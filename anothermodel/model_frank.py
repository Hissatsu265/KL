import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, MeanAbsoluteError
from torchvision.models.video import r3d_18, R3D_18_Weights

from multitask.visualize.visualize import visualize_and_save_gradcam,plot_and_save_optimization_metrics
import os
from typing import List
from torchmetrics.classification import BinarySpecificity, BinaryRecall
from torchmetrics.classification import MulticlassSpecificity, MulticlassRecall
def rmse_tt(predictions, targets):
        mse = F.mse_loss(predictions, targets)
        return torch.sqrt(mse)
class GradNormOptimizer:
    def __init__(self, model, num_tasks=2, alpha=1.5):
        self.model = model
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.initial_losses = None
        self.task_weights = nn.Parameter(torch.ones(num_tasks, requires_grad=True))
        self.weights_optimizer = torch.optim.Adam([self.task_weights], lr=0.025)
        
    def to(self, device):
        self.task_weights = self.task_weights.to(device)
        return self
        
    def compute_grad_norm_loss(self, losses, shared_params):
        if self.initial_losses is None:
            self.initial_losses = [loss.item() for loss in losses]
            
        L_ratio = torch.stack([loss / init_loss for loss, init_loss 
                             in zip(losses, self.initial_losses)])
        L_mean = torch.mean(L_ratio)
        r_weights = L_ratio / L_mean
        
        # Get weighted grad norms for each task
        grad_norms = []
        for i, loss in enumerate(losses):
            grads = torch.autograd.grad(loss, shared_params, retain_graph=True)
            grad_norm = torch.norm(torch.stack([g.norm() for g in grads]))
            grad_norms.append(grad_norm)
        
        grad_norms = torch.stack(grad_norms)
        mean_norm = torch.mean(grad_norms)
        
        target_grad_norm = grad_norms * (r_weights ** self.alpha)
        gradnorm_loss = torch.sum(torch.abs(grad_norms - target_grad_norm))
        
        return gradnorm_loss
        
    def update_weights(self, losses, shared_params):
        gradnorm_loss = self.compute_grad_norm_loss(losses, shared_params)
        
        self.weights_optimizer.zero_grad()
        gradnorm_loss.backward(retain_graph=True)  # Add retain_graph=True
        self.weights_optimizer.step()
        
        normalized_weights = F.softmax(self.task_weights, dim=0)
        return normalized_weights

class CombinedOptimizer:
    def __init__(self, model, num_tasks=2, frank_wolfe_weight=0.5, alpha=1.5):
        self.frank_wolfe = FrankWolfeOptimizer(num_tasks)
        self.gradnorm = GradNormOptimizer(model, num_tasks, alpha)
        self.frank_wolfe_weight = frank_wolfe_weight
        self.weights = torch.ones(num_tasks) / num_tasks
        self.device = None
        
    def to(self, device):
        self.device = device
        self.frank_wolfe.to(device)
        self.gradnorm.to(device)
        self.weights = self.weights.to(device)
        return self
        
    def update_weights(self, losses, shared_params):
        gn_weights = self.gradnorm.update_weights(losses, shared_params)
        
        fw_weights = self.frank_wolfe.update_weights(losses)
        
        combined_weights = (self.frank_wolfe_weight * fw_weights + 
                          (1 - self.frank_wolfe_weight) * gn_weights)
        
  
        self.weights = combined_weights
        return self.weights
class FrankWolfeOptimizer:
    def __init__(self, num_tasks=2, step_size_param=0.1):
        self.num_tasks = num_tasks
        self.weights = None
        self.device = None
        self.step_size_param = step_size_param
        self.iteration = 0
        self.loss_history = []
        
    def to(self, device):
        self.device = device
        if self.weights is None:
            self.weights = (torch.ones(self.num_tasks) / self.num_tasks).to(device)
        else:
            self.weights = self.weights.to(device)
        return self
    
    def compute_step_size(self):
        # Standard Frank-Wolfe step size
        return 2.0 / (self.iteration + 2.0)
    
    def project_to_simplex(self, gradient):
        # Find vertex of simplex in direction of negative gradient
        min_idx = torch.argmax(gradient)
        vertex = torch.zeros_like(self.weights, device=self.device)
        vertex[min_idx] = 1.0
        return vertex
    
    def update_weights(self, losses):
        if self.weights is None:
            self.device = losses[0].device
            self.weights = (torch.ones(self.num_tasks) / self.num_tasks).to(self.device)
        
        # Compute loss gradient
        losses_tensor = torch.stack([loss.detach() for loss in losses])
        
        # Project gradient onto simplex
        s = self.project_to_simplex(losses_tensor)
        
        # Compute step size
        gamma = self.compute_step_size()
        
        # Update weights using Frank-Wolfe update rule
        self.weights = (1 - gamma) * self.weights + gamma * s
        
        # Update iteration counter and store loss history
        self.iteration += 1
        self.loss_history.append([loss.item() for loss in losses])
        
        return self.weightss

class AttentionGatingModule(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionGatingModule, self).__init__()
        self.attention = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),  # Changed to GELU
            nn.Dropout(0.2),
            nn.Linear(feature_dim // 2, feature_dim),
            nn.Sigmoid()
        )
        
        # Added residual connection
        self.residual = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )

    def forward(self, shared_features, task_specific_features):
        if shared_features.size(1) != task_specific_features.size(1):
            task_specific_features = F.linear(task_specific_features, 
                                           torch.eye(shared_features.size(1)).to(task_specific_features.device))
        
        attention_weights = self.attention(shared_features)
        gated_features = shared_features * attention_weights + \
                        task_specific_features * (1 - attention_weights)
        
        # Add residual connection
        residual = self.residual(gated_features)
        return gated_features + residual

class MultiTaskAlzheimerModel(pl.LightningModule):
    def __init__(self, 
                 num_classes=2, 
                 input_shape=(1, 64, 64, 64),
                 metadata_dim=3,
                 pretrained=True):
        super(MultiTaskAlzheimerModel, self).__init__()
        
        if pretrained:
            print('dfdfdfdfdff===================================')
            self.backbone = r3d_18(weights=R3D_18_Weights.DEFAULT)
        else:
            self.backbone = r3d_18(weights=None)

        self.backbone.stem[0] = nn.Conv3d(
            input_shape[0], 64, 
            kernel_size=(3, 7, 7), 
            stride=(1, 2, 2), 
            padding=(1, 3, 3), 
            bias=False
        )
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        self.metadata_embedding = nn.Sequential(
            nn.Linear(metadata_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128)
        )
        
        # Enhanced cross attention
        self.cross_attention = nn.Sequential(
            nn.Linear(512 + 128, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.LayerNorm(512)
        )
        
        # Enhanced shared representation
        self.shared_representation = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024)
        )
        
        # Task-specific branches with enhanced attention
        self.classification_gate =AttentionGatingModule(1024)
        self.classification_branch = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        self.regression_gate = AttentionGatingModule(1024)
        self.regression_branch = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
        
        # Metrics
        self.train_classification_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_classification_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_classification_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.mmse_mae = MeanAbsoluteError()
 
        self.classification_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.regression_loss = nn.HuberLoss()
        self.multi_task_optimizer = CombinedOptimizer(self, num_tasks=2, 
                                            frank_wolfe_weight=0.4,  
                                            alpha=1.5)

        if num_classes>2:
            self.specificity = MulticlassSpecificity(num_classes=num_classes)
            self.sensitivity = MulticlassRecall(num_classes=num_classes)
        else:
            self.specificity = BinarySpecificity()  # For binary classification
            self.sensitivity = BinaryRecall()
 
        self.loss_history = {
            'classification': [],
            'regression': [],
            'weights': [],
            'grad_norms_classification': [],
            'grad_norms_regression': []
        }
        self._init_weights()
        self.multi_task_optimizer = FrankWolfeOptimizer(
            num_tasks=2,
            step_size_param=0.1
        )
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def on_fit_start(self):
        self.multi_task_optimizer.to(self.device)

    def forward(self, image, metadata):
        # Extract features using ResNet3D
        x = self.backbone(image)
        image_features = x.squeeze(-1).squeeze(-1).squeeze(-1)  # Remove spatial dimensions
        
        # Process metadata
        metadata_features = self.metadata_embedding(metadata)
        
        # Feature fusion
        fused_features = self.cross_attention(torch.cat([image_features, metadata_features], dim=1))
        
        # Shared representation
        shared_features = self.shared_representation(fused_features)
        
        # Task-specific outputs
        classification_features = self.classification_gate(shared_features, shared_features)
        classification_output = self.classification_branch(classification_features)
        
        regression_features = self.regression_gate(shared_features, shared_features)
        regression_output = self.regression_branch(regression_features)
        
        return classification_output, regression_output

    # def training_step(self, batch, batch_idx):
    #     image, label, mmse, age, gender = batch['image'], batch['label'], batch['mmse'], batch['age'], batch['gender']
    #     metadata = torch.stack([mmse, age, gender], dim=1).float()
        
    #     classification_output, regression_output = self(image, metadata)
        
    #     classification_loss = self.classification_loss(classification_output, label)
    #     regression_loss = self.regression_loss(regression_output.squeeze(), mmse)

    #     # Calculate gradient norms before the multi-task optimizer update
    #     classification_loss.backward(retain_graph=True)
    #     grad_norm_classification = torch.norm(torch.stack([
    #         torch.norm(p.grad) 
    #         for p in self.parameters() 
    #         if p.grad is not None
    #     ]))
    #     self.zero_grad()
    
    #     regression_loss.backward(retain_graph=True)
    #     grad_norm_regression = torch.norm(torch.stack([
    #         torch.norm(p.grad) 
    #         for p in self.parameters() 
    #         if p.grad is not None
    #     ]))
    #     self.zero_grad()
    #     # =========================================================================================
    #     shared_params = list(self.shared_representation.parameters())
    
    #     losses = [classification_loss.to(self.device), regression_loss.to(self.device)]
    #     weights = self.multi_task_optimizer.update_weights(losses, shared_params)
        
        
    #     total_loss = torch.sum(torch.stack(losses) * weights)
        
    #     preds = torch.argmax(classification_output, dim=1)
    #     acc = (preds == label).float().mean()
        
    #     # Logging
    #     self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
    #     self.log('train_classification_loss', classification_loss, on_step=True, on_epoch=True)
    #     self.log('train_regression_loss', regression_loss, on_step=True, on_epoch=True)
    #     self.log('train_classification_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
    #     self.log('train_classification_weight', weights[0], on_step=True, on_epoch=True)
    #     self.log('train_regression_weight', weights[1], on_step=True, on_epoch=True)
        
    #     # Save history
    #     self.loss_history['classification'].append(classification_loss.item())
    #     self.loss_history['regression'].append(regression_loss.item())
    #     self.loss_history['weights'].append(weights.tolist())
    #     self.loss_history['grad_norms_classification'].append(grad_norm_classification.item())
    #     self.loss_history['grad_norms_regression'].append(grad_norm_regression.item())

    #     return total_loss
    def training_step(self, batch, batch_idx):
        image, label, mmse, age, gender = batch['image'], batch['label'], batch['mmse'], batch['age'], batch['gender']
        metadata = torch.stack([mmse, age, gender], dim=1).float()
        
        classification_output, regression_output = self(image, metadata)
        
        classification_loss = self.classification_loss(classification_output, label)
        regression_loss = self.regression_loss(regression_output.squeeze(), mmse)
        
        classification_loss.backward(retain_graph=True)
        grad_norm_classification = torch.norm(torch.stack([
            torch.norm(p.grad) 
            for p in self.parameters() 
            if p.grad is not None
        ]))
        self.zero_grad()
    
        regression_loss.backward(retain_graph=True)
        grad_norm_regression = torch.norm(torch.stack([
            torch.norm(p.grad) 
            for p in self.parameters() 
            if p.grad is not None
        ]))
        self.zero_grad()
    #     # =========================================================================================
        # Update weights using only Frank-Wolfe
        losses = [classification_loss.to(self.device), regression_loss.to(self.device)]
        weights = self.multi_task_optimizer.update_weights(losses)
        
        total_loss = torch.sum(torch.stack(losses) * weights)
        
        # Calculate accuracy
        preds = torch.argmax(classification_output, dim=1)
        acc = (preds == label).float().mean()
        
        # Logging
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_classification_loss', classification_loss, on_step=True, on_epoch=True)
        self.log('train_regression_loss', regression_loss, on_step=True, on_epoch=True)
        self.log('train_classification_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_classification_weight', weights[0], on_step=True, on_epoch=True)
        self.log('train_regression_weight', weights[1], on_step=True, on_epoch=True)
        
        # Save history
        self.loss_history['classification'].append(classification_loss.item())
        self.loss_history['regression'].append(regression_loss.item())
        self.loss_history['weights'].append(weights.tolist())

        return total_loss

    def validation_step(self, batch, batch_idx):
        image, label, mmse, age, gender = batch['image'], batch['label'], batch['mmse'], batch['age'], batch['gender']
        metadata = torch.stack([mmse, age, gender], dim=1).float()
        
        classification_output, regression_output = self(image, metadata)
        
        classification_loss = self.classification_loss(classification_output, label)
        regression_loss = self.regression_loss(regression_output.squeeze(), mmse)
        
        losses = [classification_loss.to(self.device), regression_loss.to(self.device)]
        weights = self.multi_task_optimizer.weights  # Now this will work
        
        total_loss = torch.sum(torch.stack(losses) * weights)
        
        preds = torch.argmax(classification_output, dim=1)
        acc = (preds == label).float().mean()
        
        # Logging
        self.log('val_loss', total_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_classification_loss', classification_loss, on_epoch=True, sync_dist=True)
        self.log('val_regression_loss', regression_loss, on_epoch=True, sync_dist=True)
        self.log('val_classification_acc', acc, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_classification_weight', weights[0], on_epoch=True, sync_dist=True)
        self.log('val_regression_weight', weights[1], on_epoch=True, sync_dist=True)
        
        return total_loss

    def test_step(self, batch, batch_idx):
        image, label, mmse, age, gender = batch['image'], batch['label'], batch['mmse'], batch['age'], batch['gender']
        metadata = torch.stack([mmse, age, gender], dim=1).float()
        
        classification_output, regression_output = self(image, metadata)
        
        classification_loss = self.classification_loss(classification_output, label).to(self.device)
        regression_loss = self.regression_loss(regression_output.squeeze(), mmse).to(self.device)
        
        # weights = self.multi_task_optimizer.weights
        # total_loss = torch.sum(torch.stack([classification_loss, regression_loss]) * weights)
        weights = self.multi_task_optimizer.weights  # Now this will work
        total_loss = torch.sum(torch.stack([classification_loss, regression_loss]) * weights)
    
        preds = torch.argmax(classification_output, dim=1)
        acc = (preds == label).float().mean()
        mae = F.l1_loss(regression_output.squeeze(), mmse)
        rmse = rmse_tt(regression_output.squeeze(), mmse)
        spec = self.specificity(preds, label)
        sens = self.sensitivity(preds, label)
        
        self.log('test_total_loss', total_loss, on_epoch=True)
        self.log('test_classification_loss', classification_loss, on_epoch=True)
        self.log('test_regression_loss', regression_loss, on_epoch=True)
        self.log('test_classification_acc', acc, on_epoch=True, prog_bar=True)
        self.log('test_regression_mae', mae, on_epoch=True, prog_bar=True)
        self.log('test_regression_rmse', rmse, on_epoch=True, prog_bar=True)
        self.log('final_classification_weight', weights[0], on_epoch=True)
        self.log('final_regression_weight', weights[1], on_epoch=True)
        
        self.log('test_specificity', spec, on_epoch=True, prog_bar=True)
        self.log('test_sensitivity', sens, on_epoch=True, prog_bar=True)
        
        self.loss_history['classification'].append(classification_loss.item())
        self.loss_history['regression'].append(regression_loss.item())
        self.loss_history['weights'].append(weights.tolist())

        
        return {
            'preds': preds,
            'true_labels': label,
            'predicted_mmse': regression_output.squeeze(),
            'true_mmse': mmse,
            'final_weights': weights.cpu(),
            'specificity': spec,
            'sensitivity': sens
        }
    def on_test_end(self):
        save_path = '/home/jupyter-iec_iot13_toanlm/multitask/visualize'
        os.makedirs(save_path, exist_ok=True)
        
        if hasattr(self, 'sample_images'):
            image = self.sample_images[0]  
            visualize_and_save_gradcam(self, image, save_path, num_slices=4)
        
        
        plot_and_save_optimization_metrics(self, save_path)
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.1, 
            patience=5
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }
# ==============================================================================
   