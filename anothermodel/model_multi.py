import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, MeanAbsoluteError
from torchvision.models.video import r3d_18, R3D_18_Weights
from torchmetrics.classification import BinarySpecificity, BinaryRecall
from torchmetrics.classification import MulticlassSpecificity, MulticlassRecall
import os
import wandb
import matplotlib.pyplot as plt
import numpy as np
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
    def __init__(self, model, num_tasks=2, frank_wolfe_weight=1, alpha=1):
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
        # Compute GradNorm weights first with retain_graph=True
        gn_weights = self.gradnorm.update_weights(losses, shared_params)
        
        # Then compute Frank-Wolfe weights
        fw_weights = self.frank_wolfe.update_weights(losses)
        
        # Combine weights
        combined_weights = (self.frank_wolfe_weight * fw_weights + 
                          (1 - self.frank_wolfe_weight) * gn_weights)
        
        # Extra weight for regression task
        # combined_weights[1] *= 1.2
        
        # self.weights = F.softmax(combined_weights, dim=0)    
        self.weights = combined_weights
        return self.weights


class FrankWolfeOptimizer:
    def __init__(self, num_tasks=2, max_iter=10, beta=0.1):
        self.num_tasks = num_tasks
        self.max_iter = max_iter
        self.weights = None
        self.device = None
        self.beta = beta  # Parameter for gamma calculation
        self.iteration = 0
        self.loss_history = []
        
    def to(self, device):
        self.device = device
        if self.weights is None:
            self.weights = (torch.ones(self.num_tasks) / self.num_tasks).to(device)
        else:
            self.weights = self.weights.to(device)
        return self
        
    def compute_gradient(self, losses, prev_weights):
        # Formula 1: Enhanced gradient computation
        # gi = 0.9 * log(1 + Li) + 0.1 * wi_prev
        losses_tensor = torch.stack([loss.detach() for loss in losses])
        log_losses = torch.log(1 + losses_tensor)
        return 0.9 * log_losses + 0.1 * prev_weights
    
    def compute_gamma(self, losses):
        # Formula 2: Enhanced gamma computation
        # γ = min(1, 2/(t+2)) * exp(-β * L_mean)
        L_mean = torch.mean(torch.stack([loss.detach() for loss in losses]))
        base_gamma = min(1.0, 2.0 / (self.iteration + 2))
        return base_gamma * torch.exp(-self.beta * L_mean)
    
    def solve_linear_problem(self, gradients):
        min_idx = torch.argmin(gradients)
        s = torch.zeros_like(self.weights, device=self.device)
        s[min_idx] = 1.0
        return s
    
    def update_weights(self, losses):
        if self.weights is None:
            self.device = losses[0].device
            self.weights = (torch.ones(self.num_tasks) / self.num_tasks).to(self.device)
            
        prev_weights = self.weights.clone()
        
        # Compute enhanced gradient
        gradients = self.compute_gradient(losses, prev_weights)
        
        # Solve linear problem
        s = self.solve_linear_problem(gradients)
        
        # Compute enhanced gamma
        gamma = self.compute_gamma(losses)
        
        # Formula 3: Enhanced weight update with logarithmic barrier
        # w_new = (1 - γ) * w_prev + γ * log(1 + s)
        log_barrier = torch.log(1 + s)
        new_weights = (1 - gamma) * prev_weights + gamma * log_barrier
        
        # Formula 4: Softmax normalization
        self.weights = F.softmax(new_weights, dim=0)
        
        # Update iteration counter and store loss history
        self.iteration += 1
        self.loss_history.append([loss.item() for loss in losses])
        
        return self.weights
# class AttentionGatingModule(nn.Module):
#     def __init__(self, feature_dim):
#         super(AttentionGatingModule, self).__init__()
#         self.attention = nn.Sequential(
#             nn.LayerNorm(feature_dim),
#             nn.Linear(feature_dim, feature_dim // 2),
#             nn.GELU(),  # Changed to GELU
#             nn.Dropout(0.2),
#             nn.Linear(feature_dim // 2, feature_dim),
#             nn.Sigmoid()
#         )
        
#         # Added residual connection
#         self.residual = nn.Sequential(
#             nn.Linear(feature_dim, feature_dim),
#             nn.LayerNorm(feature_dim)
#         )

#     def forward(self, shared_features, task_specific_features):
#         if shared_features.size(1) != task_specific_features.size(1):
#             task_specific_features = F.linear(task_specific_features, 
#                                            torch.eye(shared_features.size(1)).to(task_specific_features.device))
        
#         attention_weights = self.attention(shared_features)
#         gated_features = shared_features * attention_weights + \
#                         task_specific_features * (1 - attention_weights)
        
#         # Add residual connection
#         residual = self.residual(gated_features)
#         return gated_features + residual
        
class CrossModalAttention(nn.Module):
    def __init__(self, feature_dim):
        super(CrossModalAttention, self).__init__()
        self.query_transform = nn.Linear(feature_dim, feature_dim)
        self.key_transform = nn.Linear(feature_dim, feature_dim)
        self.value_transform = nn.Linear(feature_dim, feature_dim)
        self.scaling_factor = torch.sqrt(torch.tensor(feature_dim, dtype=torch.float32))
        self.output_projection = nn.Linear(feature_dim, feature_dim)
        
    def forward(self, image_features, metadata_features):
        # Generate query from image features
        query = self.query_transform(image_features)  # [B, feature_dim]
        # Generate key and value from metadata features
        key = self.key_transform(metadata_features)   # [B, feature_dim]
        value = self.value_transform(metadata_features)  # [B, feature_dim]
        
        # Reshape for attention calculation
        # [B, 1, feature_dim]
        query = query.unsqueeze(1)
        # [B, feature_dim, 1]
        key = key.unsqueeze(2)
        
        # Calculate attention scores: [B, 1, 1]
        attention_scores = torch.bmm(query, key) / self.scaling_factor
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Reshape value for multiplication with attention weights
        # [B, 1, feature_dim]
        value = value.unsqueeze(1)
        
        # Apply attention: [B, 1, feature_dim]
        context = torch.bmm(attention_weights, value)
        # [B, feature_dim]
        context = context.squeeze(1)
        
        # Combine with original image features (residual connection)
        output = self.output_projection(context + image_features)
        
        return output

class TaskSpecificEncoder(nn.Module):
   
    def __init__(self, input_dim, output_dim):
        super(TaskSpecificEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Task-specific attention (unique to each task)
        self.task_attention = nn.Sequential(
            nn.Linear(output_dim, output_dim // 4),
            nn.GELU(),
            nn.Linear(output_dim // 4, output_dim),
            nn.Sigmoid()
        )
        
    def forward(self, shared_features):
        # Encode shared features
        encoded = self.encoder(shared_features)
        
        # Apply task-specific attention
        attention_weights = self.task_attention(encoded)
        task_specific_features = encoded * attention_weights
        
        return task_specific_features

class KnowledgeDistillation(nn.Module):
    """Knowledge distillation module for mutual learning between tasks"""
    def __init__(self, feature_dim, alpha=0.3):
        super(KnowledgeDistillation, self).__init__()
        self.alpha = alpha
        self.transform = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def forward(self, target_features, source_features):
        """Transfer knowledge from source task to target task"""
        transformed_source = self.transform(source_features)
        
        # Weighted combination of target and transformed source features
        enhanced_features = (1 - self.alpha) * target_features + self.alpha * transformed_source
        
        return enhanced_features

def rmse_tt(predictions, targets):
    mse = F.mse_loss(predictions, targets)
    return torch.sqrt(mse)

class MultiTaskAlzheimerModel(pl.LightningModule):
    def __init__(self, 
                 num_classes=2, 
                 input_shape=(1, 64, 64, 64),
                 metadata_dim=2,
                 pretrained=True,
                 shared_dim=512,
                 task_specific_dim=256):
        super(MultiTaskAlzheimerModel, self).__init__()
        
        if pretrained:
            self.image_encoder = r3d_18(weights=R3D_18_Weights.DEFAULT)
        else:
            self.image_encoder = r3d_18(weights=None)

        # Modify first convolution to match input channels
        self.image_encoder.stem[0] = nn.Conv3d(
            input_shape[0], 64, 
            kernel_size=(7, 7, 7), 
            stride=(2, 2, 2), 
            padding=(3, 3, 3), 
            bias=False
        )
        
        # Remove the classification head
        self.image_encoder = nn.Sequential(*list(self.image_encoder.children())[:-1])
        
        # Metadata encoder
        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.LayerNorm(128)
        )
        
        # Feature fusion module
        self.image_projection = nn.Linear(512, shared_dim)
        self.metadata_projection = nn.Linear(128, shared_dim)
        
        # Cross-modal attention for feature fusion
        self.cross_modal_attention = CrossModalAttention(shared_dim)
        
        # Shared representation module - THE CORE OF MULTI-TASK LEARNING
        self.shared_representation = nn.Sequential(
            nn.Linear(shared_dim, shared_dim*2),
            nn.LayerNorm(shared_dim*2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(shared_dim*2, shared_dim),
            nn.LayerNorm(shared_dim)
        )
        
        # Task-specific encoders - TASK-SPECIFIC FEATURES
        self.classification_encoder = TaskSpecificEncoder(shared_dim, task_specific_dim)
        self.regression_encoder = TaskSpecificEncoder(shared_dim, task_specific_dim)
        
        # Task-specific decoders - TASK-SPECIFIC OUTPUTS
        self.classification_decoder = nn.Sequential(
            nn.Linear(task_specific_dim, task_specific_dim//2),
            nn.LayerNorm(task_specific_dim//2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(task_specific_dim//2, num_classes)
        )
        
        self.regression_decoder = nn.Sequential(
            nn.Linear(task_specific_dim, task_specific_dim//2),
            nn.LayerNorm(task_specific_dim//2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(task_specific_dim//2, 1)
        )
        
        # Knowledge distillation modules - MUTUAL LEARNING BETWEEN TASKS
        self.cls_to_reg_distillation = KnowledgeDistillation(task_specific_dim)
        self.reg_to_cls_distillation = KnowledgeDistillation(task_specific_dim)
        
        # Loss functions
        self.classification_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.regression_loss = nn.HuberLoss()
        
        # Initialize metrics
        self._init_metrics(num_classes)
        
        # Dynamic task weighting
        self.multi_task_optimizer = CombinedOptimizer(self, num_tasks=2, 
                                            frank_wolfe_weight=0.4,  
                                            alpha=1.5)
        
        # Track distribution of samples and history
        self.class_counts = {"AD": 0, "CN": 0, "MCI": 0}
        self.current_epoch_idx = 0
        self._init_weights()
        
    def _init_metrics(self, num_classes):
        # Classification metrics
        self.train_classification_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_classification_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_classification_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        
        # Regression metrics
        self.mmse_mae = MeanAbsoluteError()
        
        # Additional metrics based on number of classes
        if num_classes > 2:
            self.specificity = MulticlassSpecificity(num_classes=num_classes)
            self.sensitivity = MulticlassRecall(num_classes=num_classes)
        else:
            self.specificity = BinarySpecificity()  
            self.sensitivity = BinaryRecall()
        
        # Result tracking
        self.test_predictions = []
        self.test_labels = []
        self.test_mmse_true = []
        self.test_mmse_pred = []
        self.loss_history = {
            'epochs': [], 'total': [], 'classification': [], 
            'regression': [], 'distillation': [], 'weights': [],
            'grad_norms_classification': [], 'grad_norms_regression': []
        }
        
    def _init_weights(self):
        """Initialize model weights for better convergence"""
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

    def encode_image(self, image):
        """Extract features from MRI image"""
        x = self.image_encoder(image)
        image_features = x.squeeze(-1).squeeze(-1).squeeze(-1)
        return self.image_projection(image_features)
        
    def encode_metadata(self, metadata):
        """Extract features from patient metadata"""
        metadata_features = self.metadata_encoder(metadata)
        return self.metadata_projection(metadata_features)
    
    def forward(self, image=None, metadata=None):
        """
        Forward pass with flexible inputs. Allows for:
        1. Both image and metadata (complete analysis)
        2. Image only (when patient data is unavailable)
        3. Metadata only (for quick preliminary assessment)
        """
        # Validate at least one input is provided
        if image is None and metadata is None:
            raise ValueError("At least one of image or metadata must be provided")
            
        # Process available inputs
        image_features = self.encode_image(image) if image is not None else None
        metadata_features = self.encode_metadata(metadata) if metadata is not None else None
        
        # Handle different input combinations
        if image_features is not None and metadata_features is not None:
            # Fusion via cross-modal attention when both inputs are available
            fused_features = self.cross_modal_attention(image_features, metadata_features)
        elif image_features is not None:
            # Image-only path
            fused_features = image_features
        else:
            # Metadata-only path
            fused_features = metadata_features
            
        # Shared representation - the core of multi-task learning
        shared_features = self.shared_representation(fused_features)
        
        # Task-specific processing
        # ==========================ddooir 1==========================================
        # cls_features = self.classification_encoder(shared_features)
        # reg_features = self.regression_encoder(shared_features)
# ===========================================================================
        cls_features = self.classification_encoder(shared_features)
        # Knowledge distillation between tasks for mutual improvement (only during training)
        if self.training:
            cls_features = self.reg_to_cls_distillation(cls_features, cls_features)
            # reg_features = self.cls_to_reg_distillation(reg_features, cls_features)
        
        # Generate task outputs
        classification_output = self.classification_decoder(cls_features)
        regression_output = self.regression_decoder(cls_features)
# ===========================================================================================================        
        return classification_output, regression_output, cls_features, cls_features

    def training_step(self, batch, batch_idx):
        # Extract batch data
        image, label, mmse, age, gender = batch['image'], batch['label'], batch['mmse'], batch['age'], batch['gender']
        metadata = torch.stack([age, gender], dim=1).float()
        
        # Forward pass
        classification_output, regression_output, cls_features, reg_features = self(image, metadata)
        
        # Calculate task losses
        classification_loss = self.classification_loss(classification_output, label)
        regression_loss = self.regression_loss(regression_output.squeeze(), mmse)

        # Calculate gradient norms for task weighting
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
        
        # Dynamic task weighting
        shared_params = list(self.shared_representation.parameters())
        losses = [classification_loss.to(self.device), regression_loss.to(self.device)]
        weights = self.multi_task_optimizer.update_weights(losses, shared_params)
        
        # Weighted total loss
        total_loss = torch.sum(torch.stack(losses) * weights)
        
        # Basic metric calculation
        preds = torch.argmax(classification_output, dim=1)
        acc = (preds == label).float().mean()
        
        # Logging
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_classification_loss', classification_loss, on_step=True, on_epoch=True)
        self.log('train_regression_loss', regression_loss, on_step=True, on_epoch=True)
        self.log('train_classification_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_classification_weight', weights[0], on_step=True, on_epoch=True)
        self.log('train_regression_weight', weights[1], on_step=True, on_epoch=True)
        self.log('grad_norm_classification', grad_norm_classification, on_step=True, on_epoch=True)
        self.log('grad_norm_regression', grad_norm_regression, on_step=True, on_epoch=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        # Extract batch data
        image, label, mmse, age, gender = batch['image'], batch['label'], batch['mmse'], batch['age'], batch['gender']
        metadata = torch.stack([age, gender], dim=1).float()
        
        # Forward pass
        classification_output, regression_output, _, _ = self(image, metadata)
        
        # Calculate task losses
        classification_loss = self.classification_loss(classification_output, label)
        regression_loss = self.regression_loss(regression_output.squeeze(), mmse)
        
        losses = [classification_loss.to(self.device), regression_loss.to(self.device)]
        weights = self.multi_task_optimizer.weights
        
        total_loss = torch.sum(torch.stack(losses) * weights)
        
        # Calculate metrics
        preds = torch.argmax(classification_output, dim=1)
        acc = (preds == label).float().mean()
        mae = F.l1_loss(regression_output.squeeze(), mmse)
        
        # Logging
        self.log('val_loss', total_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_classification_loss', classification_loss, on_epoch=True, sync_dist=True)
        self.log('val_regression_loss', regression_loss, on_epoch=True, sync_dist=True)
        self.log('val_classification_acc', acc, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_mmse_mae', mae, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return total_loss
    
    def test_step(self, batch, batch_idx):
        # Extract batch data
        image, label, mmse, age, gender = batch['image'], batch['label'], batch['mmse'], batch['age'], batch['gender']
        
        # Track class distribution
        for idx in label:
            label_val = idx.item()
            if label_val == 0:
                self.class_counts["AD"] += 1
            elif label_val == 1:
                self.class_counts["CN"] += 1
            else:
                self.class_counts["MCI"] += 1
        
        metadata = torch.stack([age, gender], dim=1).float()
        
        # Forward pass
        classification_output, regression_output, _, _ = self(image, metadata)
        
        # Calculate task losses
        classification_loss = self.classification_loss(classification_output, label)
        regression_loss = self.regression_loss(regression_output.squeeze(), mmse)
        
        weights = self.multi_task_optimizer.weights
        total_loss = torch.sum(torch.stack([classification_loss, regression_loss]) * weights)
    
        # Calculate metrics
        preds = torch.argmax(classification_output, dim=1)
        acc = (preds == label).float().mean()
        mae = F.l1_loss(regression_output.squeeze(), mmse)
        rmse = rmse_tt(regression_output.squeeze(), mmse)
        spec = self.specificity(preds, label)
        sens = self.sensitivity(preds, label)
        
        # Log metrics
        self.log('test_total_loss', total_loss, on_epoch=True)
        self.log('test_classification_loss', classification_loss, on_epoch=True)
        self.log('test_regression_loss', regression_loss, on_epoch=True)
        self.log('test_classification_acc', acc, on_epoch=True, prog_bar=True)
        self.log('test_regression_mae', mae, on_epoch=True, prog_bar=True)
        self.log('test_regression_rmse', rmse, on_epoch=True, prog_bar=True)
        self.log('test_specificity', spec, on_epoch=True, prog_bar=True)
        self.log('test_sensitivity', sens, on_epoch=True, prog_bar=True)
        
        # Store for visualization
        self.test_predictions.append(preds.cpu())
        self.test_labels.append(label.cpu())
        self.test_mmse_true.append(mmse.cpu())
        self.test_mmse_pred.append(regression_output.squeeze().cpu())
        
        return {
            'preds': preds,
            'true_labels': label,
            'predicted_mmse': regression_output.squeeze(),
            'true_mmse': mmse,
            'specificity': spec,
            'sensitivity': sens
        }
    
    def on_train_epoch_end(self):
        # Store loss values at the end of each epoch for plotting
        self.loss_history['epochs'].append(self.current_epoch_idx)
        
        # Get the current epoch's average loss values
        avg_total_loss = self.trainer.callback_metrics.get('train_loss_epoch', 0)
        avg_class_loss = self.trainer.callback_metrics.get('train_classification_loss_epoch', 0)
        avg_reg_loss = self.trainer.callback_metrics.get('train_regression_loss_epoch', 0)
        
        # Convert from tensor to float if needed
        if hasattr(avg_total_loss, 'item'):
            avg_total_loss = avg_total_loss.item()
        if hasattr(avg_class_loss, 'item'):
            avg_class_loss = avg_class_loss.item()
        if hasattr(avg_reg_loss, 'item'):
            avg_reg_loss = avg_reg_loss.item()
            
        # Store for later plotting
        self.loss_history['total'].append(avg_total_loss)
        self.loss_history['classification'].append(avg_class_loss)
        self.loss_history['regression'].append(avg_reg_loss)
        
        # Plot and log to wandb
        self.plot_loss_curves()
        
        # Increment epoch counter
        self.current_epoch_idx += 1
    
    def on_test_start(self):
        self.test_predictions = []
        self.test_labels = []
        self.test_mmse_true = []
        self.test_mmse_pred = []
        self.class_counts = {"AD": 0, "CN": 0, "MCI": 0}
        
    def on_test_end(self):
        # Print class distribution
        print("============================================")
        print(f"Number of AD samples: {self.class_counts['AD']}")
        print(f"Number of CN samples: {self.class_counts['CN']}")
        print(f"Number of MCI samples: {self.class_counts['MCI']}")
        print("============================================")
        
        # Gather all predictions and labels
        all_preds = torch.cat(self.test_predictions).numpy()
        all_labels = torch.cat(self.test_labels).numpy()
        
        all_true_mmse = torch.cat(self.test_mmse_true).numpy()
        
        # Ensure each tensor in the list has at least 1 dimension
        fixed_mmse_pred = []
        for tensor in self.test_mmse_pred:
            if tensor.dim() == 0:  # If it's a 0-dim tensor
                fixed_mmse_pred.append(tensor.unsqueeze(0))  # Convert to 1-dim tensor
            else:
                fixed_mmse_pred.append(tensor)
        
        all_pred_mmse = torch.cat(fixed_mmse_pred).numpy()
        
        # Determine class names for visualization
        class_names = ["AD", "CN", "MCI"] if hasattr(self, 'num_classes') and self.num_classes > 2 else ["CN", "MCI"]
            
        # Create visualizations
        self.plot_confusion_matrix(all_preds, all_labels, class_names)
        self.plot_mmse_predictions(all_true_mmse, all_pred_mmse)
    
    def plot_loss_curves(self):
        """Plot and log loss curves to wandb"""
        import matplotlib.pyplot as plt
        import wandb
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if len(self.loss_history['epochs']) > 0:
            ax.plot(self.loss_history['epochs'], self.loss_history['total'], 
                   label='Total Loss', marker='o')
            ax.plot(self.loss_history['epochs'], self.loss_history['classification'], 
                   label='Classification Loss', marker='s')
            ax.plot(self.loss_history['epochs'], self.loss_history['regression'], 
                   label='Regression Loss', marker='^')
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training Loss Convergence')
            ax.legend()
            ax.grid(True)
            
            # Log to wandb
            self.logger.experiment.log({"Training Loss Curves": wandb.Image(fig)})
        
        plt.close(fig)
        
    def plot_confusion_matrix(self, predictions, labels, class_names):
        """Plot and log confusion matrix to wandb"""
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        import matplotlib.pyplot as plt
        import wandb
        
        cm = confusion_matrix(labels, predictions)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        # Log to wandb
        self.logger.experiment.log({"Confusion Matrix": wandb.Image(plt)})
        plt.close()
        
    def plot_mmse_predictions(self, all_true_mmse, all_pred_mmse):
        """Create and log MMSE prediction visualizations to wandb"""
        import matplotlib.pyplot as plt
        import numpy as np
        import wandb
        
        # 1. Scatter Plot: True vs Predicted MMSE
        fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
        scatter = ax_scatter.scatter(all_true_mmse, all_pred_mmse, alpha=0.6)
        ax_scatter.plot([min(all_true_mmse), max(all_true_mmse)], 
                       [min(all_true_mmse), max(all_true_mmse)], 'r--')
        ax_scatter.set_xlabel('True MMSE Score')
        ax_scatter.set_ylabel('Predicted MMSE Score')
        ax_scatter.set_title('MMSE Prediction: True vs Predicted')
        ax_scatter.grid(True)
        self.logger.experiment.log({"MMSE Scatter Plot": wandb.Image(fig_scatter)})
        plt.close(fig_scatter)
        
        # 2. Residual Plot
        residuals = all_pred_mmse - all_true_mmse
        fig_residual, ax_residual = plt.subplots(figsize=(10, 6))
        ax_residual.scatter(all_true_mmse, residuals, alpha=0.6)
        ax_residual.axhline(y=0, color='r', linestyle='--')
        ax_residual.set_xlabel('True MMSE Score')
        ax_residual.set_ylabel('Residual (Predicted - True)')
        ax_residual.set_title('MMSE Prediction Residuals')
        ax_residual.grid(True)
        self.logger.experiment.log({"MMSE Residual Plot": wandb.Image(fig_residual)})
        plt.close(fig_residual)
        
        # 3. Distribution Plot
        fig_dist, ax_dist = plt.subplots(figsize=(10, 6))
        ax_dist.hist(all_true_mmse, bins=15, alpha=0.5, label='True MMSE')
        ax_dist.hist(all_pred_mmse, bins=15, alpha=0.5, label='Predicted MMSE')
        ax_dist.set_xlabel('MMSE Score')
        ax_dist.set_ylabel('Frequency')
        ax_dist.set_title('Distribution of True and Predicted MMSE Scores')
        ax_dist.legend()
        ax_dist.grid(True)
        self.logger.experiment.log({"MMSE Distribution Plot": wandb.Image(fig_dist)})
        plt.close(fig_dist)
        
        # 4. Additional metrics as a table
        mmse_metrics = {
            "MMSE MAE": float(F.l1_loss(torch.tensor(all_pred_mmse), torch.tensor(all_true_mmse))),
            "MMSE RMSE": float(torch.sqrt(F.mse_loss(torch.tensor(all_pred_mmse), torch.tensor(all_true_mmse)))),
            "MMSE Correlation": float(np.corrcoef(all_true_mmse, all_pred_mmse)[0, 1])
        }
        self.logger.experiment.log({"MMSE Metrics": wandb.Table(
            data=[[k, v] for k, v in mmse_metrics.items()],
            columns=["Metric", "Value"])
        })

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