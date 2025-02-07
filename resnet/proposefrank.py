import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, MeanAbsoluteError
from torchvision.models.video import r3d_18, R3D_18_Weights

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
        # self.backbone.train()
        # for module in self.backbone.modules():
        #     if isinstance(module, nn.Module):
        #         module.gradient_checkpointing_enable()
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
        
        # Losses
        self.classification_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.regression_loss = nn.HuberLoss()
        self.multi_task_optimizer = CombinedOptimizer(self, num_tasks=2, 
                                            frank_wolfe_weight=0.4,  
                                            alpha=1.5)
        
        # Loss history
        self.loss_history = {
            'classification': [],
            'regression': [],
            'weights': []
        }
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def on_fit_start(self):
        """Called when training begins"""
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

    def training_step(self, batch, batch_idx):
        image, label, mmse, age, gender = batch['image'], batch['label'], batch['mmse'], batch['age'], batch['gender']
        metadata = torch.stack([mmse, age, gender], dim=1).float()
        
        classification_output, regression_output = self(image, metadata)
        
        classification_loss = self.classification_loss(classification_output, label)
        regression_loss = self.regression_loss(regression_output.squeeze(), mmse)
        
        shared_params = list(self.shared_representation.parameters())
    
        losses = [classification_loss.to(self.device), regression_loss.to(self.device)]
        weights = self.multi_task_optimizer.update_weights(losses, shared_params)
        
        # losses = [classification_loss.to(self.device), regression_loss.to(self.device)]
        # weights = self.multi_task_optimizer.update_weights(losses)
        
        total_loss = torch.sum(torch.stack(losses) * weights)
        
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
        
        # Detailed metrics logging
        self.log('test_total_loss', total_loss, on_epoch=True)
        self.log('test_classification_loss', classification_loss, on_epoch=True)
        self.log('test_regression_loss', regression_loss, on_epoch=True)
        self.log('test_classification_acc', acc, on_epoch=True, prog_bar=True)
        self.log('test_regression_mae', mae, on_epoch=True, prog_bar=True)
        self.log('final_classification_weight', weights[0], on_epoch=True)
        self.log('final_regression_weight', weights[1], on_epoch=True)
        
        return {
            'preds': preds,
            'true_labels': label,
            'predicted_mmse': regression_output.squeeze(),
            'true_mmse': mmse,
            'final_weights': weights.cpu()
        }

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