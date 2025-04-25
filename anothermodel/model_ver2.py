import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, MeanAbsoluteError
from torchvision.models.video import r3d_18, R3D_18_Weights
from sklearn.metrics import accuracy_score, recall_score
import os
import wandb
import matplotlib.pyplot as plt
import numpy as np
from typing import List
from torchmetrics.classification import BinarySpecificity, BinaryRecall
from torchmetrics.classification import MulticlassSpecificity, MulticlassRecall

# def rmse_tt(predictions, targets):
#     mse = F.mse_loss(predictions, targets)
#     return torch.sqrt(mse)

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
    def __init__(self, model, num_tasks=2, frank_wolfe_weight=0.5,initial_lambda=0.5, alpha=1.5, eta=0.1):
        self.frank_wolfe = FrankWolfeOptimizer(num_tasks)
        self.gradnorm = GradNormOptimizer(model, num_tasks, alpha)
        self.lambda_param = initial_lambda  # Initial lambda value
        self.weights = torch.ones(num_tasks) / num_tasks
        self.device = None
        self.eta = eta  # Learning rate for lambda adaptation
        
    def to(self, device):
        self.device = device
        self.frank_wolfe.to(device)
        self.gradnorm.to(device)
        self.weights = self.weights.to(device)
        return self
   
    def update_weights(self, losses, shared_params):
        gn_weights = self.gradnorm.update_weights(losses, shared_params)
        fw_weights = self.frank_wolfe.update_weights(losses)
        
        # Tính toán tỷ lệ mất mát như trong bài báo
        loss_values = torch.stack([loss.detach() for loss in losses])
        if not hasattr(self, 'initial_losses'):
            self.initial_losses = [loss.item() for loss in losses]
        
        L_ratio = torch.stack([loss / init_loss for loss, init_loss 
                            in zip(loss_values, self.initial_losses)])
        L_mean = torch.mean(L_ratio)
        
        # Tính toán norm gradient
        grad_norms = []
        for i, loss in enumerate(losses):
            grads = torch.autograd.grad(loss, shared_params, retain_graph=True)
            grad_norm = torch.norm(torch.stack([g.norm() for g in grads]))
            grad_norms.append(grad_norm)
        
        grad_norms = torch.stack(grad_norms)
        
        # Tính toán gradient mục tiêu theo công thức trong bài báo
        alpha = 1.5  # Giá trị từ bài báo
        G_target = grad_norms * (L_ratio / L_mean) ** alpha
        
        # Tính toán sự khác biệt giữa gradient thực tế và mục tiêu
        grad_diff = torch.sum(torch.abs(grad_norms - G_target))
        
        # Tính toán lambda theo công thức trong bài báo
        lambda_input = self.eta * grad_diff
        new_lambda = torch.sigmoid(lambda_input)
        
        self.lambda_param = new_lambda.item()
        
        # Kết hợp trọng số như trong bài báo
        combined_weights = (self.lambda_param * fw_weights + 
                          (1 - self.lambda_param) * gn_weights)
        
        self.last_lambda = self.lambda_param
        self.last_grad_diff = grad_diff.item()
        
        self.weights = F.softmax(combined_weights, dim=0)
        return self.weights
    
    def get_lambda_info(self):
        return {
            'lambda': self.lambda_param,
            'gradient_difference': getattr(self, 'last_grad_diff', 0)
        }
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
class AttentionRefinementModule(nn.Module): # Đổi tên để phản ánh chức năng mới (tùy chọn)
    def __init__(self, feature_dim):
        super(AttentionRefinementModule, self).__init__()
        # Mạng attention để tính trọng số dựa trên shared features
        self.attention = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim // 2, feature_dim),
            nn.Sigmoid() # Trọng số attention trong khoảng (0, 1)
        )
        
        # Kết nối residual: biến đổi output trước khi cộng lại
        self.residual = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
    def forward(self, shared_features):
        """
        Args:
            shared_features (torch.Tensor): Đặc trưng chia sẻ từ backbone. 
                                            Shape: (batch_size, feature_dim) hoặc (batch_size, seq_len, feature_dim)

        Returns:
            torch.Tensor: Đặc trưng đã được điều chỉnh/chọn lọc cho task.
                          Shape giống như shared_features.
        """
        # 1. Tính trọng số attention chỉ dựa trên shared_features
        attention_weights = self.attention(shared_features)
        
        # 2. Áp dụng trọng số attention lên shared_features
        #    Phần tử nào quan trọng sẽ có trọng số gần 1, phần tử ít quan trọng có trọng số gần 0.
        refined_features = shared_features * attention_weights
        
        # 3. Áp dụng kết nối residual
        #    Tính residual trên refined_features và cộng vào chính nó
        residual_out = self.residual(refined_features)
        output_features = refined_features + residual_out
        
        return output_features
class TransformerFeatureEncoder(nn.Module):
    def __init__(self, feature_dim, num_heads=8, dropout=0.1):
        super(TransformerFeatureEncoder, self).__init__()
        
        # Đầu tiên, reshape feature vector thành sequence
        self.seq_len = 16  # Chia feature_dim thành các seq_len segments
        self.hidden_dim = feature_dim // self.seq_len
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.zeros(1, self.seq_len, self.hidden_dim))
        nn.init.normal_(self.pos_encoder, mean=0, std=0.02)
        
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Projection back to original dimension
        self.output_projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
    def forward(self, shared_features, task_specific_features=None):
        batch_size = shared_features.size(0)
        x = shared_features.view(batch_size, self.seq_len, self.hidden_dim)
        x = x + self.pos_encoder
        transformer_output = self.transformer_encoder(x)
        transformer_output = transformer_output.reshape(batch_size, -1)
        output = self.output_projection(transformer_output) + shared_features
        return output
class MultiTaskAlzheimerModel(pl.LightningModule):
    def __init__(self, 
                 num_classes=2, 
                 input_shape=(1, 64, 64, 64),
                 metadata_dim=2,
                 pretrained=True):
        super(MultiTaskAlzheimerModel, self).__init__()
        
        if pretrained:
            print('dfdfdfdfdff===================================')
            self.backbone = r3d_18(weights=R3D_18_Weights.DEFAULT)
        else:
            self.backbone = r3d_18(weights=None)

        self.backbone.stem[0] = nn.Conv3d(
            input_shape[0], 64, 
            kernel_size=(7, 7, 7), 
            stride=(2, 2, 2), 
            padding=(3, 3, 3), 
            bias=False
        )
        
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        print(self.backbone)
        self.metadata_embedding = nn.Sequential(
            nn.Linear(metadata_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            # nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128)
        )
        
        # Enhanced cross attention
        self.cross_attention = nn.Sequential(
            nn.Linear(512 + 128, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            # nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.LayerNorm(512)
        )
        
        # Enhanced shared representation
        self.shared_representation = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            # nn.Dropout(0.3),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024)
        )
        
        # self.classification_gate = AttentionGatingModule(1024)
        self.classification_gate = AttentionRefinementModule(1024)
        self.classification_encoder = TransformerFeatureEncoder(1024, num_heads=8, dropout=0.2)
        self.classification_branch = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            # nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # 2. Regression branch with TransformerFeatureEncoder
        # self.regression_encoder = TransformerFeatureEncoder(1024, num_heads=8, dropout=0.2)
        # self.regression_gate = AttentionGatingModule(1024)
        self.regression_gate = AttentionRefinementModule(1024)
        
        # Different architecture for regression-specific processing
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
        # self.class_names = ['AD', 'CN', 'MCI'] 
        
            
        # Metrics
        self.train_classification_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_classification_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_classification_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.mmse_mae = MeanAbsoluteError()
        # Metrics for label
        # self.test_accuracy_per_class = MulticlassAccuracy(num_classes=num_classes, average=None)
        # self.test_sensitivity_per_class = MulticlassRecall(num_classes=num_classes, average=None)
        # self.test_specificity_per_class = MulticlassSpecificity(num_classes=num_classes, average=None)

        self.classification_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.regression_loss = nn.HuberLoss()
        self.multi_task_optimizer = CombinedOptimizer(self, num_tasks=2, 
                                            frank_wolfe_weight=0.4,  
                                            alpha=1.5)

        if num_classes>2:
            self.specificity = MulticlassSpecificity(num_classes=num_classes)
            self.sensitivity = MulticlassRecall(num_classes=num_classes)
        else:
            self.specificity = BinarySpecificity()  
            self.sensitivity = BinaryRecall()
        
        # Enhanced tracking for visualization
        self.test_predictions = []
        self.test_labels = []
        self.test_mmse_true = []
        self.test_mmse_pred = []
        
        # self.test_mae = MeanAbsoluteError()
        # self.test_rmse = MeanSquaredError(squared=False) 
        # self.num_samples_per_class = [0] * num_classes
        
        self.loss_history = {
            'epochs': [],
            'total': [],
            'classification': [],
            'regression': [],
            'weights': [],
            'lambda_values': [], 
            'grad_norms_classification': [],
            'grad_norms_regression': []
        }
        
        self._init_weights()
        self.num_AD = 0
        self.num_CN = 0
        self.num_MCI = 0
        self.current_epoch_idx = 0
    
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
        x = self.backbone(image)
        image_features = x.squeeze(-1).squeeze(-1).squeeze(-1)  # Remove spatial dimensions
        
        # Process metadata
        metadata_features = self.metadata_embedding(metadata)
        
        # Feature fusion
        fused_features = self.cross_attention(torch.cat([image_features, metadata_features], dim=1))
        
        # Shared representation
        shared_features = self.shared_representation(fused_features)
        
        # Classification path - using attention gating
        classification_features = self.classification_gate(shared_features)
        classification_features = self.classification_encoder(classification_features)
        classification_output = self.classification_branch(classification_features)
        
        # Regression path - using transformer encoder
        # regression_features = self.regression_encoder(shared_features)
        regression_features = self.regression_gate(shared_features)
        regression_output = self.regression_branch(regression_features)
        
        return classification_output, regression_output

    def training_step(self, batch, batch_idx):
        image, label, mmse, age, gender = batch['image'], batch['label'], batch['mmse'], batch['age'], batch['gender']
        metadata = torch.stack([age, gender], dim=1).float()
        
        classification_output, regression_output = self(image, metadata)
        
        classification_loss = self.classification_loss(classification_output, label)
        regression_loss = self.regression_loss(regression_output.squeeze(), mmse)

        # Calculate gradient norms before the multi-task optimizer update
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
        
        shared_params = list(self.shared_representation.parameters())
    
        losses = [classification_loss.to(self.device), regression_loss.to(self.device)]
        weights = self.multi_task_optimizer.update_weights(losses, shared_params)
        
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
        lambda_info = self.multi_task_optimizer.get_lambda_info()
        
        # Log lambda value
        self.log('train_lambda', lambda_info['lambda'], on_step=True, on_epoch=True)
        self.log('train_gradient_difference', lambda_info['gradient_difference'], on_step=True, on_epoch=True)
        
        return total_loss
    
    def on_train_epoch_end(self):
        self.loss_history['epochs'].append(self.current_epoch_idx)
        
        # Get the current epoch's average loss values
        avg_total_loss = self.trainer.callback_metrics.get('train_loss_epoch', 0)
        avg_class_loss = self.trainer.callback_metrics.get('train_classification_loss_epoch', 0)
        avg_reg_loss = self.trainer.callback_metrics.get('train_regression_loss_epoch', 0)
        avg_lambda = self.trainer.callback_metrics.get('train_lambda_epoch', 0)
        
        # Convert from tensor to float if needed
        if hasattr(avg_total_loss, 'item'):
            avg_total_loss = avg_total_loss.item()
        if hasattr(avg_class_loss, 'item'):
            avg_class_loss = avg_class_loss.item()
        if hasattr(avg_reg_loss, 'item'):
            avg_reg_loss = avg_reg_loss.item()
        if hasattr(avg_lambda, 'item'):
            avg_lambda = avg_lambda.item()
            
        # Store for later plotting
        self.loss_history['total'].append(avg_total_loss)
        self.loss_history['classification'].append(avg_class_loss)
        self.loss_history['regression'].append(avg_reg_loss)
        self.loss_history['lambda_values'].append(avg_lambda)
        
        # Plot and log to wandb after each epoch
        self.plot_loss_curves()
        self.plot_lambda_curve()
        
        # Increment epoch counter
        self.current_epoch_idx += 1


    def plot_loss_curves(self):
        """Plot and log loss curves to wandb"""
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
    def plot_lambda_curve(self):
        """Plot and log lambda parameter changes to wandb"""
        if len(self.loss_history['epochs']) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.plot(self.loss_history['epochs'], self.loss_history['lambda_values'], 
                   label='Lambda Parameter', marker='o', color='purple')
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Lambda Value')
            ax.set_title('Lambda Parameter Changes Over Training')
            ax.set_ylim([0, 1])  # Lambda is between 0 and 1
            ax.grid(True)
            ax.legend()
            
            # Log to wandb
            self.logger.experiment.log({"Lambda Parameter Curve": wandb.Image(fig)})
            
            plt.close(fig)
            
            # # Also log the current task weights vs lambda in a scatter plot
            # if len(self.loss_history['epochs']) > 1:
            #     fig2, ax2 = plt.subplots(figsize=(10, 6))
                
            #     # Access the weights using multi_task_optimizer
            #     current_weights = self.multi_task_optimizer.weights.cpu().numpy()
            #     current_lambda = self.loss_history['lambda_values'][-1]
                
            #     # Create a scatter plot of weights vs lambda
            #     ax2.bar(['Classification', 'Regression'], current_weights, color=['blue', 'green'])
            #     ax2.set_title(f'Task Weights (Lambda = {current_lambda:.3f})')
            #     ax2.set_ylabel('Weight Value')
            #     ax2.set_ylim([0, 1])
                
            #     # Add lambda value as a horizontal line
            #     ax2.axhline(y=current_lambda, color='r', linestyle='--', 
            #                label=f'λ = {current_lambda:.3f}')
            #     ax2.legend()
                
            #     # Log to wandb
            #     self.logger.experiment.log({"Current Weights vs Lambda": wandb.Image(fig2)})
                
            #     plt.close(fig2)
        

    def validation_step(self, batch, batch_idx):
        image, label, mmse, age, gender = batch['image'], batch['label'], batch['mmse'], batch['age'], batch['gender']
        metadata = torch.stack([age, gender], dim=1).float()
        
        classification_output, regression_output = self(image, metadata)
        
        classification_loss = self.classification_loss(classification_output, label)
        regression_loss = self.regression_loss(regression_output.squeeze(), mmse)
        
        losses = [classification_loss.to(self.device), regression_loss.to(self.device)]
        weights = self.multi_task_optimizer.weights
        
        total_loss = torch.sum(torch.stack(losses) * weights)
        
        preds = torch.argmax(classification_output, dim=1)
        acc = (preds == label).float().mean()
        lambda_info = self.multi_task_optimizer.get_lambda_info()
        self.log('val_lambda', lambda_info['lambda'], on_epoch=True, sync_dist=True)
        
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
        
        # Track distribution of classes
        for ij in label:
            label1 = ij.item()
            if label1 == 0:
                self.num_AD += 1
            elif label1 == 1:
                self.num_CN += 1
            else:
                self.num_MCI += 1
        
        metadata = torch.stack([age, gender], dim=1).float()
        classification_output, regression_output = self(image, metadata)
        
        classification_loss = self.classification_loss(classification_output, label).to(self.device)
        regression_loss = self.regression_loss(regression_output.squeeze(), mmse).to(self.device)
        
        weights = self.multi_task_optimizer.weights
        total_loss = torch.sum(torch.stack([classification_loss, regression_loss]) * weights)
    
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
        self.log('final_classification_weight', weights[0], on_epoch=True)
        self.log('final_regression_weight', weights[1], on_epoch=True)
        self.log('test_specificity', spec, on_epoch=True, prog_bar=True)
        self.log('test_sensitivity', sens, on_epoch=True, prog_bar=True)
        
        # Store for confusion matrix and MMSE visualizations
        self.test_predictions.append(preds.cpu())
        self.test_labels.append(label.cpu())
        self.test_mmse_true.append(mmse.cpu())
        self.test_mmse_pred.append(regression_output.squeeze().cpu())
        # ==============================================================================
        # for class_idx in range(self.hparams.num_classes):
        #      self.num_samples_per_class[class_idx] += (label == class_idx).sum().item()
        
        # classification_output, regression_output = self(image, metadata)
        # preds = torch.argmax(classification_output, dim=1)
        # regression_pred_squeezed = regression_output.squeeze()
        
        # self.test_accuracy_per_class.update(preds, label)
        # self.test_sensitivity_per_class.update(preds, label)
        # self.test_specificity_per_class.update(preds, label)
        # self.test_mae.update(regression_pred_squeezed, mmse)
        # self.test_rmse.update(regression_pred_squeezed, mmse)
        # ==============================================================================
        return {
            'preds': preds,
            'true_labels': label,
            'predicted_mmse': regression_output.squeeze(),
            'true_mmse': mmse,
            'final_weights': weights.cpu(),
            'specificity': spec,
            'sensitivity': sens
        }
        
    def on_test_start(self):
        self.test_predictions = []
        self.test_labels = []
        self.test_mmse_true = []
        self.test_mmse_pred = []
        self.num_AD = 0
        self.num_CN = 0
        self.num_MCI = 0
        
    def on_test_end(self):
     
        print("============================================")
        print(f"Number of AD samples: {self.num_AD}")
        print(f"Number of CN samples: {self.num_CN}")
        print(f"Number of MCI samples: {self.num_MCI}")
        print("============================================")
        
        all_preds = torch.cat(self.test_predictions).numpy()
        all_labels = torch.cat(self.test_labels).numpy()
        
        all_true_mmse = torch.cat(self.test_mmse_true).numpy()
        
        # # Sửa lỗi: Đảm bảo mỗi tensor trong list có ít nhất 1 chiều
        # fixed_mmse_pred = []
        # for tensor in self.test_mmse_pred:
        #     if tensor.dim() == 0:  # Nếu là tensor 0 chiều
        #         fixed_mmse_pred.append(tensor.unsqueeze(0))  # Chuyển thành tensor 1 chiều
        #     else:
        #         fixed_mmse_pred.append(tensor)
        
        # all_pred_mmse = torch.cat(fixed_mmse_pred).numpy()
        
        # # Class names for confusion matrix
        # if hasattr(self, 'num_classes') and self.num_classes > 2:
        #     class_names = ["AD", "CN", "MCI"]
        # else:
        #     class_names = ["CN", "MCI"]
            
        # # Create and log MMSE prediction visualizations
        # self.plot_mmse_predictions(all_true_mmse, all_pred_mmse)
        fixed_mmse_pred = []
        for tensor in self.test_mmse_pred:
            if tensor.dim() == 0:  # Nếu là tensor 0 chiều
                fixed_mmse_pred.append(tensor.unsqueeze(0))  # Chuyển thành tensor 1 chiều
            else:
                fixed_mmse_pred.append(tensor)
        
        all_pred_mmse = torch.cat(fixed_mmse_pred).numpy()
        
        # Class names for confusion matrix
        if hasattr(self, 'num_classes') and self.num_classes > 2:
            class_names = ["AD", "CN", "MCI"]
        else:
            class_names = ["CN", "MCI"]
            
        # Create and log MMSE prediction visualizations
        self.plot_mmse_predictions(all_true_mmse, all_pred_mmse)
        
        # After other visualizations, create lambda vs weights visualization
        lambda_value = self.multi_task_optimizer.lambda_param
        # Fix: Detach weights tensor before converting to numpy
        weights = self.multi_task_optimizer.weights.detach().cpu().numpy()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.bar(['Classification', 'Regression'], weights, color=['blue', 'green'])
        ax.set_title(f'Final Task Weights (Lambda = {lambda_value:.3f})')
        ax.set_ylabel('Weight Value')
        ax.set_ylim([0, 1])
        
        # Add lambda value as a horizontal line
        ax.axhline(y=lambda_value, color='r', linestyle='--', 
                 label=f'λ = {lambda_value:.3f}')
        ax.legend()
        
        # Log to wandb
        self.logger.experiment.log({"Final Weights vs Lambda": wandb.Image(fig)})
        
        plt.close(fig)
        print("============= PER-CLASS METRICS =============")
        for i, class_name in enumerate(class_names):
            # Create binary labels for this class
            class_true = (all_labels == i).astype(int)
            class_pred = (all_preds == i).astype(int)
            
            # Calculate metrics
            class_acc = accuracy_score(class_true, class_pred)
            class_sensitivity = recall_score(class_true, class_pred, zero_division=0)
            class_specificity = recall_score(1-class_true, 1-class_pred, zero_division=0)
            
            # For MAE and RMSE, only consider samples from this class
            class_indices = np.where(all_labels == i)[0]
            if len(class_indices) > 0:
                class_mae = np.mean(np.abs(all_true_mmse[class_indices] - all_pred_mmse[class_indices]))
                class_rmse = np.sqrt(np.mean((all_true_mmse[class_indices] - all_pred_mmse[class_indices])**2))
            else:
                class_mae = float('nan')
                class_rmse = float('nan')
            
            print(f"Class {class_name}:")
            print(f"  Accuracy: {class_acc:.4f}")
            print(f"  Sensitivity: {class_sensitivity:.4f}")
            print(f"  Specificity: {class_specificity:.4f}")
            print(f"  MAE: {class_mae:.4f}")
            print(f"  RMSE: {class_rmse:.4f}")
        
    def plot_confusion_matrix(self, predictions, labels, class_names):
        """Plot and log confusion matrix to wandb"""
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
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