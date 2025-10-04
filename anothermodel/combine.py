import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, MeanAbsoluteError
from torchvision.models.video import r3d_18, R3D_18_Weights

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

class GradNormFrankWolfeOptimizer:
    """
    Combined GradNorm + Frank-Wolfe optimizer for multi-task learning
    """
    def __init__(self, model, num_tasks=2, alpha=1.5):
        self.model = model
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.initial_losses = None
        
        # Initialize task weights uniformly on the simplex
        self.weights = torch.ones(num_tasks) / num_tasks
        self.iteration = 0
        
    def to(self, device):
        self.weights = self.weights.to(device)
        if self.initial_losses is not None:
            self.initial_losses = self.initial_losses.to(device)
        return self
    
    def set_initial_losses(self, initial_losses):
        """Set initial losses L_i(0) for normalization"""
        self.initial_losses = torch.stack([loss.detach().clone() for loss in initial_losses])
        if self.weights.device != torch.device('cpu'):
            self.initial_losses = self.initial_losses.to(self.weights.device)
    
    def get_shared_parameters(self):
        """Get parameters of the last shared layer"""
        return list(self.model.shared_representation.parameters())
    
    def compute_gradient_norms(self, task_losses, shared_params):
        """Compute gradient norms G_W for each task"""
        grad_norms = []
        
        for i in range(self.num_tasks):
            # Calculate gradient of w_i * L_i w.r.t. shared parameters
            weighted_loss = self.weights[i] * task_losses[i]
            
            # Compute gradients
            grads = torch.autograd.grad(
                outputs=weighted_loss,
                inputs=shared_params,
                retain_graph=True,
                create_graph=False  # Don't need second-order gradients for Frank-Wolfe
            )
            
            # Calculate gradient norm
            grad_norm = torch.norm(torch.stack([torch.norm(g) for g in grads]))
            grad_norms.append(grad_norm)
        
        return torch.stack(grad_norms)
    
    def update_weights(self, task_losses, shared_params):
        """Update task weights using GradNorm + Frank-Wolfe algorithm"""
        
        # Set initial losses if not already set
        if self.initial_losses is None:
            self.set_initial_losses(task_losses)
            return self.weights.clone()
        
        # 1. & 2. Compute GradNorm Components
        grad_norms = self.compute_gradient_norms(task_losses, shared_params)
        
        # 3. Calculate relative learning rates and targets
        current_losses = torch.stack([loss.detach() for loss in task_losses])
        normalized_losses = current_losses / (self.initial_losses + 1e-8)
        relative_rates = normalized_losses / torch.mean(normalized_losses)
        
        # Calculate average gradient norm (as constant)
        avg_grad_norm = torch.mean(grad_norms).detach()
        
        # Calculate targets for each task
        targets = avg_grad_norm * (relative_rates ** self.alpha)
        
        # 4. Determine Frank-Wolfe Direction s_k
        priority_scores = targets - grad_norms
        best_task_index = torch.argmax(priority_scores)
        
        # Create one-hot vector for Frank-Wolfe direction
        s_k = torch.zeros(self.num_tasks, device=self.weights.device)
        s_k[best_task_index] = 1.0
        
        # 5. Update Weights with Frank-Wolfe step
        gamma = 2.0 / (self.iteration + 2.0)
        
        with torch.no_grad():
            self.weights = (1.0 - gamma) * self.weights + gamma * s_k
        
        self.iteration += 1
        
        return self.weights.clone()

class AttentionGatingModule(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionGatingModule, self).__init__()
        self.attention = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim // 2, feature_dim),
            nn.Sigmoid()
        )
        
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
        
        residual = self.residual(gated_features)
        return gated_features + residual

class MultiTaskAlzheimerModel(pl.LightningModule):
    def __init__(self, 
                 num_classes=2, 
                 input_shape=(1, 64, 64, 64),
                 metadata_dim=2,
                 pretrained=True,
                 gradnorm_alpha=1.5):
        super(MultiTaskAlzheimerModel, self).__init__()
        
        if pretrained:
            print('Loading pretrained R3D-18 model...')
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
        
        # Enhanced shared representation (last shared layer)
        self.shared_representation = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024)
        )
        
        # Task-specific branches with enhanced attention
        self.classification_gate = AttentionGatingModule(1024)
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
        
        # Initialize GradNorm + Frank-Wolfe optimizer
        self.gradnorm_fw_optimizer = GradNormFrankWolfeOptimizer(
            model=self, 
            num_tasks=2, 
            alpha=gradnorm_alpha
        )

        if num_classes > 2:
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
        
        # Enhanced loss history tracking
        self.loss_history = {
            'epochs': [],
            'total': [],
            'classification': [],
            'regression': [],
            'weights': [],
            'frank_wolfe_iterations': []
        }
        
        self._init_weights()
        self.num_AD = 0
        self.num_CN = 0
        self.num_MCI = 0
        self.current_epoch_idx = 0
        self.initial_batch_processed = False
    
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
        self.gradnorm_fw_optimizer.to(self.device)

    def forward(self, image, metadata):
        x = self.backbone(image)
        image_features = x.squeeze(-1).squeeze(-1).squeeze(-1)
        
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
        metadata = torch.stack([age, gender], dim=1).float()
        
        classification_output, regression_output = self(image, metadata)
        
        classification_loss = self.classification_loss(classification_output, label)
        regression_loss = self.regression_loss(regression_output.squeeze(), mmse)
        
        task_losses = [classification_loss, regression_loss]
        
        # Get shared parameters (last shared layer)
        shared_params = self.gradnorm_fw_optimizer.get_shared_parameters()
        
        # Set initial losses for the first batch
        if not self.initial_batch_processed:
            self.gradnorm_fw_optimizer.set_initial_losses(task_losses)
            self.initial_batch_processed = True
        
        # Update weights using GradNorm + Frank-Wolfe
        weights = self.gradnorm_fw_optimizer.update_weights(task_losses, shared_params)
        
        # Compute combined loss
        total_loss = torch.sum(torch.stack(task_losses) * weights)
        
        # Calculate metrics
        preds = torch.argmax(classification_output, dim=1)
        acc = (preds == label).float().mean()
        
        # Logging
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_classification_loss', classification_loss, on_step=True, on_epoch=True)
        self.log('train_regression_loss', regression_loss, on_step=True, on_epoch=True)
        self.log('train_classification_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_classification_weight', weights[0], on_step=True, on_epoch=True)
        self.log('train_regression_weight', weights[1], on_step=True, on_epoch=True)
        self.log('frank_wolfe_iteration', self.gradnorm_fw_optimizer.iteration, on_step=True, on_epoch=True)
        
        # Log to wandb
        if self.logger:
            wandb.log({
                "train/total_loss": total_loss.item(),
                "train/classification_loss": classification_loss.item(),
                "train/regression_loss": regression_loss.item(),
                "train/accuracy": acc.item(),
                "train/classification_weight": weights[0].item(),
                "train/regression_weight": weights[1].item(),
                "train/weight_ratio": (weights[0] / weights[1]).item(),
                "train/frank_wolfe_iteration": self.gradnorm_fw_optimizer.iteration,
                "global_step": self.global_step
            })
        
        return total_loss
    
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
        self.loss_history['frank_wolfe_iterations'].append(self.gradnorm_fw_optimizer.iteration)
        
        # Plot and log to wandb after each epoch
        self.plot_loss_curves()
        self.plot_weight_evolution()
        
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
            ax.set_title('Training Loss Convergence (GradNorm + Frank-Wolfe)')
            ax.legend()
            ax.grid(True)
            
            # Log to wandb
            if self.logger:
                self.logger.experiment.log({"Training Loss Curves": wandb.Image(fig)})
        
        plt.close(fig)
    
    def plot_weight_evolution(self):
        """Plot weight evolution over Frank-Wolfe iterations"""
        if len(self.loss_history['frank_wolfe_iterations']) > 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Get current weights
            current_weights = self.gradnorm_fw_optimizer.weights.cpu().numpy()
            
            # Plot weight evolution (simplified - showing current weights)
            ax.bar(['Classification', 'Regression'], current_weights, 
                  color=['blue', 'red'], alpha=0.7)
            ax.set_ylabel('Weight')
            ax.set_title(f'Task Weights at Iteration {self.gradnorm_fw_optimizer.iteration}')
            ax.set_ylim(0, 1)
            ax.grid(True, axis='y')
            
            # Add text annotations
            for i, weight in enumerate(current_weights):
                ax.text(i, weight + 0.02, f'{weight:.3f}', ha='center', va='bottom')
            
            # Log to wandb
            if self.logger:
                self.logger.experiment.log({"Task Weight Evolution": wandb.Image(fig)})
            
            plt.close(fig)

    def validation_step(self, batch, batch_idx):
        image, label, mmse, age, gender = batch['image'], batch['label'], batch['mmse'], batch['age'], batch['gender']
        metadata = torch.stack([age, gender], dim=1).float()
        
        classification_output, regression_output = self(image, metadata)
        
        classification_loss = self.classification_loss(classification_output, label)
        regression_loss = self.regression_loss(regression_output.squeeze(), mmse)
        
        # Use current weights for validation loss computation
        weights = self.gradnorm_fw_optimizer.weights
        total_loss = torch.sum(torch.stack([classification_loss, regression_loss]) * weights)
        
        preds = torch.argmax(classification_output, dim=1)
        acc = (preds == label).float().mean()
        
        # Logging
        self.log('val_loss', total_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_classification_loss', classification_loss, on_epoch=True, sync_dist=True)
        self.log('val_regression_loss', regression_loss, on_epoch=True, sync_dist=True)
        self.log('val_classification_acc', acc, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_classification_weight', weights[0], on_epoch=True, sync_dist=True)
        self.log('val_regression_weight', weights[1], on_epoch=True, sync_dist=True)
        
        if self.logger:
            wandb.log({
                "val/total_loss": total_loss.item(),
                "val/classification_loss": classification_loss.item(),
                "val/regression_loss": regression_loss.item(),
                "val/accuracy": acc.item(),
                "epoch": self.current_epoch_idx
            })

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
        
        classification_loss = self.classification_loss(classification_output, label)
        regression_loss = self.regression_loss(regression_output.squeeze(), mmse)
        
        weights = self.gradnorm_fw_optimizer.weights
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
        self.log('final_frank_wolfe_iterations', self.gradnorm_fw_optimizer.iteration, on_epoch=True)
        self.log('test_specificity', spec, on_epoch=True, prog_bar=True)
        self.log('test_sensitivity', sens, on_epoch=True, prog_bar=True)
        
        # Store for confusion matrix and MMSE visualizations
        self.test_predictions.append(preds.cpu())
        self.test_labels.append(label.cpu())
        self.test_mmse_true.append(mmse.cpu())
        self.test_mmse_pred.append(regression_output.squeeze().cpu())

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
        print(f"Total Frank-Wolfe iterations: {self.gradnorm_fw_optimizer.iteration}")
        print("============================================")
        
        # Collect all test predictions and ground truths
        all_preds = torch.cat(self.test_predictions).numpy()
        all_labels = torch.cat(self.test_labels).numpy()
        
        # MMSE predictions visualization
        all_true_mmse = torch.cat(self.test_mmse_true).numpy()
        all_pred_mmse = torch.cat(self.test_mmse_pred).numpy()
        
        self.plot_mmse_predictions(all_true_mmse, all_pred_mmse)

        
    def plot_mmse_predictions(self, all_true_mmse, all_pred_mmse):
        """Create and log MMSE prediction visualizations to wandb"""
        
        # 1. Scatter Plot: True vs Predicted MMSE
        fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
        scatter = ax_scatter.scatter(all_true_mmse, all_pred_mmse, alpha=0.6)
        ax_scatter.plot([min(all_true_mmse), max(all_true_mmse)], 
                       [min(all_true_mmse), max(all_true_mmse)], 'r--')
        ax_scatter.set_xlabel('True MMSE Score')
        ax_scatter.set_ylabel('Predicted MMSE Score')
        ax_scatter.set_title('MMSE Prediction: True vs Predicted (GradNorm + Frank-Wolfe)')
        ax_scatter.grid(True)
        if self.logger:
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
        if self.logger:
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
        if self.logger:
            self.logger.experiment.log({"MMSE Distribution Plot": wandb.Image(fig_dist)})
        plt.close(fig_dist)
        
        # 4. Additional metrics as a table
        mmse_metrics = {
            "MMSE MAE": float(F.l1_loss(torch.tensor(all_pred_mmse), torch.tensor(all_true_mmse))),
            "MMSE RMSE": float(torch.sqrt(F.mse_loss(torch.tensor(all_pred_mmse), torch.tensor(all_true_mmse)))),
            "MMSE Correlation": float(np.corrcoef(all_true_mmse, all_pred_mmse)[0, 1])
        }

    def configure_optimizers(self):
        # Only optimize model parameters (task weights handled by Frank-Wolfe)
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