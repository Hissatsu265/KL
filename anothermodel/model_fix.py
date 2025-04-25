import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, MeanAbsoluteError
from torchvision.models.video import r3d_18, R3D_18_Weights

from multitask.visualize.visualize import visualize_and_save_gradcam, plot_and_save_optimization_metrics, plot_confusion_matrix
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
                 metadata_dim=2,
                 pretrained=True,
                 classification_weight=0.5,  # Weight for classification loss
                 regression_weight=0.5):     # Weight for regression loss
        super(MultiTaskAlzheimerModel, self).__init__()
        
        # Save loss weights
        self.classification_weight = classification_weight
        self.regression_weight = regression_weight
        
        if pretrained:
            print('Loading pretrained model===================================')
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

        if num_classes > 2:
            self.specificity = MulticlassSpecificity(num_classes=num_classes)
            self.sensitivity = MulticlassRecall(num_classes=num_classes)
        else:
            self.specificity = BinarySpecificity()  
            self.sensitivity = BinaryRecall()
            
        self.test_predictions = []
        self.test_labels = []
        self.test_true_mmse = []
        self.test_pred_mmse = []
        
        # Store loss history for plotting
        self.loss_history = {
            'classification': [],
            'regression': [],
            'total': [],
            'epochs': []
        }
        
        self._init_weights()
        self.num_AD = 0
        self.num_CN = 0
        self.num_MCI = 0
        self.current_epoch_num = 0
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, image, metadata):
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
        metadata = torch.stack([age, gender], dim=1).float()
        
        classification_output, regression_output = self(image, metadata)
        
        classification_loss = self.classification_loss(classification_output, label)
        regression_loss = self.regression_loss(regression_output.squeeze(), mmse)
        
        # Combine losses directly using weights
        total_loss = self.classification_weight * classification_loss + self.regression_weight * regression_loss
        
        preds = torch.argmax(classification_output, dim=1)
        acc = (preds == label).float().mean()
        
        # Logging
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_classification_loss', classification_loss, on_step=True, on_epoch=True)
        self.log('train_regression_loss', regression_loss, on_step=True, on_epoch=True)
        self.log('train_classification_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return total_loss

    def on_train_epoch_end(self):
        # Store epoch-level losses for plotting
        self.current_epoch_num += 1
        self.loss_history['epochs'].append(self.current_epoch_num)
        
        # Use the logged metrics
        if hasattr(self.trainer, 'callback_metrics'):
            metrics = self.trainer.callback_metrics
            self.loss_history['classification'].append(metrics.get('train_classification_loss_epoch', 0).item())
            self.loss_history['regression'].append(metrics.get('train_regression_loss_epoch', 0).item())
            self.loss_history['total'].append(metrics.get('train_loss_epoch', 0).item())
            
            # Log loss curves to wandb
            if self.logger and isinstance(self.logger, pl.loggers.WandbLogger):
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(self.loss_history['epochs'], self.loss_history['total'], label='Total Loss', marker='o')
                ax.plot(self.loss_history['epochs'], self.loss_history['classification'], label='Classification Loss', marker='s')
                ax.plot(self.loss_history['epochs'], self.loss_history['regression'], label='Regression Loss', marker='^')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title('Training Loss Convergence')
                ax.legend()
                ax.grid(True)
                
                # Log to wandb
                self.logger.experiment.log({"Loss Convergence": wandb.Image(fig)})
                plt.close(fig)

    def validation_step(self, batch, batch_idx):
        image, label, mmse, age, gender = batch['image'], batch['label'], batch['mmse'], batch['age'], batch['gender']
        metadata = torch.stack([age, gender], dim=1).float()
        
        classification_output, regression_output = self(image, metadata)
        
        classification_loss = self.classification_loss(classification_output, label)
        regression_loss = self.regression_loss(regression_output.squeeze(), mmse)
        
        # Direct sum of losses with weights
        total_loss = self.classification_weight * classification_loss + self.regression_weight * regression_loss
        
        preds = torch.argmax(classification_output, dim=1)
        acc = (preds == label).float().mean()
        
        # Logging
        self.log('val_loss', total_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_classification_loss', classification_loss, on_epoch=True, sync_dist=True)
        self.log('val_regression_loss', regression_loss, on_epoch=True, sync_dist=True)
        self.log('val_classification_acc', acc, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return total_loss

    def test_step(self, batch, batch_idx):
        image, label, mmse, age, gender = batch['image'], batch['label'], batch['mmse'], batch['age'], batch['gender']
        
        # Count class distribution
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
        
        # Direct sum with weights
        total_loss = self.classification_weight * classification_loss + self.regression_weight * regression_loss
    
        preds = torch.argmax(classification_output, dim=1)
        acc = (preds == label).float().mean()
        mae = F.l1_loss(regression_output.squeeze(), mmse)
        rmse = rmse_tt(regression_output.squeeze(), mmse)
        spec = self.specificity(preds, label)
        sens = self.sensitivity(preds, label)
        
        # Store predictions and true values for plotting
        self.test_predictions.append(preds.cpu())
        self.test_labels.append(label.cpu())
        self.test_pred_mmse.append(regression_output.squeeze().cpu())
        self.test_true_mmse.append(mmse.cpu())
        
        # Logging metrics
        self.log('test_total_loss', total_loss, on_epoch=True)
        self.log('test_classification_loss', classification_loss, on_epoch=True)
        self.log('test_regression_loss', regression_loss, on_epoch=True)
        self.log('test_classification_acc', acc, on_epoch=True, prog_bar=True)
        self.log('test_regression_mae', mae, on_epoch=True, prog_bar=True)
        self.log('test_regression_rmse', rmse, on_epoch=True, prog_bar=True)
        self.log('test_specificity', spec, on_epoch=True, prog_bar=True)
        self.log('test_sensitivity', sens, on_epoch=True, prog_bar=True)
        
        return {
            'preds': preds,
            'true_labels': label,
            'predicted_mmse': regression_output.squeeze(),
            'true_mmse': mmse,
            'specificity': spec,
            'sensitivity': sens
        }
        
    def on_test_start(self):
        # Clear previous predictions at the start of testing
        self.test_predictions = []
        self.test_labels = []
        self.test_pred_mmse = []
        self.test_true_mmse = []
        self.num_AD = 0
        self.num_CN = 0
        self.num_MCI = 0
        
    def on_test_end(self):
        print("============================================")
        print(f"Number of AD samples: {self.num_AD}")
        print(f"Number of CN samples: {self.num_CN}")
        print(f"Number of MCI samples: {self.num_MCI}")
        print("============================================")
      
        save_path = '/home/jupyter-iec_iot13_toanlm/multitask/visualize'
        os.makedirs(save_path, exist_ok=True)
        
        all_preds = torch.cat(self.test_predictions).numpy()
        all_labels = torch.cat(self.test_labels).numpy()
        all_pred_mmse = torch.cat(self.test_pred_mmse).numpy()
        all_true_mmse = torch.cat(self.test_true_mmse).numpy()
        
        if hasattr(self, 'num_classes') and self.num_classes > 2:
            class_names = ["AD", "CN", "MCI"]
        else:
            class_names = ["CN", "MCI"]
        
        # Plot confusion matrix
        plot_confusion_matrix(
            y_true=all_labels,
            y_pred=all_preds,
            save_path=save_path,
            classes=class_names
        )
        
        # Plot MMSE scatter plot in wandb
        if self.logger and isinstance(self.logger, pl.loggers.WandbLogger):
            # 1. Scatter Plot
            fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
            scatter = ax_scatter.scatter(all_true_mmse, all_pred_mmse, alpha=0.6)
            ax_scatter.plot([min(all_true_mmse), max(all_true_mmse)], [min(all_true_mmse), max(all_true_mmse)], 'r--')
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
            
        # You can still call the original optimization metrics if needed
        # plot_and_save_optimization_metrics(self, save_path)
        
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