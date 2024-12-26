import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, MeanAbsoluteError
from torchmetrics import (
    Precision,
    Recall,
    F1Score
)
class AttentionGatingModule(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionGatingModule, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim),
            nn.Sigmoid()
        )
    
    def forward(self, shared_features, task_specific_features):
        # Đảm bảo kích thước giống nhau
        if shared_features.size(1) != task_specific_features.size(1):
            # Resize task_specific_features để khớp với shared_features
            task_specific_features = F.linear(task_specific_features, 
                                              torch.eye(shared_features.size(1)).to(task_specific_features.device))
        
        # Compute attention weights
        attention_weights = self.attention(shared_features)
        
        # Apply gating mechanism
        gated_features = shared_features * attention_weights + \
                         task_specific_features * (1 - attention_weights)
        
        return gated_features

class MultiTaskAlzheimerModel(pl.LightningModule):
    def __init__(self, 
                 num_classes=2, 
                 input_shape=(1, 64, 64, 64), 
                 metadata_dim=3):
        super(MultiTaskAlzheimerModel, self).__init__()
        
        # Backbone Efficient3D Feature Extractor
        self.initial_conv = nn.Sequential(
            nn.Conv3d(input_shape[0], 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        
        self.features = nn.Sequential(
            Efficient3DBlock(32, 64),
            nn.MaxPool3d(2, 2),
            Efficient3DBlock(64, 128),   
            nn.MaxPool3d(2, 2),
            Efficient3DBlock(128, 256),  
            nn.MaxPool3d(2, 2),
            Efficient3DBlock(256, 512),  
            nn.MaxPool3d(2, 2)
        )
        
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # Shared representation layer
        self.shared_representation = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # Attention Gating Modules
        self.classification_gate = AttentionGatingModule(1024)
        self.regression_gate = AttentionGatingModule(1024)
        
        # Task-specific branches
        # Classification Branch
        self.classification_branch = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Regression Branch (MMSE Score)
        self.regression_branch = nn.Sequential(
            nn.Linear(1024 + metadata_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1)  # Single output for MMSE score
        )
        
        # Metrics
        self.train_classification_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_classification_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_classification_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.mmse_mae = MeanAbsoluteError()
        # Metrics cho classification
        self.test_classification_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_precision = Precision(task='multiclass', num_classes=num_classes, average='macro')
        self.test_recall = Recall(task='multiclass', num_classes=num_classes, average='macro')
        self.test_f1 = F1Score(task='multiclass', num_classes=num_classes, average='macro')
    
        # Losses
        self.classification_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.MSELoss()
        
    def forward(self, x, metadata=None):
        if x.dim() == 4:
            x = x.unsqueeze(0)  # Add batch dimension if missing
    
        # Image feature extraction
        x = self.initial_conv(x)
        x = self.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
       
        shared_features = self.shared_representation(x)
    
        # Classification branch with gating
        classification_features = self.classification_gate(shared_features, shared_features)  # Thay đổi ở đây
        classification_output = self.classification_branch(classification_features)
        
        # Regression branch with gating and metadata
        if metadata is not None:
            # regression_features = self.regression_gate(shared_features, x)
            regression_features = self.regression_gate(shared_features, shared_features)
            regression_input = torch.cat([regression_features, metadata], dim=1)
            regression_output = self.regression_branch(regression_input)
        else:
            regression_output = None
        
        return classification_output, regression_output
    
    def training_step(self, batch, batch_idx):
        # Unpack batch
        images = batch['image']
        labels = batch['label']
        mmse = batch['mmse']
        age = batch['age']
        gender = batch['gender']
    
        # Chuyển đổi sang Tensor nếu cần
        mmse = torch.tensor(float(mmse)) if not isinstance(mmse, torch.Tensor) else mmse
        age = torch.tensor(float(age)) if not isinstance(age, torch.Tensor) else age
        gender = torch.tensor(float(gender)) if not isinstance(gender, torch.Tensor) else gender
        
        # Prepare metadata
        metadata = torch.stack([mmse, age, gender], dim=1).float()
        
        # Forward pass
        classification_output, regression_output = self(images, metadata)
        
        # Classification loss
        classification_loss = self.classification_loss(classification_output, labels)
        
        # Regression loss
        regression_loss = self.regression_loss(regression_output.squeeze(), mmse)
        
        # Combined loss with optional weighting
        total_loss = classification_loss + 0.5 * regression_loss
        
        # Logging
        preds = torch.argmax(classification_output, dim=1)
        acc = (preds == labels).float().mean()
        
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_classification_loss', classification_loss, on_step=True, on_epoch=True)
        self.log('train_regression_loss', regression_loss, on_step=True, on_epoch=True)
        self.log('train_classification_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        # Unpack batch
        images = batch['image']
        labels = batch['label']
        mmse = batch['mmse']
        age = batch['age']
        gender = batch['gender']
    
        # Chuyển đổi sang Tensor nếu cần
        mmse = torch.tensor(float(mmse)) if not isinstance(mmse, torch.Tensor) else mmse
        age = torch.tensor(float(age)) if not isinstance(age, torch.Tensor) else age
        gender = torch.tensor(float(gender)) if not isinstance(gender, torch.Tensor) else gender
        
        # Prepare metadata
        metadata = torch.stack([mmse, age, gender], dim=1).float()
        
        # Forward pass
        classification_output, regression_output = self(images, metadata)
        
        # Classification loss
        classification_loss = self.classification_loss(classification_output, labels)
        
        # Regression loss
        regression_loss = self.regression_loss(regression_output.squeeze(), mmse)
        
        # Combined loss
        total_loss = classification_loss + 0.5 * regression_loss
        
        # Logging
        preds = torch.argmax(classification_output, dim=1)
        acc = (preds == labels).float().mean()
        
        self.log('val_loss', total_loss, on_epoch=True, prog_bar=True)
        self.log('val_classification_loss', classification_loss, on_epoch=True)
        self.log('val_regression_loss', regression_loss, on_epoch=True)
        self.log('val_classification_acc', acc, on_epoch=True, prog_bar=True)
        
        return total_loss
    
    # def test_step(self, batch, batch_idx):
    #     # Unpack batch
    #     images = batch['image']
    #     labels = batch['label']
    #     mmse = batch['mmse']
    #     age = batch['age']
    #     gender = batch['gender']
    
    #     # Chuyển đổi sang Tensor nếu cần
    #     mmse = torch.tensor(float(mmse)) if not isinstance(mmse, torch.Tensor) else mmse
    #     age = torch.tensor(float(age)) if not isinstance(age, torch.Tensor) else age
    #     gender = torch.tensor(float(gender)) if not isinstance(gender, torch.Tensor) else gender
        
    #     # Prepare metadata
    #     metadata = torch.stack([mmse, age, gender], dim=1).float()
        
    #     # Forward pass
    #     classification_output, regression_output = self(images, metadata)
        
    #     # Classification metrics
    #     preds = torch.argmax(classification_output, dim=1)
    #     acc = (preds == labels).float().mean()
        
    #     # Regression metrics
    #     mae = F.l1_loss(regression_output.squeeze(), mmse)
        
    #     self.log('test_classification_acc', acc, on_epoch=True, prog_bar=True)
    #     self.log('test_regression_mae', mae, on_epoch=True, prog_bar=True)
        
    #     return {'preds': preds, 'true_labels': labels}
    def test_step(self, batch, batch_idx):
        # Unpack batch
        images = batch['image']
        labels = batch['label']
        mmse = batch['mmse']
        age = batch['age']
        gender = batch['gender']
    
        # Chuyển đổi sang Tensor nếu cần
        mmse = torch.tensor(float(mmse)) if not isinstance(mmse, torch.Tensor) else mmse
        age = torch.tensor(float(age)) if not isinstance(age, torch.Tensor) else age
        gender = torch.tensor(float(gender)) if not isinstance(gender, torch.Tensor) else gender
        
        # Prepare metadata
        metadata = torch.stack([mmse, age, gender], dim=1).float()
        
        # Forward pass
        classification_output, regression_output = self(images, metadata)
        
        # Classification metrics
        preds = torch.argmax(classification_output, dim=1)
        
        # Classification loss
        classification_loss = self.classification_loss(classification_output, labels)
        
        # Regression loss and metrics
        regression_loss = self.regression_loss(regression_output.squeeze(), mmse)
        mae = F.l1_loss(regression_output.squeeze(), mmse)
        
        # Calculate additional classification metrics
        accuracy = self.test_classification_accuracy(preds, labels)
        precision = self.test_precision(preds, labels)
        recall = self.test_recall(preds, labels)
        f1 = self.test_f1(preds, labels)
        
        # Total loss
        total_loss = classification_loss + 0.5 * regression_loss
        
        # Log all metrics
        self.log('test_total_loss', total_loss, on_epoch=True)
        self.log('test_classification_loss', classification_loss, on_epoch=True)
        self.log('test_regression_loss', regression_loss, on_epoch=True)
        self.log('test_classification_acc', accuracy, on_epoch=True)
        self.log('test_precision', precision, on_epoch=True)
        self.log('test_recall', recall, on_epoch=True)
        self.log('test_f1', f1, on_epoch=True)
        self.log('test_regression_mae', mae, on_epoch=True)
        
        # Return dictionary with all metrics for later analysis
        return {
            'preds': preds,
            'true_labels': labels,
            'total_loss': total_loss,
            'classification_loss': classification_loss,
            'regression_loss': regression_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mae': mae,
            'mmse_pred': regression_output.squeeze().detach(),
            'mmse_true': mmse
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

# Reuse the Efficient3DBlock from the previous implementation
class Efficient3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=6):
        super(Efficient3DBlock, self).__init__()
        
        # Depthwise separable convolution
        self.depthwise_conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.BatchNorm3d(in_channels),
            nn.ReLU6(inplace=True)
        )
        
        # Pointwise convolution with expansion
        expanded_channels = in_channels * expand_ratio
        self.pointwise_conv = nn.Sequential(
            nn.Conv3d(in_channels, expanded_channels, kernel_size=1),
            nn.BatchNorm3d(expanded_channels),
            nn.ReLU6(inplace=True)
        )
        
        # Projection convolution
        self.projection_conv = nn.Sequential(
            nn.Conv3d(expanded_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels)
        )
        
        # Shortcut connection
        self.shortcut = in_channels == out_channels
        
    def forward(self, x):
        identity = x
        
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.projection_conv(x)
        
        if self.shortcut:
            x += identity
        
        return x