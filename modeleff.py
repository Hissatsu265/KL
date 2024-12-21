import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, MeanAbsoluteError
from typing import Optional, Dict, Any

class SELayer3D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class Efficient3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=6, dropout_rate=0.2):
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
        
        # Squeeze-and-Excitation Layer
        self.se_layer = SELayer3D(out_channels)
        
        # Dropout
        self.dropout = nn.Dropout3d(p=dropout_rate)
        
        # Shortcut connection
        self.shortcut = in_channels == out_channels
        
    def forward(self, x):
        identity = x
        
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.projection_conv(x)
        x = self.se_layer(x)
        x = self.dropout(x)
        
        if self.shortcut:
            x += identity
        
        return x

class AttentionGatingModule(nn.Module):
    def __init__(self, feature_dim, reduction=4):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // reduction),
            nn.LayerNorm(feature_dim // reduction),
            nn.GELU(),
            nn.Linear(feature_dim // reduction, feature_dim),
            nn.Sigmoid()
        )
    
    def forward(self, shared_features, task_specific_features):
        # Ensure same dimensionality
        if shared_features.size(1) != task_specific_features.size(1):
            task_specific_features = F.linear(
                task_specific_features, 
                torch.eye(shared_features.size(1)).to(task_specific_features.device)
            )
        
        attention_weights = self.attention(shared_features)
        
        # Apply gating mechanism with soft interpolation
        gated_features = (shared_features * attention_weights + 
                          task_specific_features * (1 - attention_weights))
        
        return gated_features

class MultiTaskAlzheimerModel(pl.LightningModule):
    def __init__(self, 
                 num_classes=2, 
                 input_shape=(1, 64, 64, 64), 
                 metadata_dim=3,
                 learning_rate=1e-3):
        super().__init__()
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        
        # Initial Convolution
        self.initial_conv = nn.Sequential(
            nn.Conv3d(input_shape[0], 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        
        # Enhanced Feature Extraction
        self.features = nn.Sequential(
            Efficient3DBlock(32, 64, expand_ratio=4),
            nn.MaxPool3d(2, 2),
            Efficient3DBlock(64, 128, expand_ratio=6),
            nn.MaxPool3d(2, 2),
            Efficient3DBlock(128, 256, expand_ratio=8),
            nn.MaxPool3d(2, 2),
            Efficient3DBlock(256, 512, expand_ratio=10),
            nn.MaxPool3d(2, 2)
        )
        
        # Global Pooling
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # Shared Representation with Enhanced Regularization
        self.shared_representation = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6)
        )
        
        # Attention Gating Modules
        self.classification_gate = AttentionGatingModule(1024)
        self.regression_gate = AttentionGatingModule(1024)
        
        # Task-Specific Branches
        # Classification Branch
        self.classification_branch = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Regression Branch (MMSE Score)
        self.regression_branch = nn.Sequential(
            nn.Linear(1024 + metadata_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
        
        # Metrics
        self.train_classification_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_classification_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_classification_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.mmse_mae = MeanAbsoluteError()
        
        # Losses with Label Smoothing and Huber Loss
        self.classification_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.regression_loss = nn.HuberLoss(delta=1.0)
        
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
        classification_features = self.classification_gate(shared_features, shared_features)
        classification_output = self.classification_branch(classification_features)
        
        # Regression branch with gating and metadata
        if metadata is not None:
            regression_features = self.regression_gate(shared_features, shared_features)
            regression_input = torch.cat([regression_features, metadata], dim=1)
            regression_output = self.regression_branch(regression_input)
        else:
            regression_output = None
        
        return classification_output, regression_output
    
    def _common_step(self, batch, batch_idx, stage):
        # Unpack batch
        images = batch['image']
        labels = batch['label']
        mmse = batch['mmse']
        age = batch['age']
        gender = batch['gender']
    
        # Convert to tensor if needed
        mmse = torch.tensor(float(mmse)).to(images.device) if not isinstance(mmse, torch.Tensor) else mmse
        age = torch.tensor(float(age)).to(images.device) if not isinstance(age, torch.Tensor) else age
        gender = torch.tensor(float(gender)).to(images.device) if not isinstance(gender, torch.Tensor) else gender
        
        # Prepare metadata
        metadata = torch.stack([mmse, age, gender], dim=1).float()
        
        # Forward pass
        classification_output, regression_output = self(images, metadata)
        
        # Classification loss
        classification_loss = self.classification_loss(classification_output, labels)
        
        # Regression loss
        regression_loss = self.regression_loss(regression_output.squeeze(), mmse)
        
        # Adaptive loss weighting
        alpha = 0.5  
        total_loss = alpha * classification_loss + (1 - alpha) * regression_loss
        
        # Compute metrics
        preds = torch.argmax(classification_output, dim=1)
        acc = (preds == labels).float().mean()
        
        # Logging
        metrics = {
            f'{stage}_loss': total_loss,
            f'{stage}_classification_loss': classification_loss,
            f'{stage}_regression_loss': regression_loss,
            f'{stage}_classification_acc': acc
        }
        
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True)
        
        return total_loss
    
    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'val')
    
    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'test')
    
    def configure_optimizers(self):
        # Use AdamW with weight decay
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate, 
            weight_decay=1e-5
        )
        
        # Cosine Annealing with Warm Restarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10,  # Initial restart period
            T_mult=2,  # Exponential restart period increase
            eta_min=1e-5  # Minimum learning rate
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }