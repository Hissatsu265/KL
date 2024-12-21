import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, MeanAbsoluteError
from torch.nn.init import xavier_uniform_

class SpatialAttentionGatingModule(nn.Module):
    def __init__(self, feature_dim):
        super(SpatialAttentionGatingModule, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv3d(feature_dim, feature_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(feature_dim // 2, feature_dim, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, shared_features, task_specific_features):
        # Đảm bảo kích thước giống nhau
        if shared_features.dim() == 2 and task_specific_features.dim() == 2:
            attention_input = shared_features.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        else:
            attention_input = shared_features
        
        # Compute spatial attention weights
        attention_weights = self.attention(attention_input)
        
        # Apply gating mechanism
        if shared_features.dim() == 2:
            gated_features = shared_features * attention_weights.squeeze()
        else:
            gated_features = shared_features * attention_weights
        
        return gated_features

class UncertaintyWeightedLoss(nn.Module):
    def __init__(self, num_tasks=2):
        super(UncertaintyWeightedLoss, self).__init__()
        # Log variance giúp mô hình tự học trọng số
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(self, losses):
        # Tính toán loss có trọng số dựa trên uncertainty
        precision_losses = [
            loss / (2 * torch.exp(log_var)) + log_var 
            for loss, log_var in zip(losses, self.log_vars)
        ]
        return sum(precision_losses)

class Efficient3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=6):
        super(Efficient3DBlock, self).__init__()
        
        # Thêm dropout và cải thiện normalization
        expanded_channels = in_channels * expand_ratio
        
        self.block = nn.Sequential(
            # Depthwise separable convolution
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.BatchNorm3d(in_channels),
            nn.ReLU6(inplace=True),
            nn.Dropout3d(p=0.1),  # Thêm dropout
            
            # Pointwise convolution
            nn.Conv3d(in_channels, expanded_channels, kernel_size=1),
            nn.BatchNorm3d(expanded_channels),
            nn.ReLU6(inplace=True),
            
            # Projection convolution
            nn.Conv3d(expanded_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            nn.Dropout3d(p=0.1)  # Thêm dropout
        )
        
        self.shortcut = in_channels == out_channels
        
    def forward(self, x):
        identity = x
        x = self.block(x)
        
        if self.shortcut:
            x += identity
        
        return x

class MultiTaskAlzheimerModel(pl.LightningModule):
    def __init__(self, 
                 num_classes=3,  # Thay đổi để hỗ trợ 3 labels 
                 input_shape=(1, 64, 64, 64), 
                 metadata_dim=3):
        super(MultiTaskAlzheimerModel, self).__init__()
        
        # Normalize metadata
        self.register_buffer('metadata_mean', torch.tensor([15.0, 70.0, 0.5]))  # Trung bình MMSE, Age, Gender
        self.register_buffer('metadata_std', torch.tensor([5.0, 10.0, 0.5]))   # Độ lệch chuẩn
        
        # Backbone và feature extraction
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
        
        # Shared representation với layer norm
        self.shared_representation = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),  # Thay thế BatchNorm bằng LayerNorm
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # Attention Gating Modules
        self.classification_gate = SpatialAttentionGatingModule(1024)
        self.regression_gate = SpatialAttentionGatingModule(1024)
        
        # Uncertainty weighted loss
        self.uncertainty_loss = UncertaintyWeightedLoss(num_tasks=2)
        
        # Task-specific branches
        self.classification_branch = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        self.regression_branch = nn.Sequential(
            nn.Linear(1024 + metadata_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1)  
        )
        
        # Metrics
        # self.classification_metrics = {
        #     'train': Accuracy(task='multiclass', num_classes=num_classes).to(self.device),
        #     'val': Accuracy(task='multiclass', num_classes=num_classes).to(self.device),
        #     'test': Accuracy(task='multiclass', num_classes=num_classes).to(self.device)
        # }
        self.classification_metrics = {
            'train': Accuracy(task='multiclass', num_classes=num_classes),
            'val': Accuracy(task='multiclass', num_classes=num_classes),
            'test': Accuracy(task='multiclass', num_classes=num_classes)
        }
        self.mmse_mae = MeanAbsoluteError()
        
        # Losses
        self.classification_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.MSELoss()
        
        # Khởi tạo trọng số
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv3d)):
            xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def normalize_metadata(self, metadata):
        # Chuẩn hóa metadata
        return (metadata - self.metadata_mean) / self.metadata_std
    
    def forward(self, x, metadata=None):
        if x.dim() == 4:
            x = x.unsqueeze(0)
        
        x = self.initial_conv(x)
        x = self.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
       
        shared_features = self.shared_representation(x)
    
        # Classification branch với gating
        classification_features = self.classification_gate(shared_features, shared_features)
        classification_output = self.classification_branch(classification_features)
        
        # Regression branch với gating và metadata
        if metadata is not None:
            # Normalize metadata
            normalized_metadata = self.normalize_metadata(metadata)
            regression_features = self.regression_gate(shared_features, shared_features)
            regression_input = torch.cat([regression_features, normalized_metadata], dim=1)
            regression_output = self.regression_branch(regression_input)
        else:
            regression_output = None
        
        return classification_output, regression_output
    
    def training_step(self, batch, batch_idx):
        images = batch['image']
        labels = batch['label']
        mmse = batch['mmse']
        age = batch['age']
        gender = batch['gender']
        
        # Chuyển đổi metadata
        metadata = torch.stack([mmse, age, gender], dim=1).float()
        
        # Forward pass
        classification_output, regression_output = self(images, metadata)
        
        # Tính toán loss
        classification_loss = self.classification_loss(classification_output, labels)
        regression_loss = self.regression_loss(regression_output.squeeze(), mmse)
        
        # Sử dụng uncertainty weighted loss
        total_loss = self.uncertainty_loss([classification_loss, regression_loss])
        
        # Logging
        preds = torch.argmax(classification_output, dim=1)
        # acc = self.classification_metrics['train'](preds.to(self.device), labels.to(self.device))
        acc = self.classification_metrics['train'].to(self.device)(preds, labels)
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
        # acc = (preds == labels).float().mean()
        # acc = self.classification_metrics['val'](preds.to(self.device), labels.to(self.device))        
        acc = self.classification_metrics['train'].to(self.device)(preds, labels)
        self.log('val_loss', total_loss, on_epoch=True, prog_bar=True)
        self.log('val_classification_loss', classification_loss, on_epoch=True)
        self.log('val_regression_loss', regression_loss, on_epoch=True)
        self.log('val_classification_acc', acc, on_epoch=True, prog_bar=True)
        
        return total_loss
    
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
        # acc = (preds == labels).float().mean()
        # acc = self.classification_metrics['test'](preds.to(self.device), labels.to(self.device))    
        acc = self.classification_metrics['train'].to(self.device)(preds, labels)
        # Regression metrics
        mae = F.l1_loss(regression_output.squeeze(), mmse)
        
        self.log('test_classification_acc', acc, on_epoch=True, prog_bar=True)
        self.log('test_regression_mae', mae, on_epoch=True, prog_bar=True)
        
        return {'preds': preds, 'true_labels': labels}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=1e-3, 
            weight_decay=1e-5  # Thêm weight decay để giảm overfitting
        )
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