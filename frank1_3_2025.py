import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, MeanAbsoluteError
from torchvision.models.video import r3d_18

class GradientScaleLayer(nn.Module):
    def __init__(self, scale):
        super(GradientScaleLayer, self).__init__()
        self.scale = scale
        
    def forward(self, x):
        return x * self.scale

class AttentionGatingModule(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionGatingModule, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim),
            nn.Sigmoid()
        )
        # Add squeeze-and-excitation
        self.se = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 16),
            nn.ReLU(),
            nn.Linear(feature_dim // 16, feature_dim),
            nn.Sigmoid()
        )

    def forward(self, shared_features, task_specific_features):
        # Original attention mechanism
        attention_weights = self.attention(shared_features)
        # Add squeeze-and-excitation
        se_weights = self.se(shared_features)
        
        gated_features = shared_features * attention_weights * se_weights + \
                        task_specific_features * (1 - attention_weights)
        return gated_features

class AdaptiveFrankWolfeLoss(nn.Module):
    def __init__(self, num_tasks=2, device='cuda'):
        super(AdaptiveFrankWolfeLoss, self).__init__()
        self.weights = nn.Parameter(torch.ones(num_tasks).to(device) / num_tasks)
        self.num_tasks = num_tasks
        self.ema_losses = None
        self.momentum = 0.9
        
    def update_ema(self, losses):
        losses_tensor = torch.stack([l.detach() for l in losses])
        if self.ema_losses is None:
            self.ema_losses = losses_tensor
        else:
            self.ema_losses = self.momentum * self.ema_losses + (1 - self.momentum) * losses_tensor
    
    def compute_frank_wolfe_step(self, grads):
        # Normalize gradients by EMA
        if self.ema_losses is not None:
            grads = grads / (self.ema_losses + 1e-8)
        
        min_idx = torch.argmin(grads)
        s = torch.zeros_like(self.weights)
        s[min_idx] = 1.0
        return s
    
    def forward(self, losses, iteration, total_iterations):
        self.update_ema(losses)
        losses_tensor = torch.stack(losses)
        grads = losses_tensor.detach()
        
        s = self.compute_frank_wolfe_step(grads)
        gamma = 2.0 / (iteration + 2.0)
        
        with torch.no_grad():
            new_weights = (1 - gamma) * self.weights + gamma * s
            self.weights.copy_(new_weights)
        
        weighted_loss = torch.sum(self.weights * losses_tensor)
        return weighted_loss, self.weights.detach()

# class MultiTaskAlzheimerModel(pl.LightningModule):
#     def __init__(self, 
#                  num_classes=2, 
#                  input_shape=(1, 64, 64, 64),
#                  metadata_dim=3):
#         super(MultiTaskAlzheimerModel, self).__init__()
        
#         self.resnet3d = r3d_18(pretrained=True)
        
#         # Modify first conv layer
#         original_conv = self.resnet3d.stem[0]
#         self.resnet3d.stem[0] = nn.Conv3d(
#             input_shape[0], 
#             original_conv.out_channels,
#             kernel_size=original_conv.kernel_size,
#             stride=original_conv.stride,
#             padding=original_conv.padding,
#             bias=False
#         )
        
#         self.feature_dim = self.resnet3d.fc.in_features
#         self.resnet3d = nn.Sequential(*list(self.resnet3d.children())[:-1])
        
#         # Enhanced metadata embedding
#         self.metadata_embedding = nn.Sequential(
#             nn.Linear(metadata_dim, 256),
#             nn.LayerNorm(256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.3),
#             nn.Linear(256, 128),
#             nn.LayerNorm(128),
#             nn.ReLU(inplace=True)
#         )
        
#         # Enhanced feature fusion with residual connection
#         # self.cross_attention = nn.Sequential(
#         #     nn.Linear(self.feature_dim + 128, 512),
#         #     nn.LayerNorm(512),
#         #     nn.ReLU(inplace=True),
#         #     nn.Dropout(0.3),
#         #     nn.Linear(512, 512),
#         #     nn.LayerNorm(512),
#         #     nn.ReLU(inplace=True)
#         # )
#         self.cross_attention = nn.Sequential(
#             nn.Linear(self.feature_dim + 128, 1024),  # Thay đổi từ 512 thành 1024
#             nn.LayerNorm(1024),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.3),
#             nn.Linear(1024, 1024),  # Thay đổi từ 512 thành 1024
#             nn.LayerNorm(1024),
#             nn.ReLU(inplace=True)
#         )
        
#         # Gradient scaling for better balance
#         self.grad_scale_classification = GradientScaleLayer(0.5)
#         self.grad_scale_regression = GradientScaleLayer(2.0)
        
#         # Enhanced shared representation
#         self.shared_representation = nn.Sequential(
#             nn.Linear(512, 1024),
#             nn.LayerNorm(1024),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.4),
#             nn.Linear(1024, 1024),
#             nn.LayerNorm(1024),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.4)
#         )
        
#         # Enhanced classification branch
#         self.classification_gate = AttentionGatingModule(1024)
#         self.classification_branch = nn.Sequential(
#             nn.Linear(1024, 512),
#             nn.LayerNorm(512),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(512, 256),
#             nn.LayerNorm(256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(256, num_classes)
#         )
        
#         # Enhanced regression branch
#         self.regression_gate = AttentionGatingModule(1024)
#         self.regression_branch = nn.Sequential(
#             nn.Linear(1024, 512),
#             nn.LayerNorm(512),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(512, 256),
#             nn.LayerNorm(256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(256, 1)
#         )
        
#         # Metrics and losses
#         self.train_classification_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
#         self.val_classification_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
#         self.test_classification_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
#         self.mmse_mae = MeanAbsoluteError()
        
#         self.classification_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
#         self.regression_loss = nn.SmoothL1Loss(beta=0.1)  # Changed to Smooth L1 Loss
        
#         self.frank_wolfe_loss = AdaptiveFrankWolfeLoss(num_tasks=2)
#         self.current_iteration = 0
#         self.total_iterations = 0
        
#         # Add regression scaling
#         self.register_buffer('mmse_min', torch.tensor(1.0))
#         self.register_buffer('mmse_max', torch.tensor(30.0))
class MultiTaskAlzheimerModel(pl.LightningModule):
    def __init__(self, 
                 num_classes=2, 
                 input_shape=(1, 64, 64, 64),
                 metadata_dim=3):
        super(MultiTaskAlzheimerModel, self).__init__()
        self.num_classes=num_classes
        # ResNet3D backbone
        self.resnet3d = r3d_18(pretrained=True)
        original_conv = self.resnet3d.stem[0]
        self.resnet3d.stem[0] = nn.Conv3d(
            input_shape[0], 
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )
        
        self.feature_dim = self.resnet3d.fc.in_features  # Thường là 512
        self.resnet3d = nn.Sequential(*list(self.resnet3d.children())[:-1])
        
        # Metadata embedding với kích thước output nhỏ hơn
        self.metadata_embedding = nn.Sequential(
            nn.Linear(metadata_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True)
        )
        
        # Feature fusion đảm bảo output 512 chiều
        self.cross_attention = nn.Sequential(
            nn.Linear(self.feature_dim + 64, 512),  # 512 + 64 -> 512
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True)
        )
        
        # Gradient scaling
        self.grad_scale_classification = GradientScaleLayer(0.5)
        self.grad_scale_regression = GradientScaleLayer(2.0)
        
        # Shared representation giữ nguyên kích thước 512
        self.shared_representation = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4)
        )
        
        # Classification branch
        self.classification_gate = AttentionGatingModule(512)
        self.classification_branch = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
        # Regression branch
        self.regression_gate = AttentionGatingModule(512)
        self.regression_branch = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
        
        # Metrics và losses giữ nguyên
        self.train_classification_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_classification_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_classification_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.mmse_mae = MeanAbsoluteError()
        
        self.classification_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.regression_loss = nn.SmoothL1Loss(beta=0.1)
        
        self.frank_wolfe_loss = AdaptiveFrankWolfeLoss(num_tasks=2)
        self.current_iteration = 0
        self.total_iterations = 0
        
        # Thêm scaling cho regression
        self.register_buffer('mmse_min', torch.tensor(1.0))
        self.register_buffer('mmse_max', torch.tensor(30.0))

    def forward(self, image, metadata):
        # Extract features
        image_features = self.resnet3d(image)
        image_features = image_features.squeeze(-1).squeeze(-1).squeeze(-1)
        
        # Process metadata với kích thước nhỏ hơn (64)
        metadata_features = self.metadata_embedding(metadata)
        
        # Feature fusion xuống 512 chiều
        fused_features = self.cross_attention(torch.cat([image_features, metadata_features], dim=1))
        
        # Shared representation với residual connection (cả hai đều 512 chiều)
        shared_features = self.shared_representation(fused_features) + fused_features
        
        # Classification branch
        classification_features = self.classification_gate(shared_features, shared_features)
        classification_features = self.grad_scale_classification(classification_features)
        classification_output = self.classification_branch(classification_features)
        
        # Regression branch
        regression_features = self.regression_gate(shared_features, shared_features)
        regression_features = self.grad_scale_regression(regression_features)
        regression_output = self.regression_branch(regression_features)
        
        return classification_output, regression_output
    
    def normalize_mmse(self, mmse):
        return (mmse - self.mmse_min) / (self.mmse_max - self.mmse_min)
    
    def denormalize_mmse(self, normalized_mmse):
        return normalized_mmse * (self.mmse_max - self.mmse_min) + self.mmse_min
    
    # def forward(self, image, metadata):
    #     # Extract features
    #     image_features = self.resnet3d(image)
    #     image_features = image_features.squeeze(-1).squeeze(-1).squeeze(-1)
        
    #     # Process metadata with enhanced embedding
    #     metadata_features = self.metadata_embedding(metadata)
        
    #     # Enhanced feature fusion
    #     fused_features = self.cross_attention(torch.cat([image_features, metadata_features], dim=1))
        
    #     # Shared representation with residual connection
    #     shared_features = self.shared_representation(fused_features) + fused_features
        
    #     # Classification branch with gradient scaling
    #     classification_features = self.classification_gate(shared_features, shared_features)
    #     classification_features = self.grad_scale_classification(classification_features)
    #     classification_output = self.classification_branch(classification_features)
        
    #     # Regression branch with gradient scaling
    #     regression_features = self.regression_gate(shared_features, shared_features)
    #     regression_features = self.grad_scale_regression(regression_features)
    #     regression_output = self.regression_branch(regression_features)
        
    #     return classification_output, regression_output
    
    def training_step(self, batch, batch_idx):
        image, label, mmse, age, gender = batch['image'], batch['label'], batch['mmse'], batch['age'], batch['gender']
        metadata = torch.stack([self.normalize_mmse(mmse), age, gender], dim=1).float()
        
        classification_output, regression_output = self(image, metadata)
        
        # Normalize MMSE for loss calculation
        normalized_mmse = self.normalize_mmse(mmse)
        
        classification_loss = self.classification_loss(classification_output, label)
        regression_loss = self.regression_loss(regression_output.squeeze(), normalized_mmse)
        
        self.current_iteration += 1
        
        losses = [classification_loss, regression_loss]
        total_loss, weights = self.frank_wolfe_loss(
            losses, 
            self.current_iteration, 
            self.total_iterations
        )
        
        # Log metrics
        preds = torch.argmax(classification_output, dim=1)
        acc = (preds == label).float().mean()
        
        # Denormalize for MAE calculation
        denorm_regression = self.denormalize_mmse(regression_output.squeeze())
        mae = F.l1_loss(denorm_regression, mmse)
        
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_classification_loss', classification_loss, on_step=True, on_epoch=True)
        self.log('train_regression_loss', regression_loss, on_step=True, on_epoch=True)
        self.log('train_classification_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_regression_mae', mae, on_step=True, on_epoch=True, prog_bar=True)
        
        return total_loss

    def configure_optimizers(self):
        # Group parameters for different learning rates
        pretrained_params = list(self.resnet3d.parameters())
        regression_params = list(self.regression_gate.parameters()) + \
                          list(self.regression_branch.parameters())
        other_params = list(self.metadata_embedding.parameters()) + \
                      list(self.cross_attention.parameters()) + \
                      list(self.shared_representation.parameters()) + \
                      list(self.classification_gate.parameters()) + \
                      list(self.classification_branch.parameters())
        
        # Use different learning rates
        optimizer = torch.optim.AdamW([
            {'params': pretrained_params, 'lr': 1e-4, 'weight_decay': 0.01},
            {'params': regression_params, 'lr': 2e-3, 'weight_decay': 0.01},
            {'params': other_params, 'lr': 1e-3, 'weight_decay': 0.01}
        ])
        
        # Sử dụng ReduceLROnPlateau thay vì OneCycleLR
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=5,
            verbose=True,
            min_lr=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',  # Metric để monitor
                'interval': 'epoch',    # Cập nhật scheduler mỗi epoch
                'frequency': 1,         # Số epochs giữa mỗi lần cập nhật
            }
        }
    def validation_step(self, batch, batch_idx):
        image, label, mmse, age, gender = batch['image'], batch['label'], batch['mmse'], batch['age'], batch['gender']
        metadata = torch.stack([self.normalize_mmse(mmse), age, gender], dim=1).float()
        
        classification_output, regression_output = self(image, metadata)
        
        # Normalize MMSE for loss calculation
        normalized_mmse = self.normalize_mmse(mmse)
        
        classification_loss = self.classification_loss(classification_output, label)
        regression_loss = self.regression_loss(regression_output.squeeze(), normalized_mmse)
        
        losses = [classification_loss, regression_loss]
        total_loss, weights = self.frank_wolfe_loss(
            losses, 
            self.current_iteration, 
            self.total_iterations
        )
        
        # Calculate metrics
        preds = torch.argmax(classification_output, dim=1)
        acc = (preds == label).float().mean()
        
        # Denormalize for MAE calculation
        denorm_regression = self.denormalize_mmse(regression_output.squeeze())
        mae = F.l1_loss(denorm_regression, mmse)
        
        # Log all relevant metrics
        self.log('val_loss', total_loss, on_epoch=True, prog_bar=True)
        self.log('val_classification_loss', classification_loss, on_epoch=True)
        self.log('val_regression_loss', regression_loss, on_epoch=True)
        self.log('val_classification_acc', acc, on_epoch=True, prog_bar=True)
        self.log('val_regression_mae', mae, on_epoch=True, prog_bar=True)
        
        # Store predictions for potential analysis
        self.validation_step_outputs = {
            'preds': preds,
            'true_labels': label,
            'regression_preds': denorm_regression,
            'true_mmse': mmse
        }
        
        return total_loss


    def test_step(self, batch, batch_idx):
        image, label, mmse, age, gender = batch['image'], batch['label'], batch['mmse'], batch['age'], batch['gender']
        metadata = torch.stack([self.normalize_mmse(mmse), age, gender], dim=1).float()
        
        # Get model predictions
        classification_output, regression_output = self(image, metadata)
        
        # Calculate classification metrics
        preds = torch.argmax(classification_output, dim=1)
        acc = (preds == label).float().mean()
        
        # Calculate regression metrics
        denorm_regression = self.denormalize_mmse(regression_output.squeeze())
        mae = F.l1_loss(denorm_regression, mmse)
        mse = F.mse_loss(denorm_regression, mmse)
        rmse = torch.sqrt(mse)
        
        # Calculate per-class metrics
        for class_idx in range(self.num_classes):
            class_mask = (label == class_idx)
            if torch.any(class_mask):
                class_acc = (preds[class_mask] == label[class_mask]).float().mean()
                self.log(f'test_class_{class_idx}_acc', class_acc, on_epoch=True)
        
        # Log metrics
        self.log('test_classification_acc', acc, on_epoch=True)
        self.log('test_regression_mae', mae, on_epoch=True)
        self.log('test_regression_mse', mse, on_epoch=True)
        self.log('test_regression_rmse', rmse, on_epoch=True)
        
        # Calculate and log correlation coefficient between predicted and true MMSE
        if batch_idx == 0:  # Only for first batch to avoid redundant computation
            correlation_matrix = torch.corrcoef(torch.stack([denorm_regression, mmse]))
            correlation = correlation_matrix[0, 1]  # Get correlation coefficient
            self.log('test_regression_correlation', correlation, on_epoch=True)
        
        # Store predictions for later analysis
        return {
            'batch_idx': batch_idx,
            'preds': preds,
            'true_labels': label,
            'regression_preds': denorm_regression,
            'true_mmse': mmse,
            'classification_probs': F.softmax(classification_output, dim=1),
            'acc': acc,
            'mae': mae
        }
    
    # def test_epoch_end(self, outputs):
    #     # Aggregate predictions from all batches
    #     all_preds = torch.cat([x['preds'] for x in outputs])
    #     all_labels = torch.cat([x['true_labels'] for x in outputs])
    #     all_reg_preds = torch.cat([x['regression_preds'] for x in outputs])
    #     all_true_mmse = torch.cat([x['true_mmse'] for x in outputs])
        
    #     # Calculate confusion matrix
    #     num_classes = self.num_classes
    #     confusion_matrix = torch.zeros(num_classes, num_classes)
    #     for t, p in zip(all_labels, all_preds):
    #         confusion_matrix[t.long(), p.long()] += 1
        
    #     # Calculate additional metrics
    #     precision = confusion_matrix.diag() / confusion_matrix.sum(1)
    #     recall = confusion_matrix.diag() / confusion_matrix.sum(0)
    #     f1 = 2 * precision * recall / (precision + recall)
        
    #     # Log additional metrics
    #     self.log('test_macro_precision', precision.mean(), on_epoch=True)
    #     self.log('test_macro_recall', recall.mean(), on_epoch=True)
    #     self.log('test_macro_f1', f1.mean(), on_epoch=True)
        
    #     # Calculate regression error distribution
    #     errors = torch.abs(all_reg_preds - all_true_mmse)
    #     error_std = torch.std(errors)
    #     error_median = torch.median(errors)
        
    #     self.log('test_regression_error_std', error_std, on_epoch=True)
    #     self.log('test_regression_error_median', error_median, on_epoch=True)
        