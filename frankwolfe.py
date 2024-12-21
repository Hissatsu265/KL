import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, MeanAbsoluteError

class FrankWolfeOptimizer:
    def __init__(self, num_tasks=2, max_iter=5):
        self.num_tasks = num_tasks
        self.max_iter = max_iter
        # Khởi tạo trọng số ban đầu bằng nhau
        self.weights = None
        self.device = None
        
    def to(self, device):
        """Chuyển weights sang device được chỉ định"""
        self.device = device
        if self.weights is None:
            self.weights = (torch.ones(self.num_tasks) / self.num_tasks).to(device)
        else:
            self.weights = self.weights.to(device)
        return self
        
    def compute_gradient(self, losses):
        """Tính gradient cho mỗi loss"""
        return torch.stack([loss.detach() for loss in losses])
    
    def solve_linear_problem(self, gradients):
        """Giải quyết bài toán tối ưu tuyến tính"""
        min_idx = torch.argmin(gradients)
        s = torch.zeros_like(self.weights, device=self.device)
        s[min_idx] = 1.0
        return s
    
    def update_weights(self, losses):
        """Cập nhật trọng số using Frank-Wolfe algorithm"""
        if self.weights is None:
            self.device = losses[0].device
            self.weights = (torch.ones(self.num_tasks) / self.num_tasks).to(self.device)
            
        gradients = self.compute_gradient(losses)
        s = self.solve_linear_problem(gradients)
        
        # Cập nhật trọng số
        gamma = 2.0 / (self.max_iter + 2)
        self.weights = (1 - gamma) * self.weights + gamma * s
        
        # Đảm bảo tổng trọng số = 1 và không âm
        self.weights = F.softmax(self.weights, dim=0)
        
        return self.weights
class Efficient3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=6):
        super(Efficient3DBlock, self).__init__()
        
        # Convolution phân tách theo chiều sâu
        self.depthwise_conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.BatchNorm3d(in_channels),
            nn.ReLU6(inplace=True)
        )
        
        # Convolution theo điểm với phép mở rộng
        expanded_channels = in_channels * expand_ratio
        self.pointwise_conv = nn.Sequential(
            nn.Conv3d(in_channels, expanded_channels, kernel_size=1),
            nn.BatchNorm3d(expanded_channels),
            nn.ReLU6(inplace=True)
        )
        
        # Convolution chiếu
        self.projection_conv = nn.Sequential(
            nn.Conv3d(expanded_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels)
        )
        
        # Kết nối rút ngắn
        self.shortcut = in_channels == out_channels
        
    def forward(self, x):
        identity = x
        
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.projection_conv(x)
        
        if self.shortcut:
            x += identity
        
        return x

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
        if shared_features.size(1) != task_specific_features.size(1):
            task_specific_features = F.linear(task_specific_features, 
                                             torch.eye(shared_features.size(1)).to(task_specific_features.device))

        attention_weights = self.attention(shared_features)
        gated_features = shared_features * attention_weights + \
                         task_specific_features * (1 - attention_weights)
        return gated_features

class MultiTaskAlzheimerModel(pl.LightningModule):
    def __init__(self, 
                 num_classes=2, 
                 input_shape=(1, 64, 64, 64),
                 metadata_dim=3):
        super(MultiTaskAlzheimerModel, self).__init__()
        
        # Lớp đầu vào
        self.image_conv = nn.Sequential(
            nn.Conv3d(input_shape[0], 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        
        self.metadata_embedding = nn.Sequential(
            nn.Linear(metadata_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True)
        )
        
        # Trích xuất đặc trưng
        self.feature_extractor = nn.Sequential(
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
        
        # Trộn đặc trưng
        self.cross_attention = nn.Sequential(
            nn.Linear(512 + 128, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True)
        )
        
        # Biểu diễn chung
        self.shared_representation = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # Nhánh phân loại
        self.classification_gate = AttentionGatingModule(1024)
        self.classification_branch = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Nhánh hồi quy
        self.regression_gate = AttentionGatingModule(1024)
        self.regression_branch = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
        
        # Metrics
        self.train_classification_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_classification_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_classification_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.mmse_mae = MeanAbsoluteError()
        
        # Losses
        self.classification_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.regression_loss = nn.HuberLoss()
        self.multi_task_optimizer = FrankWolfeOptimizer(num_tasks=2)
        
        # Lưu trữ loss history để phân tích
        self.loss_history = {
            'classification': [],
            'regression': [],
            'weights': []
        }
    def on_fit_start(self):
        """Được gọi khi bắt đầu training"""
        # Chuyển optimizer sang đúng device
        self.multi_task_optimizer.to(self.device)
    def forward(self, image, metadata):
        # Xử lý đầu vào
        image_features = self.image_conv(image)
        metadata_features = self.metadata_embedding(metadata)
        
        # Trích xuất đặc trưng
        image_features = self.feature_extractor(image_features)
        image_features = self.global_pool(image_features).flatten(1)
        
        # Trộn đặc trưng
        fused_features = self.cross_attention(torch.cat([image_features, metadata_features], dim=1))
        
        # Biểu diễn chung
        shared_features = self.shared_representation(fused_features)
        
        # Nhánh phân loại
        classification_features = self.classification_gate(shared_features, shared_features)
        classification_output = self.classification_branch(classification_features)
        
        # Nhánh hồi quy
        regression_features = self.regression_gate(shared_features, shared_features)
        regression_output = self.regression_branch(regression_features)
        
        return classification_output, regression_output

    def training_step(self, batch, batch_idx):
        image, label, mmse, age, gender = batch['image'], batch['label'], batch['mmse'], batch['age'], batch['gender']
        metadata = torch.stack([mmse, age, gender], dim=1).float()
        
        classification_output, regression_output = self(image, metadata)
        
        # Tính individual losses
        classification_loss = self.classification_loss(classification_output, label)
        regression_loss = self.regression_loss(regression_output.squeeze(), mmse)
        
        # Cập nhật trọng số using Frank-Wolfe
        # losses = [classification_loss, regression_loss]
        losses = [classification_loss.to(self.device), regression_loss.to(self.device)]
        weights = self.multi_task_optimizer.update_weights(losses)
        
        # Tính tổng loss với trọng số mới
        total_loss = torch.sum(torch.stack(losses) * weights)
        
        # Logging
        preds = torch.argmax(classification_output, dim=1)
        acc = (preds == label).float().mean()
        
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_classification_loss', classification_loss, on_step=True, on_epoch=True)
        self.log('train_regression_loss', regression_loss, on_step=True, on_epoch=True)
        self.log('train_classification_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_classification_weight', weights[0], on_step=True, on_epoch=True)
        self.log('train_regression_weight', weights[1], on_step=True, on_epoch=True)
        
        # Lưu history
        self.loss_history['classification'].append(classification_loss.item())
        self.loss_history['regression'].append(regression_loss.item())
        self.loss_history['weights'].append(weights.tolist())
        
        return total_loss

    # def validation_step(self, batch, batch_idx):
    #     image, label, mmse, age, gender = batch['image'], batch['label'], batch['mmse'], batch['age'], batch['gender']
    #     metadata = torch.stack([mmse, age, gender], dim=1).float()
        
    #     classification_output, regression_output = self(image, metadata)
        
    #     # Sử dụng trọng số đã học được từ training
    #     classification_loss = self.classification_loss(classification_output, label)
    #     regression_loss = self.regression_loss(regression_output.squeeze(), mmse)
        
    #     weights = self.multi_task_optimizer.weights
    #     total_loss = torch.sum(torch.stack([classification_loss, regression_loss]) * weights)
        
    #     preds = torch.argmax(classification_output, dim=1)
    #     acc = (preds == label).float().mean()
        
    #     self.log('val_loss', total_loss, on_epoch=True, prog_bar=True)
    #     self.log('val_classification_loss', classification_loss, on_epoch=True)
    #     self.log('val_regression_loss', regression_loss, on_epoch=True)
    #     self.log('val_classification_acc', acc, on_epoch=True, prog_bar=True)
        
    #     return total_loss
    
    # def test_step(self, batch, batch_idx):
    #     image, label, mmse, age, gender = batch['image'], batch['label'], batch['mmse'], batch['age'], batch['gender']
    #     metadata = torch.stack([mmse, age, gender], dim=1).float()
        
    #     classification_output, regression_output = self(image, metadata)
        
    #     preds = torch.argmax(classification_output, dim=1)
    #     acc = (preds == label).float().mean()
    #     mae = F.l1_loss(regression_output.squeeze(), mmse)
        
    #     self.log('test_classification_acc', acc, on_epoch=True, prog_bar=True)
    #     self.log('test_regression_mae', mae, on_epoch=True, prog_bar=True)
        
    #     return {'preds': preds, 'true_labels': label}
    def validation_step(self, batch, batch_idx):
        image, label, mmse, age, gender = batch['image'], batch['label'], batch['mmse'], batch['age'], batch['gender']
        metadata = torch.stack([mmse, age, gender], dim=1).float()
        
        classification_output, regression_output = self(image, metadata)
        
        classification_loss = self.classification_loss(classification_output, label)
        regression_loss = self.regression_loss(regression_output.squeeze(), mmse)
        
        # Đảm bảo losses ở cùng device với weights
        losses = [classification_loss.to(self.device), regression_loss.to(self.device)]
        weights = self.multi_task_optimizer.weights  # weights đã ở đúng device
        
        total_loss = torch.sum(torch.stack(losses) * weights)
        
        preds = torch.argmax(classification_output, dim=1)
        acc = (preds == label).float().mean()
        
        self.log('val_loss', total_loss, on_epoch=True, prog_bar=True)
        self.log('val_classification_loss', classification_loss, on_epoch=True)
        self.log('val_regression_loss', regression_loss, on_epoch=True)
        self.log('val_classification_acc', acc, on_epoch=True, prog_bar=True)
        self.log('val_classification_weight', weights[0], on_epoch=True)
        self.log('val_regression_weight', weights[1], on_epoch=True)
        
        return total_loss

    def test_step(self, batch, batch_idx):
        image, label, mmse, age, gender = batch['image'], batch['label'], batch['mmse'], batch['age'], batch['gender']
        metadata = torch.stack([mmse, age, gender], dim=1).float()
        
        classification_output, regression_output = self(image, metadata)
        
        # Trong phase test, chúng ta sử dụng trọng số cuối cùng đã học được
        classification_loss = self.classification_loss(classification_output, label).to(self.device)
        regression_loss = self.regression_loss(regression_output.squeeze(), mmse).to(self.device)
        
        weights = self.multi_task_optimizer.weights
        total_loss = torch.sum(torch.stack([classification_loss, regression_loss]) * weights)
        
        
        preds = torch.argmax(classification_output, dim=1)
        acc = (preds == label).float().mean()
        mae = F.l1_loss(regression_output.squeeze(), mmse)
        
        # Log thêm các metric chi tiết
        self.log('test_total_loss', total_loss, on_epoch=True)
        self.log('test_classification_loss', classification_loss, on_epoch=True)
        self.log('test_regression_loss', regression_loss, on_epoch=True)
        self.log('test_classification_acc', acc, on_epoch=True, prog_bar=True)
        self.log('test_regression_mae', mae, on_epoch=True, prog_bar=True)
        self.log('final_classification_weight', weights[0], on_epoch=True)
        self.log('final_regression_weight', weights[1], on_epoch=True)
        
        # Lưu chi tiết để phân tích sau
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

