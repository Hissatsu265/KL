import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, MeanAbsoluteError
from torchvision.models.video import r3d_18
from torch.hub import load_state_dict_from_url

class FrankWolfeLoss(nn.Module):
    def __init__(self, num_tasks=2, device='cuda'):
        super(FrankWolfeLoss, self).__init__()
        self.weights = nn.Parameter(torch.ones(num_tasks).to(device) / num_tasks)
        self.num_tasks = num_tasks
        
    def compute_frank_wolfe_step(self, grads):
        grad_dot = torch.matmul(self.weights, grads)
        min_idx = torch.argmin(grads)
        s = torch.zeros_like(self.weights)
        s[min_idx] = 1.0
        return s
    
    def forward(self, losses, iteration, total_iterations):
        losses_tensor = torch.stack(losses)
        grads = losses_tensor.detach()
        s = self.compute_frank_wolfe_step(grads)
        gamma = 2.0 / (iteration + 2.0)
        
        with torch.no_grad():
            new_weights = (1 - gamma) * self.weights + gamma * s
            self.weights.copy_(new_weights)
        
        weighted_loss = torch.sum(self.weights * losses_tensor)
        return weighted_loss, self.weights.detach()

class AttentionGatingModule(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionGatingModule, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.BatchNorm1d(feature_dim // 2),  # Added BatchNorm
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
        
        # Load pretrained ResNet3D
        self.resnet3d = r3d_18(pretrained=True)
        
        # Modify first conv layer to accept single channel input
        original_conv = self.resnet3d.stem[0]
        self.resnet3d.stem[0] = nn.Conv3d(
            input_shape[0], 
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )
        
        # Remove the final classification layer
        self.feature_dim = self.resnet3d.fc.in_features
        self.resnet3d = nn.Sequential(*list(self.resnet3d.children())[:-1])
        
        # Enhanced metadata embedding
        self.metadata_embedding = nn.Sequential(
            nn.Linear(metadata_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        
        # Enhanced feature fusion
        self.cross_attention = nn.Sequential(
            nn.Linear(self.feature_dim + 128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Enhanced shared representation
        self.shared_representation = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Classification branch
        self.classification_gate = AttentionGatingModule(1024)
        self.classification_branch = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Enhanced regression branch
        self.regression_gate = AttentionGatingModule(1024)
        self.regression_branch = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Constrain output between 0 and 1
        )
        
        # MMSE normalization parameters
        self.register_buffer('mmse_min', torch.tensor(1.0))
        self.register_buffer('mmse_max', torch.tensor(30.0))
        
        # Metrics
        self.train_classification_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_classification_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_classification_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.mmse_mae = MeanAbsoluteError()
        
        # Enhanced losses
        self.classification_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.regression_loss = nn.SmoothL1Loss()
        self.regression_scale = 5.0  # Scaling factor for regression loss
        
        # Frank-Wolfe loss
        self.frank_wolfe_loss = FrankWolfeLoss(num_tasks=2)
        self.current_iteration = 0
        self.total_iterations = 0
        
    def normalize_mmse(self, mmse):
        return (mmse - self.mmse_min) / (self.mmse_max - self.mmse_min)
    
    def denormalize_mmse(self, normalized_mmse):
        return normalized_mmse * (self.mmse_max - self.mmse_min) + self.mmse_min

    def forward(self, image, metadata):
        # Normalize MMSE in metadata
        metadata = metadata.clone()
        metadata[:, 0] = self.normalize_mmse(metadata[:, 0])
        
        # Extract features using pretrained ResNet3D
        image_features = self.resnet3d(image)
        image_features = image_features.squeeze(-1).squeeze(-1).squeeze(-1)
        
        # Process metadata
        metadata_features = self.metadata_embedding(metadata)
        
        # Fuse features
        fused_features = self.cross_attention(torch.cat([image_features, metadata_features], dim=1))
        
        # Shared representation
        shared_features = self.shared_representation(fused_features)
        
        # Classification branch
        classification_features = self.classification_gate(shared_features, shared_features)
        classification_output = self.classification_branch(classification_features)
        
        # Regression branch
        regression_features = self.regression_gate(shared_features, shared_features)
        regression_output = self.regression_branch(regression_features)
        
        # Denormalize regression output
        regression_output = self.denormalize_mmse(regression_output)
        
        return classification_output, regression_output

    def training_step(self, batch, batch_idx):
        image, label, mmse, age, gender = batch['image'], batch['label'], batch['mmse'], batch['age'], batch['gender']
        metadata = torch.stack([mmse, age, gender], dim=1).float()
        
        classification_output, regression_output = self(image, metadata)
        
        classification_loss = self.classification_loss(classification_output, label)
        regression_loss = self.regression_loss(regression_output.squeeze(), mmse) * self.regression_scale
        
        self.current_iteration += 1
        
        losses = [classification_loss, regression_loss]
        total_loss, weights = self.frank_wolfe_loss(
            losses, 
            self.current_iteration, 
            self.total_iterations
        )
        
        # Calculate metrics
        preds = torch.argmax(classification_output, dim=1)
        acc = self.train_classification_accuracy(preds, label)
        mae = self.mmse_mae(regression_output.squeeze(), mmse)
        
        # Log metrics
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_classification_loss', classification_loss, on_step=True, on_epoch=True)
        self.log('train_regression_loss', regression_loss, on_step=True, on_epoch=True)
        self.log('train_classification_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_mmse_mae', mae, on_step=True, on_epoch=True, prog_bar=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        image, label, mmse, age, gender = batch['image'], batch['label'], batch['mmse'], batch['age'], batch['gender']
        metadata = torch.stack([mmse, age, gender], dim=1).float()
        
        classification_output, regression_output = self(image, metadata)
        
        classification_loss = self.classification_loss(classification_output, label)
        regression_loss = self.regression_loss(regression_output.squeeze(), mmse) * self.regression_scale
        
        losses = [classification_loss, regression_loss]
        total_loss, weights = self.frank_wolfe_loss(
            losses, 
            self.current_iteration, 
            self.total_iterations
        )
        
        # Calculate metrics
        preds = torch.argmax(classification_output, dim=1)
        acc = self.val_classification_accuracy(preds, label)
        mae = self.mmse_mae(regression_output.squeeze(), mmse)
        
        # Log metrics
        self.log('val_loss', total_loss, on_epoch=True, prog_bar=True)
        self.log('val_classification_loss', classification_loss, on_epoch=True)
        self.log('val_regression_loss', regression_loss, on_epoch=True)
        self.log('val_classification_acc', acc, on_epoch=True, prog_bar=True)
        self.log('val_mmse_mae', mae, on_epoch=True, prog_bar=True)
        
        return total_loss
    
    def test_step(self, batch, batch_idx):
        image, label, mmse, age, gender = batch['image'], batch['label'], batch['mmse'], batch['age'], batch['gender']
        metadata = torch.stack([mmse, age, gender], dim=1).float()
        
        classification_output, regression_output = self(image, metadata)
        
        # Calculate metrics
        preds = torch.argmax(classification_output, dim=1)
        acc = self.test_classification_accuracy(preds, label)
        mae = self.mmse_mae(regression_output.squeeze(), mmse)
        
        # Log metrics
        self.log('test_classification_acc', acc, on_epoch=True, prog_bar=True)
        self.log('test_mmse_mae', mae, on_epoch=True, prog_bar=True)
        
        return {'preds': preds, 'true_labels': label}
    
    def configure_optimizers(self):
        # Use different learning rates for pretrained and new layers
        pretrained_params = list(self.resnet3d.parameters())
        new_params = list(self.metadata_embedding.parameters()) + \
                    list(self.cross_attention.parameters()) + \
                    list(self.shared_representation.parameters()) + \
                    list(self.classification_gate.parameters()) + \
                    list(self.classification_branch.parameters()) + \
                    list(self.regression_gate.parameters()) + \
                    list(self.regression_branch.parameters())
        
        optimizer = torch.optim.Adam([
            {'params': pretrained_params, 'lr': 1e-4},
            {'params': new_params, 'lr': 1e-3}
        ])
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.1, 
            patience=5,
            min_lr=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }