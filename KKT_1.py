import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, MeanAbsoluteError
from torchvision.models.video import r3d_18
from torch.hub import load_state_dict_from_url

class KKTLoss(nn.Module):
    def __init__(self, num_tasks=3, device='cuda', alpha=0.1):
        super(KKTLoss, self).__init__()
        # Initialize task weights with simplex constraint
        self.weights = nn.Parameter(torch.ones(num_tasks).to(device) / num_tasks)
        self.num_tasks = num_tasks
        self.alpha = alpha  # Learning rate for dual variables
        self.eps = 1e-6  # Small constant for numerical stability
        
    def project_simplex(self, v):
        """Project vector v onto probability simplex."""
        v_sorted, _ = torch.sort(v, descending=True)
        cssv = torch.cumsum(v_sorted, dim=0)
        rho = torch.nonzero(v_sorted * torch.arange(1, len(v) + 1, device=v.device) > cssv - 1)[-1]
        theta = (cssv[rho] - 1) / (rho + 1)
        return torch.maximum(v - theta, torch.zeros_like(v))
    
    def compute_kkt_weights(self, losses):
        """Compute optimal task weights using KKT conditions."""
        # Gradient of Lagrangian with respect to weights
        grad_L = losses - torch.mean(losses)
        
        # Update weights using gradient descent with simplex projection
        with torch.no_grad():
            new_weights = self.weights - self.alpha * grad_L
            # Project onto probability simplex (sum to 1, non-negative)
            new_weights = self.project_simplex(new_weights)
            self.weights.copy_(new_weights)
            
        return self.weights
    
    def forward(self, losses, iteration, total_iterations):
        losses_tensor = torch.stack(losses)
        
        # Compute optimal weights using KKT conditions
        weights = self.compute_kkt_weights(losses_tensor)
        
        # Weighted sum of losses
        weighted_loss = torch.sum(weights * losses_tensor)
        
        return weighted_loss, weights.detach()

class AttentionGatingModule(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionGatingModule, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
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
        
        # Enhanced metadata processing
        self.metadata_embedding = nn.Sequential(
            nn.Linear(metadata_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        # Improved feature fusion
        self.cross_attention = nn.Sequential(
            nn.Linear(self.feature_dim + 256, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        # Enhanced shared representation
        self.shared_representation = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        # Classification branch
        self.classification_gate = AttentionGatingModule(1024)
        self.classification_branch = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        # Enhanced regression branch
        self.regression_gate = AttentionGatingModule(1024)
        self.regression_branch = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )
        
        # Auxiliary regression task
        self.aux_regression = nn.Sequential(
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
        
        # MMSE normalization parameters
        self.register_buffer('mmse_min', torch.tensor(0.0))
        self.register_buffer('mmse_max', torch.tensor(30.0))
        
        # Metrics
        self.train_classification_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_classification_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_classification_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.mmse_mae = MeanAbsoluteError()
        
        # Modified losses
        self.classification_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.regression_loss = nn.HuberLoss(delta=1.0)
        self.regression_scale = 10.0
        
        # KKT loss with task prioritization
        self.kkt_loss = KKTLoss(num_tasks=3)
        self.current_iteration = 0
        self.total_iterations = 0
    
    def normalize_mmse(self, mmse):
        return (mmse - self.mmse_min) / (self.mmse_max - self.mmse_min)
    
    def denormalize_mmse(self, normalized_mmse):
        return normalized_mmse * (self.mmse_max - self.mmse_min) + self.mmse_min

    def forward(self, image, metadata):
        # Extract features using pretrained ResNet3D
        image_features = self.resnet3d(image)
        image_features = image_features.squeeze(-1).squeeze(-1).squeeze(-1)
        
        # Process metadata with enhanced embedding
        metadata_features = self.metadata_embedding(metadata)
        
        # Improved feature fusion
        fused_features = self.cross_attention(torch.cat([image_features, metadata_features], dim=1))
        
        # Enhanced shared representation with residual connection
        shared_features = self.shared_representation(fused_features) + fused_features
        
        # Task-specific branches
        classification_features = self.classification_gate(shared_features, shared_features)
        classification_output = self.classification_branch(classification_features)
        
        regression_features = self.regression_gate(shared_features, shared_features)
        regression_output = self.regression_branch(regression_features)
        
        # Auxiliary regression output
        aux_regression_output = self.aux_regression(shared_features)
        
        return classification_output, regression_output, aux_regression_output

    def training_step(self, batch, batch_idx):
        image, label, mmse, age, gender = batch['image'], batch['label'], batch['mmse'], batch['age'], batch['gender']
        metadata = torch.stack([mmse, age, gender], dim=1).float()
        
        classification_output, regression_output, aux_regression_output = self(image, metadata)
        
        # Calculate losses with modified weights
        classification_loss = self.classification_loss(classification_output, label)
        regression_loss = self.regression_loss(regression_output.squeeze(), mmse) * self.regression_scale
        aux_regression_loss = self.regression_loss(aux_regression_output.squeeze(), mmse) * (self.regression_scale * 0.5)
        
        self.current_iteration += 1
        
        # Combined loss with KKT optimization
        losses = [classification_loss, regression_loss, aux_regression_loss]
        total_loss, weights = self.kkt_loss(
            losses, 
            self.current_iteration, 
            self.total_iterations
        )
        
        # Calculate metrics
        preds = torch.argmax(classification_output, dim=1)
        acc = self.train_classification_accuracy(preds, label)
        mae = self.mmse_mae(regression_output.squeeze(), mmse)
        aux_mae = self.mmse_mae(aux_regression_output.squeeze(), mmse)
        
        # Log metrics
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_classification_loss', classification_loss, on_step=True, on_epoch=True)
        self.log('train_regression_loss', regression_loss, on_step=True, on_epoch=True)
        self.log('train_aux_regression_loss', aux_regression_loss, on_step=True, on_epoch=True)
        self.log('train_classification_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_mmse_mae', mae, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_aux_mmse_mae', aux_mae, on_step=True, on_epoch=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        image, label, mmse, age, gender = batch['image'], batch['label'], batch['mmse'], batch['age'], batch['gender']
        metadata = torch.stack([mmse, age, gender], dim=1).float()
        
        classification_output, regression_output, aux_regression_output = self(image, metadata)
        
        # Calculate losses
        classification_loss = self.classification_loss(classification_output, label)
        regression_loss = self.regression_loss(regression_output.squeeze(), mmse) * self.regression_scale
        aux_regression_loss = self.regression_loss(aux_regression_output.squeeze(), mmse) * (self.regression_scale * 0.5)
        
        losses = [classification_loss, regression_loss, aux_regression_loss]
        total_loss, weights = self.kkt_loss(
            losses, 
            self.current_iteration, 
            self.total_iterations
        )
        
        # Calculate metrics
        preds = torch.argmax(classification_output, dim=1)
        acc = self.val_classification_accuracy(preds, label)
        mae = self.mmse_mae(regression_output.squeeze(), mmse)
        aux_mae = self.mmse_mae(aux_regression_output.squeeze(), mmse)
        
        # Log metrics
        self.log('val_loss', total_loss, on_epoch=True, prog_bar=True)
        self.log('val_classification_loss', classification_loss, on_epoch=True)
        self.log('val_regression_loss', regression_loss, on_epoch=True)
        self.log('val_aux_regression_loss', aux_regression_loss, on_epoch=True)
        self.log('val_classification_acc', acc, on_epoch=True, prog_bar=True)
        self.log('val_mmse_mae', mae, on_epoch=True, prog_bar=True)
        self.log('val_aux_mmse_mae', aux_mae, on_epoch=True)
        
        return total_loss

    def test_step(self, batch, batch_idx):
        image, label, mmse, age, gender = batch['image'], batch['label'], batch['mmse'], batch['age'], batch['gender']
        metadata = torch.stack([mmse, age, gender], dim=1).float()
        
        classification_output, regression_output, aux_regression_output = self(image, metadata)
        
        # Calculate metrics
        preds = torch.argmax(classification_output, dim=1)
        acc = self.test_classification_accuracy(preds, label)
        mae = self.mmse_mae(regression_output.squeeze(), mmse)
        aux_mae = self.mmse_mae(aux_regression_output.squeeze(), mmse)
        
        # Log metrics
        self.log('test_classification_acc', acc, on_epoch=True, prog_bar=True)
        self.log('test_mmse_mae', mae, on_epoch=True, prog_bar=True)
        self.log('test_aux_mmse_mae', aux_mae, on_epoch=True)
        
        return {'preds': preds, 'true_labels': label}

    def configure_optimizers(self):
        # Separate parameter groups with different learning rates
        pretrained_params = list(self.resnet3d.parameters())
        regression_params = list(self.regression_branch.parameters()) + \
                          list(self.regression_gate.parameters()) + \
                          list(self.aux_regression.parameters())
        other_params = list(self.metadata_embedding.parameters()) + \
                      list(self.cross_attention.parameters()) + \
                      list(self.shared_representation.parameters()) + \
                      list(self.classification_gate.parameters()) + \
                      list(self.classification_branch.parameters())
        
        optimizer = torch.optim.AdamW([
            {'params': pretrained_params, 'lr': 5e-5, 'weight_decay': 0.01},
            {'params': regression_params, 'lr': 2e-3, 'weight_decay': 0.01},
            {'params': other_params, 'lr': 1e-3, 'weight_decay': 0.01}
        ])
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }