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
        # Initialize weights for each task
        self.weights = nn.Parameter(torch.ones(num_tasks).to(device) / num_tasks)
        self.num_tasks = num_tasks
        
    def compute_frank_wolfe_step(self, grads):

        # Calculate the linear minimization problem
        grad_dot = torch.matmul(self.weights, grads)
        
        # Find the minimal gradient
        min_idx = torch.argmin(grads)
        s = torch.zeros_like(self.weights)
        s[min_idx] = 1.0
        
        return s
    
    def forward(self, losses, iteration, total_iterations):
        """
        Update weights using Frank-Wolfe algorithm
        Args:
            losses: List of task losses
            iteration: Current training iteration
            total_iterations: Total number of training iterations
        Returns:
            Total weighted loss
        """
        # Stack losses
        losses_tensor = torch.stack(losses)
        
        # Compute gradients with respect to weights
        grads = losses_tensor.detach()
        
        # Compute Frank-Wolfe step
        s = self.compute_frank_wolfe_step(grads)
        
        # Compute step size (diminishing step size rule)
        gamma = 2.0 / (iteration + 2.0)
        
        # Update weights
        with torch.no_grad():
            new_weights = (1 - gamma) * self.weights + gamma * s
            self.weights.copy_(new_weights)
        
        # Compute weighted loss
        weighted_loss = torch.sum(self.weights * losses_tensor)
        
        return weighted_loss, self.weights.detach()
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
        
        # Load pretrained ResNet3D
        self.resnet3d = r3d_18(pretrained=True)
        # self.resnet3d = r3d_50(pretrained=True)
        
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
        
        # Metadata embedding
        self.metadata_embedding = nn.Sequential(
            nn.Linear(metadata_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True)
        )
        
        # Feature fusion
        self.cross_attention = nn.Sequential(
            nn.Linear(self.feature_dim + 128, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True)
        )
        
        # Shared representation
        self.shared_representation = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # Classification branch
        self.classification_gate = AttentionGatingModule(1024)
        self.classification_branch = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Regression branch
        self.regression_gate = AttentionGatingModule(1024)
        self.regression_branch = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
        
        self.register_buffer('mmse_min', torch.tensor(1.0))
        self.register_buffer('mmse_max', torch.tensor(30.0))
        
        # Metrics
        self.train_classification_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_classification_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_classification_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.mmse_mae = MeanAbsoluteError()
        
        # Losses
        self.classification_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.regression_loss = nn.HuberLoss()

        self.frank_wolfe_loss = FrankWolfeLoss(num_tasks=2)
        self.current_iteration = 0
        self.total_iterations = 0 
    
    def forward(self, image, metadata):
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
        
        return classification_output, regression_output

    def training_step(self, batch, batch_idx):
        image, label, mmse, age, gender = batch['image'], batch['label'], batch['mmse'], batch['age'], batch['gender']
        metadata = torch.stack([mmse, age, gender], dim=1).float()
        
        classification_output, regression_output = self(image, metadata)
        
        classification_loss = self.classification_loss(classification_output, label)
        regression_loss = self.regression_loss(regression_output.squeeze(), mmse)
        
        self.current_iteration += 1
        
        # Compute total loss using Frank-Wolfe
        losses = [classification_loss, regression_loss]
        total_loss, weights = self.frank_wolfe_loss(
            losses, 
            self.current_iteration, 
            self.total_iterations
        )
        
        preds = torch.argmax(classification_output, dim=1)
        acc = (preds == label).float().mean()
        
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_classification_loss', classification_loss, on_step=True, on_epoch=True)
        self.log('train_regression_loss', regression_loss, on_step=True, on_epoch=True)
        self.log('train_classification_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        image, label, mmse, age, gender = batch['image'], batch['label'], batch['mmse'], batch['age'], batch['gender']
        metadata = torch.stack([mmse, age, gender], dim=1).float()
        
        classification_output, regression_output = self(image, metadata)
        
        classification_loss = self.classification_loss(classification_output, label)
        regression_loss = self.regression_loss(regression_output.squeeze(), mmse)
        
        losses = [classification_loss, regression_loss]
        total_loss, weights = self.frank_wolfe_loss(
            losses, 
            self.current_iteration, 
            self.total_iterations
        )
        
        preds = torch.argmax(classification_output, dim=1)
        acc = (preds == label).float().mean()
        
        self.log('val_loss', total_loss, on_epoch=True, prog_bar=True)
        self.log('val_classification_loss', classification_loss, on_epoch=True)
        self.log('val_regression_loss', regression_loss, on_epoch=True)
        self.log('val_classification_acc', acc, on_epoch=True, prog_bar=True)
        
        return total_loss
    
    def test_step(self, batch, batch_idx):
        image, label, mmse, age, gender = batch['image'], batch['label'], batch['mmse'], batch['age'], batch['gender']
        metadata = torch.stack([mmse, age, gender], dim=1).float()
        
        classification_output, regression_output = self(image, metadata)
        
        preds = torch.argmax(classification_output, dim=1)
        acc = (preds == label).float().mean()
        mae = F.l1_loss(regression_output.squeeze(), mmse)
        
        self.log('test_classification_acc', acc, on_epoch=True, prog_bar=True)
        self.log('test_regression_mae', mae, on_epoch=True, prog_bar=True)
        
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
            {'params': pretrained_params, 'lr': 1e-4},  # Lower learning rate for pretrained layers
            {'params': new_params, 'lr': 1e-3}          # Higher learning rate for new layers
        ])
        
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