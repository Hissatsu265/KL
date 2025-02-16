# import torch
# import torch.nn.functional as F
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# import cv2
# import os

# class GradCAM:
#     def __init__(self, model, target_layer):
#         self.model = model
#         self.target_layer = target_layer
#         self.gradients = None
#         self.activations = None
        
#         # Register hooks
#         target_layer.register_forward_hook(self.save_activation)
#         target_layer.register_full_backward_hook(self.save_gradient)
    
#     def save_activation(self, module, input, output):
#         self.activations = output.detach()
    
#     def save_gradient(self, module, grad_input, grad_output):
#         self.gradients = grad_output[0].detach()
    
#     def __call__(self, x, class_idx=None):
#         # Forward pass
#         self.model.eval()
#         classification_output, _ = self.model(x, torch.zeros(x.size(0), 3).to(x.device))
        
#         if class_idx is None:
#             class_idx = torch.argmax(classification_output, dim=1)
        
#         # Zero gradients
#         self.model.zero_grad()
        
#         # Target for backprop
#         one_hot = torch.zeros_like(classification_output)
#         one_hot[range(len(class_idx)), class_idx] = 1
        
#         # Backward pass
#         classification_output.backward(gradient=one_hot, retain_graph=True)
        
#         # Global average pooling of gradients
#         weights = torch.mean(self.gradients, dim=(2, 3, 4))
        
#         # Weight the activations
#         cam = torch.sum(weights[:, :, None, None, None] * self.activations, dim=1)
        
#         # ReLU and normalize
#         cam = F.relu(cam)
#         cam = F.interpolate(cam.unsqueeze(1), size=x.shape[2:], mode='trilinear', align_corners=False)
#         cam = cam.squeeze(1)
        
#         # Normalize between 0 and 1
#         cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
#         return cam

# def visualize_and_save_gradcam(model, image, save_path, target_class=None, num_slices=4):
  
#     os.makedirs(save_path, exist_ok=True)
    
#     # Get the target layer (last convolutional layer)
#     target_layer = None
#     for name, module in model.named_modules():
#         if isinstance(module, torch.nn.Conv3d):
#             target_layer = module
    
#     gradcam = GradCAM(model, target_layer)
    
#     # Generate heatmap
#     heatmap = gradcam(image.unsqueeze(0), target_class)
#     heatmap = heatmap[0].cpu().numpy()
    
#     # Select representative slices
#     D = image.shape[2]
#     slice_indices = np.linspace(D//4, 3*D//4, num_slices, dtype=int)
    
#     # Create a figure with all slices
#     fig, axes = plt.subplots(num_slices, 3, figsize=(15, 5*num_slices))
#     fig.suptitle('GradCAM Visualization for Multiple Slices', fontsize=16)
    
#     for i, slice_idx in enumerate(slice_indices):
#         # Get the slice
#         image_slice = image[0, 0, slice_idx].cpu().numpy()
#         heatmap_slice = heatmap[slice_idx]
        
#         # Original image
#         axes[i, 0].imshow(image_slice, cmap='gray')
#         axes[i, 0].set_title(f'Original Image (Slice {slice_idx})')
#         axes[i, 0].axis('off')
        
#         # Heatmap
#         axes[i, 1].imshow(heatmap_slice, cmap='jet')
#         axes[i, 1].set_title(f'GradCAM Heatmap (Slice {slice_idx})')
#         axes[i, 1].axis('off')
        
#         # Overlay
#         axes[i, 2].imshow(image_slice, cmap='gray')
#         axes[i, 2].imshow(heatmap_slice, cmap='jet', alpha=0.5)
#         axes[i, 2].set_title(f'Overlay (Slice {slice_idx})')
#         axes[i, 2].axis('off')
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_path, 'gradcam_visualization.png'), dpi=300, bbox_inches='tight')
#     plt.close()

# def plot_and_save_optimization_metrics(model, save_path):

#     os.makedirs(save_path, exist_ok=True)
    
#     classification_losses = model.loss_history['classification']
#     regression_losses = model.loss_history['regression']
#     weights = np.array(model.loss_history['weights'])
    
#     plt.figure(figsize=(15, 5))
    
#     # Plot losses
#     plt.subplot(1, 2, 1)
#     plt.plot(classification_losses, label='Classification Loss')
#     plt.plot(regression_losses, label='Regression Loss')
#     plt.title('Loss Evolution')
#     plt.xlabel('Iteration')
#     plt.ylabel('Loss')
#     plt.legend()
    
#     # Plot weights
#     plt.subplot(1, 2, 2)
#     plt.plot(weights[:, 0], label='Classification Weight')
#     plt.plot(weights[:, 1], label='Regression Weight')
#     plt.title('Task Weights Evolution')
#     plt.xlabel('Iteration')
#     plt.ylabel('Weight')
#     plt.legend()
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_path, 'optimization_metrics.png'), dpi=300, bbox_inches='tight')
#     plt.close()
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def __call__(self, x, class_idx=None):
        # Forward pass
        self.model.eval()
        classification_output, _ = self.model(x, torch.zeros(x.size(0), 3).to(x.device))
        
        if class_idx is None:
            class_idx = torch.argmax(classification_output, dim=1)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Target for backprop
        one_hot = torch.zeros_like(classification_output)
        one_hot[range(len(class_idx)), class_idx] = 1
        
        # Backward pass
        classification_output.backward(gradient=one_hot, retain_graph=True)
        
        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3, 4))
        
        # Weight the activations
        cam = torch.sum(weights[:, :, None, None, None] * self.activations, dim=1)
        
        # ReLU and normalize
        cam = F.relu(cam)
        cam = F.interpolate(cam.unsqueeze(1), size=x.shape[2:], mode='trilinear', align_corners=False)
        cam = cam.squeeze(1)
        
        # Normalize between 0 and 1
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam

def reshape_3d_image(image):
    """
    Reshape image to correct format for visualization
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    
    # If image is 1D, try to reshape it to 2D
    if len(image.shape) == 1:
        size = int(np.sqrt(image.shape[0]))
        if size * size == image.shape[0]:
            image = image.reshape(size, size)
        else:
            # Try to find factors that work
            for i in range(int(np.sqrt(image.shape[0])), 0, -1):
                if image.shape[0] % i == 0:
                    image = image.reshape(i, image.shape[0]//i)
                    break
    
    return image

def visualize_and_save_gradcam(model, image, save_path, target_class=None, num_slices=4):
    """
    Visualize GradCAM results for selected slices and save to specified path
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Ensure image is in correct format
    if isinstance(image, torch.Tensor):
        if image.dim() == 3:  # If missing batch dimension
            image = image.unsqueeze(0)
        if image.dim() == 2:  # If 2D image
            image = image.unsqueeze(0).unsqueeze(0)
    
    # Get the target layer (last convolutional layer)
    target_layer = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv3d):
            target_layer = module
    
    if target_layer is None:
        print("No Conv3d layer found in the model")
        return
    
    gradcam = GradCAM(model, target_layer)
    
    # Generate heatmap
    with torch.no_grad():
        heatmap = gradcam(image.to(next(model.parameters()).device), target_class)
        heatmap = heatmap[0].cpu().numpy()
    
    # Select representative slices
    D = image.shape[2] if image.dim() > 3 else int(np.cbrt(image.shape[-1]))
    slice_indices = np.linspace(D//4, 3*D//4, num_slices, dtype=int)
    
    # Create a figure with all slices
    fig, axes = plt.subplots(num_slices, 3, figsize=(15, 5*num_slices))
    if num_slices == 1:
        axes = axes.reshape(1, -1)
    
    for i, slice_idx in enumerate(slice_indices):
        # Get the slice and ensure it's 2D
        if image.dim() > 3:
            image_slice = image[0, 0, slice_idx].cpu().numpy()
        else:
            image_slice = reshape_3d_image(image[0, 0])
        
        heatmap_slice = reshape_3d_image(heatmap[slice_idx] if heatmap.ndim > 2 else heatmap)
        
        # Plot original image
        axes[i, 0].imshow(image_slice, cmap='gray')
        axes[i, 0].set_title(f'Original Image (Slice {slice_idx})')
        axes[i, 0].axis('off')
        
        # Plot heatmap
        axes[i, 1].imshow(heatmap_slice, cmap='jet')
        axes[i, 1].set_title(f'GradCAM Heatmap (Slice {slice_idx})')
        axes[i, 1].axis('off')
        
        # Plot overlay
        axes[i, 2].imshow(image_slice, cmap='gray')
        axes[i, 2].imshow(heatmap_slice, cmap='jet', alpha=0.5)
        axes[i, 2].set_title(f'Overlay (Slice {slice_idx})')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'gradcam_visualization.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_and_save_optimization_metrics(model, save_path):
    """
    Plot and save the optimization metrics from the training process
    """
    os.makedirs(save_path, exist_ok=True)
    
    classification_losses = model.loss_history['classification']
    regression_losses = model.loss_history['regression']
    weights = np.array(model.loss_history['weights'])
    
    plt.figure(figsize=(15, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(classification_losses, label='Classification Loss')
    plt.plot(regression_losses, label='Regression Loss')
    plt.title('Loss Evolution')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot weights
    plt.subplot(1, 2, 2)
    plt.plot(weights[:, 0], label='Classification Weight')
    plt.plot(weights[:, 1], label='Regression Weight')
    plt.title('Task Weights Evolution')
    plt.xlabel('Iteration')
    plt.ylabel('Weight')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'optimization_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()