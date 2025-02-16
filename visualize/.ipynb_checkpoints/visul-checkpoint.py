import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import os

class VisualizationUtils:
    def __init__(self, save_dir='visualization_results'):
        """
        Initialize with a directory to save all visualizations
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def plot_loss_weights(self, loss_history):
        """
        Plot and save the evolution of loss weights
        """
        weights = np.array(loss_history['weights'])
        epochs = range(len(weights))
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, weights[:, 0], 'b-', label='Classification Weight')
        plt.plot(epochs, weights[:, 1], 'r-', label='Regression Weight')
        plt.xlabel('Iterations')
        plt.ylabel('Loss Weight')
        plt.title('Loss Weight Evolution')
        plt.legend()
        plt.grid(True)
        
        save_path = os.path.join(self.save_dir, 'loss_weights.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Loss weights plot saved to: {save_path}")

    def plot_losses(self, loss_history):
        """
        Plot and save the evolution of individual losses
        """
        cls_losses = loss_history['classification']
        reg_losses = loss_history['regression']
        epochs = range(len(cls_losses))

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, cls_losses, 'b-', label='Classification Loss')
        plt.plot(epochs, reg_losses, 'r-', label='Regression Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss Value')
        plt.title('Training Losses Evolution')
        plt.legend()
        plt.grid(True)
        
        save_path = os.path.join(self.save_dir, 'training_losses.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Training losses plot saved to: {save_path}")

# class GradCAMVisualizer:
#     def __init__(self, model, save_dir='visualization_results'):
#         self.save_dir = save_dir
#         os.makedirs(save_dir, exist_ok=True)
        
#         # Get the last convolutional layer
#         self.target_layer = self._get_last_conv_layer(model)
#         self.model = model
#         self.grad_cam = GradCAM(
#             model=model,
#             target_layer=self.target_layer,
#             use_cuda=torch.cuda.is_available()
#         )

#     def _get_last_conv_layer(self, model):
#         """Find the last convolutional layer in the backbone"""
#         for module in reversed(list(model.backbone.modules())):
#             if isinstance(module, torch.nn.Conv3d):
#                 return module
#         raise ValueError("No convolutional layer found in model")

#     def generate_cam(self, input_tensor, target_class, slice_idx=None):
#         """
#         Generate and save Grad-CAM visualization
#         Args:
#             input_tensor: Input MRI scan (1, C, D, H, W)
#             target_class: Target class for visualization
#             slice_idx: Optional specific slice indices (D, H, W) to visualize
#         """
#         # Generate CAM
#         cam = self.grad_cam(input_tensor=input_tensor, target_category=target_class)
#         cam = cam[0, :]  # Take first item from batch
        
#         D, H, W = cam.shape
#         if slice_idx is None:
#             slice_idx = (D//2, H//2, W//2)
            
#         fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
#         # Sagittal view
#         axes[0].imshow(cam[slice_idx[0], :, :], cmap='jet')
#         axes[0].set_title('Sagittal View')
        
#         # Coronal view
#         axes[1].imshow(cam[:, slice_idx[1], :], cmap='jet')
#         axes[1].set_title('Coronal View')
        
#         # Axial view
#         axes[2].imshow(cam[:, :, slice_idx[2]], cmap='jet')
#         axes[2].set_title('Axial View')
        
#         plt.suptitle(f'Grad-CAM Visualization for Class {target_class}')
        
#         save_path = os.path.join(self.save_dir, f'gradcam_class_{target_class}.png')
#         plt.savefig(save_path)
#         plt.close()
#         print(f"Grad-CAM visualization saved to: {save_path}")

# Example usage in your training script:
class GradCAMVisualizer:
    def __init__(self, model, save_dir):
        self.model = model
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Get the target layer - in this case the last convolutional layer
        target_layer = self._get_target_layer()
        
        self.grad_cam = GradCAM(
            model=model,
            target_layers=[target_layer],
            use_cuda=torch.cuda.is_available()
        )
    
    def _get_target_layer(self):
        # Navigate through the backbone to find the last convolutional layer
        for name, module in self.model.backbone.named_modules():
            if isinstance(module, torch.nn.Conv3d):
                last_conv_layer = module
        return last_conv_layer
    
    def generate_cam(self, input_tensor, target_class):
        # Ensure input is on the correct device
        input_tensor = input_tensor.to(next(self.model.parameters()).device)
        
        # Generate class activation map
        grayscale_cam = self.grad_cam(input_tensor=input_tensor, 
                                    target_category=target_class)
        
        # Take a middle slice of the 3D volume for visualization
        middle_slice = input_tensor.shape[2] // 2
        image_slice = input_tensor[:, :, middle_slice, :, :].squeeze().cpu()
        
        # Normalize the image slice
        image_slice = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min())
        
        # Convert to RGB if necessary
        if image_slice.shape[0] == 1:  # If grayscale
            image_slice = torch.stack([image_slice[0]]*3)
        
        # Convert to numpy and transpose to correct format (H,W,C)
        image_slice = image_slice.permute(1, 2, 0).numpy()
        
        # Overlay the CAM
        visualization = show_cam_on_image(image_slice, 
                                        grayscale_cam[0],
                                        use_rgb=True)
        
        # Save the visualization
        plt.figure(figsize=(10, 10))
        plt.imshow(visualization)
        plt.axis('off')
        plt.savefig(os.path.join(self.save_dir, f'gradcam_class_{target_class}.png'))
        plt.close()
def save_visualizations(model, loss_history, sample_input, base_dir='results'):
    """
    Save all visualizations to specified directory
    """
    # Create visualization objects with specific save directories
    vis_utils = VisualizationUtils(save_dir=os.path.join(base_dir, 'plots'))
    grad_cam = GradCAMVisualizer(model, save_dir=os.path.join(base_dir, 'gradcam'))
    
    # Generate and save plots
    vis_utils.plot_loss_weights(loss_history)
    vis_utils.plot_losses(loss_history)
    
    # Generate and save Grad-CAM visualizations
    for class_idx in range(model.num_classes):  # For each class
        grad_cam.generate_cam(
            input_tensor=sample_input,
            target_class=class_idx
        )