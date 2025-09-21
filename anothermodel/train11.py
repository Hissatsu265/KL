# import torch
# # import torch
# torch.set_float32_matmul_precision("medium")

# import numpy as np
# import random
# from multitask.multitaskdataset import MultitaskAlzheimerDataset
# # from multitask.visualize.visualize import visualize_and_save_gradcam,plot_and_save_optimization_metrics
# import torchvision.transforms as transforms
# import random
# import wandb
# from pytorch_lightning.loggers import WandbLogger
# # from multitask.visualize.gradcam3D import GradCAM3D, save_gradcam



# # from multitask.anothermodel.model_fix import MultiTaskAlzheimerModel
# # from multitask.anothermodel.model_frank import MultiTaskAlzheimerModel
# # from multitask.anothermodel.model_gradfrank import MultiTaskAlzheimerModel
# from multitask.anothermodel.model import MultiTaskAlzheimerModel
# from multitask.anothermodel.frank import MultiTaskAlzheimerModel
# import torch.nn.functional as F

# from torch.utils.data import Dataset, DataLoader, random_split
# import pytorch_lightning as pl
# from torch import nn
# from torch.utils.data import Dataset, DataLoader
# from torch.optim import Adam
# from torch.nn.functional import cross_entropy
# from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
# from pytorch_lightning.loggers import TensorBoardLogger
# # =============================================================
# # export PYTHONPATH="${PYTHONPATH}:/root/"
# # =============================================================
# class RandomRotation3D:
#     def __init__(self, degrees):
#         self.degrees = degrees
        
#     def __call__(self, x):
#         angle = random.uniform(-self.degrees, self.degrees)
#         return x  

# class RandomTranslation3D:
#     def __init__(self, max_shift):
#         self.max_shift = max_shift
        
#     def __call__(self, x):
#         shift = random.uniform(-self.max_shift, self.max_shift)
#         return x  

# class RandomFlip3D:
#     def __init__(self, p=0.5):
#         self.p = p
        
#     def __call__(self, x):
#         if random.random() < self.p:
#             dim = random.randint(0, 2) 
#             return torch.flip(x, [dim+1])  
#         return x

# class RandomIntensity:
#     def __init__(self, factor):
#         self.factor = factor
        
#     def __call__(self, x):
#         scale = 1.0 + random.uniform(-self.factor, self.factor)
#         return x * scale

# class RandomNoise:
#     def __init__(self, std):
#         self.std = std
        
#     def __call__(self, x):
#         noise = torch.randn_like(x) * self.std
#         return x + noise
    
# class CustomDataset:
#     def __init__(self, file_path):
#         self.data=file_path
#         self.is_train=True
#         self.train_transforms = transforms.Compose([
#                 RandomRotation3D(degrees=10),  # Xoay ảnh ngẫu nhiên ±10 độ
#                 RandomTranslation3D(max_shift=0.1),  # Dịch chuyển tối đa 10%
#                 RandomFlip3D(p=0.5),  # Lật ảnh với xác suất 50%
#                 RandomIntensity(factor=0.1),  # Thay đổi độ sáng ±10%
#                 RandomNoise(std=0.02)  # Thêm nhiễu Gaussian
#             ])
    
#     def __getitem__(self, idx):
#         image = self.data[idx]['image']
#         label = self.data[idx]['classification_label']
#         metadata = self.data[idx]['regression_targets']
        
#         if not isinstance(image, torch.Tensor):
#             image = torch.tensor(image, dtype=torch.float32)
#         if image.dim() == 3:
#             image = image.unsqueeze(0) 
#         if self.is_train:
#             image = self.train_transforms(image)
            
#         label = torch.tensor(label, dtype=torch.long)
#         mmse = torch.tensor(float(metadata['mmse']), dtype=torch.float32)
#         age = torch.tensor(float(metadata['age']), dtype=torch.float32)
#         gender = torch.tensor(float(metadata['gender']), dtype=torch.float32)
#         # if (label==2):
#         #     label=label-1

#         return {
#             'image': image,
#             # 'label': label-1,
#             'label': label,
#             'mmse': mmse,
#             'age': age,
#             'gender': gender
#         }
    
#     def __len__(self):
#         return len(self.data)
# def main(wandb_logger):
#     print('start')
#     ad3y = list(torch.load('/root/multitask/data/ad3y_skull.pt', weights_only=False))  
#     mci3y = list(torch.load('/root/multitask/data/mci3y_skull.pt', weights_only=False))
#     cn3y = list(torch.load('/root/multitask/data/cn3y_skull.pt', weights_only=False))
#     ad1y = list(torch.load('/root/multitask/data/ad1y_skull.pt', weights_only=False))
#     cn2y = list(torch.load('/root/multitask/data/cn2y_skull.pt', weights_only=False))
#     ad=ad3y+ad1y
#     cn=cn2y+cn3y
#     ad=ad[:560]
#     cn=cn[:560]
    
#     print('AD: ',len(ad))
#     print('CN: ',len(cn))
#     # dataset_list = ad3y + mci3y + cn3y + ad1y + cn2y
#     dataset_list =cn+mci3y+ad + cn+mci3y+ad+ cn+mci3y+ad 
#     # dataset_list = dataset_list + mci3y  
#     dataset=CustomDataset(dataset_list)
#     # =====================================================
#     batch_size = 16
#     max_epochs = 100
#     num_classes = 3
#     input_shape = (1, 64, 64, 64) 
    
#     train_size = int(0.7 * len(dataset))
#     val_size = int(0.15 * len(dataset))
#     test_size = len(dataset) - train_size - val_size
    
#     train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=4)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size,num_workers=4)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size,num_workers=4)
#     print('==================================')
#     model = MultiTaskAlzheimerModel(num_classes=num_classes)
 
#     checkpoint_callback = ModelCheckpoint(
#         monitor='val_loss',
#         dirpath='checkpoints/',
#         filename='alzheimer-{epoch:02d}-{val_acc:.2f}',
#         save_top_k=1,
#         save_weights_only=True,
#         mode='min'
#     )
    
#     early_stop_callback = EarlyStopping(
#         monitor='val_loss',
#         min_delta=0.00,
#         patience=100,
#         verbose=True,
#         mode='min'
#     )
 
#     trainer = pl.Trainer(
#         max_epochs=max_epochs,
#         accelerator='gpu' if torch.cuda.is_available() else 'cpu',
#         # accelerator='cpu'
#         devices=1,  
#         # devices=2,
#         # strategy='ddp',
#         callbacks=[checkpoint_callback, early_stop_callback],
#         # logger=wandb_logger,
#         deterministic=False
#     )
#     trainer.fit(
#         model, 
#         train_dataloaders=train_loader,
#         val_dataloaders=val_loader,
#     )
#     trainer.test(model, dataloaders=test_loader)
#     # wandb.finish()
#     return test_loader,model

# # ============================================================
# if __name__ == '__main__':
#     print('hi')
  
#     # wandb_logger = WandbLogger(project="model_compare_ver2")
#     wandb_logger=""
#     test_loader,model =main(wandb_logger)
# Sửa đổi file main (train.py hoặc main.py)
import torch
# torch.set_float32_matmul_precision("medium")

import numpy as np
import random
from multitask.multitaskdataset import MultitaskAlzheimerDataset
import torchvision.transforms as transforms
import random
import wandb
from pytorch_lightning.loggers import WandbLogger
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Import model classes (giữ nguyên như code gốc)
from multitask.anothermodel.model import MultiTaskAlzheimerModel
from multitask.anothermodel.frank_wandb import MultiTaskAlzheimerModel
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.nn.functional import cross_entropy
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# Giữ nguyên các class transform và CustomDataset
class RandomRotation3D:
    def __init__(self, degrees):
        self.degrees = degrees
        
    def __call__(self, x):
        angle = random.uniform(-self.degrees, self.degrees)
        return x  

class RandomTranslation3D:
    def __init__(self, max_shift):
        self.max_shift = max_shift
        
    def __call__(self, x):
        shift = random.uniform(-self.max_shift, self.max_shift)
        return x  

class RandomFlip3D:
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, x):
        if random.random() < self.p:
            dim = random.randint(0, 2) 
            return torch.flip(x, [dim+1])  
        return x

class RandomIntensity:
    def __init__(self, factor):
        self.factor = factor
        
    def __call__(self, x):
        scale = 1.0 + random.uniform(-self.factor, self.factor)
        return x * scale

class RandomNoise:
    def __init__(self, std):
        self.std = std
        
    def __call__(self, x):
        noise = torch.randn_like(x) * self.std
        return x + noise
    
class CustomDataset:
    def __init__(self, file_path):
        self.data=file_path
        self.is_train=True
        self.train_transforms = transforms.Compose([
                RandomRotation3D(degrees=10),
                RandomTranslation3D(max_shift=0.1),
                RandomFlip3D(p=0.5),
                RandomIntensity(factor=0.1),
                RandomNoise(std=0.02)
            ])
    
    def __getitem__(self, idx):
        image = self.data[idx]['image']
        label = self.data[idx]['classification_label']
        metadata = self.data[idx]['regression_targets']
        
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32)
        if image.dim() == 3:
            image = image.unsqueeze(0) 
        if self.is_train:
            image = self.train_transforms(image)
            
        label = torch.tensor(label, dtype=torch.long)
        mmse = torch.tensor(float(metadata['mmse']), dtype=torch.float32)
        age = torch.tensor(float(metadata['age']), dtype=torch.float32)
        gender = torch.tensor(float(metadata['gender']), dtype=torch.float32)

        return {
            'image': image,
            'label': label,
            'mmse': mmse,
            'age': age,
            'gender': gender
        }
    
    def __len__(self):
        return len(self.data)

def plot_data_distribution(ad_count, cn_count, mci_count):
    """Vẽ biểu đồ phân phối dữ liệu"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['AD', 'CN', 'MCI']
    counts = [ad_count, cn_count, mci_count]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    bars = ax.bar(categories, counts, color=colors, alpha=0.8)
    ax.set_xlabel('Diagnosis Categories')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Dataset Distribution')
    
    # Thêm số lượng lên trên mỗi bar
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + count*0.01,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Log to wandb
    wandb.log({"Data Distribution": wandb.Image(fig)})
    plt.close(fig)

def plot_training_overview(train_size, val_size, test_size, total_size):
    """Vẽ biểu đồ chia tập dữ liệu"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Pie chart cho tỷ lệ chia data
    sizes = [train_size, val_size, test_size]
    labels = ['Training', 'Validation', 'Testing']
    colors = ['#FF9999', '#66B2FF', '#99FF99']
    
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Dataset Split Distribution')
    
    # Bar chart cho số lượng samples
    ax2.bar(labels, sizes, color=colors, alpha=0.8)
    ax2.set_xlabel('Dataset Split')
    ax2.set_ylabel('Number of Samples')
    ax2.set_title('Sample Count by Split')
    
    # Thêm số lượng lên trên mỗi bar
    for i, (label, size) in enumerate(zip(labels, sizes)):
        ax2.text(i, size + total_size*0.01, f'{size}', 
                ha='center', va='bottom', fontweight='bold')
    
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Log to wandb
    wandb.log({"Training Overview": wandb.Image(fig)})
    plt.close(fig)

def main(config=None):
    # Khởi tạo wandb
    wandb.init(
        project="multitask-alzheimer-detection",
        config={
            "batch_size": 16,
            "max_epochs": 100,
            "num_classes": 3,
            "input_shape": (1, 64, 64, 64),
            "learning_rate": 1e-3,
            "architecture": "R3D-18 + Multi-task",
            "optimizer": "Frank-Wolfe",
            "augmentation": True
        }
    )
    
    # Cập nhật config nếu có
    if config:
        wandb.config.update(config)
    
    config = wandb.config
    
    print('Loading data...')
    ad3y = list(torch.load('/root/multitask/data/ad3y_skull.pt', weights_only=False))  
    mci3y = list(torch.load('/root/multitask/data/mci3y_skull.pt', weights_only=False))
    cn3y = list(torch.load('/root/multitask/data/cn3y_skull.pt', weights_only=False))
    ad1y = list(torch.load('/root/multitask/data/ad1y_skull.pt', weights_only=False))
    cn2y = list(torch.load('/root/multitask/data/cn2y_skull.pt', weights_only=False))
    
    ad = ad3y + ad1y
    cn = cn2y + cn3y
    ad = ad[:560]
    cn = cn[:560]
    
    print(f'AD: {len(ad)}')
    print(f'CN: {len(cn)}')
    print(f'MCI: {len(mci3y)}')
    
    # Vẽ biểu đồ phân phối dữ liệu
    plot_data_distribution(len(ad), len(cn), len(mci3y))
    
    dataset_list = cn + mci3y + ad + cn + mci3y + ad + cn + mci3y + ad 
    dataset = CustomDataset(dataset_list)
    
    # Chia dataset
    batch_size = config.batch_size
    max_epochs = config.max_epochs
    num_classes = config.num_classes
    
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    # Vẽ biểu đồ chia dataset
    plot_training_overview(train_size, val_size, test_size, len(dataset))
    
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)
    
    print('Initializing model...')
    model = MultiTaskAlzheimerModel(num_classes=num_classes)
    
    # Khởi tạo wandb logger
    wandb_logger = WandbLogger(project="multitask-alzheimer-detection")
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename='alzheimer-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,
        save_weights_only=False,
        mode='min'
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.001,
        patience=15,
        verbose=True,
        mode='min'
    )
    
    # Trainer với wandb logger
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=wandb_logger,
        deterministic=False,
        log_every_n_steps=10
    )
    
    print('Starting training...')
    trainer.fit(
        model, 
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    
    print('Running test...')
    trainer.test(model, dataloaders=test_loader)
    
    # Log model architecture
    wandb.log({"model_summary": str(model)})
    
    # Finish wandb run
    wandb.finish()
    
    return test_loader, model

if __name__ == '__main__':
    print('Starting Multi-task Alzheimer Detection Training...')
    
    custom_config = {
        "experiment_name": "frank_wolfe_r3d18",
        "notes": "Multi-task learning with Frank-Wolfe optimizer and R3D-18 backbone"
    }
    
    test_loader, model = main(custom_config)