import torch
# import torch
torch.set_float32_matmul_precision("medium")

import numpy as np
import random
from multitask.multitaskdataset import MultitaskAlzheimerDataset
# from multitask.visualize.visualize import visualize_and_save_gradcam,plot_and_save_optimization_metrics
import torchvision.transforms as transforms
import random
import wandb
from pytorch_lightning.loggers import WandbLogger
# from multitask.visualize.gradcam3D import GradCAM3D, save_gradcam



# from multitask.anothermodel.model_fix import MultiTaskAlzheimerModel
# from multitask.anothermodel.model_frank import MultiTaskAlzheimerModel
# from multitask.anothermodel.model_gradfrank import MultiTaskAlzheimerModel
from multitask.anothermodel.model import MultiTaskAlzheimerModel
from multitask.anothermodel.frank import MultiTaskAlzheimerModel
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.nn.functional import cross_entropy
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
# =============================================================
# export PYTHONPATH="${PYTHONPATH}:/home/toan/"
# =============================================================
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
                RandomRotation3D(degrees=10),  # Xoay ảnh ngẫu nhiên ±10 độ
                RandomTranslation3D(max_shift=0.1),  # Dịch chuyển tối đa 10%
                RandomFlip3D(p=0.5),  # Lật ảnh với xác suất 50%
                RandomIntensity(factor=0.1),  # Thay đổi độ sáng ±10%
                RandomNoise(std=0.02)  # Thêm nhiễu Gaussian
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
        # if (label==2):
        #     label=label-1

        return {
            'image': image,
            # 'label': label-1,
            'label': label,
            'mmse': mmse,
            'age': age,
            'gender': gender
        }
    
    def __len__(self):
        return len(self.data)
def main(wandb_logger):
    print('start')
    ad3y = list(torch.load('/home/toan/multitask/data/ad3y_skull.pt', weights_only=False))  
    mci3y = list(torch.load('/home/toan/multitask/data/mci3y_skull.pt', weights_only=False))
    cn3y = list(torch.load('/home/toan/multitask/data/cn3y_skull.pt', weights_only=False))
    ad1y = list(torch.load('/home/toan/multitask/data/ad1y_skull.pt', weights_only=False))
    cn2y = list(torch.load('/home/toan/multitask/data/cn2y_skull.pt', weights_only=False))
    ad=ad3y+ad1y
    cn=cn2y+cn3y
    ad=ad[:560]
    cn=cn[:560]
    
    print('AD: ',len(ad))
    print('CN: ',len(cn))
    # dataset_list = ad3y + mci3y + cn3y + ad1y + cn2y
    dataset_list =cn+mci3y+ad + cn+mci3y+ad+ cn+mci3y+ad 
    # dataset_list = dataset_list + mci3y  
    dataset=CustomDataset(dataset_list)
    # =====================================================
    batch_size = 16
    max_epochs = 100
    num_classes = 3
    input_shape = (1, 64, 64, 64) 
    
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,num_workers=4)
    print('==================================')
    model = MultiTaskAlzheimerModel(num_classes=num_classes)
 
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename='alzheimer-{epoch:02d}-{val_acc:.2f}',
        save_top_k=1,
        save_weights_only=True,
        mode='min'
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=100,
        verbose=True,
        mode='min'
    )
 
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        # accelerator='cpu'
        devices=1,  
        # devices=2,
        # strategy='ddp',
        callbacks=[checkpoint_callback, early_stop_callback],
        # logger=wandb_logger,
        deterministic=False
    )
    trainer.fit(
        model, 
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    trainer.test(model, dataloaders=test_loader)
    # wandb.finish()
    return test_loader,model

# ============================================================
if __name__ == '__main__':
    print('hi')
  
    # wandb_logger = WandbLogger(project="model_compare_ver2")
    wandb_logger=""
    test_loader,model =main(wandb_logger)
