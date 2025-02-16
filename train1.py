import torch
import numpy as np
import random
from multitask.multitaskdataset import MultitaskAlzheimerDataset
from multitask.visualize.visualize import visualize_and_save_gradcam,plot_and_save_optimization_metrics
import torchvision.transforms as transforms
import random
# from multitask.model_v1 import MultiTaskAlzheimerModel
# from multitask.model import MultiTaskAlzheimerModel
# from multitask.KKT import MultiTaskAlzheimerModel
# from multitask.crosseff import MultiTaskAlzheimerModel
# from multitask.frankwolfe import MultiTaskAlzheimerModel
# from multitask.resnet_original import MultiTaskAlzheimerModel
# from multitask.resnet18 import MultiTaskAlzheimerModel
# from multitask.resnet_frank import MultiTaskAlzheimerModel
# from multitask.resnet.frankwolfe_update import MultiTaskAlzheimerModel
# from multitask.resnet.extra import MultiTaskAlzheimerMosdel
from multitask.resnet.proposefrank import MultiTaskAlzheimerModel
# from multitask.resnet.nguyenban import MultiTaskAlzheimerModel
# from multitask.frank_1 import MultiTaskAlzheimerModel
# from multitask.frank1_3_2025 import MultiTaskAlzheimerModel

# from multitask.KKT_1 import MultiTaskAlzheimerModel
# import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.nn.functional import cross_entropy
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
# =============================================================
# export PYTHONPATH="${PYTHONPATH}:/home/jupyter-iec_iot13_toanlm/"
# =============================================================
class RandomRotation3D:
    def __init__(self, degrees):
        self.degrees = degrees
        
    def __call__(self, x):
        angle = random.uniform(-self.degrees, self.degrees)
        # Implement 3D rotation here
        return x  # Placeholder for actual implementation

class RandomTranslation3D:
    def __init__(self, max_shift):
        self.max_shift = max_shift
        
    def __call__(self, x):
        shift = random.uniform(-self.max_shift, self.max_shift)
        # Implement 3D translation here
        return x  # Placeholder for actual implementation

class RandomFlip3D:
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, x):
        if random.random() < self.p:
            dim = random.randint(0, 2)  # Chọn ngẫu nhiên chiều để lật
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
        # self.data = torch.load(file_path)
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
        # if(label!=2 or label !=1):
        #     print()
        #     print('husssssssssssssssssssssssssssssssssssssssssssssss')
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
def main():
    print('start')
    ad3y = list(torch.load('/home/jupyter-iec_iot13_toanlm/multitask/data/data_skull_v1/ad3y_skull.pt'))  
    mci3y = list(torch.load('/home/jupyter-iec_iot13_toanlm/multitask/data/data_skull_v1/mci3y_skull.pt'))
    cn3y = list(torch.load('/home/jupyter-iec_iot13_toanlm/multitask/data/data_skull_v1/cn3y_skull.pt'))
    ad1y = list(torch.load('/home/jupyter-iec_iot13_toanlm/multitask/data/data_skull_v1/ad1y_skull.pt'))
    cn2y = list(torch.load('/home/jupyter-iec_iot13_toanlm/multitask/data/data_skull_v1/cn2y_skull.pt'))
    ad=ad3y+ad1y
    cn=cn2y+cn3y
    ad=ad[:560]
    cn=cn[:560]
    
    print('AD: ',len(ad))
    print('CN: ',len(cn))
    # dataset_list = ad3y + mci3y + cn3y + ad1y + cn2y
    dataset_list =cn+ad+ad+cn+cn+ad
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
        save_top_k=3,
        mode='min'
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=10,
        verbose=True,
        mode='min'
    )
    
    # logger = TensorBoardLogger('tb_logs', name='alzheimer_3d_classification')
    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        # accelerator='cpu'
        # devices=[1],  
        devices=2,
        strategy='ddp',
        callbacks=[checkpoint_callback, early_stop_callback],
        # logger=logger,
        deterministic=False
    )
    # Training
    trainer.fit(
        model, 
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )
    # Testing
    trainer.test(model, dataloaders=test_loader)
    # save_path = '/home/jupyter-iec_iot13_toanlm/multitask/visualize'
    # test_batch = next(iter(test_loader))
    # image = test_batch['image'][0]
    # visualize_and_save_gradcam(model, image, save_path, num_slices=4)
    # plot_and_save_optimization_metrics(model, save_path)
    # visualize_results(model, model.loss_history, sample_input)
# ============================================================
if __name__ == '__main__':
    print('hi')
    main()

    