import torch
import numpy as np
import random
from multitask.multitaskdataset import MultitaskAlzheimerDataset
from multitask.visualize.visualize import visualize_and_save_gradcam,plot_and_save_optimization_metrics
import torchvision.transforms as transforms

import wandb
from pytorch_lightning.loggers import WandbLogger
from multitask.visualize.gradcam3D import GradCAM3D, save_gradcam
from multitask.multitaskdataset import MultitaskAlzheimerDataset
from multitask.anothermodel.modelrep11 import MultiTaskAlzheimerModel
import torch
import nibabel as nib
from scipy.ndimage import zoom

# =============================================================
# export PYTHONPATH="${PYTHONPATH}:/home/jupyter-iec_iot13_toanlm/"
# =============================================================
import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from multitask.anothermodel.model1 import MultiTaskAlzheimerModel
from torchvision.models.video import r3d_18, R3D_18_Weights

import time

# from multitask.anothermodel.model_fix import MultiTaskAlzheimerModel
# from multitask.anothermodel.model_frank import MultiTaskAlzheimerModel
# from multitask.anothermodel.model_grad import MultiTaskAlzheimerModel
# from multitask.anothermodel.model_gradfrank import MultiTaskAlzheimerModel
# from multitask.anothermodel.model import MultiTaskAlzheimerModel
from multitask.anothermodel.model1 import MultiTaskAlzheimerModel
from multitask.anothermodel.modelrep11 import MultiTaskAlzheimerModel
# from multitask.anothermodel.model_multi import MultiTaskAlzheimerModel
# from multitask.anothermodel.model_ver2 import MultiTaskAlzheimerModel
import torch.nn.functional as F
import copy

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
class AugmentedDataset(Dataset):
        def __init__(self, original_dataset, augmentation_factor=3):
            self.original_dataset = original_dataset
            self.augmentation_factor = augmentation_factor
            
        def __len__(self):
            return len(self.original_dataset) * self.augmentation_factor
            
        def __getitem__(self, idx):
            original_idx = idx % len(self.original_dataset)
            sample = self.original_dataset[original_idx]
            
            if idx // len(self.original_dataset) > 0:
                # Tạo bản sao để tránh ảnh hưởng đến mẫu gốc
                augmented_sample = copy.deepcopy(sample)
                
                image = augmented_sample['image']
                
                augmentation_round = idx // len(self.original_dataset)
                
                if augmentation_round == 1:
                    transforms_set1 = transforms.Compose([
                        RandomRotation3D(degrees=15),
                        RandomFlip3D(p=0.7),
                        RandomIntensity(factor=0.15)
                    ])
                    image = transforms_set1(image)
                else:  # augmentation_round == 2
                    # Áp dụng bộ augmentation thứ hai
                    transforms_set2 = transforms.Compose([
                        RandomTranslation3D(max_shift=0.15),
                        RandomNoise(std=0.03),
                        RandomIntensity(factor=0.12)
                    ])
                    image = transforms_set2(image)
                
                augmented_sample['image'] = image
                return augmented_sample
            else:
                return sample
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
        self.is_train=False
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
    dataset_list =cn+ad
    # dataset_list = dataset_list + mci3y  
    dataset=CustomDataset(dataset_list)
    # =====================================================
    batch_size = 16
    max_epochs = 10
    num_classes = 2
    input_shape = (1, 64, 64, 64) 
    
    train_size = int(0.7 * len(dataset))
    test_size = 225
    
    val_size = len(dataset) - train_size - test_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    for i in test_dataset:
        print(i)
    # train_dataset=train_dataset+train_dataset+train_dataset
    # test_dataset=test_dataset+test_dataset+test_dataset
    
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=4)
    augmented_train_dataset = AugmentedDataset(train_dataset, augmentation_factor=3)    
    train_loader = DataLoader(augmented_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size,num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,num_workers=4)
    
    print('==================================')
    model = MultiTaskAlzheimerModel(num_classes=num_classes)
 
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename='alzheimer-{epoch:02d}-{val_acc:.2f}',
        save_top_k=1,
        save_weights_only=False,
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
        # devices=[1],  
        devices=2,
        strategy='ddp',
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=wandb_logger,
        deterministic=False
    )
    s=time.time()
    trainer.fit(
        model, 
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    s1=time.time()
    
    trainer.test(model, dataloaders=test_loader)
    torch.save(model.state_dict(), 'model_weights.pth')

    e=time.time()
    print("===============================================================")
    print("Time: ",e-s)
    print("Time: ",(s1-s)/100)
    wandb.finish()
    return test_loader,model
# ============================================================================
def normalize(img):
    # Normalize ảnh để có mean=0 và std=1
    if isinstance(img, torch.Tensor):
        mean = torch.mean(img)
        std = torch.std(img)
        return (img - mean) / (std + 1e-8)
    else:
        mean = np.mean(img)
        std = np.std(img)
        return (img - mean) / (std + 1e-8)

def smoothing(img, sigma=1.0):
    # Áp dụng Gaussian smoothing
    if isinstance(img, torch.Tensor):
        img_np = img.numpy()
        smoothed_img = gaussian_filter(img_np, sigma=sigma)
        return torch.tensor(smoothed_img, dtype=torch.float32)
    else:
        return gaussian_filter(img, sigma=sigma)

def preprocess_mri_image(image_path, target_shape=(64, 64, 64)):
    """
    Tiền xử lý ảnh MRI giống như trong quá trình training
    """
    # Load ảnh từ file
    img = nib.load(image_path)
    img_data = img.get_fdata()
    
    # Chuyển đổi dữ liệu ảnh thành tensor
    img_tensor = torch.tensor(img_data, dtype=torch.float32).unsqueeze(0)
    
    # Resize ảnh nếu cần
    if img_tensor.shape[1:] != target_shape:
        img_tensor = F.interpolate(img_tensor.unsqueeze(0), 
                                    size=target_shape, 
                                    mode='trilinear', 
                                    align_corners=False).squeeze(0)
    
    # Áp dụng normalize
    img_tensor = normalize(img_tensor)
    
    # Áp dụng smoothing
    img_tensor = smoothing(img_tensor)
    
    # Định dạng lại tensor để phù hợp với đầu vào mô hình (batch_size, channels, D, H, W)
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)  # Thêm chiều channel
    if img_tensor.dim() == 4:
        img_tensor = img_tensor.unsqueeze(0)  # Thêm chiều batch
        
    return img_tensor

# Load mô hình
def load_model(weights_path, device='cuda'):
    model = MultiTaskAlzheimerModel(num_classes=2)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Hàm dự đoán
def predict(model, image_path, age, gender, device='cuda'):
   
    image_tensor = preprocess_mri_image(image_path)
    image_tensor = image_tensor.to(device)
    
    # Tạo metadata tensor
    metadata = torch.tensor([[float(age), float(gender)]], dtype=torch.float32).to(device)
    
    # In thông tin để debug
    print(f"Image tensor shape: {image_tensor.shape}")
    print(f"Image tensor min: {image_tensor.min()}, max: {image_tensor.max()}, mean: {image_tensor.mean()}")
    print(f"Metadata: {metadata}")
    
    # Dự đoán
    with torch.no_grad():
        classification_logits, mmse_pred = model(image_tensor, metadata)
        
        # In ra logits để debug
        print(f"Raw logits: {classification_logits}")
        
        # Tính xác suất cho mỗi lớp
        class_probs = torch.softmax(classification_logits, dim=1)
        print(f"Class probabilities: {class_probs}")
        
        predicted_class = torch.argmax(class_probs, dim=1).item()
        
        # Lấy xác suất của từng lớp
        class_probabilities = class_probs[0].cpu().numpy()
        
        # Lấy giá trị MMSE dự đoán
        mmse_prediction = mmse_pred.item()
    
    results = {
        'predicted_class': predicted_class,  # 0: CN (Normal), 1: AD (Alzheimer's Disease)
        'class_probabilities': class_probabilities,
        'predicted_mmse': mmse_prediction
    }
    
    # In kết quả dự đoán
    class_name = "CN (Bình thường)" if predicted_class == 0 else "AD (Bệnh Alzheimer)"
    cn_probability = class_probabilities[0] * 100
    ad_probability = class_probabilities[1] * 100
    
    print("\nKết quả dự đoán:")
    print(f"Phân loại: {class_name}")
    print(f"Xác suất CN (Bình thường): {cn_probability:.2f}%")
    print(f"Xác suất AD (Bệnh Alzheimer): {ad_probability:.2f}%")
    print(f"Điểm MMSE dự đoán: {mmse_prediction:.2f}")
    
    return results
def check_model(model):
    """
    Kiểm tra mô hình đã load để đảm bảo không có vấn đề
    """
    # Kiểm tra parameters
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {param_count}")
    
    # Kiểm tra có parameters nào là NaN không
    has_nan = any(torch.isnan(p).any() for p in model.parameters())
    print(f"Has NaN parameters: {has_nan}")
    
    # Kiểm tra gradient
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Parameter {name} requires gradient")
    
    return not has_nan
# ============================================================
if __name__ == '__main__':
    print('hi')
    # wandb.init(
    #     project="model_compare_ver2",
    #     config={
    #         "architecture": "R3D_18",
    #         "dataset": "ADNI",
    #         "batch_size": 16,
    #         "learning_rate": 1e-3,
    #         "num_classes": 2,
    #         "epochs": 100,
    #     }
    # )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wandb_logger = WandbLogger(project="model_compare_ver2")
    test_loader,model =main(wandb_logger)
    results = predict(model, "/home/jupyter-iec_iot13_toanlm/testmodel/cn1.nii", 76, 1, device)
# from multitask.anothermodel.model_ver2 import MultiTaskAlzheimerModel
# from pytorch_lightning import LightningModule

# # Khởi tạo model với cùng cấu trúc
# model = MultiTaskAlzheimerModel(num_classes=2)

# # Load checkpoint đã lưu
# checkpoint_path = 'checkpoints/best_model-epoch.ckpt'  # thay bằng đúng file bạn lưu được
# model = model.load_from_checkpoint(checkpoint_path)

# ============================================================================================
# model = MultiTaskAlzheimerModel(num_classes=2)
# model.load_state_dict(torch.load('model_weights.pth'))
# model.eval()