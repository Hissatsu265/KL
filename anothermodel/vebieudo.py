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
import torch
import numpy as np

import random
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split

import wandb
import torch
import nibabel as nib
from scipy.ndimage import zoom

# =============================================================
# export PYTHONPATH="${PYTHONPATH}:/home/jupyter-iec_iot13_toanlm/"
# =============================================================
from scipy.ndimage import gaussian_filter
# =====================================================================
def normalize(img):
    if isinstance(img, torch.Tensor):
        min_val = img.min()
        max_val = img.max()
        return (img - min_val) / (max_val - min_val + 1e-8)
    else:
        min_val = np.min(img)
        max_val = np.max(img)
        return (img - min_val) / (max_val - min_val + 1e-8)

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
class CustomDataset:
    def __init__(self, file_path):
        self.data=file_path
        self.is_train=False
        # self.train_transforms = transforms.Compose([
        #         RandomRotation3D(degrees=10),  # Xoay ảnh ngẫu nhiên ±10 độ
        #         RandomTranslation3D(max_shift=0.1),  # Dịch chuyển tối đa 10%
        #         RandomFlip3D(p=0.5),  # Lật ảnh với xác suất 50%
        #         RandomIntensity(factor=0.1),  # Thay đổi độ sáng ±10%
        #         RandomNoise(std=0.02)  # Thêm nhiễu Gaussian
        #     ])
    
    def __getitem__(self, idx):
        image = self.data[idx]['image']
        label = self.data[idx]['classification_label']
        metadata = self.data[idx]['regression_targets']
        
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32)
        if image.dim() == 3:
            image = image.unsqueeze(0) 
  
            
        label = torch.tensor(label, dtype=torch.long)
        mmse = torch.tensor(float(metadata['mmse']), dtype=torch.float32)
        age = torch.tensor(float(metadata['age']), dtype=torch.float32)
        gender = torch.tensor(float(metadata['gender']), dtype=torch.float32)
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
# Hàm dự đoán
def predict(model, image_path, age, gender, device='cuda'):
    
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

    # dataset=CustomDataset(cn3y)
    t=0
    for sample in test_dataset:
        # print(i)
        # image_tensor = sample['image'].to(device)  # shape: [1, C, H, W]
        image_tensor = sample['image'].unsqueeze(1).to(device)
        # print("Min:", image_tensor.min().item())
        # print("Max:", image_tensor.max().item())
        # print("Mean:", image_tensor.mean().item())
        # print("-=p=p=p=p=pd=apd=psd=apd=pa=pd=dp=")
        # print(image_tensor.shape)
        age = sample['age'].item()                # lấy giá trị float từ tensor
        gender = sample['gender'].item()
        metadata = torch.tensor([[float(age), float(gender)]], dtype=torch.float32).to(device)
        print('True score: ',sample['mmse'].item())
        with torch.no_grad():
            classification_logits, mmse_pred = model(image_tensor, metadata)
            
            
            # print(f"Raw logits: {classification_logits}")
            
            
            class_probs = torch.softmax(classification_logits, dim=1)
            # print(f"Class probabilities: {class_probs}")
            
            predicted_class = torch.argmax(class_probs, dim=1).item()
            
            
            class_probabilities = class_probs[0].cpu().numpy()
            
            
            mmse_prediction = mmse_pred.item()
            print("predict score: ", mmse_prediction )
        # results = {
        #     'predicted_class': predicted_class,  # 0: CN (Normal), 1: AD (Alzheimer's Disease)
        #     'class_probabilities': class_probabilities,
        #     'predicted_mmse': mmse_prediction
        # }
        
        # # In kết quả dự đoán
        # class_name = "CN (Bình thường)" if predicted_class == 0 else "AD (Bệnh Alzheimer)"
        # cn_probability = class_probabilities[0] * 100
        # ad_probability = class_probabilities[1] * 100
        
        # print("\nKết quả dự đoán:")
        # print(f"Phân loại: {class_name}")
        # print(f"Xác suất CN (Bình thường): {cn_probability:.2f}%")
        # print(f"Xác suất AD (Bệnh Alzheimer): {ad_probability:.2f}%")
        # print(f"Điểm MMSE dự đoán: {mmse_prediction:.2f}")
        
    
    
    
    results = {
        'predicted_class': '',  # 0: CN (Normal), 1: AD (Alzheimer's Disease)
        'class_probabilities':'',
        'predicted_mmse':''
    }
    
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

# Hàm sử dụng chính để dự đoán
def predict_alzheimer_disease(image_path, age, gender, model_path='model_weights.pth'):

    # Thiết lập device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load mô hình
    model = load_model(model_path, device)
    
    # Kiểm tra mô hình
    if not check_model(model):
        print("WARNING: Phát hiện vấn đề với mô hình!")
    
    # Debug mode - thử với một tensor giả
    print("\nTesting with dummy data...")
    dummy_image = torch.randn(1, 1, 64, 64, 64).to(device)
    dummy_metadata = torch.tensor([[float(age), float(gender)]]).to(device)
    
    with torch.no_grad():
        try:
            dummy_outputs = model(dummy_image, dummy_metadata)
            print(f"Dummy test successful! Output shapes: {[o.shape for o in dummy_outputs]}")
        except Exception as e:
            print(f"Error with dummy test: {e}")
            return None
    
    # Thực hiện dự đoán thực tế
    print("\nPerforming actual prediction...")
    try:
        results = predict(model, image_path, age, gender, device)
        return "hehe"
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return None


image_path = '/home/jupyter-iec_iot13_toanlm/testmodel/cn1.nii'
age = 79.0
gender = 1.0  

results = predict_alzheimer_disease(
    image_path=image_path,
    age=65.0,
    gender=0.0, 
    model_path="model_weights.pth"
)
print(results)