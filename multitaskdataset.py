import torch
import numpy as np
import random
class MultitaskAlzheimerDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, augment_metadata=True):
        loaded_data = torch.load(data_path)

        self.images = [item[0] for item in loaded_data]
        self.labels = [item[1] for item in loaded_data]
        
        if augment_metadata:
            self.metadata = generate_synthetic_metadata(
                len(self.labels), 
                self.labels
            )
        else:
            self.metadata = None
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        metadata = self.metadata[idx]
        return {
            'image': image,
            'classification_label': label,  
            'regression_targets': {
                'mmse': metadata['mmse'],
                'age': metadata['age'],
                'gender': metadata['gender']
            }
        }