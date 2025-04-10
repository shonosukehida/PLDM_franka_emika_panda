import torch  
from torch.utils.data import Dataset 
import numpy as np 
import os 
import pickle

class FrankaDataset(Dataset):
    def __init__(self, data_path, image_path, transform=None):
        #load data.p 
        with open(data_path, 'rb') as f:
            self.episodes = pickle.load(f)
        
        #load images.npy 
        self.images = np.load(image_path)
        self.transform = transform 
        
        self.total_length = sum(len(ep['observations'])-1 for ep in self.episodes)
        self.idx_map
        