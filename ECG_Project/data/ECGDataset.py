import torch
from torch.utils.data import Dataset
import numpy as np

class ECGDataset(Dataset):
    def __init__(self, path_noisy, path_clean):
        self.X_noisy = np.load(path_noisy).astype(np.float32)
        self.X_clean = np.load(path_clean).astype(np.float32)

    def __len__(self):
        return len(self.X_noisy)

    def __getitem__(self, idx):
        x = torch.tensor(self.X_noisy[idx].T)  # [12, 5000]
        y = torch.tensor(self.X_clean[idx].T)
        return x, y
