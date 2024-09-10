from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
import urllib.request


class SlidingWindowDataset(Dataset):
    def __init__(self, target, window_size, step_size, shift_size, past_cov=None, future_cov=None):
        if past_cov is not None:
            assert len(target) == len(past_cov), "Target and covariates have different sizes"
        if future_cov is not None:
            assert len(target) == len(future_cov), "Target and covariates have different sizes"
        self.target = target
        self.past_cov = past_cov
        self.future_cov = future_cov
        self.window_size = window_size
        self.step_size = step_size
        self.shift_size = shift_size
        self.dataset_size = (len(self.target) - self.window_size - self.shift_size + 1) // self.step_size
        assert self.dataset_size > 0, "Dataset is too short for given window and shift sizes"
    
    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, idx):
        past_range = np.arange(idx*self.step_size, idx*self.step_size + self.window_size)
        future_range = past_range + self.shift_size
        
        past_target = self.target[past_range]
        future_target = self.target[future_range]
        past_cov = self.past_cov[past_range] if self.past_cov is not None else [[0]]
        future_cov = self.future_cov[past_range] if self.future_cov is not None else [[0]]
        
        return (
            torch.as_tensor(past_cov, dtype=torch.float32), 
            torch.as_tensor(past_target, dtype=torch.float32), 
            torch.as_tensor(future_cov, dtype=torch.float32), 
            torch.as_tensor(future_target, dtype=torch.float32)
        )


def electricity_dataset():
    urllib.request.urlretrieve(
        'https://raw.githubusercontent.com/vpozdnyakov/deeptcn/main/datasets/electricity/electricity_train.csv', 
        'electricity_train.csv'
    )
    urllib.request.urlretrieve(
        'https://raw.githubusercontent.com/vpozdnyakov/deeptcn/main/datasets/electricity/electricity_test.csv', 
        'electricity_test.csv'
    )
    electricity_train = pd.read_csv('electricity_train.csv', index_col=0)
    electricity_train.index = pd.to_datetime(electricity_train.index)
    electricity_test = pd.read_csv('electricity_test.csv', index_col=0)
    electricity_test.index = pd.to_datetime(electricity_test.index)
    return electricity_train, electricity_test