import torch
from torch.utils.data import Dataset

class RBCDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = [torch.tensor(seq.clone().detach(), dtype = torch.float32) for seq in sequences]
        self.labels = [torch.tensor(label.clone().detach(), dtype = torch.float32) for label in labels]

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]