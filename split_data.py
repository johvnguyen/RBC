import os
import random
from pathlib import Path
import torch

def split_dataset(directory, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1"

    random.seed(seed)
    directory = Path(directory)

    all_files = sorted([f for f in directory.glob("*.pth") if f.is_file()])
    random.shuffle(all_files)

    n = len(all_files)
    train_end = int(train_ratio * n)
    val_end = train_end + int(val_ratio * n)

    train_files = all_files[:train_end]
    val_files = all_files[train_end:val_end]
    test_files = all_files[val_end:]

    return train_files, val_files, test_files

def merge_pth_files(file_list, output_path):
    merged_data = []

    for file in file_list:
        data = torch.load(file)
        
        # You may need to adjust this depending on what's inside the .pth files
        if isinstance(data, list):
            merged_data.extend(data)
        else:
            merged_data.append(data)
    
    torch.save(merged_data, output_path)

    return

# Example usage
train, val, test = split_dataset("data/datasets/")

print("Train:", len(train))
print("Val:", len(val))
print("Test:", len(test))

merge_pth_files(train, "data/datasets/splits/train.pth")
merge_pth_files(val, "data/datasets/splits/val.pth")
merge_pth_files(test, "data/datasets/splits/test.pth")
