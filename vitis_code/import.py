import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pathlib
import os
data_dir = pathlib.Path('/kaggle/input/datasets/radarcommunsignaldata2026train')
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])
train_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
num_classes = len(train_dataset.classes)
print(f"Đã load {len(train_dataset)} ảnh, gồm {num_classes} classes: {train_dataset.classes}")
