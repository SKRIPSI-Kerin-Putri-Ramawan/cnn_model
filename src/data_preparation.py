import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os
import matplotlib.pyplot as plt

# Konfigurasi Path
DATASET_PATH = 'data/Pepper_Dataset'
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

def prepare_data():
    print("Mengecek dataset...")
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Folder {DATASET_PATH} tidak ditemukan!")
        return None, None

    # 1. Transformasi Data (Augmentasi & Normalisasi)
    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Skala 0-1
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Standar ImageNet
    ])

    val_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. Load Dataset
    full_dataset = datasets.ImageFolder(root=DATASET_PATH)
    
    # Split 80% Train, 20% Val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Apply transforms secara terpisah
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    # 3. Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"\nKelas ditemukan: {full_dataset.classes}")
    print(f"Jumlah sampel training: {len(train_dataset)}")
    print(f"Jumlah sampel validasi: {len(val_dataset)}")

    return train_loader, val_loader

if __name__ == "__main__":
    train_loader, val_loader = prepare_data()
    if train_loader:
        # Cek satu batch
        images, labels = next(iter(train_loader))
        print(f"Batch images shape: {images.shape}")
        print(f"Batch labels shape: {labels.shape}")
