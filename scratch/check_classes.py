from torchvision import datasets
import os

DATASET_PATH = 'data/Pepper_Dataset'
if os.path.exists(DATASET_PATH):
    dataset = datasets.ImageFolder(root=DATASET_PATH)
    print(dataset.class_to_idx)
else:
    print("Dataset path not found")
