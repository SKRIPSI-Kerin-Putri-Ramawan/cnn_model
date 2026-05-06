import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from src.data_preparation import prepare_data
from src.model_architecture import create_model
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def evaluate_and_visualize(model_path='models/best_pepper_model.pth'):
    # 0. Set Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menggunakan perangkat: {device}")

    # 1. Load Data
    _, val_loader = prepare_data()
    if val_loader is None:
        return

    # 2. Load Model
    print(f"Memuat model dari {model_path}...")
    model = create_model(num_classes=4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 3. Evaluate
    print("\nMengevaluasi model pada data validasi...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 4. Metrics
    class_names = ['Bacterial Spot', 'Cerespora', 'Healthy', 'Leaf_Curl']
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # 5. Confusion Matrix Visualization
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Prediksi')
    plt.ylabel('Kenyataan (True)')
    plt.title('Confusion Matrix - Pepper Disease Detection')
    plt.show()

if __name__ == "__main__":
    # Pastikan model .pth ada sebelum evaluasi
    evaluate_and_visualize()
