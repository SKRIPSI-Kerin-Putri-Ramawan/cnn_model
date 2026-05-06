import torch
import torch.nn as nn
import torch.optim as optim
from src.data_preparation import prepare_data
from src.model_architecture import create_model
import os

def train_pepper_model(epochs=30):
    # 0. Set Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menggunakan perangkat: {device}")

    # 1. Load Data
    train_loader, val_loader = prepare_data()
    if train_loader is None:
        return

    # 2. Create Model & Move to Device
    model = create_model(num_classes=4).to(device)
    
    # 3. Define Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    best_val_acc = 0.0
    patience = 5
    counter = 0

    print("\nStarting Training Phase...")
    
    for epoch in range(epochs):
        # --- Training Loop ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        train_acc = 100. * correct / total
        
        # --- Validation Loop ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}% | Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")

        # Save Best Model (Checkpoint)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'models/best_pepper_model.pth')
            print(f"--> Model terbaik disimpan dengan Akurasi: {val_acc:.2f}%")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping dipicu.")
                break

    # Save Final Model
    torch.save(model.state_dict(), 'models/pepper_disease_model_final.pth')
    print("\nTraining selesai! Model disimpan di folder 'models/'")

if __name__ == "__main__":
    if not os.path.exists('models'):
        os.makedirs('models')
    train_pepper_model(epochs=30)
