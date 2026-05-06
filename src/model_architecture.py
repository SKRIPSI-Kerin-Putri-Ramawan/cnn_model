import torch
import torch.nn as nn
import torch.nn.functional as F

class PepperCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(PepperCNN, self).__init__()
        # Layer Konvolusi 1: (3, 224, 224) -> (32, 111, 111) setelah pool
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Layer Konvolusi 2: (32, 111, 111) -> (64, 54, 54) setelah pool
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        
        # Layer Konvolusi 3: (64, 54, 54) -> (128, 26, 26) setelah pool
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        
        # Classifier
        # Input size: 128 * 26 * 26 = 86528
        self.fc1 = nn.Linear(128 * 26 * 26, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(-1, 128 * 26 * 26) # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        # Note: Softmax biasanya tidak disertakan di sini jika menggunakan CrossEntropyLoss di PyTorch
        return x

def create_model(num_classes=4):
    """
    Fungsi pembantu untuk membuat instance model.
    """
    return PepperCNN(num_classes=num_classes)

if __name__ == "__main__":
    # Verifikasi arsitektur
    model = create_model()
    print("Ringkasan Arsitektur Model PyTorch:")
    print(model)
    
    # Test dengan dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"\nInput Shape: {dummy_input.shape}")
    print(f"Output Shape: {output.shape}")
