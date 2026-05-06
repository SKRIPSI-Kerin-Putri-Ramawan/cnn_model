import torch
from torchvision import transforms
from PIL import Image
from src.model_architecture import create_model
import os

def predict_disease(image_path):
    # 1. Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Load Model
    model = create_model(num_classes=4)
    model.load_state_dict(torch.load('models/best_pepper_model.pth', map_location=device))
    model.to(device)
    model.eval()
    
    # 3. Preprocess Image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    # 4. Inference
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    # 5. Output
    class_names = ['Cerespora', 'Leaf_Curl', 'Bacterial Spot', 'Healthy']
    result = class_names[predicted.item()]
    
    return result, confidence.item()

if __name__ == "__main__":
    test_img = 'data/Pepper_Dataset/Healthy/healthy_1.jpg' # Contoh path
    if os.path.exists(test_img):
        label, conf = predict_disease(test_img)
        print(f"Hasil Prediksi: {label} (Confidence: {conf*100:.2f}%)")
    else:
        print("File gambar contoh tidak ditemukan.")
