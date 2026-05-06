from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torchvision import transforms
from PIL import Image
import os
from werkzeug.utils import secure_filename
from src.model_architecture import create_model

app = Flask(__name__)
CORS(app)

# Konfigurasi
MODEL_PATH = 'models/best_pepper_model.pth'
CLASS_NAMES = ['Bacterial Spot', 'Cerespora', 'Healthy', 'Leaf_Curl']
UPLOAD_FOLDER = 'uploads'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Global Model Variable
model = None

def get_model():
    global model
    if model is None:
        model = create_model(num_classes=4)
        if os.path.exists(MODEL_PATH):
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            print(f"Model dimuat dari {MODEL_PATH} ({DEVICE})")
        model.to(DEVICE)
        model.eval()
    return model

# Transformasi Gambar
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file dikirim'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nama file kosong'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        # Load model & preprocess
        net = get_model()
        img = Image.open(filepath).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)

        # Inference
        with torch.no_grad():
            outputs = net(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

        result = {
            'class': CLASS_NAMES[pred.item()],
            'confidence': float(conf.item()),
            'status': 'success'
        }
        
    except Exception as e:
        result = {'error': str(e), 'status': 'failed'}
    
    return jsonify(result)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'active',
        'device': str(DEVICE),
        'cuda_available': torch.cuda.is_available()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
