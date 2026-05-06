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

CLASS_METADATA = {
    'Bacterial Spot': {
        'title': 'Bercak Bakteri',
        'description': 'Analisis neural mendeteksi lesi nekrotik basah dengan halo klorotik pada kutikula daun, sangat konsisten dengan isolat Xanthomonas campestris.',
        'treatments': [
            {'title': "Karantina Spesimen", 'desc': "Pisahkan tanaman yang terinfeksi untuk mencegah penyebaran lateral."},
            {'title': "Bakterisida Tembaga", 'desc': "Semprotkan pada seluruh permukaan daun setiap 10 hari."},
            {'title': "Sterilisasi Alat", 'desc': "Bersihkan gunting pangkas dengan alkohol 70% setelah penggunaan."}
        ]
    },
    'Cerespora': {
        'title': 'Bercak Daun Cercospora',
        'description': 'Terdeteksi bintik-bintik kecil berbentuk bulat dengan pusat berwarna abu-abu, ciri khas infeksi jamur Cercospora capsici.',
        'treatments': [
            {'title': "Kurangi Kelembapan", 'desc': "Pastikan jarak tanam cukup agar sirkulasi udara baik."},
            {'title': "Fungisida", 'desc': "Gunakan fungisida berbahan aktif mankozeb atau klorotalonil."},
            {'title': "Sanitasi Lahan", 'desc': "Bersihkan sisa-sisa tanaman yang terinfeksi dari area penanaman."}
        ]
    },
    'Healthy': {
        'title': 'Sehat',
        'description': 'Jaringan daun tampak sehat tanpa adanya tanda-tanda patogen atau defisiensi nutrisi yang signifikan.',
        'treatments': [
            {'title': "Pemeliharaan Rutin", 'desc': "Lanjutkan penyiraman dan pemupukan terjadwal."},
            {'title': "Pemantauan Berkala", 'desc': "Lakukan inspeksi visual setiap minggu untuk deteksi dini."},
            {'title': "Nutrisi Optimal", 'desc': "Pastikan tanaman mendapatkan asupan N-P-K yang seimbang."}
        ]
    },
    'Leaf_Curl': {
        'title': 'Daun Keriting',
        'description': 'Daun menunjukkan gejala menggulung dan mengecil, kemungkinan disebabkan oleh infeksi virus (Gemini virus) atau serangan kutu daun.',
        'treatments': [
            {'title': "Pengendalian Vektor", 'desc': "Kendalikan kutu kebul atau trips yang menjadi pembawa virus."},
            {'title': "Pencabutan Tanaman", 'desc': "Tanaman yang terinfeksi virus parah sebaiknya dicabut dan dimusnahkan."},
            {'title': "Nutrisi Tambahan", 'desc': "Berikan pupuk daun untuk membantu pemulihan vigor tanaman."}
        ]
    }
}

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

        class_name = CLASS_NAMES[pred.item()]
        metadata = CLASS_METADATA.get(class_name, {
            'title': class_name,
            'description': 'Hasil klasifikasi tidak dikenal.',
            'treatments': []
        })

        result = {
            'class': class_name,
            'confidence': float(conf.item()),
            'status': 'success',
            **metadata
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
