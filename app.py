from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app) # Mengizinkan request dari domain berbeda (penting untuk web/N8N)

# Konfigurasi
MODEL_PATH = 'best_pepper_model.keras'
CLASS_NAMES = ['Cerespora', 'Leaf_Curl', 'Bacterial Spot', 'Healthy']
UPLOAD_FOLDER = 'temp_uploads'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load Model secara global agar tidak lambat saat request
print(f"Memuat model {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)

def process_and_predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = float(np.max(predictions)) * 100
    
    return predicted_class, confidence

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file yang diunggah'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nama file kosong'}), 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        try:
            label, confidence = process_and_predict(file_path)
            
            # Hapus file setelah diproses agar tidak menumpuk
            os.remove(file_path)
            
            return jsonify({
                'status': 'success',
                'prediction': label,
                'confidence': f"{confidence:.2f}%",
                'message': f"Daun terdeteksi memiliki {label}"
            })
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({'status': 'API Aktif', 'model': 'Pepper Disease Detector'})

if __name__ == '__main__':
    # Jalankan pada port 5000
    app.run(host='0.0.0.0', port=5000, debug=False)
