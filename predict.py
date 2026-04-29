import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Konfigurasi
MODEL_PATH = 'best_pepper_model.keras'
CLASS_NAMES = ['Cerespora', 'Leaf_Curl', 'Bacterial Spot', 'Healthy'] # Sesuaikan dengan urutan folder

def predict_disease(img_path):
    if not os.path.exists(img_path):
        print(f"Error: File {img_path} tidak ditemukan.")
        return

    # 1. Load Model
    print(f"Memuat model {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)

    # 2. Preprocess Image
    print(f"Memproses gambar {img_path}...")
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Batch dimension
    img_array /= 255.0 # Normalisasi

    # 3. Predict
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = 100 * np.max(predictions)

    print("\n--- HASIL PREDIKSI ---")
    print(f"Penyakit Terdeteksi: {predicted_class}")
    print(f"Tingkat Kepercayaan: {confidence:.2f}%")
    print("----------------------")

if __name__ == "__main__":
    # Masukkan path gambar di sini
    # Contoh: py predict.py dataset_test/test_image.jpg
    import sys
    if len(sys.argv) > 1:
        predict_disease(sys.argv[1])
    else:
        print("Gunakan: python predict.py <path_gambar>")
