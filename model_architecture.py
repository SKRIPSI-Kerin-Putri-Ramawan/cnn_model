import tensorflow as tf
from tensorflow.keras import layers, models

def create_model(input_shape=(224, 224, 3), num_classes=4):
    """
    Membangun arsitektur CNN untuk deteksi penyakit daun paprika.
    """
    model = models.Sequential([
        # --- Feature Extractor ---
        # Layer Konvolusi 1: Mendeteksi fitur dasar (tepi, tekstur)
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        # Layer Konvolusi 2: Mendeteksi fitur yang lebih kompleks
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Layer Konvolusi 3: Fitur tingkat tinggi
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # --- Transition ---
        # Flatten: Mengubah tensor 3D menjadi vector 1D
        layers.Flatten(),
        
        # --- Classifier ---
        # Dense Layer: Penalaran tingkat tinggi
        layers.Dense(128, activation='relu'),
        
        # Dropout: Mencegah overfitting (Opsional tapi disarankan)
        layers.Dropout(0.5),
        
        # Output Layer: Klasifikasi akhir (Softmax untuk multi-kelas)
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

if __name__ == "__main__":
    # Membuat model
    pepper_model = create_model()
    
    # Menampilkan ringkasan arsitektur (Phase 2 Finalization)
    print("Ringkasan Arsitektur Model CNN:")
    pepper_model.summary()
    
    # Verifikasi input dan output shape
    print(f"\nInput Shape: {pepper_model.input_shape}")
    print(f"Output Shape: {pepper_model.output_shape}")
