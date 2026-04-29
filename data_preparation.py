import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt

# Konfigurasi Path
DATASET_PATH = 'Pepper_Dataset'
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

def prepare_data():
    print("Mengecek dataset...")
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Folder {DATASET_PATH} tidak ditemukan!")
        return None, None

    # 1. Image Data Generator dengan Augmentasi & Normalisasi
    # Rescale 1./255 adalah bagian dari Normalisasi (0-1)
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2 # Split 20% untuk validasi
    )

    # 2. Generator untuk Data Training
    train_generator = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    # 3. Generator untuk Data Validasi
    validation_generator = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    print(f"\nKelas ditemukan: {list(train_generator.class_indices.keys())}")
    print(f"Jumlah sampel training: {train_generator.samples}")
    print(f"Jumlah sampel validasi: {validation_generator.samples}")

    return train_generator, validation_generator

def visualize_samples(generator):
    # Mengambil satu batch data
    images, labels = next(generator)
    
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(f"Kelas: {list(generator.class_indices.keys())[labels[i].argmax()]}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    print("Visualisasi sampel selesai.")

if __name__ == "__main__":
    train_gen, val_gen = prepare_data()
    if train_gen:
        # Menampilkan beberapa sampel untuk verifikasi
        # visualize_samples(train_gen)
        pass
