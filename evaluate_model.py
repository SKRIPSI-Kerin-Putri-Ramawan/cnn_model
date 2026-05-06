import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from src.data_preparation import prepare_data

def evaluate_and_visualize(model_path='models/best_pepper_model.keras'):
    print(f"Memuat model dari {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # 1. Load Validation Data
    _, val_gen = prepare_data()
    
    # 2. Evaluate
    print("\nMengevaluasi model pada data validasi...")
    loss, acc = model.evaluate(val_gen)
    print(f"Validation Accuracy: {acc*100:.2f}%")
    print(f"Validation Loss: {loss:.4f}")

    # Catatan: Visualisasi History memerlukan objek 'history' dari model.fit()
    # Jika kita memuat model dari file, kita tidak memiliki history kecuali kita menyimpannya secara terpisah.
    # Namun, kita bisa mensimulasikan hasil akhir atau menyarankan pengguna melihat log training.
    
def plot_history(history):
    """
    Fungsi ini digunakan jika dipanggil langsung setelah model.fit()
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.show()

if __name__ == "__main__":
    evaluate_and_visualize()
