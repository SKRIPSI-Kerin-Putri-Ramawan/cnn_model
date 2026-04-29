import tensorflow as tf
from data_preparation import prepare_data
from model_architecture import create_model
import os

def train_pepper_model(epochs=10):
    # 1. Load Data
    train_gen, val_gen = prepare_data()
    if train_gen is None:
        print("Data tidak siap. Membatalkan training.")
        return
    
    # 2. Create Model
    model = create_model(input_shape=(224, 224, 3), num_classes=4)
    
    # 3. Compile Model (Phase 3: Compile the Model)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nStarting Training Phase...")
    
    # 4. Callbacks (Optimization)
    # EarlyStopping: Berhenti jika val_loss tidak membaik
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=3, 
        restore_best_weights=True
    )
    
    # ModelCheckpoint: Simpan model terbaik selama training
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'best_pepper_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )

    # 5. Training (Phase 3: Train/Fit the Model)
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=[early_stop, checkpoint]
    )
    
    # 6. Save Final Model
    model.save('pepper_disease_model_final.keras')
    print("\nTraining selesai! Model disimpan sebagai 'pepper_disease_model_final.keras'")
    
    return history

if __name__ == "__main__":
    # Menjalankan training (Default 10 epoch untuk demo)
    history = train_pepper_model(epochs=10)
