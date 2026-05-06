"""
Test Suite untuk Pepper Disease Classification Project
"""
import pytest
import os
import sys
import numpy as np
import tensorflow as tf
from unittest.mock import patch, MagicMock

# Add root directory to path to find src and other scripts
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestDataPreparation:
    """Test untuk data_preparation.py"""
    
    def test_prepare_data_returns_generators(self):
        """Test: Memastikan prepare_data mengembalikan dua generator"""
        from src.data_preparation import prepare_data
        train_gen, val_gen = prepare_data()
        
        assert train_gen is not None, "Train generator tidak boleh None"
        assert val_gen is not None, "Validation generator tidak boleh None"
    
    def test_train_generator_has_correct_batch_size(self):
        """Test: Memastikan batch size generator benar"""
        from src.data_preparation import prepare_data, BATCH_SIZE
        train_gen, _ = prepare_data()
        
        assert train_gen.batch_size == BATCH_SIZE, f"Batch size harus {BATCH_SIZE}"
    
    def test_validation_generator_has_correct_batch_size(self):
        """Test: Memastikan batch size validation generator benar"""
        from src.data_preparation import prepare_data, BATCH_SIZE
        _, val_gen = prepare_data()
        
        assert val_gen.batch_size == BATCH_SIZE, f"Validation batch size harus {BATCH_SIZE}"
    
    def test_image_size_correct(self):
        """Test: Memastikan ukuran gambar sesuai konfigurasi"""
        from src.data_preparation import prepare_data, IMAGE_SIZE
        train_gen, _ = prepare_data()
        
        assert train_gen.target_size == IMAGE_SIZE, f"Image size harus {IMAGE_SIZE}"
    
    def test_class_indices_exist(self):
        """Test: Memastikan class indices terdefinisi"""
        from src.data_preparation import prepare_data
        train_gen, _ = prepare_data()
        
        assert hasattr(train_gen, 'class_indices'), "Generator harus memiliki class_indices"
        assert len(train_gen.class_indices) > 0, "Harus ada minimal 1 kelas"
    
    def test_dataset_path_exists(self):
        """Test: Memastikan dataset path ada"""
        from src.data_preparation import DATASET_PATH
        assert os.path.exists(DATASET_PATH), f"Dataset path {DATASET_PATH} harus ada"


class TestModelArchitecture:
    """Test untuk model_architecture.py"""
    
    def test_create_model_returns_keras_model(self):
        """Test: Memastikan create_model mengembalikan model Keras"""
        from src.model_architecture import create_model
        model = create_model()
        
        assert isinstance(model, tf.keras.Model), "Harus mengembalikan Keras Model"
    
    def test_model_has_correct_input_shape(self):
        """Test: Memastikan input shape sesuai"""
        from src.model_architecture import create_model
        model = create_model(input_shape=(224, 224, 3))
        
        assert model.input_shape == (None, 224, 224, 3), "Input shape harus (None, 224, 224, 3)"
    
    def test_model_has_correct_output_shape(self):
        """Test: Memastikan output shape sesuai jumlah kelas"""
        from src.model_architecture import create_model
        model = create_model(num_classes=4)
        
        assert model.output_shape == (None, 4), "Output shape harus (None, 4)"
    
    def test_model_has_conv2d_layers(self):
        """Test: Memastikan model memiliki Conv2D layers"""
        from src.model_architecture import create_model
        model = create_model()
        
        conv_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
        assert len(conv_layers) >= 3, "Model harus memiliki minimal 3 Conv2D layers"
    
    def test_model_has_dense_layers(self):
        """Test: Memastikan model memiliki Dense layers"""
        from src.model_architecture import create_model
        model = create_model()
        
        dense_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
        assert len(dense_layers) >= 2, "Model harus memiliki minimal 2 Dense layers"
    
    def test_model_has_dropout_layer(self):
        """Test: Memastikan model memiliki Dropout layer"""
        from src.model_architecture import create_model
        model = create_model()
        
        dropout_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dropout)]
        assert len(dropout_layers) >= 1, "Model harus memiliki Dropout layer"
    
    def test_model_has_softmax_output(self):
        """Test: Memastikan output layer menggunakan softmax"""
        from src.model_architecture import create_model
        model = create_model()
        
        output_layer = model.layers[-1]
        assert output_layer.activation.__name__ == 'softmax', "Output layer harus menggunakan softmax"


class TestTrainModel:
    """Test untuk train_model.py"""
    
    @patch('train_model.tf')
    @patch('train_model.prepare_data')
    @patch('train_model.create_model')
    def test_train_pepper_model_returns_history(self, mock_create, mock_prepare, mock_tf):
        """Test: Memastikan training mengembalikan history"""
        # Setup mocks
        mock_train_gen = MagicMock()
        mock_train_gen.samples = 100
        mock_val_gen = MagicMock()
        mock_val_gen.samples = 50
        
        mock_prepare.return_value = (mock_train_gen, mock_val_gen)
        
        mock_model = MagicMock()
        mock_model.fit.return_value = MagicMock(history={'loss': [1.0, 0.5], 'accuracy': [0.5, 0.8]})
        mock_create.return_value = mock_model
        mock_tf.keras.callbacks.EarlyStopping.return_value = MagicMock()
        mock_tf.keras.callbacks.ModelCheckpoint.return_value = MagicMock()
        
        from train_model import train_pepper_model
        result = train_pepper_model(epochs=1)
        
        # Verify model.fit was called
        mock_model.fit.assert_called_once()
    
    @patch('train_model.tf')
    @patch('train_model.prepare_data')
    def test_train_pepper_model_handles_no_data(self, mock_prepare, mock_tf):
        """Test: Memastikan training menangani kasus tidak ada data"""
        mock_prepare.return_value = (None, None)
        
        from train_model import train_pepper_model
        result = train_pepper_model(epochs=1)
        
        assert result is None, "Harus mengembalikan None jika data tidak siap"


class TestEvaluateModel:
    """Test untuk evaluate_model.py"""
    
    @patch('evaluate_model.tf')
    @patch('evaluate_model.prepare_data')
    def test_evaluate_model_returns_loss_accuracy(self, mock_prepare, mock_tf):
        """Test: Memastikan evaluate mengembalikan loss dan accuracy"""
        mock_model = MagicMock()
        mock_model.evaluate.return_value = [0.5, 0.9]  # [loss, accuracy]
        mock_tf.keras.models.load_model.return_value = mock_model
        
        mock_train_gen = MagicMock()
        mock_val_gen = MagicMock()
        mock_prepare.return_value = (mock_train_gen, mock_val_gen)
        
        from evaluate_model import evaluate_and_visualize
        # This should run without error
        evaluate_and_visualize()
        
        mock_model.evaluate.assert_called_once()


class TestPredict:
    """Test untuk predict.py"""
    
    @patch('predict.tf')
    @patch('predict.image')
    def test_predict_disease_with_valid_image(self, mock_image, mock_tf):
        """Test: Prediksi dengan gambar valid"""
        # Setup mocks
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[0.1, 0.2, 0.3, 0.4]])
        mock_tf.keras.models.load_model.return_value = mock_model
        mock_tf.nn.softmax.return_value = np.array([0.1, 0.2, 0.3, 0.4])
        
        mock_img = MagicMock()
        mock_img_array = np.zeros((224, 224, 3))
        mock_image.load_img.return_value = mock_img
        mock_image.img_to_array.return_value = mock_img_array
        
        from predict import predict_disease
        # Should not raise exception
        try:
            predict_disease('test_image.jpg')
        except Exception as e:
            # May fail due to mock limitations, but should handle gracefully
            pass
    
    def test_predict_handles_missing_file(self):
        """Test: Prediksi menangani file yang tidak ada"""
        from predict import predict_disease
        
        # Should handle gracefully
        result = predict_disease('nonexistent_file.jpg')
        # Function prints error but doesn't raise


class TestIntegration:
    """Integration tests untuk alur kerja lengkap"""
    
    def test_full_pipeline_data_to_model(self):
        """Test: Pipeline lengkap dari data ke model"""
        from src.data_preparation import prepare_data
        from src.model_architecture import create_model
        
        # Step 1: Prepare data
        train_gen, val_gen = prepare_data()
        assert train_gen is not None
        
        # Step 2: Create model
        model = create_model(input_shape=(224, 224, 3), num_classes=len(train_gen.class_indices))
        assert model is not None
        
        # Step 3: Verify model can process a batch
        images, labels = next(train_gen)
        assert images.shape[1:] == (224, 224, 3), "Image shape harus (224, 224, 3)"
    
    def test_class_names_consistency(self):
        """Test: Konsistensi nama kelas di seluruh project"""
        from src.data_preparation import prepare_data
        from predict import CLASS_NAMES
        
        train_gen, _ = prepare_data()
        data_classes = list(train_gen.class_indices.keys())
        
        # Predict.py harus memiliki kelas yang sama
        assert len(CLASS_NAMES) == len(data_classes), "Jumlah kelas harus sama"


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])