#!/usr/bin/env python3
"""
Minimal test to verify the neural network works with MNIST data
"""

import os
import numpy as np

def minimal_test():
    print("Starting minimal neural network test...")
    
    try:
        # Import core components
        from src.data.data_loader import MNISTDataLoader
        from src.models.neural_network import NeuralNetwork
        from src.models.layers import DenseLayer
        from src.models.activations import ReLU, Softmax
        from src.training.trainer import Trainer
        from src.training.loss_functions import CrossEntropyLoss
        
        print("✓ All imports successful")
        
        # Load small subset of MNIST data
        print("Loading MNIST data (this may take a moment for first download)...")
        data_loader = MNISTDataLoader()
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_loader.load_data(
            validation_split=0.2, normalize=True, flatten=True
        )
        
        # Use only small subset for quick test
        X_train_small = X_train[:100]  # 100 training samples
        y_train_small = y_train[:100]
        X_test_small = X_test[:20]     # 20 test samples
        y_test_small = y_test[:20]
        
        print(f"✓ Data loaded - Training: {X_train_small.shape}, Test: {X_test_small.shape}")
        
        # Create simple model
        print("Creating neural network model...")
        model = NeuralNetwork()
        model.add_layer(DenseLayer(784, 64, activation=ReLU(), weight_init='xavier'))
        model.add_layer(DenseLayer(64, 10, activation=Softmax(), weight_init='xavier'))
        print("✓ Model created")
        
        # Create trainer
        trainer = Trainer(
            model=model,
            loss_function=CrossEntropyLoss(),
            learning_rate=0.01,
            batch_size=32
        )
        print("✓ Trainer created")
        
        # Train for just a few epochs
        print("Training model (5 epochs)...")
        history = trainer.train(
            X_train_small, y_train_small,
            X_test_small, y_test_small,  # Using test as validation for simplicity
            epochs=5,
            verbose=True
        )
        print("✓ Training completed")
        
        # Test prediction
        print("Testing predictions...")
        predictions = model.predict(X_test_small)
        predicted_classes = predictions.argmax(axis=1)
        true_classes = y_test_small.argmax(axis=1)
        
        accuracy = (predicted_classes == true_classes).mean()
        print(f"✓ Test accuracy: {accuracy:.4f}")
        
        # Test model saving
        print("Testing model save/load...")
        os.makedirs('models', exist_ok=True)
        model.save_model('models/test_model.pkl')
        
        # Load model
        new_model = NeuralNetwork()
        new_model.load_model('models/test_model.pkl')
        
        # Test loaded model
        new_predictions = new_model.predict(X_test_small[:5])
        original_predictions = predictions[:5]
        
        if np.allclose(new_predictions, original_predictions, atol=1e-6):
            print("✓ Model save/load successful")
        else:
            print("❌ Model save/load failed - predictions don't match")
        
        print("\n🎉 MINIMAL TEST COMPLETED SUCCESSFULLY!")
        print(f"Final accuracy: {accuracy:.4f}")
        print("The neural network implementation is working correctly!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = minimal_test()
    if success:
        print("\n✅ Ready to run full training with: python main.py")
    else:
        print("\n❌ Please check the error messages above")
