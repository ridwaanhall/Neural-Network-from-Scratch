#!/usr/bin/env python3
"""
Test script to check model loading functionality
"""

import os
import glob
from src.models.neural_network import NeuralNetwork

def test_model_loading():
    """Test loading the latest saved model"""
    print("Testing model loading...")
    
    # Find the latest model
    model_files = glob.glob("models/mnist_model_*.pkl")
    if not model_files:
        print("No model files found!")
        return False
        
    latest_model = max(model_files, key=os.path.getmtime)
    print(f"Loading: {latest_model}")
    
    try:
        # Create neural network and load model
        model = NeuralNetwork()
        model.load_model(latest_model)
        
        print(f"Model loaded successfully!")
        print(f"Number of layers: {len(model.layers)}")
        
        # Test prediction with dummy data
        import numpy as np
        dummy_input = np.random.random((1, 784))
        prediction = model.predict(dummy_input)
        print(f"Prediction shape: {prediction.shape}")
        print(f"Prediction sum (should be ~1.0): {prediction.sum():.3f}")
        
        return True
        
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_loading()
    if success:
        print("✅ Model loading test PASSED")
    else:
        print("❌ Model loading test FAILED")
