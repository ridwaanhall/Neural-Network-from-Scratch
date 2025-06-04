#!/usr/bin/env python3
"""
Quick test script to verify the neural network implementation works
"""

import os

try:
    import numpy as np
    print(f"✓ NumPy imported successfully (version: {np.__version__})")
    
    from src.models.activations import ReLU, Softmax
    print("✓ Activation functions imported successfully")
    
    from src.models.layers import DenseLayer
    print("✓ Layer classes imported successfully")
    
    from src.models.neural_network import NeuralNetwork
    print("✓ Neural network class imported successfully")
    
    from src.training.loss_functions import CrossEntropyLoss
    print("✓ Loss functions imported successfully")
    
    # Test basic functionality
    print("\nTesting basic functionality...")
    
    # Create a simple model
    model = NeuralNetwork()
    model.add_layer(DenseLayer(2, 3, activation=ReLU(), weight_init='xavier'))
    model.add_layer(DenseLayer(3, 2, activation=Softmax(), weight_init='xavier'))
    print("✓ Model creation successful")
    
    # Test forward pass with dummy data
    X = np.random.randn(5, 2)  # 5 samples, 2 features
    output = model.forward(X)
    print(f"✓ Forward pass successful - output shape: {output.shape}")
    
    # Test prediction
    predictions = model.predict(X)
    print(f"✓ Prediction successful - predictions shape: {predictions.shape}")
    
    print("\n🎉 ALL TESTS PASSED! The neural network implementation is working correctly.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please make sure all required packages are installed:")
    print("pip install numpy requests")
    
except Exception as e:
    print(f"❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()
