#!/usr/bin/env python3
"""
Debug script to examine saved model structure
"""

import pickle
import glob
import os

def debug_model_structure():
    """Debug the structure of a saved model"""
    # Find the latest model
    model_files = glob.glob("models/mnist_model_*.pkl")
    if not model_files:
        print("No model files found!")
        return
        
    latest_model = max(model_files, key=os.path.getmtime)
    print(f"Examining: {latest_model}")
    
    with open(latest_model, 'rb') as f:
        data = pickle.load(f)
    
    print("\n=== MODEL DATA STRUCTURE ===")
    print("Keys:", list(data.keys()))
    print("Architecture:", data['architecture'])
    print("Number of parameters:", len(data['parameters']))
    
    print("\n=== PARAMETER DETAILS ===")
    for i, param in enumerate(data['parameters']):
        if param is None:
            print(f"Parameter {i}: None (Dropout layer)")
        else:
            print(f"Parameter {i}: Dense layer")
            if 'weights' in param:
                print(f"  Weights shape: {param['weights'].shape}")
            if 'bias' in param:
                print(f"  Bias shape: {param['bias'].shape}")
    
    print("\n=== EXPECTED NETWORK STRUCTURE ===")
    print("Based on main.py, the expected structure for 'default' architecture is:")
    print("1. DenseLayer(784, 256) + ReLU")
    print("2. DropoutLayer(0.2)")  
    print("3. DenseLayer(256, 128) + ReLU")
    print("4. DropoutLayer(0.2)")
    print("5. DenseLayer(128, 10) + Softmax")
    
    print("\nThis should correspond to parameters:")
    print("Parameter 0: Dense layer weights (784, 256)")
    print("Parameter 1: None (Dropout)")
    print("Parameter 2: Dense layer weights (256, 128)")
    print("Parameter 3: None (Dropout)")
    print("Parameter 4: Dense layer weights (128, 10)")

if __name__ == "__main__":
    debug_model_structure()
