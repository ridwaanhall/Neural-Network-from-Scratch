#!/usr/bin/env python3
"""
Debug test script to isolate the error in test.py
"""

import numpy as np
import os

def main():
    try:
        print("Starting debug test...")
        
        # Import components
        from src.data.data_loader import MNISTDataLoader
        from src.models.neural_network import NeuralNetwork
        from src.utils.metrics import MetricsTracker
        
        print("✓ Imports successful")
        
        # Find latest model
        import glob
        model_files = glob.glob("models/mnist_model_*.pkl")
        if not model_files:
            print("❌ No model files found!")
            return False
            
        latest_model = max(model_files, key=os.path.getmtime)
        print(f"Found model: {latest_model}")
        
        # Load model
        model = NeuralNetwork()
        model.load_model(latest_model)
        print("✓ Model loaded")
        
        # Load test data
        print("Loading test data...")
        data_loader = MNISTDataLoader()
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_loader.load_data(
            validation_split=0.2, normalize=True, flatten=True
        )
        print(f"✓ Test data loaded: {X_test.shape}, {y_test.shape}")
        
        # Use small subset for debugging
        X_test_small = X_test[:100]
        y_test_small = y_test[:100]
        
        print("Making predictions...")
        predictions_proba = model.predict(X_test_small)
        predictions_classes = model.predict_classes(X_test_small)
        
        print(f"✓ Predictions made")
        print(f"  predictions_proba shape: {predictions_proba.shape if predictions_proba is not None else 'None'}")
        print(f"  predictions_classes shape: {predictions_classes.shape if predictions_classes is not None else 'None'}")
        
        # Convert y_test to class indices if needed
        if y_test_small.ndim > 1 and y_test_small.shape[1] > 1:
            true_classes = np.argmax(y_test_small, axis=1)
        else:
            true_classes = y_test_small
            
        print(f"  true_classes shape: {true_classes.shape if true_classes is not None else 'None'}")
        
        # Check for None values
        if predictions_proba is None:
            print("❌ predictions_proba is None!")
            return False
        if predictions_classes is None:
            print("❌ predictions_classes is None!")
            return False
        if true_classes is None:
            print("❌ true_classes is None!")
            return False
            
        print("Testing MetricsTracker...")
        num_classes = 10
        class_names = [str(i) for i in range(num_classes)]
        
        metrics_tracker = MetricsTracker(num_classes=num_classes, class_names=class_names)
        print("✓ MetricsTracker created")
        
        print("Calling update...")
        metrics_tracker.update(true_classes, predictions_classes, predictions_proba)
        print("✓ MetricsTracker update successful")
        
        print("Calling compute_metrics...")
        detailed_metrics = metrics_tracker.compute_metrics()
        print("✓ MetricsTracker compute_metrics successful")
        
        print(f"Detailed metrics: {detailed_metrics}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()
