#!/usr/bin/env python3
"""
Neural Network from Scratch - Final Demonstration
=================================================

This script demonstrates the key capabilities of our neural network implementation
and provides a comprehensive overview of what we've built.

Key Features Demonstrated:
1. Multiple model architectures
2. Training with different configurations  
3. Comprehensive evaluation
4. Model persistence
5. Visualization capabilities
"""

import os
import numpy as np
from datetime import datetime

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def print_section(title):
    """Print a formatted section"""
    print(f"\n📋 {title}")
    print("-" * (len(title) + 4))

def demonstrate_components():
    """Demonstrate individual components"""
    print_header("NEURAL NETWORK FROM SCRATCH - COMPONENT DEMONSTRATION")
    try:
        # Import all components
        from src.models.activations import ReLU, Sigmoid, Tanh, Softmax, LeakyReLU
        from src.models.layers import DenseLayer, DropoutLayer
        from src.models.neural_network import NeuralNetwork
        from src.training.loss_functions import CrossEntropyLoss, MeanSquaredError
        from src.training.trainer import Trainer
        from src.utils.metrics import calculate_metrics
        
        print_section("1. Activation Functions")
        activations = [ReLU(), Sigmoid(), Tanh(), Softmax(), LeakyReLU(0.1)]
        test_input = np.array([[-2, -1, 0, 1, 2]])
        
        for activation in activations:
            output = activation.forward(test_input)
            print(f"  {activation.__class__.__name__:12}: {output.flatten()}")
        
        print_section("2. Layer Types")
        # Dense layer
        dense = DenseLayer(3, 2, activation=ReLU(), weight_init='xavier')
        test_data = np.random.randn(5, 3)
        dense_output = dense.forward(test_data)
        print(f"  Dense Layer:   Input {test_data.shape} → Output {dense_output.shape}")
        
        # Dropout layer
        dropout = DropoutLayer(0.5)
        dropout_output = dropout.forward(test_data, training=True)
        print(f"  Dropout Layer: Input {test_data.shape} → Output {dropout_output.shape}")
        
        print_section("3. Loss Functions")
        y_true = np.array([[0, 1, 0], [1, 0, 0]])  # One-hot encoded
        y_pred = np.array([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]])  # Predictions
        
        ce_loss = CrossEntropyLoss()
        mse_loss = MeanSquaredError()
        
        print(f"  Cross-Entropy Loss: {ce_loss.forward(y_true, y_pred):.4f}")
        print(f"  Mean Squared Error: {mse_loss.forward(y_true, y_pred):.4f}")
        
        print_section("4. Model Architecture Flexibility")
        
        # Simple model
        simple_model = NeuralNetwork()
        simple_model.add_layer(DenseLayer(784, 128, activation=ReLU()))
        simple_model.add_layer(DenseLayer(128, 10, activation=Softmax()))
        print(f"  Simple Model:  {len(simple_model.layers)} layers")
        
        # Complex model
        complex_model = NeuralNetwork()
        complex_model.add_layer(DenseLayer(784, 512, activation=ReLU()))
        complex_model.add_layer(DropoutLayer(0.3))
        complex_model.add_layer(DenseLayer(512, 256, activation=ReLU()))
        complex_model.add_layer(DropoutLayer(0.3))
        complex_model.add_layer(DenseLayer(256, 128, activation=ReLU()))
        complex_model.add_layer(DropoutLayer(0.2))
        complex_model.add_layer(DenseLayer(128, 10, activation=Softmax()))
        print(f"  Complex Model: {len(complex_model.layers)} layers")
        
        print_section("5. Training Features")
        trainer_features = [
            "✓ SGD with Momentum",
            "✓ Learning Rate Scheduling (step, exponential, plateau)",
            "✓ Early Stopping with Patience",
            "✓ Batch Training",
            "✓ Progress Tracking",
            "✓ History Logging"
        ]
        
        for feature in trainer_features:
            print(f"  {feature}")
        
        print_section("6. Evaluation Metrics")
        # Simulate some predictions for metrics demonstration
        y_true_classes = np.array([0, 1, 2, 1, 0, 2, 1, 0])
        y_pred_classes = np.array([0, 1, 1, 1, 0, 2, 2, 0])
        
        metrics = calculate_metrics(y_true_classes, y_pred_classes, 
                                  class_names=['0', '1', '2'])
        
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        
        print_section("7. Weight Initialization Methods")
        init_methods = ['xavier', 'he', 'random']
        for method in init_methods:
            layer = DenseLayer(100, 50, weight_init=method)
            weight_std = np.std(layer.weights)
            print(f"  {method.capitalize():8}: Weight std = {weight_std:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during component demonstration: {e}")
        import traceback
        traceback.print_exc()
        return False

def demonstrate_mini_training():
    """Demonstrate training on a small dataset"""
    print_header("MINI TRAINING DEMONSTRATION")
    try:
        from src.data.data_loader import MNISTDataLoader
        from src.models.neural_network import NeuralNetwork
        from src.models.layers import DenseLayer
        from src.models.activations import ReLU, Softmax
        from src.training.trainer import Trainer
        from src.training.loss_functions import CrossEntropyLoss
        
        print_section("Loading MNIST Data (Small Subset)")
        data_loader = MNISTDataLoader()
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_loader.load_data(
            validation_split=0.2, normalize=True, flatten=True
        )
        
        # Use very small subset for quick demo
        X_train_mini = X_train[:50]
        y_train_mini = y_train[:50]
        X_test_mini = X_test[:20]
        y_test_mini = y_test[:20]
        
        print(f"  Training samples: {X_train_mini.shape[0]}")
        print(f"  Test samples:     {X_test_mini.shape[0]}")
        print(f"  Input features:   {X_train_mini.shape[1]}")
        print(f"  Output classes:   {y_train_mini.shape[1]}")
        
        print_section("Creating Model")
        model = NeuralNetwork()
        model.add_layer(DenseLayer(784, 32, activation=ReLU(), weight_init='xavier'))
        model.add_layer(DenseLayer(32, 10, activation=Softmax(), weight_init='xavier'))
        print(f"  Model architecture: 784 → 32 → 10")
        print(f"  Total parameters: {sum(layer.weights.size + layer.biases.size for layer in model.layers if hasattr(layer, 'weights'))}")
        
        print_section("Training (3 epochs)")
        trainer = Trainer(
            model=model,
            loss_function=CrossEntropyLoss(),
            learning_rate=0.01,
            batch_size=16,
            momentum=0.9
        )
        
        # Quick training
        history = trainer.train(
            X_train_mini, y_train_mini,
            X_test_mini, y_test_mini,
            epochs=3,
            verbose=True
        )
        
        print_section("Results")
        predictions = model.predict(X_test_mini)
        predicted_classes = predictions.argmax(axis=1)
        true_classes = y_test_mini.argmax(axis=1)
        accuracy = (predicted_classes == true_classes).mean()
        
        print(f"  Final accuracy: {accuracy:.4f}")
        print(f"  Training completed successfully!")
        
        # Show a few predictions
        print(f"\n  Sample predictions:")
        for i in range(min(5, len(predicted_classes))):
            confidence = predictions[i].max()
            print(f"    Sample {i+1}: True={true_classes[i]}, Predicted={predicted_classes[i]}, Confidence={confidence:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during mini training: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_project_summary():
    """Show comprehensive project summary"""
    print_header("PROJECT SUMMARY")
    
    print_section("🎯 What We Built")
    features = [
        "Complete neural network from scratch using only NumPy",
        "Multiple activation functions (ReLU, Sigmoid, Tanh, Softmax, LeakyReLU)",
        "Flexible layer system (Dense, Dropout) with easy extensibility",
        "Advanced training features (momentum, learning rate scheduling, early stopping)",
        "Multiple loss functions (Cross-entropy, MSE, Binary cross-entropy, Huber)",
        "Comprehensive evaluation metrics and visualization",
        "Professional data pipeline with automatic MNIST downloading",
        "Model persistence (save/load) functionality",
        "Clean OOP architecture with separation of concerns",
        "Command-line interface for easy usage",
        "Extensive logging and error handling",
        "Production-ready code with proper documentation"
    ]
    
    for i, feature in enumerate(features, 1):
        print(f"  {i:2d}. {feature}")
    
    print_section("🏗️ Architecture")
    components = {
        "Data Pipeline": "Automatic MNIST download, preprocessing, validation split",
        "Model Core": "NeuralNetwork class with flexible layer stacking",
        "Layers": "DenseLayer (fully connected), DropoutLayer (regularization)",
        "Activations": "ReLU, Sigmoid, Tanh, Softmax, LeakyReLU with derivatives",
        "Loss Functions": "Cross-entropy, MSE, Binary cross-entropy, Huber loss",
        "Training": "Trainer class with SGD, momentum, learning rate scheduling",
        "Evaluation": "Comprehensive metrics, confusion matrix, visualizations",
        "Utilities": "Plotting, metrics calculation, model serialization"
    }
    
    for component, description in components.items():
        print(f"  {component:15}: {description}")
    
    print_section("📊 Performance Expectations")
    performance = [
        "Simple architecture (2 layers): ~85-90% accuracy",
        "Default architecture (3 layers + dropout): ~92-95% accuracy", 
        "Deep architecture (4 layers + dropout): ~95-97% accuracy",
        "Training time: 1-10 minutes depending on architecture and epochs",
        "Memory usage: <1GB RAM for full MNIST dataset"
    ]
    
    for perf in performance:
        print(f"  • {perf}")
    
    print_section("🚀 How to Use")
    usage_examples = [
        "python main.py                    # Full training pipeline",
        "python main.py --quick_test       # Quick test with small dataset",
        "python main.py --architecture deep --epochs 100  # Deep model training",
        "python train.py                   # Standalone training",
        "python test.py                    # Model evaluation",
        "python minimal_test.py            # Basic functionality test"
    ]
    
    for example in usage_examples:
        print(f"  {example}")
    
    print_section("🎓 Learning Value")
    learning_points = [
        "Mathematical foundations of neural networks",
        "Backpropagation algorithm implementation",
        "Gradient descent optimization techniques",
        "Weight initialization strategies",
        "Regularization methods (dropout)",
        "Training best practices and monitoring",
        "Model evaluation and interpretation",
        "Software engineering for ML projects",
        "Object-oriented design patterns",
        "Professional code organization and documentation"
    ]
    
    for point in learning_points:
        print(f"  • {point}")

def main():
    """Main demonstration function"""
    print(f"Neural Network from Scratch - Demonstration")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Demonstrate components
        if not demonstrate_components():
            print("❌ Component demonstration failed")
            return
        
        # Show project summary
        show_project_summary()
        
        # Ask if user wants to run mini training
        print_header("OPTIONAL: MINI TRAINING DEMO")
        print("Would you like to see a quick training demonstration?")
        print("This will download MNIST data (if not already present) and train a small model.")
        print("Note: This may take a few minutes for the first run.")
        
        response = input("\nRun mini training demo? (y/n): ").lower().strip()
        
        if response == 'y' or response == 'yes':
            if not demonstrate_mini_training():
                print("❌ Mini training demonstration failed")
                return
        
        print_header("DEMONSTRATION COMPLETED SUCCESSFULLY! 🎉")
        print("""
Next Steps:
1. Run 'python main.py --quick_test' for a quick full pipeline test
2. Run 'python main.py' for complete training with default settings  
3. Read USAGE_GUIDE.md for detailed usage instructions
4. Explore the src/ directory to understand the implementation
5. Modify architectures and parameters to experiment

The neural network implementation is ready for use!
        """)
        
    except KeyboardInterrupt:
        print("\n\nDemonstration interrupted by user.")
    except Exception as e:
        print(f"\n❌ Demonstration failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
