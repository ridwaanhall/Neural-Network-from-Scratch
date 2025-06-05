"""
Training Script for MNIST Neural Network

This script trains a neural network from scratch on the MNIST dataset.
It demonstrates the complete training pipeline with proper configuration
and monitoring capabilities.
"""

import numpy as np
import os
import time
import sys
from datetime import datetime

# Add the parent directory to the Python path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_loader import MNISTDataLoader, DataPreprocessor
from src.models.neural_network import NeuralNetwork
from src.training.trainer import Trainer
from src.utils.visualization import create_visualization_report
from src.utils.metrics import classification_report


def main():
    """Main training function."""
    print("=" * 80)
    print("MNIST Neural Network Training from Scratch")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Configuration
    config = {
        # Data parameters
        'data_dir': 'data',
        'validation_split': 0.15,
        'normalize': True,
        'flatten': True,
        'one_hot': True,
        
        # Network architecture
        'input_size': 784,  # 28x28 flattened
        'hidden_layers': [256, 128, 64],  # Three hidden layers
        'output_size': 10,  # 10 digit classes
        'activation': 'relu',
        'output_activation': 'softmax',
        'weight_init': 'he',  # Good for ReLU
        'use_dropout': True,
        'dropout_rate': 0.3,
        
        # Training parameters
        'epochs': 50,
        'batch_size': 128,
        'learning_rate': 0.001,
        'momentum': 0.9,
        'loss': 'cross_entropy',
        
        # Training optimization
        'patience': 10,  # Early stopping patience
        'min_delta': 1e-4,  # Minimum improvement threshold
        'lr_scheduler': {
            'type': 'step',
            'step_size': 15,
            'gamma': 0.5
        },
        
        # Logging and saving
        'save_best': True,
        'verbose': 2,  # Detailed progress
        'save_model': True,
        'model_path': 'models/mnist_nn_model.pkl',
        'create_report': True
    }
    
    print("Configuration:")
    for key, value in config.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value}")
        else:
            print(f"  {key}: {value}")
    print()
    
    # Step 1: Load and preprocess data
    print("Step 1: Loading and preprocessing MNIST dataset...")
    print("-" * 50)
    
    data_loader = MNISTDataLoader(data_dir=config['data_dir'])
    
    try:
        X_train, y_train, X_test, y_test = data_loader.load_data(
            normalize=config['normalize'],
            flatten=config['flatten'],
            one_hot=config['one_hot']
        )
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Training labels shape: {y_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        print(f"Test labels shape: {y_test.shape}")
        print(f"Data type: {X_train.dtype}")
        print(f"Value range: [{X_train.min():.3f}, {X_train.max():.3f}]")
        print()
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Split training data into train and validation
    print("Creating train/validation split...")
    X_train, X_val, y_train, y_val = DataPreprocessor.train_validation_split(
        X_train, y_train, 
        validation_split=config['validation_split'],
        shuffle=True,
        random_seed=42
    )
    
    print(f"Final training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    print()
    
    # Step 2: Create neural network model
    print("Step 2: Creating neural network model...")
    print("-" * 50)
    
    model = NeuralNetwork(
        input_size=config['input_size'],
        hidden_layers=config['hidden_layers'],
        output_size=config['output_size'],
        activation=config['activation'],
        output_activation=config['output_activation'],
        loss=config['loss'],
        learning_rate=config['learning_rate'],
        weight_init=config['weight_init'],
        use_dropout=config['use_dropout'],
        dropout_rate=config['dropout_rate']
    )
    
    # Print model summary
    model.summary()
    print()
    
    # Step 3: Create trainer and start training
    print("Step 3: Training the neural network...")
    print("-" * 50)
    
    trainer = Trainer(
        model=model,
        patience=config['patience'],
        min_delta=config['min_delta'],
        lr_scheduler=config['lr_scheduler'],
        save_best=config['save_best'],
        verbose=config['verbose']
    )
    
    # Start training
    start_time = time.time()
    
    try:
        history = trainer.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            momentum=config['momentum']
        )
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        return
    except Exception as e:
        print(f"\nError during training: {e}")
        return
    
    # Step 4: Evaluate on test set
    print("\nStep 4: Evaluating on test set...")
    print("-" * 50)
    
    # Get test predictions
    test_predictions = model.predict(X_test)
    test_pred_classes = model.predict_classes(X_test)
    
    # Evaluate model
    test_results = trainer.evaluate_model(X_test, y_test, verbose=True)
    
    # Get detailed classification report
    y_test_classes = np.argmax(y_test, axis=1) if y_test.ndim > 1 else y_test
    class_names = [str(i) for i in range(10)]
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test_classes, test_pred_classes, 
                              config['output_size'], class_names))
    
    # Step 5: Save model
    if config['save_model']:
        print("\nStep 5: Saving trained model...")
        print("-" * 50)
        
        os.makedirs(os.path.dirname(config['model_path']), exist_ok=True)
        model.save_model(config['model_path'])
        print()
    
    # Step 6: Create visualization report
    if config['create_report']:
        print("Step 6: Creating visualization report...")
        print("-" * 50)
        
        try:
            create_visualization_report(
                model=model,
                history=history,
                X_test=X_test,
                y_test=y_test_classes,
                y_pred=test_pred_classes,
                confusion_mat=test_results['confusion_matrix'],
                save_dir='logs'
            )
        except ImportError:
            print("Matplotlib not available. Skipping visualization report.")
        except Exception as e:
            print(f"Error creating visualization report: {e}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Final Test Accuracy: {test_results['test_accuracy']:.4f}")
    print(f"Final Test Loss: {test_results['test_loss']:.4f}")
    print(f"Total Training Time: {training_time:.2f} seconds")
    print(f"Best Validation Loss: {trainer.best_val_loss:.4f}")
    
    if history['train_loss']:
        print(f"Final Training Loss: {history['train_loss'][-1]:.4f}")
        print(f"Final Training Accuracy: {history['train_accuracy'][-1]:.4f}")
    
    if config['save_model']:
        print(f"Model saved to: {config['model_path']}")
    
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    
    try:
        main()
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nPress Enter to exit...")
    input()
