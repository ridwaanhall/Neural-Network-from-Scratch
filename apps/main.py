#!/usr/bin/env python3
"""
Main execution script for MNIST Neural Network from Scratch
============================================================

This script provides a complete end-to-end pipeline demonstration of the neural network
implementation. It includes data loading, model training, evaluation, and visualization.

Author: Ridwan Halim (ridwaanhall)
Date: June 04, 2025
"""

import os
import sys
import logging
import argparse
from datetime import datetime

from src.data.data_loader import MNISTDataLoader
from src.models.neural_network import NeuralNetwork
from src.models.layers import DenseLayer, DropoutLayer
from src.models.activations import ReLU, Sigmoid, Softmax
from src.training.trainer import Trainer
from src.utils.visualization import plot_training_history, plot_confusion_matrix, plot_sample_predictions
from src.utils.metrics import calculate_metrics

def setup_logging():
    """Setup logging configuration"""
    os.makedirs('logs', exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'logs/main_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def create_model(architecture='default'):
    """
    Create neural network model with specified architecture
    
    Args:
        architecture (str): Model architecture type ('default', 'deep', 'simple')
    
    Returns:
        NeuralNetwork: Configured neural network model
    """
    if architecture == 'simple':
        # Simple 2-layer network
        model = NeuralNetwork()
        model.add_layer(DenseLayer(784, 128, activation=ReLU(), weight_init='xavier'))
        model.add_layer(DenseLayer(128, 10, activation=Softmax(), weight_init='xavier'))
        
    elif architecture == 'deep':
        # Deeper network with dropout
        model = NeuralNetwork()
        model.add_layer(DenseLayer(784, 512, activation=ReLU(), weight_init='xavier'))
        model.add_layer(DropoutLayer(0.3))
        model.add_layer(DenseLayer(512, 256, activation=ReLU(), weight_init='xavier'))
        model.add_layer(DropoutLayer(0.3))
        model.add_layer(DenseLayer(256, 128, activation=ReLU(), weight_init='xavier'))
        model.add_layer(DropoutLayer(0.2))
        model.add_layer(DenseLayer(128, 10, activation=Softmax(), weight_init='xavier'))
        
    else:  # default
        # Balanced 3-layer network
        model = NeuralNetwork()
        model.add_layer(DenseLayer(784, 256, activation=ReLU(), weight_init='xavier'))
        model.add_layer(DropoutLayer(0.2))
        model.add_layer(DenseLayer(256, 128, activation=ReLU(), weight_init='xavier'))
        model.add_layer(DropoutLayer(0.2))
        model.add_layer(DenseLayer(128, 10, activation=Softmax(), weight_init='xavier'))
    
    return model

def train_model(model, X_train, y_train, X_val, y_val, config):
    """
    Train the neural network model
    
    Args:
        model: Neural network model
        X_train, y_train: Training data and labels
        X_val, y_val: Validation data and labels
        config (dict): Training configuration
    
    Returns:
        Trainer: Trained trainer object with history
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting model training...")
      # Initialize trainer
    trainer = Trainer(
        model=model,
        patience=config.get('early_stopping_patience', 10),
        min_delta=1e-4,
        lr_scheduler={'type': config.get('lr_schedule', 'plateau')},
        save_best=True,
        verbose=1
    )
      # Train the model
    history = trainer.train(
        X_train, y_train,
        X_val, y_val,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        momentum=config.get('momentum', 0.9),
        shuffle=True
    )
    
    logger.info("Training completed!")
    return trainer

def evaluate_model(model, X_test, y_test, class_names=None):
    """
    Evaluate the trained model on test data
    
    Args:
        model: Trained neural network model
        X_test, y_test: Test data and labels
        class_names: Names of classes for visualization
    
    Returns:
        dict: Evaluation metrics
    """
    logger = logging.getLogger(__name__)
    logger.info("Evaluating model on test data...")
    
    # Make predictions
    predictions = model.predict(X_test)
    predicted_classes = predictions.argmax(axis=1)
    true_classes = y_test.argmax(axis=1)
    
    # Calculate comprehensive metrics
    metrics = calculate_metrics(true_classes, predicted_classes, class_names)
    
    # Log results
    logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Test Precision: {metrics['precision']:.4f}")
    logger.info(f"Test Recall: {metrics['recall']:.4f}")
    logger.info(f"Test F1-Score: {metrics['f1_score']:.4f}")
    
    return metrics, predictions

def save_model_and_results(model, trainer, metrics, config, timestamp=None):
    """Save trained model and results"""
    logger = logging.getLogger(__name__)
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Use provided timestamp or generate new one
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model_path = f'models/mnist_model_{timestamp}.pkl'
    
    model.save_model(model_path)
    logger.info(f"Model saved to: {model_path}")
    
    # Save training configuration and results
    results = {
        'config': config,
        'final_metrics': metrics,
        'training_history': trainer.history if hasattr(trainer, 'history') else None
    }
    
    import pickle
    results_path = f'models/results_{timestamp}.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    logger.info(f"Results saved to: {results_path}")
    return model_path, results_path

def create_visualizations(trainer, metrics, X_test, y_test, predictions, save_plots=True, timestamp=None):
    """Create and optionally save visualization plots"""
    logger = logging.getLogger(__name__)
    logger.info("Creating visualizations...")
    
    # Use provided timestamp or generate new one
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # Plot training history if available
        if hasattr(trainer.model, 'history') and trainer.model.history:
            training_history_path = f'logs/training_history_{timestamp}.png' if save_plots else None
            plot_training_history(trainer.model.history, save_path=training_history_path)
            logger.info(f"Training history plot saved to: {training_history_path}")
        else:
            logger.warning("No training history available for plotting")
        
        # Calculate metrics and plot confusion matrix
        true_classes = y_test.argmax(axis=1)
        predicted_classes = predictions.argmax(axis=1)
        class_names = [str(i) for i in range(10)]  # MNIST digit classes
        
        # Calculate comprehensive metrics including confusion matrix
        from src.utils.metrics import calculate_metrics
        metrics = calculate_metrics(true_classes, predicted_classes, class_names)
        
        confusion_matrix_path = f'logs/confusion_matrix_{timestamp}.png' if save_plots else None
        plot_confusion_matrix(
            metrics['confusion_matrix'],
            class_names=class_names,
            save_path=confusion_matrix_path
        )
        
        # Plot sample predictions
        sample_predictions_path = f'logs/sample_predictions_{timestamp}.png' if save_plots else None
        plot_sample_predictions(
            X_test, y_test, predictions,
            save_path=sample_predictions_path
        )
        
        logger.info("Visualizations created successfully!")
        
    except Exception as e:
        logger.warning(f"Could not create visualizations: {e}")
        logger.warning("This might be due to missing matplotlib. Install it with: pip install matplotlib")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='MNIST Neural Network from Scratch')
    parser.add_argument('--architecture', choices=['simple', 'default', 'deep'], 
                       default='default', help='Model architecture')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--no_plots', action='store_true', help='Skip plot generation')
    parser.add_argument('--quick_test', action='store_true', help='Run quick test with small dataset')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("="*60)
    logger.info("MNIST Neural Network from Scratch - Starting Pipeline")
    logger.info("="*60)
    
    try:
        # Configuration
        config = {
            'architecture': args.architecture,
            'epochs': 5 if args.quick_test else args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'momentum': 0.9,
            'early_stopping_patience': 5 if args.quick_test else 10,
            'lr_schedule': 'plateau'
        }
        
        logger.info(f"Configuration: {config}")
          # Step 1: Load and preprocess data
        logger.info("Step 1: Loading MNIST dataset...")
        data_loader = MNISTDataLoader()
        
        if args.quick_test:
            # Use smaller subset for quick testing
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_loader.load_data(
                validation_split=0.2, normalize=True, flatten=True
            )
            # Take only first 1000 samples for quick test
            X_train, y_train = X_train[:1000], y_train[:1000]
            X_val, y_val = X_val[:200], y_val[:200]
            X_test, y_test = X_test[:200], y_test[:200]
            logger.info("Using subset of data for quick test")
        else:
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_loader.load_data(
                validation_split=0.2, normalize=True, flatten=True
            )
        
        logger.info(f"Data loaded - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        # Step 2: Create model
        logger.info(f"Step 2: Creating {config['architecture']} neural network model...")
        model = create_model(config['architecture'])
        logger.info(f"Model created with {len(model.layers)} layers")
        
        # Step 3: Train model
        logger.info("Step 3: Training the model...")
        trainer = train_model(model, X_train, y_train, X_val, y_val, config)
          # Step 4: Evaluate model
        logger.info("Step 4: Evaluating the model...")
        metrics, predictions = evaluate_model(model, X_test, y_test)
        
        # Generate timestamp for consistent file naming
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Step 5: Save model and results
        logger.info("Step 5: Saving model and results...")
        model_path, results_path = save_model_and_results(model, trainer, metrics, config, timestamp)
        
        # Step 6: Create visualizations
        if not args.no_plots:
            logger.info("Step 6: Creating visualizations...")
            create_visualizations(trainer, metrics, X_test, y_test, predictions, save_plots=True, timestamp=timestamp)
        
        # Final summary
        logger.info("="*60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"Final Test Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Results saved to: {results_path}")
        logger.info("Check the 'logs' folder for visualizations and detailed logs")
        logger.info("="*60)
        
        return model, metrics
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        logger.error("Check the logs for detailed error information")
        raise

if __name__ == "__main__":
    model, metrics = main()
