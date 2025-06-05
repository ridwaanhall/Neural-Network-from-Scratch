"""
Testing Script for MNIST Neural Network

This script loads a trained neural network model and evaluates it on the test set.
It provides comprehensive testing metrics and visualizations.
"""

import numpy as np
import os
import sys
from datetime import datetime

# Add the parent directory to the Python path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_loader import MNISTDataLoader
from src.models.neural_network import NeuralNetwork
from src.utils.metrics import classification_report, MetricsTracker
from src.utils.visualization import (plot_confusion_matrix, plot_sample_predictions, 
                                   plot_class_distribution)


def load_model(model_path):
    """
    Load a trained model from file.
    
    Args:
        model_path (str): Path to the saved model
        
    Returns:
        NeuralNetwork: Loaded model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Create a new model instance and load the saved parameters
    model = NeuralNetwork()
    model.load_model(model_path)
    return model


def evaluate_model_comprehensive(model, X_test, y_test, class_names=None):
    """
    Perform comprehensive evaluation of the model.
    
    Args:
        model: Trained neural network model
        X_test (np.ndarray): Test input data
        y_test (np.ndarray): Test labels
        class_names (list): Names of classes
        
    Returns:
        dict: Comprehensive evaluation results
    """
    print("Performing comprehensive model evaluation...")
      # Get predictions
    print("Making predictions on test data...")
    try:
        predictions_proba = model.predict(X_test)
        print(f"✓ Predictions completed - shape: {predictions_proba.shape}")
    except Exception as e:
        print(f"❌ Error making predictions: {e}")
        raise
    
    try:
        predictions_classes = model.predict_classes(X_test)
        print(f"✓ Class predictions completed - shape: {predictions_classes.shape}")
    except Exception as e:
        print(f"❌ Error making class predictions: {e}")
        raise
    
    # Convert one-hot to class indices if necessary
    print("Converting labels...")
    if y_test.ndim > 1 and y_test.shape[1] > 1:
        true_classes = np.argmax(y_test, axis=1)
        print(f"✓ Converted one-hot labels to class indices - shape: {true_classes.shape}")
    else:
        true_classes = y_test
        print(f"✓ Using labels as-is - shape: {true_classes.shape}")
    
    # Calculate basic metrics
    print("Calculating basic metrics...")
    try:
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        print(f"✓ Basic metrics calculated - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")
    except Exception as e:
        print(f"❌ Error calculating basic metrics: {e}")
        raise
      # Get confusion matrix
    print("Calculating confusion matrix...")
    try:
        confusion_mat = model.get_confusion_matrix(X_test, y_test)
        print(f"✓ Confusion matrix calculated - shape: {confusion_mat.shape}")
    except Exception as e:
        print(f"❌ Error calculating confusion matrix: {e}")
        raise
    
    # Use metrics tracker for detailed metrics
    num_classes = 10  # MNIST has 10 digit classes
    
    # Ensure class_names is properly initialized
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]
    
    print("Setting up MetricsTracker...")
    try:
        # Check data types and shapes before MetricsTracker
        print(f"Data validation:")
        print(f"  num_classes: {num_classes}")
        print(f"  true_classes: type={type(true_classes)}, shape={true_classes.shape}, dtype={true_classes.dtype}")
        print(f"  predictions_classes: type={type(predictions_classes)}, shape={predictions_classes.shape}, dtype={predictions_classes.dtype}")
        print(f"  predictions_proba: type={type(predictions_proba)}, shape={predictions_proba.shape}, dtype={predictions_proba.dtype}")
        print(f"  class_names: {class_names}")
        
        # Check for any NaN or infinite values
        if np.any(np.isnan(predictions_proba)):
            print("⚠️  Warning: NaN values found in predictions_proba")
        if np.any(np.isinf(predictions_proba)):
            print("⚠️  Warning: Infinite values found in predictions_proba")
        
        # Check value ranges
        print(f"  true_classes range: {np.min(true_classes)} to {np.max(true_classes)}")
        print(f"  predictions_classes range: {np.min(predictions_classes)} to {np.max(predictions_classes)}")
        print(f"  predictions_proba range: {np.min(predictions_proba):.6f} to {np.max(predictions_proba):.6f}")
        
        metrics_tracker = MetricsTracker(num_classes=num_classes, class_names=class_names)
        print("✓ MetricsTracker created successfully")
        
        print("Updating MetricsTracker...")
        metrics_tracker.update(true_classes, predictions_classes, predictions_proba)
        print("✓ MetricsTracker updated successfully")
        
        print("Computing detailed metrics...")
        detailed_metrics = metrics_tracker.compute_metrics()
        print("✓ Detailed metrics computed successfully")
        
    except Exception as e:
        print(f"❌ Error in MetricsTracker processing: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to basic metrics
        detailed_metrics = {
            'precision_macro': 0.0,
            'recall_macro': 0.0,
            'f1_macro': 0.0,
            'precision_weighted': 0.0,
            'recall_weighted': 0.0,
            'f1_weighted': 0.0
        }
    
    # Calculate per-class accuracy
    per_class_accuracy = np.diag(confusion_mat) / np.sum(confusion_mat, axis=1)
    
    # Find misclassified samples
    misclassified_indices = np.where(true_classes != predictions_classes)[0]
    misclassification_rate = len(misclassified_indices) / len(true_classes)
    
    # Calculate confidence statistics
    max_confidences = np.max(predictions_proba, axis=1)
    avg_confidence = np.mean(max_confidences)
    correct_confidences = max_confidences[true_classes == predictions_classes]
    incorrect_confidences = max_confidences[true_classes != predictions_classes]
    
    results = {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'confusion_matrix': confusion_mat,
        'per_class_accuracy': per_class_accuracy,
        'detailed_metrics': detailed_metrics,
        'predictions_proba': predictions_proba,
        'predictions_classes': predictions_classes,
        'true_classes': true_classes,
        'misclassified_indices': misclassified_indices,
        'misclassification_rate': misclassification_rate,
        'avg_confidence': avg_confidence,
        'correct_avg_confidence': np.mean(correct_confidences) if len(correct_confidences) > 0 else 0,
        'incorrect_avg_confidence': np.mean(incorrect_confidences) if len(incorrect_confidences) > 0 else 0
    }
    
    return results


def print_test_results(results, class_names=None):
    """
    Print detailed test results.
    
    Args:
        results (dict): Test results from evaluate_model_comprehensive
        class_names (list): Names of classes
    """
    print("\n" + "=" * 80)
    print("COMPREHENSIVE TEST RESULTS")
    print("=" * 80)
    
    # Basic metrics
    print(f"Test Loss: {results['test_loss']:.6f}")
    print(f"Test Accuracy: {results['test_accuracy']:.6f} ({results['test_accuracy']*100:.2f}%)")
    print(f"Misclassification Rate: {results['misclassification_rate']:.6f} ({results['misclassification_rate']*100:.2f}%)")
    print()
    
    # Confidence statistics
    print("Prediction Confidence Statistics:")
    print(f"  Average Confidence: {results['avg_confidence']:.4f}")
    print(f"  Correct Predictions Avg Confidence: {results['correct_avg_confidence']:.4f}")
    print(f"  Incorrect Predictions Avg Confidence: {results['incorrect_avg_confidence']:.4f}")
    print()
    
    # Detailed metrics
    detailed = results['detailed_metrics']
    print("Detailed Metrics:")
    print(f"  Precision (Macro): {detailed['precision_macro']:.4f}")
    print(f"  Recall (Macro): {detailed['recall_macro']:.4f}")
    print(f"  F1-Score (Macro): {detailed['f1_macro']:.4f}")
    print(f"  Precision (Weighted): {detailed['precision_weighted']:.4f}")
    print(f"  Recall (Weighted): {detailed['recall_weighted']:.4f}")
    print(f"  F1-Score (Weighted): {detailed['f1_weighted']:.4f}")
    
    if 'top_1_accuracy' in detailed:
        print(f"  Top-1 Accuracy: {detailed['top_1_accuracy']:.4f}")
        print(f"  Top-2 Accuracy: {detailed['top_2_accuracy']:.4f}")
        print(f"  Top-3 Accuracy: {detailed['top_3_accuracy']:.4f}")
    print()
    
    # Per-class accuracy
    print("Per-Class Accuracy:")
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(results['per_class_accuracy']))]
    
    for i, (class_name, accuracy) in enumerate(zip(class_names, results['per_class_accuracy'])):
        print(f"  {class_name}: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print()
    
    # Confusion matrix summary
    cm = results['confusion_matrix']
    print("Confusion Matrix Summary:")
    print(f"  Diagonal Sum (Correct): {np.sum(np.diag(cm))}")
    print(f"  Off-diagonal Sum (Incorrect): {np.sum(cm) - np.sum(np.diag(cm))}")
    print(f"  Total Samples: {np.sum(cm)}")


def analyze_errors(results, X_test, class_names=None, num_examples=5):
    """
    Analyze and display error cases.
    
    Args:
        results (dict): Test results
        X_test (np.ndarray): Test input data
        class_names (list): Names of classes
        num_examples (int): Number of error examples to show
    """
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS")
    print("=" * 60)
    
    misclassified_indices = results['misclassified_indices']
    true_classes = results['true_classes']
    pred_classes = results['predictions_classes']
    pred_proba = results['predictions_proba']
    
    if len(misclassified_indices) == 0:
        print("No misclassified samples found!")
        return
    
    print(f"Total misclassified samples: {len(misclassified_indices)}")
    print(f"Misclassification rate: {len(misclassified_indices)/len(true_classes)*100:.2f}%")
    print()
    
    # Analyze error patterns
    error_pairs = {}
    for idx in misclassified_indices:
        true_label = true_classes[idx]
        pred_label = pred_classes[idx]
        pair = (true_label, pred_label)
        error_pairs[pair] = error_pairs.get(pair, 0) + 1
    
    # Sort by frequency
    sorted_errors = sorted(error_pairs.items(), key=lambda x: x[1], reverse=True)
    
    print("Most Common Error Patterns:")
    for i, ((true_label, pred_label), count) in enumerate(sorted_errors[:10]):
        if class_names:
            true_name = class_names[true_label]
            pred_name = class_names[pred_label]
        else:
            true_name = str(true_label)
            pred_name = str(pred_label)
        
        percentage = count / len(misclassified_indices) * 100
        print(f"  {i+1}. {true_name} → {pred_name}: {count} times ({percentage:.1f}% of errors)")
    print()
    
    # Show examples of misclassified samples
    print(f"Examples of Misclassified Samples (showing {min(num_examples, len(misclassified_indices))}):")
    
    # Select random misclassified samples
    selected_indices = np.random.choice(misclassified_indices, 
                                      min(num_examples, len(misclassified_indices)), 
                                      replace=False)
    
    for i, idx in enumerate(selected_indices):
        true_label = true_classes[idx]
        pred_label = pred_classes[idx]
        confidence = pred_proba[idx, pred_label]
        
        if class_names:
            true_name = class_names[true_label]
            pred_name = class_names[pred_label]
        else:
            true_name = str(true_label)
            pred_name = str(pred_label)
        
        print(f"  Sample {i+1}: True: {true_name}, Predicted: {pred_name}, "
              f"Confidence: {confidence:.3f}")


def find_latest_model():
    """Find the most recent model file in the models directory."""
    models_dir = 'models'
    if not os.path.exists(models_dir):
        return None
    
    # Check for the default model name first
    default_model = os.path.join(models_dir, 'mnist_nn_model.pkl')
    if os.path.exists(default_model):
        return default_model
    
    # If default model doesn't exist, look for timestamped models
    model_files = [f for f in os.listdir(models_dir) if f.startswith('mnist_model_') and f.endswith('.pkl')]
    
    if not model_files:
        return None
    
    # Sort by modification time to get the latest
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True)
    return os.path.join(models_dir, model_files[0])


def main():
    """Main testing function."""
    print("=" * 80)
    print("MNIST Neural Network Testing")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Configuration
    model_path = find_latest_model()
    if model_path is None:
        print("❌ No trained model found in models/ directory.")
        print("Please train a model first using: python main.py or python train.py")
        return
    
    data_dir = 'data'
    class_names = [str(i) for i in range(10)]  # MNIST digit names
      # Step 1: Load the trained model
    print("Step 1: Loading trained model...")
    print("-" * 50)
    
    print(f"Found model: {model_path}")
    
    try:
        model = load_model(model_path)
        print(f"✓ Model loaded successfully from: {model_path}")
        model.summary()
        print()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Please make sure you have trained a model using train.py first.")
        return
    
    # Step 2: Load test data
    print("Step 2: Loading test data...")
    print("-" * 50)
    
    try:
        data_loader = MNISTDataLoader(data_dir=data_dir)
        _, _, X_test, y_test = data_loader.load_data(
            normalize=True,
            flatten=True,
            one_hot=True
        )
        
        print(f"Test data loaded successfully:")
        print(f"  Test samples: {len(X_test)}")
        print(f"  Input shape: {X_test.shape}")
        print(f"  Labels shape: {y_test.shape}")
        print()
        
    except Exception as e:
        print(f"Error loading test data: {e}")
        return
    
    # Step 3: Comprehensive evaluation
    print("Step 3: Comprehensive model evaluation...")
    print("-" * 50)
    
    try:
        results = evaluate_model_comprehensive(model, X_test, y_test, class_names)
        print("Evaluation completed successfully.")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return
    
    # Step 4: Display results
    print_test_results(results, class_names)
    
    # Step 5: Error analysis
    analyze_errors(results, X_test, class_names)
    
    # Step 6: Generate classification report
    print("\n" + "=" * 60)
    print("DETAILED CLASSIFICATION REPORT")
    print("=" * 60)
    
    metrics_tracker = MetricsTracker(num_classes=10, class_names=class_names)
    metrics_tracker.update(results['true_classes'], results['predictions_classes'], 
                         results['predictions_proba'])
    print(metrics_tracker.get_classification_report())
    
    # Step 7: Visualizations (if matplotlib is available)
    print("\nStep 7: Creating visualizations...")
    print("-" * 50)
    
    try:
        # Plot confusion matrix
        plot_confusion_matrix(results['confusion_matrix'], 
                            class_names=class_names,
                            save_path='logs/test_confusion_matrix.png')
        
        # Plot sample predictions (including errors)
        plot_sample_predictions(X_test, results['true_classes'], 
                              results['predictions_classes'],
                              num_samples=16,
                              save_path='logs/test_sample_predictions.png',
                              class_names=class_names)
        
        # Plot class distribution
        plot_class_distribution(results['true_classes'], 
                              class_names=class_names,
                              save_path='logs/test_class_distribution.png',
                              title="Test Set Class Distribution")
        
        print("Visualizations saved to logs/ directory.")
        
    except ImportError:
        print("Matplotlib not available. Skipping visualizations.")
    except Exception as e:
        print(f"Error creating visualizations: {e}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("TESTING SUMMARY")
    print("=" * 80)
    print(f"Model Performance:")
    print(f"  Test Accuracy: {results['test_accuracy']:.4f} ({results['test_accuracy']*100:.2f}%)")
    print(f"  Test Loss: {results['test_loss']:.6f}")
    print(f"  Average Confidence: {results['avg_confidence']:.4f}")
    print(f"  Samples Tested: {len(X_test):,}")
    print(f"  Misclassified: {len(results['misclassified_indices']):,}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    try:
        main()
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nPress Enter to exit...")
    input()
