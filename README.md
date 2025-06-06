# Neural Network from Scratch - Professional Implementation

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-Only-green.svg)](https://numpy.org)
[![Accuracy](https://img.shields.io/badge/MNIST%20Accuracy-98.06%25-brightgreen.svg)](/)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg)](/)

A complete, professional neural network implementation built entirely from scratch using only NumPy for MNIST digit classification. This project achieves **98.06% test accuracy** with a clean, object-oriented architecture and comprehensive documentation.

## 🎯 Key Features

- ✅ **Pure NumPy Implementation** - No TensorFlow, PyTorch, or scikit-learn for ML core
- ✅ **98.06% Test Accuracy** - Proven performance on MNIST dataset  
- ✅ **Professional Architecture** - Clean OOP design with separation of concerns
- ✅ **6 Activation Functions** - ReLU, Sigmoid, Tanh, Softmax, LeakyReLU, Linear
- ✅ **5 Loss Functions** - CrossEntropy, MSE, BCE, CategoricalCE, Huber Loss
- ✅ **4 Weight Initializers** - Xavier, He, Random, Zeros initialization
- ✅ **Advanced Training** - SGD with momentum, learning rate scheduling, early stopping
- ✅ **Comprehensive CLI** - Full command-line interface with 20+ configurable parameters
- ✅ **Interactive GUI** - Real-time digit recognition with drawing canvas
- ✅ **Organized Visualizations** - Timestamped directories for train/test/main runs
- ✅ **Model Persistence** - Save/load trained models with full state restoration

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Quick test (recommended first run)
python apps/main.py --epochs 2 --quick_test

# 3. Training with custom parameters
python apps/train.py --epochs 10 --batch-size 64 --learning-rate 0.001

# 4. Test a trained model
python apps/test.py --model-path models/mnist_model_YYYYMMDD_HHMMSS.pkl

# 5. Launch interactive GUI
python apps/play_app.py
```

## 📖 Command Line Interface Documentation

### 🏋️ Training Script (`train.py`)

Dedicated training script with comprehensive hyperparameter control and model customization options.

```bash
python apps/train.py [OPTIONS]
```

**Usage Examples:**

```bash
# Basic training with default parameters
python apps/train.py

# Custom architecture with deeper network
python apps/train.py --hidden-layers 512 256 128 64 --epochs 100

# Regression-style training with MSE loss
python apps/train.py --loss mse --output-activation linear --weight-init xavier

# Xavier initialization with sigmoid activation
python apps/train.py --activation sigmoid --weight-init xavier --epochs 50

# Advanced learning rate scheduling
python apps/train.py --lr-step-size 10 --lr-gamma 0.8 --epochs 100

# Regularized training to prevent overfitting
python apps/train.py --dropout-rate 0.4 --patience 15 --validation-split 0.2

# Silent training for batch processing
python apps/train.py --verbose 0 --no-report
```

**Complete Parameter Reference:**

| Parameter | Type | Default | Choices | Description |
|-----------|------|---------|---------|-------------|
| **Training Parameters** | | | | |
| `--epochs` | int | 50 | - | Number of training epochs |
| `--batch-size` | int | 128 | - | Mini-batch size for gradient updates |
| `--learning-rate` | float | 0.001 | - | Initial learning rate for optimizer |
| `--momentum` | float | 0.9 | - | Momentum coefficient for SGD optimizer |
| **Network Architecture** | | | | |
| `--hidden-layers` | int[] | [256, 128, 64] | - | Hidden layer sizes (space-separated) |
| `--dropout-rate` | float | 0.3 | - | Dropout probability for regularization |
| `--activation` | choice | relu | `relu`, `sigmoid`, `tanh` | Activation function for hidden layers |
| `--weight-init` | choice | he | `xavier`, `he`, `random`, `zeros` | Weight initialization method |
| `--loss` | choice | cross_entropy | `mse`, `cross_entropy`, `binary_crossentropy`, `categorical_crossentropy_smooth`, `huber` | Loss function to use for training |
| `--output-activation` | choice | softmax | `softmax`, `sigmoid`, `linear`, `tanh` | Output layer activation function |
| **Training Optimization** | | | | |
| `--patience` | int | 10 | - | Early stopping patience (epochs without improvement) |
| `--validation-split` | float | 0.15 | - | Fraction of training data used for validation |
| `--min-delta` | float | 0.0001 | - | Minimum improvement threshold for early stopping |
| `--lr-step-size` | int | 15 | - | Step size for learning rate scheduler |
| `--lr-gamma` | float | 0.5 | - | Learning rate decay factor |
| `--no-lr-scheduler` | flag | False | - | Disable learning rate scheduler |
| **Saving and Logging** | | | | |
| `--no-save` | flag | False | - | Skip saving the trained model |
| `--no-report` | flag | False | - | Skip generating visualization reports |
| `--model-path` | str | auto | - | Custom model save path (default: timestamped) |
| `--verbose` | choice | 2 | `0`, `1`, `2` | Verbosity: `0`=silent, `1`=progress, `2`=detailed |

**Parameter Combinations Guide:**

```bash
# Classification (recommended for MNIST)
python apps/train.py --loss cross_entropy --output-activation softmax --weight-init he

# Regression-style training
python apps/train.py --loss mse --output-activation linear --weight-init xavier

# Sigmoid-based network (classic approach)
python apps/train.py --activation sigmoid --weight-init xavier --loss cross_entropy

# Huber loss for robust training
python apps/train.py --loss huber --output-activation linear --weight-init he

# High-performance setup with advanced scheduling
python apps/train.py --epochs 200 --batch-size 64 --lr-step-size 10 --lr-gamma 0.8 --patience 20
```

### 🧪 Testing Script (`test.py`)

Comprehensive model evaluation with detailed performance analysis and error visualization.

```bash
python apps/test.py [OPTIONS]
```

**Usage Examples:**

```bash
# Test latest trained model
python apps/test.py

# Test specific model with detailed analysis
python apps/test.py --model-path models/best_model.pkl --verbose

# Quick evaluation without plots for CI/CD
python apps/test.py --no-visualizations

# Extensive error analysis
python apps/test.py --error-examples 20 --verbose
```

**Available Options:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--model-path` | str | auto | Path to trained model (default: latest in models/) |
| `--no-visualizations` | flag | False | Skip generating evaluation plots and reports |
| `--verbose` | flag | False | Enable detailed evaluation output and metrics |
| `--error-examples` | int | 5 | Number of misclassified examples to analyze |

### 🎯 Main Pipeline (`main.py`)

Complete end-to-end pipeline with predefined architectures and streamlined workflow.

```bash
python apps/main.py [OPTIONS]
```

**Usage Examples:**

```bash
# Complete pipeline with default balanced architecture
python apps/main.py

# Quick verification test (recommended for first run)
python apps/main.py --quick_test

# Deep architecture for maximum accuracy
python apps/main.py --arch deep --epochs 100

# Simple architecture for fast experimentation
python apps/main.py --arch simple --epochs 10

# Production training without visualizations
python apps/main.py --arch deep --no_plots --epochs 200
```

**Available Options:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--arch` | choice | default | Predefined architecture: `simple`, `default`, `deep` |
| `--epochs` | int | 50 | Number of training epochs |
| `--batch_size` | int | 128 | Training batch size |
| `--learning_rate` | float | 0.001 | Learning rate for optimization |
| `--no_plots` | flag | False | Skip all visualization generation |
| `--quick_test` | flag | False | Fast test with reduced dataset (1000 samples) |

**Architecture Specifications:**

| Architecture | Layers | Expected Accuracy | Training Time |
|-------------|--------|------------------|---------------|
| **Simple** | 784→128→10 | 85-90% | 1-2 minutes |
| **Default** | 784→256→128→10 + Dropout | 92-95% | 3-5 minutes |
| **Deep** | 784→512→256→128→10 + Dropout | 95-98% | 1-2 hours |

## 📊 Performance Results

Our implementation achieves **98.06% accuracy** on the MNIST test set with organized visualization outputs:

**Latest Achievement (June 6, 2025):**

- **Test Accuracy:** 98.06%
- **Configuration:** Deep architecture, 150 epochs, plateau learning rate scheduling
- **Training Time:** ~1 hour 20 minutes
- **Architecture:** 784→512→256→128→10 with dropout layers

**Training Run Visualizations** (`logs/run_train_YYYYMMDD_HHMMSS/`):

- Training history plots with loss and accuracy curves  
- Confusion matrices (standard and normalized)
- Sample predictions with confidence scores
- Class distribution analysis
- Weight distribution visualizations

**Test Run Visualizations** (`logs/run_test_YYYYMMDD_HHMMSS/`):

- Model evaluation metrics and confusion matrices
- Error analysis with misclassification patterns
- Test-specific performance visualizations

## 📁 Project Architecture

```txt
Neural-Network-from-Scratch/
├── 📁 src/                             # Core Implementation
│   ├── 📁 data/
│   │   ├── __init__.py
│   │   └── data_loader.py              # MNIST data pipeline
│   ├── 📁 models/
│   │   ├── __init__.py
│   │   ├── activations.py              # 5 activation functions
│   │   ├── layers.py                   # Dense & Dropout layers
│   │   └── neural_network.py           # Main NeuralNetwork class
│   ├── 📁 training/
│   │   ├── __init__.py
│   │   ├── loss_functions.py           # 4 loss functions
│   │   └── trainer.py                  # Advanced training system
│   └── 📁 utils/
│       ├── __init__.py
│       ├── metrics.py                  # Evaluation metrics
│       └── visualization.py            # Professional plotting
├── 📁 apps/                            # Main Apps
│   ├── main.py                         # Complete pipeline
│   ├── train.py                        # Standalone training
│   ├── test.py                         # Model evaluation
│   ├── demo.py                         # Component demo
│   └── play_app.py                     # Interactive GUI
├── 📁 debug/
│   ├── debug_model.py                  # Examine saved model structure
│   └── debug_test.py                   # Isolate the error in test.py
├── 📁 test/
│   ├── minimal_test.py                 # Verify the NN works with MNIST data
│   ├── test_basic.py                   # Verify the NN implementation works
│   └── test_model_loading.py           # Check model loading functionality
├── 📁 data/                            # MNIST dataset
├── 📁 models/                          # Saved models
├── 📁 logs/                            # Training logs & organized visualizations
│   ├── 📁 run_train_YYYYMMDD_HHMMSS/   # Training run outputs
│   ├── 📁 run_test_YYYYMMDD_HHMMSS/    # Test run outputs  
│   └── 📁 run_main_YYYYMMDD_HHMMSS/    # Main pipeline outputs
├── 📄 requirements.txt                 # Dependencies
├── 📄 README.md                        # This file
├── 📄 USAGE_GUIDE.md                   # Detailed usage
└── 📄 PROJECT_STATUS.md                # Complete status
```

## 🧠 Core Components

### Neural Network Architecture

- **Flexible Model Building** - Add layers dynamically with `add_layer()`
- **Forward Propagation** - Optimized NumPy matrix operations
- **Backpropagation** - Automatic gradient computation with chain rule
- **Model Serialization** - Complete save/load functionality

### Activation Functions (6 Types)

```python
ReLU()          # Rectified Linear Unit
Sigmoid()       # Logistic activation  
Tanh()          # Hyperbolic tangent
Softmax()       # Probability distribution
LeakyReLU()     # Parameterized ReLU
Linear()        # Identity function (for regression)
```

### Loss Functions (5 Types)

```python
CrossEntropyLoss()           # Multiclass classification
MeanSquaredError()           # Regression tasks
BinaryCrossEntropyLoss()     # Binary classification  
CategoricalCrossEntropyLoss() # Smoothed categorical cross-entropy
HuberLoss()                  # Robust loss function
```

### Training Features

- **SGD with Momentum** - Accelerated optimization
- **Learning Rate Scheduling** - Adaptive learning rates
- **Early Stopping** - Prevent overfitting
- **Mini-batch Training** - Memory-efficient processing
- **Progress Tracking** - Real-time monitoring

## 🎮 Usage Examples

### Interactive GUI Application

```bash
# Launch drawing interface for digit recognition
python apps/play_app.py
```

### Programmatic Interface

```python
from src.models.neural_network import NeuralNetwork
from src.models.layers import DenseLayer, DropoutLayer
from src.models.activations import ReLU, Softmax
from src.training.trainer import Trainer

# Create custom model
model = NeuralNetwork()
model.add_layer(DenseLayer(784, 256, activation=ReLU()))
model.add_layer(DropoutLayer(0.2))
model.add_layer(DenseLayer(256, 128, activation=ReLU()))
model.add_layer(DenseLayer(128, 10, activation=Softmax()))

# Train with advanced features
trainer = Trainer(model, patience=15, save_best=True)
history = trainer.train(X_train, y_train, X_val, y_val, 
                       epochs=50, batch_size=64, momentum=0.9)
```

## 🏗️ Model Architectures

### Simple (Quick Testing)

```txt
Input(784) → Dense(128, ReLU) → Dense(10, Softmax)
Performance: 85-90% accuracy, 1-2 minutes training
```

### Default (Balanced)

```txt
Input(784) → Dense(256, ReLU) → Dropout(0.2) → Dense(128, ReLU) → Dropout(0.2) → Dense(10, Softmax)
Performance: 92-95% accuracy, 3-5 minutes training
```

### Deep (Maximum Accuracy)

```txt  
Input(784) → Dense(512, ReLU) → Dropout(0.3) → Dense(256, ReLU) → Dropout(0.3) → Dense(128, ReLU) → Dropout(0.2) → Dense(10, Softmax)
Performance: 95-98% accuracy, 1-2 hours training (98.06% achieved with plateau scheduling)
```

## 📋 Requirements*

```txt
numpy>=2.2.6              # Core mathematical operations
torch>=2.7.1+cpu          # MNIST dataset downloading only
torchvision>=0.22.1+cpu   # Dataset utilities
matplotlib>=3.10.3        # Visualization (optional)
tkinter                   # GUI apps (standard library)
```

**Note: Core dependencies are listed above. See `requirements.txt` for the complete dependency list. Library versions may vary depending on your system and Python environment - the versions shown are from our development setup.*

## 🏆 Educational Value

This implementation demonstrates:

- **Mathematical Foundations**: Forward propagation, backpropagation, gradient descent
- **Software Engineering**: OOP design, modular architecture, error handling
- **Machine Learning**: Training strategies, regularization, evaluation metrics
- **Data Science**: Visualization, performance analysis, model interpretation

## 📄 Documentation

- **README.md** - Project overview and quick start (this file)
- **USAGE_GUIDE.md** - Comprehensive usage instructions
- **PROJECT_STATUS.md** - Detailed project status and implementation details

## 🚀 Getting Started

1. **Clone and Setup**:

   ```bash
   git clone <repository-url>
   cd Neural-Network-from-Scratch
   pip install -r requirements.txt
   ```

2. **Quick Verification**:

   ```bash
   python test/minimal_test.py
   ```

3. **Run Demo**:

   ```bash
   python apps/demo.py
   ```

4. **Train Your First Model**:

   ```bash
   python apps/main.py --quick_test
   ```

5. **Full Training**:

   ```bash
   python apps/main.py
   ```

## 🎯 Project Status

✅ **FULLY OPERATIONAL** - Production ready with 98.06% test accuracy achieved!

This implementation successfully demonstrates professional neural network development from scratch using only NumPy, achieving state-of-the-art results on MNIST digit classification.
