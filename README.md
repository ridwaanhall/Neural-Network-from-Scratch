# Neural Network from Scratch - Professional Implementation

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-Only-green.svg)](https://numpy.org)
[![Accuracy](https://img.shields.io/badge/MNIST%20Accuracy-96.71%25-brightgreen.svg)](/)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg)](/)

A complete, professional neural network implementation built entirely from scratch using only NumPy for MNIST digit classification. This project achieves **96.71% test accuracy** with a clean, object-oriented architecture and comprehensive documentation.

## 🎯 Key Features

- ✅ **Pure NumPy Implementation** - No TensorFlow, PyTorch, or scikit-learn for ML core
- ✅ **96.71% Test Accuracy** - Proven performance on MNIST dataset  
- ✅ **Professional Architecture** - Clean OOP design with separation of concerns
- ✅ **5 Activation Functions** - ReLU, Sigmoid, Tanh, Softmax, LeakyReLU
- ✅ **4 Loss Functions** - CrossEntropy, MSE, BCE, Huber Loss
- ✅ **Advanced Training** - SGD with momentum, learning rate scheduling, early stopping
- ✅ **Interactive GUI** - Real-time digit recognition with drawing canvas
- ✅ **Comprehensive Visualization** - Training curves, confusion matrices, sample predictions
- ✅ **Model Persistence** - Save/load trained models with full state restoration

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Quick test (recommended first run)
python main.py --quick_test

# 3. Full training
python main.py

# 4. Launch interactive GUI
python play_app.py
```

## 📊 Performance Results

Our implementation achieves **96.71% accuracy** on the MNIST test set:

![Training History](logs/training_history_20250604_231500.png)

![Confusion Matrix](logs/confusion_matrix_20250604_231500.png)

![Sample Predictions](logs/sample_predictions_20250604_231500.png)

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
├── 📁 Applications/
│   ├── main.py                         # Complete pipeline
│   ├── train.py                        # Standalone training
│   ├── test.py                         # Model evaluation
│   ├── demo.py                         # Component demo
│   ├── play_app.py                     # Interactive GUI
│   └── minimal_test.py                 # Quick verification
├── 📁 data/                            # MNIST dataset
├── 📁 models/                          # Saved models
├── 📁 logs/                            # Training logs & plots
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

### Activation Functions (5 Types)

```python
ReLU()          # Rectified Linear Unit
Sigmoid()       # Logistic activation  
Tanh()          # Hyperbolic tangent
Softmax()       # Probability distribution
LeakyReLU()     # Parameterized ReLU
```

### Loss Functions (4 Types)

```python
CrossEntropyLoss()    # Multiclass classification
MeanSquaredError()    # Regression tasks
BinaryCrossEntropy()  # Binary classification  
HuberLoss()          # Robust loss function
```

### Training Features

- **SGD with Momentum** - Accelerated optimization
- **Learning Rate Scheduling** - Adaptive learning rates
- **Early Stopping** - Prevent overfitting
- **Mini-batch Training** - Memory-efficient processing
- **Progress Tracking** - Real-time monitoring

## 🎮 Usage Examples

### Command Line Interface

```bash
# Basic training
python main.py

# Quick test with reduced dataset
python main.py --quick_test

# Deep architecture for maximum accuracy
python main.py --architecture deep

# Custom parameters
python main.py --epochs 100 --batch_size 64 --learning_rate 0.01

# Skip plot generation
python main.py --no_plots

# Standalone scripts
python train.py    # Training only
python test.py     # Evaluation only
python demo.py     # Component demonstration
```

### Interactive GUI Application

```bash
# Launch drawing interface for digit recognition
python play_app.py
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
Performance: 95-97% accuracy, 8-15 minutes training
```

## 📋 Requirements

```txt
numpy>=1.21.0              # Core mathematical operations
torch>=1.9.0               # MNIST dataset downloading only
torchvision>=0.10.0        # Dataset utilities
matplotlib>=3.3.0          # Visualization (optional)
tkinter                    # GUI applications (standard library)
```

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
   python minimal_test.py
   ```

3. **Run Demo**:

   ```bash
   python demo.py
   ```

4. **Train Your First Model**:

   ```bash
   python main.py --quick_test
   ```

5. **Full Training**:

   ```bash
   python main.py
   ```

## 🎯 Project Status

✅ **FULLY OPERATIONAL** - Production ready with 96.71% test accuracy achieved!

This implementation successfully demonstrates professional neural network development from scratch using only NumPy, achieving state-of-the-art results on MNIST digit classification.
