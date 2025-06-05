# Neural Network from Scratch - Complete Usage Guide

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-Only-green.svg)](https://numpy.org)
[![Accuracy](https://img.shields.io/badge/MNIST%20Accuracy-96.71%25-brightgreen.svg)](/)

## 🎯 Professional MNIST Neural Network Implementation

This comprehensive guide covers all aspects of using our production-ready neural network implementation built entirely from scratch using only NumPy for MNIST digit classification, achieving **96.71% test accuracy**.

---

## 🚀 Quick Start Guide

### 1. Installation & Setup

```bash
# Clone the repository
git clone <repository-url>
cd nn-scratch

# Install dependencies
pip install -r requirements.txt

# Verify installation
python minimal_test.py
```

### 2. First Run (Recommended)

```bash
# Quick test with reduced dataset (fastest way to verify everything works)
python apps/main.py --quick_test

# Expected output: ~85-90% accuracy in 1-2 minutes
```

### 3. Full Training

```bash
# Complete training with default architecture
python apps/main.py

# Expected output: ~95% accuracy in 5-10 minutes
```

### 4. Interactive GUI Application

```bash
# Launch digit recognition interface
python apps/play_app.py

# Draw digits and test the model in real-time
```

---

## 📁 Complete Project Structure

```txt
nn-scratch/
├── 📁 src/                             # Core Implementation
│   ├── 📁 data/
│   │   ├── __init__.py
│   │   └── data_loader.py              # MNIST data pipeline with PyTorch integration
│   ├── 📁 models/
│   │   ├── __init__.py
│   │   ├── activations.py              # 5 activation functions (ReLU, Sigmoid, Tanh, Softmax, LeakyReLU)
│   │   ├── layers.py                   # Dense & Dropout layers with weight initialization
│   │   └── neural_network.py           # Main NeuralNetwork class with save/load
│   ├── 📁 training/
│   │   ├── __init__.py
│   │   ├── loss_functions.py           # 4 loss functions (CrossEntropy, MSE, BCE, Huber)
│   │   └── trainer.py                  # Advanced training with early stopping & scheduling
│   └── 📁 utils/
│       ├── __init__.py
│       ├── metrics.py                  # Comprehensive evaluation metrics
│       └── visualization.py            # Professional plotting utilities
├── 📁 Applications/
│   ├── main.py                         # Complete end-to-end pipeline
│   ├── train.py                        # Standalone training script
│   ├── test.py                         # Comprehensive evaluation script
│   ├── demo.py                         # Component demonstration
│   ├── play_app.py                     # Interactive GUI for digit recognition
│   ├── minimal_test.py                 # Quick functionality verification
│   ├── test_basic.py                   # Basic import and component test
│   ├── debug_test.py                   # Debug utilities
│   └── test_model_loading.py           # Model loading verification
├── 📁 data/                            # MNIST dataset storage
├── 📁 models/                          # Trained model storage with timestamps
├── 📁 logs/                            # Training logs, plots, and visualizations
├── 📄 requirements.txt                 # Project dependencies
├── 📄 README.md                        # Project overview and quick start
├── 📄 USAGE_GUIDE.md                   # This comprehensive guide
├── 📄 PROJECT_STATUS.md                # Complete project status
└── 📄 setup.bat / setup.ps1            # Environment setup scripts
```

---

## 🎮 Command Line Applications

### Main Pipeline (`main.py`)

The primary application that provides a complete end-to-end machine learning pipeline:

```bash
# Basic usage
python apps/main.py                         # Full training with default settings

# Architecture options
python apps/main.py --architecture simple   # Quick 2-layer network
python apps/main.py --architecture default  # Balanced 3-layer network (default)
python apps/main.py --architecture deep     # Deep 4-layer network for maximum accuracy

# Training parameters
python apps/main.py --epochs 100            # Custom epoch count
python apps/main.py --batch_size 64         # Custom batch size
python apps/main.py --learning_rate 0.01    # Custom learning rate

# Testing and debugging
python apps/main.py --quick_test            # Reduced dataset for quick testing
python apps/main.py --no_plots             # Skip visualization generation

# Combined options
python apps/main.py --architecture deep --epochs 50 --batch_size 128
```

### Standalone Training (`train.py`)

Focused training script with advanced features:

```bash
# Basic training
python apps/train.py

# Custom parameters
python apps/train.py --epochs 50 --batch_size 32 --learning_rate 0.001

# Advanced features
python apps/train.py --early_stopping --patience 10 --save_best
```

### Model Evaluation (`test.py`)

Comprehensive model testing and evaluation:

```bash
# Test latest model
python apps/test.py

# Test specific model
python apps/test.py --model_path models/mnist_model_20250604_231500.pkl

# Detailed evaluation with plots
python apps/test.py --detailed --save_plots
```

### Component Demo (`demo.py`)

Demonstrates individual components and their capabilities:

```bash
# Run component demonstrations
python apps/demo.py

# Shows: activation functions, loss functions, layer types, etc.
```

### Quick Testing Scripts

```bash
# Minimal functionality test (fastest verification)
python minimal_test.py

# Basic import and component test
python test_basic.py

# Debug utilities
python debug_test.py

# Model loading verification
python test_model_loading.py
```

---

## 🎮 Interactive GUI Application

### Digit Recognition App (`play_app.py`)

Launch an interactive Tkinter-based application for real-time digit recognition:

```bash
python apps/play_app.py
```

**Features:**

- ✅ **Drawing Canvas** - Draw digits with mouse
- ✅ **Real-time Prediction** - Instant classification with confidence scores
- ✅ **Model Selection** - Choose from available trained models
- ✅ **Clear/Reset** - Clear canvas for new drawings
- ✅ **Professional UI** - Clean, user-friendly interface

**Usage:**

1. Launch the application
2. Select a trained model from the dropdown
3. Draw a digit (0-9) on the canvas
4. View the prediction and confidence score
5. Clear and try again

---

## 🏗️ Model Architectures

### Simple Architecture (Quick Testing)

```txt
Input(784) → Dense(128, ReLU) → Dense(10, Softmax)

Performance: 85-90% accuracy in 10-20 epochs
Training Time: 1-2 minutes
Memory Usage: <500MB
Use Case: Rapid prototyping and testing
```

### Default Architecture (Balanced Performance)

```txt
Input(784) → Dense(256, ReLU) → Dropout(0.2) → Dense(128, ReLU) → Dropout(0.2) → Dense(10, Softmax)

Performance: 92-95% accuracy in 30-50 epochs  
Training Time: 3-5 minutes
Memory Usage: <800MB
Use Case: Standard production deployment
```

### Deep Architecture (Maximum Accuracy)

```txt
Input(784) → Dense(512, ReLU) → Dropout(0.3) → Dense(256, ReLU) → Dropout(0.3) → 
         Dense(128, ReLU) → Dropout(0.2) → Dense(10, Softmax)

Performance: 95-97% accuracy in 50-100 epochs
Training Time: 8-15 minutes  
Memory Usage: <1GB
Use Case: Maximum accuracy requirements
```

---

## 💻 Programmatic Interface

### Basic Model Creation

```python
from src.models.neural_network import NeuralNetwork
from src.models.layers import DenseLayer, DropoutLayer
from src.models.activations import ReLU, Softmax

# Create model
model = NeuralNetwork()

# Add layers
model.add_layer(DenseLayer(784, 256, activation=ReLU()))
model.add_layer(DropoutLayer(0.2))
model.add_layer(DenseLayer(256, 128, activation=ReLU()))
model.add_layer(DenseLayer(128, 10, activation=Softmax()))

print(f"Model created with {len(model.layers)} layers")
```

### Advanced Training

```python
from src.training.trainer import Trainer
from src.training.loss_functions import CrossEntropyLoss
from src.data.data_loader import load_mnist_data

# Load data
X_train, y_train, X_test, y_test = load_mnist_data()

# Create trainer with advanced features
trainer = Trainer(
    model=model,
    loss_function=CrossEntropyLoss(),
    learning_rate=0.001,
    momentum=0.9,
    patience=15,
    save_best=True,
    min_improvement=0.001
)

# Train with validation split
history = trainer.train(
    X_train, y_train,
    validation_split=0.1,
    epochs=50,
    batch_size=64,
    verbose=True
)
```

### Model Evaluation

```python
from src.utils.metrics import evaluate_model, plot_confusion_matrix
from src.utils.visualization import plot_training_history, plot_sample_predictions

# Evaluate model
accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Generate visualizations
plot_training_history(history)
plot_confusion_matrix(model, X_test, y_test)
plot_sample_predictions(model, X_test, y_test, num_samples=10)
```

### Model Persistence

```python
# Save model
model.save('models/my_custom_model.pkl')

# Load model
from src.models.neural_network import NeuralNetwork
loaded_model = NeuralNetwork.load('models/my_custom_model.pkl')

# Verify loaded model
predictions = loaded_model.predict(X_test[:10])
print(f"Loaded model predictions: {predictions}")
```

---

## 🔧 Configuration Options

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 30 | Number of training epochs |
| `batch_size` | 32 | Mini-batch size for training |
| `learning_rate` | 0.001 | Learning rate for optimization |
| `momentum` | 0.9 | Momentum for SGD optimizer |
| `validation_split` | 0.1 | Fraction of data for validation |
| `patience` | 10 | Early stopping patience |
| `min_improvement` | 0.001 | Minimum improvement for early stopping |

### Architecture Options

| Architecture | Layers | Parameters | Training Time | Expected Accuracy |
|--------------|--------|------------|---------------|-------------------|
| `simple` | 2 | ~100K | 1-2 min | 85-90% |
| `default` | 3 | ~200K | 3-5 min | 92-95% |
| `deep` | 4 | ~400K | 8-15 min | 95-97% |

### Activation Functions

```python
from src.models.activations import ReLU, Sigmoid, Tanh, Softmax, LeakyReLU

# Available activations
ReLU()          # Rectified Linear Unit (default for hidden layers)
Sigmoid()       # Logistic activation
Tanh()          # Hyperbolic tangent
Softmax()       # Probability distribution (output layer)
LeakyReLU(alpha=0.01)  # Leaky ReLU with configurable slope
```

### Loss Functions

```python
from src.training.loss_functions import CrossEntropyLoss, MeanSquaredError, BinaryCrossEntropy, HuberLoss

# Available loss functions
CrossEntropyLoss()    # Multiclass classification (default)
MeanSquaredError()    # Regression tasks
BinaryCrossEntropy()  # Binary classification
HuberLoss(delta=1.0)  # Robust loss function
```

---

## 📊 Output Files and Logs

### Generated Files

After training, the following files are created:

```txt
models/
├── mnist_model_YYYYMMDD_HHMMSS.pkl    # Trained model
└── results_YYYYMMDD_HHMMSS.pkl        # Training results

logs/
├── main_YYYYMMDD_HHMMSS.log           # Training log
├── training_history_YYYYMMDD_HHMMSS.png  # Loss/accuracy curves
├── confusion_matrix_YYYYMMDD_HHMMSS.png  # Confusion matrix heatmap
└── sample_predictions_YYYYMMDD_HHMMSS.png # Sample prediction visualizations
```

### Log File Contents

Training logs contain:

- Model architecture details
- Training parameters
- Epoch-by-epoch progress
- Performance metrics
- Error messages and warnings

---

## 🔍 Troubleshooting

### Common Issues

#### 1. Import Errors

```bash
# Ensure you're in the project root directory
cd nn-scratch

# Check Python path
python -c "import sys; print(sys.path)"

# Install missing dependencies
pip install -r requirements.txt
```

#### 2. MNIST Download Issues

```bash
# Clear existing data and re-download
rm -rf data/MNIST
python -c "from src.data.data_loader import load_mnist_data; load_mnist_data()"
```

#### 3. Memory Issues

```bash
# Use smaller batch size
python apps/main.py --batch_size 16

# Use simple architecture
python apps/main.py --architecture simple
```

#### 4. Training Too Slow

```bash
# Use quick test mode
python apps/main.py --quick_test

# Reduce epochs
python apps/main.py --epochs 10
```

### Performance Optimization

**Speed Up Training:**

- Use larger batch sizes (64-128) if memory allows
- Reduce image resolution (already optimized for MNIST)
- Use simple architecture for quick iterations
- Enable early stopping to avoid overtraining

**Improve Accuracy:**

- Use deep architecture
- Increase training epochs
- Add dropout for regularization
- Experiment with learning rate scheduling

---

## 📚 Advanced Usage Examples

### Custom Architecture

```python
from src.models.neural_network import NeuralNetwork
from src.models.layers import DenseLayer, DropoutLayer
from src.models.activations import ReLU, Tanh, Softmax

# Create a custom 5-layer network
model = NeuralNetwork()
model.add_layer(DenseLayer(784, 512, activation=ReLU()))
model.add_layer(DropoutLayer(0.3))
model.add_layer(DenseLayer(512, 256, activation=ReLU()))
model.add_layer(DropoutLayer(0.2))
model.add_layer(DenseLayer(256, 128, activation=Tanh()))
model.add_layer(DropoutLayer(0.1))
model.add_layer(DenseLayer(128, 64, activation=ReLU()))
model.add_layer(DenseLayer(64, 10, activation=Softmax()))
```

### Learning Rate Scheduling

```python
from src.training.trainer import Trainer

# Create trainer with learning rate scheduling
trainer = Trainer(
    model=model,
    learning_rate=0.01,
    lr_schedule='step',      # 'step', 'exp', or 'plateau'
    lr_decay=0.5,           # Decay factor
    lr_patience=5           # Steps/epochs before decay
)
```

### Batch Processing

```python
# Process large datasets in batches
def process_large_dataset(model, X, batch_size=1000):
    predictions = []
    for i in range(0, len(X), batch_size):
        batch = X[i:i+batch_size]
        batch_pred = model.predict(batch)
        predictions.extend(batch_pred)
    return np.array(predictions)
```

---

## 🎯 Best Practices

### Training Recommendations

1. **Start Small**: Use `--quick_test` for initial verification
2. **Monitor Progress**: Check training logs and plots
3. **Use Early Stopping**: Prevent overfitting with patience parameter
4. **Save Regularly**: Models are automatically saved with timestamps
5. **Validate Results**: Always test on unseen data

### Development Workflow

1. **Test Components**: Run `minimal_test.py` first
2. **Quick Iteration**: Use simple architecture for experimentation
3. **Full Training**: Use deep architecture for final models
4. **Evaluate Thoroughly**: Use comprehensive evaluation scripts
5. **Document Changes**: Update logs and documentation

### Production Deployment

1. **Model Selection**: Choose best performing saved model
2. **Validation**: Test thoroughly on holdout data  
3. **Monitoring**: Track performance metrics
4. **Backup**: Keep multiple model versions
5. **Documentation**: Maintain usage and performance records

---

## 🏆 Expected Results

### Performance Benchmarks

| Architecture | Accuracy | Training Time | Model Size | Memory Usage |
|--------------|----------|---------------|------------|--------------|
| Simple | 85-90% | 1-2 min | ~1MB | <500MB |
| Default | 92-95% | 3-5 min | ~2MB | <800MB |
| Deep | 95-97% | 8-15 min | ~5MB | <1GB |

### Achieved Results (Verified)

- **Maximum Accuracy**: 96.71% on MNIST test set
- **Training Speed**: 5-10 minutes for standard architecture
- **Memory Efficiency**: <1GB RAM for complete training
- **Model Size**: 2-5MB saved model files
- **Convergence**: Stable training in 30-50 epochs

---

This comprehensive guide covers all aspects of using the neural network implementation. For additional information, see:

- **README.md** - Project overview and quick start
- **PROJECT_STATUS.md** - Detailed implementation status
- **Source Code** - Comprehensive inline documentation

**Happy learning and experimenting!** 🚀
