#!/usr/bin/env python3
"""
Interactive MNIST Digit Recognition App
========================================

This application provides a graphical user interface for testing trained MNIST models.
Users can select a saved model, draw a digit on a canvas, and see the model's predictions.

Features:
- Model selection from saved models
- Drawing canvas for digit input
- Real-time prediction with confidence scores
- Clear and predict buttons

Author: Ridwan Halim (ridwaanhall)
Date: June 04, 2025
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import os
import pickle
from PIL import Image, ImageDraw
import glob
import sys

# Add the parent directory to the Python path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our neural network components
from src.models.neural_network import NeuralNetwork


class DigitRecognitionApp:
    """Interactive GUI application for digit recognition using trained MNIST models."""
    
    def __init__(self, root):
        """Initialize the application."""
        self.root = root
        self.root.title("MNIST Digit Recognition - Neural Network from Scratch")
        self.root.geometry("800x700")
        self.root.resizable(True, True)
        
        # Application state
        self.model = None
        self.model_info = None
        self.canvas_size = 280  # 28x28 scaled up by 10
        self.drawing = False
        self.last_x = None
        self.last_y = None
        
        # Create PIL image for drawing (28x28 grayscale)
        self.pil_image = Image.new('L', (28, 28), 0)  # Black background
        self.pil_draw = ImageDraw.Draw(self.pil_image)
        
        self.setup_ui()
        self.load_available_models()
        
    def setup_ui(self):
        """Setup the user interface."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="MNIST Digit Recognition", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Model selection frame
        model_frame = ttk.LabelFrame(main_frame, text="Model Selection", padding="10")
        model_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        model_frame.columnconfigure(1, weight=1)
        
        ttk.Label(model_frame, text="Select Model:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, 
                                       state="readonly", width=50)
        self.model_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        self.model_combo.bind('<<ComboboxSelected>>', self.on_model_selected)
        
        self.load_model_btn = ttk.Button(model_frame, text="Load Model", 
                                        command=self.load_selected_model)
        self.load_model_btn.grid(row=0, column=2)
        
        # Model info frame
        self.info_frame = ttk.LabelFrame(main_frame, text="Model Information", padding="10")
        self.info_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.info_text = tk.Text(self.info_frame, height=4, width=70, state=tk.DISABLED)
        self.info_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Drawing and prediction frame
        content_frame = ttk.Frame(main_frame)
        content_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        content_frame.columnconfigure(0, weight=1)
        content_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)
        
        # Drawing canvas frame
        canvas_frame = ttk.LabelFrame(content_frame, text="Draw a Digit (0-9)", padding="10")
        canvas_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
        # Canvas for drawing
        self.canvas = tk.Canvas(canvas_frame, width=self.canvas_size, height=self.canvas_size, 
                               bg='black', cursor='pencil')
        self.canvas.grid(row=0, column=0, pady=(0, 10))
        
        # Canvas controls
        canvas_controls = ttk.Frame(canvas_frame)
        canvas_controls.grid(row=1, column=0)
        
        self.clear_btn = ttk.Button(canvas_controls, text="Clear", command=self.clear_canvas)
        self.clear_btn.grid(row=0, column=0, padx=(0, 10))
        
        self.predict_btn = ttk.Button(canvas_controls, text="Predict", 
                                     command=self.predict_digit, state=tk.DISABLED)
        self.predict_btn.grid(row=0, column=1)
        
        # Prediction results frame
        results_frame = ttk.LabelFrame(content_frame, text="Prediction Results", padding="10")
        results_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        results_frame.columnconfigure(0, weight=1)
        
        # Predicted digit display
        self.prediction_label = ttk.Label(results_frame, text="Predicted Digit: -", 
                                         font=("Arial", 14, "bold"))
        self.prediction_label.grid(row=0, column=0, pady=(0, 10))
        
        self.confidence_label = ttk.Label(results_frame, text="Confidence: -", 
                                         font=("Arial", 12))
        self.confidence_label.grid(row=1, column=0, pady=(0, 20))
        
        # Probability distribution
        ttk.Label(results_frame, text="Probability Distribution:", 
                 font=("Arial", 11, "bold")).grid(row=2, column=0, sticky=tk.W)
        
        # Frame for probability bars
        self.prob_frame = ttk.Frame(results_frame)
        self.prob_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        self.prob_frame.columnconfigure(1, weight=1)
        
        # Create probability display widgets
        self.prob_labels = []
        self.prob_bars = []
        for i in range(10):
            # Digit label
            digit_label = ttk.Label(self.prob_frame, text=f"{i}:", width=3)
            digit_label.grid(row=i, column=0, sticky=tk.W, pady=1)
            
            # Progress bar for probability
            prob_bar = ttk.Progressbar(self.prob_frame, length=150, mode='determinate')
            prob_bar.grid(row=i, column=1, sticky=(tk.W, tk.E), padx=(5, 10), pady=1)
            
            # Percentage label
            percent_label = ttk.Label(self.prob_frame, text="0.00%", width=8)
            percent_label.grid(row=i, column=2, sticky=tk.E, pady=1)
            
            self.prob_bars.append(prob_bar)
            self.prob_labels.append(percent_label)
        
        # Bind canvas events
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Please select and load a model to begin")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
    
    def load_available_models(self):
        """Load list of available trained models."""
        models_dir = "models"
        if not os.path.exists(models_dir):
            self.status_var.set("No models directory found")
            return
        
        # Find all .pkl model files
        model_files = glob.glob(os.path.join(models_dir, "mnist_model_*.pkl"))
        
        if not model_files:
            self.status_var.set("No trained models found in models directory")
            return
        
        # Sort by modification time (newest first)
        model_files.sort(key=os.path.getmtime, reverse=True)
        
        # Extract just the filename for display
        model_names = [os.path.basename(f) for f in model_files]
        
        self.model_combo['values'] = model_names
        if model_names:
            self.model_combo.set(model_names[0])  # Select newest model by default
            self.status_var.set(f"Found {len(model_names)} trained models")
    
    def on_model_selected(self, event):
        """Handle model selection from dropdown."""
        self.status_var.set(f"Selected: {self.model_var.get()}")
    
    def load_selected_model(self):
        """Load the selected model."""
        model_name = self.model_var.get()
        if not model_name:
            messagebox.showwarning("Warning", "Please select a model first")
            return
        
        model_path = os.path.join("models", model_name)
        
        try:
            self.status_var.set("Loading model...")
            self.root.update()
            
            # Load the model
            self.model = NeuralNetwork()
            self.model.load_model(model_path)
            
            # Try to load corresponding results file for model info
            results_name = model_name.replace("mnist_model_", "results_")
            results_path = os.path.join("models", results_name)
            
            if os.path.exists(results_path):
                with open(results_path, 'rb') as f:
                    self.model_info = pickle.load(f)
            else:
                self.model_info = None
            
            # Update UI
            self.update_model_info()
            self.predict_btn.config(state=tk.NORMAL)
            self.status_var.set(f"Model loaded successfully: {model_name}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.status_var.set("Failed to load model")
    
    def update_model_info(self):
        """Update the model information display."""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        
        if self.model_info:
            info = self.model_info
            config = info.get('config', {})
            metrics = info.get('final_metrics', {})
            
            info_text = f"Architecture: {config.get('architecture', 'Unknown')}\n"
            info_text += f"Test Accuracy: {metrics.get('accuracy', 0):.4f} "
            info_text += f"({metrics.get('accuracy', 0)*100:.2f}%)\n"
            info_text += f"Training Epochs: {config.get('epochs', 'Unknown')}, "
            info_text += f"Batch Size: {config.get('batch_size', 'Unknown')}\n"
            info_text += f"Model Layers: {len(self.model.layers) if self.model else 'Unknown'}"
        else:
            info_text = f"Model loaded successfully\n"
            info_text += f"Layers: {len(self.model.layers) if self.model else 'Unknown'}\n"
            info_text += "No additional information available"
        
        self.info_text.insert(1.0, info_text)
        self.info_text.config(state=tk.DISABLED)
    
    def start_draw(self, event):
        """Start drawing on canvas."""
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y
    
    def draw(self, event):
        """Draw on canvas."""
        if self.drawing:
            # Draw on tkinter canvas
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y, 
                                   fill='white', width=12, capstyle=tk.ROUND, smooth=tk.TRUE)
            
            # Draw on PIL image (scaled down to 28x28)
            x1, y1 = self.last_x / 10, self.last_y / 10
            x2, y2 = event.x / 10, event.y / 10
            self.pil_draw.line([x1, y1, x2, y2], fill=255, width=2)
            
            self.last_x = event.x
            self.last_y = event.y
    
    def stop_draw(self, event):
        """Stop drawing."""
        self.drawing = False
    
    def clear_canvas(self):
        """Clear the drawing canvas."""
        self.canvas.delete("all")
        self.pil_image = Image.new('L', (28, 28), 0)
        self.pil_draw = ImageDraw.Draw(self.pil_image)
        
        # Reset prediction display
        self.prediction_label.config(text="Predicted Digit: -")
        self.confidence_label.config(text="Confidence: -")
        for i in range(10):
            self.prob_bars[i]['value'] = 0
            self.prob_labels[i].config(text="0.00%")
        
        self.status_var.set("Canvas cleared")
    
    def predict_digit(self):
        """Predict the drawn digit using the loaded model."""
        if not self.model:
            messagebox.showwarning("Warning", "Please load a model first")
            return
        
        try:
            # Convert PIL image to numpy array
            img_array = np.array(self.pil_image)
            
            # Check if image is blank
            if np.max(img_array) == 0:
                messagebox.showinfo("Info", "Please draw a digit first")
                return
            
            # Normalize and flatten the image
            img_normalized = img_array.astype(np.float32) / 255.0
            img_flattened = img_normalized.flatten().reshape(1, -1)
            
            # Make prediction
            self.status_var.set("Making prediction...")
            self.root.update()
            
            predictions = self.model.predict(img_flattened)
            probabilities = predictions[0]  # Get probabilities for the single image
            
            # Get predicted digit and confidence
            predicted_digit = np.argmax(probabilities)
            confidence = probabilities[predicted_digit]
            
            # Update prediction display
            self.prediction_label.config(text=f"Predicted Digit: {predicted_digit}")
            self.confidence_label.config(text=f"Confidence: {confidence:.2%}")
            
            # Update probability bars
            for i in range(10):
                prob_percent = probabilities[i] * 100
                self.prob_bars[i]['value'] = prob_percent
                self.prob_labels[i].config(text=f"{prob_percent:.2f}%")
            
            self.status_var.set(f"Prediction complete - Digit: {predicted_digit} ({confidence:.2%})")
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
            self.status_var.set("Prediction failed")


def main():
    """Main function to run the application."""
    try:
        # Create and run the application
        root = tk.Tk()
        app = DigitRecognitionApp(root)
        
        # Center the window
        root.update_idletasks()
        x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
        y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
        root.geometry(f"+{x}+{y}")
        
        print("Starting MNIST Digit Recognition App...")
        print("Draw a digit and click 'Predict' to test your trained model!")
        
        root.mainloop()
        
    except ImportError as e:
        print(f"Missing required dependency: {e}")
        print("Please install required packages:")
        print("pip install pillow")
    except Exception as e:
        print(f"Error starting application: {e}")


if __name__ == "__main__":
    main()
