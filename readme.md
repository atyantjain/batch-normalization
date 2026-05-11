# 🧠 Batch Normalization Practice

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive exploration of **Batch Normalization** implementation using both high-level frameworks (PyTorch) and low-level manual calculations (NumPy). This repository serves as a practical guide to understanding how normalizing activations can stabilize and accelerate neural network training.

---

## 🌟 Key Features

- **PyTorch Implementation**: A complete training pipeline using `nn.BatchNorm1d` for a binary classification task.
- **Manual Calculation**: Step-by-step NumPy implementation in `scratch.py` to demystify the mathematics behind the algorithm.
- **Data Pipeline**: Integration with Excel datasets using `pandas` and `scikit-learn` for preprocessing.
- **Visualization**: Real-time loss tracking using `matplotlib` to observe convergence.

---

## 📂 Project Structure

```text
├── Batch normalization.py  # Main PyTorch training script
├── scratch.py               # Manual NumPy implementation of BatchNorm
├── data.xlsx               # Dataset containing study hours and sleep patterns
├── llm.ipynb               # Experimental notebook for model development
└── readme.md               # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

Ensure you have the following installed:
- Python 3.8+
- PyTorch
- Pandas
- Scikit-learn
- Matplotlib
- Openpyxl (for Excel support)

### Installation

```bash
# Clone the repository
git clone https://github.com/atyantjain/batch-normalization.git

# Install dependencies
pip install torch pandas scikit-learn matplotlib openpyxl
```

---

## 🛠️ Usage

### 1. Training with PyTorch
Run the main script to train a model with Batch Normalization on the provided dataset:

```bash
python "Batch normalization.py"
```

The script will:
1. Load and preprocess the student performance data.
2. Initialize a neural network with a `BatchNorm1d` layer.
3. Train for 7 epochs using Adam optimizer and Cross-Entropy loss.
4. Display a plot of the training loss.

### 2. Understanding the Math
Run the scratch script to see the underlying mechanics:

```bash
python scratch.py
```

---

## 📐 How Batch Normalization Works

Batch Normalization follows these core steps for each batch:

1.  **Calculate Mean**: $\mu_B = \frac{1}{m} \sum_{i=1}^m x_i$
2.  **Calculate Variance**: $\sigma_B^2 = \frac{1}{m} \sum_{i=1}^m (x_i - \mu_B)^2$
3.  **Normalize**: $\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$
4.  **Scale and Shift**: $y_i = \gamma \hat{x}_i + \beta$

Where $\gamma$ and $\beta$ are learnable parameters that allow the network to undo the normalization if it helps the model learn better.

---

## 📊 Results

The model tracks the average loss per epoch. With Batch Normalization, you should observe a smooth decline in loss even with a small number of epochs, demonstrating the stability provided by the layer.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

<div align="center">
  <sub>Built with ❤️ by Atyant Jain</sub>
</div>
