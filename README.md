# Basic DNN MNIST Classification

A PyTorch implementation of a Deep Neural Network for MNIST digit classification with automated CI/CD pipeline.

## 🚀 Automated Testing

[![Run Model](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME/actions/workflows/run_model.yml/badge.svg)](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME/actions/workflows/run_model.yml)

**Click the badge above to see live training logs and test results!**

## 📊 Model Analysis

| Requirement | Status | Details |
|-------------|--------|---------|
| **Total Parameter Count** | ✅ **16,570 parameters** | Lightweight model with efficient parameter usage |
| **Batch Normalization** | ✅ **5 layers** | Applied after each convolutional layer for stable training |
| **Dropout** | ✅ **1 layer** | Dropout rate of 0.05 applied before final classification |
| **Fully Connected Layer** | ✅ **2 layers** | Two FC layers: 588→16→10 (no GAP used) |

## 🏗️ Model Architecture

```
Input: 28×28×1 (MNIST images)
├── Conv Block 1 (3 layers)
│   ├── Conv2d(1→8) + BatchNorm + ReLU
│   ├── Conv2d(8→12) + BatchNorm + ReLU  
│   └── Conv2d(12→12) + BatchNorm + ReLU
├── MaxPool2d(2×2) + 1×1 Conv (12→8)
├── Conv Block 2 (2 layers)
│   ├── Conv2d(8→12) + BatchNorm + ReLU
│   └── Conv2d(12→12) + BatchNorm + ReLU
├── MaxPool2d(2×2) + 1×1 Conv (12→8)
├── Conv Block 3 (2 layers)
│   ├── Conv2d(8→12) + BatchNorm + ReLU
│   └── Conv2d(12→12)
└── Classifier
    ├── Flatten (12×7×7 = 588)
    ├── Linear(588→16) + ReLU
    ├── Dropout(0.05)
    └── Linear(16→10) + LogSoftmax
```

## 🔧 Local Setup

### Prerequisites
```bash
pip install torch torchvision matplotlib numpy
```

### Run Locally
```bash
# Check parameter count
python -c "
import torch
import torch.nn as nn
import torch.nn.functional as F
# [Model class definition from your code]
model = Net()
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total parameters: {total_params}')
"

# Run full training and testing
python basic_dnn_mnist.py
```

## 📈 CI/CD Pipeline

This repository includes an automated GitHub Actions workflow that:

1. **Runs on every push** to main/master branch
2. **Installs dependencies** automatically
3. **Calculates parameter count** and displays model analysis
4. **Runs full training** for 20 epochs
5. **Shows test accuracy** results
6. **Uploads results** as artifacts

### View Live Results
- Click the **"Run Model"** badge above
- Go to **Actions** tab in your repository
- Click on the latest workflow run
- View the **logs** to see training progress and test accuracy

## 🎯 Expected Results

- **Total Parameters**: 16,570
- **Training Accuracy**: ~98%+ after 20 epochs
- **Test Accuracy**: ~98%+ on MNIST test set
- **Training Time**: ~2-3 minutes on GitHub Actions

## 📁 Repository Structure

```
├── basic_dnn_mnist.py          # Main training script
├── .github/workflows/          # CI/CD pipeline
│   └── run_model.yml          # GitHub Actions workflow
├── README.md                   # This file
└── requirements.txt            # Dependencies (optional)
```

## 🔗 GitHub Actions Features

- ✅ **Automatic execution** on code push
- ✅ **Parameter count analysis**
- ✅ **Full training and testing**
- ✅ **Live logs** showing progress
- ✅ **Test accuracy results**
- ✅ **Artifact upload** for results

## 📝 Usage

1. **Push your code** to GitHub
2. **Check Actions tab** for automatic execution
3. **View live logs** to see training progress
4. **Download artifacts** for detailed results

---

*This repository demonstrates a complete CI/CD pipeline for machine learning model training and testing.*
