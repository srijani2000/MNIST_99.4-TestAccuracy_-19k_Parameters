import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from assignment6 import Net

def count_parameters(model):
    """Count and format the total number of parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total_params': total_params,
        'trainable_params': trainable_params
    }

def test_parameter_count(capsys):
    model = Net()
    params = count_parameters(model)
    
    # Print parameter count for GitHub Actions log
    print("\n" + "="*50)
    print(f"Model Parameter Count Summary:")
    print(f"Total Parameters: {params['total_params']:,}")
    print(f"Trainable Parameters: {params['trainable_params']:,}")
    print("="*50 + "\n")
    
    assert params['total_params'] < 20000, f"Model has {params['total_params']:,} parameters, which exceeds the limit of 20,000"

def test_batch_norm_usage():
    model = Net()
    has_batch_norm = any(isinstance(module, torch.nn.BatchNorm2d) for module in model.modules())
    assert has_batch_norm, "Model should use Batch Normalization"

def test_dropout_usage():
    model = Net()
    has_dropout = any(isinstance(module, torch.nn.Dropout) for module in model.modules())
    assert has_dropout, "Model should use Dropout"

def test_gap_or_fc():
    model = Net()
    has_gap = any(isinstance(module, torch.nn.AvgPool2d) for module in model.modules())
    has_fc = any(isinstance(module, torch.nn.Linear) for module in model.modules())
    assert has_gap or has_fc, "Model should use either Global Average Pooling or Fully Connected Layer"