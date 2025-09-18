import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 12, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(12)
        self.conv3 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(2,2)
        self.antman1 = nn.Conv2d(12, 8, kernel_size=1)
        self.conv4 = nn.Conv2d(8, 12, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(12)
        self.conv5 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(12)
        self.pool2 = nn.MaxPool2d(2,2)
        self.antman2 = nn.Conv2d(12, 8, kernel_size=1)
        self.conv5addon = nn.Conv2d(8, 12, kernel_size=3, padding=1)
        self.bn5addon = nn.BatchNorm2d(12)
        self.conv6 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(12*7*7, 16)
        self.dropout = nn.Dropout(0.05)
        self.fc2 = nn.Linear(16, 10)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.antman1(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = self.antman2(x)
        x = F.relu(self.bn5addon(self.conv5addon(x)))
        x = self.conv6(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class TestModel:
    def test_model_creation(self):
        """Test that model can be created successfully"""
        model = Net()
        assert isinstance(model, nn.Module)
    
    def test_model_forward_pass(self):
        """Test that model forward pass works with correct input shape"""
        model = Net()
        model.eval()
        
        # Test with batch of 1
        x = torch.randn(1, 1, 28, 28)
        output = model(x)
        
        assert output.shape == (1, 10)
        assert torch.allclose(torch.exp(output).sum(dim=1), torch.ones(1), atol=1e-6)
    
    def test_model_forward_pass_batch(self):
        """Test that model forward pass works with batch input"""
        model = Net()
        model.eval()
        
        # Test with batch of 4
        x = torch.randn(4, 1, 28, 28)
        output = model(x)
        
        assert output.shape == (4, 10)
        assert torch.allclose(torch.exp(output).sum(dim=1), torch.ones(4), atol=1e-6)
    
    def test_model_parameter_count(self):
        """Test that model has correct number of parameters"""
        model = Net()
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params == 16570
        assert trainable_params == 16570
        assert total_params == trainable_params
    
    def test_model_has_batch_normalization(self):
        """Test that model has batch normalization layers"""
        model = Net()
        bn_layers = [module for module in model.modules() if isinstance(module, nn.BatchNorm2d)]
        assert len(bn_layers) == 5
    
    def test_model_has_dropout(self):
        """Test that model has dropout layers"""
        model = Net()
        dropout_layers = [module for module in model.modules() if isinstance(module, nn.Dropout)]
        assert len(dropout_layers) == 1
    
    def test_model_has_fully_connected_layers(self):
        """Test that model has fully connected layers"""
        model = Net()
        fc_layers = [module for module in model.modules() if isinstance(module, nn.Linear)]
        assert len(fc_layers) == 2
    
    def test_model_no_gap(self):
        """Test that model does not use Global Average Pooling"""
        model = Net()
        gap_layers = [module for module in model.modules() if isinstance(module, nn.AdaptiveAvgPool2d)]
        assert len(gap_layers) == 0
    
    def test_model_output_shape(self):
        """Test that model output has correct shape for MNIST"""
        model = Net()
        model.eval()
        
        x = torch.randn(1, 1, 28, 28)
        output = model(x)
        
        # Should output log probabilities for 10 classes
        assert output.shape == (1, 10)
        assert output.dtype == torch.float32
    
    def test_model_gradient_flow(self):
        """Test that model can compute gradients"""
        model = Net()
        model.train()
        
        x = torch.randn(1, 1, 28, 28)
        target = torch.tensor([5])
        
        output = model(x)
        loss = F.nll_loss(output, target)
        loss.backward()
        
        # Check that gradients are computed
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
