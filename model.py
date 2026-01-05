"""
CNN Model for Brain Tumor Detection
Convolutional Neural Network with multiple layers for binary classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BrainTumorCNN(nn.Module):
    """
    Convolutional Neural Network for Brain Tumor Detection
    
    Architecture:
    - 3 Convolutional blocks (Conv2d + BatchNorm + ReLU + MaxPool)
    - Dropout for regularization
    - Fully connected layers for classification
    """
    
    def __init__(self, num_classes=2, dropout_rate=0.5):
        """
        Args:
            num_classes: Number of output classes (default: 2 for yes/no)
            dropout_rate: Dropout probability for regularization
        """
        super(BrainTumorCNN, self).__init__()
        
        # First Convolutional Block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second Convolutional Block
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Third Convolutional Block
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fourth Convolutional Block
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Fully Connected Layers
        # Input size calculation: 256 * (image_size/16)^2
        # Assuming input image size is 224x224: 256 * 14 * 14 = 50176
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # First block
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Second block
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Third block
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Fourth block
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x


def get_model(num_classes=2, dropout_rate=0.5, device='cpu'):
    """
    Helper function to create and initialize the model
    
    Args:
        num_classes: Number of output classes
        dropout_rate: Dropout probability
        device: Device to load model on ('cpu' or 'cuda')
        
    Returns:
        Initialized model
    """
    model = BrainTumorCNN(num_classes=num_classes, dropout_rate=dropout_rate)
    model = model.to(device)
    return model


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = get_model(num_classes=2, device=device)
    print("\nModel Architecture:")
    print(model)
    
    # Test with a dummy input
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

