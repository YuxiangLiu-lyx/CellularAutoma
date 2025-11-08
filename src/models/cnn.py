"""
CNN model for Game of Life prediction using 3x3 convolutions
"""
import torch
import torch.nn as nn


class GameOfLifeCNN(nn.Module):
    """
    CNN for predicting next state in Game of Life.
    
    Uses 3x3 convolutions to capture local neighborhood information,
    matching the Game of Life rule structure.
    """
    
    def __init__(self, hidden_channels=16, padding_mode='circular'):
        """
        Initialize CNN model.
        
        Args:
            hidden_channels: Number of hidden channels in convolutional layers
            padding_mode: 'circular' for periodic boundaries, 'zeros' for fixed boundaries
        """
        super(GameOfLifeCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1, 
                               padding_mode=padding_mode)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input state (batch, 1, H, W)
            
        Returns:
            Predicted next state (batch, 1, H, W)
        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return x


class DeepGameOfLifeCNN(nn.Module):
    """
    Deeper CNN variant with multiple 3x3 convolutional layers.
    """
    
    def __init__(self, hidden_channels=32, num_layers=2, padding_mode='circular'):
        """
        Initialize deep CNN model.
        
        Args:
            hidden_channels: Number of hidden channels
            num_layers: Number of 3x3 convolutional layers
            padding_mode: 'circular' for periodic boundaries, 'zeros' for fixed boundaries
        """
        super(DeepGameOfLifeCNN, self).__init__()
        
        layers = []
        
        # First layer
        layers.append(nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1,
                               padding_mode=padding_mode))
        layers.append(nn.ReLU())
        
        # Middle layers
        for _ in range(num_layers - 1):
            layers.append(nn.Conv2d(hidden_channels, hidden_channels, 
                                   kernel_size=3, padding=1, padding_mode=padding_mode))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Conv2d(hidden_channels, 1, kernel_size=1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input state (batch, 1, H, W)
            
        Returns:
            Predicted next state (batch, 1, H, W)
        """
        return self.network(x)


def count_parameters(model):
    """
    Count trainable parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test models
    print("Testing CNN models...")
    print("=" * 60)
    
    # Test simple model
    model_simple = GameOfLifeCNN(hidden_channels=16)
    print(f"\nSimple CNN:")
    print(f"  Parameters: {count_parameters(model_simple):,}")
    
    test_input = torch.randn(4, 1, 32, 32)
    output = model_simple(test_input)
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Test deep model
    model_deep = DeepGameOfLifeCNN(hidden_channels=32, num_layers=2)
    print(f"\nDeep CNN:")
    print(f"  Parameters: {count_parameters(model_deep):,}")
    
    output_deep = model_deep(test_input)
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {output_deep.shape}")
    print(f"  Output range: [{output_deep.min():.3f}, {output_deep.max():.3f}]")
    
    print("\n" + "=" * 60)
    print("Model tests passed!")

