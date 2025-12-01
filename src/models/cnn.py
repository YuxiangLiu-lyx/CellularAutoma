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


class MinimalGameOfLifeCNN(nn.Module):
    """
    Minimal CNN that explicitly extracts center value and neighbor count.
    
    Architecture:
      Layer 1: Conv2d(1 → 2, 3x3) - Extract center + neighbor info
      Layer 2: Conv2d(2 → hidden, 1x1) - Combine information
      Layer 3: Conv2d(hidden → 1, 1x1) - Map to output
    
    This matches the Game of Life rule structure:
      - Know center value (0 or 1)
      - Count alive neighbors (0-8)
      - Apply decision rule
    """
    
    def __init__(self, hidden_channels=8, padding_mode='circular'):
        """
        Initialize minimal CNN model.
        
        Args:
            hidden_channels: Number of hidden channels in middle layer
            padding_mode: 'circular' for periodic boundaries
        """
        super(MinimalGameOfLifeCNN, self).__init__()
        
        self.extract = nn.Conv2d(1, 2, kernel_size=3, padding=1, 
                                 padding_mode=padding_mode)
        self.relu1 = nn.ReLU()
        self.combine = nn.Conv2d(2, hidden_channels, kernel_size=1)
        self.relu2 = nn.ReLU()
        self.output = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input state (batch, 1, H, W)
            
        Returns:
            Predicted next state (batch, 1, H, W)
        """
        x = self.extract(x)
        x = self.relu1(x)
        x = self.combine(x)
        x = self.relu2(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x


class UltraMinimalGameOfLifeCNN(nn.Module):
    """
    Ultra-minimal CNN with just 2 layers.
    
    Architecture:
      Layer 1: Conv2d(1 → 2, 3x3) - Extract center + neighbor count
      Layer 2: Conv2d(2 → 1, 1x1) - Direct rule mapping
    
    Total parameters: 1*2*9 + 2 + 2*1 + 1 = 21 parameters
    """
    
    def __init__(self, padding_mode='circular'):
        """
        Initialize ultra-minimal CNN model.
        
        Args:
            padding_mode: 'circular' for periodic boundaries
        """
        super(UltraMinimalGameOfLifeCNN, self).__init__()
        
        self.extract = nn.Conv2d(1, 2, kernel_size=3, padding=1, 
                                 padding_mode=padding_mode)
        self.relu = nn.ReLU()
        self.decide = nn.Conv2d(2, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input state (batch, 1, H, W)
            
        Returns:
            Predicted next state (batch, 1, H, W)
        """
        x = self.extract(x)
        x = self.relu(x)
        x = self.decide(x)
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
    print("Testing CNN models...")
    print("=" * 80)
    
    test_input = torch.randn(4, 1, 32, 32)
    
    models = [
        ("Ultra-Minimal (2 channels)", UltraMinimalGameOfLifeCNN()),
        ("Minimal (h=4)", MinimalGameOfLifeCNN(hidden_channels=4)),
        ("Minimal (h=8)", MinimalGameOfLifeCNN(hidden_channels=8)),
        ("Standard (h=8)", GameOfLifeCNN(hidden_channels=8)),
        ("Standard (h=16)", GameOfLifeCNN(hidden_channels=16)),
        ("Deep (h=32, L=2)", DeepGameOfLifeCNN(hidden_channels=32, num_layers=2)),
    ]
    
    print(f"\n{'Model':<30} {'Parameters':<15} {'Output Range':<20}")
    print("-" * 80)
    
    for name, model in models:
        params = count_parameters(model)
        output = model(test_input)
        print(f"{name:<30} {params:<15,} [{output.min():.3f}, {output.max():.3f}]")
    
    print("\n" + "=" * 80)
    print("Architecture Comparison")
    print("=" * 80)
    print("Ultra-Minimal: 1→2(3x3) → 2→1(1x1)")
    print("  - 2 channels to extract center + neighbors")
    print("  - Direct mapping to output")
    print("  - Parameters: ~21")
    print("\nMinimal: 1→2(3x3) → 2→h(1x1) → h→1(1x1)")
    print("  - 2 channels for center + neighbors")
    print("  - Hidden layer to combine information")
    print("  - Parameters: ~20 + 3h")
    print("\nStandard: 1→h(3x3) → h→1(1x1)")
    print("  - h channels to extract features")
    print("  - Direct mapping to output")
    print("  - Parameters: ~11h")

