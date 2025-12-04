"""CNN architectures for Game of Life prediction."""
import torch
import torch.nn as nn


class GameOfLifeCNN(nn.Module):
    """Baseline CNN using 3x3 convolutions to model local rules."""

    def __init__(self, hidden_channels=16, padding_mode='circular'):
        super(GameOfLifeCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1, 
                               padding_mode=padding_mode)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return x


class MinimalGameOfLifeCNN(nn.Module):
    """Compact CNN that separates center value and neighbor count."""

    def __init__(self, hidden_channels=8, padding_mode='circular'):
        super(MinimalGameOfLifeCNN, self).__init__()
        self.extract = nn.Conv2d(1, 2, kernel_size=3, padding=1, 
                                 padding_mode=padding_mode)
        self.relu1 = nn.ReLU()
        self.combine = nn.Conv2d(2, hidden_channels, kernel_size=1)
        self.relu2 = nn.ReLU()
        self.output = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.extract(x)
        x = self.relu1(x)
        x = self.combine(x)
        x = self.relu2(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x


class UltraMinimalGameOfLifeCNN(nn.Module):
    """Two-layer CNN that directly maps neighbor features to the next state."""

    def __init__(self, padding_mode='circular'):
        super(UltraMinimalGameOfLifeCNN, self).__init__()
        self.extract = nn.Conv2d(1, 2, kernel_size=3, padding=1, 
                                 padding_mode=padding_mode)
        self.relu = nn.ReLU()
        self.decide = nn.Conv2d(2, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.extract(x)
        x = self.relu(x)
        x = self.decide(x)
        x = self.sigmoid(x)
        return x


class DeepGameOfLifeCNN(nn.Module):
    """Stacked 3x3 convolutional network."""

    def __init__(self, hidden_channels=32, num_layers=2, padding_mode='circular'):
        super(DeepGameOfLifeCNN, self).__init__()
        layers = []
        layers.append(nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1,
                               padding_mode=padding_mode))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Conv2d(hidden_channels, hidden_channels, 
                                   kernel_size=3, padding=1, padding_mode=padding_mode))
            layers.append(nn.ReLU())
        layers.append(nn.Conv2d(hidden_channels, 1, kernel_size=1))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def count_parameters(model):
    """Return the number of trainable parameters."""
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
