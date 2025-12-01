"""
Train minimal CNN architectures to find the minimum capacity needed.
Tests different hidden_channels: 4, 8, 9, 16, 32
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.models.cnn import GameOfLifeCNN, count_parameters
from src.utils.data_loader import create_dataloader


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for state_t, state_t1 in loader:
        state_t = state_t.to(device)
        state_t1 = state_t1.to(device)
        
        optimizer.zero_grad()
        output = model(state_t)
        loss = criterion(output, state_t1)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        pred = (output > 0.5).float()
        correct += (pred == state_t1).sum().item()
        total += state_t1.numel()
    
    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for state_t, state_t1 in loader:
            state_t = state_t.to(device)
            state_t1 = state_t1.to(device)
            
            output = model(state_t)
            loss = criterion(output, state_t1)
            
            total_loss += loss.item()
            
            pred = (output > 0.5).float()
            correct += (pred == state_t1).sum().item()
            total += state_t1.numel()
    
    return total_loss / len(loader), correct / total


def train_model(model_config, num_epochs, data_dir, device, save_dir):
    """
    Train a model with specific configuration.
    
    Args:
        model_config: Dictionary with 'name', 'model', 'save_name'
        num_epochs: Number of training epochs
        data_dir: Directory containing train.h5 and val.h5
        device: Device to train on
        save_dir: Directory to save models
        
    Returns:
        Dictionary with training results
    """
    print(f"\n{'='*60}")
    print(f"Training: {model_config['name']}")
    print(f"{'='*60}")
    
    model = model_config['model']
    model = model.to(device)
    
    print(f"Parameters: {count_parameters(model):,}")
    
    train_loader = create_dataloader(
        data_dir / "train.h5",
        batch_size=64,
        shuffle=True
    )
    
    val_loader = create_dataloader(
        data_dir / "val.h5",
        batch_size=64,
        shuffle=False
    )
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_acc = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = save_dir / f"{model_config['save_name']}.pth"
            torch.save({
                'model_config': model_config['name'],
                'model_state_dict': model.state_dict(),
                'val_accuracy': val_acc,
                'val_loss': val_loss,
            }, save_path)
    
    print(f"Best validation accuracy: {best_val_acc:.6f}")
    
    return {
        'name': model_config['name'],
        'parameters': count_parameters(model),
        'best_val_acc': best_val_acc,
        'history': history
    }


def main():
    """Main training function."""
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "processed"
    save_dir = project_root / "models" / "minimal"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    model_configs = [
        {
            'name': 'CNN (4 channels)',
            'model': GameOfLifeCNN(hidden_channels=4, padding_mode='circular'),
            'save_name': 'cnn_4ch'
        },
    ]
    
    num_epochs = 20
    results = []
    
    for config in model_configs:
        result = train_model(
            model_config=config,
            num_epochs=num_epochs,
            data_dir=data_dir,
            device=device,
            save_dir=save_dir
        )
        results.append(result)
    
    print("\n" + "="*80)
    print("ARCHITECTURE COMPARISON")
    print("="*80)
    print(f"{'Model':<30} {'Parameters':<15} {'Best Val Acc':<15}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['name']:<30} {r['parameters']:<15,} {r['best_val_acc']:<15.6f}")
    
    min_params = min(r['parameters'] for r in results)
    best_acc = max(r['best_val_acc'] for r in results)
    
    print("-" * 80)
    print(f"\nBest accuracy: {best_acc:.6f}")
    print(f"Minimum parameters: {min_params:,}")
    
    print("\nModels achieving >{best_acc-0.001:.4f} accuracy:")
    for r in results:
        if r['best_val_acc'] >= best_acc - 0.001:
            print(f"  {r['name']}: {r['parameters']:,} params")
    
    print("\n" + "="*80)
    print("Architecture Details")
    print("="*80)
    print("GameOfLifeCNN (h=4):")
    print("  Structure: 1→4(3x3) → 4→1(1x1)")
    print("  4 channels extract local features from 3x3 neighborhood")
    print(f"  Total parameters: {results[0]['parameters']:,}")


if __name__ == "__main__":
    main()

