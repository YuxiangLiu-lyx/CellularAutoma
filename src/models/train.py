"""
Training script for Game of Life CNN
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.cnn import GameOfLifeCNN, DeepGameOfLifeCNN, count_parameters
from src.utils.data_loader import create_dataloader


def train_epoch(model, loader, criterion, optimizer, device):
    """
    Train for one epoch.
    
    Args:
        model: CNN model
        loader: DataLoader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        
    Returns:
        Average loss for epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    for state_t, state_t1 in tqdm(loader, desc="Training"):
        state_t = state_t.to(device)
        state_t1 = state_t1.to(device)
        
        optimizer.zero_grad()
        
        output = model(state_t)
        loss = criterion(output, state_t1)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def evaluate(model, loader, criterion, device):
    """
    Evaluate model on validation set.
    
    Args:
        model: CNN model
        loader: DataLoader
        criterion: Loss function
        device: Device
        
    Returns:
        Dictionary with loss and accuracy
    """
    model.eval()
    total_loss = 0
    total_correct = 0
    total_pixels = 0
    
    with torch.no_grad():
        for state_t, state_t1 in loader:
            state_t = state_t.to(device)
            state_t1 = state_t1.to(device)
            
            output = model(state_t)
            loss = criterion(output, state_t1)
            
            total_loss += loss.item()
            
            # Calculate accuracy
            pred = (output > 0.5).float()
            correct = (pred == state_t1).sum().item()
            total_correct += correct
            total_pixels += state_t1.numel()
    
    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_pixels
    
    return {'loss': avg_loss, 'accuracy': accuracy}


def train_model(model_type='simple', hidden_channels=16, num_layers=2,
                num_epochs=20, batch_size=32, learning_rate=0.001,
                save_path='experiments/cnn/model.pt', data_dir='data/processed'):
    """
    Train CNN model on Game of Life data.
    
    Args:
        model_type: 'simple' or 'deep'
        hidden_channels: Number of hidden channels
        num_layers: Number of layers (for deep model)
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        save_path: Path to save trained model
        data_dir: Directory containing train.h5 and val.h5
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    if model_type == 'simple':
        model = GameOfLifeCNN(hidden_channels=hidden_channels)
    else:
        model = DeepGameOfLifeCNN(hidden_channels=hidden_channels, 
                                 num_layers=num_layers)
    
    model = model.to(device)
    print(f"\nModel: {model_type}")
    print(f"Parameters: {count_parameters(model):,}")
    
    # Create data loaders
    print("\nLoading data...")
    project_root = Path(__file__).parent.parent.parent
    train_path = project_root / data_dir / "train.h5"
    val_path = project_root / data_dir / "val.h5"
    
    train_loader = create_dataloader(str(train_path), 
                                    batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(str(val_path),
                                  batch_size=batch_size, shuffle=False)
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Setup training
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate
        val_results = evaluate(model, val_loader, criterion, device)
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_results['loss']:.4f}")
        print(f"  Val Accuracy: {val_results['accuracy']:.4f}")
        
        # Save best model
        if val_results['accuracy'] > best_val_acc:
            best_val_acc = val_results['accuracy']
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_results['accuracy'],
                'val_loss': val_results['loss'],
            }, save_path)
            print(f"  Saved best model (acc: {best_val_acc:.4f})")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Game of Life CNN')
    parser.add_argument('--model', type=str, default='simple', 
                       choices=['simple', 'deep'], help='Model type')
    parser.add_argument('--hidden', type=int, default=16, 
                       help='Hidden channels')
    parser.add_argument('--layers', type=int, default=2,
                       help='Number of layers (for deep model)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--save', type=str, default='experiments/cnn/model.pt',
                       help='Save path')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                       help='Data directory containing train.h5 and val.h5')
    
    args = parser.parse_args()
    
    train_model(
        model_type=args.model,
        hidden_channels=args.hidden,
        num_layers=args.layers,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_path=args.save,
        data_dir=args.data_dir
    )

