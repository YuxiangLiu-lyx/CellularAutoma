"""
Train 10-channel CNN to verify if it's sufficient.
Based on L1 regularization findings that 10 channels are enough.
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
    
    for state_t, state_t1 in tqdm(loader, desc="Training"):
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


def main():
    """Main training function."""
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "processed"
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    print("\n" + "="*70)
    print("Training 10-Channel CNN (Direct Training)")
    print("="*70)
    print("Hypothesis: 10 channels are sufficient based on L1 pruning results")
    
    model = GameOfLifeCNN(hidden_channels=10, padding_mode='circular')
    model = model.to(device)
    
    print(f"\nModel Architecture:")
    print(f"  Structure: 1→10(3x3) → 10→1(1x1)")
    print(f"  Parameters: {count_parameters(model):,}")
    print(f"  vs 16-channel: 177 parameters")
    print(f"  Reduction: {(1 - count_parameters(model)/177)*100:.1f}%")
    
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
    
    print(f"\nData:")
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    max_epochs = 100
    target_accuracy = 1.0
    best_val_acc = 0
    
    save_path = project_root / "experiments" / "direct_training" / "cnn_10ch.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("Training")
    print("="*70)
    print(f"Target: {target_accuracy*100:.2f}% accuracy or {max_epochs} epochs")
    
    epoch_history = []
    
    for epoch in range(max_epochs):
        print(f"\nEpoch {epoch+1}/{max_epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        epoch_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'hidden_channels': 10,
                'model_state_dict': model.state_dict(),
                'val_accuracy': val_acc,
                'val_loss': val_loss,
                'epoch': epoch + 1,
                'training_method': 'direct_10ch',
            }, save_path)
            print(f"  Saved best model (acc: {best_val_acc:.6f})")
        
        if val_acc >= target_accuracy:
            print(f"\n{'='*70}")
            print(f"SUCCESS: Target accuracy {target_accuracy*100:.2f}% reached!")
            print(f"Converged at epoch {epoch+1}")
            print(f"{'='*70}")
            break
    
    print("\n" + "="*70)
    print("Training Complete")
    print("="*70)
    print(f"Best validation accuracy: {best_val_acc:.6f} ({best_val_acc*100:.2f}%)")
    print(f"Model saved to: {save_path}")
    
    if best_val_acc >= target_accuracy:
        print(f"\n✓ SUCCESS: 10 channels are sufficient!")
        print(f"  - Achieved {best_val_acc*100:.2f}% accuracy")
        print(f"  - Parameters: {count_parameters(model):,} (vs 177 for 16-ch)")
        print(f"  - No pruning needed - can train 10-ch model directly")
    else:
        print(f"\n✗ INSUFFICIENT: 10 channels may not be enough")
        print(f"  - Best achieved: {best_val_acc*100:.2f}%")
        print(f"  - Did not reach 100% in {max_epochs} epochs")
        print(f"  - May need more channels or different architecture")
    
    print("\n" + "="*70)
    print("Comparison with Other Approaches")
    print("="*70)
    print(f"{'Approach':<30} {'Channels':<12} {'Parameters':<15} {'Accuracy':<12}")
    print("-" * 70)
    print(f"{'Original':<30} {'16':<12} {'177':<15} {'100.00%':<12}")
    print(f"{'L1 Pruned (16→10)':<30} {'10 active':<12} {'~111':<15} {'100.00%':<12}")
    print(f"{'Direct 10-channel':<30} {'10':<12} {f'{count_parameters(model)}':<15} {f'{best_val_acc*100:.2f}%':<12}")
    
    print("\n" + "="*70)
    print("Analysis")
    print("="*70)
    
    if best_val_acc >= 0.999:
        print("10 channels trained from scratch can achieve excellent performance!")
        print("This validates the L1 pruning findings.")
        print("\nKey insight:")
        print("  - L1 regularization correctly identified the minimum channel count")
        print("  - Can skip 16-channel training and prune directly to 10 channels")
        print("  - Or train 10-channel model from the beginning")
    elif best_val_acc >= 0.95:
        print("10 channels show promise but may need:")
        print("  - More training epochs")
        print("  - Different learning rate")
        print("  - Different initialization")
    else:
        print("10 channels may be insufficient for direct training.")
        print("Possible reasons:")
        print("  - L1 pruning benefits from 16-channel initialization")
        print("  - 10 channels alone may have optimization difficulties")
        print("  - May need 11-12 channels for stable training")
    
    # Save training history and documentation
    import json
    from datetime import datetime
    
    output_dir = save_path.parent
    
    history_path = output_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump({
            'config': {
                'hidden_channels': 10,
                'max_epochs': max_epochs,
                'target_accuracy': target_accuracy,
            },
            'results': {
                'best_val_acc': float(best_val_acc),
                'converged': bool(best_val_acc >= target_accuracy),
                'final_epoch': epoch_history[-1]['epoch'],
            },
            'history': epoch_history
        }, f, indent=2)
    
    # Save README
    readme_path = output_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(f"# 10-Channel CNN Direct Training\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"## Summary\n\n")
        f.write(f"- **Architecture**: GameOfLifeCNN with 10 channels\n")
        f.write(f"- **Training Method**: Direct training (no pruning)\n")
        f.write(f"- **Parameters**: {count_parameters(model):,}\n")
        f.write(f"- **Best Accuracy**: {best_val_acc*100:.2f}%\n")
        f.write(f"- **Converged**: {'Yes' if best_val_acc >= target_accuracy else 'No'}\n")
        f.write(f"- **Training Epochs**: {epoch_history[-1]['epoch']}\n\n")
        
        f.write(f"## Motivation\n\n")
        f.write(f"Based on L1 regularization experiments, we found that only 10 out of 16 channels are needed.\n")
        f.write(f"This experiment tests if training a 10-channel model directly (without pruning) can achieve 100% accuracy.\n\n")
        
        f.write(f"## Results\n\n")
        if best_val_acc >= target_accuracy:
            f.write(f"✓ **SUCCESS**: 10 channels are sufficient!\n\n")
            f.write(f"- Achieved {best_val_acc*100:.2f}% accuracy\n")
            f.write(f"- Validates L1 pruning findings\n")
            f.write(f"- Can train 10-channel model directly without pruning\n")
        else:
            f.write(f"⚠ **PARTIAL SUCCESS**: Reached {best_val_acc*100:.2f}% accuracy\n\n")
            f.write(f"- Did not reach 100% in {max_epochs} epochs\n")
            f.write(f"- May need more training or different approach\n")
        
        f.write(f"\n## Comparison\n\n")
        f.write(f"| Approach | Channels | Parameters | Accuracy |\n")
        f.write(f"|----------|----------|------------|----------|\n")
        f.write(f"| Original 16-ch | 16 | 177 | 100.00% |\n")
        f.write(f"| L1 Pruned (16→10) | 10 active | ~111 | 100.00% |\n")
        f.write(f"| Direct 10-ch | 10 | {count_parameters(model)} | {best_val_acc*100:.2f}% |\n\n")
        
        f.write(f"## Files\n\n")
        f.write(f"- `cnn_10ch.pth` - Trained model parameters\n")
        f.write(f"- `training_history.json` - Epoch-by-epoch training log\n")
        f.write(f"- `README.md` - This file\n\n")
        
        f.write(f"## Usage\n\n")
        f.write(f"```python\n")
        f.write(f"import torch\n")
        f.write(f"from src.models.cnn import GameOfLifeCNN\n\n")
        f.write(f"checkpoint = torch.load('experiments/direct_training/cnn_10ch.pth')\n")
        f.write(f"model = GameOfLifeCNN(hidden_channels=10, padding_mode='circular')\n")
        f.write(f"model.load_state_dict(checkpoint['model_state_dict'])\n")
        f.write(f"```\n")
    
    print(f"\nAll files saved to: {output_dir}")
    print(f"  - cnn_10ch.pth (model)")
    print(f"  - training_history.json (training log)")
    print(f"  - README.md (documentation)")


if __name__ == "__main__":
    main()

