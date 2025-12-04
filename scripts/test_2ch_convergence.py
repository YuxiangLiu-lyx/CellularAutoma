"""
Test 2-channel CNN convergence with L1 regularization.
Train 5 runs to test the absolute minimum architecture.
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import json
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from src.models.cnn import GameOfLifeCNN, count_parameters
from src.utils.data_loader import create_dataloader


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def compute_l1_loss(model, lambda_l1):
    """Compute L1 regularization loss."""
    l1_loss = 0
    for param in model.conv1.parameters():
        l1_loss += torch.sum(torch.abs(param))
    return lambda_l1 * l1_loss


def train_one_run(run_id, seed, lambda_l1, train_loader, val_loader, criterion, device, 
                  max_epochs=100, target_accuracy=1.0, save_dir=None, patience=20):
    """
    Train one 2-channel model run with L1 regularization.
    
    Args:
        run_id: Run identifier
        seed: Random seed
        lambda_l1: L1 regularization strength
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        device: Computing device
        max_epochs: Maximum training epochs
        target_accuracy: Target accuracy to reach
        save_dir: Directory to save model
        patience: Early stopping patience (epochs without improvement)
        
    Returns:
        Dictionary with run results
    """
    print(f"\n{'='*70}")
    print(f"Run {run_id}/5 (seed={seed})")
    print(f"{'='*70}")
    
    set_seed(seed)
    
    # Create 2-channel model
    model = GameOfLifeCNN(hidden_channels=2, padding_mode='circular')
    model = model.to(device)
    
    num_params = count_parameters(model)
    print(f"Parameters: {num_params}")
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_acc = 0
    convergence_epoch = -1
    epoch_history = []
    epochs_without_improvement = 0
    
    for epoch in range(max_epochs):
        model.train()
        train_loss = 0
        train_bce_loss = 0
        train_l1_loss = 0
        train_correct = 0
        train_total = 0
        
        for state_t, state_t1 in train_loader:
            state_t = state_t.to(device)
            state_t1 = state_t1.to(device)
            
            optimizer.zero_grad()
            output = model(state_t)
            
            bce_loss = criterion(output, state_t1)
            l1_loss = compute_l1_loss(model, lambda_l1)
            loss = bce_loss + l1_loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_bce_loss += bce_loss.item()
            train_l1_loss += l1_loss.item()
            pred = (output > 0.5).float()
            train_correct += (pred == state_t1).sum().item()
            train_total += state_t1.numel()
        
        train_loss = train_loss / len(train_loader)
        train_bce_loss = train_bce_loss / len(train_loader)
        train_l1_loss = train_l1_loss / len(train_loader)
        train_acc = train_correct / train_total
        
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for state_t, state_t1 in val_loader:
                state_t = state_t.to(device)
                state_t1 = state_t1.to(device)
                
                output = model(state_t)
                loss = criterion(output, state_t1)
                
                val_loss += loss.item()
                pred = (output > 0.5).float()
                val_correct += (pred == state_t1).sum().item()
                val_total += state_t1.numel()
        
        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        
        epoch_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_bce_loss': train_bce_loss,
            'train_l1_loss': train_l1_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })
        
        # Check for improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        if (epoch + 1) % 10 == 0 or val_acc >= target_accuracy:
            print(f"  Epoch {epoch+1}: Train Acc = {train_acc:.6f}, Val Acc = {val_acc:.6f}, L1 = {train_l1_loss:.6f}")
        
        # Check convergence
        if val_acc >= target_accuracy and convergence_epoch == -1:
            convergence_epoch = epoch + 1
            print(f"  Converged at epoch {convergence_epoch}")
            
            if save_dir:
                model_path = save_dir / f"run_{run_id:02d}_seed_{seed}.pth"
                torch.save({
                    'run_id': run_id,
                    'seed': seed,
                    'hidden_channels': 2,
                    'lambda_l1': lambda_l1,
                    'convergence_epoch': convergence_epoch,
                    'model_state_dict': model.state_dict(),
                    'val_accuracy': val_acc,
                }, model_path)
            
            break
        
        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            print(f"    Best val acc: {best_val_acc:.6f}")
            break
    
    if convergence_epoch == -1:
        print(f"  Did not converge (best: {best_val_acc:.6f})")
    
    # Analyze channel weights
    conv1_weights = model.conv1.weight.data.cpu().numpy()
    channel_l1_norms = []
    for i in range(2):
        l1_norm = np.sum(np.abs(conv1_weights[i]))
        channel_l1_norms.append(float(l1_norm))
    
    return {
        'run_id': run_id,
        'seed': seed,
        'converged': convergence_epoch != -1,
        'convergence_epoch': convergence_epoch,
        'best_val_acc': float(best_val_acc),
        'channel_l1_norms': channel_l1_norms,
        'epoch_history': epoch_history,
        'stopped_early': epochs_without_improvement >= patience and convergence_epoch == -1
    }


def main():
    """Main experiment function."""
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "processed"
    
    output_dir = project_root / "experiments" / "2ch_convergence"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print("="*70)
    print("2-Channel CNN Convergence Test with L1 Regularization")
    print("="*70)
    print(f"Device: {device}")
    print(f"Configuration:")
    print(f"  - Architecture: 2-channel CNN")
    print(f"  - L1 regularization: lambda=0.001")
    print(f"  - Number of runs: 5")
    print(f"  - Max epochs per run: 100")
    print(f"  - Early stopping patience: 20 epochs")
    print(f"  - Target accuracy: 100%")
    print(f"\nOutput directory: {output_dir}")
    
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
    
    print(f"\nTraining samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    criterion = nn.BCELoss()
    lambda_l1 = 0.001
    num_runs = 5
    
    # Use different seeds for diversity
    base_seed = 300
    seeds = [base_seed + i * 100 for i in range(num_runs)]
    
    print(f"\nStarting {num_runs} training runs...")
    print(f"Seeds: {seeds[0]} to {seeds[-1]}")
    
    results = []
    start_time = datetime.now()
    
    for i, seed in enumerate(seeds, 1):
        result = train_one_run(
            run_id=i,
            seed=seed,
            lambda_l1=lambda_l1,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            max_epochs=100,
            target_accuracy=1.0,
            save_dir=models_dir,
            patience=20
        )
        results.append(result)
    
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    convergence_epochs = [r['convergence_epoch'] for r in results if r['converged']]
    converged_count = len(convergence_epochs)
    early_stopped = sum(1 for r in results if r.get('stopped_early', False))
    
    print(f"\nConverged: {converged_count}/5 runs")
    print(f"Early stopped (no hope): {early_stopped}/5 runs")
    
    if convergence_epochs:
        print(f"\nConvergence Statistics:")
        print(f"  Mean:   {np.mean(convergence_epochs):.1f} epochs")
        print(f"  Median: {np.median(convergence_epochs):.1f} epochs")
        print(f"  Std:    {np.std(convergence_epochs):.1f} epochs")
        print(f"  Min:    {np.min(convergence_epochs)} epochs")
        print(f"  Max:    {np.max(convergence_epochs)} epochs")
        
        print(f"\nDistribution:")
        bins = [0, 10, 20, 30, 50, 100]
        for i in range(len(bins)-1):
            count = sum(1 for e in convergence_epochs if bins[i] < e <= bins[i+1])
            if count > 0:
                print(f"  {bins[i]+1}-{bins[i+1]} epochs: {count} runs")
    
    print(f"\nDetailed Results:")
    print(f"{'Run':<6} {'Seed':<8} {'Converged':<12} {'Epoch':<10} {'Best Acc':<12} {'Early Stop':<12}")
    print("-" * 70)
    
    for r in results:
        converged = "Yes" if r['converged'] else "No"
        epoch = r['convergence_epoch'] if r['converged'] else "-"
        early = "Yes" if r.get('stopped_early', False) else "No"
        print(f"{r['run_id']:<6} {r['seed']:<8} {converged:<12} {str(epoch):<10} "
              f"{r['best_val_acc']:.6f}    {early:<12}")
    
    # Analyze channel usage
    print(f"\n{'='*70}")
    print("Channel L1 Norms Analysis")
    print(f"{'='*70}")
    
    for r in results:
        if r['converged']:
            norms = r['channel_l1_norms']
            print(f"Run {r['run_id']:2d}: Ch0={norms[0]:6.3f}, Ch1={norms[1]:6.3f}")
    
    if converged_count > 0:
        all_ch0 = [r['channel_l1_norms'][0] for r in results if r['converged']]
        all_ch1 = [r['channel_l1_norms'][1] for r in results if r['converged']]
        
        print(f"\nAverage L1 norms (converged runs only):")
        print(f"  Channel 0: {np.mean(all_ch0):.3f} ± {np.std(all_ch0):.3f}")
        print(f"  Channel 1: {np.mean(all_ch1):.3f} ± {np.std(all_ch1):.3f}")
    
    # Save summary
    summary = {
        'experiment': {
            'name': '2-Channel CNN Convergence Test with L1',
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_time_seconds': total_time,
        },
        'configuration': {
            'architecture': '2-channel CNN',
            'lambda_l1': lambda_l1,
            'num_runs': num_runs,
            'max_epochs': 100,
            'early_stopping_patience': 20,
            'target_accuracy': 1.0,
            'seeds': seeds,
        },
        'statistics': {
            'converged_runs': converged_count,
            'early_stopped_runs': early_stopped,
            'total_runs': num_runs,
            'convergence_rate': converged_count / num_runs,
            'mean_convergence_epoch': float(np.mean(convergence_epochs)) if convergence_epochs else None,
            'median_convergence_epoch': float(np.median(convergence_epochs)) if convergence_epochs else None,
            'std_convergence_epoch': float(np.std(convergence_epochs)) if convergence_epochs else None,
            'min_convergence_epoch': int(np.min(convergence_epochs)) if convergence_epochs else None,
            'max_convergence_epoch': int(np.max(convergence_epochs)) if convergence_epochs else None,
        },
        'results': results
    }
    
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_path}")
    
    # Create README
    readme_path = output_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(f"# 2-Channel CNN Convergence Test with L1 Regularization\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"## Experiment Design\n\n")
        f.write(f"- **Objective**: Test if 2-channel CNN (absolute minimum) can reach 100% accuracy\n")
        f.write(f"- **Motivation**: Find the absolute theoretical minimum - can just 2 channels solve Game of Life?\n")
        f.write(f"- **Configuration**: lambda_l1 = 0.001, early stopping patience = 20\n")
        f.write(f"- **Number of runs**: 5\n")
        f.write(f"- **Seeds**: {seeds[0]} to {seeds[-1]}\n")
        f.write(f"- **Parameters per model**: 23 (1×2×3×3+2=20, 2×1×1×1+1=3)\n")
        f.write(f"- **Parameter reduction**: ~87.0% vs standard 16-channel (177 params)\n\n")
        
        f.write(f"## Results\n\n")
        f.write(f"- **Convergence rate**: {converged_count}/5 ({converged_count/5*100:.1f}%)\n")
        
        if convergence_epochs:
            f.write(f"- **Mean convergence**: {np.mean(convergence_epochs):.1f} epochs\n")
            f.write(f"- **Median convergence**: {np.median(convergence_epochs):.1f} epochs\n")
            f.write(f"- **Range**: {np.min(convergence_epochs)}-{np.max(convergence_epochs)} epochs\n")
            f.write(f"- **Std deviation**: {np.std(convergence_epochs):.1f} epochs\n\n")
            
            f.write(f"## Channel Usage\n\n")
            if converged_count > 0:
                all_ch0 = [r['channel_l1_norms'][0] for r in results if r['converged']]
                all_ch1 = [r['channel_l1_norms'][1] for r in results if r['converged']]
                
                f.write(f"Average L1 norms across converged runs:\n")
                f.write(f"- Channel 0: {np.mean(all_ch0):.3f} ± {np.std(all_ch0):.3f}\n")
                f.write(f"- Channel 1: {np.mean(all_ch1):.3f} ± {np.std(all_ch1):.3f}\n\n")
                
                f.write(f"This shows how both channels are being utilized in successful runs.\n\n")
            
            f.write(f"## Analysis\n\n")
            if converged_count == 5:
                f.write(f"All 5 runs converged.\n\n")
                f.write(f"2 channels are sufficient for Game of Life:\n")
                f.write(f"- Minimal architecture achieved (23 parameters)\n")
                f.write(f"- ~87.0% parameter reduction from 16-channel baseline\n\n")
            elif converged_count >= 3:
                f.write(f"{converged_count}/5 runs converged.\n\n")
                f.write(f"2 channels are nearly sufficient but sensitive to initialization.\n")
            elif converged_count >= 2:
                f.write(f"{converged_count}/5 runs converged.\n\n")
                f.write(f"2 channels are possible but unreliable. Consider 3-4 channels.\n")
            else:
                f.write(f"Only {converged_count}/5 runs converged.\n\n")
                f.write(f"2 channels may be insufficient for reliable training.\n")
        else:
            f.write(f"\n**FAILED**: No runs converged to 100% accuracy.\n\n")
            f.write(f"2 channels are **below the minimum** required capacity.\n")
            f.write(f"This establishes the lower bound: need at least 3 channels.\n")
        
        f.write(f"\n## Architecture Verification\n\n")
        f.write(f"```\n")
        f.write(f"Input:  (batch, 1, H, W)          # Single channel Game of Life state\n")
        f.write(f"   ↓\n")
        f.write(f"Conv1:  (batch, 2, H, W)          # 2 hidden channels (1→2, 3×3 kernel)\n")
        f.write(f"   ↓\n")
        f.write(f"ReLU\n")
        f.write(f"   ↓\n")
        f.write(f"Conv2:  (batch, 1, H, W)          # Back to single channel (2→1, 1×1 kernel)\n")
        f.write(f"   ↓\n")
        f.write(f"Sigmoid\n")
        f.write(f"   ↓\n")
        f.write(f"Output: (batch, 1, H, W)          # Predicted next state\n")
        f.write(f"```\n\n")
        f.write(f"Architecture: **Single channel → Multi-channel (2) → Single channel**\n\n")
        
        f.write(f"## Comparison with Other Architectures\n\n")
        f.write(f"| Architecture | Channels | Parameters | Conv1 Params | Conv2 Params | Convergence Rate |\n")
        f.write(f"|--------------|----------|------------|--------------|--------------|------------------|\n")
        f.write(f"| Standard     | 16       | ~177       | 1×16×9+16=160| 16×1×1+1=17  | 100% (baseline)  |\n")
        f.write(f"| Pruned       | 4        | ~29        | 1×4×9+4=40   | 4×1×1+1=5    | ? (see 4ch)      |\n")
        f.write(f"| Minimal      | 3        | ~22        | 1×3×9+3=30   | 3×1×1+1=4    | ? (see 3ch)      |\n")
        f.write(f"| **This test**| **2**    | **23**     | **1×2×9+2=20**| **2×1×1+1=3** | **{converged_count/num_runs*100:.0f}%** |\n\n")
        
        f.write(f"## Files\n\n")
        f.write(f"- `summary.json` - Complete experimental data\n")
        f.write(f"- `models/` - Saved models from successful runs\n")
        f.write(f"- `README.md` - This file\n\n")
        
        f.write(f"## Next Steps\n\n")
        if converged_count >= 4:
            f.write(f"- Analyze learned weights\n")
            f.write(f"- Test different initialization schemes\n")
        elif converged_count >= 2:
            f.write(f"- Investigate successful vs failed runs\n")
            f.write(f"- Consider 3 channels for reliable performance\n")
        else:
            f.write(f"- Consider 3+ channels for reliable training\n")
    
    print(f"README saved to: {readme_path}")
    
    print("\n" + "="*70)
    print("Experiment Complete")
    print("="*70)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"All results saved to: {output_dir}")
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    if converged_count == 5:
        print(f"2 channels sufficient: all runs converged (23 params)")
        print(f"Average convergence: {np.mean(convergence_epochs):.1f} epochs")
    elif converged_count >= 3:
        print(f"2 channels nearly sufficient: {converged_count}/5 converged")
    elif converged_count >= 2:
        print(f"2 channels marginal: {converged_count}/5 converged")
        print(f"Consider 3 channels for reliability")
    else:
        print(f"2 channels insufficient: only {converged_count}/5 converged")
        print(f"Recommend 3+ channels")
    print("="*70)


if __name__ == "__main__":
    main()
