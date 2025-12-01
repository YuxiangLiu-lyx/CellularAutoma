"""
Test 4-channel CNN convergence.
Train 10 runs to see if 4 channels are sufficient to reach 100% accuracy.
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
                  max_epochs=100, target_accuracy=1.0, save_dir=None):
    """
    Train one 4-channel model run.
    
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
        
    Returns:
        Dictionary with run results
    """
    print(f"\n{'='*70}")
    print(f"Run {run_id}/10 (seed={seed})")
    print(f"{'='*70}")
    
    set_seed(seed)
    
    # Create 4-channel model
    model = GameOfLifeCNN(hidden_channels=4, padding_mode='circular')
    model = model.to(device)
    
    num_params = count_parameters(model)
    print(f"Parameters: {num_params}")
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_acc = 0
    convergence_epoch = -1
    epoch_history = []
    
    for epoch in range(max_epochs):
        model.train()
        train_loss = 0
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
            pred = (output > 0.5).float()
            train_correct += (pred == state_t1).sum().item()
            train_total += state_t1.numel()
        
        train_loss = train_loss / len(train_loader)
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
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        if (epoch + 1) % 10 == 0 or val_acc >= target_accuracy:
            print(f"  Epoch {epoch+1}: Val Acc = {val_acc:.6f}")
        
        if val_acc >= target_accuracy and convergence_epoch == -1:
            convergence_epoch = epoch + 1
            print(f"  ✓ Converged at epoch {convergence_epoch}")
            
            if save_dir:
                model_path = save_dir / f"run_{run_id:02d}_seed_{seed}.pth"
                torch.save({
                    'run_id': run_id,
                    'seed': seed,
                    'hidden_channels': 4,
                    'lambda_l1': lambda_l1,
                    'convergence_epoch': convergence_epoch,
                    'model_state_dict': model.state_dict(),
                    'val_accuracy': val_acc,
                }, model_path)
            
            break
    
    if convergence_epoch == -1:
        print(f"  ✗ Did not converge (best: {best_val_acc:.6f})")
    
    return {
        'run_id': run_id,
        'seed': seed,
        'converged': convergence_epoch != -1,
        'convergence_epoch': convergence_epoch,
        'best_val_acc': float(best_val_acc),
        'epoch_history': epoch_history
    }


def main():
    """Main experiment function."""
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "processed"
    
    output_dir = project_root / "experiments" / "4ch_convergence"
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
    print("4-Channel CNN Convergence Test")
    print("="*70)
    print(f"Device: {device}")
    print(f"Configuration:")
    print(f"  - Architecture: 4-channel CNN")
    print(f"  - L1 regularization: lambda=0.001")
    print(f"  - Number of runs: 10")
    print(f"  - Max epochs per run: 100")
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
    
    criterion = nn.BCELoss()
    lambda_l1 = 0.001
    num_runs = 10
    
    base_seed = 100
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
            save_dir=models_dir
        )
        results.append(result)
    
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    convergence_epochs = [r['convergence_epoch'] for r in results if r['converged']]
    converged_count = len(convergence_epochs)
    
    print(f"\nConverged: {converged_count}/10 runs")
    
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
    print(f"{'Run':<6} {'Seed':<8} {'Converged':<12} {'Epoch':<10} {'Best Acc':<12}")
    print("-" * 70)
    
    for r in results:
        converged = "Yes" if r['converged'] else "No"
        epoch = r['convergence_epoch'] if r['converged'] else "-"
        print(f"{r['run_id']:<6} {r['seed']:<8} {converged:<12} {str(epoch):<10} {r['best_val_acc']:.6f}")
    
    # Save summary
    summary = {
        'experiment': {
            'name': '4-Channel CNN Convergence Test',
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_time_seconds': total_time,
        },
        'configuration': {
            'architecture': '4-channel CNN',
            'lambda_l1': lambda_l1,
            'num_runs': num_runs,
            'max_epochs': 100,
            'target_accuracy': 1.0,
        },
        'statistics': {
            'converged_runs': converged_count,
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
        f.write(f"# 4-Channel CNN Convergence Test\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"## Experiment Design\n\n")
        f.write(f"- **Objective**: Test if 4-channel CNN can reach 100% accuracy\n")
        f.write(f"- **Motivation**: Push the pruning limit to see minimum viable channels\n")
        f.write(f"- **Configuration**: lambda_l1 = 0.001\n")
        f.write(f"- **Number of runs**: 10\n")
        f.write(f"- **Seeds**: {seeds[0]} to {seeds[-1]}\n\n")
        
        f.write(f"## Results\n\n")
        f.write(f"- **Convergence rate**: {converged_count}/10 ({converged_count/10*100:.1f}%)\n")
        
        if convergence_epochs:
            f.write(f"- **Mean convergence**: {np.mean(convergence_epochs):.1f} epochs\n")
            f.write(f"- **Median convergence**: {np.median(convergence_epochs):.1f} epochs\n")
            f.write(f"- **Range**: {np.min(convergence_epochs)}-{np.max(convergence_epochs)} epochs\n")
            f.write(f"- **Std deviation**: {np.std(convergence_epochs):.1f} epochs\n\n")
            
            f.write(f"## Analysis\n\n")
            if converged_count == 10:
                f.write(f"**✓ SUCCESS**: All 10 runs converged to 100% accuracy!\n\n")
                f.write(f"4 channels are **sufficient** for this task. This shows:\n")
                f.write(f"- Extremely minimal architecture can still solve Game of Life\n")
                f.write(f"- Can train 4-channel models directly\n")
                f.write(f"- Significant parameter reduction: 4ch has ~29 params vs 16ch with 177 params\n")
            elif converged_count >= 7:
                f.write(f"**Mostly successful**: {converged_count}/10 runs converged.\n\n")
                f.write(f"4 channels are **likely sufficient** but may need:\n")
                f.write(f"- Better initialization\n")
                f.write(f"- Hyperparameter tuning\n")
                f.write(f"- More training epochs for some cases\n")
            else:
                f.write(f"**Limited success**: Only {converged_count}/10 runs converged.\n\n")
                f.write(f"4 channels may be **too few** for reliable training. Consider:\n")
                f.write(f"- Using 5-6 channels instead\n")
                f.write(f"- Different training strategy\n")
                f.write(f"- Pruning from larger model may be more reliable\n")
        else:
            f.write(f"\n**FAILED**: No runs converged to 100% accuracy.\n\n")
            f.write(f"4 channels are **insufficient** for this task.\n")
        
        f.write(f"\n## Files\n\n")
        f.write(f"- `summary.json` - Complete experimental data\n")
        f.write(f"- `models/` - Saved models from successful runs\n")
        f.write(f"- `README.md` - This file\n")
    
    print(f"README saved to: {readme_path}")
    
    print("\n" + "="*70)
    print("Experiment Complete")
    print("="*70)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"All results saved to: {output_dir}")
    
    # Print conclusion
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    if converged_count == 10:
        print("✓ 4 channels are SUFFICIENT - all runs converged!")
        print(f"  Average convergence: {np.mean(convergence_epochs):.1f} epochs")
    elif converged_count >= 7:
        print(f"⚠ 4 channels are LIKELY SUFFICIENT - {converged_count}/10 converged")
        print(f"  May need tuning for 100% reliability")
    elif converged_count >= 5:
        print(f"⚠ 4 channels are MARGINAL - only {converged_count}/10 converged")
        print(f"  Consider using 5-6 channels instead")
    else:
        print(f"✗ 4 channels are INSUFFICIENT - only {converged_count}/10 converged")
        print(f"  Recommend using at least 6 channels")
    print("="*70)


if __name__ == "__main__":
    main()
