"""
Test 2-channel CNN convergence WITH L2 regularization.
Compare with L1 and no-regularization versions to evaluate L2's effect.
Train 5 runs to test convergence.
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


def train_one_run(run_id, seed, train_loader, val_loader, criterion, device, 
                  weight_decay=0.001, max_epochs=100, target_accuracy=1.0, 
                  save_dir=None, patience=20):
    """
    Train one 2-channel model run WITH L2 regularization.
    
    Args:
        run_id: Run identifier
        seed: Random seed
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        device: Computing device
        weight_decay: L2 regularization strength
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
    
    # Adam with weight decay (L2 regularization)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)
    
    best_val_acc = 0
    convergence_epoch = -1
    epoch_history = []
    epochs_without_improvement = 0
    
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
            
            # Only BCE loss (L2 is in optimizer via weight_decay)
            loss = criterion(output, state_t1)
            
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
        
        # Check for improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        if (epoch + 1) % 10 == 0 or val_acc >= target_accuracy:
            print(f"  Epoch {epoch+1}: Train Acc = {train_acc:.6f}, Val Acc = {val_acc:.6f}")
        
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
                    'weight_decay': weight_decay,
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
    channel_l2_norms = []
    for i in range(2):
        l1_norm = np.sum(np.abs(conv1_weights[i]))
        l2_norm = np.sqrt(np.sum(conv1_weights[i]**2))
        channel_l1_norms.append(float(l1_norm))
        channel_l2_norms.append(float(l2_norm))
    
    return {
        'run_id': run_id,
        'seed': seed,
        'converged': convergence_epoch != -1,
        'convergence_epoch': convergence_epoch,
        'best_val_acc': float(best_val_acc),
        'channel_l1_norms': channel_l1_norms,
        'channel_l2_norms': channel_l2_norms,
        'epoch_history': epoch_history,
        'stopped_early': epochs_without_improvement >= patience and convergence_epoch == -1
    }


def main():
    """Main experiment function."""
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "processed"
    
    output_dir = project_root / "experiments" / "2ch_l2"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    weight_decay = 0.001
    
    print("="*70)
    print("2-Channel CNN Test WITH L2 Regularization")
    print("="*70)
    print(f"Device: {device}")
    print(f"Configuration:")
    print(f"  - Architecture: 2-channel CNN")
    print(f"  - L2 regularization: weight_decay={weight_decay}")
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
    num_runs = 5
    
    # Use different seeds
    base_seed = 6000
    seeds = [base_seed + i * 100 for i in range(num_runs)]
    
    print(f"\nStarting {num_runs} training runs...")
    print(f"Seeds: {seeds[0]} to {seeds[-1]}")
    
    results = []
    start_time = datetime.now()
    
    for i, seed in enumerate(seeds, 1):
        result = train_one_run(
            run_id=i,
            seed=seed,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            weight_decay=weight_decay,
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
    
    print(f"\nDetailed Results:")
    print(f"{'Run':<6} {'Seed':<8} {'Converged':<12} {'Epoch':<10} {'Best Acc':<12} {'Early Stop':<12}")
    print("-" * 70)
    
    for r in results:
        converged = "Yes" if r['converged'] else "No"
        epoch = r['convergence_epoch'] if r['converged'] else "-"
        early = "Yes" if r.get('stopped_early', False) else "No"
        print(f"{r['run_id']:<6} {r['seed']:<8} {converged:<12} {str(epoch):<10} "
              f"{r['best_val_acc']:.6f}    {early:<12}")
    
    # Save summary
    summary = {
        'experiment': {
            'name': '2-Channel CNN Test WITH L2',
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_time_seconds': total_time,
        },
        'configuration': {
            'architecture': '2-channel CNN',
            'weight_decay': weight_decay,
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
        },
        'results': results
    }
    
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_path}")
    
    # Conclusion
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print(f"WITH L2 (weight_decay={weight_decay}): {converged_count}/5 converged ({converged_count/5*100:.1f}%)")
    print(f"Compare with L1 and no-regularization results!")
    print("="*70)


if __name__ == "__main__":
    main()
