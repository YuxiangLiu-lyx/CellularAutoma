"""
Comprehensive 2-Channel CNN Regularization Comparison.

Compare L1, L2, and no regularization on 2-channel Game of Life CNN.
Each configuration runs 30 times to establish statistical significance.
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
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
    """Compute L1 regularization loss on conv1 parameters."""
    l1_loss = 0
    for param in model.conv1.parameters():
        l1_loss += torch.sum(torch.abs(param))
    return lambda_l1 * l1_loss


def train_single_run(run_id, seed, reg_type, reg_strength, train_loader, val_loader, 
                     criterion, device, max_epochs=100, target_accuracy=1.0, patience=20):
    """
    Train one 2-channel model run with specified regularization.
    
    Args:
        run_id: Run identifier
        seed: Random seed
        reg_type: 'l1', 'l2', or 'none'
        reg_strength: Regularization strength (lambda_l1 or weight_decay)
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        device: Computing device
        max_epochs: Maximum training epochs
        target_accuracy: Target accuracy to reach
        patience: Early stopping patience
        
    Returns:
        Dictionary with run results
    """
    set_seed(seed)
    
    # Create 2-channel model
    model = GameOfLifeCNN(hidden_channels=2, padding_mode='circular')
    model = model.to(device)
    
    # Setup optimizer based on regularization type
    if reg_type == 'l2':
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=reg_strength)
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    
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
            
            # Base loss
            bce_loss = criterion(output, state_t1)
            
            # Add L1 regularization if needed
            if reg_type == 'l1':
                l1_loss = compute_l1_loss(model, reg_strength)
                loss = bce_loss + l1_loss
            else:
                loss = bce_loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = (output > 0.5).float()
            train_correct += (pred == state_t1).sum().item()
            train_total += state_t1.numel()
        
        train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation
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
            'train_loss': float(train_loss),
            'train_acc': float(train_acc),
            'val_loss': float(val_loss),
            'val_acc': float(val_acc)
        })
        
        # Check for improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Check convergence
        if val_acc >= target_accuracy and convergence_epoch == -1:
            convergence_epoch = epoch + 1
            break
        
        # Early stopping
        if epochs_without_improvement >= patience:
            break
    
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
        'reg_type': reg_type,
        'reg_strength': float(reg_strength),
        'converged': convergence_epoch != -1,
        'convergence_epoch': convergence_epoch,
        'final_val_acc': float(val_acc),
        'best_val_acc': float(best_val_acc),
        'total_epochs_trained': len(epoch_history),
        'stopped_early': epochs_without_improvement >= patience and convergence_epoch == -1,
        'channel_l1_norms': channel_l1_norms,
        'channel_l2_norms': channel_l2_norms,
        'epoch_history': epoch_history
    }


def main():
    """Main experiment function."""
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "processed"
    
    output_dir = project_root / "experiments" / "2ch_comprehensive"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print("="*70)
    print("COMPREHENSIVE 2-CHANNEL CNN REGULARIZATION COMPARISON")
    print("="*70)
    print(f"Device: {device}")
    print(f"\nExperiment Design:")
    print(f"  - Architecture: 2-channel CNN (minimum capacity)")
    print(f"  - Regularization types: L1, L2, None")
    print(f"  - Runs per type: 30")
    print(f"  - Total runs: 90")
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
    num_runs_per_type = 30
    
    # Configuration for each regularization type
    configs = [
        {'name': 'L1', 'reg_type': 'l1', 'reg_strength': 0.001, 'base_seed': 10000},
        {'name': 'L2', 'reg_type': 'l2', 'reg_strength': 0.001, 'base_seed': 20000},
        {'name': 'None', 'reg_type': 'none', 'reg_strength': 0.0, 'base_seed': 30000}
    ]
    
    all_results = {}
    start_time = datetime.now()
    
    for config in configs:
        name = config['name']
        reg_type = config['reg_type']
        reg_strength = config['reg_strength']
        base_seed = config['base_seed']
        
        print(f"\n{'='*70}")
        print(f"Testing: {name} Regularization")
        print(f"{'='*70}")
        if reg_type == 'l1':
            print(f"  L1 lambda: {reg_strength}")
        elif reg_type == 'l2':
            print(f"  L2 weight_decay: {reg_strength}")
        else:
            print(f"  No regularization")
        print(f"  Seeds: {base_seed} to {base_seed + (num_runs_per_type-1)*10}")
        
        results = []
        seeds = [base_seed + i * 10 for i in range(num_runs_per_type)]
        
        for i, seed in enumerate(seeds, 1):
            print(f"\n  Run {i}/{num_runs_per_type} (seed={seed})...", end='', flush=True)
            
            result = train_single_run(
                run_id=i,
                seed=seed,
                reg_type=reg_type,
                reg_strength=reg_strength,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                device=device,
                max_epochs=100,
                target_accuracy=1.0,
                patience=20
            )
            
            results.append(result)
            
            if result['converged']:
                print(f" Converged at epoch {result['convergence_epoch']}")
            else:
                print(f" Did not converge (best: {result['best_val_acc']:.4f})")
        
        # Save individual results for this regularization type
        results_file = output_dir / f"results_{reg_type}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'regularization': name,
                'reg_type': reg_type,
                'reg_strength': reg_strength,
                'num_runs': num_runs_per_type,
                'seeds': seeds,
                'results': results
            }, f, indent=2)
        print(f"\n  Saved to: {results_file}")
        
        all_results[name] = results
    
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    
    # Generate summary statistics
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    summary_stats = {}
    
    for name, results in all_results.items():
        converged = [r for r in results if r['converged']]
        converged_count = len(converged)
        convergence_rate = converged_count / num_runs_per_type
        
        if converged:
            convergence_epochs = [r['convergence_epoch'] for r in converged]
            mean_epoch = np.mean(convergence_epochs)
            median_epoch = np.median(convergence_epochs)
            std_epoch = np.std(convergence_epochs)
            min_epoch = np.min(convergence_epochs)
            max_epoch = np.max(convergence_epochs)
        else:
            mean_epoch = None
            median_epoch = None
            std_epoch = None
            min_epoch = None
            max_epoch = None
        
        # Best accuracy among non-converged runs
        non_converged = [r for r in results if not r['converged']]
        if non_converged:
            best_non_converged_acc = max(r['best_val_acc'] for r in non_converged)
        else:
            best_non_converged_acc = None
        
        summary_stats[name] = {
            'total_runs': num_runs_per_type,
            'converged_count': converged_count,
            'convergence_rate': float(convergence_rate),
            'mean_convergence_epoch': float(mean_epoch) if mean_epoch else None,
            'median_convergence_epoch': float(median_epoch) if median_epoch else None,
            'std_convergence_epoch': float(std_epoch) if std_epoch else None,
            'min_convergence_epoch': int(min_epoch) if min_epoch else None,
            'max_convergence_epoch': int(max_epoch) if max_epoch else None,
            'best_non_converged_accuracy': float(best_non_converged_acc) if best_non_converged_acc else None
        }
        
        print(f"\n{name} Regularization:")
        print(f"  Converged: {converged_count}/{num_runs_per_type} ({convergence_rate*100:.1f}%)")
        if converged:
            print(f"  Mean convergence epoch: {mean_epoch:.1f}")
            print(f"  Median convergence epoch: {median_epoch:.1f}")
            print(f"  Std: {std_epoch:.1f} epochs")
            print(f"  Range: {min_epoch}-{max_epoch} epochs")
        if non_converged:
            print(f"  Best non-converged accuracy: {best_non_converged_acc:.4f}")
    
    # Save final summary
    final_summary = {
        'experiment': 'Comprehensive 2-Channel CNN Regularization Comparison',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_time_seconds': total_time,
        'total_time_minutes': total_time / 60,
        'configuration': {
            'architecture': '2-channel CNN',
            'num_runs_per_type': num_runs_per_type,
            'total_runs': num_runs_per_type * 3,
            'max_epochs': 100,
            'early_stopping_patience': 20,
            'target_accuracy': 1.0,
            'regularization_configs': {
                'L1': {'lambda': 0.001},
                'L2': {'weight_decay': 0.001},
                'None': {}
            }
        },
        'summary_statistics': summary_stats
    }
    
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(final_summary, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Experiment Complete!")
    print(f"{'='*70}")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Results saved to: {output_dir}")
    print(f"  - results_l1.json: L1 regularization (30 runs)")
    print(f"  - results_l2.json: L2 regularization (30 runs)")
    print(f"  - results_none.json: No regularization (30 runs)")
    print(f"  - summary.json: Final statistics")
    
    # Quick comparison
    print(f"\n{'='*70}")
    print("CONVERGENCE RATE COMPARISON")
    print(f"{'='*70}")
    rates = [(name, stats['convergence_rate']) for name, stats in summary_stats.items()]
    rates.sort(key=lambda x: x[1], reverse=True)
    
    for i, (name, rate) in enumerate(rates, 1):
        count = summary_stats[name]['converged_count']
        print(f"{i}. {name:6s}: {count}/30 ({rate*100:.1f}%)")
    
    best = rates[0]
    print(f"\nBest performing: {best[0]} with {best[1]*100:.1f}% convergence rate")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
