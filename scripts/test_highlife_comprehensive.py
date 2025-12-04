"""
Comprehensive HighLife 2-Channel CNN Training.

Train 2-channel CNN on HighLife rule (similar to GoL, one extra birth condition).
Compare L1, L2, and no regularization.
Each configuration runs 30 times.

HighLife vs GoL:
- GoL: survive 2-3, birth 3
- HighLife: survive 2-3, birth 3 OR 6

Optimized: data preloaded to GPU, fast training.
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

from src.models.cnn import GameOfLifeCNN


class HighLife:
    """
    HighLife cellular automaton - similar to Game of Life.
    
    Rules:
        - Live cell with 2-3 neighbors survives (same as GoL)
        - Dead cell with 3 OR 6 neighbors becomes alive (GoL: only 3)
        - All other cells die or remain dead
    """
    
    def __init__(self, grid_size=(32, 32)):
        self.height, self.width = grid_size
    
    def step(self, state):
        """Compute next state."""
        neighbors = self._count_neighbors(state)
        
        # HighLife rule: survival 2-3, birth 3 or 6
        survive = (neighbors == 2) | (neighbors == 3)
        birth = (neighbors == 3) | (neighbors == 6)
        
        next_state = ((state == 1) & survive) | ((state == 0) & birth)
        
        return next_state.astype(np.uint8)
    
    def _count_neighbors(self, state):
        """Count alive neighbors with periodic boundaries."""
        neighbors = np.zeros_like(state, dtype=int)
        
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                shifted = np.roll(np.roll(state, di, axis=0), dj, axis=1)
                neighbors += shifted
        
        return neighbors


def generate_highlife_data(num_samples, grid_size=(32, 32), density_range=(0.1, 0.5)):
    """Generate HighLife training data."""
    simulator = HighLife(grid_size)
    
    states_t = []
    states_t1 = []
    
    for _ in range(num_samples):
        density = np.random.uniform(*density_range)
        state = (np.random.random(grid_size) < density).astype(np.uint8)
        next_state = simulator.step(state)
        
        states_t.append(state)
        states_t1.append(next_state)
    
    X = np.array(states_t)[:, np.newaxis, :, :].astype(np.float32)
    Y = np.array(states_t1)[:, np.newaxis, :, :].astype(np.float32)
    
    return torch.tensor(X), torch.tensor(Y)


def set_seed(seed):
    """Set random seed."""
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


def extract_weights(model):
    """Extract all model weights."""
    conv1_w = model.conv1.weight.data.cpu().numpy()
    conv1_b = model.conv1.bias.data.cpu().numpy()
    conv2_w = model.conv2.weight.data.cpu().numpy()
    conv2_b = model.conv2.bias.data.cpu().numpy()
    
    return {
        'conv1': {
            'channel_0': {
                'kernel': conv1_w[0, 0].tolist(),
                'bias': float(conv1_b[0])
            },
            'channel_1': {
                'kernel': conv1_w[1, 0].tolist(),
                'bias': float(conv1_b[1])
            }
        },
        'conv2': {
            'channel_0_weight': float(conv2_w[0, 0, 0, 0]),
            'channel_1_weight': float(conv2_w[0, 1, 0, 0]),
            'bias': float(conv2_b[0])
        }
    }


def train_single_run(run_id, seed, reg_type, reg_strength, train_X, train_Y,
                     val_X, val_Y, criterion, device, save_dir=None,
                     max_epochs=100, target_accuracy=1.0, patience=20, batch_size=64):
    """Train one 2-channel model run with specified regularization."""
    set_seed(seed)
    
    model = GameOfLifeCNN(hidden_channels=2, padding_mode='circular')
    model = model.to(device)
    
    if reg_type == 'l2':
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=reg_strength)
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_acc = 0
    convergence_epoch = -1
    epoch_history = []
    epochs_without_improvement = 0
    
    n_samples = len(train_X)
    
    for epoch in range(max_epochs):
        model.train()
        train_correct = 0
        train_total = 0
        
        indices = torch.randperm(n_samples, device=train_X.device)
        
        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_X = train_X[batch_idx]
            batch_Y = train_Y[batch_idx]
            
            optimizer.zero_grad()
            output = model(batch_X)
            
            bce_loss = criterion(output, batch_Y)
            
            if reg_type == 'l1':
                l1_loss = compute_l1_loss(model, reg_strength)
                loss = bce_loss + l1_loss
            else:
                loss = bce_loss
            
            loss.backward()
            optimizer.step()
            
            pred = (output > 0.5).float()
            train_correct += (pred == batch_Y).sum().item()
            train_total += batch_Y.numel()
        
        train_acc = train_correct / train_total
        
        # Validation - one batch
        model.eval()
        with torch.no_grad():
            output = model(val_X)
            val_loss = criterion(output, val_Y).item()
            pred = (output > 0.5).float()
            val_correct = (pred == val_Y).sum().item()
            val_acc = val_correct / val_Y.numel()
        
        epoch_history.append({
            'epoch': epoch + 1,
            'train_acc': float(train_acc),
            'val_acc': float(val_acc)
        })
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        if val_acc >= target_accuracy and convergence_epoch == -1:
            convergence_epoch = epoch + 1
            break
        
        if epochs_without_improvement >= patience:
            break
    
    weights = extract_weights(model)
    
    # Save checkpoint if converged
    checkpoint_path = None
    if save_dir and convergence_epoch != -1:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = save_dir / f"run_{run_id:02d}_{reg_type}_seed_{seed}.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'seed': seed,
            'reg_type': reg_type,
            'reg_strength': reg_strength,
            'convergence_epoch': convergence_epoch,
            'val_accuracy': best_val_acc,
            'weights': weights
        }, checkpoint_path)
    
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
        'weights': weights,
        'checkpoint_path': str(checkpoint_path) if checkpoint_path else None,
        'epoch_history': epoch_history
    }


def main():
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "experiments" / "highlife_comprehensive"
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
    print("COMPREHENSIVE HIGHLIFE 2-CHANNEL CNN TRAINING")
    print("="*70)
    print(f"Device: {device}")
    print(f"\nHighLife Rule:")
    print(f"  - Live cell survives with 2-3 neighbors (same as GoL)")
    print(f"  - Dead cell births with 3 OR 6 neighbors (GoL: only 3)")
    print(f"\nExperiment Design:")
    print(f"  - Architecture: 2-channel CNN")
    print(f"  - Regularization types: L1, L2, None")
    print(f"  - Runs per type: 100")
    print(f"  - Total runs: 300")
    print(f"  - Max epochs per run: 100")
    print(f"  - Early stopping patience: 20 epochs")
    print(f"  - Target accuracy: 100%")
    print(f"\nOutput directory: {output_dir}")
    
    # Generate and preload data
    print(f"\nGenerating HighLife data...")
    np.random.seed(42)  # Fixed seed for data generation
    train_X, train_Y = generate_highlife_data(10000, grid_size=(32, 32))
    val_X, val_Y = generate_highlife_data(2000, grid_size=(32, 32))
    
    # Move to device once
    train_X = train_X.to(device)
    train_Y = train_Y.to(device)
    val_X = val_X.to(device)
    val_Y = val_Y.to(device)
    
    print(f"Training samples: {len(train_X)}")
    print(f"Validation samples: {len(val_X)}")
    print(f"Data moved to: {device}")
    
    criterion = nn.BCELoss()
    num_runs_per_type = 100
    
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
                train_X=train_X,
                train_Y=train_Y,
                val_X=val_X,
                val_Y=val_Y,
                criterion=criterion,
                device=device,
                save_dir=models_dir,
                max_epochs=100,
                target_accuracy=1.0,
                patience=20,
                batch_size=64
            )
            
            results.append(result)
            
            if result['converged']:
                print(f" Converged at epoch {result['convergence_epoch']}")
            else:
                print(f" Did not converge (best: {result['best_val_acc']:.4f})")
        
        # Save results for this regularization type
        results_file = output_dir / f"results_{reg_type}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'rule': 'HighLife',
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
    
    # Summary
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
            std_epoch = np.std(convergence_epochs)
            min_epoch = np.min(convergence_epochs)
            max_epoch = np.max(convergence_epochs)
        else:
            mean_epoch = None
            std_epoch = None
            min_epoch = None
            max_epoch = None
        
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
            'std_convergence_epoch': float(std_epoch) if std_epoch else None,
            'min_convergence_epoch': int(min_epoch) if min_epoch else None,
            'max_convergence_epoch': int(max_epoch) if max_epoch else None,
            'best_non_converged_accuracy': float(best_non_converged_acc) if best_non_converged_acc else None
        }
        
        print(f"\n{name} Regularization:")
        print(f"  Converged: {converged_count}/{num_runs_per_type} ({convergence_rate*100:.1f}%)")
        if converged:
            print(f"  Mean convergence epoch: {mean_epoch:.1f}")
            print(f"  Std: {std_epoch:.1f} epochs")
            print(f"  Range: {min_epoch}-{max_epoch} epochs")
        if non_converged:
            print(f"  Best non-converged accuracy: {best_non_converged_acc:.4f}")
    
    # Save summary
    saved_models = list(models_dir.glob("*.pth"))
    
    final_summary = {
        'experiment': 'Comprehensive HighLife 2-Channel CNN Training',
        'rule': {
            'name': 'HighLife',
            'survival': '2-3 neighbors (same as GoL)',
            'birth': '3 OR 6 neighbors'
        },
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
            'train_samples': 10000,
            'val_samples': 2000,
            'regularization_configs': {
                'L1': {'lambda': 0.001},
                'L2': {'weight_decay': 0.001},
                'None': {}
            }
        },
        'summary_statistics': summary_stats,
        'saved_models_count': len(saved_models)
    }
    
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(final_summary, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Experiment Complete!")
    print(f"{'='*70}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Results saved to: {output_dir}")
    print(f"  - results_l1.json")
    print(f"  - results_l2.json")
    print(f"  - results_none.json")
    print(f"  - summary.json")
    print(f"  - models/: {len(saved_models)} converged model checkpoints")
    
    # Comparison
    print(f"\n{'='*70}")
    print("CONVERGENCE RATE COMPARISON")
    print(f"{'='*70}")
    rates = [(name, stats['convergence_rate']) for name, stats in summary_stats.items()]
    rates.sort(key=lambda x: x[1], reverse=True)
    
    for i, (name, rate) in enumerate(rates, 1):
        count = summary_stats[name]['converged_count']
        print(f"{i}. {name:6s}: {count}/{num_runs_per_type} ({rate*100:.1f}%)")
    
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
