"""
Maze Transfer Learning: Fine-tune from pretrained GoL model.

Use pretrained GoL model as initialization, fine-tune all parameters on Maze rule.
Run 100 times, track convergence epoch for each run.

Maze vs GoL:
- GoL: survive 2-3, birth 3
- Maze: survive 1-5, birth 3

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


class Maze:
    """
    Maze cellular automaton - different survival range from Game of Life.
    
    Rules:
        - Live cell with 1-5 neighbors survives (GoL: 2-3)
        - Dead cell with 3 neighbors becomes alive (same as GoL)
        - All other cells die or remain dead
    """
    
    def __init__(self, grid_size=(32, 32)):
        self.height, self.width = grid_size
    
    def step(self, state):
        """Compute next state."""
        neighbors = self._count_neighbors(state)
        
        # Maze rule: survival 1-5, birth 3
        survive = (neighbors >= 1) & (neighbors <= 5)
        birth = (neighbors == 3)
        
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


def generate_maze_data(num_samples, grid_size=(32, 32), density_range=(0.1, 0.5)):
    """Generate Maze training data."""
    simulator = Maze(grid_size)
    
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


def train_finetune(pretrained_path, seed, train_X, train_Y, val_X, val_Y,
                   criterion, device, save_dir=None, max_epochs=100,
                   target_accuracy=1.0, patience=20, batch_size=64):
    """Fine-tune all parameters from pretrained GoL model."""
    set_seed(seed)
    
    # Load pretrained model
    checkpoint = torch.load(pretrained_path, map_location=device, weights_only=False)
    
    model = GameOfLifeCNN(hidden_channels=2, padding_mode='circular')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Optimize ALL parameters
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_acc = 0
    convergence_epoch = -1
    epochs_without_improvement = 0
    history = []
    
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
            loss = criterion(output, batch_Y)
            loss.backward()
            optimizer.step()
            
            pred = (output > 0.5).float()
            train_correct += (pred == batch_Y).sum().item()
            train_total += batch_Y.numel()
        
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        with torch.no_grad():
            output = model(val_X)
            val_loss = criterion(output, val_Y).item()
            pred = (output > 0.5).float()
            val_correct = (pred == val_Y).sum().item()
            val_acc = val_correct / val_Y.numel()
        
        history.append({
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
        
        checkpoint_path = save_dir / f"finetune_seed_{seed}.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'seed': seed,
            'method': 'finetune_all',
            'convergence_epoch': convergence_epoch,
            'val_accuracy': best_val_acc,
            'weights': weights
        }, checkpoint_path)
    
    return {
        'seed': seed,
        'converged': convergence_epoch != -1,
        'convergence_epoch': convergence_epoch,
        'best_val_acc': float(best_val_acc),
        'epochs_trained': len(history),
        'weights': weights,
        'checkpoint_path': str(checkpoint_path) if checkpoint_path else None,
        'history': history
    }


def find_all_converged_models(project_root):
    """Find all converged 2-channel GoL models."""
    model_dirs = [
        project_root / "experiments" / "2ch_convergence" / "models",
        project_root / "experiments" / "2ch_no_l1" / "models",
        project_root / "experiments" / "2ch_l2" / "models",
        project_root / "experiments" / "2ch_comprehensive" / "models",
    ]
    
    converged_models = []
    
    for model_dir in model_dirs:
        if not model_dir.exists():
            continue
        
        for model_path in model_dir.glob("*.pth"):
            try:
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                # Check if model converged (has convergence_epoch)
                if checkpoint.get('convergence_epoch') is not None:
                    converged_models.append({
                        'path': model_path,
                        'seed': checkpoint.get('seed'),
                        'convergence_epoch': checkpoint.get('convergence_epoch'),
                        'reg_type': checkpoint.get('reg_type', 'unknown'),
                        'lambda_l1': checkpoint.get('lambda_l1'),
                        'weight_decay': checkpoint.get('weight_decay')
                    })
            except Exception as e:
                print(f"  Warning: Could not load {model_path.name}: {e}")
                continue
    
    return converged_models


def main():
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "experiments" / "maze_transfer"
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all converged models
    print("="*70)
    print("Finding all converged 2-channel GoL models...")
    print("="*70)
    
    converged_models = find_all_converged_models(project_root)
    
    if not converged_models:
        print("Error: No converged models found!")
        return
    
    print(f"\nFound {len(converged_models)} converged models:")
    for i, model_info in enumerate(converged_models, 1):
        reg_info = ""
        if model_info['lambda_l1']:
            reg_info = f"L1={model_info['lambda_l1']}"
        elif model_info['weight_decay']:
            reg_info = f"L2={model_info['weight_decay']}"
        else:
            reg_info = "None"
        
        print(f"  {i}. {model_info['path'].name} (seed={model_info['seed']}, {reg_info}, epoch={model_info['convergence_epoch']})")
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print("\n" + "="*70)
    print("MAZE TRANSFER LEARNING: FINE-TUNE FROM ALL CONVERGED GOL MODELS")
    print("="*70)
    print(f"Device: {device}")
    print(f"\nMaze Rule:")
    print(f"  - Live cell survives with 1-5 neighbors (GoL: 2-3)")
    print(f"  - Dead cell births with 3 neighbors (same as GoL)")
    print(f"\nExperiment Design:")
    print(f"  - Method: Fine-tune ALL parameters from GoL pretrained weights")
    print(f"  - Number of runs: {len(converged_models)} (one per converged model)")
    print(f"  - Max epochs: 100, Patience: 20, Target: 100%")
    
    # Generate and preload data
    print(f"\nGenerating Maze data...")
    np.random.seed(42)  # Fixed seed for data generation
    train_X, train_Y = generate_maze_data(10000, grid_size=(32, 32))
    val_X, val_Y = generate_maze_data(2000, grid_size=(32, 32))
    
    # Move to device once
    train_X = train_X.to(device)
    train_Y = train_Y.to(device)
    val_X = val_X.to(device)
    val_Y = val_Y.to(device)
    
    print(f"Training samples: {len(train_X)}")
    print(f"Validation samples: {len(val_X)}")
    print(f"Data moved to: {device}")
    
    criterion = nn.BCELoss()
    num_runs = len(converged_models)
    
    print(f"\n{'='*70}")
    print(f"Starting {num_runs} runs (one per converged model)...")
    print(f"{'='*70}")
    
    results = []
    start_time = datetime.now()
    
    for i, model_info in enumerate(converged_models, 1):
        pretrained_path = model_info['path']
        original_seed = model_info['seed']
        
        # Use a new seed for training (different from original)
        train_seed = 20000 + i * 10
        
        print(f"\n  Run {i}/{num_runs} (init: {pretrained_path.name}, seed={train_seed})...", end='', flush=True)
        
        result = train_finetune(
            pretrained_path=pretrained_path,
            seed=train_seed,
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
        
        # Add original model info to result
        result['pretrained_model'] = str(pretrained_path.name)
        result['pretrained_seed'] = original_seed
        result['pretrained_reg_type'] = model_info['reg_type']
        result['pretrained_convergence_epoch'] = model_info['convergence_epoch']
        
        results.append(result)
        
        if result['converged']:
            print(f" Converged at epoch {result['convergence_epoch']}")
        else:
            print(f" Did not converge (best: {result['best_val_acc']:.4f})")
    
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    
    # Summary
    converged = [r for r in results if r['converged']]
    converged_count = len(converged)
    convergence_rate = converged_count / num_runs
    
    convergence_epochs = [r['convergence_epoch'] for r in converged]
    
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Converged: {converged_count}/{num_runs} ({convergence_rate*100:.1f}%)")
    
    if converged:
        mean_epoch = np.mean(convergence_epochs)
        median_epoch = np.median(convergence_epochs)
        std_epoch = np.std(convergence_epochs)
        min_epoch = np.min(convergence_epochs)
        max_epoch = np.max(convergence_epochs)
        
        print(f"\nConvergence Epoch Statistics:")
        print(f"  Mean: {mean_epoch:.1f}")
        print(f"  Median: {median_epoch:.1f}")
        print(f"  Std: {std_epoch:.1f}")
        print(f"  Range: {min_epoch}-{max_epoch}")
        print(f"\nConvergence Epochs (all {converged_count} converged runs):")
        print(f"  {convergence_epochs}")
    else:
        print("\nNo runs converged.")
    
    # Save results
    saved_models = list(models_dir.glob("*.pth"))
    
    output = {
        'experiment': 'Maze Transfer Learning: Fine-tune from Pretrained GoL',
        'rule': {
            'name': 'Maze',
            'survival': '1-5 neighbors',
            'birth': '3 neighbors'
        },
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'pretrained_model': str(pretrained_path.name),
        'total_time_seconds': total_time,
        'total_time_minutes': total_time / 60,
        'configuration': {
            'num_runs': num_runs,
            'pretrained_models_count': len(converged_models),
            'pretrained_models': [str(m['path'].name) for m in converged_models],
            'train_samples': len(train_X),
            'val_samples': len(val_X),
            'max_epochs': 100,
            'patience': 20,
            'target_accuracy': 1.0,
            'method': 'fine-tune_all_parameters'
        },
        'summary': {
            'converged_count': converged_count,
            'convergence_rate': float(convergence_rate),
            'convergence_epochs': convergence_epochs,
            'mean_convergence_epoch': float(np.mean(convergence_epochs)) if convergence_epochs else None,
            'median_convergence_epoch': float(np.median(convergence_epochs)) if convergence_epochs else None,
            'std_convergence_epoch': float(np.std(convergence_epochs)) if convergence_epochs else None,
            'min_convergence_epoch': int(np.min(convergence_epochs)) if convergence_epochs else None,
            'max_convergence_epoch': int(np.max(convergence_epochs)) if convergence_epochs else None
        },
        'saved_models_count': len(saved_models),
        'results': results
    }
    
    output_file = output_dir / "transfer_results.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Experiment Complete!")
    print(f"{'='*70}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Results saved to: {output_file}")
    print(f"Models saved to: {models_dir} ({len(saved_models)} converged models)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
