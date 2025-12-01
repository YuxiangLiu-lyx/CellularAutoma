"""
Minimal network experiment on random logic functions.
Tests whether a 2-hidden-neuron network can learn logic with noisy inputs.
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


class LogicGateMLP(nn.Module):
    """Two-layer MLP for binary classification."""
    
    def __init__(self, input_size=32, hidden_size=2):
        super(LogicGateMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


def generate_random_logic_function(num_useful_inputs=5, seed=42):
    """
    Generate logic function using subset of inputs.
    
    Returns:
        logic_func: Callable that maps 32-bit input to binary output
        useful_indices: Indices of inputs used in the function
    """
    np.random.seed(seed)
    useful_indices = sorted(np.random.choice(32, num_useful_inputs, replace=False))
    
    if num_useful_inputs >= 5:
        idx = useful_indices
        def logic_func(x):
            xor_part = int(x[idx[0]]) ^ int(x[idx[1]])
            and_part = int(x[idx[3]]) and (not int(x[idx[4]]))
            or_part = int(x[idx[2]]) or and_part
            return int(xor_part and or_part)
            
    elif num_useful_inputs >= 3:
        idx = useful_indices
        def logic_func(x):
            xor_part = int(x[idx[0]]) ^ int(x[idx[1]])
            return int(xor_part and int(x[idx[2]]))
    else:
        idx = useful_indices
        def logic_func(x):
            return int(int(x[idx[0]]) and int(x[idx[1]]))
    
    return logic_func, useful_indices


def generate_dataset(logic_func, num_samples=10000, seed=42):
    """Generate binary input-output pairs for given logic function."""
    np.random.seed(seed)
    X = np.random.randint(0, 2, size=(num_samples, 32)).astype(np.float32)
    y = np.array([logic_func(x) for x in X]).astype(np.float32)
    return X, y


def train_and_evaluate(model, train_loader, val_loader, test_data, device, 
                      max_epochs=200, target_accuracy=1.0):
    """Train model and return performance metrics."""
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    X_test, y_test = test_data
    X_test = torch.from_numpy(X_test).to(device)
    y_test = torch.from_numpy(y_test).to(device)
    
    best_val_acc = 0
    convergence_epoch = -1
    history = []
    
    for epoch in range(max_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            output = model(batch_x).squeeze()
            loss = criterion(output, batch_y)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = (output > 0.5).float()
            train_correct += (pred == batch_y).sum().item()
            train_total += len(batch_y)
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                output = model(batch_x).squeeze()
                loss = criterion(output, batch_y)
                
                val_loss += loss.item()
                pred = (output > 0.5).float()
                val_correct += (pred == batch_y).sum().item()
                val_total += len(batch_y)
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        # Test accuracy
        with torch.no_grad():
            test_output = model(X_test).squeeze()
            test_pred = (test_output > 0.5).float()
            test_acc = (test_pred == y_test).sum().item() / len(y_test)
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'test_acc': test_acc
        })
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        if (epoch + 1) % 20 == 0 or val_acc >= target_accuracy:
            print(f"  Epoch {epoch+1:3d}: Train={train_acc:.4f}, Val={val_acc:.4f}, Test={test_acc:.4f}")
        
        if val_acc >= target_accuracy and convergence_epoch == -1:
            convergence_epoch = epoch + 1
            print(f"  Converged at epoch {convergence_epoch}")
            break
    
    final_test_acc = history[-1]['test_acc']
    
    return {
        'converged': convergence_epoch != -1,
        'convergence_epoch': convergence_epoch,
        'best_val_acc': best_val_acc,
        'final_test_acc': final_test_acc,
        'history': history
    }


def analyze_learned_features(model, useful_indices):
    """Compute feature importance and compare with ground truth."""
    input_weights = model.fc1.weight.data.cpu().numpy()
    input_importance = np.sum(np.abs(input_weights), axis=0)
    
    top_k = len(useful_indices)
    top_indices = np.argsort(input_importance)[-top_k:][::-1]
    
    overlap = len(set(top_indices) & set(useful_indices))
    precision = overlap / len(top_indices)
    recall = overlap / len(useful_indices)
    
    return {
        'input_importance': input_importance.tolist(),
        'top_k_indices': top_indices.tolist(),
        'useful_indices': useful_indices,
        'overlap': overlap,
        'precision': precision,
        'recall': recall
    }


def main():
    print("="*70)
    print("Logic Gates Experiment: 32->2->1 Network")
    print("="*70)
    
    num_useful_inputs = 5
    num_train = 8000
    num_val = 1000
    num_test = 1000
    hidden_size = 2
    batch_size = 64
    max_epochs = 200
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  Network: 32 -> {hidden_size} -> 1")
    print(f"  Useful inputs: {num_useful_inputs}/32")
    print(f"  Train/val/test: {num_train}/{num_val}/{num_test}")
    print(f"  Max epochs: {max_epochs}")
    
    print("\n" + "="*70)
    print("Generating Logic Function")
    print("="*70)
    
    logic_func, useful_indices = generate_random_logic_function(
        num_useful_inputs=num_useful_inputs,
        seed=42
    )
    
    print(f"Indices: {useful_indices}")
    print(f"Formula: (x[{useful_indices[0]}] XOR x[{useful_indices[1]}]) AND "
          f"(x[{useful_indices[2]}] OR (x[{useful_indices[3]}] AND NOT x[{useful_indices[4]}]))")
    
    print("\nGenerating datasets...")
    X_train, y_train = generate_dataset(logic_func, num_train, seed=42)
    X_val, y_val = generate_dataset(logic_func, num_val, seed=100)
    X_test, y_test = generate_dataset(logic_func, num_test, seed=200)
    
    print(f"Train: {X_train.shape}, y_mean={y_train.mean():.3f}")
    print(f"Val:   {X_val.shape}, y_mean={y_val.mean():.3f}")
    print(f"Test:  {X_test.shape}, y_mean={y_test.mean():.3f}")
    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val)
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    
    print("\n" + "="*70)
    print("Training")
    print("="*70)
    
    model = LogicGateMLP(input_size=32, hidden_size=hidden_size)
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params} (fc1: {32*hidden_size + hidden_size}, fc2: {hidden_size + 1})")
    
    results = train_and_evaluate(
        model, train_loader, val_loader, (X_test, y_test),
        device, max_epochs=max_epochs, target_accuracy=1.0
    )
    
    print("\n" + "="*70)
    print("Feature Analysis")
    print("="*70)
    
    feature_analysis = analyze_learned_features(model, useful_indices)
    
    print(f"\nGround truth: {useful_indices}")
    print(f"Model top-{len(useful_indices)}: {feature_analysis['top_k_indices']}")
    print(f"Overlap: {feature_analysis['overlap']}/{len(useful_indices)} "
          f"(P={feature_analysis['precision']:.2f}, R={feature_analysis['recall']:.2f})")
    
    print("\nTop 10 inputs by importance:")
    importance = np.array(feature_analysis['input_importance'])
    top_10_idx = np.argsort(importance)[-10:][::-1]
    for idx in top_10_idx:
        marker = "*" if idx in useful_indices else " "
        print(f"  {marker} x[{idx:2d}]: {importance[idx]:.4f}")
    
    print("\n" + "="*70)
    print("Results")
    print("="*70)
    
    if results['converged']:
        print(f"Converged at epoch {results['convergence_epoch']}")
    else:
        print(f"Did not converge")
    
    print(f"Best val acc:  {results['best_val_acc']:.4f}")
    print(f"Final test acc: {results['final_test_acc']:.4f}")
    output_dir = Path(__file__).parent.parent / "experiments" / "logic_gates"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_dict = {
        'experiment': 'Minimal network on random logic',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'architecture': f'{32}-{hidden_size}-1',
            'useful_inputs': num_useful_inputs,
            'indices': useful_indices,
            'samples': {'train': num_train, 'val': num_val, 'test': num_test},
            'max_epochs': max_epochs,
        },
        'results': results,
        'features': feature_analysis,
    }
    
    results_path = output_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\nSaved: {results_path}")
    print("="*70)


if __name__ == "__main__":
    main()
