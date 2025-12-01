"""
Evaluate minimal CNN model on test dataset.
"""
import sys
from pathlib import Path
import torch
import numpy as np
import h5py

sys.path.append(str(Path(__file__).parent.parent))

from src.models.cnn import GameOfLifeCNN, count_parameters
from src.evaluation.metrics import pixel_accuracy


def load_model(model_path, device):
    """Load trained model."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model = GameOfLifeCNN(hidden_channels=4, padding_mode='circular')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from: {model_path}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Validation accuracy: {checkpoint.get('val_accuracy', 'N/A'):.6f}")
    
    return model


def evaluate_on_test_set(model, test_h5_path, device, batch_size=64):
    """Evaluate model on test dataset."""
    with h5py.File(test_h5_path, 'r') as f:
        states_t = f['states_t'][:]
        states_t1 = f['states_t1'][:]
    
    num_samples = len(states_t)
    all_predictions = []
    
    print(f"\nEvaluating on {num_samples} test samples...")
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_t = states_t[i:i+batch_size]
            batch_t = torch.from_numpy(batch_t).float().unsqueeze(1).to(device)
            
            output = model(batch_t)
            pred = (output > 0.5).cpu().numpy().squeeze(1)
            all_predictions.append(pred)
    
    predictions = np.concatenate(all_predictions, axis=0)
    accuracy = pixel_accuracy(predictions, states_t1)
    
    return accuracy, predictions, states_t1


def main():
    """Main evaluation function."""
    project_root = Path(__file__).parent.parent
    model_path = project_root / "models" / "minimal" / "cnn_4ch.pth"
    test_path = project_root / "data" / "processed" / "test_random.h5"
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first:")
        print("  python scripts/train_minimal.py")
        return
    
    if not test_path.exists():
        print(f"Error: Test data not found at {test_path}")
        print("Please generate dataset first:")
        print("  python scripts/generate_dataset.py")
        return
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    print("="*70)
    
    model = load_model(model_path, device)
    
    print("\n" + "="*70)
    print("Testing on random test set")
    print("="*70)
    
    accuracy, predictions, ground_truth = evaluate_on_test_set(
        model, test_path, device
    )
    
    print(f"\nTest accuracy: {accuracy:.6f} ({accuracy*100:.2f}%)")
    
    alive_cells_true = np.mean(ground_truth)
    alive_cells_pred = np.mean(predictions)
    print(f"\nAlive cells in ground truth: {alive_cells_true*100:.2f}%")
    print(f"Alive cells in predictions: {alive_cells_pred*100:.2f}%")
    
    print("\n" + "="*70)
    print("Architecture Details")
    print("="*70)
    print("GameOfLifeCNN (4 channels):")
    print("  Layer 1: Conv2d(1 → 4, kernel=3x3, circular)")
    print("    - Extract 4 different local features")
    print("  Layer 2: Conv2d(4 → 1, kernel=1x1)")
    print("    - Combine features to predict next state")
    print(f"\nTotal parameters: {count_parameters(model):,}")
    print(f"Previous model (h=16): 177 parameters")
    print(f"Current model (h=4): {count_parameters(model):,} parameters")


if __name__ == "__main__":
    main()

