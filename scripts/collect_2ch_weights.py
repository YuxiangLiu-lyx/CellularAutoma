"""
Collect weights from all saved 2-channel converged models.
Organize by regularization type (None, L1, L2).
"""
import sys
from pathlib import Path
import torch
import json
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))


def extract_weights(checkpoint_path):
    """Extract weights from a saved model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict']
    
    conv1_w = state_dict['conv1.weight'].numpy()
    conv1_b = state_dict['conv1.bias'].numpy()
    conv2_w = state_dict['conv2.weight'].numpy()
    conv2_b = state_dict['conv2.bias'].numpy()
    
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
        },
        'metadata': {
            'seed': checkpoint.get('seed'),
            'convergence_epoch': checkpoint.get('convergence_epoch'),
            'val_accuracy': checkpoint.get('val_accuracy'),
            'lambda_l1': checkpoint.get('lambda_l1'),
            'weight_decay': checkpoint.get('weight_decay')
        }
    }


def main():
    project_root = Path(__file__).parent.parent
    
    # Define model paths by regularization type
    model_paths = {
        'none': [
            project_root / 'experiments' / '2ch_no_l1' / 'models' / 'run_01_seed_5000.pth',
        ],
        'l1': [
            project_root / 'experiments' / '2ch_convergence' / 'models' / 'run_01_seed_300.pth',
            project_root / 'experiments' / '2ch_convergence' / 'models' / 'run_02_seed_400.pth',
            project_root / 'experiments' / '2ch_convergence' / 'models' / 'run_04_seed_600.pth',
            project_root / 'experiments' / '2ch_convergence' / 'models' / 'run_05_seed_700.pth',
        ],
        'l2': [
            project_root / 'experiments' / '2ch_l2' / 'models' / 'run_02_seed_6100.pth',
            project_root / 'experiments' / '2ch_l2' / 'models' / 'run_04_seed_6300.pth',
        ],
    }
    
    print("="*70)
    print("Collecting 2-Channel Model Weights")
    print("="*70)
    
    all_weights = {}
    
    for reg_type, paths in model_paths.items():
        print(f"\n{reg_type.upper()} regularization:")
        all_weights[reg_type] = []
        
        for path in paths:
            if path.exists():
                print(f"  Loading: {path.name}")
                weights = extract_weights(path)
                all_weights[reg_type].append(weights)
            else:
                print(f"  Not found: {path.name}")
    
    # Save to JSON
    output = {
        'description': '2-channel CNN weights from converged models',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'architecture': {
            'hidden_channels': 2,
            'conv1': '1 -> 2 channels, 3x3 kernel, circular padding',
            'conv2': '2 -> 1 channels, 1x1 kernel',
            'activation': 'ReLU after conv1, Sigmoid after conv2'
        },
        'regularization': {
            'none': 'No regularization',
            'l1': 'L1 on conv1, lambda=0.001',
            'l2': 'L2 weight_decay=0.001'
        },
        'models': all_weights
    }
    
    output_file = project_root / 'experiments' / '2ch_weights_summary.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    for reg_type in ['none', 'l1', 'l2']:
        count = len(all_weights[reg_type])
        print(f"  {reg_type.upper():6s}: {count} models")
    
    print(f"\nSaved to: {output_file}")
    print("="*70)


if __name__ == "__main__":
    main()
