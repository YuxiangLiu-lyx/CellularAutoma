"""
Data loading utilities for Game of Life datasets
"""
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class GameOfLifeDataset(Dataset):
    """
    PyTorch Dataset for Game of Life prediction.
    
    Loads data from HDF5 file containing state_t and state_t+1 pairs.
    """
    
    def __init__(self, h5_path, transform=None):
        """
        Initialize dataset.
        
        Args:
            h5_path: Path to HDF5 file
            transform: Optional transform to apply
        """
        self.h5_path = h5_path
        self.transform = transform
        
        with h5py.File(h5_path, 'r') as f:
            self.length = len(f['states_t'])
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        """
        Get a single training sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (state_t, state_t1) as tensors
        """
        with h5py.File(self.h5_path, 'r') as f:
            state_t = f['states_t'][idx].astype(np.float32)
            state_t1 = f['states_t1'][idx].astype(np.float32)
        
        # Add channel dimension: (H, W) -> (1, H, W)
        state_t = torch.from_numpy(state_t).unsqueeze(0)
        state_t1 = torch.from_numpy(state_t1).unsqueeze(0)
        
        if self.transform:
            state_t = self.transform(state_t)
            state_t1 = self.transform(state_t1)
        
        return state_t, state_t1


def create_dataloader(h5_path, batch_size=32, shuffle=True, num_workers=0):
    """
    Create DataLoader for training or evaluation.
    
    Args:
        h5_path: Path to HDF5 file
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        
    Returns:
        DataLoader object
    """
    dataset = GameOfLifeDataset(h5_path)
    
    pin_memory = torch.cuda.is_available()
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return loader


if __name__ == "__main__":
    # Test data loader
    import sys
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "data" / "processed" / "train.h5"
    
    if not data_path.exists():
        print(f"Error: {data_path} not found")
        print("Please run: python3 scripts/generate_dataset.py")
        sys.exit(1)
    
    print("Testing data loader...")
    print("=" * 60)
    
    dataset = GameOfLifeDataset(data_path)
    print(f"Dataset size: {len(dataset)}")
    
    # Test single sample
    state_t, state_t1 = dataset[0]
    print(f"Sample shape: {state_t.shape}")
    print(f"Value range: [{state_t.min():.1f}, {state_t.max():.1f}]")
    
    # Test dataloader
    loader = create_dataloader(data_path, batch_size=32, shuffle=True)
    batch_t, batch_t1 = next(iter(loader))
    print(f"\nBatch shape: {batch_t.shape}")
    print(f"Batch dtype: {batch_t.dtype}")
    
    print("\n" + "=" * 60)
    print("Data loader test passed!")

