"""
Visualize model predictions vs ground truth
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional


def visualize_prediction_comparison(true_trajectory: np.ndarray,
                                   pred_trajectory: np.ndarray,
                                   pattern_name: str = "Pattern",
                                   save_path: Optional[str] = None,
                                   num_steps_to_show: int = 10):
    """
    Compare predicted trajectory with ground truth side by side.
    
    Args:
        true_trajectory: Ground truth (T, H, W)
        pred_trajectory: Predicted trajectory (T, H, W)
        pattern_name: Name for title
        save_path: Path to save figure
        num_steps_to_show: Number of timesteps to visualize
    """
    num_steps = min(num_steps_to_show, len(true_trajectory))
    indices = np.linspace(0, len(true_trajectory) - 1, num_steps, dtype=int)
    
    fig, axes = plt.subplots(2, num_steps, figsize=(num_steps * 2, 4))
    
    if num_steps == 1:
        axes = axes.reshape(2, 1)
    
    for i, idx in enumerate(indices):
        # True trajectory
        axes[0, i].imshow(true_trajectory[idx], cmap='binary', interpolation='nearest')
        axes[0, i].set_title(f"True t={idx}", fontsize=10)
        axes[0, i].axis('off')
        
        # Predicted trajectory
        axes[1, i].imshow(pred_trajectory[idx], cmap='binary', interpolation='nearest')
        
        # Calculate accuracy
        acc = np.mean(true_trajectory[idx] == pred_trajectory[idx])
        axes[1, i].set_title(f"Pred t={idx}\nAcc:{acc:.2%}", fontsize=10)
        axes[1, i].axis('off')
    
    fig.suptitle(f"{pattern_name}: Prediction vs Truth", fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_difference_map(true_state: np.ndarray,
                            pred_state: np.ndarray,
                            title: str = "Difference",
                            save_path: Optional[str] = None):
    """
    Visualize where predictions differ from ground truth.
    
    Args:
        true_state: Ground truth state (H, W)
        pred_state: Predicted state (H, W)
        title: Plot title
        save_path: Path to save
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # True
    axes[0].imshow(true_state, cmap='binary', interpolation='nearest')
    axes[0].set_title(f"Ground Truth\n{np.sum(true_state)} alive cells")
    axes[0].axis('off')
    
    # Predicted
    axes[1].imshow(pred_state, cmap='binary', interpolation='nearest')
    axes[1].set_title(f"Prediction\n{np.sum(pred_state)} alive cells")
    axes[1].axis('off')
    
    # Difference (red=false positive, blue=false negative, black=correct)
    diff = np.zeros((*true_state.shape, 3))
    
    # Correct predictions: black
    correct = (true_state == pred_state)
    diff[correct] = [0, 0, 0]
    
    # False positives (predicted alive, actually dead): red
    false_pos = (pred_state == 1) & (true_state == 0)
    diff[false_pos] = [1, 0, 0]
    
    # False negatives (predicted dead, actually alive): blue
    false_neg = (pred_state == 0) & (true_state == 1)
    diff[false_neg] = [0, 0, 1]
    
    axes[2].imshow(diff, interpolation='nearest')
    axes[2].set_title(f"Difference\nRed: False+, Blue: False-\nAcc: {np.mean(correct):.2%}")
    axes[2].axis('off')
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved difference map to {save_path}")
    else:
        plt.show()
    
    plt.close()


def count_alive_cells(trajectory: np.ndarray) -> np.ndarray:
    """
    Count number of alive cells at each timestep.
    
    Args:
        trajectory: Trajectory array (T, H, W)
        
    Returns:
        Array of alive cell counts (T,)
    """
    return np.sum(trajectory, axis=(1, 2))

