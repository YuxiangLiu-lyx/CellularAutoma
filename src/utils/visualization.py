"""
Visualization tools for Game of Life
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
from typing import Optional


def visualize_state(state: np.ndarray, 
                   title: str = "Game of Life",
                   save_path: Optional[str] = None,
                   figsize: tuple = (8, 8),
                   show_grid: bool = True) -> None:
    """
    Visualize a single Game of Life state.
    
    Args:
        state: State array (H x W)
        title: Plot title
        save_path: Path to save figure, None for display only
        figsize: Figure size
        show_grid: Whether to show grid lines
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.imshow(state, cmap='binary', interpolation='nearest')
    ax.set_title(title, fontsize=16, pad=10)
    
    if show_grid:
        h, w = state.shape
        ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_trajectory(trajectory: np.ndarray,
                        pattern_name: str = "Pattern",
                        save_path: Optional[str] = None,
                        figsize: tuple = (16, 4),
                        num_frames_to_show: int = 8,
                        show_grid: bool = True) -> None:
    """
    Visualize multiple frames from a trajectory.
    
    Args:
        trajectory: Trajectory array (T, H, W)
        pattern_name: Pattern name for title
        save_path: Path to save figure
        figsize: Figure size
        num_frames_to_show: Number of frames to display
        show_grid: Whether to show grid lines
    """
    num_steps = len(trajectory)
    indices = np.linspace(0, num_steps - 1, num_frames_to_show, dtype=int)
    
    fig, axes = plt.subplots(1, num_frames_to_show, figsize=figsize)
    
    for i, (ax, idx) in enumerate(zip(axes, indices)):
        ax.imshow(trajectory[idx], cmap='binary', interpolation='nearest')
        ax.set_title(f"t={idx}", fontsize=12)
        
        if show_grid:
            h, w = trajectory[idx].shape
            ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
            ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    fig.suptitle(f"{pattern_name} Evolution", fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved trajectory to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_animation(trajectory: np.ndarray,
                    pattern_name: str = "Pattern",
                    save_path: Optional[str] = None,
                    fps: int = 10,
                    figsize: tuple = (8, 8),
                    show_grid: bool = True) -> None:
    """
    Create animated GIF from trajectory.
    
    Args:
        trajectory: Trajectory array (T, H, W)
        pattern_name: Pattern name for title
        save_path: Path to save GIF file
        fps: Frames per second
        figsize: Figure size
        show_grid: Whether to show grid lines
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(trajectory[0], cmap='binary', interpolation='nearest', animated=True)
    
    if show_grid:
        h, w = trajectory[0].shape
        ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    ax.set_xticks([])
    ax.set_yticks([])
    title = ax.set_title(f"{pattern_name} - Step 0", fontsize=16)
    
    def update(frame):
        im.set_array(trajectory[frame])
        title.set_text(f"{pattern_name} - Step {frame}")
        return [im, title]
    
    anim = FuncAnimation(fig, update, frames=len(trajectory), 
                        interval=1000//fps, blit=True, repeat=True)
    
    if save_path:
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer)
        print(f"Saved animation to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_pattern_grid(patterns_dict: dict,
                          save_path: Optional[str] = None,
                          figsize: tuple = (15, 10),
                          show_grid: bool = True) -> None:
    """
    Visualize multiple patterns in a grid.
    
    Args:
        patterns_dict: Dictionary of {name: pattern_array}
        save_path: Path to save figure
        figsize: Figure size
        show_grid: Whether to show grid lines
    """
    num_patterns = len(patterns_dict)
    ncols = min(4, num_patterns)
    nrows = (num_patterns + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()
    
    for ax, (name, pattern) in zip(axes, patterns_dict.items()):
        ax.imshow(pattern, cmap='binary', interpolation='nearest')
        ax.set_title(name, fontsize=12)
        
        if show_grid:
            h, w = pattern.shape
            ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
            ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    for ax in axes[num_patterns:]:
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved pattern grid to {save_path}")
    else:
        plt.show()
    
    plt.close()
