"""Conway's Game of Life simulator."""
import numpy as np
from typing import Tuple, Optional


class GameOfLife:
    """Game of Life simulator with periodic boundary conditions."""

    def __init__(self, grid_size: Tuple[int, int] = (32, 32)):
        """Create simulator with the given grid dimensions."""
        self.height, self.width = grid_size
    
    def step(self, state: np.ndarray) -> np.ndarray:
        """Compute the next state for the provided grid."""
        neighbors = self._count_neighbors(state)
        next_state = ((state == 1) & ((neighbors == 2) | (neighbors == 3))) | \
                     ((state == 0) & (neighbors == 3))
        return next_state.astype(np.uint8)
    
    def _count_neighbors(self, state: np.ndarray) -> np.ndarray:
        """Count live neighbors for each cell using periodic boundaries."""
        neighbors = np.zeros_like(state, dtype=int)
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                shifted = np.roll(np.roll(state, di, axis=0), dj, axis=1)
                neighbors += shifted
        return neighbors
    
    def simulate(self, initial_state: np.ndarray, num_steps: int) -> np.ndarray:
        """Simulate evolution for multiple steps and return the full trajectory."""
        trajectory = np.zeros((num_steps + 1, self.height, self.width), dtype=np.uint8)
        trajectory[0] = initial_state
        current_state = initial_state.copy()
        for t in range(1, num_steps + 1):
            current_state = self.step(current_state)
            trajectory[t] = current_state
        return trajectory


def place_pattern(grid_size: Tuple[int, int], 
                  pattern: np.ndarray, 
                  position: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Place a pattern on a grid, centered by default or at a given corner."""
    grid = np.zeros(grid_size, dtype=np.uint8)
    ph, pw = pattern.shape
    h, w = grid_size
    if position is None:
        start_h = (h - ph) // 2
        start_w = (w - pw) // 2
    else:
        start_h, start_w = position
    
    end_h = min(start_h + ph, h)
    end_w = min(start_w + pw, w)
    actual_ph = end_h - start_h
    actual_pw = end_w - start_w
    
    grid[start_h:end_h, start_w:end_w] = pattern[:actual_ph, :actual_pw]
    
    return grid
