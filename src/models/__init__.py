"""Neural network models for Game of Life prediction."""

from .cnn import GameOfLifeCNN, DeepGameOfLifeCNN, count_parameters

__all__ = ['GameOfLifeCNN', 'DeepGameOfLifeCNN', 'count_parameters']
