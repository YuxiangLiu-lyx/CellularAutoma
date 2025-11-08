"""Evaluation metrics and analysis tools."""

from .metrics import (
    pixel_accuracy,
    multi_step_accuracy,
    find_collapse_step,
    hamming_distance,
    confusion_matrix,
    evaluate_on_dataset
)

__all__ = [
    'pixel_accuracy',
    'multi_step_accuracy',
    'find_collapse_step',
    'hamming_distance',
    'confusion_matrix',
    'evaluate_on_dataset'
]
