"""
Evaluation metrics for Game of Life prediction
"""
import numpy as np
from typing import Callable, Optional


def pixel_accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Calculate pixel-wise accuracy.
    
    Args:
        pred: Predicted states (N, H, W) or (H, W)
        true: True states (N, H, W) or (H, W)
        
    Returns:
        Accuracy as float between 0 and 1
    """
    return np.mean(pred == true)


def alive_cell_accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Calculate accuracy only on alive cells (where true == 1).
    
    This metric is crucial for sparse patterns like gliders.
    If model predicts all dead, pixel_accuracy can be high but this will be 0.
    
    Args:
        pred: Predicted states (N, H, W) or (H, W)
        true: True states (N, H, W) or (H, W)
        
    Returns:
        Accuracy on alive cells, or 1.0 if no alive cells
    """
    alive_mask = (true == 1)
    num_alive = np.sum(alive_mask)
    
    if num_alive == 0:
        return 1.0
    
    correct_alive = np.sum((pred == true) & alive_mask)
    return correct_alive / num_alive


def pattern_preservation_score(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Calculate how well the pattern is preserved.
    
    Combines:
    - Alive cell recall: did we keep the alive cells?
    - Dead cell precision: did we avoid false alives?
    
    Args:
        pred: Predicted states
        true: True states
        
    Returns:
        Score between 0 and 1
    """
    # Recall on alive cells
    alive_mask = (true == 1)
    num_true_alive = np.sum(alive_mask)
    
    if num_true_alive == 0:
        # No alive cells, check if prediction is also empty
        return 1.0 if np.sum(pred) == 0 else 0.0
    
    recall = np.sum((pred == 1) & alive_mask) / num_true_alive
    
    # Precision on predicted alive cells
    num_pred_alive = np.sum(pred == 1)
    if num_pred_alive == 0:
        precision = 0.0
    else:
        precision = np.sum((pred == 1) & alive_mask) / num_pred_alive
    
    # F1 score
    if recall + precision == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def multi_step_accuracy(predictor: Callable, 
                       initial_state: np.ndarray,
                       true_trajectory: np.ndarray,
                       metric='pixel') -> np.ndarray:
    """
    Evaluate prediction accuracy over multiple steps.
    
    Args:
        predictor: Function that takes state and returns next state
        initial_state: Initial state (H, W)
        true_trajectory: True trajectory (T, H, W)
        metric: 'pixel', 'alive', or 'pattern' for different metrics
        
    Returns:
        Array of accuracies for each time step
    """
    num_steps = len(true_trajectory) - 1
    accuracies = np.zeros(num_steps)
    
    current_state = initial_state.copy()
    
    # Select metric function
    if metric == 'alive':
        metric_func = alive_cell_accuracy
    elif metric == 'pattern':
        metric_func = pattern_preservation_score
    else:
        metric_func = pixel_accuracy
    
    for t in range(num_steps):
        next_state = predictor(current_state)
        accuracies[t] = metric_func(next_state, true_trajectory[t + 1])
        current_state = next_state
    
    return accuracies


def find_collapse_step(predictor: Callable,
                       initial_state: np.ndarray,
                       true_trajectory: np.ndarray,
                       threshold: float = 0.95,
                       metric: str = 'pattern') -> int:
    """
    Find the step where prediction collapses.
    
    Uses pattern preservation score by default to avoid issues with sparse patterns.
    
    Args:
        predictor: Function that takes state and returns next state
        initial_state: Initial state (H, W)
        true_trajectory: True trajectory (T, H, W)
        threshold: Accuracy threshold
        metric: 'pixel', 'alive', or 'pattern' (recommended)
        
    Returns:
        Step number where accuracy first drops below threshold, or -1 if never
    """
    accuracies = multi_step_accuracy(predictor, initial_state, true_trajectory, metric=metric)
    
    collapse_indices = np.where(accuracies < threshold)[0]
    
    if len(collapse_indices) > 0:
        return int(collapse_indices[0])
    return -1


def hamming_distance(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Calculate normalized Hamming distance (1 - accuracy).
    
    Args:
        pred: Predicted states
        true: True states
        
    Returns:
        Hamming distance as float between 0 and 1
    """
    return 1.0 - pixel_accuracy(pred, true)


def confusion_matrix(pred: np.ndarray, true: np.ndarray) -> dict:
    """
    Calculate confusion matrix for binary prediction.
    
    Args:
        pred: Predicted states
        true: True states
        
    Returns:
        Dictionary with TP, TN, FP, FN counts
    """
    pred_flat = pred.flatten()
    true_flat = true.flatten()
    
    tp = np.sum((pred_flat == 1) & (true_flat == 1))
    tn = np.sum((pred_flat == 0) & (true_flat == 0))
    fp = np.sum((pred_flat == 1) & (true_flat == 0))
    fn = np.sum((pred_flat == 0) & (true_flat == 1))
    
    return {
        'true_positive': int(tp),
        'true_negative': int(tn),
        'false_positive': int(fp),
        'false_negative': int(fn),
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
    }


def evaluate_on_dataset(predictor: Callable,
                       states_t: np.ndarray,
                       states_t1: np.ndarray) -> dict:
    """
    Evaluate predictor on entire dataset.
    
    Args:
        predictor: Function that takes state and returns next state
        states_t: Current states (N, H, W)
        states_t1: True next states (N, H, W)
        
    Returns:
        Dictionary of evaluation metrics
    """
    num_samples = len(states_t)
    predictions = np.zeros_like(states_t1)
    
    for i in range(num_samples):
        predictions[i] = predictor(states_t[i])
    
    accuracy = pixel_accuracy(predictions, states_t1)
    cm = confusion_matrix(predictions, states_t1)
    
    return {
        'accuracy': accuracy,
        'hamming_distance': 1.0 - accuracy,
        **cm
    }

