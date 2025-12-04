"""Evaluation metrics for Game of Life prediction."""
import numpy as np
from typing import Callable, Optional


def pixel_accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    """Return pixel-wise accuracy."""
    return np.mean(pred == true)


def alive_cell_accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    """Return accuracy restricted to cells that are alive in the target."""
    alive_mask = (true == 1)
    num_alive = np.sum(alive_mask)
    
    if num_alive == 0:
        return 1.0
    
    correct_alive = np.sum((pred == true) & alive_mask)
    return correct_alive / num_alive


def pattern_preservation_score(pred: np.ndarray, true: np.ndarray) -> float:
    """Return an F1-style score balancing alive recall and dead precision."""
    alive_mask = (true == 1)
    num_true_alive = np.sum(alive_mask)
    
    if num_true_alive == 0:
        return 1.0 if np.sum(pred) == 0 else 0.0
    
    recall = np.sum((pred == 1) & alive_mask) / num_true_alive
    
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
    """Evaluate trajectory accuracy over multiple steps using a chosen metric."""
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
    """Return the first timestep where accuracy drops below threshold, or -1."""
    accuracies = multi_step_accuracy(predictor, initial_state, true_trajectory, metric=metric)
    
    collapse_indices = np.where(accuracies < threshold)[0]
    
    if len(collapse_indices) > 0:
        return int(collapse_indices[0])
    return -1


def hamming_distance(pred: np.ndarray, true: np.ndarray) -> float:
    """Return normalized Hamming distance (1 - accuracy)."""
    return 1.0 - pixel_accuracy(pred, true)


def confusion_matrix(pred: np.ndarray, true: np.ndarray) -> dict:
    """Return confusion matrix counts and derived precision/recall."""
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
    """Evaluate a predictor on a dataset and return summary metrics."""
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
