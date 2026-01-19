"""
Calibration for TAP-Score

Select threshold to control false alarm rate on clean data.
"""

import numpy as np
from typing import List, Tuple


def compute_threshold(
    clean_scores: np.ndarray,
    target_false_alarm: float = 0.05,
) -> float:
    """
    Compute threshold for target false alarm rate.

    Args:
        clean_scores: TAP scores on clean expert (obs, action) pairs
        target_false_alarm: Target false alarm rate (e.g., 0.05 = 5%)

    Returns:
        Threshold τ such that P(score < τ | clean) ≈ target_false_alarm
    """
    # Lower scores = more likely OOD
    # We want: fraction of clean with score < τ ≈ target_false_alarm
    threshold = np.quantile(clean_scores, target_false_alarm)

    actual_rate = np.mean(clean_scores < threshold)
    print(f"Threshold: {threshold:.4f}")
    print(f"Target false alarm: {target_false_alarm:.2%}")
    print(f"Actual false alarm: {actual_rate:.2%}")

    return threshold


def evaluate_detection(
    clean_scores: np.ndarray,
    perturbed_scores: np.ndarray,
    threshold: float = None,
) -> dict:
    """
    Evaluate OOD detection performance.

    Args:
        clean_scores: TAP scores on clean data
        perturbed_scores: TAP scores on perturbed data
        threshold: Optional threshold (if None, compute AUROC only)

    Returns:
        Dictionary with metrics
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    # Labels: 0 = clean (should have high score), 1 = perturbed (should have low score)
    # We flip: use -score so that higher means more likely OOD
    labels = np.concatenate([np.zeros(len(clean_scores)), np.ones(len(perturbed_scores))])
    scores = np.concatenate([-clean_scores, -perturbed_scores])  # Negate so higher = more OOD

    auroc = roc_auc_score(labels, scores)

    results = {
        'auroc': auroc,
        'clean_mean': float(np.mean(clean_scores)),
        'clean_std': float(np.std(clean_scores)),
        'perturbed_mean': float(np.mean(perturbed_scores)),
        'perturbed_std': float(np.std(perturbed_scores)),
    }

    if threshold is not None:
        # Compute detection metrics at threshold
        clean_flagged = np.mean(clean_scores < threshold)
        perturbed_flagged = np.mean(perturbed_scores < threshold)

        results['threshold'] = threshold
        results['clean_flag_rate'] = float(clean_flagged)
        results['perturbed_flag_rate'] = float(perturbed_flagged)

    return results


def compute_episode_score(
    scores: np.ndarray,
    method: str = 'mean',
) -> float:
    """
    Aggregate per-timestep scores to episode score.

    Args:
        scores: (T,) per-timestep TAP scores
        method: 'mean', 'min', or 'median'

    Returns:
        Single episode score
    """
    if method == 'mean':
        return float(np.mean(scores))
    elif method == 'min':
        return float(np.min(scores))
    elif method == 'median':
        return float(np.median(scores))
    else:
        raise ValueError(f"Unknown method: {method}")


class TAPCalibrator:
    """
    Calibrator for TAP-Score thresholding.

    Usage:
        calibrator = TAPCalibrator(target_false_alarm=0.05)
        calibrator.fit(clean_episode_scores)
        is_ood = calibrator.predict(new_score)
    """

    def __init__(self, target_false_alarm: float = 0.05):
        self.target_false_alarm = target_false_alarm
        self.threshold = None

    def fit(self, clean_scores: np.ndarray):
        """Fit threshold on clean scores."""
        self.threshold = compute_threshold(clean_scores, self.target_false_alarm)
        return self

    def predict(self, score: float) -> bool:
        """Predict if score indicates OOD."""
        if self.threshold is None:
            raise ValueError("Calibrator not fitted. Call fit() first.")
        return score < self.threshold

    def predict_batch(self, scores: np.ndarray) -> np.ndarray:
        """Predict OOD for batch of scores."""
        if self.threshold is None:
            raise ValueError("Calibrator not fitted. Call fit() first.")
        return scores < self.threshold
