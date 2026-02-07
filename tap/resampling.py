"""
TAP-Guided Resampling

At each replan step:
1. Sample K action chunks from policy (or perturbations of expert)
2. Score each with TAP
3. Execute the highest scoring chunk

This converts TAP from "detection only" to "active intervention."
"""

import numpy as np
import torch


class TAPResampler:
    """Resampling strategy using TAP-Score to select best action chunk."""

    def __init__(self, model, device, k=5):
        """
        Args:
            model: TAP-Score model
            device: torch device
            k: Number of action candidates to sample
        """
        self.model = model
        self.device = device
        self.k = k
        self.model.eval()

    def select_best_action(self, obs, action_candidates):
        """
        Select the best action chunk from candidates using TAP-Score.

        Args:
            obs: (T, C, H, W) observation window
            action_candidates: list of K action chunks, each (H, action_dim)

        Returns:
            best_action: (H, action_dim) highest scoring action chunk
            best_score: scalar TAP score of selected action
            all_scores: list of all K scores
        """
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)  # (1, T, C, H, W)

        scores = []
        with torch.no_grad():
            for action in action_candidates:
                action_tensor = torch.from_numpy(action).unsqueeze(0).to(self.device)
                score = self.model(obs_tensor, action_tensor).item()
                scores.append(score)

        best_idx = np.argmax(scores)
        return action_candidates[best_idx], scores[best_idx], scores

    def generate_candidates_from_expert(self, expert_action, noise_levels=None):
        """
        Generate K action candidates by perturbing expert action.

        For simulation: creates candidates with varying noise levels,
        demonstrating that TAP can select the cleanest one.

        Args:
            expert_action: (H, action_dim) expert action chunk
            noise_levels: list of K noise standard deviations

        Returns:
            candidates: list of K action chunks
        """
        if noise_levels is None:
            # Default: range from clean to heavily noisy
            noise_levels = [0.0] + [0.05 * (i + 1) for i in range(self.k - 1)]

        candidates = []
        for noise_std in noise_levels[:self.k]:
            if noise_std == 0:
                candidates.append(expert_action.copy())
            else:
                noisy = expert_action + np.random.randn(*expert_action.shape).astype(np.float32) * noise_std
                candidates.append(noisy)

        return candidates


def evaluate_resampling(model, dataset, device, perturbation=None, n_samples=200, k=5):
    """
    Evaluate TAP-guided resampling vs random selection.

    For each sample:
    1. Get observation (optionally perturbed)
    2. Generate K action candidates (expert + noisy variants)
    3. Compare: TAP-selected score vs random selection vs always-expert

    Args:
        model: TAP-Score model
        dataset: TAPDataset
        device: torch device
        perturbation: Optional perturbation to apply to observations
        n_samples: Number of samples to evaluate
        k: Number of candidates per sample

    Returns:
        dict with comparison metrics
    """
    from tqdm import tqdm

    resampler = TAPResampler(model, device, k=k)

    tap_selected_scores = []
    random_selected_scores = []
    expert_scores = []
    tap_selected_correct = 0  # Times TAP selected the clean expert action

    num_positives = len(dataset.valid_indices)
    indices = np.random.choice(num_positives, min(n_samples, num_positives), replace=False)

    for idx in tqdm(indices, desc="Evaluating resampling"):
        sample = dataset._get_positive(idx)
        obs = sample['obs'].numpy()  # (T, C, H, W)
        expert_action = sample['actions'].numpy()  # (H, action_dim)

        # Apply perturbation to observation if specified
        if perturbation is not None:
            obs = perturbation(obs[np.newaxis, ...])[0]  # Add/remove batch dim

        # Generate candidates: expert + noisy variants
        candidates = resampler.generate_candidates_from_expert(expert_action)

        # TAP-guided selection
        best_action, best_score, all_scores = resampler.select_best_action(obs, candidates)
        tap_selected_scores.append(best_score)

        # Random selection
        random_idx = np.random.randint(k)
        random_selected_scores.append(all_scores[random_idx])

        # Expert (always select clean action = index 0)
        expert_scores.append(all_scores[0])

        # Did TAP select the clean expert?
        if np.argmax(all_scores) == 0:
            tap_selected_correct += 1

    return {
        'tap_mean_score': float(np.mean(tap_selected_scores)),
        'random_mean_score': float(np.mean(random_selected_scores)),
        'expert_mean_score': float(np.mean(expert_scores)),
        'tap_selected_expert_rate': tap_selected_correct / len(indices),
        'improvement_over_random': float(np.mean(tap_selected_scores) - np.mean(random_selected_scores)),
    }
