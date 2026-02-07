"""
TAP-Score Dataset: Positive/Negative Sampling

Creates training data for TAP-Score:
- Positive: (obs window, matching expert actions)
- Negative: (obs window, mismatched actions)

With observation augmentations to learn invariance to benign visual changes.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import zarr
from pathlib import Path


class ObservationAugmentor:
    """Apply random augmentations to observations during training.

    This teaches TAP that correct actions should still score high
    even if lighting/noise changes - preventing "pixel shift detector" behavior.
    """

    def __init__(self, p=0.3):
        """
        Args:
            p: Probability of applying each augmentation type (default 0.3 for moderate augmentation)
        """
        self.p = p

    def __call__(self, obs):
        """
        Args:
            obs: (T, C, H, W) observation window, values in [0, 1]
        Returns:
            augmented obs with same shape
        """
        obs = obs.copy()

        # Brightness (multiply by random factor) - moderate range
        if np.random.rand() < self.p:
            factor = np.random.uniform(0.8, 1.2)
            obs = np.clip(obs * factor, 0, 1)

        # Gaussian noise - light noise only
        if np.random.rand() < self.p:
            noise_std = np.random.uniform(0.01, 0.08)
            noise = np.random.randn(*obs.shape).astype(np.float32) * noise_std
            obs = np.clip(obs + noise, 0, 1)

        # Cutout (random black rectangle) - smaller patches
        if np.random.rand() < self.p:
            T, C, H, W = obs.shape
            cut_size = np.random.randint(5, 15)
            x = np.random.randint(0, W - cut_size)
            y = np.random.randint(0, H - cut_size)
            obs[:, :, y:y+cut_size, x:x+cut_size] = 0

        # Small translation (shift and pad) - smaller shifts
        if np.random.rand() < self.p:
            T, C, H, W = obs.shape
            shift_x = np.random.randint(-3, 4)
            shift_y = np.random.randint(-3, 4)
            new_obs = np.zeros_like(obs)
            src_x = max(0, shift_x)
            src_y = max(0, shift_y)
            dst_x = max(0, -shift_x)
            dst_y = max(0, -shift_y)
            w = W - abs(shift_x)
            h = H - abs(shift_y)
            new_obs[:, :, dst_y:dst_y+h, dst_x:dst_x+w] = obs[:, :, src_y:src_y+h, src_x:src_x+w]
            obs = new_obs

        return obs.astype(np.float32)


class TAPDataset(Dataset):
    """
    Dataset for TAP-Score training.

    Samples (observation window, action chunk, label) triplets.
    Positives: real expert pairs
    Negatives: mismatched or noisy actions
    """

    def __init__(
        self,
        data_path,
        obs_window=2,
        action_chunk=16,
        negative_ratio=1.0,
        noise_std=0.1,
        transform=None,
        episode_filter=None,
        augment=False,
        augment_prob=0.5,
    ):
        """
        Args:
            data_path: Path to zarr dataset (Push-T format)
            obs_window: Number of observation frames
            action_chunk: Number of actions in chunk
            negative_ratio: Ratio of negatives to positives
            noise_std: Std of Gaussian noise for noisy negatives
            transform: Optional transform for observations
            episode_filter: Optional set of episode indices to include (for train/val split)
            augment: If True, apply random observation augmentations
            augment_prob: Probability of each augmentation type
        """
        self.obs_window = obs_window
        self.action_chunk = action_chunk
        self.negative_ratio = negative_ratio
        self.noise_std = noise_std
        self.transform = transform
        self.episode_filter = episode_filter
        self.augment = augment
        self.augmentor = ObservationAugmentor(p=augment_prob) if augment else None

        # Load data
        self.data_path = Path(data_path)
        self._load_data()

        # Create index of valid samples
        self._create_index()

    def _load_data(self):
        """Load Push-T zarr dataset."""
        root = zarr.open_group(str(self.data_path), mode='r')

        # Push-T structure: data/img, data/action, meta/episode_ends
        self.images = root['data']['img'][:]  # (N, H, W, C)
        self.actions = root['data']['action'][:]  # (N, action_dim)
        self.episode_ends = root['meta']['episode_ends'][:]

        # Auto-detect action dimension
        self.action_dim = self.actions.shape[-1]

        # Compute episode starts
        self.episode_starts = np.concatenate([[0], self.episode_ends[:-1]])
        self.num_episodes = len(self.episode_ends)

        print(f"Loaded {self.num_episodes} episodes, {len(self.images)} frames, action_dim={self.action_dim}")

    def _create_index(self):
        """Create index of valid (episode, timestep) pairs."""
        self.valid_indices = []

        min_len = self.obs_window + self.action_chunk

        for ep_idx in range(self.num_episodes):
            # Skip episodes not in filter (for train/val split)
            if self.episode_filter is not None and ep_idx not in self.episode_filter:
                continue

            start = self.episode_starts[ep_idx]
            end = self.episode_ends[ep_idx]
            ep_len = end - start

            if ep_len < min_len:
                continue

            # Valid timesteps: can get obs_window before and action_chunk after
            for t in range(self.obs_window - 1, ep_len - self.action_chunk):
                global_t = start + t
                self.valid_indices.append((ep_idx, global_t))

        print(f"Created {len(self.valid_indices)} valid samples")

    def __len__(self):
        # Each valid index gives 1 positive + negative_ratio negatives
        return int(len(self.valid_indices) * (1 + self.negative_ratio))

    def _get_obs_window(self, global_t):
        """Get observation window ending at global_t."""
        start_t = global_t - self.obs_window + 1
        obs = self.images[start_t:global_t + 1]  # (T, H, W, C)

        # Convert to (T, C, H, W) and normalize to [0, 1]
        obs = obs.astype(np.float32) / 255.0
        obs = np.transpose(obs, (0, 3, 1, 2))

        return obs

    def _get_action_chunk(self, global_t):
        """Get action chunk starting at global_t."""
        actions = self.actions[global_t:global_t + self.action_chunk]
        return actions.astype(np.float32)

    def _get_random_action_chunk(self, exclude_ep_idx, exclude_t):
        """Get action chunk from different episode/time."""
        while True:
            # Pick random valid index
            idx = np.random.randint(len(self.valid_indices))
            ep_idx, global_t = self.valid_indices[idx]

            # Make sure it's different enough
            if ep_idx != exclude_ep_idx or abs(global_t - exclude_t) > self.action_chunk * 2:
                return self._get_action_chunk(global_t)

    def __getitem__(self, idx):
        """
        Get sample.

        First half of indices: positives
        Second half: negatives (cycling through shuffle, noise, permute types)
        """
        num_positives = len(self.valid_indices)

        if idx < num_positives:
            # Positive sample
            return self._get_positive(idx)
        else:
            # Negative sample
            neg_offset = idx - num_positives
            base_idx = neg_offset % num_positives
            neg_type = neg_offset % 3  # Cycle through all 3 negative types
            return self._get_negative(base_idx, neg_type)

    def _get_positive(self, idx):
        """Get positive (matching) sample."""
        ep_idx, global_t = self.valid_indices[idx]

        obs = self._get_obs_window(global_t)
        actions = self._get_action_chunk(global_t)

        # Apply observation augmentation (teaches invariance to visual changes)
        if self.augmentor is not None:
            obs = self.augmentor(obs)

        if self.transform:
            obs = self.transform(obs)

        return {
            'obs': torch.from_numpy(obs),
            'actions': torch.from_numpy(actions),
            'label': torch.tensor(1.0),
        }

    def _get_negative(self, idx, neg_type):
        """Get negative sample."""
        ep_idx, global_t = self.valid_indices[idx]

        obs = self._get_obs_window(global_t)

        if neg_type % 3 == 0:
            # Shuffle: actions from different time/episode
            actions = self._get_random_action_chunk(ep_idx, global_t)
        elif neg_type % 3 == 1:
            # Noise: add Gaussian noise to correct actions
            actions = self._get_action_chunk(global_t)
            actions = actions + np.random.randn(*actions.shape).astype(np.float32) * self.noise_std
        else:
            # Permute: shuffle order within chunk
            actions = self._get_action_chunk(global_t)
            perm = np.random.permutation(self.action_chunk)
            actions = actions[perm]

        # Apply observation augmentation (same augmentations for negatives)
        if self.augmentor is not None:
            obs = self.augmentor(obs)

        if self.transform:
            obs = self.transform(obs)

        return {
            'obs': torch.from_numpy(obs),
            'actions': torch.from_numpy(actions),
            'label': torch.tensor(0.0),
        }


def create_dataloaders(data_path, batch_size=32, train_split=0.9, augment=False, **kwargs):
    """Create train and validation dataloaders with episode-based splitting.

    Episodes are split to prevent data leakage between train and validation sets.
    This ensures the model is evaluated on truly unseen data.

    Args:
        data_path: Path to zarr dataset
        batch_size: Batch size for dataloaders
        train_split: Fraction of episodes for training
        augment: If True, apply observation augmentations to training data
        **kwargs: Additional arguments for TAPDataset
    """
    from torch.utils.data import DataLoader

    # First, load data to get episode information
    root = zarr.open_group(str(data_path), mode='r')
    episode_ends = root['meta']['episode_ends'][:]
    num_episodes = len(episode_ends)

    # Split episodes (not samples) for proper train/val separation
    episode_indices = np.arange(num_episodes)
    np.random.shuffle(episode_indices)

    train_ep_count = int(num_episodes * train_split)
    train_episodes = set(episode_indices[:train_ep_count].tolist())
    val_episodes = set(episode_indices[train_ep_count:].tolist())

    print(f"Episode split: {len(train_episodes)} train, {len(val_episodes)} val")

    # Create separate datasets with episode filtering
    # Augmentation only for training, not validation
    train_dataset = TAPDataset(data_path, episode_filter=train_episodes, augment=augment, **kwargs)
    val_dataset = TAPDataset(data_path, episode_filter=val_episodes, augment=False, **kwargs)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader
