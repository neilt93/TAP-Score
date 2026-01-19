"""
Contrastive TAP-Score: Ranking-based Action-Proposal Scoring

Instead of binary classification, TAP learns to rank the correct action
among multiple candidates. This fixes the collapse problem where BCE-based
TAP either overfits to observations or ignores them.

Key insight: A constant score cannot win because the model must rank
the true action above negatives. The observation encoder cannot be ignored
because the same negative actions appear across many observations.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import zarr
from pathlib import Path


class ContrastiveObsEncoder(nn.Module):
    """CNN encoder for observations, outputs normalized embedding."""

    def __init__(self, obs_channels=3, hidden_dim=128):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(obs_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        # Project to embedding space
        self.fc = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, obs):
        """
        Args:
            obs: (B, T, C, H, W) observation window
        Returns:
            embedding: (B, hidden_dim) L2-normalized
        """
        B, T, C, H, W = obs.shape

        # Flatten batch and time, encode each frame
        obs_flat = obs.view(B * T, C, H, W)
        features = self.conv(obs_flat)  # (B*T, 256)
        features = features.view(B, T, -1)  # (B, T, 256)

        # Pool across time (mean pooling)
        features = features.mean(dim=1)  # (B, 256)

        # Project and normalize
        embedding = self.fc(features)  # (B, hidden_dim)
        embedding = F.normalize(embedding, dim=-1)

        return embedding


class ContrastiveActionEncoder(nn.Module):
    """MLP encoder for action chunks, outputs normalized embedding."""

    def __init__(self, action_dim=2, chunk_size=16, hidden_dim=128):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(action_dim * chunk_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, actions):
        """
        Args:
            actions: (B, M, H, action_dim) M action candidates
        Returns:
            embeddings: (B, M, hidden_dim) L2-normalized
        """
        B, M, H, action_dim = actions.shape

        # Flatten action chunks
        actions_flat = actions.view(B * M, -1)  # (B*M, H*action_dim)
        embeddings = self.mlp(actions_flat)  # (B*M, hidden_dim)
        embeddings = embeddings.view(B, M, -1)  # (B, M, hidden_dim)
        embeddings = F.normalize(embeddings, dim=-1)

        return embeddings


class ContrastiveTAPScore(nn.Module):
    """
    Contrastive TAP-Score: Rank correct action among candidates.

    Given observation window and M action candidates, outputs logits
    where higher = more compatible with the observation.
    """

    def __init__(
        self,
        obs_channels=3,
        action_dim=2,
        obs_window=2,
        action_chunk=16,
        hidden_dim=128,
        temperature=0.1,
    ):
        super().__init__()

        self.obs_window = obs_window
        self.action_chunk = action_chunk
        self.hidden_dim = hidden_dim
        self.temperature = temperature

        self.obs_encoder = ContrastiveObsEncoder(obs_channels, hidden_dim)
        self.action_encoder = ContrastiveActionEncoder(action_dim, action_chunk, hidden_dim)

    def forward(self, obs, actions):
        """
        Compute compatibility scores between observation and action candidates.

        Args:
            obs: (B, T, C, H, W) observation window
            actions: (B, M, H, action_dim) M action candidates

        Returns:
            logits: (B, M) compatibility scores (higher = more compatible)
        """
        # Encode observation: (B, hidden_dim)
        obs_emb = self.obs_encoder(obs)

        # Encode actions: (B, M, hidden_dim)
        action_embs = self.action_encoder(actions)

        # Dot product similarity: (B, M)
        # obs_emb: (B, hidden_dim) -> (B, 1, hidden_dim)
        # action_embs: (B, M, hidden_dim)
        logits = torch.bmm(action_embs, obs_emb.unsqueeze(-1)).squeeze(-1)  # (B, M)
        logits = logits / self.temperature

        return logits

    def get_loss(self, obs, actions):
        """
        InfoNCE loss: cross-entropy where target is index 0 (the positive).

        Args:
            obs: (B, T, C, H, W)
            actions: (B, M, H, action_dim) where actions[:, 0] is positive

        Returns:
            loss: scalar
            accuracy: top-1 retrieval accuracy
        """
        logits = self.forward(obs, actions)  # (B, M)

        # Target is always 0 (positive action is first)
        targets = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        loss = F.cross_entropy(logits, targets)

        # Compute accuracy
        preds = logits.argmax(dim=-1)
        accuracy = (preds == 0).float().mean()

        return loss, accuracy

    def score_single(self, obs, action, n_negatives=16, negative_actions=None):
        """
        Score a single (obs, action) pair by comparing against negatives.

        This is used at evaluation time to get a scalar score in [0, 1].

        Args:
            obs: (T, C, H, W) single observation window
            action: (H, action_dim) single action chunk
            n_negatives: number of negative actions to sample
            negative_actions: (N, H, action_dim) pool of negative actions to sample from

        Returns:
            score: scalar in [0, 1] - probability that action is best among candidates
        """
        if negative_actions is None:
            raise ValueError("Must provide negative_actions pool for scoring")

        # Sample random negatives
        n_pool = negative_actions.shape[0]
        neg_indices = np.random.choice(n_pool, min(n_negatives, n_pool), replace=False)
        neg_actions = negative_actions[neg_indices]  # (n_neg, H, action_dim)

        # Construct candidate set: [proposal, neg1, neg2, ...]
        action = action.unsqueeze(0)  # (1, H, action_dim)
        candidates = torch.cat([action, neg_actions], dim=0)  # (1+n_neg, H, action_dim)

        # Add batch dimension
        obs = obs.unsqueeze(0)  # (1, T, C, H, W)
        candidates = candidates.unsqueeze(0)  # (1, 1+n_neg, H, action_dim)

        # Get logits
        with torch.no_grad():
            logits = self.forward(obs, candidates)  # (1, 1+n_neg)

        # Score is softmax probability of the proposal (index 0)
        probs = F.softmax(logits, dim=-1)
        score = probs[0, 0].item()

        return score


class ContrastiveTAPDataset(Dataset):
    """
    Dataset for contrastive TAP training.

    Returns (obs, [positive_action, neg1, neg2, ...]) for each sample.
    """

    def __init__(
        self,
        data_path,
        obs_window=2,
        action_chunk=16,
        n_negatives=15,
        hard_negative_ratio=0.5,
        noise_std=0.1,
        augment=False,
        augment_prob=0.3,
        episode_filter=None,
    ):
        """
        Args:
            data_path: Path to zarr dataset
            obs_window: Number of observation frames
            action_chunk: Number of actions in chunk
            n_negatives: Number of negative actions per sample
            hard_negative_ratio: Fraction of negatives that are "hard" (corrupted versions)
            noise_std: Noise std for corrupted negatives
            augment: Apply observation augmentations
            augment_prob: Probability of each augmentation
            episode_filter: Optional set of episode indices
        """
        self.obs_window = obs_window
        self.action_chunk = action_chunk
        self.n_negatives = n_negatives
        self.hard_negative_ratio = hard_negative_ratio
        self.noise_std = noise_std
        self.augment = augment
        self.episode_filter = episode_filter

        # Load data
        self.data_path = Path(data_path)
        self._load_data()
        self._create_index()

        # Simple augmentation
        if augment:
            from .dataset import ObservationAugmentor
            self.augmentor = ObservationAugmentor(p=augment_prob)
        else:
            self.augmentor = None

    def _load_data(self):
        root = zarr.open_group(str(self.data_path), mode='r')
        self.images = root['data']['img'][:]
        self.actions = root['data']['action'][:]
        self.episode_ends = root['meta']['episode_ends'][:]
        self.action_dim = self.actions.shape[-1]
        self.episode_starts = np.concatenate([[0], self.episode_ends[:-1]])
        self.num_episodes = len(self.episode_ends)
        print(f"Loaded {self.num_episodes} episodes, {len(self.images)} frames, action_dim={self.action_dim}")

    def _create_index(self):
        self.valid_indices = []
        min_len = self.obs_window + self.action_chunk

        for ep_idx in range(self.num_episodes):
            if self.episode_filter is not None and ep_idx not in self.episode_filter:
                continue

            start = self.episode_starts[ep_idx]
            end = self.episode_ends[ep_idx]
            ep_len = end - start

            if ep_len < min_len:
                continue

            for t in range(self.obs_window - 1, ep_len - self.action_chunk):
                global_t = start + t
                self.valid_indices.append((ep_idx, global_t))

        print(f"Created {len(self.valid_indices)} valid samples")

    def __len__(self):
        return len(self.valid_indices)

    def _get_obs_window(self, global_t):
        start_t = global_t - self.obs_window + 1
        obs = self.images[start_t:global_t + 1]
        obs = obs.astype(np.float32) / 255.0
        obs = np.transpose(obs, (0, 3, 1, 2))
        return obs

    def _get_action_chunk(self, global_t):
        actions = self.actions[global_t:global_t + self.action_chunk]
        return actions.astype(np.float32)

    def _get_random_action_chunk(self, exclude_ep_idx, exclude_t):
        """Get action from different time/episode."""
        while True:
            idx = np.random.randint(len(self.valid_indices))
            ep_idx, global_t = self.valid_indices[idx]
            if ep_idx != exclude_ep_idx or abs(global_t - exclude_t) > self.action_chunk * 2:
                return self._get_action_chunk(global_t)

    def _get_hard_negative(self, positive_action, neg_type):
        """Create hard negative by corrupting positive action."""
        if neg_type == 0:
            # Add noise
            return positive_action + np.random.randn(*positive_action.shape).astype(np.float32) * self.noise_std
        elif neg_type == 1:
            # Permute order
            perm = np.random.permutation(self.action_chunk)
            return positive_action[perm]
        else:
            # Mirror x or y
            mirrored = positive_action.copy()
            if np.random.rand() > 0.5:
                mirrored[:, 0] = -mirrored[:, 0]  # Mirror x
            else:
                mirrored[:, 1] = -mirrored[:, 1]  # Mirror y
            return mirrored

    def __getitem__(self, idx):
        ep_idx, global_t = self.valid_indices[idx]

        # Get observation window
        obs = self._get_obs_window(global_t)

        # Apply augmentation
        if self.augmentor is not None:
            obs = self.augmentor(obs)

        # Get positive action (index 0)
        positive_action = self._get_action_chunk(global_t)

        # Get negative actions
        n_hard = int(self.n_negatives * self.hard_negative_ratio)
        n_random = self.n_negatives - n_hard

        negatives = []

        # Hard negatives (corrupted versions of positive)
        for i in range(n_hard):
            neg_type = i % 3
            negatives.append(self._get_hard_negative(positive_action, neg_type))

        # Random negatives (from other times/episodes)
        for _ in range(n_random):
            negatives.append(self._get_random_action_chunk(ep_idx, global_t))

        # Stack: [positive, neg1, neg2, ...]
        all_actions = np.stack([positive_action] + negatives, axis=0)  # (1+M, H, action_dim)

        return {
            'obs': torch.from_numpy(obs),
            'actions': torch.from_numpy(all_actions),
        }


def create_contrastive_dataloaders(data_path, batch_size=32, train_split=0.9, **kwargs):
    """Create train and validation dataloaders for contrastive TAP."""
    from torch.utils.data import DataLoader

    root = zarr.open_group(str(data_path), mode='r')
    episode_ends = root['meta']['episode_ends'][:]
    num_episodes = len(episode_ends)

    episode_indices = np.arange(num_episodes)
    np.random.shuffle(episode_indices)

    train_ep_count = int(num_episodes * train_split)
    train_episodes = set(episode_indices[:train_ep_count].tolist())
    val_episodes = set(episode_indices[train_ep_count:].tolist())

    print(f"Episode split: {len(train_episodes)} train, {len(val_episodes)} val")

    # Training with augmentation, validation without
    train_dataset = ContrastiveTAPDataset(data_path, episode_filter=train_episodes, augment=True, **kwargs)
    val_dataset = ContrastiveTAPDataset(data_path, episode_filter=val_episodes, augment=False, **kwargs)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader


def build_contrastive_tap_model(config=None):
    """Build contrastive TAP model from config."""
    config = config or {}

    return ContrastiveTAPScore(
        obs_channels=config.get("obs_channels", 3),
        action_dim=config.get("action_dim", 2),
        obs_window=config.get("obs_window", 2),
        action_chunk=config.get("action_chunk", 16),
        hidden_dim=config.get("hidden_dim", 128),
        temperature=config.get("temperature", 0.1),
    )
