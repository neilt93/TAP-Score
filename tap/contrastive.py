"""
Contrastive TAP-Score: Ranking-based Action-Proposal Scoring

Instead of binary classification, TAP learns to rank the correct action
among multiple candidates. This fixes the collapse problem where BCE-based
TAP either overfits to observations or ignores them.

Key insight: A constant score cannot win because the model must rank
the true action above negatives. The observation encoder cannot be ignored
because the same negative actions appear across many observations.

Supports both image-based (CNN) and state-based (MLP) observations.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import zarr
from pathlib import Path


class ContrastiveObsEncoder(nn.Module):
    """CNN encoder for image observations, outputs normalized embedding."""

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


class ContrastiveStateObsEncoder(nn.Module):
    """MLP encoder for state-based observations, outputs normalized embedding.

    Supports delta encoding: concatenates (s_t, s_t - s_{t-1}) for each timestep,
    which helps the model learn that actions correlate with state changes.
    """

    def __init__(self, obs_dim, obs_window=2, hidden_dim=128, use_deltas=False):
        super().__init__()
        self.obs_dim = obs_dim
        self.obs_window = obs_window
        self.use_deltas = use_deltas

        # Input dimension depends on whether we use deltas
        if use_deltas:
            # Each timestep has (state, delta) except first which has (state, zeros)
            input_dim = obs_dim * obs_window * 2  # state + delta for each timestep
        else:
            input_dim = obs_dim * obs_window

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, obs):
        """
        Args:
            obs: (B, T, obs_dim) state observation window
        Returns:
            embedding: (B, hidden_dim) L2-normalized
        """
        B, T, D = obs.shape

        if self.use_deltas:
            # Compute deltas: s_t - s_{t-1}
            # First timestep delta is zeros
            deltas = torch.zeros_like(obs)
            deltas[:, 1:, :] = obs[:, 1:, :] - obs[:, :-1, :]

            # Concatenate state and delta: (B, T, 2*D)
            obs_with_deltas = torch.cat([obs, deltas], dim=-1)
            obs_flat = obs_with_deltas.view(B, -1)  # (B, T * 2 * obs_dim)
        else:
            obs_flat = obs.view(B, -1)  # (B, T * obs_dim)

        # Process through MLP
        embedding = self.mlp(obs_flat)  # (B, hidden_dim)
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

    Supports both image-based (obs_type='image') and state-based (obs_type='state') observations.
    """

    def __init__(
        self,
        obs_channels=3,
        action_dim=2,
        obs_window=2,
        action_chunk=16,
        hidden_dim=128,
        temperature=0.1,
        obs_type='image',  # 'image' or 'state'
        obs_dim=None,  # Required if obs_type='state'
        **kwargs,  # Additional options like use_deltas
    ):
        super().__init__()

        self.obs_window = obs_window
        self.action_chunk = action_chunk
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.obs_type = obs_type
        self.use_deltas = kwargs.get('use_deltas', False)

        # Choose encoder based on observation type
        if obs_type == 'image':
            self.obs_encoder = ContrastiveObsEncoder(obs_channels, hidden_dim)
        elif obs_type == 'state':
            if obs_dim is None:
                raise ValueError("obs_dim must be specified for state observations")
            use_deltas = kwargs.get('use_deltas', False)
            self.obs_encoder = ContrastiveStateObsEncoder(obs_dim, obs_window, hidden_dim, use_deltas=use_deltas)
        else:
            raise ValueError(f"Unknown obs_type: {obs_type}. Use 'image' or 'state'")

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
            # Mirror random action dimension (works for any action_dim)
            mirrored = positive_action.copy()
            dim_to_negate = np.random.randint(positive_action.shape[1])
            mirrored[:, dim_to_negate] = -mirrored[:, dim_to_negate]
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


class ContrastiveStateTAPDataset(Dataset):
    """
    Dataset for contrastive TAP training with state-based observations.

    Supports zarr format with 'obs' key (state vectors) instead of 'img'.

    Features:
    - Hard negatives from nearest-neighbor states (different obs, similar state)
    - Magnitude-preserving hard negatives (rotation, permutation)
    - Standard corrupted negatives (noise, permute, mirror)
    """

    def __init__(
        self,
        data_path,
        obs_window=2,
        action_chunk=16,
        n_negatives=15,
        hard_negative_ratio=0.5,
        noise_std=0.1,
        episode_filter=None,
        use_nn_negatives=True,  # Use nearest-neighbor hard negatives
        nn_negative_ratio=0.3,  # Fraction of negatives from NN states
        n_neighbors=50,  # Number of neighbors to consider
    ):
        self.obs_window = obs_window
        self.action_chunk = action_chunk
        self.n_negatives = n_negatives
        self.hard_negative_ratio = hard_negative_ratio
        self.noise_std = noise_std
        self.episode_filter = episode_filter
        self.use_nn_negatives = use_nn_negatives
        self.nn_negative_ratio = nn_negative_ratio
        self.n_neighbors = n_neighbors

        self.data_path = Path(data_path)
        self._load_data()
        self._create_index()

        # Build state index for nearest neighbor lookup
        if use_nn_negatives:
            self._build_state_index()

    def _load_data(self):
        root = zarr.open_group(str(self.data_path), mode='r')
        self.obs = root['data']['obs'][:]  # State observations
        self.actions = root['data']['action'][:]
        self.episode_ends = root['meta']['episode_ends'][:]
        self.obs_dim = self.obs.shape[-1]
        self.action_dim = self.actions.shape[-1]
        self.episode_starts = np.concatenate([[0], self.episode_ends[:-1]])
        self.num_episodes = len(self.episode_ends)
        print(f"Loaded {self.num_episodes} episodes, {len(self.obs)} frames")
        print(f"  obs_dim={self.obs_dim}, action_dim={self.action_dim}")

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

    def _build_state_index(self):
        """Build index for fast nearest-neighbor state lookup."""
        print("Building state index for NN hard negatives...")

        # Collect states at valid indices
        self.state_vectors = []
        for ep_idx, global_t in self.valid_indices:
            state = self.obs[global_t].astype(np.float32)
            self.state_vectors.append(state)
        self.state_vectors = np.array(self.state_vectors)

        # Normalize for cosine similarity
        norms = np.linalg.norm(self.state_vectors, axis=1, keepdims=True) + 1e-8
        self.state_vectors_normalized = self.state_vectors / norms

        print(f"  Built index with {len(self.state_vectors)} states")

    def _get_nn_action(self, idx, exclude_range=10):
        """Get action from a nearest-neighbor state (different time, similar state)."""
        query = self.state_vectors_normalized[idx]

        # Compute similarities to all states
        similarities = self.state_vectors_normalized @ query

        # Exclude nearby timesteps (within same episode context)
        for i in range(max(0, idx - exclude_range), min(len(similarities), idx + exclude_range + 1)):
            similarities[i] = -np.inf

        # Get top-k neighbors
        top_k_indices = np.argpartition(similarities, -self.n_neighbors)[-self.n_neighbors:]
        top_k_indices = top_k_indices[similarities[top_k_indices] > -np.inf]

        if len(top_k_indices) == 0:
            # Fallback to random
            return self._get_random_action_chunk(-1, -1)

        # Sample from neighbors
        nn_idx = np.random.choice(top_k_indices)
        _, global_t = self.valid_indices[nn_idx]
        return self._get_action_chunk(global_t)

    def __len__(self):
        return len(self.valid_indices)

    def _get_obs_window(self, global_t):
        start_t = global_t - self.obs_window + 1
        obs = self.obs[start_t:global_t + 1]
        return obs.astype(np.float32)

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
        """Create hard negative by corrupting positive action.

        Types 0-2: Standard corruptions (noise, permute, mirror)
        Types 3-4: Magnitude-preserving corruptions (rotation, axis swap)
        """
        if neg_type == 0:
            # Add noise
            return positive_action + np.random.randn(*positive_action.shape).astype(np.float32) * self.noise_std
        elif neg_type == 1:
            # Permute temporal order
            perm = np.random.permutation(self.action_chunk)
            return positive_action[perm]
        elif neg_type == 2:
            # Mirror dimension
            mirrored = positive_action.copy()
            dim_to_negate = np.random.randint(positive_action.shape[1])
            mirrored[:, dim_to_negate] = -mirrored[:, dim_to_negate]
            return mirrored
        elif neg_type == 3:
            # Rotation (magnitude-preserving, 2D)
            if positive_action.shape[1] == 2:
                angle = np.random.uniform(np.pi/6, np.pi)  # 30-180 degrees
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
                return (positive_action @ rot.T).astype(np.float32)
            else:
                # For higher dims, apply random rotation
                d = positive_action.shape[1]
                q, _ = np.linalg.qr(np.random.randn(d, d).astype(np.float32))
                return (positive_action @ q.T).astype(np.float32)
        else:
            # Axis swap (magnitude-preserving)
            swapped = positive_action.copy()
            d = positive_action.shape[1]
            perm = np.random.permutation(d)
            return swapped[:, perm].astype(np.float32)

    def __getitem__(self, idx):
        ep_idx, global_t = self.valid_indices[idx]

        obs = self._get_obs_window(global_t)
        positive_action = self._get_action_chunk(global_t)

        # Split negatives into categories
        n_nn = int(self.n_negatives * self.nn_negative_ratio) if self.use_nn_negatives else 0
        n_hard = int((self.n_negatives - n_nn) * self.hard_negative_ratio)
        n_random = self.n_negatives - n_nn - n_hard

        negatives = []

        # Nearest-neighbor negatives (actions from similar states)
        for _ in range(n_nn):
            negatives.append(self._get_nn_action(idx))

        # Hard negatives (corrupted versions of positive, including magnitude-preserving)
        for i in range(n_hard):
            neg_type = i % 5  # 5 types of hard negatives
            negatives.append(self._get_hard_negative(positive_action, neg_type))

        # Random negatives (from other times/episodes)
        for _ in range(n_random):
            negatives.append(self._get_random_action_chunk(ep_idx, global_t))

        all_actions = np.stack([positive_action] + negatives, axis=0)

        return {
            'obs': torch.from_numpy(obs),
            'actions': torch.from_numpy(all_actions),
        }


def create_contrastive_dataloaders(data_path, batch_size=32, train_split=0.9, obs_type='image', **kwargs):
    """Create train and validation dataloaders for contrastive TAP.

    Args:
        data_path: Path to zarr dataset
        batch_size: Batch size
        train_split: Fraction of episodes for training
        obs_type: 'image' for image observations, 'state' for state vectors
        **kwargs: Additional arguments for dataset
    """
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
    print(f"Observation type: {obs_type}")

    # Choose dataset class based on observation type
    if obs_type == 'image':
        DatasetClass = ContrastiveTAPDataset
        train_dataset = DatasetClass(data_path, episode_filter=train_episodes, augment=True, **kwargs)
        val_dataset = DatasetClass(data_path, episode_filter=val_episodes, augment=False, **kwargs)
    else:
        DatasetClass = ContrastiveStateTAPDataset
        train_dataset = DatasetClass(data_path, episode_filter=train_episodes, **kwargs)
        val_dataset = DatasetClass(data_path, episode_filter=val_episodes, **kwargs)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader


def build_contrastive_tap_model(config=None):
    """Build contrastive TAP model from config.

    Config keys:
        obs_type: 'image' or 'state' (default: 'image')
        obs_dim: Required if obs_type='state'
        obs_channels: For image observations (default: 3)
        action_dim: Action dimension (default: 2)
        obs_window: Observation window size (default: 2)
        action_chunk: Action chunk size (default: 16)
        hidden_dim: Hidden dimension (default: 128)
        temperature: Temperature for contrastive loss (default: 0.1)
        use_deltas: Use delta encoding for state observations (default: False)
    """
    config = config or {}

    return ContrastiveTAPScore(
        obs_channels=config.get("obs_channels", 3),
        action_dim=config.get("action_dim", 2),
        obs_window=config.get("obs_window", 2),
        action_chunk=config.get("action_chunk", 16),
        hidden_dim=config.get("hidden_dim", 128),
        temperature=config.get("temperature", 0.1),
        obs_type=config.get("obs_type", "image"),
        obs_dim=config.get("obs_dim"),
        use_deltas=config.get("use_deltas", False),
    )
