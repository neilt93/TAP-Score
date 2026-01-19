"""
TAP-Score Model: Temporal Action-Proposal Scorer

Scores (observation window, action chunk) pairs to determine
if the action looks expert-like for the given observation context.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ObservationEncoder(nn.Module):
    """CNN encoder for observation images with batch normalization."""

    def __init__(self, obs_channels=3, hidden_dim=128, use_batchnorm=True):
        super().__init__()

        # CNN with optional batch normalization for training stability
        if use_batchnorm:
            self.conv = nn.Sequential(
                nn.Conv2d(obs_channels, 32, 3, stride=2, padding=1),  # 48x48
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 24x24
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 12x12
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 6x6
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),  # 1x1
                nn.Flatten(),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(obs_channels, 32, 3, stride=2, padding=1),  # 48x48
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 24x24
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 12x12
                nn.ReLU(),
                nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 6x6
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),  # 1x1
                nn.Flatten(),
            )

        self.fc = nn.Linear(256, hidden_dim)

    def forward(self, obs):
        """
        Args:
            obs: (B, T, C, H, W) observation window
        Returns:
            features: (B, T, hidden_dim)
        """
        B, T, C, H, W = obs.shape

        # Flatten batch and time
        obs_flat = obs.view(B * T, C, H, W)
        features = self.conv(obs_flat)
        features = self.fc(features)

        # Reshape back
        features = features.view(B, T, -1)
        return features


class ActionEncoder(nn.Module):
    """MLP encoder for action chunks."""

    def __init__(self, action_dim=2, chunk_size=16, hidden_dim=128):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(action_dim * chunk_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, actions):
        """
        Args:
            actions: (B, H, action_dim) action chunk
        Returns:
            features: (B, hidden_dim)
        """
        B = actions.shape[0]
        actions_flat = actions.view(B, -1)
        return self.mlp(actions_flat)


class TAPScore(nn.Module):
    """
    Temporal Action-Proposal Score model.

    Scores (observation window, action chunk) → [0, 1]
    High score = looks like expert action
    Low score = off-manifold / OOD
    """

    def __init__(
        self,
        obs_channels=3,
        action_dim=2,
        obs_window=2,
        action_chunk=16,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
    ):
        super().__init__()

        self.obs_window = obs_window
        self.action_chunk = action_chunk
        self.hidden_dim = hidden_dim

        # Encoders
        self.obs_encoder = ObservationEncoder(obs_channels, hidden_dim)
        self.action_encoder = ActionEncoder(action_dim, action_chunk, hidden_dim)

        # Positional embedding for observation window
        self.pos_embed = nn.Parameter(torch.randn(1, obs_window + 1, hidden_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, obs, actions):
        """
        Args:
            obs: (B, T, C, H, W) observation window
            actions: (B, H, action_dim) action chunk

        Returns:
            score: (B,) scores in [0, 1]
        """
        B = obs.shape[0]

        # Encode observations: (B, T, hidden_dim)
        obs_features = self.obs_encoder(obs)

        # Encode actions: (B, hidden_dim)
        action_features = self.action_encoder(actions)

        # Concatenate: (B, T+1, hidden_dim)
        # Action features as additional token
        action_token = action_features.unsqueeze(1)  # (B, 1, hidden_dim)
        tokens = torch.cat([obs_features, action_token], dim=1)

        # Add positional embedding
        tokens = tokens + self.pos_embed[:, :tokens.shape[1], :]

        # Transformer
        features = self.transformer(tokens)

        # Use last token (action) for classification
        features = features[:, -1, :]  # (B, hidden_dim)

        # Classify
        logits = self.classifier(features).squeeze(-1)  # (B,)
        scores = torch.sigmoid(logits)

        return scores

    def get_loss(self, obs, actions, labels):
        """
        Compute binary cross-entropy loss.

        Args:
            obs: (B, T, C, H, W)
            actions: (B, H, action_dim)
            labels: (B,) binary labels (1 = positive, 0 = negative)

        Returns:
            loss: scalar
        """
        scores = self.forward(obs, actions)
        loss = F.binary_cross_entropy(scores, labels.float())
        return loss


def build_tap_model(config=None):
    """Build TAP model from config."""
    config = config or {}

    return TAPScore(
        obs_channels=config.get("obs_channels", 3),
        action_dim=config.get("action_dim", 2),
        obs_window=config.get("obs_window", 2),
        action_chunk=config.get("action_chunk", 16),
        hidden_dim=config.get("hidden_dim", 128),
        num_layers=config.get("num_layers", 2),
        num_heads=config.get("num_heads", 4),
    )
