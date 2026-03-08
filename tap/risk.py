"""
Risk-based TAP-Score: Binary failure predictor P(fail_within_H | obs, action).

Instead of contrastive ranking against negatives, this model directly predicts
whether the current (obs, action) pair will lead to failure within H env steps.

Trained on real rollout data from both clean and perturbed regimes.
"""
from __future__ import annotations

import glob
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, WeightedRandomSampler


class RiskObsEncoder(nn.Module):
    """MLP encoder for state observations (no L2 normalization)."""

    def __init__(self, obs_dim: int, obs_window: int = 2, hidden_dim: int = 128):
        super().__init__()
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

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (B, T, obs_dim)
        Returns:
            (B, hidden_dim) — un-normalized embedding
        """
        B = obs.shape[0]
        return self.mlp(obs.view(B, -1))


class RiskActionEncoder(nn.Module):
    """MLP encoder for action chunks (no L2 normalization)."""

    def __init__(self, action_dim: int = 7, chunk_size: int = 8, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(action_dim * chunk_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            action: (B, H, action_dim)
        Returns:
            (B, hidden_dim) — un-normalized embedding
        """
        B = action.shape[0]
        return self.mlp(action.view(B, -1))


class RiskTAPScore(nn.Module):
    """Binary risk predictor: P(fail_within_H | obs, action).

    Reuses the same encoder architectures as ContrastiveTAPScore but
    with a concatenation + sigmoid head instead of dot-product scoring.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 7,
        obs_window: int = 2,
        action_chunk: int = 8,
        hidden_dim: int = 128,
        obs_only: bool = False,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.obs_window = obs_window
        self.action_chunk = action_chunk
        self.hidden_dim = hidden_dim
        self.obs_only = obs_only

        self.obs_encoder = RiskObsEncoder(obs_dim, obs_window, hidden_dim)
        if not obs_only:
            self.action_encoder = RiskActionEncoder(action_dim, action_chunk, hidden_dim)
            risk_input_dim = hidden_dim * 2
        else:
            self.action_encoder = None
            risk_input_dim = hidden_dim
        self.risk_head = nn.Sequential(
            nn.Linear(risk_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, obs: torch.Tensor, action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            obs:    (B, obs_window, obs_dim)
            action: (B, action_chunk, action_dim) — ignored when obs_only=True
        Returns:
            risk_logit: (B,) — raw logit (positive = more likely to fail)
        """
        obs_emb = self.obs_encoder(obs)      # (B, hidden_dim)
        if self.obs_only:
            combined = obs_emb
        else:
            act_emb = self.action_encoder(action)  # (B, hidden_dim)
            combined = torch.cat([obs_emb, act_emb], dim=-1)
        logit = self.risk_head(combined)
        return logit.squeeze(-1)

    def predict_risk(
        self, obs: torch.Tensor, action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return P(fail) in [0, 1]."""
        return torch.sigmoid(self.forward(obs, action))

    def encode(
        self, obs: torch.Tensor, action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return per-step embedding before risk head."""
        obs_emb = self.obs_encoder(obs)      # (B, hidden_dim)
        if self.obs_only:
            return obs_emb
        act_emb = self.action_encoder(action)  # (B, hidden_dim)
        return torch.cat([obs_emb, act_emb], dim=-1)


class TemporalRiskModel(nn.Module):
    """Episode-level risk from a sequence of per-step embeddings.

    Uses a frozen or fine-tuned RiskTAPScore for per-step encoding,
    then a GRU over time to learn temporal risk patterns (sustained drift
    vs sharp events). Outputs episode-level P(fail) and per-step risk.
    """

    def __init__(
        self,
        step_model: RiskTAPScore,
        gru_hidden: int = 64,
        gru_layers: int = 1,
        freeze_step_model: bool = True,
    ):
        super().__init__()
        self.step_model = step_model
        self.freeze_step_model = freeze_step_model
        if freeze_step_model:
            for p in self.step_model.parameters():
                p.requires_grad = False

        emb_dim = step_model.hidden_dim * 2  # concat of obs + action embeddings
        self.gru = nn.GRU(emb_dim, gru_hidden, num_layers=gru_layers, batch_first=True)
        # Per-step risk head (from GRU hidden state)
        self.step_head = nn.Linear(gru_hidden, 1)
        # Episode-level risk head (from final GRU hidden state)
        self.episode_head = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden),
            nn.ReLU(),
            nn.Linear(gru_hidden, 1),
        )

    def forward(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            obs_seq:    (B, T, obs_window, obs_dim)
            action_seq: (B, T, action_chunk, action_dim)
            lengths:    (B,) — actual sequence lengths (for masking)
        Returns:
            dict with:
                step_logits:    (B, T) — per-step fail_within_H logits
                episode_logit:  (B,) — episode-level failure logit
        """
        B, T = obs_seq.shape[:2]

        # Encode each step
        obs_flat = obs_seq.reshape(B * T, *obs_seq.shape[2:])
        act_flat = action_seq.reshape(B * T, *action_seq.shape[2:])

        if self.freeze_step_model:
            with torch.no_grad():
                emb = self.step_model.encode(obs_flat, act_flat)  # (B*T, emb_dim)
        else:
            emb = self.step_model.encode(obs_flat, act_flat)

        emb = emb.reshape(B, T, -1)  # (B, T, emb_dim)

        # Pack if variable length
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                emb, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False
            )
            gru_out, h_n = self.gru(packed)
            gru_out, _ = nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True, total_length=T)
        else:
            gru_out, h_n = self.gru(emb)  # (B, T, gru_hidden), (1, B, gru_hidden)

        # Per-step risk
        step_logits = self.step_head(gru_out).squeeze(-1)  # (B, T)

        # Episode-level risk from final hidden state
        h_final = h_n[-1]  # (B, gru_hidden)
        episode_logit = self.episode_head(h_final).squeeze(-1)  # (B,)

        return {
            "step_logits": step_logits,
            "episode_logit": episode_logit,
        }

    def predict_episode_risk(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return P(episode_fail) in [0, 1]."""
        out = self.forward(obs_seq, action_seq, lengths)
        return torch.sigmoid(out["episode_logit"])

    def predict_step_risk(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return per-step P(fail_within_H) in [0, 1]. Shape: (B, T)."""
        out = self.forward(obs_seq, action_seq, lengths)
        return torch.sigmoid(out["step_logits"])


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class RiskRolloutDataset(Dataset):
    """Loads per-episode NPZ files and creates (obs, action, label) samples.

    Each NPZ contains one rollout with decision-level data.

    Label modes:
    - "onset_window": label=1 if step in [onset, onset+H] AND episode failed.
    - "episode_window": label=episode_outcome for all steps in [onset, onset+W].
      Produces more positives by using the episode outcome as target for every
      post-onset step in the early window. W controlled by fail_horizon.
    """

    def __init__(
        self,
        rollout_dir: str,
        obs_window: int = 2,
        action_chunk: int = 8,
        fail_horizon: int = 64,
        regime_filter: Optional[str] = None,
        task: Optional[str] = None,
        label_mode: str = "onset_window",
        post_onset_only: bool = False,
    ):
        self.obs_window = obs_window
        self.action_chunk = action_chunk
        self.fail_horizon = fail_horizon
        self.label_mode = label_mode
        self.post_onset_only = post_onset_only

        npz_files = sorted(glob.glob(str(Path(rollout_dir) / "*.npz")))
        if not npz_files:
            raise FileNotFoundError(f"No .npz files in {rollout_dir}")

        self.samples: List[dict] = []
        self.episode_ids: List[int] = []  # episode index per sample (for splitting)

        for ep_idx, npz_path in enumerate(npz_files):
            if Path(npz_path).name.startswith("_"):
                continue  # skip checkpoint files
            data = np.load(npz_path, allow_pickle=True)
            regime = str(data["regime"])
            if regime_filter is not None and regime != regime_filter:
                continue

            episode_success = bool(data["episode_success"])
            obs_seq = data["obs_seq"]          # (T, obs_dim)
            action_chunks = data["action_chunks"]  # (n_decisions, H, action_dim)
            decision_steps = data["decision_steps"]  # (n_decisions,)

            # Get onset for this episode
            onset = self._get_onset(data, regime, task)

            for d_idx in range(len(decision_steps)):
                step = int(decision_steps[d_idx])

                # Skip pre-onset steps if requested
                if post_onset_only and onset is not None and step < onset:
                    continue

                # Build obs window: obs_window frames ending at step
                obs_indices = []
                for w in range(obs_window):
                    t = step - (obs_window - 1 - w)
                    t = max(0, min(t, len(obs_seq) - 1))
                    obs_indices.append(t)
                obs_win = obs_seq[obs_indices].astype(np.float32)  # (obs_window, obs_dim)

                # Action chunk: truncate or pad to action_chunk
                act = action_chunks[d_idx]  # (H, action_dim)
                if act.shape[0] >= action_chunk:
                    act = act[:action_chunk]
                else:
                    pad = np.tile(act[-1:], (action_chunk - act.shape[0], 1))
                    act = np.concatenate([act, pad], axis=0)
                act = act.astype(np.float32)

                # Label depends on mode
                if label_mode == "episode_window":
                    # Episode-outcome label for steps in [onset, onset + W].
                    # Steps outside the window are excluded (not labeled 0).
                    if onset is None:
                        # Clean regime: include all steps, label = episode outcome
                        label = 0.0 if episode_success else 1.0
                    elif onset <= step <= onset + fail_horizon:
                        label = 0.0 if episode_success else 1.0
                    else:
                        continue  # skip steps outside the early window
                elif label_mode == "onset_window":
                    # Original: label=1 only if failed AND in [onset, onset+H]
                    if episode_success:
                        label = 0.0
                    elif onset is None:
                        label = 0.0
                    else:
                        label = 1.0 if onset <= step <= onset + fail_horizon else 0.0
                else:
                    raise ValueError(f"Unknown label_mode: {label_mode}")

                self.samples.append({
                    "obs": obs_win,
                    "action": act,
                    "label": label,
                })
                self.episode_ids.append(ep_idx)

        self.episode_ids = np.array(self.episode_ids)
        n_fail = sum(1 for s in self.samples if s["label"] > 0.5)
        n_safe = len(self.samples) - n_fail
        print(f"RiskRolloutDataset: {len(self.samples)} samples "
              f"({n_safe} safe, {n_fail} fail) from {len(npz_files)} episodes"
              f", mode={label_mode}"
              + (f", regime={regime_filter}" if regime_filter else ""))

    @staticmethod
    def _get_onset(data, regime: str, task: Optional[str]) -> Optional[int]:
        """Get perturbation onset for an episode. Returns None for clean."""
        if regime == "clean":
            return None
        if "onset" in data.keys():
            return int(data["onset"])
        # Fallback: compute from regime definition
        if task is not None:
            from scripts.collect_risk_rollouts import get_onset as _get_onset
            seed = int(data["seed"]) if "seed" in data.keys() else 0
            return _get_onset(regime, task, seed=seed)
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "obs": torch.from_numpy(s["obs"]),
            "action": torch.from_numpy(s["action"]),
            "label": torch.tensor(s["label"], dtype=torch.float32),
        }


class EpisodeSequenceDataset(Dataset):
    """Loads per-episode NPZ files and returns full sequences for temporal modeling.

    Each item is (obs_seq, action_seq, step_labels, episode_label, length).
    """

    def __init__(
        self,
        rollout_dir: str,
        obs_window: int = 2,
        action_chunk: int = 8,
        fail_horizon: int = 64,
        max_seq_len: int = 50,
        task: Optional[str] = None,
    ):
        self.obs_window = obs_window
        self.action_chunk = action_chunk
        self.fail_horizon = fail_horizon
        self.max_seq_len = max_seq_len

        npz_files = sorted(glob.glob(str(Path(rollout_dir) / "ep_*.npz")))
        if not npz_files:
            raise FileNotFoundError(f"No ep_*.npz files in {rollout_dir}")

        self.episodes: List[dict] = []

        for npz_path in npz_files:
            data = np.load(npz_path, allow_pickle=True)
            episode_success = bool(data["episode_success"])
            regime = str(data["regime"]) if "regime" in data.keys() else "clean"
            obs_seq = data["obs_seq"]              # (T_obs, obs_dim)
            action_chunks = data["action_chunks"]  # (n_decisions, H, action_dim)
            decision_steps = data["decision_steps"]  # (n_decisions,)

            if len(decision_steps) == 0:
                continue

            # Get onset for this episode
            onset = RiskRolloutDataset._get_onset(data, regime, task)

            n_decisions = len(decision_steps)
            obs_dim = obs_seq.shape[-1]
            act_dim = action_chunks.shape[-1]

            # Build per-decision obs windows and action chunks
            obs_windows = np.zeros((n_decisions, obs_window, obs_dim), dtype=np.float32)
            act_chunks = np.zeros((n_decisions, action_chunk, act_dim), dtype=np.float32)
            step_labels = np.zeros(n_decisions, dtype=np.float32)

            for d_idx in range(n_decisions):
                step = int(decision_steps[d_idx])

                # Obs window
                for w in range(obs_window):
                    t = step - (obs_window - 1 - w)
                    t = max(0, min(t, len(obs_seq) - 1))
                    obs_windows[d_idx, w] = obs_seq[t]

                # Action chunk (pad if short)
                act = action_chunks[d_idx]
                if act.shape[0] >= action_chunk:
                    act_chunks[d_idx] = act[:action_chunk]
                else:
                    act_chunks[d_idx, :act.shape[0]] = act
                    act_chunks[d_idx, act.shape[0]:] = act[-1:]

                # Step-level label: onset-window
                if episode_success:
                    step_labels[d_idx] = 0.0
                elif onset is None:
                    step_labels[d_idx] = 0.0
                else:
                    step_labels[d_idx] = 1.0 if onset <= step <= onset + fail_horizon else 0.0

            self.episodes.append({
                "obs_windows": obs_windows,
                "act_chunks": act_chunks,
                "step_labels": step_labels,
                "episode_label": 0.0 if episode_success else 1.0,
                "length": n_decisions,
            })

        n_fail = sum(1 for e in self.episodes if e["episode_label"] > 0.5)
        print(f"EpisodeSequenceDataset: {len(self.episodes)} episodes "
              f"({len(self.episodes) - n_fail} success, {n_fail} fail)")

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        ep = self.episodes[idx]
        T = min(ep["length"], self.max_seq_len)

        # Truncate to max_seq_len
        obs = ep["obs_windows"][:T]
        act = ep["act_chunks"][:T]
        step_labels = ep["step_labels"][:T]

        # Pad to max_seq_len
        if T < self.max_seq_len:
            pad_obs = np.zeros((self.max_seq_len - T, *obs.shape[1:]), dtype=np.float32)
            pad_act = np.zeros((self.max_seq_len - T, *act.shape[1:]), dtype=np.float32)
            pad_labels = np.zeros(self.max_seq_len - T, dtype=np.float32)
            obs = np.concatenate([obs, pad_obs], axis=0)
            act = np.concatenate([act, pad_act], axis=0)
            step_labels = np.concatenate([step_labels, pad_labels], axis=0)

        return {
            "obs_seq": torch.from_numpy(obs),          # (max_seq_len, obs_window, obs_dim)
            "action_seq": torch.from_numpy(act),        # (max_seq_len, action_chunk, action_dim)
            "step_labels": torch.from_numpy(step_labels),  # (max_seq_len,)
            "episode_label": torch.tensor(ep["episode_label"], dtype=torch.float32),
            "length": torch.tensor(T, dtype=torch.long),
        }


class MultiRegimeRiskDataset(Dataset):
    """Wraps a single rollout directory with multiple regimes, balanced sampling."""

    def __init__(
        self,
        rollout_dir: str,
        regimes: Optional[List[str]] = None,
        obs_window: int = 2,
        action_chunk: int = 8,
        fail_horizon: int = 64,
        task: Optional[str] = None,
        label_mode: str = "onset_window",
        post_onset_only: bool = False,
    ):
        if regimes is None:
            regimes = ["clean", "zero80", "freeze80", "dropout80"]

        self.regime_datasets: List[RiskRolloutDataset] = []
        for regime in regimes:
            try:
                ds = RiskRolloutDataset(
                    rollout_dir, obs_window=obs_window,
                    action_chunk=action_chunk, fail_horizon=fail_horizon,
                    regime_filter=regime, task=task,
                    label_mode=label_mode, post_onset_only=post_onset_only,
                )
                if len(ds) > 0:
                    self.regime_datasets.append(ds)
            except FileNotFoundError:
                print(f"  Warning: no data for regime '{regime}', skipping")

        if not self.regime_datasets:
            raise ValueError(f"No valid regime data in {rollout_dir}")

        # Concatenate all samples
        self.all_samples = []
        self.all_episode_ids = []
        offset = 0
        for ds in self.regime_datasets:
            self.all_samples.extend(ds.samples)
            self.all_episode_ids.extend(ds.episode_ids + offset)
            offset += (ds.episode_ids.max() + 1) if len(ds.episode_ids) > 0 else 0

        self.all_episode_ids = np.array(self.all_episode_ids)
        n_fail = sum(1 for s in self.all_samples if s["label"] > 0.5)
        print(f"MultiRegimeRiskDataset: {len(self.all_samples)} total samples, "
              f"{n_fail} fail ({100 * n_fail / max(len(self.all_samples), 1):.1f}%)")

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):
        s = self.all_samples[idx]
        return {
            "obs": torch.from_numpy(s["obs"]),
            "action": torch.from_numpy(s["action"]),
            "label": torch.tensor(s["label"], dtype=torch.float32),
        }

    @property
    def episode_ids(self):
        return self.all_episode_ids


# ---------------------------------------------------------------------------
# Hard negative mining
# ---------------------------------------------------------------------------

def mine_hard_negatives(
    model: RiskTAPScore,
    dataset: Dataset,
    device: torch.device,
    top_k_ratio: float = 0.3,
    batch_size: int = 256,
) -> np.ndarray:
    """Score all failure samples; return indices of hardest (model thinks safe).

    Returns sample indices into the dataset that should be oversampled.
    """
    model.eval()
    fail_indices = []
    fail_scores = []

    # Collect all failure sample indices
    for idx in range(len(dataset)):
        s = dataset[idx]
        if s["label"].item() > 0.5:
            fail_indices.append(idx)

    if not fail_indices:
        return np.array([], dtype=np.int64)

    # Score in batches
    for start in range(0, len(fail_indices), batch_size):
        batch_idx = fail_indices[start:start + batch_size]
        obs_list, act_list = [], []
        for idx in batch_idx:
            s = dataset[idx]
            obs_list.append(s["obs"])
            act_list.append(s["action"])

        obs_batch = torch.stack(obs_list).to(device)
        act_batch = torch.stack(act_list).to(device)

        with torch.no_grad():
            risk = model.predict_risk(obs_batch, act_batch)
        fail_scores.extend(risk.cpu().numpy().tolist())

    fail_scores = np.array(fail_scores)
    # Hard negatives: failure samples where model predicts LOW risk (thinks safe)
    n_hard = max(1, int(len(fail_indices) * top_k_ratio))
    hard_order = np.argsort(fail_scores)[:n_hard]  # lowest risk first
    hard_indices = np.array(fail_indices)[hard_order]

    print(f"  Hard mining: {n_hard}/{len(fail_indices)} failure samples selected "
          f"(risk range [{fail_scores[hard_order].min():.3f}, {fail_scores[hard_order].max():.3f}])")

    return hard_indices


def build_weighted_sampler(
    dataset: Dataset,
    hard_indices: np.ndarray,
    oversample_factor: float = 3.0,
) -> WeightedRandomSampler:
    """Create a weighted sampler that oversamples hard failure examples."""
    weights = np.ones(len(dataset), dtype=np.float64)
    weights[hard_indices] = oversample_factor
    return WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
