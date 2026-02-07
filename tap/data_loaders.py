"""Unified data loading for multiple benchmarks."""

from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import zarr

from .benchmarks import get_benchmark_config


def load_zarr_episodes(data_path: str) -> List[Dict[str, np.ndarray]]:
    """Load episodes from zarr format (Push-T, Kitchen, Block Push)."""
    root = zarr.open_group(str(data_path), mode='r')

    # Get data arrays
    images = root['data']['img'][:]
    actions = root['data']['action'][:]
    episode_ends = root['meta']['episode_ends'][:]

    # Split into episodes
    episode_starts = np.concatenate([[0], episode_ends[:-1]])
    episodes = []
    for i in range(len(episode_ends)):
        start, end = episode_starts[i], episode_ends[i]
        episodes.append({
            'images': images[start:end],
            'actions': actions[start:end]
        })

    return episodes


def load_hdf5_episodes(data_path: str) -> List[Dict[str, np.ndarray]]:
    """Load episodes from HDF5 format (Robomimic)."""
    import h5py

    episodes = []
    with h5py.File(data_path, 'r') as f:
        demos = f['data']
        demo_keys = sorted([k for k in demos.keys() if k.startswith('demo_')])

        for demo_key in demo_keys:
            demo = demos[demo_key]

            # Robomimic stores images under obs/
            # Try agentview_image first, then other camera options
            obs = demo['obs']
            if 'agentview_image' in obs:
                images = obs['agentview_image'][:]
            elif 'image' in obs:
                images = obs['image'][:]
            else:
                # Find any image key
                image_keys = [k for k in obs.keys() if 'image' in k.lower()]
                if image_keys:
                    images = obs[image_keys[0]][:]
                else:
                    raise ValueError(f"No image data found in {demo_key}")

            actions = demo['actions'][:]

            episodes.append({
                'images': images,
                'actions': actions.astype(np.float32)
            })

    return episodes


def load_episodes(data_path: str, benchmark: str) -> List[Dict[str, np.ndarray]]:
    """
    Load episodes for a given benchmark using the appropriate format loader.

    Args:
        data_path: Path to the data directory or file
        benchmark: Benchmark name (pusht, lift, kitchen, blockpush)

    Returns:
        List of episode dictionaries with 'images' and 'actions' keys
    """
    config = get_benchmark_config(benchmark)
    data_format = config["data_format"]

    if data_format == "zarr":
        episodes = load_zarr_episodes(data_path)
    elif data_format == "hdf5":
        episodes = load_hdf5_episodes(data_path)
    else:
        raise ValueError(f"Unknown data format: {data_format}")

    # Validate episodes
    if len(episodes) == 0:
        raise ValueError(f"No episodes loaded from {data_path}")

    # Check action dimension matches config
    expected_action_dim = config["action_dim"]
    actual_action_dim = episodes[0]['actions'].shape[1]
    if actual_action_dim != expected_action_dim:
        print(f"Warning: Expected action_dim={expected_action_dim}, got {actual_action_dim}")

    return episodes


def get_data_stats(episodes: List[Dict[str, np.ndarray]]) -> Dict[str, Any]:
    """Get statistics about loaded episodes."""
    n_episodes = len(episodes)
    episode_lengths = [len(ep['actions']) for ep in episodes]

    all_actions = np.concatenate([ep['actions'] for ep in episodes], axis=0)

    return {
        'n_episodes': n_episodes,
        'total_timesteps': sum(episode_lengths),
        'episode_length_range': (min(episode_lengths), max(episode_lengths)),
        'mean_episode_length': np.mean(episode_lengths),
        'action_dim': all_actions.shape[1],
        'action_range': (all_actions.min(), all_actions.max()),
        'image_shape': episodes[0]['images'].shape[1:],
    }


def print_data_stats(episodes: List[Dict[str, np.ndarray]], benchmark: str):
    """Print statistics about loaded data."""
    stats = get_data_stats(episodes)
    config = get_benchmark_config(benchmark)

    print(f"\n{'='*50}")
    print(f"Loaded {config['name']} ({benchmark})")
    print(f"{'='*50}")
    print(f"Episodes:        {stats['n_episodes']}")
    print(f"Total timesteps: {stats['total_timesteps']}")
    print(f"Episode lengths: {stats['episode_length_range'][0]} - {stats['episode_length_range'][1]}")
    print(f"Mean length:     {stats['mean_episode_length']:.1f}")
    print(f"Action dim:      {stats['action_dim']} (expected: {config['action_dim']})")
    print(f"Action range:    [{stats['action_range'][0]:.3f}, {stats['action_range'][1]:.3f}]")
    print(f"Image shape:     {stats['image_shape']}")
    print(f"{'='*50}\n")
