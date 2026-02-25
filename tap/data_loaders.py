"""Unified data loading for multiple benchmarks."""

from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import zarr

from .benchmarks import get_benchmark_config


def load_zarr_image_episodes(data_path: str) -> List[Dict[str, np.ndarray]]:
    """Load episodes from zarr format with image observations (Push-T)."""
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
            'actions': actions[start:end],
            'obs_type': 'image'
        })

    return episodes


def load_zarr_state_episodes(data_path: str) -> List[Dict[str, np.ndarray]]:
    """Load episodes from zarr format with state observations (Block Push)."""
    root = zarr.open_group(str(data_path), mode='r')

    # Get data arrays
    obs = root['data']['obs'][:]
    actions = root['data']['action'][:]
    episode_ends = root['meta']['episode_ends'][:]

    # Split into episodes
    episode_starts = np.concatenate([[0], episode_ends[:-1]])
    episodes = []
    for i in range(len(episode_ends)):
        start, end = episode_starts[i], episode_ends[i]
        episodes.append({
            'obs': obs[start:end],
            'actions': actions[start:end],
            'obs_type': 'state'
        })

    return episodes


def load_zarr_episodes(data_path: str, obs_type: str = 'image') -> List[Dict[str, np.ndarray]]:
    """Load episodes from zarr format, auto-detecting observation type."""
    if obs_type == 'image':
        return load_zarr_image_episodes(data_path)
    else:
        return load_zarr_state_episodes(data_path)


def load_hdf5_episodes(data_path: str) -> List[Dict[str, np.ndarray]]:
    """Load episodes from HDF5 format (Robomimic) with image observations."""
    import h5py

    episodes = []
    with h5py.File(data_path, 'r') as f:
        demos = f['data']
        demo_keys = sorted([k for k in demos.keys() if k.startswith('demo_')],
                           key=lambda k: int(k.split('_')[1]))

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


def load_hdf5_lowdim_episodes(data_path: str, obs_keys: List[str]) -> List[Dict[str, np.ndarray]]:
    """Load episodes from HDF5 with lowdim state observations (Robomimic).

    Concatenates the specified obs_keys into a flat state vector per timestep.
    Validates shapes and prints summary statistics.

    Args:
        data_path: Path to HDF5 file
        obs_keys: List of observation keys to concatenate (e.g. ['object', 'robot0_eef_pos', ...])

    Returns:
        List of episode dicts with 'obs' (T, obs_dim) and 'actions' (T, action_dim)
    """
    import h5py

    episodes = []
    with h5py.File(data_path, 'r') as f:
        demos = f['data']
        demo_keys = sorted((k for k in demos.keys() if k.startswith('demo_')),
                           key=lambda k: int(k.split('_')[1]))

        # Print obs key shapes from first demo for validation
        first_demo = demos[demo_keys[0]]
        print(f"  HDF5 obs keys in first demo:")
        total_obs_dim = 0
        for key in obs_keys:
            if key not in first_demo['obs']:
                available = list(first_demo['obs'].keys())
                raise ValueError(f"obs key '{key}' not found in HDF5. Available: {available}")
            shape = first_demo['obs'][key].shape
            print(f"    {key}: {shape}")
            total_obs_dim += shape[-1]
        print(f"  Concatenated obs_dim: {total_obs_dim}")
        print(f"  Action shape: {first_demo['actions'].shape}")
        print(f"  Number of demos: {len(demo_keys)}")

        for demo_key in demo_keys:
            demo = demos[demo_key]
            obs_parts = [demo['obs'][k][:] for k in obs_keys]
            obs = np.concatenate(obs_parts, axis=-1)  # (T, obs_dim)
            actions = demo['actions'][:]
            episodes.append({
                'obs': obs.astype(np.float32),
                'actions': actions.astype(np.float32),
                'obs_type': 'state',
            })

    # Print episode length stats
    lengths = [len(ep['actions']) for ep in episodes]
    print(f"  Episode lengths: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")

    return episodes


def load_episodes(data_path: str, benchmark: str) -> List[Dict[str, np.ndarray]]:
    """
    Load episodes for a given benchmark using the appropriate format loader.

    Args:
        data_path: Path to the data directory or file
        benchmark: Benchmark name (pusht, blockpush, kitchen)

    Returns:
        List of episode dictionaries with 'images'/'obs' and 'actions' keys
    """
    config = get_benchmark_config(benchmark)
    data_format = config["data_format"]
    obs_type = config.get("obs_type", "image")

    if data_format == "zarr":
        episodes = load_zarr_episodes(data_path, obs_type=obs_type)
    elif data_format == "hdf5":
        if obs_type == "state" and "obs_keys" in config:
            episodes = load_hdf5_lowdim_episodes(data_path, config["obs_keys"])
        else:
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

    # Determine observation type
    obs_type = episodes[0].get('obs_type', 'image')

    stats = {
        'n_episodes': n_episodes,
        'total_timesteps': sum(episode_lengths),
        'episode_length_range': (min(episode_lengths), max(episode_lengths)),
        'mean_episode_length': np.mean(episode_lengths),
        'action_dim': all_actions.shape[1],
        'action_range': (all_actions.min(), all_actions.max()),
        'obs_type': obs_type,
    }

    if obs_type == 'image':
        stats['obs_shape'] = episodes[0]['images'].shape[1:]
    else:
        stats['obs_shape'] = (episodes[0]['obs'].shape[1],)

    return stats


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
    print(f"Obs type:        {stats['obs_type']}")
    print(f"Obs shape:       {stats['obs_shape']}")
    print(f"{'='*50}\n")
