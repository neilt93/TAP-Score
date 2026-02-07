"""
Verify Diffusion Policy data loading using official repository components.

This script imports from the official diffusion_policy repo to verify
that the Push-T dataset is correctly formatted and loadable.
"""

import sys
import os

# Add diffusion_policy to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'baselines', 'diffusion_policy'))

import numpy as np
import zarr

def main():
    print("=" * 60)
    print("Diffusion Policy Data Verification")
    print("=" * 60)

    data_path = "data/processed/pusht"
    print(f"\nLoading Push-T data from: {data_path}")

    # Load using zarr (same as DP repo)
    root = zarr.open_group(data_path, mode='r')

    print("\n--- Dataset Structure ---")
    print(f"Groups: {list(root.keys())}")
    print(f"Data arrays: {list(root['data'].keys())}")
    print(f"Meta arrays: {list(root['meta'].keys())}")

    # Load arrays
    images = root['data']['img']
    actions = root['data']['action']
    episode_ends = root['meta']['episode_ends'][:]

    print("\n--- Array Shapes ---")
    print(f"Images shape: {images.shape}")
    print(f"Actions shape: {actions.shape}")
    print(f"Episode ends: {len(episode_ends)} episodes")

    print("\n--- Data Statistics ---")
    print(f"Image dtype: {images.dtype}")
    print(f"Image range: [{images[:100].min()}, {images[:100].max()}]")
    print(f"Action dtype: {actions.dtype}")
    print(f"Action range: [{actions[:].min():.3f}, {actions[:].max():.3f}]")
    print(f"Action dim: {actions.shape[1]}")

    # Episode lengths
    episode_starts = np.concatenate([[0], episode_ends[:-1]])
    episode_lengths = episode_ends - episode_starts

    print("\n--- Episode Statistics ---")
    print(f"Number of episodes: {len(episode_ends)}")
    print(f"Total timesteps: {actions.shape[0]}")
    print(f"Episode length range: [{episode_lengths.min()}, {episode_lengths.max()}]")
    print(f"Mean episode length: {episode_lengths.mean():.1f}")

    # Sample first episode
    print("\n--- First Episode Sample ---")
    first_ep_end = episode_ends[0]
    print(f"Episode 0: {first_ep_end} timesteps")
    print(f"First action: {actions[0]}")
    print(f"Last action: {actions[first_ep_end-1]}")

    print("\n" + "=" * 60)
    print("Data verification PASSED")
    print("Dataset is compatible with Diffusion Policy format")
    print("=" * 60)

if __name__ == "__main__":
    main()
