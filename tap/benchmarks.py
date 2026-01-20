"""Benchmark configuration registry for TAP-Score evaluation."""

from pathlib import Path

# Benchmark configurations
BENCHMARKS = {
    "pusht": {
        "name": "Push-T",
        "action_dim": 2,
        "obs_channels": 3,
        "image_size": 96,
        "obs_window": 2,
        "action_chunk": 16,
        "data_format": "zarr",
        "data_subdir": "pusht",
        "download_url": "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip",
        "description": "2D pushing task with T-block",
    },
    "lift": {
        "name": "Robomimic Lift",
        "action_dim": 7,
        "obs_channels": 3,
        "image_size": 84,
        "obs_window": 2,
        "action_chunk": 16,
        "data_format": "hdf5",
        "data_subdir": "lift",
        "download_url": "https://diffusion-policy.cs.columbia.edu/data/training/robomimic_image/lift_ph_image.hdf5",
        "description": "7-DoF robotic arm lifting task",
    },
    "kitchen": {
        "name": "Kitchen",
        "action_dim": 9,
        "obs_channels": 3,
        "image_size": 64,
        "obs_window": 2,
        "action_chunk": 16,
        "data_format": "zarr",
        "data_subdir": "kitchen",
        "download_url": "https://diffusion-policy.cs.columbia.edu/data/training/kitchen.zip",
        "description": "9-DoF kitchen manipulation task",
    },
    "blockpush": {
        "name": "Block Push",
        "action_dim": 2,
        "obs_channels": 3,
        "image_size": 96,
        "obs_window": 2,
        "action_chunk": 16,
        "data_format": "zarr",
        "data_subdir": "blockpush",
        "download_url": "https://diffusion-policy.cs.columbia.edu/data/training/blockpush.zip",
        "description": "2D block pushing task",
    },
}


def get_benchmark_config(benchmark: str) -> dict:
    """Get configuration for a benchmark."""
    if benchmark not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {benchmark}. Available: {list(BENCHMARKS.keys())}")
    return BENCHMARKS[benchmark]


def get_data_path(benchmark: str, data_root: str = "data/processed") -> Path:
    """Get the data path for a benchmark."""
    config = get_benchmark_config(benchmark)
    return Path(data_root) / config["data_subdir"]


def list_benchmarks() -> list:
    """List all available benchmarks."""
    return list(BENCHMARKS.keys())


def print_benchmark_info():
    """Print information about all benchmarks."""
    print("Available Benchmarks:")
    print("-" * 70)
    for key, cfg in BENCHMARKS.items():
        print(f"  {key:12} | {cfg['name']:20} | action_dim={cfg['action_dim']} | {cfg['data_format']}")
    print("-" * 70)
