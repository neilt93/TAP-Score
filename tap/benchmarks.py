"""Benchmark configuration registry for TAP-Score evaluation."""

from pathlib import Path

# Benchmark configurations
BENCHMARKS = {
    "pusht": {
        "name": "Push-T",
        "action_dim": 2,
        "obs_type": "image",  # Uses CNN encoder
        "obs_channels": 3,
        "image_size": 96,
        "obs_window": 2,
        "action_chunk": 16,
        "data_format": "zarr",
        "data_subdir": "pusht",
        "zarr_obs_key": "img",  # Key in zarr for observations
        "download_url": "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip",
        "description": "2D pushing task with T-block (image observations)",
    },
    "blockpush": {
        "name": "Block Push",
        "action_dim": 2,
        "obs_type": "state",  # Uses MLP encoder
        "obs_dim": 16,  # State observation dimension
        "obs_window": 2,
        "action_chunk": 16,
        "data_format": "zarr",
        "data_subdir": "block_pushing",  # Fixed: actual directory name
        "zarr_file": "multimodal_push_seed.zarr",  # Specific zarr file
        "zarr_obs_key": "obs",  # Key in zarr for observations
        "download_url": "https://diffusion-policy.cs.columbia.edu/data/training/block_pushing.zip",
        "description": "2D block pushing task (state observations)",
    },
    "kitchen": {
        "name": "Kitchen",
        "action_dim": 9,
        "obs_type": "state",  # Uses MLP encoder (no images available)
        "obs_dim": 60,  # State observation dimension
        "obs_window": 2,
        "action_chunk": 16,
        "data_format": "npy",  # Different format
        "data_subdir": "kitchen",
        "download_url": "https://diffusion-policy.cs.columbia.edu/data/training/kitchen.zip",
        "description": "9-DoF kitchen manipulation task (state observations)",
    },
    "robomimic_lift_lowdim": {
        "name": "Lift (lowdim)",
        "action_dim": 7,
        "obs_type": "state",
        "obs_dim": 19,  # Validated: object(10) + eef_pos(3) + eef_quat(4) + gripper_qpos(2)
        "obs_window": 2,
        "action_chunk": 10,  # Placeholder — derive from DP config (horizon) at runtime
        "data_format": "hdf5",
        "data_subdir": "robomimic/datasets/lift/ph",
        "hdf5_file": "low_dim.hdf5",
        "obs_keys": ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
        "description": "7-DoF robotic lift task (state observations)",
    },
    "robomimic_can_lowdim": {
        "name": "Can (lowdim)",
        "action_dim": 7,
        "obs_type": "state",
        "obs_dim": 23,  # Validated: object(14) + eef_pos(3) + eef_quat(4) + gripper_qpos(2)
        "obs_window": 2,
        "action_chunk": 10,  # Placeholder — derive from DP config (horizon) at runtime
        "data_format": "hdf5",
        "data_subdir": "robomimic/datasets/can/ph",
        "hdf5_file": "low_dim.hdf5",
        "obs_keys": ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
        "description": "7-DoF robotic can-picking task (state observations)",
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
    base_path = Path(data_root) / config["data_subdir"]

    # If zarr_file or hdf5_file is specified, append it to the path
    if "zarr_file" in config:
        return base_path / config["zarr_file"]
    if "hdf5_file" in config:
        return base_path / config["hdf5_file"]
    return base_path


def list_benchmarks() -> list:
    """List all available benchmarks."""
    return list(BENCHMARKS.keys())


def print_benchmark_info():
    """Print information about all benchmarks."""
    print("Available Benchmarks:")
    print("-" * 80)
    for key, cfg in BENCHMARKS.items():
        print(f"  {key:25} | {cfg['name']:20} | action_dim={cfg['action_dim']} | {cfg['data_format']}")
    print("-" * 80)
