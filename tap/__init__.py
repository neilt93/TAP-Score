"""TAP-Score: Temporal Action-Proposal Scoring for Off-Manifold Detection."""

from .model import TAPScore, build_tap_model
from .dataset import TAPDataset, create_dataloaders, ObservationAugmentor
from .perturbations import get_perturbation, PERTURBATIONS
from .calibration import TAPCalibrator, compute_threshold, evaluate_detection
from .resampling import TAPResampler, evaluate_resampling

__all__ = [
    'TAPScore',
    'build_tap_model',
    'TAPDataset',
    'create_dataloaders',
    'ObservationAugmentor',
    'get_perturbation',
    'PERTURBATIONS',
    'TAPCalibrator',
    'compute_threshold',
    'evaluate_detection',
    'TAPResampler',
    'evaluate_resampling',
]
