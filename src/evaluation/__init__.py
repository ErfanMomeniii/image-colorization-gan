"""
Evaluation Module

Contains metrics and evaluation utilities for image colorization:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- Colorfulness metric
- Visual comparison utilities
- Experiment tracking and comparison
"""

from .evaluator import (
    Evaluator,
    calculate_psnr,
    calculate_ssim,
    calculate_colorfulness,
    evaluate_batch
)

from .experiments import (
    ExperimentTracker,
    create_hyperparameter_table
)

__all__ = [
    'Evaluator',
    'calculate_psnr',
    'calculate_ssim',
    'calculate_colorfulness',
    'evaluate_batch',
    'ExperimentTracker',
    'create_hyperparameter_table'
]
