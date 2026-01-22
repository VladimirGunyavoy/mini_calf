"""
Training configuration for CALF reinforcement learning.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingConfig:
    """Configuration for CALF training parameters."""

    # Episode settings
    num_episodes: int = 500
    max_steps_per_episode: int = 750
    batch_size: int = 64
    start_training_step: int = 100

    # Exploration
    exploration_noise: float = 0.5

    # Evaluation
    eval_interval: int = 10

    # Random seed
    seed: int = 42

    # Reward scaling
    reward_scale: float = 10.0  # Scale rewards for better learning

    # Resume training
    resume_training: bool = True
    resume_checkpoint: str = "trained_calf_final.pth"

    # CALF-specific parameters
    lambda_relax: float = 0.99995  # Relaxation factor (lower = faster P_relax decrease)
    nu_bar: float = 0.01  # Lyapunov decrease threshold
    kappa_low_coef: float = 0.01  # Lower K_∞ coefficient
    kappa_up_coef: float = 1000.0  # Upper K_∞ coefficient

    # Environment boundaries
    goal_epsilon: float = 0.05  # Distance to goal for early termination
    boundary_limit: float = 5.0  # Position boundary for early termination

    # System type: 'point_mass' or 'differential_drive'
    system_type: str = 'point_mass'

    # Differential drive specific parameters
    max_v: float = 1.0  # Max linear velocity
    max_omega: float = 2.0  # Max angular velocity
    goal_angle_tolerance: float = 0.1  # Goal angle tolerance (radians)

    @classmethod
    def from_preset(cls, preset_name: str) -> 'TrainingConfig':
        """
        Create configuration from preset.

        Parameters
        ----------
        preset_name : str
            One of: 'quick' (fast testing), 'standard' (default), 'thorough' (long training)

        Returns
        -------
        TrainingConfig
            Configuration instance with preset values
        """
        presets = {
            'quick': {
                'num_episodes': 100,
                'max_steps_per_episode': 500,
                'batch_size': 64,
                'start_training_step': 50,
            },
            'standard': {
                'num_episodes': 500,
                'max_steps_per_episode': 750,
                'batch_size': 64,
                'start_training_step': 100,
            },
            'thorough': {
                'num_episodes': 1000,
                'max_steps_per_episode': 1000,
                'batch_size': 128,
                'start_training_step': 200,
            }
        }

        if preset_name not in presets:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(presets.keys())}")

        return cls(**presets[preset_name])
