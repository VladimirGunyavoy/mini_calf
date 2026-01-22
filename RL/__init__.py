# RL/__init__.py
"""
Reinforcement Learning components for CALF.

Provides:
- BaseEnv: Abstract base class for environments
- PointMassEnv: Point mass environment (1D position + velocity)
- DifferentialDriveEnv: Differential drive environment (2D position + orientation)
- CALFController: Critic as Lyapunov Function controller
- TD3: Twin Delayed DDPG algorithm
- ReplayBuffer: Experience replay buffer
"""

from .base_env import BaseEnv
from .simple_env import PointMassEnv, pd_nominal_policy
from .differential_drive_env import DifferentialDriveEnv, move_to_point_policy
from .td3 import TD3, ReplayBuffer
from .calf import CALFController

__all__ = [
    # Base classes
    'BaseEnv',
    
    # Environments
    'PointMassEnv',
    'DifferentialDriveEnv',
    
    # Policies
    'pd_nominal_policy',
    'move_to_point_policy',
    
    # RL algorithms
    'TD3',
    'ReplayBuffer',
    'CALFController',
]


def create_env(system_type: str, **kwargs):
    """
    Factory function to create environment by type.
    
    Parameters:
    -----------
    system_type : str
        'point_mass' or 'differential_drive'
    **kwargs
        Additional parameters for the environment
        
    Returns:
    --------
    BaseEnv
        Environment instance
    """
    if system_type == 'point_mass':
        return PointMassEnv(**kwargs)
    elif system_type == 'differential_drive':
        return DifferentialDriveEnv(**kwargs)
    else:
        raise ValueError(f"Unknown system type: {system_type}. "
                        f"Available: 'point_mass', 'differential_drive'")


def create_nominal_policy(system_type: str, **kwargs):
    """
    Factory function to create nominal policy by system type.
    
    Parameters:
    -----------
    system_type : str
        'point_mass' or 'differential_drive'
    **kwargs
        Additional parameters for the policy
        
    Returns:
    --------
    callable
        Policy function: state -> action
    """
    if system_type == 'point_mass':
        return pd_nominal_policy(**kwargs)
    elif system_type == 'differential_drive':
        return move_to_point_policy(**kwargs)
    else:
        raise ValueError(f"Unknown system type: {system_type}. "
                        f"Available: 'point_mass', 'differential_drive'")
