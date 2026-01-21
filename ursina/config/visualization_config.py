"""
Visualization configuration for CALF system.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class VisualizationConfig:
    """Configuration for visualization parameters."""

    # Visual agents
    n_agents: int = 5  # Number of agents to visualize simultaneously

    # Trail settings
    trail_max_length: int = 600
    trail_decimation: int = 1  # Capture every N-th point (1 = no decimation)
    trail_rebuild_freq: int = 15  # Rebuild mesh every N additions (higher = better performance)

    # Heatmap settings
    heatmap_enabled: bool = True
    heatmap_grid_size: int = 21  # Grid resolution (NxN)
    heatmap_update_freq: int = 500  # Update every N steps
    heatmap_height_scale: float = 1.0  # Height scale for visualization (reduced from 2.0)
    agent_height_epsilon: float = 0.05  # Agent height offset from surface (reduced from 0.15)

    # Grid overlay
    grid_enabled: bool = True
    grid_node_size: float = 0.04  # Size of grid nodes
    grid_line_thickness: int = 3  # Thickness of grid lines
    grid_sample_step: int = 1  # Sample every N-th point from heatmap grid

    @classmethod
    def from_preset(cls, preset_name: str) -> 'VisualizationConfig':
        """
        Create configuration from preset.

        Parameters
        ----------
        preset_name : str
            One of: 'low' (weak PC), 'medium' (balanced), 'high' (powerful PC)

        Returns
        -------
        VisualizationConfig
            Configuration instance with preset values
        """
        presets = {
            'low': {
                'n_agents': 3,
                'trail_max_length': 300,
                'trail_decimation': 3,
                'trail_rebuild_freq': 30,
                'heatmap_enabled': True,
                'heatmap_grid_size': 15,
                'heatmap_update_freq': 1000,
                'grid_enabled': True,
            },
            'medium': {
                'n_agents': 5,
                'trail_max_length': 600,
                'trail_decimation': 1,
                'trail_rebuild_freq': 15,
                'heatmap_enabled': True,
                'heatmap_grid_size': 21,
                'heatmap_update_freq': 500,
                'grid_enabled': True,
            },
            'high': {
                'n_agents': 10,
                'trail_max_length': 1000,
                'trail_decimation': 1,
                'trail_rebuild_freq': 10,
                'heatmap_enabled': True,
                'heatmap_grid_size': 31,
                'heatmap_update_freq': 250,
                'grid_enabled': True,
            }
        }

        if preset_name not in presets:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(presets.keys())}")

        return cls(**presets[preset_name])

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'n_agents': self.n_agents,
            'trail_max_length': self.trail_max_length,
            'trail_decimation': self.trail_decimation,
            'trail_rebuild_freq': self.trail_rebuild_freq,
            'heatmap_enabled': self.heatmap_enabled,
            'heatmap_grid_size': self.heatmap_grid_size,
            'heatmap_update_freq': self.heatmap_update_freq,
            'heatmap_height_scale': self.heatmap_height_scale,
            'agent_height_epsilon': self.agent_height_epsilon,
            'grid_enabled': self.grid_enabled,
            'grid_node_size': self.grid_node_size,
            'grid_line_thickness': self.grid_line_thickness,
            'grid_sample_step': self.grid_sample_step,
        }
