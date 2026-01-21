"""
Application-level configuration combining training and visualization settings.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import json
from pathlib import Path

from .training_config import TrainingConfig
from .visualization_config import VisualizationConfig


@dataclass
class AppConfig:
    """
    Combined application configuration.

    Combines training and visualization configurations into a single
    unified config object.
    """

    training: TrainingConfig
    visualization: VisualizationConfig

    def __init__(
        self,
        training: Optional[TrainingConfig] = None,
        visualization: Optional[VisualizationConfig] = None
    ):
        """
        Initialize AppConfig.

        Parameters
        ----------
        training : TrainingConfig, optional
            Training configuration. If None, uses default.
        visualization : VisualizationConfig, optional
            Visualization configuration. If None, uses default.
        """
        self.training = training if training is not None else TrainingConfig()
        self.visualization = visualization if visualization is not None else VisualizationConfig()

    @classmethod
    def from_preset(
        cls,
        preset_name: str,
        training_preset: Optional[str] = None,
        visualization_preset: Optional[str] = None
    ) -> 'AppConfig':
        """
        Create configuration from preset.

        Parameters
        ----------
        preset_name : str
            Combined preset name: 'low', 'medium', 'high'
            This sets both training and visualization presets together.
        training_preset : str, optional
            Override training preset ('quick', 'standard', 'thorough')
        visualization_preset : str, optional
            Override visualization preset ('low', 'medium', 'high')

        Returns
        -------
        AppConfig
            Configuration instance with preset values

        Examples
        --------
        >>> # Use combined preset
        >>> config = AppConfig.from_preset('medium')
        >>>
        >>> # Mix presets
        >>> config = AppConfig.from_preset('medium', training_preset='thorough')
        """
        # Preset mappings
        preset_mapping = {
            'low': {
                'training': 'quick',
                'visualization': 'low'
            },
            'medium': {
                'training': 'standard',
                'visualization': 'medium'
            },
            'high': {
                'training': 'thorough',
                'visualization': 'high'
            }
        }

        if preset_name not in preset_mapping:
            raise ValueError(
                f"Unknown preset: {preset_name}. "
                f"Available: {list(preset_mapping.keys())}"
            )

        # Use overrides if provided
        training_preset = training_preset or preset_mapping[preset_name]['training']
        visualization_preset = visualization_preset or preset_mapping[preset_name]['visualization']

        return cls(
            training=TrainingConfig.from_preset(training_preset),
            visualization=VisualizationConfig.from_preset(visualization_preset)
        )

    @classmethod
    def from_file(cls, filepath: str) -> 'AppConfig':
        """
        Load configuration from JSON file.

        Parameters
        ----------
        filepath : str
            Path to JSON configuration file

        Returns
        -------
        AppConfig
            Loaded configuration
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")

        with open(path, 'r') as f:
            data = json.load(f)

        training_data = data.get('training', {})
        visualization_data = data.get('visualization', {})

        return cls(
            training=TrainingConfig(**training_data),
            visualization=VisualizationConfig(**visualization_data)
        )

    def to_file(self, filepath: str):
        """
        Save configuration to JSON file.

        Parameters
        ----------
        filepath : str
            Path where to save configuration
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'training': self.training.__dict__,
            'visualization': self.visualization.to_dict()
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns
        -------
        dict
            Configuration as dictionary
        """
        return {
            'training': self.training.__dict__,
            'visualization': self.visualization.to_dict()
        }

    def __repr__(self) -> str:
        """String representation of configuration."""
        return (
            f"AppConfig(\n"
            f"  Training: {self.training.num_episodes} episodes, "
            f"batch_size={self.training.batch_size}\n"
            f"  Visualization: {self.visualization.n_agents} agents, "
            f"grid_size={self.visualization.heatmap_grid_size}\n"
            f")"
        )
