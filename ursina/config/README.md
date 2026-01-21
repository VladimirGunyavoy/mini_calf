# CALF Configuration System

Centralized configuration management for CALF training and visualization.

## Quick Start

```python
from config import AppConfig

# Use preset (recommended)
config = AppConfig.from_preset('medium')

# Access configuration
print(f"Episodes: {config.training.num_episodes}")
print(f"Visual agents: {config.visualization.n_agents}")

# Run training with config
# (Implementation in STAGE 3)
```

## Available Presets

### Combined Presets
- `'low'` - For weak PCs (quick training + minimal visualization)
- `'medium'` - Balanced (standard training + medium visualization) **[DEFAULT]**
- `'high'` - For powerful PCs (thorough training + rich visualization)

### Training Presets
- `'quick'` - 100 episodes, 500 steps/episode
- `'standard'` - 500 episodes, 750 steps/episode **[DEFAULT]**
- `'thorough'` - 1000 episodes, 1000 steps/episode

### Visualization Presets
- `'low'` - 3 agents, grid 15x15, update every 1000 steps
- `'medium'` - 5 agents, grid 21x21, update every 500 steps **[DEFAULT]**
- `'high'` - 10 agents, grid 31x31, update every 250 steps

## Usage Examples

### Basic Usage
```python
from config import AppConfig

# Default configuration
config = AppConfig()

# Use preset
config = AppConfig.from_preset('high')
```

### Mixed Presets
```python
# Thorough training with medium visualization
config = AppConfig.from_preset('medium', training_preset='thorough')

# Quick training with high-quality visualization
config = AppConfig.from_preset('low', visualization_preset='high')
```

### Custom Configuration
```python
from config import TrainingConfig, VisualizationConfig, AppConfig

# Custom training
training = TrainingConfig(
    num_episodes=300,
    batch_size=128,
    exploration_noise=0.3
)

# Custom visualization
visualization = VisualizationConfig(
    n_agents=7,
    heatmap_grid_size=25,
    trail_max_length=800
)

# Combine
config = AppConfig(training=training, visualization=visualization)
```

### Save/Load Configuration
```python
# Save to file
config = AppConfig.from_preset('high')
config.to_file('my_config.json')

# Load from file
config = AppConfig.from_file('my_config.json')
```

## Configuration Parameters

### TrainingConfig
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_episodes` | int | 500 | Number of training episodes |
| `max_steps_per_episode` | int | 750 | Maximum steps per episode |
| `batch_size` | int | 64 | Training batch size |
| `start_training_step` | int | 100 | When to start training |
| `exploration_noise` | float | 0.5 | Exploration noise std |
| `reward_scale` | float | 10.0 | Reward scaling factor |
| `lambda_relax` | float | 0.99995 | CALF relaxation factor |
| `nu_bar` | float | 0.01 | Lyapunov decrease threshold |
| `kappa_low_coef` | float | 0.01 | Lower K_infinity coefficient |
| `kappa_up_coef` | float | 1000.0 | Upper K_infinity coefficient |

### VisualizationConfig
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_agents` | int | 5 | Number of visual agents |
| `trail_max_length` | int | 600 | Maximum trail points |
| `trail_decimation` | int | 1 | Trail point decimation |
| `trail_rebuild_freq` | int | 15 | Mesh rebuild frequency |
| `heatmap_enabled` | bool | True | Enable critic heatmap |
| `heatmap_grid_size` | int | 21 | Heatmap grid resolution |
| `heatmap_update_freq` | int | 500 | Heatmap update frequency |
| `heatmap_height_scale` | float | 2.0 | Height scale factor |
| `grid_enabled` | bool | True | Enable grid overlay |
| `grid_node_size` | float | 0.04 | Grid node size |

## Testing

Run configuration tests:
```bash
cd c:\GitHub\Learn\CALF\ursina
py -3.12 tests/test_config.py
```

## Implementation Notes

- Built with Python `@dataclass` for clean, type-safe configuration
- Supports JSON serialization for saving/loading
- All presets are validated in unit tests
- Future: STAGE 3 will integrate this config system into main.py
