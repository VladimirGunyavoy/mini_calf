"""
Test configuration system.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import AppConfig, TrainingConfig, VisualizationConfig


def test_default_config():
    """Test default configuration creation."""
    print("Testing default config...")
    config = AppConfig()
    print(f"  Training episodes: {config.training.num_episodes}")
    print(f"  Visual agents: {config.visualization.n_agents}")
    print(f"  Heatmap grid: {config.visualization.heatmap_grid_size}")
    assert config.training.num_episodes == 500
    assert config.visualization.n_agents == 5
    print("  [OK] Default config works")


def test_preset_low():
    """Test 'low' preset."""
    print("\nTesting 'low' preset...")
    config = AppConfig.from_preset('low')
    print(f"  Training episodes: {config.training.num_episodes}")
    print(f"  Visual agents: {config.visualization.n_agents}")
    print(f"  Heatmap grid: {config.visualization.heatmap_grid_size}")
    assert config.training.num_episodes == 100  # quick preset
    assert config.visualization.n_agents == 3
    assert config.visualization.heatmap_grid_size == 15
    print("  [OK] Low preset works")


def test_preset_medium():
    """Test 'medium' preset."""
    print("\nTesting 'medium' preset...")
    config = AppConfig.from_preset('medium')
    print(f"  Training episodes: {config.training.num_episodes}")
    print(f"  Visual agents: {config.visualization.n_agents}")
    print(f"  Heatmap grid: {config.visualization.heatmap_grid_size}")
    assert config.training.num_episodes == 500  # standard preset
    assert config.visualization.n_agents == 5
    assert config.visualization.heatmap_grid_size == 21
    print("  [OK] Medium preset works")


def test_preset_high():
    """Test 'high' preset."""
    print("\nTesting 'high' preset...")
    config = AppConfig.from_preset('high')
    print(f"  Training episodes: {config.training.num_episodes}")
    print(f"  Visual agents: {config.visualization.n_agents}")
    print(f"  Heatmap grid: {config.visualization.heatmap_grid_size}")
    assert config.training.num_episodes == 1000  # thorough preset
    assert config.visualization.n_agents == 10
    assert config.visualization.heatmap_grid_size == 31
    print("  [OK] High preset works")


def test_mixed_preset():
    """Test mixed presets."""
    print("\nTesting mixed presets...")
    config = AppConfig.from_preset('medium', training_preset='thorough')
    print(f"  Training episodes: {config.training.num_episodes} (should be thorough)")
    print(f"  Visual agents: {config.visualization.n_agents} (should be medium)")
    assert config.training.num_episodes == 1000  # thorough
    assert config.visualization.n_agents == 5  # medium
    print("  [OK] Mixed presets work")


def test_to_dict():
    """Test conversion to dictionary."""
    print("\nTesting to_dict()...")
    config = AppConfig.from_preset('medium')
    data = config.to_dict()
    print(f"  Keys: {list(data.keys())}")
    assert 'training' in data
    assert 'visualization' in data
    assert data['training']['num_episodes'] == 500
    assert data['visualization']['n_agents'] == 5
    print("  [OK] to_dict() works")


def test_repr():
    """Test string representation."""
    print("\nTesting __repr__()...")
    config = AppConfig.from_preset('medium')
    repr_str = repr(config)
    print(f"  {repr_str}")
    assert 'AppConfig' in repr_str
    assert '500 episodes' in repr_str
    assert '5 agents' in repr_str
    print("  [OK] __repr__() works")


def test_individual_configs():
    """Test individual config classes."""
    print("\nTesting individual configs...")

    training = TrainingConfig.from_preset('standard')
    assert training.num_episodes == 500
    print("  [OK] TrainingConfig works")

    vis = VisualizationConfig.from_preset('medium')
    assert vis.n_agents == 5
    print("  [OK] VisualizationConfig works")


if __name__ == '__main__':
    print("=" * 60)
    print("CALF Configuration System Tests")
    print("=" * 60)

    try:
        test_default_config()
        test_preset_low()
        test_preset_medium()
        test_preset_high()
        test_mixed_preset()
        test_to_dict()
        test_repr()
        test_individual_configs()

        print("\n" + "=" * 60)
        print("[SUCCESS] ALL TESTS PASSED")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
