"""
Quick demo of config system - show how easy it is to use.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config import AppConfig

def print_separator():
    print("\n" + "=" * 60 + "\n")

def demo_basic_usage():
    """Demo 1: Basic usage."""
    print("DEMO 1: Basic Usage")
    print("-" * 60)

    # Default config
    config = AppConfig()
    print(f"Default config:")
    print(f"  Training: {config.training.num_episodes} episodes")
    print(f"  Visualization: {config.visualization.n_agents} agents")

    print_separator()

def demo_presets():
    """Demo 2: Using presets."""
    print("DEMO 2: Presets")
    print("-" * 60)

    for preset in ['low', 'medium', 'high']:
        config = AppConfig.from_preset(preset)
        print(f"\nPreset '{preset}':")
        print(f"  Training: {config.training.num_episodes} episodes, "
              f"batch={config.training.batch_size}")
        print(f"  Visualization: {config.visualization.n_agents} agents, "
              f"grid={config.visualization.heatmap_grid_size}x{config.visualization.heatmap_grid_size}")
        print(f"  Heatmap update: every {config.visualization.heatmap_update_freq} steps")

    print_separator()

def demo_mixed_presets():
    """Demo 3: Mixed presets."""
    print("DEMO 3: Mixed Presets")
    print("-" * 60)

    # Long training with simple visualization
    config = AppConfig.from_preset('medium', training_preset='thorough')
    print("Thorough training + Medium visualization:")
    print(f"  Training: {config.training.num_episodes} episodes (thorough)")
    print(f"  Visualization: {config.visualization.n_agents} agents (medium)")

    print()

    # Quick training with rich visualization
    config = AppConfig.from_preset('low', visualization_preset='high')
    print("Quick training + High visualization:")
    print(f"  Training: {config.training.num_episodes} episodes (quick)")
    print(f"  Visualization: {config.visualization.n_agents} agents (high)")

    print_separator()

def demo_repr():
    """Demo 4: String representation."""
    print("DEMO 4: String Representation")
    print("-" * 60)

    config = AppConfig.from_preset('medium')
    print("\nrepr(config):")
    print(repr(config))

    print_separator()

def demo_dict_export():
    """Demo 5: Dictionary export."""
    print("DEMO 5: Dictionary Export")
    print("-" * 60)

    config = AppConfig.from_preset('high')
    data = config.to_dict()

    print("\nConfig as dictionary:")
    print(f"  Training keys: {list(data['training'].keys())[:5]}...")
    print(f"  Visualization keys: {list(data['visualization'].keys())[:5]}...")
    print(f"\nSample values:")
    print(f"  training.num_episodes: {data['training']['num_episodes']}")
    print(f"  training.lambda_relax: {data['training']['lambda_relax']}")
    print(f"  visualization.n_agents: {data['visualization']['n_agents']}")

    print_separator()

def demo_save_load():
    """Demo 6: Save/load to file."""
    print("DEMO 6: Save/Load to File")
    print("-" * 60)

    import tempfile
    import os
    import json

    # Create and save config
    config = AppConfig.from_preset('high')
    temp_path = os.path.join(tempfile.gettempdir(), 'calf_config_demo.json')

    config.to_file(temp_path)
    print(f"Saved config to: {temp_path}")

    # Show file contents (first few lines)
    with open(temp_path, 'r') as f:
        data = json.load(f)
    print(f"\nFile contents (partial):")
    print(f"  training.num_episodes: {data['training']['num_episodes']}")
    print(f"  visualization.n_agents: {data['visualization']['n_agents']}")

    # Load back
    loaded = AppConfig.from_file(temp_path)
    print(f"\nLoaded config from file:")
    print(f"  Training: {loaded.training.num_episodes} episodes")
    print(f"  Visualization: {loaded.visualization.n_agents} agents")

    # Cleanup
    os.remove(temp_path)
    print(f"\nCleaned up: {temp_path}")

    print_separator()

def demo_usage_in_code():
    """Demo 7: How to use in actual code."""
    print("DEMO 7: Usage in Code")
    print("-" * 60)

    print("\nExample: Replace train_calf_visual.py constants\n")
    print("OLD CODE (train_calf_visual.py):")
    print("  NUM_EPISODES = 500")
    print("  BATCH_SIZE = 64")
    print("  N_AGENTS_VISUAL = 5")
    print("  HEATMAP_GRID_SIZE = 21")
    print()
    print("NEW CODE (with config system):")
    print("  from config import AppConfig")
    print("  config = AppConfig.from_preset('medium')")
    print()
    print("  # Access parameters:")
    print("  num_episodes = config.training.num_episodes")
    print("  batch_size = config.training.batch_size")
    print("  n_agents = config.visualization.n_agents")
    print("  grid_size = config.visualization.heatmap_grid_size")
    print()
    print("BENEFITS:")
    print("  - One-line preset switching")
    print("  - No scattered constants")
    print("  - Easy experimentation")
    print("  - Type-safe with IDE autocomplete")

    print_separator()

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("       CALF Configuration System - Interactive Demo")
    print("=" * 60)

    demo_basic_usage()
    demo_presets()
    demo_mixed_presets()
    demo_repr()
    demo_dict_export()
    demo_save_load()
    demo_usage_in_code()

    print("=" * 60)
    print("Demo complete! Config system is ready to use.")
    print("=" * 60)
    print("\nNext: STAGE 3 will integrate this into main.py")
    print()
