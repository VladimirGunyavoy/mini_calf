"""
Integration test: Verify config system can replace train_calf_visual.py constants.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import AppConfig


def test_config_matches_train_calf_visual():
    """
    Test that config values match the constants in train_calf_visual.py.

    This verifies that the config system correctly captures all the
    hardcoded constants from the original training script.
    """
    print("Testing config matches train_calf_visual.py constants...")

    # Load medium preset (should match train_calf_visual.py defaults)
    config = AppConfig.from_preset('medium')

    # Training constants from train_calf_visual.py
    print("\n  Training Config:")
    assert config.training.num_episodes == 500, "NUM_EPISODES mismatch"
    print(f"    NUM_EPISODES: {config.training.num_episodes} [OK]")

    assert config.training.max_steps_per_episode == 750, "MAX_STEPS_PER_EPISODE mismatch"
    print(f"    MAX_STEPS_PER_EPISODE: {config.training.max_steps_per_episode} [OK]")

    assert config.training.batch_size == 64, "BATCH_SIZE mismatch"
    print(f"    BATCH_SIZE: {config.training.batch_size} [OK]")

    assert config.training.start_training_step == 100, "START_TRAINING_STEP mismatch"
    print(f"    START_TRAINING_STEP: {config.training.start_training_step} [OK]")

    assert config.training.exploration_noise == 0.5, "EXPLORATION_NOISE mismatch"
    print(f"    EXPLORATION_NOISE: {config.training.exploration_noise} [OK]")

    assert config.training.reward_scale == 10.0, "REWARD_SCALE mismatch"
    print(f"    REWARD_SCALE: {config.training.reward_scale} [OK]")

    # CALF parameters
    print("\n  CALF Config:")
    assert config.training.lambda_relax == 0.99995, "LAMBDA_RELAX mismatch"
    print(f"    LAMBDA_RELAX: {config.training.lambda_relax} [OK]")

    assert config.training.nu_bar == 0.01, "NU_BAR mismatch"
    print(f"    NU_BAR: {config.training.nu_bar} [OK]")

    assert config.training.kappa_low_coef == 0.01, "KAPPA_LOW_COEF mismatch"
    print(f"    KAPPA_LOW_COEF: {config.training.kappa_low_coef} [OK]")

    assert config.training.kappa_up_coef == 1000.0, "KAPPA_UP_COEF mismatch"
    print(f"    KAPPA_UP_COEF: {config.training.kappa_up_coef} [OK]")

    # Visualization constants
    print("\n  Visualization Config:")
    assert config.visualization.n_agents == 5, "N_AGENTS_VISUAL mismatch"
    print(f"    N_AGENTS_VISUAL: {config.visualization.n_agents} [OK]")

    assert config.visualization.trail_max_length == 600, "TRAIL_MAX_LENGTH mismatch"
    print(f"    TRAIL_MAX_LENGTH: {config.visualization.trail_max_length} [OK]")

    assert config.visualization.trail_decimation == 1, "TRAIL_DECIMATION mismatch"
    print(f"    TRAIL_DECIMATION: {config.visualization.trail_decimation} [OK]")

    assert config.visualization.trail_rebuild_freq == 15, "TRAIL_REBUILD_FREQ mismatch"
    print(f"    TRAIL_REBUILD_FREQ: {config.visualization.trail_rebuild_freq} [OK]")

    assert config.visualization.heatmap_enabled == True, "HEATMAP_ENABLED mismatch"
    print(f"    HEATMAP_ENABLED: {config.visualization.heatmap_enabled} [OK]")

    assert config.visualization.heatmap_grid_size == 21, "HEATMAP_GRID_SIZE mismatch"
    print(f"    HEATMAP_GRID_SIZE: {config.visualization.heatmap_grid_size} [OK]")

    assert config.visualization.heatmap_update_freq == 500, "HEATMAP_UPDATE_FREQ mismatch"
    print(f"    HEATMAP_UPDATE_FREQ: {config.visualization.heatmap_update_freq} [OK]")

    assert config.visualization.heatmap_height_scale == 2.0, "HEATMAP_HEIGHT_SCALE mismatch"
    print(f"    HEATMAP_HEIGHT_SCALE: {config.visualization.heatmap_height_scale} [OK]")

    assert config.visualization.grid_enabled == True, "GRID_OVERLAY_ENABLED mismatch"
    print(f"    GRID_OVERLAY_ENABLED: {config.visualization.grid_enabled} [OK]")

    assert config.visualization.grid_node_size == 0.04, "GRID_NODE_SIZE mismatch"
    print(f"    GRID_NODE_SIZE: {config.visualization.grid_node_size} [OK]")

    print("\n  [SUCCESS] All constants match!")


def test_preset_switching():
    """Test that preset switching works as expected."""
    print("\nTesting preset switching...")

    # Low preset
    low = AppConfig.from_preset('low')
    print(f"  Low: {low.visualization.n_agents} agents, {low.training.num_episodes} episodes")
    assert low.visualization.n_agents == 3
    assert low.training.num_episodes == 100

    # Medium preset
    medium = AppConfig.from_preset('medium')
    print(f"  Medium: {medium.visualization.n_agents} agents, {medium.training.num_episodes} episodes")
    assert medium.visualization.n_agents == 5
    assert medium.training.num_episodes == 500

    # High preset
    high = AppConfig.from_preset('high')
    print(f"  High: {high.visualization.n_agents} agents, {high.training.num_episodes} episodes")
    assert high.visualization.n_agents == 10
    assert high.training.num_episodes == 1000

    print("  [SUCCESS] Preset switching works!")


def test_save_load_config():
    """Test saving and loading configuration to/from file."""
    print("\nTesting save/load to file...")

    import tempfile
    import os

    # Create a config
    config = AppConfig.from_preset('high')

    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        temp_path = f.name

    try:
        config.to_file(temp_path)
        print(f"  Saved to: {temp_path}")

        # Load back
        loaded = AppConfig.from_file(temp_path)
        print(f"  Loaded from: {temp_path}")

        # Verify values match
        assert loaded.training.num_episodes == config.training.num_episodes
        assert loaded.visualization.n_agents == config.visualization.n_agents
        assert loaded.training.lambda_relax == config.training.lambda_relax

        print("  [SUCCESS] Save/load works!")

    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == '__main__':
    print("=" * 60)
    print("CALF Config Integration Tests")
    print("=" * 60)

    try:
        test_config_matches_train_calf_visual()
        test_preset_switching()
        test_save_load_config()

        print("\n" + "=" * 60)
        print("[SUCCESS] ALL INTEGRATION TESTS PASSED")
        print("=" * 60)
        print("\nConfig system is ready for STAGE 3 integration!")

    except AssertionError as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
