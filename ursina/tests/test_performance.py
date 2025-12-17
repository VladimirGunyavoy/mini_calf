"""
Performance Test - Phase 4.3 & 4.4
Testing VectorizedEnvironment with different numbers of agents
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import time
from ursina import *

from core import Player, setup_scene
from managers import (
    WindowManager,
    ZoomManager,
    ObjectManager,
    ColorManager,
)
from physics import VectorizedEnvironment
from physics.policies import PDPolicy
from visuals import PointVisual

# Test configurations
TEST_CONFIGS = [
    {"n_agents": 10, "color": color.cyan, "name": "10 agents"},
    {"n_agents": 50, "color": color.yellow, "name": "50 agents"},
    {"n_agents": 100, "color": color.orange, "name": "100 agents"},
]

CURRENT_TEST = 0  # Start with first test

# IMPORTANT: Setup window before creating Ursina app
WindowManager.setup_before_app(monitor="left")

app = Ursina()

# Initialize components
player = Player()
color_manager = ColorManager()
window_manager = WindowManager(color_manager=color_manager, monitor="left")
zoom_manager = ZoomManager(player=player)
object_manager = ObjectManager(zoom_manager=zoom_manager)

# Setup scene
ground, grid, lights, frame = setup_scene(color_manager, object_manager)

print("\n" + "="*70)
print("PHASE 4.3 & 4.4: PERFORMANCE TEST")
print("="*70)
print("\nTesting VectorizedEnvironment with different numbers of agents")
print("Press SPACE to cycle through tests (10, 50, 100 agents)")
print("Press R to restart current test")
print("Press Q/Escape to quit")
print("\n" + "="*70)

# Global variables
vec_env = None
visual_points = []
frame_count = 0
fps_samples = []
stats_interval = 300  # Print stats every 300 frames

def create_test(n_agents: int, agent_color):
    """Create a new test with specified number of agents."""
    global vec_env, visual_points, frame_count, fps_samples
    
    # Clear previous test
    for point in visual_points:
        if hasattr(point, 'entity'):
            destroy(point.entity)
    visual_points.clear()
    
    # Reset counters
    frame_count = 0
    fps_samples.clear()
    
    # Create PD policy
    pd_policy = PDPolicy(kp=1.0, kd=0.8, target=np.array([0.0]), dim=1)
    
    # Create vectorized environment
    vec_env = VectorizedEnvironment(
        n_envs=n_agents,
        policy=pd_policy,
        dt=0.01,
        seed=42
    )
    
    print(f"\n[TEST] Creating {n_agents} agents...")
    
    # Create visual objects
    for i in range(n_agents):
        point_visual = PointVisual(
            scale=0.06,  # Smaller for many agents
            color=agent_color,
            zoom_manager=zoom_manager
        )
        visual_points.append(point_visual)
    
    print(f"[OK] Created {n_agents} visual objects")
    vec_env.print_stats()

# Start first test
test_config = TEST_CONFIGS[CURRENT_TEST]
print(f"\n>>> Starting Test 1/{len(TEST_CONFIGS)}: {test_config['name']}")
create_test(test_config["n_agents"], test_config["color"])

def update():
    """Update each frame"""
    global frame_count, fps_samples
    
    # Measure frame time
    frame_start = time.perf_counter()
    
    # 1. Vectorized environment step (batch process all agents)
    states = vec_env.step()
    
    # 2. Update visualization (phase space representation)
    for i, state in enumerate(states):
        x = state[0]  # Position
        v = state[1]  # Velocity
        
        # Phase space: (x, 0, v)
        visual_points[i].set_position((x, 0, v))
    
    frame_end = time.perf_counter()
    frame_time = (frame_end - frame_start) * 1000  # milliseconds
    
    # Calculate FPS
    current_fps = 1000.0 / frame_time if frame_time > 0 else 0
    fps_samples.append(current_fps)
    
    # Keep only last 60 samples
    if len(fps_samples) > 60:
        fps_samples.pop(0)
    
    # 3. Statistics
    frame_count += 1
    if frame_count % stats_interval == 0:
        positions = vec_env.get_positions()
        velocities = vec_env.get_velocities()
        
        avg_fps = np.mean(fps_samples) if fps_samples else 0
        min_fps = np.min(fps_samples) if fps_samples else 0
        max_fps = np.max(fps_samples) if fps_samples else 0
        
        print(f"\n[Frame {frame_count}] Test {CURRENT_TEST+1}/{len(TEST_CONFIGS)}: {TEST_CONFIGS[CURRENT_TEST]['name']}")
        print(f"  FPS: avg={avg_fps:.1f}, min={min_fps:.1f}, max={max_fps:.1f}")
        print(f"  Frame time: avg={1000/avg_fps:.2f}ms")
        print(f"  Position: min={positions.min():.3f}, max={positions.max():.3f}, mean={positions.mean():.3f}")
        print(f"  Velocity: min={velocities.min():.3f}, max={velocities.max():.3f}, mean={velocities.mean():.3f}")
        
        dist_to_target = np.abs(positions)
        print(f"  Distance to target: max={dist_to_target.max():.3f}, mean={dist_to_target.mean():.3f}")

def input(key):
    """Input handling"""
    global CURRENT_TEST
    
    if key == 'q' or key == 'escape':
        print("\nExiting...")
        application.quit()
    
    # Camera
    if key == 'w':
        player.y += 1
    if key == 's':
        player.y -= 1
    if key == 'a':
        player.x -= 1
    if key == 'd':
        player.x += 1
    
    # Zoom
    if key == 'scroll up':
        zoom_manager.zoom_in()
    if key == 'scroll down':
        zoom_manager.zoom_out()
    
    # Reset current test
    if key == 'r':
        print(f"\n[RESET] Restarting test {CURRENT_TEST+1}/{len(TEST_CONFIGS)}")
        test_config = TEST_CONFIGS[CURRENT_TEST]
        create_test(test_config["n_agents"], test_config["color"])
    
    # Cycle through tests
    if key == 'space':
        CURRENT_TEST = (CURRENT_TEST + 1) % len(TEST_CONFIGS)
        test_config = TEST_CONFIGS[CURRENT_TEST]
        print(f"\n>>> Starting Test {CURRENT_TEST+1}/{len(TEST_CONFIGS)}: {test_config['name']}")
        create_test(test_config["n_agents"], test_config["color"])

print("\nStarting performance test...")
app.run()
