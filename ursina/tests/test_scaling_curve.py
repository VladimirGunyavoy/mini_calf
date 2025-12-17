"""
Scaling Curve Test - Extended Performance Testing
Building a performance curve for VectorizedEnvironment
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

# Extended test configurations for building a scaling curve
TEST_CONFIGS = [
    {"n_agents": 10, "color": color.cyan, "name": "10 agents"},
    {"n_agents": 25, "color": color.azure, "name": "25 agents"},
    {"n_agents": 50, "color": color.lime, "name": "50 agents"},
    {"n_agents": 75, "color": color.yellow, "name": "75 agents"},
    {"n_agents": 100, "color": color.orange, "name": "100 agents"},
    {"n_agents": 150, "color": color.salmon, "name": "150 agents"},
    {"n_agents": 200, "color": color.red, "name": "200 agents"},
]

CURRENT_TEST = 0
AUTO_CYCLE = True  # Automatically cycle through all tests
FRAMES_PER_TEST = 600  # 10 seconds at 60 FPS

# Performance data collection
performance_data = []

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
print("SCALING CURVE TEST - Extended Performance Analysis")
print("="*70)
print(f"\nTesting {len(TEST_CONFIGS)} configurations:")
for i, cfg in enumerate(TEST_CONFIGS, 1):
    print(f"  {i}. {cfg['name']}")
print(f"\nEach test runs for {FRAMES_PER_TEST} frames")
print("Press SPACE to cycle manually (disables auto-cycle)")
print("Press A to toggle auto-cycle")
print("Press R to restart current test")
print("Press Q/Escape to quit and see results")
print("\n" + "="*70)

# Global variables
vec_env = None
visual_points = []
frame_count = 0
test_frame_count = 0
fps_samples = []

def create_test(n_agents: int, agent_color):
    """Create a new test with specified number of agents."""
    global vec_env, visual_points, frame_count, test_frame_count, fps_samples
    
    # Clear previous test
    for point in visual_points:
        if hasattr(point, 'entity'):
            destroy(point.entity)
    visual_points.clear()
    
    # Reset counters
    frame_count = 0
    test_frame_count = 0
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
    
    print(f"\n[TEST {CURRENT_TEST+1}/{len(TEST_CONFIGS)}] Creating {n_agents} agents...")
    
    # Create visual objects (smaller for many agents)
    scale = max(0.04, 0.08 - (n_agents / 2000))  # Smaller for more agents
    for i in range(n_agents):
        point_visual = PointVisual(
            scale=scale,
            color=agent_color,
            zoom_manager=zoom_manager
        )
        visual_points.append(point_visual)
    
    print(f"[OK] Created {n_agents} visual objects (scale={scale:.3f})")
    vec_env.print_stats()

# Start first test
test_config = TEST_CONFIGS[CURRENT_TEST]
print(f"\n>>> Starting Test {CURRENT_TEST+1}/{len(TEST_CONFIGS)}: {test_config['name']}")
create_test(test_config["n_agents"], test_config["color"])

def update():
    """Update each frame"""
    global frame_count, test_frame_count, fps_samples, CURRENT_TEST, AUTO_CYCLE
    
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
    
    # Keep only last 60 samples for running average
    if len(fps_samples) > 60:
        fps_samples.pop(0)
    
    frame_count += 1
    test_frame_count += 1
    
    # Print progress every 150 frames
    if frame_count % 150 == 0:
        avg_fps = np.mean(fps_samples) if fps_samples else 0
        progress = (test_frame_count / FRAMES_PER_TEST) * 100
        print(f"  Frame {test_frame_count}/{FRAMES_PER_TEST} ({progress:.0f}%) - FPS: {avg_fps:.1f}")
    
    # Auto-cycle to next test after FRAMES_PER_TEST frames
    if AUTO_CYCLE and test_frame_count >= FRAMES_PER_TEST:
        # Save performance data
        n_agents = TEST_CONFIGS[CURRENT_TEST]["n_agents"]
        avg_fps = np.mean(fps_samples) if fps_samples else 0
        min_fps = np.min(fps_samples) if fps_samples else 0
        max_fps = np.max(fps_samples) if fps_samples else 0
        avg_frame_time = 1000.0 / avg_fps if avg_fps > 0 else 0
        
        performance_data.append({
            "n_agents": n_agents,
            "avg_fps": avg_fps,
            "min_fps": min_fps,
            "max_fps": max_fps,
            "avg_frame_time": avg_frame_time
        })
        
        print(f"\n[COMPLETE] Test {CURRENT_TEST+1}/{len(TEST_CONFIGS)}")
        print(f"  Agents: {n_agents}")
        print(f"  FPS: avg={avg_fps:.1f}, min={min_fps:.1f}, max={max_fps:.1f}")
        print(f"  Frame time: avg={avg_frame_time:.2f}ms")
        
        # Move to next test
        CURRENT_TEST = (CURRENT_TEST + 1) % len(TEST_CONFIGS)
        
        if CURRENT_TEST == 0:
            # Completed full cycle - print summary
            print("\n" + "="*70)
            print("SCALING CURVE - COMPLETE RESULTS")
            print("="*70)
            print(f"\n{'Agents':<10} {'Avg FPS':<12} {'Frame Time':<15} {'Performance'}")
            print("-" * 70)
            
            for data in performance_data:
                n = data["n_agents"]
                fps = data["avg_fps"]
                ft = data["avg_frame_time"]
                
                # Performance assessment
                if fps > 1000:
                    perf = "Excellent"
                elif fps > 500:
                    perf = "Very Good"
                elif fps > 200:
                    perf = "Good"
                elif fps > 60:
                    perf = "Acceptable"
                else:
                    perf = "Poor"
                
                print(f"{n:<10} {fps:<12.1f} {ft:<15.2f} {perf}")
            
            print("\n" + "="*70)
            print("Press SPACE to restart, Q to quit")
        
        test_config = TEST_CONFIGS[CURRENT_TEST]
        print(f"\n>>> Starting Test {CURRENT_TEST+1}/{len(TEST_CONFIGS)}: {test_config['name']}")
        create_test(test_config["n_agents"], test_config["color"])

def input(key):
    """Input handling"""
    global CURRENT_TEST, AUTO_CYCLE
    
    if key == 'q' or key == 'escape':
        # Print final summary before exiting
        if performance_data:
            print("\n" + "="*70)
            print("FINAL RESULTS")
            print("="*70)
            print(f"\n{'Agents':<10} {'Avg FPS':<12} {'Min FPS':<12} {'Max FPS':<12} {'Frame Time'}")
            print("-" * 70)
            
            for data in performance_data:
                n = data["n_agents"]
                avg_fps = data["avg_fps"]
                min_fps = data["min_fps"]
                max_fps = data["max_fps"]
                ft = data["avg_frame_time"]
                
                print(f"{n:<10} {avg_fps:<12.1f} {min_fps:<12.1f} {max_fps:<12.1f} {ft:.2f}ms")
            
            print("\n" + "="*70)
        
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
    
    # Toggle auto-cycle
    if key == 'a':
        AUTO_CYCLE = not AUTO_CYCLE
        status = "ENABLED" if AUTO_CYCLE else "DISABLED"
        print(f"\n[AUTO-CYCLE] {status}")
    
    # Reset current test
    if key == 'r':
        print(f"\n[RESET] Restarting test {CURRENT_TEST+1}/{len(TEST_CONFIGS)}")
        test_config = TEST_CONFIGS[CURRENT_TEST]
        create_test(test_config["n_agents"], test_config["color"])
    
    # Manual cycle through tests (disables auto)
    if key == 'space':
        AUTO_CYCLE = False
        CURRENT_TEST = (CURRENT_TEST + 1) % len(TEST_CONFIGS)
        test_config = TEST_CONFIGS[CURRENT_TEST]
        print(f"\n>>> Manual switch to Test {CURRENT_TEST+1}/{len(TEST_CONFIGS)}: {test_config['name']}")
        create_test(test_config["n_agents"], test_config["color"])

print(f"\nStarting scaling curve test (auto-cycle: {AUTO_CYCLE})...")
app.run()
