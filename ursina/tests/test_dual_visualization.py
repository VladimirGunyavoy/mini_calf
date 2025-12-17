"""
Test Dual Visualization (TD3 vs PD) - Phase 6
Визуализация двух групп агентов для сравнения TD3 и PD политик
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ursina import Ursina, camera, Vec3, color, Text, time as ursina_time, Entity
import numpy as np

from physics import VectorizedEnvironment
from physics.policies import PDPolicy, TD3Policy
from visuals import SimpleTrail
from core import Player
from core.scene_setup import setup_lighting, create_ground
from managers import ColorManager

# Global variables for update function
vec_env_td3 = None
vec_env_pd = None
points_td3 = []
points_pd = []
trails_td3 = []
trails_pd = []
stats_text = None
frame_count = 0
fps_samples = []
start_time = 0
episode_steps = 0
max_episode_steps = 2000

# Statistics tracking
td3_successes = 0
pd_successes = 0
td3_total_distance = 0
pd_total_distance = 0
td3_resets = 0
pd_resets = 0
td3_total_steps_to_goal = 0
pd_total_steps_to_goal = 0

def update():
    """Global update function called by Ursina every frame"""
    global frame_count, start_time, fps_samples, episode_steps
    global td3_successes, pd_successes, td3_total_distance, pd_total_distance
    global td3_resets, pd_resets, td3_total_steps_to_goal, pd_total_steps_to_goal
    
    # Step both simulations
    vec_env_td3.step()
    vec_env_pd.step()
    episode_steps += 1
    
    # Reset counters for this frame
    frame_td3_distance = 0
    frame_pd_distance = 0
    
    # Update TD3 group (LEFT side)
    for i in range(len(points_td3)):
        state = vec_env_td3.envs[i].state
        x, v = state[0], state[1]
        
        # Phase space: X-axis = position, Z-axis = velocity
        # LEFT side: negative Z offset
        position = (x, 0.1, v - 5)  # Shift left by -5 on Z
        
        # Calculate distance to goal
        distance_to_goal = np.sqrt(x**2 + v**2)
        frame_td3_distance += distance_to_goal
        
        # Reset conditions: reached goal or max steps
        if distance_to_goal < 0.1 or episode_steps >= max_episode_steps:
            td3_resets += 1
            
            if distance_to_goal < 0.1:
                td3_successes += 1
                td3_total_steps_to_goal += episode_steps
            
            # Clear trail
            trails_td3[i].clear()
            
            # Set new random initial position
            x_new = np.random.uniform(-3, 3)
            v_new = np.random.uniform(-1, 1)
            vec_env_td3.envs[i].state = np.array([x_new, v_new])
        
        # Update point position
        points_td3[i].position = position
        
        # Add point to trail
        trails_td3[i].add_point(position)
    
    # Update PD group (RIGHT side)
    for i in range(len(points_pd)):
        state = vec_env_pd.envs[i].state
        x, v = state[0], state[1]
        
        # Phase space: X-axis = position, Z-axis = velocity
        # RIGHT side: positive Z offset
        position = (x, 0.1, v + 5)  # Shift right by +5 on Z
        
        # Calculate distance to goal
        distance_to_goal = np.sqrt(x**2 + v**2)
        frame_pd_distance += distance_to_goal
        
        # Reset conditions: reached goal or max steps
        if distance_to_goal < 0.1 or episode_steps >= max_episode_steps:
            pd_resets += 1
            
            if distance_to_goal < 0.1:
                pd_successes += 1
                pd_total_steps_to_goal += episode_steps
            
            # Clear trail
            trails_pd[i].clear()
            
            # Set new random initial position
            x_new = np.random.uniform(-3, 3)
            v_new = np.random.uniform(-1, 1)
            vec_env_pd.envs[i].state = np.array([x_new, v_new])
        
        # Update point position
        points_pd[i].position = position
        
        # Add point to trail
        trails_pd[i].add_point(position)
    
    # Update distances
    td3_total_distance += frame_td3_distance
    pd_total_distance += frame_pd_distance
    
    # Reset episode counter if needed
    if episode_steps >= max_episode_steps:
        episode_steps = 0
    
    # FPS calculation
    frame_count += 1
    if frame_count % 60 == 0:
        elapsed = ursina_time.time() - start_time
        fps = 60 / elapsed if elapsed > 0 else 0
        fps_samples.append(fps)
        start_time = ursina_time.time()
        
        if len(fps_samples) > 10:
            fps_samples.pop(0)
        
        avg_fps = np.mean(fps_samples) if fps_samples else 0
        
        # Calculate average distances
        avg_td3_dist = frame_td3_distance / len(points_td3) if points_td3 else 0
        avg_pd_dist = frame_pd_distance / len(points_pd) if points_pd else 0
        
        # Calculate success rates
        td3_success_rate = (td3_successes / td3_resets * 100) if td3_resets > 0 else 0
        pd_success_rate = (pd_successes / pd_resets * 100) if pd_resets > 0 else 0
        
        # Calculate average steps to goal
        avg_td3_steps = (td3_total_steps_to_goal / td3_successes) if td3_successes > 0 else 0
        avg_pd_steps = (pd_total_steps_to_goal / pd_successes) if pd_successes > 0 else 0
        
        # Winner indicator
        winner_text = ""
        if td3_success_rate > pd_success_rate:
            winner_text = " <- BETTER"
        elif pd_success_rate > td3_success_rate:
            winner_text = " <- BETTER"
        
        # Update stats with comprehensive comparison
        stats_text.text = f'''Phase 6: Dual Visualization (TD3 vs PD)
===== TD3 (RED, LEFT) =====
Successes: {td3_successes}/{td3_resets} ({td3_success_rate:.1f}%){winner_text if td3_success_rate > pd_success_rate else ''}
Avg distance: {avg_td3_dist:.2f}
Avg steps to goal: {avg_td3_steps:.0f}

===== PD (GREEN, RIGHT) =====
Successes: {pd_successes}/{pd_resets} ({pd_success_rate:.1f}%){winner_text if pd_success_rate > td3_success_rate else ''}
Avg distance: {avg_pd_dist:.2f}
Avg steps to goal: {avg_pd_steps:.0f}

FPS: {avg_fps:.0f} | Step: {episode_steps}/{max_episode_steps}'''

def main():
    """Test dual visualization: TD3 vs PD"""
    global vec_env_td3, vec_env_pd, points_td3, points_pd
    global trails_td3, trails_pd, stats_text, frame_count, fps_samples, start_time, episode_steps
    global td3_successes, pd_successes, td3_resets, pd_resets
    global td3_total_steps_to_goal, pd_total_steps_to_goal
    
    # Create app
    app = Ursina()
    
    # Setup
    color_manager = ColorManager()
    player = Player()
    
    # Scene setup
    create_ground(color_manager, object_manager=None)
    setup_lighting(color_manager)
    
    # Camera - positioned to see both groups
    camera.position = (0, 25, 0)  # Higher, centered
    camera.rotation_x = 45
    
    # Configuration
    n_agents = 25  # 25 per group = 50 total
    
    # Policies
    td3_policy = TD3Policy(action_dim=1, action_scale=0.3)  # Random stub
    pd_policy = PDPolicy(
        kp=1.0,
        kd=0.8,
        target=np.array([0.0, 0.0])
    )
    
    # Vectorized environments with SAME seed for fair comparison
    seed = 42
    
    vec_env_td3 = VectorizedEnvironment(
        n_envs=n_agents,
        policy=td3_policy,
        dt=0.01,
        seed=seed
    )
    
    vec_env_pd = VectorizedEnvironment(
        n_envs=n_agents,
        policy=pd_policy,
        dt=0.01,
        seed=seed
    )
    
    vec_env_td3.reset()
    vec_env_pd.reset()
    
    # Set SAME initial positions for both groups (synchronized)
    np.random.seed(seed)
    initial_states = []
    for i in range(n_agents):
        x = np.random.uniform(-3, 3)
        v = np.random.uniform(-1, 1)
        initial_states.append([x, v])
    
    # Apply same initial conditions to both groups
    for i in range(n_agents):
        vec_env_td3.envs[i].state = np.array(initial_states[i])
        vec_env_pd.envs[i].state = np.array(initial_states[i])
    
    print(f"\n[OK] Created 2 groups of {n_agents} agents with synchronized initial states")
    
    # Create TD3 group (RED, LEFT)
    for i in range(n_agents):
        state = vec_env_td3.envs[i].state
        x, v = state[0], state[1]
        pos = (x, 0.1, v - 5)  # LEFT side
        
        # Visual point
        point_visual = Entity(
            model='sphere',
            color=color.red,
            scale=0.12,
            position=pos
        )
        points_td3.append(point_visual)
        
        # Trail (red)
        trail = SimpleTrail(
            trail_color=color.red,
            max_length=600,
            decimation=5,
            rebuild_frequency=20
        )
        trails_td3.append(trail)
        trail.add_point(pos)
    
    print(f"[OK] Created TD3 group (RED, LEFT): {n_agents} agents")
    
    # Create PD group (GREEN, RIGHT)
    for i in range(n_agents):
        state = vec_env_pd.envs[i].state
        x, v = state[0], state[1]
        pos = (x, 0.1, v + 5)  # RIGHT side
        
        # Visual point
        point_visual = Entity(
            model='sphere',
            color=color.green,
            scale=0.12,
            position=pos
        )
        points_pd.append(point_visual)
        
        # Trail (green)
        trail = SimpleTrail(
            trail_color=color.green,
            max_length=600,
            decimation=5,
            rebuild_frequency=20
        )
        trails_pd.append(trail)
        trail.add_point(pos)
    
    print(f"[OK] Created PD group (GREEN, RIGHT): {n_agents} agents")
    
    # Add visual separators (optional)
    # Center line at Z=0
    Entity(
        model='cube',
        color=color.white,
        scale=(10, 0.05, 0.1),
        position=(0, 0, 0),
        alpha=0.3
    )
    
    # Stats UI (larger for more detailed stats)
    stats_text = Text(
        text='',
        position=(-0.85, 0.48),
        scale=0.9,
        color=color.white,
        origin=(-0.5, 0.5)
    )
    
    # Performance tracking
    frame_count = 0
    fps_samples = []
    start_time = ursina_time.time()
    episode_steps = 0
    td3_successes = 0
    pd_successes = 0
    td3_resets = 0
    pd_resets = 0
    td3_total_steps_to_goal = 0
    pd_total_steps_to_goal = 0
    
    # Instructions
    instructions = Text(
        text='[Phase 6] RED (LEFT) = TD3 (random stub) | GREEN (RIGHT) = PD controller',
        position=(0, -0.45),
        scale=1.0,
        color=color.white,
        origin=(0, 0)
    )
    
    print("\n[OK] Starting dual visualization...")
    print("    TD3 (RED): Random actions (stub)")
    print("    PD (GREEN): Deterministic controller")
    print("    Initial conditions: SYNCHRONIZED")
    
    # Run
    app.run()


if __name__ == '__main__':
    print("[Phase 6.1-6.3] Testing dual visualization: TD3 vs PD...")
    main()


