"""
Test Simple Trails with 50 Agents - Phase 5.4 & 5.5
Визуализация траекторий для 50 точек с сбросом по завершению эпизода
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ursina import Ursina, camera, Vec3, color, Text, time as ursina_time, Entity
import numpy as np

from physics import VectorizedEnvironment
from physics.policies import PDPolicy
from visuals import SimpleTrail
from core import Player
from core.scene_setup import setup_lighting, create_ground
from managers import ColorManager

# Global variables for update function
vec_env = None
points = []
trails = []
stats_text = None
frame_count = 0
fps_samples = []
start_time = 0
episode_steps = 0
max_episode_steps = 2000  # Reset after 2000 steps (more time for longer trails)

def update():
    """Global update function called by Ursina every frame"""
    global frame_count, start_time, fps_samples, episode_steps
    
    # Step simulation
    vec_env.step()
    episode_steps += 1
    
    # Check for episode reset
    reset_count = 0
    for i in range(len(points)):
        state = vec_env.envs[i].state
        x, v = state[0], state[1]
        
        # Reset conditions: reached goal or max steps
        distance_to_goal = np.sqrt(x**2 + v**2)
        
        if distance_to_goal < 0.1 or episode_steps >= max_episode_steps:
            # Clear trail
            trails[i].clear()
            
            # Set new random initial position (reset)
            x_new = np.random.uniform(-3, 3)
            v_new = np.random.uniform(-1, 1)
            vec_env.envs[i].state = np.array([x_new, v_new])
            
            reset_count += 1
    
    # Reset episode counter if needed
    if episode_steps >= max_episode_steps:
        episode_steps = 0
    
    # Update visual points and trails
    for i in range(len(points)):
        state = vec_env.envs[i].state
        
        # Phase space: X-axis = position, Z-axis = velocity
        x = state[0]
        v = state[1]
        position = (x, 0.1, v)
        
        # Update point position
        points[i].position = position
        
        # Add point to trail
        trails[i].add_point(position)
    
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
        
        # Update stats
        stats_text.text = f'''Phase 5.4-5.5: 50 Agents with Trails
Agents: {len(points)}
FPS: {avg_fps:.0f}
Episode step: {episode_steps}/{max_episode_steps}
Trails active: {len([t for t in trails if len(t.positions) > 0])}'''

def main():
    """Test trails with 50 agents and episode reset"""
    global vec_env, points, trails, stats_text, frame_count, fps_samples, start_time, episode_steps
    
    # Create app
    app = Ursina()
    
    # Setup
    color_manager = ColorManager()
    player = Player()
    
    # Scene setup
    create_ground(color_manager, object_manager=None)
    setup_lighting(color_manager)
    
    # Camera - farther back to see all 50 agents
    camera.position = (0, 20, -30)
    camera.rotation_x = 35
    
    # Configuration
    n_agents = 50
    
    # Policy: PD controller
    pd_policy = PDPolicy(
        kp=1.0,
        kd=0.8,
        target=np.array([0.0, 0.0])
    )
    
    # Vectorized environment
    vec_env = VectorizedEnvironment(
        n_envs=n_agents,
        policy=pd_policy,
        dt=0.01,
        seed=42
    )
    
    vec_env.reset()
    
    # Set initial positions with larger spread
    np.random.seed(42)
    for i in range(n_agents):
        x = np.random.uniform(-3, 3)
        v = np.random.uniform(-1, 1)
        vec_env.envs[i].state = np.array([x, v])
    
    print(f"\n[DEBUG] Created {n_agents} agents with spread initial positions")
    
    # Create visual points and trails
    # Use fewer unique colors, cycling through them
    colors_list = [
        color.red, color.green, color.blue, color.yellow, 
        color.cyan, color.magenta, color.orange, color.pink
    ]
    
    for i in range(n_agents):
        state = vec_env.envs[i].state
        x, v = state[0], state[1]
        pos = (x, 0.1, v)
        
        # Visual point with smaller scale for 50 agents
        point_visual = Entity(
            model='sphere',
            color=colors_list[i % len(colors_list)],
            scale=0.1,  # Smaller for 50 agents
            position=pos
        )
        points.append(point_visual)
        
        # Trail with higher decimation for performance
        trail = SimpleTrail(
            trail_color=colors_list[i % len(colors_list)],
            max_length=600,  # Longer trails
            decimation=5,  # Every 5th point (higher decimation)
            rebuild_frequency=20  # Rebuild even less frequently
        )
        trails.append(trail)
        trail.add_point(pos)
    
    print(f"[OK] Created {n_agents} visual points with optimized trails")
    
    # Stats UI
    stats_text = Text(
        text='',
        position=(-0.85, 0.45),
        scale=1.2,
        color=color.white
    )
    
    # Performance tracking
    frame_count = 0
    fps_samples = []
    start_time = ursina_time.time()
    episode_steps = 0
    
    # Run
    app.run()


if __name__ == '__main__':
    print("[Phase 5.4-5.5] Testing 50 agents with trails and episode reset...")
    main()


