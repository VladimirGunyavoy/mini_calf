"""
Test Simple Trails - Phase 5.2
Визуализация траекторий для 10 точек с PD политикой
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
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

def update():
    """Global update function called by Ursina every frame"""
    global frame_count, start_time, fps_samples
    
    # Step simulation
    vec_env.step()
    
    # Update visual points and trails
    for i in range(len(points)):
        state = vec_env.envs[i].state
        
        # Phase space: X-axis = position, Z-axis = velocity
        x = state[0]
        v = state[1]
        position = (x, 0.1, v)  # Y=0.1 to be visible above floor
        
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
        
        # Keep last 10 samples
        if len(fps_samples) > 10:
            fps_samples.pop(0)
        
        avg_fps = np.mean(fps_samples) if fps_samples else 0
        
        # Update stats
        stats_text.text = f'''Phase 5.2: Simple Trails Test
Agents: {len(points)}
FPS: {avg_fps:.0f}
Trails: {len([t for t in trails if len(t.positions) > 0])} active'''

def main():
    """Test simple trails with 10 agents"""
    global vec_env, points, trails, stats_text, frame_count, fps_samples, start_time
    
    # Create app
    app = Ursina()
    
    # Setup
    color_manager = ColorManager()
    player = Player()
    
    # Scene setup (minimal - without ObjectManager)
    create_ground(color_manager, object_manager=None)
    setup_lighting(color_manager)
    
    # Camera
    camera.position = (0, 15, -25)
    camera.rotation_x = 30
    
    # Configuration
    n_agents = 10
    
    # Policy: PD controller targeting origin
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
    
    # Reset to get initial states with larger spread
    vec_env.reset()
    
    # Override initial states with larger spread for better visualization
    np.random.seed(42)
    for i in range(n_agents):
        # Random positions in range [-3, 3] for position, [-1, 1] for velocity
        x = np.random.uniform(-3, 3)
        v = np.random.uniform(-1, 1)
        vec_env.envs[i].state = np.array([x, v])
    
    # Print initial positions
    print("\n[DEBUG] Initial agent positions (spread out):")
    for i in range(n_agents):
        state = vec_env.envs[i].state
        print(f"  Agent {i}: x={state[0]:.2f}, v={state[1]:.2f}")
    
    # Create visual points and trails for each agent
    points = []
    trails = []
    colors_list = [
        color.red, color.green, color.blue, color.yellow, color.cyan,
        color.magenta, color.orange, color.pink, color.lime, color.violet
    ]
    
    for i in range(n_agents):
        # Get initial position
        state = vec_env.envs[i].state
        x, v = state[0], state[1]
        pos = (x, 0.1, v)  # Y=0.1 to be visible above floor
        
        # Visual point (sphere) with unique color
        point_visual = Entity(
            model='sphere',
            color=colors_list[i % len(colors_list)],
            scale=0.15,  # Larger for visibility
            position=pos
        )
        points.append(point_visual)
        
        # Trail with same color as point
        trail = SimpleTrail(
            trail_color=colors_list[i % len(colors_list)],
            max_length=300,
            decimation=2,  # Every 2nd point for performance
            rebuild_frequency=10  # Rebuild mesh every 10 additions
        )
        trails.append(trail)
        
        # Add initial point to trail
        trail.add_point(pos)
    
    print(f"[OK] Created {n_agents} visual points with trails")
    
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
    
    # Run
    app.run()


if __name__ == '__main__':
    print("[Phase 5] Testing Simple Trails with 10 agents...")
    main()
