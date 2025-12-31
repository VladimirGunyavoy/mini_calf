"""
Test TD3 with Single Point Visualization
=========================================

Phase 9.3: Test real TD3 agent with PointSystem and visualization.
"""

import sys
from pathlib import Path
import numpy as np
from ursina import *

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from physics.policies.td3_policy import TD3Policy
from physics.point_system import PointSystem
from visuals.point_visual import PointVisual
from visuals.multi_color_trail import MultiColorTrail
from ursina import Vec3, Vec4, color

# Global variables for update function
policy = None
point_system = None
point_visual = None
trail = None
step_counter = 0
distance_history = []
action_history = []
stats_text = None


def update():
    """Global update function - called by Ursina every frame"""
    global step_counter, distance_history, action_history
    
    if point_system is None or policy is None:
        return
    
    # Get current state
    state = point_system.get_state()
    
    # Get action from TD3 policy
    action = policy.get_action(state)
    
    # Set action and step
    point_system.u = float(action[0])
    point_system.step()
    
    # Update visual
    new_state = point_system.get_state()
    x, v = new_state
    
    # Phase space position
    point_visual.position = Vec3(x, 0, v)
    
    # Add to trail
    trail.add_point((x, 0, v), mode='td3')
    
    # Statistics
    distance = np.linalg.norm(new_state)
    distance_history.append(distance)
    action_history.append(action[0])
    
    step_counter += 1
    
    # Update stats text
    avg_distance = np.mean(distance_history[-100:]) if len(distance_history) > 0 else 0
    avg_action = np.mean(np.abs(action_history[-100:])) if len(action_history) > 0 else 0
    
    stats_text.text = f'''TD3 Agent - Single Point

Step: {step_counter}
State: [{x:.3f}, {v:.3f}]
Action: {action[0]:.3f}
Distance: {distance:.4f}

Avg Distance (100 steps): {avg_distance:.4f}
Avg |Action| (100 steps): {avg_action:.4f}

Controls:
  R - Reset
  ESC - Quit
'''
    
    # Check convergence
    if distance < 0.1 and step_counter > 100:
        print(f"\n[OK] Converged at step {step_counter}!")
        print(f"Final state: [{x:.4f}, {v:.4f}]")
        print(f"Final distance: {distance:.4f}")


def input(key):
    """Global input function - called by Ursina for key presses"""
    global step_counter, distance_history, action_history
    
    if key == 'r':
        # Reset
        print("\nResetting...")
        step_counter = 0
        distance_history = []
        action_history = []
        
        # Random initial state
        new_state = np.random.uniform(-3, 3, 2)
        point_system.state = new_state
        
        # Clear trail
        trail.clear()
        
        print(f"New initial state: {new_state}")
    
    elif key == 'escape' or key == 'q':
        application.quit()


def main():
    """Single point with real TD3 agent"""
    app = Ursina()

    # Setup camera
    camera.position = (0, 15, 0)
    camera.look_at((0, 0, 0))
    camera.rotation_x = 90

    # Dark background
    Sky(color=Vec4(0.04, 0.04, 0.08, 1))

    # Ground
    Entity(
        model='plane',
        scale=40,
        color=Vec4(0.12, 0.12, 0.16, 1),
        position=(0, -0.1, 0)
    )

    # Goal marker
    goal_marker = Entity(
        model='sphere',
        color=Vec4(0.3, 0.8, 0.3, 0.5),
        scale=0.2,
        position=(0, 0, 0)
    )

    # Load TD3 policy
    global policy, point_system, point_visual, trail, stats_text
    global step_counter, distance_history, action_history
    
    print("\n" + "=" * 60)
    print("Loading TD3 Agent")
    print("=" * 60)

    model_path = Path(__file__).parent.parent.parent / "RL" / "calf_model.pth"

    try:
        policy = TD3Policy.create_from_checkpoint(
            checkpoint_path=str(model_path),
            state_dim=2,
            action_dim=1,
            max_action=5.0
        )
        print("[OK] TD3 agent loaded successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to load TD3 agent: {e}")
        print("Using stub mode instead")
        policy = TD3Policy(agent=None, action_dim=1, action_scale=0.5)

    # Create point system
    initial_state = np.array([3.0, 0.0])  # Start at x=3, v=0
    point_system = PointSystem(
        dt=0.01,
        initial_state=initial_state,
        controller=None  # Will use policy directly
    )

    # Create visual representation
    point_visual = Entity(
        model='sphere',
        color=Vec4(0.2, 0.3, 0.8, 1),  # Blue
        scale=0.15,
        position=(initial_state[0], 0, initial_state[1])
    )

    # Create trail
    trail = MultiColorTrail(
        max_length=1000,
        decimation=1,
        rebuild_frequency=5
    )

    # UI text
    stats_text = Text(
        text='',
        position=(-0.85, 0.45),
        scale=1.2,
        color=color.white
    )

    print("\n" + "=" * 60)
    print("Visualization started!")
    print("=" * 60)
    print("Press R to reset, ESC to quit")

    app.run()


if __name__ == "__main__":
    main()







