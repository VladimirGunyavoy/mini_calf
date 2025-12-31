"""
Test TD3 vs CALF - Dual Visualization
======================================

Phase 9.5: Side-by-side comparison of real TD3 agent vs CALF policy.
"""

import sys
from pathlib import Path
import numpy as np
from ursina import *

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from physics.policies.td3_policy import TD3Policy
from physics.policies.pd_policy import PDPolicy
from physics.policies.calf_policy import CALFPolicy
from physics.vectorized_env import VectorizedEnvironment
from visuals.multi_color_trail import MultiColorTrail
from ursina import Vec3, Vec4, color


# Global variables
vec_env_td3 = None
vec_env_calf = None
points_td3 = []
points_calf = []
trails_td3 = []
trails_calf = []
stats = {
    'td3_success': 0,
    'calf_success': 0,
    'td3_resets': 0,
    'calf_resets': 0,
    'td3_distances': [],
    'calf_distances': [],
    'td3_steps_to_goal': [],
    'calf_steps_to_goal': [],
    'step_counter': 0
}


def update():
    """Update loop"""
    global vec_env_td3, vec_env_calf, points_td3, points_calf
    global trails_td3, trails_calf, stats

    # TD3 group step
    vec_env_td3.step()
    new_states_td3 = vec_env_td3.get_states()

    # CALF group step
    vec_env_calf.step()
    new_states_calf = vec_env_calf.get_states()

    # Update visuals for TD3 (left side, offset=-8)
    for i in range(vec_env_td3.n_envs):
        x, v = new_states_td3[i]
        x_offset = x - 8  # Shift left
        points_td3[i].position = Vec3(x_offset, 0, v)
        trails_td3[i].add_point((x_offset, 0, v), mode='td3')

        # Track distance
        distance = np.linalg.norm(new_states_td3[i])
        stats['td3_distances'].append(distance)

        # Check for success (close to goal)
        if distance < 0.15:
            stats['td3_success'] += 1
            stats['td3_steps_to_goal'].append(stats['step_counter'])
            stats['td3_resets'] += 1
            trails_td3[i].clear()
            # Reset to random position
            new_state = np.array([np.random.uniform(-2, 2), np.random.uniform(-0.5, 0.5)])
            vec_env_td3.envs[i].state = new_state

    # Update visuals for CALF (right side, offset=+8)
    for i in range(vec_env_calf.n_envs):
        x, v = new_states_calf[i]
        x_offset = x + 8  # Shift right
        mode = vec_env_calf.policy.get_mode_for_env(i)

        # Color based on CALF mode
        if mode == 'td3':
            points_calf[i].color = Vec4(0.2, 0.3, 0.8, 1)  # Blue
        elif mode == 'relax':
            points_calf[i].color = Vec4(0.2, 0.6, 0.3, 1)  # Green
        elif mode == 'fallback':
            points_calf[i].color = Vec4(0.8, 0.4, 0.15, 1)  # Orange

        points_calf[i].position = Vec3(x_offset, 0, v)
        trails_calf[i].add_point((x_offset, 0, v), mode=mode)

        # Track distance
        distance = np.linalg.norm(new_states_calf[i])
        stats['calf_distances'].append(distance)

        # Check for success (close to goal)
        if distance < 0.15:
            stats['calf_success'] += 1
            stats['calf_steps_to_goal'].append(stats['step_counter'])
            stats['calf_resets'] += 1
            trails_calf[i].clear()
            # Reset to random position
            new_state = np.array([np.random.uniform(-2, 2), np.random.uniform(-0.5, 0.5)])
            vec_env_calf.envs[i].state = new_state

    stats['step_counter'] += 1

    # Update stats display
    update_stats_display()


def update_stats_display():
    """Update statistics text"""
    td3_avg_dist = np.mean(stats['td3_distances'][-100:]) if stats['td3_distances'] else 0
    calf_avg_dist = np.mean(stats['calf_distances'][-100:]) if stats['calf_distances'] else 0

    td3_success_rate = stats['td3_success'] / max(1, stats['td3_resets']) * 100
    calf_success_rate = stats['calf_success'] / max(1, stats['calf_resets']) * 100

    td3_avg_steps = np.mean(stats['td3_steps_to_goal']) if stats['td3_steps_to_goal'] else 0
    calf_avg_steps = np.mean(stats['calf_steps_to_goal']) if stats['calf_steps_to_goal'] else 0

    # Determine winner
    better_policy = ""
    if calf_success_rate > td3_success_rate + 5:
        better_policy = "CALF BETTER"
    elif td3_success_rate > calf_success_rate + 5:
        better_policy = "TD3 BETTER"
    else:
        better_policy = "TIED"

    stats_text.text = f'''TD3 vs CALF Comparison

Step: {stats['step_counter']}

=== TD3 (LEFT, BLUE) ===
Success: {stats['td3_success']}/{stats['td3_resets']} ({td3_success_rate:.1f}%)
Avg Distance: {td3_avg_dist:.4f}
Avg Steps to Goal: {td3_avg_steps:.0f}

=== CALF (RIGHT, MULTI-COLOR) ===
Success: {stats['calf_success']}/{stats['calf_resets']} ({calf_success_rate:.1f}%)
Avg Distance: {calf_avg_dist:.4f}
Avg Steps to Goal: {calf_avg_steps:.0f}

>>> {better_policy} <<<

Controls:
  R - Reset all
  ESC - Quit
'''


def main():
    """Main visualization"""
    global vec_env_td3, vec_env_calf
    global points_td3, points_calf, trails_td3, trails_calf
    global stats_text

    app = Ursina()

    # Setup camera
    camera.position = (0, 25, 0)
    camera.look_at((0, 0, 0))
    camera.rotation_x = 90

    # Dark background
    Sky(color=Vec4(0.04, 0.04, 0.08, 1))

    # Ground
    Entity(
        model='plane',
        scale=60,
        color=Vec4(0.12, 0.12, 0.16, 1),
        position=(0, -0.1, 0)
    )

    print("\n" + "=" * 60)
    print("Loading Policies")
    print("=" * 60)

    # Load TD3 policy
    model_path = Path(__file__).parent.parent.parent / "RL" / "calf_model.pth"

    try:
        td3_policy = TD3Policy.create_from_checkpoint(
            checkpoint_path=str(model_path),
            state_dim=2,
            action_dim=1,
            max_action=5.0
        )
        print("[OK] TD3 policy loaded!")
    except Exception as e:
        print(f"[ERROR] Failed to load TD3: {e}")
        print("Using stub mode")
        td3_policy = TD3Policy(agent=None, action_dim=1, action_scale=0.5)

    # Create CALF policy (TD3 + PD fallback)
    pd_policy = PDPolicy(
        kp=1.0,
        kd=1.0,
        target=np.array([0.0]),  # 1D target
        dim=1  # 1D control
    )

    try:
        td3_for_calf = TD3Policy.create_from_checkpoint(
            checkpoint_path=str(model_path),
            state_dim=2,
            action_dim=1,
            max_action=5.0
        )
        calf_policy = CALFPolicy(
            td3_policy=td3_for_calf,
            pd_policy=pd_policy,
            fallback_threshold=0.3,
            relax_threshold=0.6,
            target=np.array([0.0, 0.0])  # 2D target for safety metric (position [x,v])
        )
        print("[OK] CALF policy created!")
    except Exception as e:
        print(f"[ERROR] Failed to create CALF: {e}")
        # Fallback to PD only
        calf_policy = pd_policy

    # Configuration
    n_agents = 15
    seed = 42

    # Create vectorized environments
    print(f"\nCreating {n_agents} agents per group...")

    vec_env_td3 = VectorizedEnvironment(
        n_envs=n_agents,
        policy=td3_policy,
        dt=0.01,
        seed=seed
    )

    vec_env_calf = VectorizedEnvironment(
        n_envs=n_agents,
        policy=calf_policy,
        dt=0.01,
        seed=seed
    )

    vec_env_td3.reset()
    vec_env_calf.reset()

    # Goal markers
    Entity(
        model='sphere',
        color=Vec4(0.8, 0.8, 0.3, 0.3),
        scale=0.25,
        position=(-8, 0, 0)  # TD3 goal
    )
    Entity(
        model='sphere',
        color=Vec4(0.8, 0.8, 0.3, 0.3),
        scale=0.25,
        position=(8, 0, 0)  # CALF goal
    )

    # Boundary boxes
    # TD3 box (left)
    Entity(
        model='cube',
        color=Vec4(0.2, 0.3, 0.8, 0.1),
        scale=(10, 0.1, 10),
        position=(-8, 0, 0),
        collider='box'
    )

    # CALF box (right)
    Entity(
        model='cube',
        color=Vec4(0.2, 0.6, 0.3, 0.1),
        scale=(10, 0.1, 10),
        position=(8, 0, 0),
        collider='box'
    )

    # Create visual objects
    print("Creating visuals...")

    # TD3 group (blue)
    for i in range(n_agents):
        state = vec_env_td3.envs[i].get_state()
        x, v = state
        x_offset = x - 8

        point = Entity(
            model='sphere',
            color=Vec4(0.2, 0.3, 0.8, 1),
            scale=0.1,
            position=(x_offset, 0, v)
        )
        points_td3.append(point)

        trail = MultiColorTrail(
            max_length=600,
            decimation=2,
            rebuild_frequency=10
        )
        trails_td3.append(trail)

    # CALF group (multi-color)
    for i in range(n_agents):
        state = vec_env_calf.envs[i].get_state()
        x, v = state
        x_offset = x + 8

        point = Entity(
            model='sphere',
            color=Vec4(0.8, 0.4, 0.15, 1),  # Start orange
            scale=0.1,
            position=(x_offset, 0, v)
        )
        points_calf.append(point)

        trail = MultiColorTrail(
            max_length=600,
            decimation=2,
            rebuild_frequency=10
        )
        trails_calf.append(trail)

    # Statistics text
    stats_text = Text(
        text='',
        position=(-0.85, 0.45),
        scale=1.0,
        color=color.white
    )

    # Labels
    Text(
        text='TD3',
        position=(-0.5, -0.4),
        scale=2,
        color=Vec4(0.2, 0.3, 0.8, 1)
    )
    Text(
        text='CALF',
        position=(0.35, -0.4),
        scale=2,
        color=Vec4(0.2, 0.6, 0.3, 1)
    )

    def input(key):
        if key == 'r':
            print("\nResetting all agents...")
            vec_env_td3.reset()
            vec_env_calf.reset()

            for trail in trails_td3 + trails_calf:
                trail.clear()

            # Reset stats
            stats['td3_success'] = 0
            stats['calf_success'] = 0
            stats['td3_resets'] = 0
            stats['calf_resets'] = 0
            stats['td3_distances'] = []
            stats['calf_distances'] = []
            stats['td3_steps_to_goal'] = []
            stats['calf_steps_to_goal'] = []
            stats['step_counter'] = 0

        elif key == 'escape':
            application.quit()

    print("\n" + "=" * 60)
    print("Dual Visualization Started!")
    print("=" * 60)
    print(f"LEFT (Blue): {n_agents} agents with pure TD3")
    print(f"RIGHT (Multi-color): {n_agents} agents with CALF")
    print("\nPress R to reset, ESC to quit")
    print("=" * 60)

    app.run()


if __name__ == "__main__":
    main()







