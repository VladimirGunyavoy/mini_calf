"""
Train pure TD3 agent with Ursina trajectory visualization
Phase 10.1: Training visualization for pure TD3 (no CALF)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from ursina import *

# Import RL components
from RL.td3 import TD3, ReplayBuffer
from RL.simple_env import PointMassEnv

# Import Ursina components
from core import Player, setup_scene
from managers import (
    InputManager, WindowManager, ZoomManager,
    ObjectManager, ColorManager, UIManager
)
from visuals import MultiColorTrail, CriticHeatmap, GridOverlay

# Training parameters
NUM_EPISODES = 500
MAX_STEPS_PER_EPISODE = 750
BATCH_SIZE = 64
START_TRAINING_STEP = 1000
EXPLORATION_NOISE = 0.3
EVAL_INTERVAL = 10
SEED = 42

# Visualization parameters
N_AGENTS_VISUAL = 25  # Number of agents to visualize simultaneously
TRAIL_MAX_LENGTH = 600
TRAIL_DECIMATION = 1  # No decimation - capture every point
TRAIL_REBUILD_FREQ = 15  # Rebuild frequency (higher = better performance, less smooth)

# Critic heatmap parameters
HEATMAP_ENABLED = True
HEATMAP_GRID_SIZE = 11  # 10x10 grid
HEATMAP_UPDATE_FREQ = 100  # Update every 100 steps
HEATMAP_HEIGHT_SCALE = 2.0  # Height scale for visualization
AGENT_HEIGHT_EPSILON = 0.15  # Base epsilon: surface/grid at epsilon, agents at Q-height + 2*epsilon

# Grid overlay parameters
GRID_OVERLAY_ENABLED = True  # Show grid on heatmap surface
GRID_NODE_SIZE = 0.12  # Size of grid nodes
GRID_LINE_THICKNESS = 3  # Thickness of grid lines
GRID_SAMPLE_STEP = 1  # Sample every N-th point from heatmap grid (1 = all points)

# Episode termination parameters
GOAL_EPSILON = 0.05  # Distance to goal for early termination
BOUNDARY_LIMIT = 5.0  # Position boundary for early termination

# Set random seeds
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================================
# SETUP URSINA BEFORE APP
# ============================================================================
WindowManager.setup_before_app(monitor="left")
app = Ursina()

# ============================================================================
# INITIALIZE COMPONENTS
# ============================================================================
player = Player()
color_manager = ColorManager()
window_manager = WindowManager(color_manager=color_manager, monitor="left")
zoom_manager = ZoomManager(player=player)
object_manager = ObjectManager(zoom_manager=zoom_manager)
input_manager = InputManager(zoom_manager=zoom_manager, player=player)
ui_manager = UIManager(
    color_manager=color_manager,
    player=player,
    zoom_manager=zoom_manager
)

# Setup scene
ground, grid, lights, frame = setup_scene(color_manager, object_manager)

print("\n" + "="*70)
print("PHASE 10.1: TD3 TRAINING WITH VISUALIZATION")
print("="*70)

# ============================================================================
# TRAINING SETUP
# ============================================================================

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

# Create environment
env = PointMassEnv(dt=0.01, max_action=5.0, goal_radius=0.1)
env.max_steps = MAX_STEPS_PER_EPISODE  # Override default max_steps
print(f"\nEnvironment created:")
print(f"  State dim: {env.state_dim}")
print(f"  Action dim: {env.action_dim}")
print(f"  Max action: {env.max_action}")
print(f"  Goal radius: {env.goal_radius}")
print(f"  Max steps per episode: {env.max_steps}")
print(f"  Goal epsilon: {GOAL_EPSILON}")
print(f"  Boundary limit: +/-{BOUNDARY_LIMIT}")

# Create TD3 agent
td3_agent = TD3(
    state_dim=env.state_dim,
    action_dim=env.action_dim,
    max_action=env.max_action,
    hidden_dim=64,
    discount=0.99,
    tau=0.005,
    policy_noise=0.2,
    noise_clip=0.5,
    policy_freq=2,
    lr=3e-4,
    device=device
)

# Replay buffer
replay_buffer = ReplayBuffer(
    state_dim=env.state_dim,
    action_dim=env.action_dim,
    max_size=100000
)

print(f"\nTD3 agent created on {device}")
print(f"Replay buffer size: {replay_buffer.max_size}")

# ============================================================================
# VISUALIZATION SETUP
# ============================================================================

# Create multiple environments for visualization
visual_envs = []
visual_points = []
visual_trails = []

for i in range(N_AGENTS_VISUAL):
    # Create environment
    vis_env = PointMassEnv(dt=0.01, max_action=5.0, goal_radius=0.1)
    vis_env.max_steps = MAX_STEPS_PER_EPISODE  # Override default max_steps
    vis_env.reset()
    visual_envs.append(vis_env)

    # Get initial position
    state = vis_env.state
    x, v = state[0], state[1]
    pos = (x, 0.1, v)

    # Create visual point
    point = object_manager.create_object(
        name=f'td3_point_{i}',
        model='sphere',
        position=pos,
        scale=0.12,
        color_val=Vec4(0.2, 0.3, 0.8, 1)  # Blue
    )
    visual_points.append(point)

    # Create trail
    trail = MultiColorTrail(
        max_length=TRAIL_MAX_LENGTH,
        decimation=TRAIL_DECIMATION,
        rebuild_frequency=TRAIL_REBUILD_FREQ
    )
    visual_trails.append(trail)
    trail.add_point(pos, mode='td3')
    
    # Регистрируем trail в ZoomManager для реакции на зум
    zoom_manager.register_object(trail, f'trail_{i}')

print(f"\n[OK] Created {N_AGENTS_VISUAL} visual agents with trails")

# ============================================================================
# CRITIC HEATMAP
# ============================================================================

critic_heatmap = None
grid_overlay = None

if HEATMAP_ENABLED:
    critic_heatmap = CriticHeatmap(
        td3_agent=td3_agent,
        grid_size=HEATMAP_GRID_SIZE,
        x_range=(-BOUNDARY_LIMIT, BOUNDARY_LIMIT),
        v_range=(-BOUNDARY_LIMIT, BOUNDARY_LIMIT),
        height_scale=HEATMAP_HEIGHT_SCALE,
        update_frequency=HEATMAP_UPDATE_FREQ,
        surface_epsilon=AGENT_HEIGHT_EPSILON  # Surface at epsilon above ground
    )
    # Регистрируем heatmap в ZoomManager для реакции на зум
    zoom_manager.register_object(critic_heatmap, 'critic_heatmap')
    print(f"[OK] Critic heatmap created ({HEATMAP_GRID_SIZE}x{HEATMAP_GRID_SIZE})")
    
    # Grid overlay will be created after training starts
    # (to avoid showing random noise from untrained critic)
    grid_overlay = None

# Goal arrow pointing up (like Y axis) - will be created after training starts
# Using custom arrow model from assets (via object_manager)
goal_arrow = None

# Boundary frame (thin lines showing the zone limits)
# Create 4 edges of the boundary zone at ±BOUNDARY_LIMIT
boundary_color = Vec4(0.2, 0.3, 0.8, 0.6)
edge_thickness = 0.05

# Top edge (z = +BOUNDARY_LIMIT)
object_manager.create_object(
    name='boundary_top',
    model='cube',
    position=(0, 0, BOUNDARY_LIMIT),
    scale=(BOUNDARY_LIMIT * 2, edge_thickness, edge_thickness),
    color_val=boundary_color
)

# Bottom edge (z = -BOUNDARY_LIMIT)
object_manager.create_object(
    name='boundary_bottom',
    model='cube',
    position=(0, 0, -BOUNDARY_LIMIT),
    scale=(BOUNDARY_LIMIT * 2, edge_thickness, edge_thickness),
    color_val=boundary_color
)

# Left edge (x = -BOUNDARY_LIMIT)
object_manager.create_object(
    name='boundary_left',
    model='cube',
    position=(-BOUNDARY_LIMIT, 0, 0),
    scale=(edge_thickness, edge_thickness, BOUNDARY_LIMIT * 2),
    color_val=boundary_color
)

# Right edge (x = +BOUNDARY_LIMIT)
object_manager.create_object(
    name='boundary_right',
    model='cube',
    position=(BOUNDARY_LIMIT, 0, 0),
    scale=(edge_thickness, edge_thickness, BOUNDARY_LIMIT * 2),
    color_val=boundary_color
)

# ============================================================================
# TRAINING STATISTICS
# ============================================================================

training_stats = {
    'episode': 0,
    'total_steps': 0,
    'episode_reward': 0.0,
    'episode_length': 0,
    'current_state': None,
    'current_action': None,
    'exploration_noise': EXPLORATION_NOISE,
    'training_started': False,
    'episode_rewards': [],
    'episode_lengths': [],
    'final_distances': [],
    'avg_critic_loss': 0.0,
    'avg_actor_loss': 0.0,
    'buffer_size': 0,
    'success_count': 0,
    'paused': False
}

# UI text for stats
stats_text = Text(
    text='',
    position=(-0.85, 0.45),
    scale=1.0,
    color=color.white
)

# Episode state
current_state = env.reset()
training_stats['current_state'] = current_state

print("\n" + "="*70)
print("[OK] TRAINING READY")
print("="*70)
print(f"\nVisualization Features:")
if HEATMAP_ENABLED:
    print(f"  - Critic heatmap: {HEATMAP_GRID_SIZE}x{HEATMAP_GRID_SIZE} grid (height = Q^2 for steeper surface)")
    print(f"  - Update frequency: every {HEATMAP_UPDATE_FREQ} steps")
    print(f"  - Surface height: epsilon ({AGENT_HEIGHT_EPSILON}) above ground")
    print(f"  - Grid overlay: epsilon ({AGENT_HEIGHT_EPSILON}) above ground, sample_step={GRID_SAMPLE_STEP}")
    print(f"  - Agent height: Q-value^2 + 2*epsilon ({2*AGENT_HEIGHT_EPSILON}) (agents float above grid)")
    print(f"  - Agent Q-values: interpolated from cached grid (synchronized with surface)")
    print(f"  - Trails follow agent height")
    print(f"  - Goal arrow: purple arrow at origin (0,0) at Q-value height, pointing up")
    if GRID_OVERLAY_ENABLED:
        print(f"  - Grid will appear after training starts (at step {START_TRAINING_STEP})")
print(f"\nControls:")
print(f"  P - Pause/Resume training")
print(f"  Q - Quit")
print(f"  WASD - Move camera")
print(f"  Scroll - Zoom")
print()

def get_agent_height(state):
    """
    Get Y coordinate for agent based on critic Q-value

    Parameters:
    -----------
    state : np.ndarray
        State vector [x, v]

    Returns:
    --------
    float
        Y coordinate (Q-value height + 2*epsilon)
    """
    if critic_heatmap is not None and training_stats['training_started']:
        # Use cached Q-values from grid (synchronized with surface/grid)
        q_height = critic_heatmap.get_q_value_for_state(state, use_cached=True)
        return q_height + 2 * AGENT_HEIGHT_EPSILON  # Agents at 2*epsilon
    else:
        # Before training starts, use default height
        return 0.1

def get_agent_heights_batch(states):
    """
    Get Y coordinates for a batch of agents based on critic Q-values (optimized)

    Parameters:
    -----------
    states : np.ndarray
        Batch of state vectors, shape (batch_size, 2) where each is [x, v]

    Returns:
    --------
    np.ndarray
        Y coordinates, shape (batch_size,)
    """
    if critic_heatmap is not None and training_stats['training_started']:
        # Batch interpolation from cached Q-values grid
        q_heights = critic_heatmap.get_q_value_for_states_batch(states, use_cached=True)
        return q_heights + 2 * AGENT_HEIGHT_EPSILON  # Agents at 2*epsilon
    else:
        # Before training starts, use default height
        return np.full(len(states), 0.1)

def update():
    """Main training loop - called every frame"""
    global current_state, training_stats

    if training_stats['paused']:
        return

    # Training step
    state = current_state
    total_steps = training_stats['total_steps']

    # Select action
    if total_steps < START_TRAINING_STEP:
        # Random exploration at the start
        action = np.random.uniform(-env.max_action, env.max_action, size=env.action_dim)
    else:
        # TD3 policy with exploration noise
        action = td3_agent.select_action(state, noise=training_stats['exploration_noise'])
        training_stats['training_started'] = True

    training_stats['current_action'] = action

    # Execute action in environment
    next_state, reward, done, info = env.step(action)

    # Check early termination conditions
    distance = np.linalg.norm(next_state)
    position = abs(next_state[0])

    if distance < GOAL_EPSILON:
        done = True
        info['in_goal'] = True
        info['termination_reason'] = 'goal_reached'
    elif position > BOUNDARY_LIMIT:
        done = True
        info['termination_reason'] = 'out_of_bounds'

    # Store in replay buffer
    replay_buffer.add(state, action, next_state, reward, float(done))
    training_stats['buffer_size'] = replay_buffer.size

    # Train TD3
    if total_steps >= START_TRAINING_STEP and replay_buffer.size >= BATCH_SIZE:
        train_info = td3_agent.train(replay_buffer, BATCH_SIZE)
        training_stats['avg_critic_loss'] = train_info['critic_loss']
        if train_info['actor_loss'] is not None:
            training_stats['avg_actor_loss'] = train_info['actor_loss']

    # Update statistics
    training_stats['episode_reward'] += reward
    training_stats['episode_length'] += 1
    training_stats['total_steps'] += 1

    # Update state
    current_state = next_state

    # Check episode termination
    if done:
        # Record episode statistics
        training_stats['episode_rewards'].append(training_stats['episode_reward'])
        training_stats['episode_lengths'].append(training_stats['episode_length'])
        training_stats['final_distances'].append(info['distance_to_goal'])

        if info['in_goal']:
            training_stats['success_count'] += 1

        # Reset for next episode
        training_stats['episode'] += 1
        current_state = env.reset()
        training_stats['episode_reward'] = 0.0
        training_stats['episode_length'] = 0

        # Evaluation interval
        if training_stats['episode'] % EVAL_INTERVAL == 0:
            print(f"\nEpisode {training_stats['episode']}:")
            avg_reward = np.mean(training_stats['episode_rewards'][-EVAL_INTERVAL:])
            avg_length = np.mean(training_stats['episode_lengths'][-EVAL_INTERVAL:])
            avg_distance = np.mean(training_stats['final_distances'][-EVAL_INTERVAL:])
            success_rate = training_stats['success_count'] / training_stats['episode'] * 100
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Length: {avg_length:.0f}")
            print(f"  Avg Final Distance: {avg_distance:.4f}")
            print(f"  Success Rate: {success_rate:.1f}%")
            print(f"  Buffer Size: {replay_buffer.size}")

            # Save checkpoint
            if training_stats['episode'] % (EVAL_INTERVAL * 5) == 0:
                checkpoint_path = Path(__file__).parent / f"checkpoints/td3_episode_{training_stats['episode']}.pth"
                checkpoint_path.parent.mkdir(exist_ok=True)
                td3_agent.save(str(checkpoint_path))
                print(f"  Checkpoint saved: {checkpoint_path}")

    # Update visual agents (continuous flow - reset immediately when done)
    # Batch inference for all agents (more efficient than 25 individual calls)
    if training_stats['training_started']:
        # Collect all states
        vis_states = np.array([env.state for env in visual_envs])
        # Batch inference: 1 forward pass instead of 25
        vis_actions = td3_agent.select_action_batch(vis_states, noise=0.0)
    else:
        # Random actions during exploration
        vis_actions = np.random.uniform(-env.max_action, env.max_action, size=(len(visual_envs), env.action_dim))

    # Step all environments and collect next states
    vis_next_states = []
    vis_done_flags = []

    for i in range(len(visual_envs)):
        vis_env = visual_envs[i]
        vis_action = vis_actions[i]

        # Step visual environment
        vis_next_state, vis_reward, vis_done, vis_info = vis_env.step(vis_action)

        # Check early termination for visual agents
        vis_distance = np.linalg.norm(vis_next_state)
        vis_position = abs(vis_next_state[0])

        if vis_distance < GOAL_EPSILON:
            vis_done = True
        elif vis_position > BOUNDARY_LIMIT:
            vis_done = True

        vis_next_states.append(vis_next_state)
        vis_done_flags.append(vis_done)

    # Batch compute heights for all agents (1 call instead of 25-50)
    vis_next_states_array = np.array(vis_next_states)
    vis_heights = get_agent_heights_batch(vis_next_states_array)

    # Update positions with batch-computed heights
    for i in range(len(visual_envs)):
        vis_next_state = vis_next_states[i]
        vis_done = vis_done_flags[i]

        # Update position (обновляем real_position для зума)
        x, v = vis_next_state[0], vis_next_state[1]
        y = vis_heights[i]  # Y from batch computation
        position = (x, y, v)
        visual_points[i].real_position = np.array(position)
        # Применяем текущую трансформацию (без пересчета всех объектов)
        visual_points[i].apply_transform(zoom_manager.a_transformation, zoom_manager.b_translation)
        visual_trails[i].add_point(position, mode='td3')

        # Continuous flow: reset immediately when done (no batching)
        if vis_done:
            visual_trails[i].clear()
            visual_envs[i].reset()
            # Update position to new starting point (обновляем real_position для зума)
            new_state = visual_envs[i].state
            x, v = new_state[0], new_state[1]
            y = get_agent_height(new_state)  # Single call for reset (rare)
            new_position = (x, y, v)
            visual_points[i].real_position = np.array(new_position)
            # Применяем текущую трансформацию
            visual_points[i].apply_transform(zoom_manager.a_transformation, zoom_manager.b_translation)
            visual_trails[i].add_point(new_position, mode='td3')

    # Update critic heatmap
    if critic_heatmap is not None and training_stats['training_started']:
        # Check if heatmap was updated
        old_step = critic_heatmap.step_counter
        critic_heatmap.update(total_steps)
        
        # Update grid overlay and goal position if heatmap was updated
        if critic_heatmap.step_counter != old_step:
            # Create grid overlay on first heatmap update (after training starts)
            global grid_overlay, goal_arrow
            if grid_overlay is None and GRID_OVERLAY_ENABLED:
                grid_overlay = GridOverlay(
                    critic_heatmap=critic_heatmap,
                    node_size=GRID_NODE_SIZE,
                    line_thickness=GRID_LINE_THICKNESS,
                    sample_step=GRID_SAMPLE_STEP,
                    grid_epsilon_multiplier=1.0  # Grid at same height as surface (epsilon)
                )
                zoom_manager.register_object(grid_overlay, 'grid_overlay')
                print("[OK] Grid overlay created (after training started)")
            
            # Create goal arrow on first heatmap update (after training starts)
            if goal_arrow is None:
                goal_arrow = object_manager.create_object(
                    name='goal_arrow',
                    model='assets/arrow.obj',
                    position=(0, 0, 0),
                    rotation=(0, 0, 0),  # Pointing up (Y-axis)
                    scale=2,
                    color_val=Vec4(0.7, 0.2, 0.9, 1.0)  # Purple
                )
                goal_arrow.unlit = True
                print("[OK] Goal arrow created (after training started)")
            
            # Update grid together with heatmap (same frequency)
            if grid_overlay is not None:
                grid_overlay.update()
            
            # Update goal arrow position (if created)
            if goal_arrow is not None:
                goal_state = np.array([0.0, 0.0])
                goal_height = critic_heatmap.get_q_value_for_state(goal_state, use_cached=True)
                
                # Position arrow at goal height
                goal_arrow.real_position = np.array([0, goal_height, 0])
                
                # Apply zoom transformation
                goal_arrow.apply_transform(zoom_manager.a_transformation, zoom_manager.b_translation)

    # Update statistics display
    avg_reward = np.mean(training_stats['episode_rewards'][-10:]) if training_stats['episode_rewards'] else 0
    success_rate = training_stats['success_count'] / max(1, training_stats['episode']) * 100

    # Get Q-value range and performance stats for display
    q_min, q_max = (0, 0)
    heatmap_perf = None
    if critic_heatmap is not None:
        q_min, q_max = critic_heatmap.get_q_range()
        heatmap_perf = critic_heatmap.get_performance_stats()

    # Build performance section
    perf_section = ""
    if heatmap_perf and heatmap_perf['update_count'] > 0:
        perf_section = f'''
=== Heatmap Performance ===
Updates: {heatmap_perf['update_count']}
Avg: {heatmap_perf['avg_time_ms']:.2f}ms ({heatmap_perf['avg_fps']:.1f} FPS)
EMA: {heatmap_perf['ema_time_ms']:.2f}ms ({heatmap_perf['ema_fps']:.1f} FPS)
'''

    stats_text.text = f'''TD3 Training Progress

Episode: {training_stats['episode']} / {NUM_EPISODES}
Total Steps: {training_stats['total_steps']}

=== Current Episode ===
Reward: {training_stats['episode_reward']:.2f}
Length: {training_stats['episode_length']}
Action: {training_stats['current_action'][0]:.3f}

=== Statistics (last 10 episodes) ===
Avg Reward: {avg_reward:.2f}
Success Rate: {success_rate:.1f}%

=== Training ===
Status: {"TRAINING" if training_stats['training_started'] else "EXPLORATION"}
Noise: {training_stats['exploration_noise']:.3f}
Buffer: {training_stats['buffer_size']} / {replay_buffer.max_size}
Critic Loss: {training_stats['avg_critic_loss']:.4f}
Actor Loss: {training_stats['avg_actor_loss']:.4f}

=== Critic Q-values ===
Min: {q_min:.2f}
Max: {q_max:.2f}{perf_section}
Press P to pause'''

    # Update managers
    if hasattr(input_manager, 'update'):
        input_manager.update()
    if hasattr(zoom_manager, 'update'):
        zoom_manager.update()
    if hasattr(object_manager, 'update'):
        object_manager.update()
    ui_manager.update()

    # Auto-stop after NUM_EPISODES
    if training_stats['episode'] >= NUM_EPISODES:
        print("\n" + "="*70)
        print(f"Training completed: {NUM_EPISODES} episodes")
        print("="*70)

        # Save final model
        final_model_path = Path(__file__).parent / "trained_td3_final.pth"
        td3_agent.save(str(final_model_path))
        print(f"\nFinal model saved: {final_model_path}")

        # Print final statistics
        print(f"\nFinal Statistics:")
        print(f"  Total Steps: {training_stats['total_steps']}")
        print(f"  Success Count: {training_stats['success_count']}")
        print(f"  Success Rate: {success_rate:.1f}%")
        print(f"  Avg Reward (last 50): {np.mean(training_stats['episode_rewards'][-50:]):.2f}")

        application.quit()

def input(key):
    """Handle input"""
    if key == 'p':
        training_stats['paused'] = not training_stats['paused']
        print(f"\nTraining {'PAUSED' if training_stats['paused'] else 'RESUMED'}")
    else:
        input_manager.handle_input(key)

app.run()
