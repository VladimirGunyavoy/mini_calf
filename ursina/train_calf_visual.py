"""
Train CALF agent with Ursina trajectory visualization
CALF: Critic as Lyapunov Function - Full implementation with mode switching visualization
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from ursina import *

# Import RL components
from RL.calf import CALFController
from RL.td3 import ReplayBuffer
from RL.simple_env import PointMassEnv, pd_nominal_policy

# Import Ursina components
from core import Player, setup_scene
from managers import (
    InputManager, WindowManager, ZoomManager,
    ObjectManager, ColorManager, UIManager
)
from visuals import LineTrail, CriticHeatmap, GridOverlay

# Training parameters
NUM_EPISODES = 500
MAX_STEPS_PER_EPISODE = 750
BATCH_SIZE = 64
START_TRAINING_STEP = 100
EXPLORATION_NOISE = 0.5
EVAL_INTERVAL = 10
SEED = 42
REWARD_SCALE = 10  # Scale rewards for better learning (env rewards are ~-2 to -0.1)

# Resume training from checkpoint
RESUME_TRAINING = True  # Set to True to continue training from checkpoint
RESUME_CHECKPOINT = "trained_calf_final.pth"  # Checkpoint to resume from

# CALF parameters
LAMBDA_RELAX = 0.99995  # Relaxation factor (меньше значение = быстрее уменьшается P_relax)
NU_BAR = 0.01  # Lyapunov decrease threshold
KAPPA_LOW_COEF = 0.01  # Lower K_∞ coefficient (уменьшено с 0.5 для расширения диапазона)
KAPPA_UP_COEF = 1000.0  # Upper K_∞ coefficient (увеличено с 2.0 для расширения диапазона)

# Visualization parameters
N_AGENTS_VISUAL = 5  # Number of agents to visualize simultaneously
TRAIL_MAX_LENGTH = 600
TRAIL_DECIMATION = 1  # No decimation - capture every point
TRAIL_REBUILD_FREQ = 15  # Rebuild frequency (higher = better performance)

# Critic heatmap parameters
HEATMAP_ENABLED = True
HEATMAP_GRID_SIZE = 21  # 20x20 grid (2x denser than before)
HEATMAP_UPDATE_FREQ = 500  # Update every 500 steps (~5 seconds with dt=0.01)
HEATMAP_HEIGHT_SCALE = 2.0  # Height scale for visualization
AGENT_HEIGHT_EPSILON = 0.15  # Base epsilon: surface/grid at epsilon, agents at Q-height + 2*epsilon

# Grid overlay parameters
GRID_OVERLAY_ENABLED = True  # Show grid on heatmap surface
GRID_NODE_SIZE = 0.04  # Size of grid nodes (reduced 3x)
GRID_LINE_THICKNESS = 3  # Thickness of grid lines
GRID_SAMPLE_STEP = 1  # Sample every N-th point from heatmap grid

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

# Explicitly enable FPS counter (make sure it's visible)
window.fps_counter.enabled = True
window.fps_counter.position = (0.75, 0.48)
window.fps_counter.color = color.white
window.fps_counter.scale = 1.0

print("\n" + "="*70)
print("CALF TRAINING WITH VISUALIZATION")
print("Critic as Lyapunov Function - Mode Switching Visualization")
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
print(f"\nEnvironment: PointMassEnv")
print(f"State dim: {env.state_dim}, Action dim: {env.action_dim}")
print(f"Max action: {env.max_action}, Goal radius: {env.goal_radius}")

# Nominal safe policy (PD controller)
nominal_policy = pd_nominal_policy(max_action=env.max_action, kp=2.0, kd=0.4)
print(f"\nNominal Policy: PD Controller (kp=1.0, kd=1.0)")

# Create CALF controller
calf_agent = CALFController(
    state_dim=env.state_dim,
    action_dim=env.action_dim,
    max_action=env.max_action,
    nominal_policy=nominal_policy,
    goal_region_radius=env.goal_radius,
    nu_bar=NU_BAR,
    kappa_low_coef=KAPPA_LOW_COEF,
    kappa_up_coef=KAPPA_UP_COEF,
    lambda_relax=LAMBDA_RELAX,
    hidden_dim=64,
    lr=3e-4,
    device=device,
    discount=0.99,
    tau=0.005,
    policy_noise=0.2,
    noise_clip=0.5,
    policy_freq=2
)

print(f"\nCALF Parameters:")
print(f"  Lambda_relax: {LAMBDA_RELAX}")
print(f"  Nu_bar (Lyapunov threshold): {NU_BAR}")
print(f"  Kappa coefficients: [{KAPPA_LOW_COEF}, {KAPPA_UP_COEF}]")
print(f"  Reward scale: {REWARD_SCALE}x (env rewards: ~-2.0 to -0.1)")

# Load checkpoint if resuming
if RESUME_TRAINING:
    checkpoint_path = Path(__file__).parent / RESUME_CHECKPOINT
    if checkpoint_path.exists():
        calf_agent.load(str(checkpoint_path))
        print(f"\n{'='*70}")
        print(f"RESUMING TRAINING FROM CHECKPOINT")
        print(f"{'='*70}")
        print(f"Loaded checkpoint: {checkpoint_path}")

        # Reset P_relax to 0 for strict certification
        calf_agent.P_relax = 0.0
        print(f"Set P_relax = 0.0 (no relaxation - strict mode)")

        stats = calf_agent.get_statistics()
        print(f"Loaded statistics:")
        print(f"  Total steps: {stats['total_steps']}")
        print(f"  Certification rate: {stats['certification_rate']:.3f}")
        print(f"  Intervention rate: {stats['intervention_rate']:.3f}")
        print(f"  Relax rate: {stats['relax_rate']:.3f}")
    else:
        print(f"\nWARNING: Checkpoint not found at {checkpoint_path}")
        print(f"Starting training from scratch...")

# Replay buffer
replay_buffer = ReplayBuffer(
    state_dim=env.state_dim,
    action_dim=env.action_dim,
    max_size=100000
)

# ============================================================================
# VISUAL AGENTS SETUP - НОВАЯ АРХИТЕКТУРА С КОЛЬЦЕВЫМ БУФЕРОМ
# ============================================================================

# Create visual environments
visual_envs = [PointMassEnv(dt=0.01, max_action=5.0, goal_radius=0.1)
               for _ in range(N_AGENTS_VISUAL)]

# Reset all visual environments
for ve in visual_envs:
    ve.reset()

# Step counters for visual agents (for MAX_STEPS_PER_EPISODE limit)
visual_step_counters = [0] * N_AGENTS_VISUAL

# НОВЫЙ ПОДХОД: Создаем визуальные агенты + траектории с кольцевым буфером
# Каждый агент состоит из:
# 1. Шарик агента (visual_point)
# 2. Кольцевой буфер точек траектории (trail_points)

class VisualAgent:
    """Визуальный агент с встроенной траекторией (кольцевой буфер)"""
    def __init__(self, object_manager, zoom_manager, agent_id, max_trail_length=200, decimation=2, point_size=0.03, point_color=None):
        # Шарик агента
        if point_color is None:
            point_color = Vec4(0.2, 0.4, 1.0, 1)  # Blue (default)

        self.visual_point = object_manager.create_object(
            name=f'calf_point_{agent_id}',
            model='sphere',
            position=(0, 0, 0),
            scale=0.06,
            color_val=point_color
        )

        # КОЛЬЦЕВОЙ БУФЕР для траектории (LineTrail)
        self.trail = LineTrail(
            max_points=max_trail_length,
            line_thickness=2,
            decimation=decimation,
            rebuild_freq=8
        )

        # Регистрируем траекторию в ZoomManager
        zoom_manager.register_object(self.trail, f'trail_{agent_id}')

        # Реальная позиция (для зума)
        self.real_position = np.array([0, 0, 0], dtype=float)

    def update_position(self, position, mode='td3'):
        """Обновить позицию агента и добавить точку в траекторию"""
        # Конвертируем позицию в numpy array
        if isinstance(position, tuple) or isinstance(position, list):
            real_pos = np.array(position, dtype=float)
        else:
            real_pos = position

        # Сохраняем реальную позицию (для зума)
        self.real_position = real_pos

        # КРИТИЧНО: Получаем текущие трансформации ДО применения
        a_trans = zoom_manager.a_transformation
        b_trans = zoom_manager.b_translation

        # Обновляем позицию шарика агента
        # 1. Сохраняем реальную позицию
        self.visual_point.real_position = self.real_position
        # 2. Применяем трансформацию зума вручную (как в старом коде)
        self.visual_point.apply_transform(a_trans, b_trans)

        # Маппинг режимов: CALF использует 'nominal', визуализация использует 'fallback'
        display_mode = mode
        if mode == 'nominal':
            display_mode = 'fallback'

        # Обновляем цвет шарика в зависимости от режима
        if display_mode == 'td3':
            self.visual_point.color = Vec4(0.2, 0.4, 1.0, 1)  # Blue
        elif display_mode == 'relax':
            self.visual_point.color = Vec4(0.2, 0.7, 0.3, 1)  # Green
        elif display_mode == 'fallback':
            self.visual_point.color = Vec4(1.0, 0.5, 0.1, 1)  # Orange

        # Добавляем точку в траекторию с ТЕМИ ЖЕ трансформациями
        # Используем display_mode для корректного отображения
        self.trail.add_point(self.real_position, mode=display_mode,
                           a_transform=a_trans, b_translate=b_trans)

    def clear_trail(self):
        """Очистить траекторию"""
        self.trail.clear()

# Создаем визуальных агентов
visual_agents = []
for i in range(N_AGENTS_VISUAL):
    agent = VisualAgent(
        object_manager=object_manager,
        zoom_manager=zoom_manager,
        agent_id=i,
        max_trail_length=200,
        decimation=1  # ИСПРАВЛЕНО: decimation=1 чтобы траектория не отставала
    )
    visual_agents.append(agent)

print(f"\n{N_AGENTS_VISUAL} visual agents initialized with ring buffer trails")

# ============================================================================
# TRAINING AGENT VISUALIZATION
# ============================================================================

# Create visual agent for the main training agent (with exploration noise)
# Use orange/red color to distinguish from visual agents (blue)
training_agent_visual = VisualAgent(
    object_manager=object_manager,
    zoom_manager=zoom_manager,
    agent_id=9999,  # Special ID for training agent
    max_trail_length=TRAIL_MAX_LENGTH,
    decimation=TRAIL_DECIMATION,
    point_color=Vec4(1.0, 0.5, 0.0, 1)  # Orange color
)

print(f"Training agent visualization initialized (orange)")

# ============================================================================
# Q-CERTIFICATE TIMELINE GRAPH
# ============================================================================

# Graph for Q† evolution over time (normalized coordinates)
# Origin at (6, 0, 0), X-axis = simulation time, Y-axis = Q† value
q_cert_timeline = LineTrail(
    max_points=TRAIL_MAX_LENGTH,
    line_thickness=3,
    decimation=TRAIL_DECIMATION,
    rebuild_freq=TRAIL_REBUILD_FREQ
)
zoom_manager.register_object(q_cert_timeline, 'q_cert_timeline')

# Graph metadata
q_cert_graph_origin = np.array([3.0, 0.0, 0.0])  # Origin point in world space (ближе к центру)
q_cert_step_counter = 0  # Step counter for X-axis (time)
q_cert_max_display_steps = MAX_STEPS_PER_EPISODE  # Max steps to display (for X scaling to [0, 2])
q_cert_min_value = 0.0  # Track min Q† value for scaling
q_cert_max_value = -100.0  # Track max Q† value for scaling

print(f"Q-certificate timeline graph initialized at origin {q_cert_graph_origin}")

# ============================================================================
# CRITIC HEATMAP SETUP
# ============================================================================

critic_heatmap = None
grid_overlay = None

if HEATMAP_ENABLED:
    critic_heatmap = CriticHeatmap(
        td3_agent=calf_agent.td3,  # Use CALF's internal TD3
        grid_size=HEATMAP_GRID_SIZE,
        x_range=(-BOUNDARY_LIMIT, BOUNDARY_LIMIT),
        v_range=(-BOUNDARY_LIMIT, BOUNDARY_LIMIT),
        height_scale=HEATMAP_HEIGHT_SCALE,
        update_frequency=HEATMAP_UPDATE_FREQ,
        surface_epsilon=AGENT_HEIGHT_EPSILON
    )
    # Register heatmap in ZoomManager for zoom reactions
    zoom_manager.register_object(critic_heatmap, 'critic_heatmap')

    if GRID_OVERLAY_ENABLED:
        grid_overlay = GridOverlay(
            critic_heatmap=critic_heatmap,
            node_size=GRID_NODE_SIZE,
            line_thickness=GRID_LINE_THICKNESS,
            sample_step=GRID_SAMPLE_STEP
        )
        # Register grid overlay in ZoomManager
        zoom_manager.register_object(grid_overlay, 'grid_overlay')

    print(f"\nCritic Heatmap initialized:")
    print(f"  Grid size: {HEATMAP_GRID_SIZE}x{HEATMAP_GRID_SIZE}")
    print(f"  Update frequency: every {HEATMAP_UPDATE_FREQ} steps")

# ============================================================================
# TRAINING STATE
# ============================================================================

# Main training environment
current_state = env.reset()
episode_reward = 0
episode_length = 0

# Training statistics
training_stats = {
    'episode': 0,
    'total_steps': 0,
    'episode_reward': 0,
    'episode_length': 0,
    'episode_rewards': [],
    'episode_lengths': [],
    'success_count': 0,
    'avg_critic_loss': 0.0,
    'avg_actor_loss': 0.0,
    'buffer_size': 0,
    'exploration_noise': EXPLORATION_NOISE,
    'training_started': False,
    'paused': False
}

# Visualization toggles
heatmap_visible = True  # Heatmap visibility toggle (H key)
grid_visible = True  # Grid overlay visibility toggle (G key)

# Goal indicator (arrow)
goal_arrow = None

# Create UI text
stats_text = Text(
    text='',
    position=(-0.75, 0.45),
    scale=0.8,
    origin=(-0.5, 0.5),
    background=True
)

# Create legend text for trail colors
legend_text = Text(
    text='TRAIL COLORS:\n'
         '<blue>Blue</blue> = TD3 (certified)\n'
         '<green>Green</green> = Relax (uncertified, relaxed)\n'
         '<orange>Orange</orange> = Fallback (nominal policy)\n'
         '\n'
         'AGENTS:\n'
         '<color:rgb(255,128,0)>Orange point</color> = Training agent (CALF mode switching)\n'
         '<blue>Blue points</blue> = Visual agents (pure TD3 policy)',
    position=(0.4, 0.45),
    scale=0.7,
    origin=(-0.5, 0.5),
    background=True
)

print("\n" + "="*70)
print("CONTROLS")
print("="*70)
print(f"  P - Pause/Resume training")
print(f"  Q - Quit")
print(f"  WASD - Move camera")
print(f"  Scroll - Zoom")
print(f"  + / = - Add 1 visual agent")
print(f"  - / _ - Remove 1 visual agent")
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
    global current_state, training_stats, q_cert_step_counter, q_cert_min_value, q_cert_max_value

    if training_stats['paused']:
        return

    # Training step
    if training_stats['total_steps'] < START_TRAINING_STEP:
        # Initial exploration with nominal policy
        action = nominal_policy(current_state)
    else:
        # CALF action selection
        training_stats['training_started'] = True
        action = calf_agent.select_action(current_state, exploration_noise=EXPLORATION_NOISE)

    # Step environment
    next_state, reward, done, info = env.step(action)

    # Scale reward for better learning
    reward = reward * REWARD_SCALE

    # Check early termination
    distance = np.linalg.norm(next_state)
    position = abs(next_state[0])

    if distance < GOAL_EPSILON:
        done = True
    elif position > BOUNDARY_LIMIT:
        done = True
    elif training_stats['episode_length'] >= MAX_STEPS_PER_EPISODE:
        done = True  # Max steps reached

    # Store transition
    replay_buffer.add(current_state, action, next_state, reward, float(done))
    training_stats['buffer_size'] = replay_buffer.size

    # Train
    if training_stats['total_steps'] >= START_TRAINING_STEP:
        train_info = calf_agent.train(replay_buffer, BATCH_SIZE)
        training_stats['avg_critic_loss'] = train_info['critic_loss']
        if train_info['actor_loss'] is not None:
            training_stats['avg_actor_loss'] = train_info['actor_loss']

    # Update state
    current_state = next_state
    training_stats['episode_reward'] += reward
    training_stats['episode_length'] += 1
    training_stats['total_steps'] += 1

    # Update training agent visualization
    if training_stats['training_started']:
        # Get height for training agent
        train_agent_height = get_agent_height(next_state)
        x, v = next_state[0], next_state[1]
        train_position = (x, train_agent_height, v)

        # Update position (this will automatically update trail with mode color)
        # Determine mode from last action source (CALF uses 'td3', 'relax', 'nominal')
        if len(calf_agent.action_sources) > 0:
            mode = calf_agent.action_sources[-1]  # Last action source
        else:
            mode = 'td3'  # Default

        training_agent_visual.update_position(train_position, mode=mode)

        # Update Q-certificate timeline graph
        # X-axis: normalized time (step counter / normalization)
        # Y-axis: normalized Q† value (q_cert / normalization)
        q_cert_step_counter += 1

        if calf_agent.q_cert is not None:
            # Update min/max Q† values for scaling
            q_cert_min_value = min(q_cert_min_value, calf_agent.q_cert)
            q_cert_max_value = max(q_cert_max_value, calf_agent.q_cert)

            # Normalize X to [0, 2] based on max episode steps
            graph_x = 2.0 * (q_cert_step_counter / q_cert_max_display_steps)

            # Normalize Y to [0, 2] based on min/max Q† values
            if abs(q_cert_max_value - q_cert_min_value) > 0.001:
                # Scale to [0, 2] range
                graph_y = 2.0 * (calf_agent.q_cert - q_cert_min_value) / (q_cert_max_value - q_cert_min_value)
            else:
                # If all values are same, put at middle
                graph_y = 1.0

            # Transform to world coordinates: origin + (x_offset, y_offset, 0)
            # Graph X-axis = simulation X-axis (right)
            # Graph Y-axis = simulation Y-axis (upward)
            graph_position = q_cert_graph_origin + np.array([graph_x, graph_y, 0.0])

            # Add point to timeline with current mode color
            # Get current zoom transformations
            a_trans = zoom_manager.a_transformation
            b_trans = zoom_manager.b_translation

            # Map mode for visualization (same as VisualAgent)
            display_mode = mode
            if mode == 'nominal':
                display_mode = 'fallback'

            q_cert_timeline.add_point(graph_position, mode=display_mode,
                                     a_transform=a_trans, b_translate=b_trans)

    # Episode termination
    if done:
        training_stats['episode_rewards'].append(training_stats['episode_reward'])
        training_stats['episode_lengths'].append(training_stats['episode_length'])

        if info['in_goal']:
            training_stats['success_count'] += 1

        # Clear training agent trail on episode end
        training_agent_visual.clear_trail()

        # Clear Q-certificate timeline graph on episode end
        q_cert_timeline.clear()
        q_cert_step_counter = 0
        q_cert_min_value = 0.0
        q_cert_max_value = -100.0

        # Reset CALF certificate for new episode
        if training_stats['training_started']:
            calf_agent.reset_certificate()

        # Reset for new episode
        current_state = env.reset()
        training_stats['episode_reward'] = 0
        training_stats['episode_length'] = 0
        training_stats['episode'] += 1

        # Update training agent to new starting position
        if training_stats['training_started']:
            train_agent_height = get_agent_height(current_state)
            x, v = current_state[0], current_state[1]
            train_position = (x, train_agent_height, v)
            training_agent_visual.update_position(train_position, mode='td3')

        # Evaluation
        if training_stats['episode'] % EVAL_INTERVAL == 0:
            print(f"\nEpisode {training_stats['episode']} / {NUM_EPISODES}")
            print(f"  Avg Reward (last 10): {np.mean(training_stats['episode_rewards'][-10:]):.2f} (scaled x{REWARD_SCALE})")
            print(f"  Success Rate: {training_stats['success_count'] / training_stats['episode'] * 100:.1f}%")

            # CALF statistics
            calf_stats = calf_agent.get_statistics()
            print(f"  CALF Stats:")
            print(f"    P_relax: {calf_stats['P_relax']:.10f}")
            print(f"    Certification rate: {calf_stats['certification_rate']:.3f}")
            print(f"    Intervention rate: {calf_stats['intervention_rate']:.3f}")
            print(f"    Relax rate: {calf_stats['relax_rate']:.3f}")

        # Save checkpoint
        if training_stats['episode'] % (EVAL_INTERVAL * 5) == 0:
            checkpoint_path = Path(__file__).parent / f"checkpoints/calf_episode_{training_stats['episode']}.pth"
            checkpoint_path.parent.mkdir(exist_ok=True)
            calf_agent.save(str(checkpoint_path))
            print(f"  Checkpoint saved: {checkpoint_path}")

    # Update visual agents (continuous flow - reset immediately when done)
    # Batch inference for all agents with mode tracking
    # NOTE: update_state=False - visualization agents don't affect training state

    # Skip visual agent updates if there are no agents
    if len(visual_envs) > 0:
        if training_stats['training_started']:
            # Collect all states
            vis_states = np.array([env.state for env in visual_envs])
            # ВАЖНО: Визуальные агенты используют чистую TD3 политику без CALF сертификации
            # Это правильно, т.к. они нужны только для демонстрации обученной политики
            # Только тренировочный агент использует CALF с сертификацией
            vis_actions = calf_agent.td3.select_action_batch(vis_states, noise=0.0)
            vis_modes = ['td3'] * len(visual_envs)  # Визуальные агенты всегда TD3 (без сертификации)
        else:
            # Random actions during exploration
            vis_actions = np.random.uniform(-env.max_action, env.max_action, size=(len(visual_envs), env.action_dim))
            vis_modes = ['td3'] * len(visual_envs)  # Default mode

        # Step all environments and collect next states
        vis_next_states = []
        vis_done_flags = []

        for i in range(len(visual_envs)):
            vis_env = visual_envs[i]
            vis_action = vis_actions[i]

            # Increment step counter for this visual agent
            visual_step_counters[i] += 1

            # Step visual environment
            vis_next_state, vis_reward, vis_done, vis_info = vis_env.step(vis_action)

            # Check early termination for visual agents
            vis_distance = np.linalg.norm(vis_next_state)
            vis_position = abs(vis_next_state[0])

            if vis_distance < GOAL_EPSILON:
                vis_done = True
            elif vis_position > BOUNDARY_LIMIT:
                vis_done = True
            elif visual_step_counters[i] >= MAX_STEPS_PER_EPISODE:
                vis_done = True  # Max steps reached - reset agent

            vis_next_states.append(vis_next_state)
            vis_done_flags.append(vis_done)

        # Batch compute heights for all agents (1 call instead of 25-50)
        vis_next_states_array = np.array(vis_next_states)
        vis_heights = get_agent_heights_batch(vis_next_states_array)

        # Update positions with batch-computed heights - НОВАЯ АРХИТЕКТУРА
        for i in range(len(visual_envs)):
            vis_next_state = vis_next_states[i]
            vis_done = vis_done_flags[i]
            mode = vis_modes[i]  # Get mode for this agent

            # Update position через VisualAgent (автоматически обновляет траекторию!)
            x, v = vis_next_state[0], vis_next_state[1]
            y = vis_heights[i]  # Y from batch computation
            position = (x, y, v)

            # НОВЫЙ СПОСОБ: один метод обновляет агента И траекторию
            visual_agents[i].update_position(position, mode=mode)

            # Continuous flow: reset immediately when done
            if vis_done:
                # Очистить траекторию (кольцевой буфер просто скрывает точки)
                visual_agents[i].clear_trail()

                visual_envs[i].reset()
                visual_step_counters[i] = 0  # Reset step counter

                # Update position to new starting point
                new_state = visual_envs[i].state
                x, v = new_state[0], new_state[1]
                y = get_agent_height(new_state)  # Single call for reset (rare)
                new_position = (x, y, v)

                # Обновляем позицию агента после сброса
                visual_agents[i].update_position(new_position, mode='td3')

    # Update critic heatmap (only if visible)
    if critic_heatmap is not None and training_stats['training_started'] and heatmap_visible:
        critic_heatmap.update(training_stats['total_steps'])

        # Update grid overlay ONLY when heatmap actually updates (same frequency!) and if visible
        if grid_overlay is not None and grid_visible and training_stats['total_steps'] % HEATMAP_UPDATE_FREQ == 0:
            grid_overlay.update()

        # Update goal arrow position
        global goal_arrow
        if goal_arrow is None:
            goal_arrow = object_manager.create_object(
                name='goal_arrow',
                model='assets/arrow.obj',
                position=(0, 0, 0),
                rotation=(180, 0, 0),
                scale=0.5,
                color_val=Vec4(0.2, 0.8, 0.2, 1)  # Green
            )
            goal_arrow.unlit = True

        if goal_arrow is not None:
            # Position at goal
            goal_y = critic_heatmap.get_q_value_for_state(np.array([0, 0]), use_cached=True)
            goal_arrow.real_position = np.array([0, goal_y + 0.5, 0])

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

    # Get CALF statistics
    calf_stats = calf_agent.get_statistics()

    # Get current Q-value for display
    current_q = 0.0
    q_cert_val = 0.0
    current_mode = 'exploration'
    k_low_val = 0.0
    k_up_val = 0.0
    neg_q_val = 0.0
    state_norm = 0.0
    if training_stats['training_started']:
        # Get Q-value for current state and last action
        state_tensor = torch.FloatTensor(current_state.reshape(1, -1)).to(device)
        action_tensor = torch.FloatTensor(action.reshape(1, -1)).to(device)
        with torch.no_grad():
            q_val, _ = calf_agent.td3.critic(state_tensor, action_tensor)
            current_q = q_val.item()

        # Get certified Q-value
        if calf_agent.q_cert is not None:
            q_cert_val = calf_agent.q_cert

        # Get current mode
        if len(calf_agent.action_sources) > 0:
            current_mode = calf_agent.action_sources[-1]

        # Get K_infinity bounds
        state_norm = np.linalg.norm(current_state)
        k_low_val = calf_agent.kappa_low(state_norm)
        k_up_val = calf_agent.kappa_up(state_norm)
        neg_q_val = -current_q

    # Build performance section
    perf_section = ""
    if heatmap_perf and heatmap_perf['update_count'] > 0:
        perf_section = f'''
=== Heatmap Performance ===
Updates: {heatmap_perf['update_count']}
Avg: {heatmap_perf['avg_time_ms']:.2f}ms ({heatmap_perf['avg_fps']:.1f} FPS)
EMA: {heatmap_perf['ema_time_ms']:.2f}ms ({heatmap_perf['ema_fps']:.1f} FPS)
'''

    stats_text.text = f'''CALF Training Progress

Episode: {training_stats['episode']} / {NUM_EPISODES}
Total Steps: {training_stats['total_steps']}
Visual Agents: {len(visual_agents)} (+/- to adjust)

=== Current Episode ===
Reward: {training_stats['episode_reward']:.2f} (x{REWARD_SCALE})
Length: {training_stats['episode_length']}

=== Overall ===
Avg Reward (10): {avg_reward:.2f} (x{REWARD_SCALE})
Success Rate: {success_rate:.1f}%

=== Training ===
Status: {"TRAINING" if training_stats['training_started'] else "EXPLORATION"}
Noise: {training_stats['exploration_noise']:.3f}
Buffer: {training_stats['buffer_size']} / {replay_buffer.max_size}
Critic Loss: {training_stats['avg_critic_loss']:.4f}
Actor Loss: {training_stats['avg_actor_loss']:.4f}

=== CALF Statistics ===
Current Mode: {current_mode.upper()}
P_relax: {calf_stats['P_relax']:.8f}
Certification: {calf_stats['certification_rate']:.3f}
Intervention: {calf_stats['intervention_rate']:.3f}
Relax: {calf_stats['relax_rate']:.3f}

=== Q-values ===
Q(s,a) current: {current_q:.4f}
Q_cert (Q†): {q_cert_val:.4f}
Delta_Q (Q - Q†): {current_q - q_cert_val:.4f}
Threshold nu_bar: {NU_BAR:.4f}

=== K_infinity Bounds ===
|s| (state norm): {state_norm:.4f}
k_low (0.5*|s|^2): {k_low_val:.4f}
-Q(s,a): {neg_q_val:.4f}
k_up (2.0*|s|^2): {k_up_val:.4f}
Valid: {"YES" if k_low_val <= neg_q_val <= k_up_val else "NO"}

Grid Min: {q_min:.2f}
Grid Max: {q_max:.2f}{perf_section}
Press P to pause'''

    # Update managers
    if hasattr(input_manager, 'update'):
        input_manager.update()
    if hasattr(zoom_manager, 'update'):
        zoom_manager.update()
    if hasattr(object_manager, 'update'):
        object_manager.update()

    # Check if training is complete
    if training_stats['episode'] >= NUM_EPISODES:
        print(f"\n{'='*70}")
        print("TRAINING COMPLETE!")
        print(f"{'='*70}")
        print(f"Total Episodes: {training_stats['episode']}")
        print(f"Success Rate: {success_rate:.1f}%")

        # Final CALF statistics
        print(f"\nFinal CALF Statistics:")
        print(f"  P_relax: {calf_stats['P_relax']:.10f}")
        print(f"  Certification rate: {calf_stats['certification_rate']:.3f}")
        print(f"  Intervention rate: {calf_stats['intervention_rate']:.3f}")
        print(f"  Relax rate: {calf_stats['relax_rate']:.3f}")

        # Save final model
        final_path = Path(__file__).parent / "trained_calf_final.pth"
        calf_agent.save(str(final_path))
        print(f"\nFinal model saved: {final_path}")


        application.quit()

def input(key):
    """Handle input"""
    global visual_agents, visual_envs, visual_step_counters, heatmap_visible, grid_visible

    if key == 'p':
        training_stats['paused'] = not training_stats['paused']
        print(f"\nTraining {'PAUSED' if training_stats['paused'] else 'RESUMED'}")

    # Toggle heatmap visibility (H key)
    elif key == 'h':
        heatmap_visible = not heatmap_visible
        if critic_heatmap is not None and hasattr(critic_heatmap, 'surface_entity') and critic_heatmap.surface_entity:
            critic_heatmap.surface_entity.visible = heatmap_visible
        print(f"\nHeatmap {'VISIBLE' if heatmap_visible else 'HIDDEN'}")

    # Toggle grid overlay visibility (G key)
    elif key == 'g':
        grid_visible = not grid_visible
        if grid_overlay is not None:
            # Toggle visibility of all node and line entities
            for node in grid_overlay.node_entities:
                if node:
                    node.visible = grid_visible
            for line in grid_overlay.line_entities:
                if line:
                    line.visible = grid_visible
        print(f"\nGrid overlay {'VISIBLE' if grid_visible else 'HIDDEN'}")

    # Increase number of visual agents (+ or =)
    elif key == '+' or key == '=':
        num_to_add = 1
        print(f"\n[Agents] Adding {num_to_add} visual agent...")

        for _ in range(num_to_add):
            # Create new environment
            new_env = PointMassEnv(dt=0.01, max_action=5.0, goal_radius=0.1)
            new_env.reset()
            visual_envs.append(new_env)

            # Create new visual agent
            agent_id = len(visual_agents)
            new_agent = VisualAgent(
                object_manager=object_manager,
                zoom_manager=zoom_manager,
                agent_id=agent_id,
                max_trail_length=TRAIL_MAX_LENGTH,
                decimation=TRAIL_DECIMATION
            )
            visual_agents.append(new_agent)

            # Add step counter
            visual_step_counters.append(0)

        print(f"[Agents] Total visual agents: {len(visual_agents)}, envs: {len(visual_envs)}, counters: {len(visual_step_counters)}")

    # Decrease number of visual agents (- or _)
    elif key == '-' or key == '_':
        num_to_remove = min(1, len(visual_agents))  # Don't go below 0

        if num_to_remove == 0:
            print(f"\n[Agents] No agents to remove!")
        else:
            print(f"\n[Agents] Removing {num_to_remove} visual agent...")

            # Remove last N agents
            for _ in range(num_to_remove):
                # Get agent index before removing
                agent_idx = len(visual_agents) - 1

                # Remove visual agent (destroy trail and point)
                agent = visual_agents.pop()

                # Unregister from ZoomManager BEFORE destroying
                zoom_manager.unregister_object(f'trail_{agent_idx}')

                # Now destroy entities
                agent.trail.clear()
                if hasattr(agent.trail, 'entity') and agent.trail.entity:
                    destroy(agent.trail.entity)
                if agent.visual_point:
                    destroy(agent.visual_point)

                # Remove environment and counter
                visual_envs.pop()
                visual_step_counters.pop()

            print(f"[Agents] Total visual agents: {len(visual_agents)}, envs: {len(visual_envs)}, counters: {len(visual_step_counters)}")

    else:
        input_manager.handle_input(key)

app.run()
