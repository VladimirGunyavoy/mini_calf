"""
Evaluate trained CALF agent with Ursina trajectory visualization
Loads pre-trained model and visualizes with P_relax = 0 (no relaxation)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from ursina import *

# Import RL components
from RL.calf import CALFController
from RL.simple_env import PointMassEnv, pd_nominal_policy

# Import Ursina components
from core import Player, setup_scene
from managers import (
    InputManager, WindowManager, ZoomManager,
    ObjectManager, ColorManager, UIManager
)
from visuals import LineTrail, CriticHeatmap, GridOverlay

# ============================================================================
# EVALUATION PARAMETERS
# ============================================================================

# Model to load
CHECKPOINT_PATH = Path(__file__).parent / "trained_calf_final.pth"

# Evaluation parameters
NUM_EPISODES = 100
MAX_STEPS_PER_EPISODE = 750
EXPLORATION_NOISE = 0.0  # No exploration during evaluation
SEED = 42

# CALF parameters (matching training)
NU_BAR = 0.01
KAPPA_LOW_COEF = 0.01
KAPPA_UP_COEF = 1000.0

# Visualization parameters
N_AGENTS_VISUAL = 5
TRAIL_MAX_LENGTH = 600
TRAIL_DECIMATION = 1
TRAIL_REBUILD_FREQ = 15

# Critic heatmap parameters
HEATMAP_ENABLED = True
HEATMAP_GRID_SIZE = 21
HEATMAP_UPDATE_FREQ = 500
HEATMAP_HEIGHT_SCALE = 2.0
AGENT_HEIGHT_EPSILON = 0.15

# Grid overlay parameters
GRID_OVERLAY_ENABLED = True
GRID_NODE_SIZE = 0.04
GRID_LINE_THICKNESS = 3
GRID_SAMPLE_STEP = 1

# Episode termination parameters
GOAL_EPSILON = 0.05
BOUNDARY_LIMIT = 5.0

# Set random seeds
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================================
# SETUP URSINA BEFORE APP
# ============================================================================
WindowManager.setup_before_app(monitor="main")
app = Ursina()

# ============================================================================
# INITIALIZE COMPONENTS
# ============================================================================
player = Player()
color_manager = ColorManager()
window_manager = WindowManager(color_manager=color_manager, monitor="main")
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

# Enable FPS counter
window.fps_counter.enabled = True
window.fps_counter.position = (0.75, 0.48)
window.fps_counter.color = color.white
window.fps_counter.scale = 1.0

print("\n" + "="*70)
print("CALF EVALUATION WITH VISUALIZATION")
print("Trained Model Evaluation - P_relax = 0 (No Relaxation)")
print("="*70)

# ============================================================================
# LOAD TRAINED MODEL
# ============================================================================

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")

# Create environment
env = PointMassEnv(dt=0.01, max_action=5.0, goal_radius=0.1)
print(f"\nEnvironment: PointMassEnv")
print(f"State dim: {env.state_dim}, Action dim: {env.action_dim}")

# Nominal safe policy
nominal_policy = pd_nominal_policy(max_action=env.max_action, kp=2.0, kd=0.4)

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
    lambda_relax=1.0,  # No relax decay during evaluation
    hidden_dim=64,
    lr=3e-4,
    device=device,
    discount=0.99,
    tau=0.005,
    policy_noise=0.2,
    noise_clip=0.5,
    policy_freq=2
)

# Load trained model
if CHECKPOINT_PATH.exists():
    calf_agent.load(str(CHECKPOINT_PATH))
    print(f"\nLoaded model from: {CHECKPOINT_PATH}")

    # Set P_relax to 0 (no relaxation)
    calf_agent.P_relax = 0.0
    print(f"Set P_relax = 0.0 (no relaxation)")

    # Print loaded statistics
    stats = calf_agent.get_statistics()
    print(f"\nLoaded model statistics:")
    print(f"  Total steps: {stats['total_steps']}")
    print(f"  Nominal interventions: {stats['nominal_interventions']}")
    print(f"  Relax events: {stats['relax_events']}")
else:
    print(f"\nERROR: Checkpoint not found at {CHECKPOINT_PATH}")
    print("Please train the model first using train_calf_visual.py")
    application.quit()
    sys.exit(1)

# ============================================================================
# VISUAL AGENTS SETUP
# ============================================================================

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
        self.visual_point.real_position = self.real_position
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

        # Добавляем точку в траекторию
        self.trail.add_point(self.real_position, mode=display_mode,
                           a_transform=a_trans, b_translate=b_trans)

    def clear_trail(self):
        """Очистить траекторию"""
        self.trail.clear()

# Create visual environments
visual_envs = [PointMassEnv(dt=0.01, max_action=5.0, goal_radius=0.1)
               for _ in range(N_AGENTS_VISUAL)]

# Reset all visual environments
for ve in visual_envs:
    ve.reset()

# Step counters for visual agents
visual_step_counters = [0] * N_AGENTS_VISUAL

# Создаем визуальных агентов
visual_agents = []
for i in range(N_AGENTS_VISUAL):
    agent = VisualAgent(
        object_manager=object_manager,
        zoom_manager=zoom_manager,
        agent_id=i,
        max_trail_length=200,
        decimation=1
    )
    visual_agents.append(agent)

print(f"\n{N_AGENTS_VISUAL} visual agents initialized")

# ============================================================================
# TRAINING AGENT VISUALIZATION
# ============================================================================

# Create visual agent for evaluation
training_agent_visual = VisualAgent(
    object_manager=object_manager,
    zoom_manager=zoom_manager,
    agent_id=9999,
    max_trail_length=TRAIL_MAX_LENGTH,
    decimation=TRAIL_DECIMATION,
    point_color=Vec4(1.0, 0.5, 0.0, 1)  # Orange color
)

print(f"Evaluation agent visualization initialized (orange)")

# ============================================================================
# Q-CERTIFICATE TIMELINE GRAPH
# ============================================================================

q_cert_timeline = LineTrail(
    max_points=TRAIL_MAX_LENGTH,
    line_thickness=3,
    decimation=TRAIL_DECIMATION,
    rebuild_freq=TRAIL_REBUILD_FREQ
)
zoom_manager.register_object(q_cert_timeline, 'q_cert_timeline')

q_cert_graph_origin = np.array([6.0, 0.0, 0.0])
q_cert_step_counter = 0
q_cert_normalization = 2.0

print(f"Q-certificate timeline graph initialized")

# ============================================================================
# CRITIC HEATMAP SETUP
# ============================================================================

critic_heatmap = None
grid_overlay = None

if HEATMAP_ENABLED:
    critic_heatmap = CriticHeatmap(
        td3_agent=calf_agent.td3,
        grid_size=HEATMAP_GRID_SIZE,
        x_range=(-BOUNDARY_LIMIT, BOUNDARY_LIMIT),
        v_range=(-BOUNDARY_LIMIT, BOUNDARY_LIMIT),
        height_scale=HEATMAP_HEIGHT_SCALE,
        update_frequency=HEATMAP_UPDATE_FREQ,
        surface_epsilon=AGENT_HEIGHT_EPSILON
    )
    zoom_manager.register_object(critic_heatmap, 'critic_heatmap')

    if GRID_OVERLAY_ENABLED:
        grid_overlay = GridOverlay(
            critic_heatmap=critic_heatmap,
            node_size=GRID_NODE_SIZE,
            line_thickness=GRID_LINE_THICKNESS,
            sample_step=GRID_SAMPLE_STEP
        )
        zoom_manager.register_object(grid_overlay, 'grid_overlay')

    print(f"\nCritic Heatmap initialized")

# ============================================================================
# EVALUATION STATE
# ============================================================================

current_state = env.reset()
episode_reward = 0
episode_length = 0

# Evaluation statistics
eval_stats = {
    'episode': 0,
    'total_steps': 0,
    'episode_reward': 0,
    'episode_length': 0,
    'episode_rewards': [],
    'episode_lengths': [],
    'success_count': 0,
    'paused': False
}

# Visualization toggles
heatmap_visible = True
grid_visible = True

# Goal indicator
goal_arrow = None

# Create UI text
stats_text = Text(
    text='',
    position=(-0.75, 0.45),
    scale=0.8,
    origin=(-0.5, 0.5),
    background=True
)

# Create legend text
legend_text = Text(
    text='TRAIL COLORS:\n'
         '<blue>Blue</blue> = TD3 (certified)\n'
         '<orange>Orange</orange> = Fallback (nominal policy)\n'
         '\n'
         'MODE: EVALUATION (P_relax = 0)\n'
         '<color:rgb(255,128,0)>Orange point</color> = Evaluation agent',
    position=(0.4, 0.45),
    scale=0.7,
    origin=(-0.5, 0.5),
    background=True
)

print("\n" + "="*70)
print("CONTROLS")
print("="*70)
print(f"  P - Pause/Resume")
print(f"  Q - Quit")
print(f"  WASD - Move camera")
print(f"  Scroll - Zoom")
print()

def get_agent_height(state):
    """Get Y coordinate for agent based on critic Q-value"""
    if critic_heatmap is not None:
        q_height = critic_heatmap.get_q_value_for_state(state, use_cached=True)
        return q_height + 2 * AGENT_HEIGHT_EPSILON
    else:
        return 0.1

def get_agent_heights_batch(states):
    """Get Y coordinates for a batch of agents (optimized)"""
    if critic_heatmap is not None:
        q_heights = critic_heatmap.get_q_value_for_states_batch(states, use_cached=True)
        return q_heights + 2 * AGENT_HEIGHT_EPSILON
    else:
        return np.full(len(states), 0.1)

def update():
    """Main evaluation loop - called every frame"""
    global current_state, eval_stats, q_cert_step_counter

    if eval_stats['paused']:
        return

    # CALF action selection (no exploration noise, P_relax = 0)
    action = calf_agent.select_action(current_state, exploration_noise=EXPLORATION_NOISE)

    # Step environment
    next_state, reward, done, info = env.step(action)

    # Check early termination
    distance = np.linalg.norm(next_state)
    position = abs(next_state[0])

    if distance < GOAL_EPSILON:
        done = True
    elif position > BOUNDARY_LIMIT:
        done = True
    elif eval_stats['episode_length'] >= MAX_STEPS_PER_EPISODE:
        done = True

    # Update state
    current_state = next_state
    eval_stats['episode_reward'] += reward
    eval_stats['episode_length'] += 1
    eval_stats['total_steps'] += 1

    # Update evaluation agent visualization
    train_agent_height = get_agent_height(next_state)
    x, v = next_state[0], next_state[1]
    train_position = (x, train_agent_height, v)

    # Determine mode
    if len(calf_agent.action_sources) > 0:
        mode = calf_agent.action_sources[-1]
    else:
        mode = 'td3'

    training_agent_visual.update_position(train_position, mode=mode)

    # Update Q-certificate timeline graph
    q_cert_step_counter += 1

    if calf_agent.q_cert is not None:
        graph_x = q_cert_step_counter / q_cert_normalization
        graph_y = calf_agent.q_cert / q_cert_normalization
        graph_position = q_cert_graph_origin + np.array([graph_x, graph_y, 0.0])

        a_trans = zoom_manager.a_transformation
        b_trans = zoom_manager.b_translation

        q_cert_timeline.add_point(graph_position, mode=mode,
                                 a_transform=a_trans, b_translate=b_trans)

    # Episode termination
    if done:
        eval_stats['episode_rewards'].append(eval_stats['episode_reward'])
        eval_stats['episode_lengths'].append(eval_stats['episode_length'])

        if info['in_goal']:
            eval_stats['success_count'] += 1

        # Clear trails
        training_agent_visual.clear_trail()
        q_cert_timeline.clear()
        q_cert_step_counter = 0

        # Reset certificate
        calf_agent.reset_certificate()

        # Reset for new episode
        current_state = env.reset()
        eval_stats['episode_reward'] = 0
        eval_stats['episode_length'] = 0
        eval_stats['episode'] += 1

        # Update position
        train_agent_height = get_agent_height(current_state)
        x, v = current_state[0], current_state[1]
        train_position = (x, train_agent_height, v)
        training_agent_visual.update_position(train_position, mode='td3')

        # Print progress
        if eval_stats['episode'] % 10 == 0:
            print(f"\nEpisode {eval_stats['episode']} / {NUM_EPISODES}")
            print(f"  Avg Reward (last 10): {np.mean(eval_stats['episode_rewards'][-10:]):.2f}")
            print(f"  Success Rate: {eval_stats['success_count'] / eval_stats['episode'] * 100:.1f}%")

            calf_stats = calf_agent.get_statistics()
            print(f"  CALF Stats:")
            print(f"    Certification rate: {calf_stats['certification_rate']:.3f}")
            print(f"    Intervention rate: {calf_stats['intervention_rate']:.3f}")

    # Update visual agents
    if len(visual_envs) > 0:
        vis_states = np.array([env.state for env in visual_envs])
        vis_actions, vis_modes = calf_agent.select_action_batch(
            vis_states, exploration_noise=0.0, return_modes=True, update_state=False
        )

        vis_next_states = []
        vis_done_flags = []

        for i in range(len(visual_envs)):
            vis_env = visual_envs[i]
            vis_action = vis_actions[i]
            visual_step_counters[i] += 1

            vis_next_state, vis_reward, vis_done, vis_info = vis_env.step(vis_action)

            vis_distance = np.linalg.norm(vis_next_state)
            vis_position = abs(vis_next_state[0])

            if vis_distance < GOAL_EPSILON:
                vis_done = True
            elif vis_position > BOUNDARY_LIMIT:
                vis_done = True
            elif visual_step_counters[i] >= MAX_STEPS_PER_EPISODE:
                vis_done = True

            vis_next_states.append(vis_next_state)
            vis_done_flags.append(vis_done)

        vis_next_states_array = np.array(vis_next_states)
        vis_heights = get_agent_heights_batch(vis_next_states_array)

        for i in range(len(visual_envs)):
            vis_next_state = vis_next_states[i]
            vis_done = vis_done_flags[i]
            mode = vis_modes[i]

            x, v = vis_next_state[0], vis_next_state[1]
            y = vis_heights[i]
            position = (x, y, v)

            visual_agents[i].update_position(position, mode=mode)

            if vis_done:
                visual_agents[i].clear_trail()
                visual_envs[i].reset()
                visual_step_counters[i] = 0

                new_state = visual_envs[i].state
                x, v = new_state[0], new_state[1]
                y = get_agent_height(new_state)
                new_position = (x, y, v)
                visual_agents[i].update_position(new_position, mode='td3')

    # Update critic heatmap
    if critic_heatmap is not None and heatmap_visible:
        critic_heatmap.update(eval_stats['total_steps'])

        if grid_overlay is not None and grid_visible and eval_stats['total_steps'] % HEATMAP_UPDATE_FREQ == 0:
            grid_overlay.update()

        # Update goal arrow
        global goal_arrow
        if goal_arrow is None:
            goal_arrow = object_manager.create_object(
                name='goal_arrow',
                model='assets/arrow.obj',
                position=(0, 0, 0),
                rotation=(180, 0, 0),
                scale=0.5,
                color_val=Vec4(0.2, 0.8, 0.2, 1)
            )
            goal_arrow.unlit = True

        if goal_arrow is not None:
            goal_y = critic_heatmap.get_q_value_for_state(np.array([0, 0]), use_cached=True)
            goal_arrow.real_position = np.array([0, goal_y + 0.5, 0])
            goal_arrow.apply_transform(zoom_manager.a_transformation, zoom_manager.b_translation)

    # Update statistics display
    avg_reward = np.mean(eval_stats['episode_rewards'][-10:]) if eval_stats['episode_rewards'] else 0
    success_rate = eval_stats['success_count'] / max(1, eval_stats['episode']) * 100

    q_min, q_max = (0, 0)
    heatmap_perf = None
    if critic_heatmap is not None:
        q_min, q_max = critic_heatmap.get_q_range()
        heatmap_perf = critic_heatmap.get_performance_stats()

    calf_stats = calf_agent.get_statistics()

    # Get current Q-value for display
    current_q = 0.0
    q_cert_val = 0.0
    current_mode = 'evaluation'
    k_low_val = 0.0
    k_up_val = 0.0
    neg_q_val = 0.0
    state_norm = 0.0

    state_tensor = torch.FloatTensor(current_state.reshape(1, -1)).to(device)
    action_tensor = torch.FloatTensor(action.reshape(1, -1)).to(device)
    with torch.no_grad():
        q_val, _ = calf_agent.td3.critic(state_tensor, action_tensor)
        current_q = q_val.item()

    if calf_agent.q_cert is not None:
        q_cert_val = calf_agent.q_cert

    if len(calf_agent.action_sources) > 0:
        current_mode = calf_agent.action_sources[-1]

    state_norm = np.linalg.norm(current_state)
    k_low_val = calf_agent.kappa_low(state_norm)
    k_up_val = calf_agent.kappa_up(state_norm)
    neg_q_val = -current_q

    perf_section = ""
    if heatmap_perf and heatmap_perf['update_count'] > 0:
        perf_section = f'''
=== Heatmap Performance ===
Updates: {heatmap_perf['update_count']}
Avg: {heatmap_perf['avg_time_ms']:.2f}ms ({heatmap_perf['avg_fps']:.1f} FPS)
EMA: {heatmap_perf['ema_time_ms']:.2f}ms ({heatmap_perf['ema_fps']:.1f} FPS)
'''

    stats_text.text = f'''CALF Evaluation Progress

Episode: {eval_stats['episode']} / {NUM_EPISODES}
Total Steps: {eval_stats['total_steps']}
Visual Agents: {len(visual_agents)}

=== Current Episode ===
Reward: {eval_stats['episode_reward']:.2f}
Length: {eval_stats['episode_length']}

=== Overall ===
Avg Reward (10): {avg_reward:.2f}
Success Rate: {success_rate:.1f}%

=== CALF Statistics (P_relax = 0) ===
Current Mode: {current_mode.upper()}
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
k_low (0.01*|s|^2): {k_low_val:.4f}
-Q(s,a): {neg_q_val:.4f}
k_up (1000*|s|^2): {k_up_val:.4f}
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

    # Check if evaluation is complete
    if eval_stats['episode'] >= NUM_EPISODES:
        print(f"\n{'='*70}")
        print("EVALUATION COMPLETE!")
        print(f"{'='*70}")
        print(f"Total Episodes: {eval_stats['episode']}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"\nFinal CALF Statistics:")
        print(f"  Certification rate: {calf_stats['certification_rate']:.3f}")
        print(f"  Intervention rate: {calf_stats['intervention_rate']:.3f}")
        print(f"  Relax rate: {calf_stats['relax_rate']:.3f}")

        application.quit()

def input(key):
    """Handle input"""
    global visual_agents, visual_envs, visual_step_counters, heatmap_visible, grid_visible

    if key == 'p':
        eval_stats['paused'] = not eval_stats['paused']
        print(f"\nEvaluation {'PAUSED' if eval_stats['paused'] else 'RESUMED'}")

    elif key == 'h':
        heatmap_visible = not heatmap_visible
        if critic_heatmap is not None and hasattr(critic_heatmap, 'surface_entity') and critic_heatmap.surface_entity:
            critic_heatmap.surface_entity.visible = heatmap_visible
        print(f"\nHeatmap {'VISIBLE' if heatmap_visible else 'HIDDEN'}")

    elif key == 'g':
        grid_visible = not grid_visible
        if grid_overlay is not None:
            for node in grid_overlay.node_entities:
                if node:
                    node.visible = grid_visible
            for line in grid_overlay.line_entities:
                if line:
                    line.visible = grid_visible
        print(f"\nGrid overlay {'VISIBLE' if grid_visible else 'HIDDEN'}")

    else:
        input_manager.handle_input(key)

app.run()
