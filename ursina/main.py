"""
CALF Training - Clean Architecture (Stage 3)
Main entry point with modular component separation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

# Import configuration
from config import AppConfig

# Import core components
from core import CALFApplication

# Import training components
from training import CALFTrainer, TrainingVisualizer

# Import RL components
from RL.calf import CALFController
from RL.td3 import ReplayBuffer
from RL.simple_env import PointMassEnv, pd_nominal_policy

# Import visuals
from visuals import LineTrail, CriticHeatmap, GridOverlay

# Import profiler
from utils import PerformanceProfiler

# ============================================================================
# CONFIGURATION
# ============================================================================

# Load configuration (can easily switch presets here)
config = AppConfig.from_preset('medium')

# Override specific parameters if needed
# config.training.num_episodes = 1000
# config.visualization.n_agents = 10

# Resume training from checkpoint
RESUME_TRAINING = True
RESUME_CHECKPOINT = "trained_calf_final.pth"

# Set random seeds
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Create profiler
profiler = PerformanceProfiler(ema_alpha=0.1)
print("\nPerformance profiler enabled")

# Frame counter for profiler updates
frame_counter = 0

# ============================================================================
# APPLICATION SETUP
# ============================================================================

# Create and setup application
app = CALFApplication(config).setup()

# Create training environment
env = app.create_training_env()

# Create nominal safe policy (PD controller)
nominal_policy = pd_nominal_policy(max_action=env.max_action, kp=2.0, kd=0.4)
print(f"\nNominal Policy: PD Controller (kp=2.0, kd=0.4)")

# ============================================================================
# CALF AGENT SETUP
# ============================================================================

# Create CALF controller
calf_agent = CALFController(
    state_dim=env.state_dim,
    action_dim=env.action_dim,
    max_action=env.max_action,
    nominal_policy=nominal_policy,
    goal_region_radius=env.goal_radius,
    nu_bar=config.training.nu_bar,
    kappa_low_coef=config.training.kappa_low_coef,
    kappa_up_coef=config.training.kappa_up_coef,
    lambda_relax=config.training.lambda_relax,
    hidden_dim=64,
    lr=3e-4,
    device=app.device,
    discount=0.99,
    tau=0.005,
    policy_noise=0.2,
    noise_clip=0.5,
    policy_freq=2
)

print(f"\nCALF Parameters:")
print(f"  Lambda_relax: {config.training.lambda_relax}")
print(f"  Nu_bar (Lyapunov threshold): {config.training.nu_bar}")
print(f"  Kappa coefficients: [{config.training.kappa_low_coef}, {config.training.kappa_up_coef}]")
print(f"  Reward scale: {config.training.reward_scale}x")

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
# TRAINER SETUP
# ============================================================================

trainer = CALFTrainer(
    calf_agent=calf_agent,
    env=env,
    replay_buffer=replay_buffer,
    nominal_policy=nominal_policy,
    config=config.training
)

# ============================================================================
# VISUALIZER SETUP
# ============================================================================

visualizer = TrainingVisualizer(
    object_manager=app.object_manager,
    zoom_manager=app.zoom_manager,
    config=config.visualization,
    device=app.device
)

# Setup visual agents
visual_envs = app.create_visual_envs(config.visualization.n_agents)
visualizer.setup_visual_agents(visual_envs)

# Setup training agent visualization
visualizer.setup_training_agent()

# Setup Q-certificate timeline
q_cert_timeline = LineTrail(
    max_points=config.visualization.trail_max_length,
    line_thickness=3,
    decimation=config.visualization.trail_decimation,
    rebuild_freq=config.visualization.trail_rebuild_freq
)
app.zoom_manager.register_object(q_cert_timeline, 'q_cert_timeline')

q_cert_graph_origin = np.array([3.0, 0.0, 0.0])
q_cert_max_display_steps = config.training.max_steps_per_episode

visualizer.setup_q_cert_timeline(
    q_cert_timeline=q_cert_timeline,
    graph_origin=q_cert_graph_origin,
    max_display_steps=q_cert_max_display_steps
)

print(f"Q-certificate timeline graph initialized at origin {q_cert_graph_origin}")

# Setup critic heatmap
critic_heatmap = None
grid_overlay = None

if config.visualization.heatmap_enabled:
    critic_heatmap = CriticHeatmap(
        td3_agent=calf_agent.td3,
        grid_size=config.visualization.heatmap_grid_size,
        x_range=(-config.training.boundary_limit, config.training.boundary_limit),
        v_range=(-config.training.boundary_limit, config.training.boundary_limit),
        height_scale=config.visualization.heatmap_height_scale,
        update_frequency=config.visualization.heatmap_update_freq,
        surface_epsilon=config.visualization.agent_height_epsilon
    )
    app.zoom_manager.register_object(critic_heatmap, 'critic_heatmap')

    if config.visualization.grid_enabled:
        grid_overlay = GridOverlay(
            critic_heatmap=critic_heatmap,
            node_size=config.visualization.grid_node_size,
            line_thickness=config.visualization.grid_line_thickness,
            sample_step=config.visualization.grid_sample_step
        )
        app.zoom_manager.register_object(grid_overlay, 'grid_overlay')

    visualizer.setup_heatmap(critic_heatmap, grid_overlay)

    print(f"\nCritic Heatmap initialized:")
    print(f"  Grid size: {config.visualization.heatmap_grid_size}x{config.visualization.heatmap_grid_size}")
    print(f"  Update frequency: every {config.visualization.heatmap_update_freq} steps")

# Setup UI
visualizer.setup_ui(config.training)

# Setup profiler UI (bottom-right corner)
from ursina import Text
profiler_text = Text(
    text='Profiler initializing...',
    position=(0.6, -0.35),
    origin=(0, 0),
    scale=0.6,
    background=True,
    use_tags=False  # Disable tag parsing to avoid color tag conflicts
)

# Print controls
print("\n" + "="*70)
print("CONTROLS")
print("="*70)
print(f"  P - Pause/Resume training")
print(f"  Q - Quit")
print(f"  WASD - Move camera")
print(f"  Scroll - Zoom")
print(f"  + / = - Add 1 visual agent")
print(f"  - / _ - Remove 1 visual agent")
print(f"  H - Toggle heatmap visibility")
print(f"  G - Toggle grid visibility")
print(f"  F1 - Export profiler data to CSV")
print(f"  F2 - Toggle profiler on/off")
print(f"  F3 - Reset profiler statistics")
print()

# ============================================================================
# MAIN LOOP
# ============================================================================

def update():
    """Main training loop - called every frame"""
    global frame_counter

    with profiler.measure('total_frame'):
        if trainer.paused:
            return

        # Training step
        with profiler.measure('training_step'):
            next_state, done, info = trainer.train_step()

        # Update training agent visualization
        if trainer.training_started:
            with profiler.measure('training_agent_viz'):
                visualizer.update_training_agent(next_state, calf_agent)
                visualizer.update_q_cert_timeline(calf_agent)

        # Episode termination
        if done:
            with profiler.measure('episode_end'):
                # Handle episode end
                new_state = trainer.handle_episode_end(info)

                # Clear visualizations
                visualizer.clear_training_agent_trail()
                visualizer.clear_q_cert_timeline()

                # Update training agent to new starting position
                if trainer.training_started:
                    visualizer.update_training_agent(new_state, calf_agent)

        # Update visual agents (batch processing)
        with profiler.measure('visual_agents'):
            visualizer.update_visual_agents(
                policy=calf_agent.td3,
                training_started=trainer.training_started,
                max_steps_per_episode=config.training.max_steps_per_episode,
                goal_epsilon=config.training.goal_epsilon,
                boundary_limit=config.training.boundary_limit
            )

        # Update critic heatmap
        if trainer.training_started:
            with profiler.measure('heatmap'):
                visualizer.update_heatmap(trainer.total_steps)

        # Update statistics display
        with profiler.measure('stats_display'):
            visualizer.update_stats_display(
                trainer=trainer,
                calf_agent=calf_agent,
                current_state=trainer.current_state,
                action=np.zeros(env.action_dim),  # Last action not stored separately
                config=config.training
            )

        # Update managers
        with profiler.measure('managers'):
            app.update_managers()

        # Update profiler display every 60 frames (~1 second at 60 FPS)
        frame_counter += 1
        if frame_counter % 60 == 0:
            with profiler.measure('profiler_update'):
                profiler_text.text = profiler.get_report(top_n=5, sort_by='ema')

        # Check if training is complete
        if trainer.is_complete():
            trainer.finalize()
            app.quit()

# ============================================================================
# INPUT HANDLER
# ============================================================================

def input(key):
    """Handle input"""
    # First, delegate to input_manager for base functionality (zoom, camera, etc.)
    app.input_manager.handle_input(key)

    # Then handle training-specific keys
    if key == 'p':
        trainer.paused = not trainer.paused
        print(f"\nTraining {'PAUSED' if trainer.paused else 'RESUMED'}")

    elif key == 'h':
        visualizer.toggle_heatmap_visibility()

    elif key == 'g':
        visualizer.toggle_grid_visibility()

    elif key in ['+', '=']:
        visualizer.add_visual_agent()

    elif key in ['-', '_']:
        visualizer.remove_visual_agent()

    elif key == 'f1':
        # Export profiler data to CSV
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"profiler_export_{timestamp}.csv"
        profiler.export_to_csv(filename)
        print(f"\nProfiler data exported to: {filename}")

    elif key == 'f2':
        # Toggle profiler on/off
        if profiler._enabled:
            profiler.disable()
            profiler_text.text = "Profiler: DISABLED"
            print("\nProfiler disabled")
        else:
            profiler.enable()
            print("\nProfiler enabled")

    elif key == 'f3':
        # Reset profiler statistics
        profiler.reset()
        profiler_text.text = "Profiler: RESET"
        print("\nProfiler statistics reset")

# ============================================================================
# START APPLICATION
# ============================================================================

app.run()
