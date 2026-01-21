# CALF Architecture Documentation (Stage 3)

**Last Updated**: 2026-01-21
**Status**: Clean Modular Architecture (Stage 3 Complete)

---

## 📋 Overview

The CALF project has been refactored into a clean, modular architecture with clear separation of concerns:

- **CALFApplication**: Application setup and lifecycle
- **CALFTrainer**: Training logic and state management
- **TrainingVisualizer**: Visualization components

This replaces the original monolithic 974-line `train_calf_visual.py` with a 317-line `main.py` and reusable components.

---

## 🏗️ Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                         main.py                              │
│                    (Entry Point - 317 lines)                 │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ AppConfig    │  │ CALF Agent   │  │ Replay Buffer    │  │
│  │ (Stage 2)    │  │ (RL/calf.py) │  │ (RL/td3.py)      │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────────┘  │
│         │                  │                  │              │
│         ▼                  ▼                  ▼              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         CALFApplication (core/application.py)        │   │
│  │  • Ursina setup         • Scene initialization      │   │
│  │  • Manager creation     • Device configuration      │   │
│  │  • Environment factory  • Lifecycle management      │   │
│  └────────────────┬────────────────────────────────────┘   │
│                   │                                          │
│         ┌─────────┴─────────┐                               │
│         ▼                   ▼                                │
│  ┌──────────────┐    ┌─────────────────┐                   │
│  │ CALFTrainer  │    │ TrainingViz     │                   │
│  │              │    │                 │                   │
│  │ • train_step │    │ • update_agents │                   │
│  │ • episode_end│    │ • update_heatmap│                   │
│  │ • statistics │    │ • update_ui     │                   │
│  └──────────────┘    └─────────────────┘                   │
│                                                               │
└───────────────────────────────────────────────────────────── ┘
         │                              │
         ▼                              ▼
  ┌──────────────┐            ┌─────────────────┐
  │   Managers   │            │    Visuals      │
  │  • Window    │            │  • LineTrail    │
  │  • Zoom      │            │  • Heatmap      │
  │  • Object    │            │  • GridOverlay  │
  │  • Input     │            │  • VisualAgent  │
  │  • UI        │            └─────────────────┘
  └──────────────┘
```

---

## 📁 File Structure

```
ursina/
├── main.py                       # NEW Main entry (317 lines) ⭐
├── old_main.py                   # Legacy main (reference)
├── train_calf_visual.py          # Original implementation (974 lines)
│
├── config/                       # Configuration System (Stage 2) ⭐
│   ├── __init__.py
│   ├── training_config.py        # Training parameters
│   ├── visualization_config.py   # Visualization parameters
│   └── app_config.py             # Combined config + presets
│
├── core/                         # Core Components
│   ├── __init__.py
│   ├── application.py            # NEW CALFApplication ⭐
│   ├── player.py                 # Camera controller
│   ├── scene_setup.py            # Scene initialization
│   ├── frame.py
│   └── state_buffer.py
│
├── training/                     # NEW Training Module (Stage 3) ⭐
│   ├── __init__.py
│   ├── trainer.py                # CALFTrainer - training logic
│   └── visualizer.py             # TrainingVisualizer + VisualAgent
│
├── managers/                     # System Managers
│   ├── input_manager.py
│   ├── zoom_manager.py
│   ├── object_manager.py
│   ├── ui_manager.py
│   ├── window_manager.py
│   └── color_manager.py
│
├── visuals/                      # Visualization Components
│   ├── line_trail.py             # Trajectory trails (ring buffer)
│   ├── critic_heatmap.py         # Q-value heatmap
│   ├── grid_overlay.py           # Grid visualization
│   ├── point_trail.py
│   └── multi_color_trail.py
│
├── physics/                      # Physics & Policies
│   ├── simulation_engine.py
│   └── policies/
│       ├── td3_policy.py
│       ├── calf_policy.py
│       └── ...
│
└── llm/                          # Documentation
    └── stage_2/
        ├── 00_START_HERE.md
        ├── 01_OPTIMIZATION_PLAN.md
        ├── STAGE_3_SUMMARY.md
        └── ...
```

⭐ = New or significantly modified in Stage 3

---

## 🧩 Component Details

### 1. CALFApplication (core/application.py)

**Purpose**: Manages Ursina application lifecycle and component initialization.

**Responsibilities**:
- Initialize Ursina application
- Create and configure all managers (Window, Zoom, Object, Input, UI, Color)
- Setup 3D scene (ground, grid, lights, frame)
- Configure PyTorch device (CPU/CUDA)
- Provide factory methods for environment creation

**Key Methods**:

```python
class CALFApplication:
    def __init__(self, config: AppConfig)
    def setup(self) -> CALFApplication  # Chaining
    def create_training_env(self) -> PointMassEnv
    def create_visual_envs(self, n_agents: int) -> List[PointMassEnv]
    def update_managers(self)  # Called every frame
    def run(self)  # Start app loop
    def quit(self)  # Exit app
```

**Usage**:

```python
config = AppConfig.from_preset('medium')
app = CALFApplication(config).setup()

env = app.create_training_env()
visual_envs = app.create_visual_envs(25)

app.run()
```

---

### 2. CALFTrainer (training/trainer.py)

**Purpose**: Encapsulates training loop logic and state management.

**Responsibilities**:
- Manage training state (current state, episode, steps)
- Select actions (exploration vs CALF)
- Step environment and store transitions
- Train the agent with replay buffer
- Handle episode termination
- Track statistics (rewards, success rate, losses)
- Save checkpoints periodically
- Finalize training and save final model

**Key Methods**:

```python
class CALFTrainer:
    def __init__(self, calf_agent, env, replay_buffer, nominal_policy, config)

    def should_train(self) -> bool
    def train_step(self) -> Tuple[np.ndarray, bool, dict]
    def handle_episode_end(self, info: dict) -> np.ndarray
    def get_stats(self) -> dict
    def is_complete(self) -> bool
    def finalize(self)
```

**Internal State**:
- `current_state`: Current environment state
- `episode`: Episode counter
- `total_steps`: Total training steps
- `episode_reward`: Cumulative reward for current episode
- `episode_length`: Steps in current episode
- `episode_rewards`: History of episode rewards
- `episode_lengths`: History of episode lengths
- `success_count`: Number of successful episodes
- `training_started`: Whether training has begun (vs exploration)
- `paused`: Training pause state

**Usage**:

```python
trainer = CALFTrainer(calf_agent, env, replay_buffer, nominal_policy, config.training)

# In main loop
next_state, done, info = trainer.train_step()

if done:
    new_state = trainer.handle_episode_end(info)

if trainer.is_complete():
    trainer.finalize()
```

---

### 3. TrainingVisualizer (training/visualizer.py)

**Purpose**: Manages all visualization components.

**Responsibilities**:
- Manage visual agents (spheres + trajectories)
- Update training agent visualization (orange sphere)
- Batch compute agent positions with Q-value heights
- Update critic heatmap and grid overlay
- Visualize Q-certificate timeline
- Update UI text displays (stats, legend)
- Handle dynamic agent addition/removal
- Toggle visibility (heatmap, grid)

**Key Methods**:

```python
class TrainingVisualizer:
    def __init__(self, object_manager, zoom_manager, config, device)

    # Setup
    def setup_visual_agents(self, visual_envs: List[PointMassEnv])
    def setup_training_agent(self)
    def setup_heatmap(self, critic_heatmap, grid_overlay=None)
    def setup_ui(self, config)
    def setup_q_cert_timeline(self, q_cert_timeline, graph_origin, max_steps)

    # Update
    def update_visual_agents(self, policy, training_started, max_steps, goal_eps, boundary)
    def update_training_agent(self, state, calf_agent)
    def update_heatmap(self, total_steps)
    def update_stats_display(self, trainer, calf_agent, current_state, action, config)
    def update_q_cert_timeline(self, calf_agent)

    # Utilities
    def get_agent_height(self, state) -> float
    def get_agent_heights_batch(self, states) -> np.ndarray
    def clear_training_agent_trail(self)
    def clear_q_cert_timeline(self)

    # Dynamic control
    def add_visual_agent(self)
    def remove_visual_agent(self)
    def toggle_heatmap_visibility(self)
    def toggle_grid_visibility(self)
```

**Helper Classes**:

```python
class VisualAgent:
    """Encapsulates agent sphere + trajectory"""
    def __init__(self, object_manager, zoom_manager, agent_id, max_trail_length, decimation, point_color)
    def update_position(self, position, mode='td3')
    def clear_trail(self)
```

**Usage**:

```python
visualizer = TrainingVisualizer(
    object_manager=app.object_manager,
    zoom_manager=app.zoom_manager,
    config=config.visualization,
    device=app.device
)

# Setup
visualizer.setup_visual_agents(visual_envs)
visualizer.setup_training_agent()
visualizer.setup_heatmap(critic_heatmap, grid_overlay)
visualizer.setup_ui(config.training)

# In main loop
visualizer.update_visual_agents(calf_agent.td3, trainer.training_started, ...)
visualizer.update_training_agent(next_state, calf_agent)
visualizer.update_heatmap(trainer.total_steps)
visualizer.update_stats_display(trainer, calf_agent, state, action, config.training)
```

---

## 🔄 Main Loop Flow

```python
def update():
    """Main training loop - called every frame by Ursina"""

    # 1. Training Step
    next_state, done, info = trainer.train_step()

    # 2. Update Training Agent Visualization
    if trainer.training_started:
        visualizer.update_training_agent(next_state, calf_agent)
        visualizer.update_q_cert_timeline(calf_agent)

    # 3. Handle Episode End
    if done:
        new_state = trainer.handle_episode_end(info)
        visualizer.clear_training_agent_trail()
        visualizer.clear_q_cert_timeline()

        if trainer.training_started:
            visualizer.update_training_agent(new_state, calf_agent)

    # 4. Update Visual Agents (batch processing)
    visualizer.update_visual_agents(
        policy=calf_agent.td3,
        training_started=trainer.training_started,
        max_steps_per_episode=config.training.max_steps_per_episode,
        goal_epsilon=config.training.goal_epsilon,
        boundary_limit=config.training.boundary_limit
    )

    # 5. Update Heatmap
    if trainer.training_started:
        visualizer.update_heatmap(trainer.total_steps)

    # 6. Update UI Display
    visualizer.update_stats_display(
        trainer=trainer,
        calf_agent=calf_agent,
        current_state=trainer.current_state,
        action=np.zeros(env.action_dim),
        config=config.training
    )

    # 7. Update Managers (input, zoom, objects)
    app.update_managers()

    # 8. Check Completion
    if trainer.is_complete():
        trainer.finalize()
        app.quit()
```

---

## 🎨 Data Flow

### Training Flow

```
┌─────────┐
│ Config  │
└────┬────┘
     │
     ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ Application  │─────▶│   Trainer    │─────▶│ CALF Agent   │
│   .setup()   │      │ .train_step()│      │ .select_act()│
└──────────────┘      └──────┬───────┘      └──────────────┘
                             │
                             ▼
                      ┌──────────────┐
                      │ Environment  │
                      │   .step()    │
                      └──────┬───────┘
                             │
                             ▼
                      ┌──────────────┐
                      │ReplayBuffer  │
                      │   .add()     │
                      └──────────────┘
```

### Visualization Flow

```
┌──────────────┐
│   Trainer    │
│  (state)     │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────┐
│     TrainingVisualizer       │
├──────────────────────────────┤
│ • get_agent_heights_batch()  │◄──── CriticHeatmap (Q-values)
│ • update_visual_agents()     │
│ • update_training_agent()    │
│ • update_heatmap()           │
│ • update_stats_display()     │
└──────┬───────────────────────┘
       │
       ▼
┌──────────────────┐
│  VisualAgent[]   │
│  • sphere        │
│  • LineTrail     │
└──────────────────┘
```

---

## 🔧 Extending the Architecture

### Adding a New Visualization

1. Extend `TrainingVisualizer`:

```python
# training/visualizer.py

class TrainingVisualizer:
    def setup_my_visualization(self, params):
        self.my_viz = MyVisualization(params)

    def update_my_visualization(self, data):
        self.my_viz.update(data)
```

2. Call from `main.py`:

```python
visualizer.setup_my_visualization(params)

def update():
    visualizer.update_my_visualization(data)
```

### Adding a New Training Strategy

1. Create new trainer class:

```python
# training/my_trainer.py

class MyTrainer(CALFTrainer):
    def train_step(self):
        # Custom training logic
        pass
```

2. Use in `main.py`:

```python
trainer = MyTrainer(agent, env, buffer, policy, config)
```

### Adding a New Application Mode

1. Extend `CALFApplication`:

```python
# core/application.py

class CALFApplication:
    def setup_headless(self):
        # No Ursina, just training
        pass
```

2. Use for headless training:

```python
app = CALFApplication(config).setup_headless()
```

---

## 📊 Benefits of New Architecture

### 1. Maintainability
- **Before**: 974-line monolith - hard to navigate
- **After**: Clear component boundaries - easy to find code

### 2. Testability
- **Before**: Difficult to unit test (everything intertwined)
- **After**: Mock components easily (e.g., mock visualizer for training tests)

### 3. Reusability
- **Before**: Copy-paste code between files
- **After**: Import and reuse components

### 4. Extensibility
- **Before**: Modifications affect entire file
- **After**: Extend specific components without side effects

### 5. Readability
- **Before**: Need to read 974 lines to understand flow
- **After**: Read 317-line main.py for overview, dive into components as needed

---

## 🚀 Performance Improvements

### Stage 1: Batch Operations ✅
- `select_action_batch()` in CALFController
- Batch GPU inference (1 call vs N calls)
- Vectorized certificate checking
- Expected: 32-65x speedup in action selection

### Stage 2: Configuration System ✅
- No scattered constants
- Easy preset switching
- Better experimentation workflow

### Stage 3: Clean Architecture ✅
- 67% code reduction in main file
- Better code organization
- Foundation for future optimizations

### Stage 4: Profiling (Next)
- Performance monitoring
- Bottleneck identification
- Data-driven optimization

---

## 📝 Migration Guide

### From train_calf_visual.py to main.py

**Old Code**:
```python
# train_calf_visual.py (974 lines)
NUM_EPISODES = 500
BATCH_SIZE = 64
# ... 50+ constants

# Setup (100+ lines)
app = Ursina()
player = Player()
# ... managers, scene, agents

# Training loop (200+ lines)
def update():
    # Training logic
    # Visualization logic
    # Manager updates
    # UI updates
```

**New Code**:
```python
# main.py (317 lines)
config = AppConfig.from_preset('medium')
app = CALFApplication(config).setup()
trainer = CALFTrainer(...)
visualizer = TrainingVisualizer(...)

def update():
    next_state, done, info = trainer.train_step()
    visualizer.update_visual_agents(...)
    visualizer.update_training_agent(...)
    app.update_managers()
```

---

## 🎓 Best Practices

1. **Use Configuration System**: Don't hardcode parameters
2. **Follow Component Boundaries**: Don't mix training and visualization logic
3. **Use Factory Methods**: Let Application create environments
4. **Batch Operations**: Always prefer batch APIs when available
5. **Document Extensions**: Update this file when adding components

---

**Version**: Stage 3
**Authors**: Claude Code (Stage 3 refactoring)
**Last Review**: 2026-01-21
