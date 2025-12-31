"""
Test CALF Policy - Single Point with MultiColor Trail
=====================================================

Phase 8.2: Визуализация одной точки с мультицветной траекторией.

Демонстрирует:
1. Одна точка с CALF политикой
2. Траектория меняет цвет при переключении режимов
3. История переключений визуально видна в траектории
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from ursina import *
from physics import VectorizedEnvironment
from physics.policies import CALFPolicy, TD3Policy, PDPolicy
from visuals import MultiColorTrail


# ============================================================================
# CONFIGURATION
# ============================================================================

SEED = 42

# CALF thresholds
FALLBACK_THRESHOLD = 0.3
RELAX_THRESHOLD = 0.6

print("\n" + "="*70)
print("TEST: CALF Policy - Single Point with MultiColor Trail")
print("="*70)
print(f"\nConfiguration:")
print(f"   Mode colors: BLUE=td3, GREEN=relax, ORANGE=fallback")
print(f"   Trajectory changes color when mode switches")
print()


# ============================================================================
# URSINA SETUP
# ============================================================================

app = Ursina()

# Scene setup - темная схема
Sky(color=Vec4(0.04, 0.04, 0.08, 1))
ground = Entity(
    model='plane',
    scale=40,
    color=Vec4(0.12, 0.12, 0.16, 1),
    collider='box'
)

# Camera
camera.position = (0, 10, -10)
camera.rotation_x = 45

# Goal marker (желтый)
Entity(
    model='sphere',
    color=Vec4(0.7, 0.6, 0.2, 1),
    scale=0.4,
    position=(0, 0, 0)
)

# Boundary circle
circle_points = []
for angle in np.linspace(0, 2*np.pi, 64):
    x = 5 * np.cos(angle)
    z = 5 * np.sin(angle)
    circle_points.append(Vec3(x, 0.05, z))

Entity(
    model=Mesh(vertices=circle_points, mode='line', thickness=2),
    color=Vec4(0.5, 0.5, 0.2, 0.4)
)


# ============================================================================
# CALF POLICY SETUP
# ============================================================================

print("[Creating CALF policy...]")

# Sub-policies (снижаем коэффициенты PD для медленного движения)
td3_policy = TD3Policy(action_dim=1, action_scale=0.2)
pd_policy = PDPolicy(kp=0.5, kd=0.4, target=np.array([0.0]), dim=1)

# CALF policy
calf_policy = CALFPolicy(
    td3_policy=td3_policy,
    pd_policy=pd_policy,
    fallback_threshold=FALLBACK_THRESHOLD,
    relax_threshold=RELAX_THRESHOLD,
    target=np.array([0.0]),
    dim=1
)

print(f"[OK] CALF Policy created")


# ============================================================================
# VECTORIZED ENVIRONMENT (1 agent)
# ============================================================================

print("[Creating VectorizedEnvironment...]")

vec_env = VectorizedEnvironment(
    n_envs=1,
    policy=calf_policy,
    dt=0.01,
    seed=SEED
)

vec_env.reset()

# Set initial condition (very far from goal to see all mode switches)
vec_env.envs[0].state = np.array([6.0, 0.0])  # Начинаем далеко

print(f"[OK] VectorizedEnvironment created")


# ============================================================================
# VISUAL OBJECTS
# ============================================================================

print("[Creating visual objects...]")

# Get initial state
state = vec_env.envs[0].state
x, v = state[0], state[1]
initial_position = (x, 0.1, v)

# Point (color will update based on mode)
point = Entity(
    model='sphere',
    color=Vec4(0.2, 0.3, 0.8, 1),  # Initial: blue
    scale=0.2,
    position=initial_position
)

# MultiColor Trail (увеличили длину для длинной траектории)
trail = MultiColorTrail(
    max_length=1500,
    decimation=1,  # Каждый кадр для точного отслеживания переключений
    rebuild_frequency=5  # Чаще перестраиваем для отображения переключений
)
trail.add_point(initial_position, calf_policy.current_mode)

print(f"[OK] Visual objects created with MultiColorTrail")


# ============================================================================
# STATISTICS
# ============================================================================

mode_history = {'td3': 0, 'relax': 0, 'fallback': 0}
episode_steps = 0
max_episode_steps = 5000  # Увеличили для длинной траектории
last_mode = calf_policy.current_mode

stats_text = Text(
    text='',
    position=(-0.85, 0.48),
    scale=0.9,
    color=Vec4(0.7, 0.7, 0.7, 1),
    origin=(-0.5, 0.5)
)

instructions_text = Text(
    text='[Phase 8.2] Single point - trajectory changes color on mode switch',
    position=(0, -0.45),
    scale=0.9,
    color=Vec4(0.7, 0.7, 0.7, 1),
    origin=(0, 0)
)


# ============================================================================
# UPDATE LOOP
# ============================================================================

def update():
    """Update loop called by Ursina every frame"""
    global episode_steps, mode_history, last_mode
    
    # Step environment
    vec_env.step()
    episode_steps += 1
    
    # Get state and mode
    state = vec_env.envs[0].state
    x, v = state[0], state[1]
    position = (x, 0.1, v)
    
    # Get current mode (ВАЖНО: используем get_mode_for_env для batch режима!)
    mode = calf_policy.get_mode_for_env(0)  # Агент 0
    
    # Calculate distance and safety
    distance = np.sqrt(x**2 + v**2)
    safety = calf_policy.get_safety_metric(state)
    
    # Track mode switches
    if mode != last_mode:
        print(f"[Mode switch at step {episode_steps}] {last_mode} -> {mode} (x={x:.2f}, safety={safety:.3f})")
        last_mode = mode
    
    # Debug: print every 60 frames
    if episode_steps % 60 == 0:
        print(f"[Step {episode_steps:4d}] x={x:6.2f}, v={v:6.2f}, dist={distance:6.2f}, safety={safety:.3f}, mode={mode}")
    
    # Update mode history
    mode_history[mode] += 1
    
    # Update point color based on mode
    point.color = MultiColorTrail.MODE_COLORS.get(mode, Vec4(1, 1, 1, 1))
    
    # Update position
    point.position = position
    
    # Add point to trail with mode
    trail.add_point(position, mode)
    
    # Reset conditions
    distance_to_goal = np.sqrt(x**2 + v**2)
    if distance_to_goal < 0.1 or episode_steps >= max_episode_steps:
        print(f"\n[Episode complete at step {episode_steps}]")
        print(f"   Final distance: {distance_to_goal:.3f}")
        print(f"   Mode distribution: {mode_history}")
        
        # Reset (начинаем далеко для демонстрации переключений)
        trail.clear()
        mode_history = {'td3': 0, 'relax': 0, 'fallback': 0}
        x_new = np.random.uniform(5, 7)  # Начинаем далеко от цели
        v_new = np.random.uniform(-0.5, 0.5)
        vec_env.envs[0].state = np.array([x_new, v_new])
        episode_steps = 0
        print(f"\n[Reset] Starting new episode from x={x_new:.2f}")
    
    # Update statistics
    total_steps = sum(mode_history.values())
    td3_pct = (mode_history['td3'] / total_steps * 100) if total_steps > 0 else 0
    relax_pct = (mode_history['relax'] / total_steps * 100) if total_steps > 0 else 0
    fallback_pct = (mode_history['fallback'] / total_steps * 100) if total_steps > 0 else 0
    
    distance = np.sqrt(x**2 + v**2)
    safety = calf_policy.get_safety_metric(state)
    
    stats_text.text = f'''Phase 8.2: MultiColor Trail (Single Point)

Current Mode: {mode.upper()}
Position: x={x:.2f}, v={v:.2f}
Distance: {distance:.2f}
Safety: {safety:.3f}

Mode Time Distribution:
  TD3 (BLUE):      {td3_pct:.1f}%
  Relax (GREEN):   {relax_pct:.1f}%
  Fallback (ORANGE): {fallback_pct:.1f}%

Step: {episode_steps}/{max_episode_steps}
FPS: {int(1/time.dt) if time.dt > 0 else 0}'''


# ============================================================================
# INPUT HANDLER
# ============================================================================

def input(key):
    """Handle keyboard input"""
    if key == 'escape' or key == 'q':
        print("\n[Exiting...]")
        application.quit()
    
    # Adjust thresholds
    if key == 'up arrow':
        calf_policy.fallback_threshold = min(1.0, calf_policy.fallback_threshold + 0.05)
        print(f"[Fallback threshold: {calf_policy.fallback_threshold:.2f}]")
    
    if key == 'down arrow':
        calf_policy.fallback_threshold = max(0.0, calf_policy.fallback_threshold - 0.05)
        print(f"[Fallback threshold: {calf_policy.fallback_threshold:.2f}]")
    
    if key == 'right arrow':
        calf_policy.relax_threshold = min(1.0, calf_policy.relax_threshold + 0.05)
        print(f"[Relax threshold: {calf_policy.relax_threshold:.2f}]")
    
    if key == 'left arrow':
        calf_policy.relax_threshold = max(0.0, calf_policy.relax_threshold - 0.05)
        print(f"[Relax threshold: {calf_policy.relax_threshold:.2f}]")
    
    # Reset trajectory
    if key == 'r':
        print("[Resetting trajectory...]")
        trail.clear()
        mode_history['td3'] = 0
        mode_history['relax'] = 0
        mode_history['fallback'] = 0


# ============================================================================
# RUN
# ============================================================================

print("\n" + "="*70)
print("[STARTING VISUALIZATION]")
print("="*70)
print("\nControls:")
print("   UP/DOWN arrows: Adjust fallback threshold")
print("   LEFT/RIGHT arrows: Adjust relax threshold")
print("   R: Reset trajectory")
print("   ESC or Q: Quit")
print("\nWatch the trajectory:")
print("   - BLUE segments: TD3 mode (close to goal)")
print("   - GREEN segments: Relax mode (medium distance)")
print("   - ORANGE segments: Fallback mode (far from goal)")
print("\nMode switches are printed to console.")
print()

app.run()







