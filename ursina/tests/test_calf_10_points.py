"""
Test CALF Policy - 10 Points with Mode Visualization
====================================================

Phase 7.5: Визуализация 10 точек с CALF политикой.

Демонстрирует:
1. 10 агентов с CALF политикой
2. Цветовое кодирование режимов:
   - BLUE = TD3 mode
   - GREEN = Relax mode
   - ORANGE = Fallback mode
3. Траектории для каждого агента
4. Статистика режимов
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from ursina import *
from physics import VectorizedEnvironment
from physics.policies import CALFPolicy, TD3Policy, PDPolicy
from visuals import SimpleTrail


# ============================================================================
# CONFIGURATION
# ============================================================================

N_AGENTS = 10
SEED = 42

# Mode colors (нормализованные к 0-1)
MODE_COLORS = {
    'td3': Vec4(0.2, 0.3, 0.8, 1),      # Синий (темный)
    'relax': Vec4(0.2, 0.6, 0.3, 1),    # Зеленый (темный)
    'fallback': Vec4(0.8, 0.4, 0.15, 1) # Оранжевый (темный)
}

# CALF thresholds
FALLBACK_THRESHOLD = 0.3
RELAX_THRESHOLD = 0.6

print("\n" + "="*70)
print("TEST: CALF Policy - 10 Points Visualization")
print("="*70)
print(f"\nConfiguration:")
print(f"   Agents: {N_AGENTS}")
print(f"   Fallback threshold: {FALLBACK_THRESHOLD}")
print(f"   Relax threshold: {RELAX_THRESHOLD}")
print(f"   Mode colors: BLUE=td3, GREEN=relax, ORANGE=fallback")
print()


# ============================================================================
# URSINA SETUP
# ============================================================================

app = Ursina()

# Scene setup - темная цветовая схема
Sky(color=Vec4(0.04, 0.04, 0.08, 1))  # Очень темно-синий небо
ground = Entity(
    model='plane',
    scale=40,
    color=Vec4(0.12, 0.12, 0.16, 1),  # Темно-серый пол
    collider='box'
)

# Camera
camera.position = (0, 12, -12)
camera.rotation_x = 45

# Goal marker (желтый at origin)
Entity(
    model='sphere',
    color=Vec4(0.7, 0.6, 0.2, 1),  # Желтый
    scale=0.3,
    position=(0, 0, 0)
)

# Boundary circle (for reference) - темный
circle_points = []
for angle in np.linspace(0, 2*np.pi, 64):
    x = 5 * np.cos(angle)
    z = 5 * np.sin(angle)
    circle_points.append(Vec3(x, 0.05, z))

Entity(
    model=Mesh(vertices=circle_points, mode='line', thickness=2),
    color=Vec4(0.5, 0.5, 0.2, 0.4)  # Желтый круг с прозрачностью
)


# ============================================================================
# CALF POLICY SETUP
# ============================================================================

print("[Creating CALF policy...]")

# Sub-policies
td3_policy = TD3Policy(action_dim=1, action_scale=0.3)
pd_policy = PDPolicy(kp=1.0, kd=0.8, target=np.array([0.0]), dim=1)

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
# VECTORIZED ENVIRONMENT
# ============================================================================

print("[Creating VectorizedEnvironment...]")

vec_env = VectorizedEnvironment(
    n_envs=N_AGENTS,
    policy=calf_policy,
    dt=0.01,
    seed=SEED
)

vec_env.reset()

# Set random initial conditions (scattered around)
np.random.seed(SEED)
for i in range(N_AGENTS):
    x = np.random.uniform(-4, 4)
    v = np.random.uniform(-1, 1)
    vec_env.envs[i].state = np.array([x, v])

print(f"[OK] VectorizedEnvironment created with {N_AGENTS} agents")


# ============================================================================
# VISUAL OBJECTS
# ============================================================================

print("[Creating visual objects...]")

points = []
trails = []

for i in range(N_AGENTS):
    state = vec_env.envs[i].state
    x, v = state[0], state[1]
    position = (x, 0.1, v)
    
    # Point (начальный цвет - приглушенный синий)
    point = Entity(
        model='sphere',
        color=MODE_COLORS['td3'],  # Приглушенный синий
        scale=0.15,
        position=position
    )
    points.append(point)
    
    # Trail (decimation=1 для точного отслеживания переключений режимов)
    trail = SimpleTrail(
        trail_color=color.white,  # Will be updated based on mode
        max_length=500,
        decimation=1,  # Каждый кадр - чтобы не пропустить переключения
        rebuild_frequency=15
    )
    trails.append(trail)
    trail.add_point(position)

print(f"[OK] Created {N_AGENTS} visual objects with trails")


# ============================================================================
# STATISTICS
# ============================================================================

mode_counts = {'td3': 0, 'relax': 0, 'fallback': 0}
episode_steps = 0
max_episode_steps = 3000

stats_text = Text(
    text='',
    position=(-0.85, 0.48),
    scale=0.9,
    color=Vec4(0.7, 0.7, 0.7, 1),  # Серый текст
    origin=(-0.5, 0.5)
)

instructions_text = Text(
    text='[Phase 7.5] CALF: BLUE=TD3 | GREEN=Relax | ORANGE=Fallback',
    position=(0, -0.45),
    scale=1.0,
    color=Vec4(0.7, 0.7, 0.7, 1),  # Серый текст
    origin=(0, 0)
)


# ============================================================================
# UPDATE LOOP
# ============================================================================

def update():
    """Update loop called by Ursina every frame"""
    global episode_steps, mode_counts
    
    # Step vectorized environment (batch)
    vec_env.step()
    episode_steps += 1
    
    # Reset mode counts
    mode_counts = {'td3': 0, 'relax': 0, 'fallback': 0}
    
    # Update each agent
    for i in range(N_AGENTS):
        state = vec_env.envs[i].state
        x, v = state[0], state[1]
        position = (x, 0.1, v)
        
        # Get mode for this agent
        mode = calf_policy.get_mode_for_env(i)
        mode_counts[mode] += 1
        
        # Update color based on mode
        point_color = MODE_COLORS.get(mode, color.white)
        points[i].color = point_color
        
        # Update position and trail
        points[i].position = position
        trails[i].add_point(position)
        
        # Reset conditions
        distance_to_goal = np.sqrt(x**2 + v**2)
        if distance_to_goal < 0.1 or episode_steps >= max_episode_steps:
            # Reset agent
            trails[i].clear()
            x_new = np.random.uniform(-4, 4)
            v_new = np.random.uniform(-1, 1)
            vec_env.envs[i].state = np.array([x_new, v_new])
    
    # Reset episode counter
    if episode_steps >= max_episode_steps:
        episode_steps = 0
    
    # Update statistics
    stats_text.text = f'''Phase 7.5: CALF Policy Visualization

Mode Distribution:
  TD3 (BLUE):      {mode_counts["td3"]}/{N_AGENTS} ({mode_counts["td3"]/N_AGENTS*100:.0f}%)
  Relax (GREEN):   {mode_counts["relax"]}/{N_AGENTS} ({mode_counts["relax"]/N_AGENTS*100:.0f}%)
  Fallback (ORANGE): {mode_counts["fallback"]}/{N_AGENTS} ({mode_counts["fallback"]/N_AGENTS*100:.0f}%)

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
    
    # Adjust thresholds dynamically
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


# ============================================================================
# RUN
# ============================================================================

print("\n" + "="*70)
print("[STARTING VISUALIZATION]")
print("="*70)
print("\nControls:")
print("   UP/DOWN arrows: Adjust fallback threshold")
print("   LEFT/RIGHT arrows: Adjust relax threshold")
print("   ESC or Q: Quit")
print("\nWatch the agents:")
print("   - BLUE agents: Close to goal (TD3 mode)")
print("   - GREEN agents: Medium distance (Relax mode)")
print("   - ORANGE agents: Far from goal (Fallback mode)")
print()

app.run()


