"""
Test CALF Policy - 10 Points with MultiColor Trails
===================================================

Phase 8.3: Визуализация 10 точек с мультицветными траекториями.

Демонстрирует:
1. 10 агентов с CALF политикой
2. Каждая траектория меняет цвет при переключении режимов
3. История переключений каждого агента визуально видна
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

N_AGENTS = 10
SEED = 42

# CALF thresholds
FALLBACK_THRESHOLD = 0.3
RELAX_THRESHOLD = 0.6

print("\n" + "="*70)
print("TEST: CALF Policy - 10 Points with MultiColor Trails")
print("="*70)
print(f"\nConfiguration:")
print(f"   Agents: {N_AGENTS}")
print(f"   Mode colors: BLUE=td3, GREEN=relax, ORANGE=fallback")
print(f"   Each trail shows mode switching history")
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
camera.position = (0, 12, -12)
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
    x = 6 * np.cos(angle)
    z = 6 * np.sin(angle)
    circle_points.append(Vec3(x, 0.05, z))

Entity(
    model=Mesh(vertices=circle_points, mode='line', thickness=2),
    color=Vec4(0.5, 0.5, 0.2, 0.4)
)


# ============================================================================
# CALF POLICY SETUP
# ============================================================================

print("[Creating CALF policy...]")

# Sub-policies (медленные для демонстрации)
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

# Set scattered initial conditions (far from goal)
np.random.seed(SEED)
for i in range(N_AGENTS):
    x = np.random.uniform(5, 7)  # Далеко от цели
    v = np.random.uniform(-0.5, 0.5)
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
    
    # Point (цвет будет обновляться по режиму)
    point = Entity(
        model='sphere',
        color=Vec4(0.8, 0.4, 0.15, 1),  # Оранжевый (fallback)
        scale=0.15,
        position=position
    )
    points.append(point)
    
    # MultiColor Trail
    trail = MultiColorTrail(
        max_length=1000,
        decimation=1,
        rebuild_frequency=5
    )
    # Get initial mode
    mode = calf_policy.get_mode_for_env(i)
    trail.add_point(position, mode)
    trails.append(trail)

print(f"[OK] Created {N_AGENTS} visual objects with MultiColorTrails")


# ============================================================================
# STATISTICS
# ============================================================================

mode_counts = {'td3': 0, 'relax': 0, 'fallback': 0}
episode_steps = 0
max_episode_steps = 5000

stats_text = Text(
    text='',
    position=(-0.85, 0.48),
    scale=0.9,
    color=Vec4(0.7, 0.7, 0.7, 1),
    origin=(-0.5, 0.5)
)

instructions_text = Text(
    text='[Phase 8.3] 10 agents - trails change color on mode switch',
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
    global episode_steps, mode_counts
    
    # Step vectorized environment
    vec_env.step()
    episode_steps += 1
    
    # Reset mode counts
    mode_counts = {'td3': 0, 'relax': 0, 'fallback': 0}
    
    # Update each agent
    for i in range(N_AGENTS):
        state = vec_env.envs[i].state
        x, v = state[0], state[1]
        position = (x, 0.1, v)
        
        # Get mode for this agent (ВАЖНО: через get_mode_for_env!)
        mode = calf_policy.get_mode_for_env(i)
        mode_counts[mode] += 1
        
        # Update point color based on mode
        point_color = MultiColorTrail.MODE_COLORS.get(mode, Vec4(1, 1, 1, 1))
        points[i].color = point_color
        
        # Update position
        points[i].position = position
        
        # Add point to trail with mode
        trails[i].add_point(position, mode)
        
        # Reset conditions
        distance_to_goal = np.sqrt(x**2 + v**2)
        if distance_to_goal < 0.1 or episode_steps >= max_episode_steps:
            # Reset agent
            trails[i].clear()
            x_new = np.random.uniform(5, 7)
            v_new = np.random.uniform(-0.5, 0.5)
            vec_env.envs[i].state = np.array([x_new, v_new])
    
    # Reset episode counter
    if episode_steps >= max_episode_steps:
        episode_steps = 0
        print(f"\n[Episode complete - resetting all agents]")
    
    # Update statistics
    stats_text.text = f'''Phase 8.3: MultiColor Trails (10 Points)

Mode Distribution:
  TD3 (BLUE):        {mode_counts["td3"]}/{N_AGENTS} ({mode_counts["td3"]/N_AGENTS*100:.0f}%)
  Relax (GREEN):     {mode_counts["relax"]}/{N_AGENTS} ({mode_counts["relax"]/N_AGENTS*100:.0f}%)
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
    
    # Clear all trails
    if key == 'c':
        print("[Clearing all trails...]")
        for trail in trails:
            trail.clear()


# ============================================================================
# RUN
# ============================================================================

print("\n" + "="*70)
print("[STARTING VISUALIZATION]")
print("="*70)
print("\nControls:")
print("   UP/DOWN arrows: Adjust fallback threshold")
print("   LEFT/RIGHT arrows: Adjust relax threshold")
print("   C: Clear all trails")
print("   ESC or Q: Quit")
print("\nWatch the trails:")
print("   - Each agent has its own multicolor trail")
print("   - ORANGE: Far from goal (Fallback mode)")
print("   - GREEN: Medium distance (Relax mode)")
print("   - BLUE: Close to goal (TD3 mode)")
print()

app.run()


