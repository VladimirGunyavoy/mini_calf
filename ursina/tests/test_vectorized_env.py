"""
Тест VectorizedEnvironment - Phase 4.2
Запуск 10 точек с PD политикой
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from ursina import *

from core import Player, setup_scene
from managers import (
    WindowManager,
    ZoomManager,
    ObjectManager,
    ColorManager,
    UIManager,
)
from physics import VectorizedEnvironment
from physics.policies import PDPolicy
from visuals import PointVisual

# ВАЖНО: Устанавливаем размер и позицию окна ДО создания приложения Ursina
WindowManager.setup_before_app(monitor="left")

app = Ursina()

# ============================================================================
# ИНИЦИАЛИЗАЦИЯ КОМПОНЕНТОВ
# ============================================================================

player = Player()
color_manager = ColorManager()

window_manager = WindowManager(color_manager=color_manager, monitor="left")
zoom_manager = ZoomManager(player=player)
object_manager = ObjectManager(zoom_manager=zoom_manager)

# Настройка сцены
ground, grid, lights, frame = setup_scene(color_manager, object_manager)

# ============================================================================
# PHASE 4.2: VECTORIZED ENVIRONMENT - 10 точек с PD
# ============================================================================

print("\n" + "="*70)
print("PHASE 4.2: VECTORIZED ENVIRONMENT TEST")
print("="*70)

# Параметры
N_AGENTS = 10
TARGET_POS = 0.0  # Целевая позиция

# Создаём PD политику для всех агентов
pd_policy = PDPolicy(kp=1.0, kd=0.8, target=np.array([TARGET_POS]), dim=1)

# Создаём векторизованную среду
vec_env = VectorizedEnvironment(
    n_envs=N_AGENTS,
    policy=pd_policy,
    dt=0.01,
    seed=42  # Для воспроизводимости
)

print(f"\nCreated {N_AGENTS} agents via VectorizedEnvironment")
print(f"   - Policy: PDPolicy (Kp={pd_policy.kp}, Kd={pd_policy.kd})")
print(f"   - Target: x={TARGET_POS}")
print(f"   - All agents will converge to x={TARGET_POS}")
print()

# Создаём визуальные представления для каждого агента
visual_points = []
for i in range(N_AGENTS):
    point_visual = PointVisual(
        scale=0.08,
        color=color.cyan,  # Cyan for PD
        zoom_manager=zoom_manager  # Pass zoom_manager for proper scaling
    )
    visual_points.append(point_visual)

print(f"[OK] Created {N_AGENTS} visual objects")

# Initial states
initial_states = vec_env.get_states()
for i, state in enumerate(initial_states):
    x, v = state
    print(f"  Agent {i}: x={x:+.2f}, v={v:+.2f}")

vec_env.print_stats()

# Statistics counters
frame_count = 0
stats_interval = 300  # Print stats every 300 frames (~5 seconds at 60 FPS)

def update():
    """Update each frame"""
    global frame_count
    
    # 1. Vectorized environment step (batch process all agents)
    states = vec_env.step()
    
    # 2. Update visualization (phase space representation)
    for i, state in enumerate(states):
        x = state[0]  # Position
        v = state[1]  # Velocity
        
        # Update visual object position in phase space: (x, 0, v)
        # X-axis = position, Z-axis = velocity
        # This visualizes the phase space dynamics!
        visual_points[i].set_position((x, 0, v))
    
    # 3. Statistics
    frame_count += 1
    if frame_count % stats_interval == 0:
        positions = vec_env.get_positions()
        velocities = vec_env.get_velocities()
        
        print(f"\n[Frame {frame_count}] Stats:")
        print(f"  x: min={positions.min():.3f}, max={positions.max():.3f}, mean={positions.mean():.3f}")
        print(f"  v: min={velocities.min():.3f}, max={velocities.max():.3f}, mean={velocities.mean():.3f}")
        
        # Check convergence
        dist_to_target = np.abs(positions - TARGET_POS)
        print(f"  Distance to target: max={dist_to_target.max():.3f}, mean={dist_to_target.mean():.3f}")

def input(key):
    """Input handling"""
    if key == 'q' or key == 'escape':
        print("\nExiting...")
        application.quit()
    
    # Camera
    if key == 'w':
        player.y += 1
    if key == 's':
        player.y -= 1
    if key == 'a':
        player.x -= 1
    if key == 'd':
        player.x += 1
    
    # Zoom
    if key == 'scroll up':
        zoom_manager.zoom_in()
    if key == 'scroll down':
        zoom_manager.zoom_out()
    
    # Reset
    if key == 'r':
        print("\nResetting environment...")
        vec_env.reset()
        vec_env.print_stats()

print("\nControls:")
print("   WASD - camera movement")
print("   Scroll - zoom")
print("   R - reset environment")
print("   Q/Escape - quit")
print("\nStarting simulation...")

app.run()
