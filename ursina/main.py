"""
Основной файл для среды тренировки с Ursina
Версия с возможностью свободного полёта, полом и frame
Включает InputManager, ZoomManager и ObjectManager
"""

import numpy as np

from ursina import *
from core import Player, setup_scene
from managers import (
    InputManager,
    WindowManager,
    ZoomManager,
    ObjectManager,
    ColorManager,
    UIManager,
    GeneralObjectManager
)
# Импорт модуля physics (физические системы)
from physics import PointSystem, SimulationEngine, VectorizedEnvironment
from physics.policies import PDPolicy, TD3Policy, CALFPolicy, PolicyAdapter, RandomSwitchPolicy
from visuals import PointVisual, SimpleTrail, MultiColorTrail
from pathlib import Path

# ВАЖНО: Устанавливаем размер и позицию окна ДО создания приложения Ursina
WindowManager.setup_before_app(monitor="left")

app = Ursina()

# ============================================================================
# ИНИЦИАЛИЗАЦИЯ КОМПОНЕНТОВ
# ============================================================================

# Базовые независимые компоненты
player = Player()
color_manager = ColorManager()

# Менеджеры (порядок создания менее критичен)
window_manager = WindowManager(color_manager=color_manager, monitor="left")
zoom_manager = ZoomManager(player=player)
object_manager = ObjectManager(zoom_manager=zoom_manager)
input_manager = InputManager(zoom_manager=zoom_manager, player=player)
ui_manager = UIManager(
    color_manager=color_manager,
    player=player,
    zoom_manager=zoom_manager
)

# Симуляция и связь с визуализацией
simulation_engine = SimulationEngine()
general_object_manager = GeneralObjectManager(
    simulation_engine=simulation_engine,
    object_manager=object_manager,
    zoom_manager=zoom_manager
)

# ============================================================================
# НАСТРОЙКА СЦЕНЫ - единая функция
# ============================================================================

ground, grid, lights, frame = setup_scene(color_manager, object_manager)

# ============================================================================
# ТЕСТОВЫЕ ОБЪЕКТЫ - создаем через ObjectManager
# ============================================================================

# # Стрелка
# object_manager.create_object(
#     name='my_arrow',
#     model='assets/arrow.obj',
#     position=(0.5, 0.5, 0.5),
#     scale=1.0,
#     color_val=color.red
# )

# # Кубики на осях
# k = 0.05
# object_manager.create_object(
#     name='my_cube_1',
#     model='cube',
#     position=(1, 0, 0),
#     scale=k * np.array([1, 1, 1]),
#     color_val=color.blue
# )

# object_manager.create_object(
#     name='my_cube_2',
#     model='cube',
#     position=(0, 1, 0),
#     scale=k * np.array([1, 1, 1]),
#     color_val=color.green
# )

# object_manager.create_object(
#     name='my_cube_3',
#     model='cube',
#     position=(0, 0, 1),
#     scale=k * np.array([1, 1, 1]),
#     color_val=color.yellow
# )

# Вывести статистику
object_manager.print_stats()

# ============================================================================
# ОБЩИЕ ОБЪЕКТЫ - создаем через GeneralObjectManager
# ============================================================================

# ============================================================================
# PHASE 9: TD3 vs CALF - Сравнение реального TD3 с CALF
# ============================================================================

print("\n" + "="*70)
print("PHASE 9: TD3 vs CALF DUAL VISUALIZATION")
print("="*70)

# Параметры для dual визуализации
N_AGENTS_PER_GROUP = 15  # 15 агентов на группу = 30 всего
SEED = 42

print(f"\nLoading Policies...")

# Путь к обученной модели TD3
model_path = Path(__file__).parent.parent / "RL" / "calf_model.pth"

# Загрузка TD3 политики (реальный агент)
try:
    td3_policy = TD3Policy.create_from_checkpoint(
        checkpoint_path=str(model_path),
        state_dim=2,
        action_dim=1,
        max_action=5.0
    )
    print("[OK] TD3 policy loaded (real agent on CUDA)!")
except Exception as e:
    print(f"[WARNING] Failed to load TD3: {e}")
    print("Using stub mode instead")
    td3_policy = TD3Policy(agent=None, action_dim=1, action_scale=0.3)

# Создание CALF политики (TD3 + PD fallback)
pd_policy = PDPolicy(
    kp=1.0,
    kd=1.0,
    target=np.array([0.0]),  # 1D target for control
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
        target=np.array([0.0, 0.0])  # 2D target for safety metric
    )
    print("[OK] CALF policy created!")
except Exception as e:
    print(f"[WARNING] Failed to create CALF: {e}")
    calf_policy = pd_policy

print(f"\nCreating 2 groups of {N_AGENTS_PER_GROUP} agents:")
print(f"   - LEFT (BLUE):  Real TD3 agent")
print(f"   - RIGHT (MULTI): CALF (TD3 + PD fallback)")
print(f"   - Initial conditions: SYNCHRONIZED (same seed)")
print()

# Векторизованные среды с одинаковым seed
vec_env_td3 = VectorizedEnvironment(
    n_envs=N_AGENTS_PER_GROUP,
    policy=td3_policy,
    dt=0.01,
    seed=SEED
)

vec_env_calf = VectorizedEnvironment(
    n_envs=N_AGENTS_PER_GROUP,
    policy=calf_policy,
    dt=0.01,
    seed=SEED
)

vec_env_td3.reset()
vec_env_calf.reset()

print(f"[OK] VectorizedEnvironments created with synchronized states")

# Визуальные объекты и траектории
points_td3 = []
points_calf = []
trails_td3 = []
trails_calf = []

# Статистика для отслеживания
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

# Создаем TD3 группу (BLUE, LEFT)
for i in range(N_AGENTS_PER_GROUP):
    state = vec_env_td3.envs[i].state
    x, v = state[0], state[1]
    pos = (x - 8, 0.1, v)  # LEFT side (shift by X=-8)

    point = object_manager.create_object(
        name=f'td3_point_{i}',
        model='sphere',
        position=pos,
        scale=0.1,
        color_val=Vec4(0.2, 0.3, 0.8, 1)  # Blue
    )
    points_td3.append(point)

    trail = MultiColorTrail(
        max_length=600,
        decimation=2,
        rebuild_frequency=10
    )
    trails_td3.append(trail)
    trail.add_point(pos, mode='td3')

print(f"[OK] Created TD3 group (BLUE, LEFT): {N_AGENTS_PER_GROUP} agents")

# Создаем CALF группу (MULTI-COLOR, RIGHT)
for i in range(N_AGENTS_PER_GROUP):
    state = vec_env_calf.envs[i].state
    x, v = state[0], state[1]
    pos = (x + 8, 0.1, v)  # RIGHT side (shift by X=+8)

    point = object_manager.create_object(
        name=f'calf_point_{i}',
        model='sphere',
        position=pos,
        scale=0.1,
        color_val=Vec4(0.8, 0.4, 0.15, 1)  # Start orange (fallback)
    )
    points_calf.append(point)

    trail = MultiColorTrail(
        max_length=600,
        decimation=2,
        rebuild_frequency=10
    )
    trails_calf.append(trail)
    trail.add_point(pos, mode='fallback')

print(f"[OK] Created CALF group (MULTI-COLOR, RIGHT): {N_AGENTS_PER_GROUP} agents")

# Желтые сферы в центрах симуляций (цели)
# TD3 goal (LEFT): x=-8, z=0
object_manager.create_object(
    name='td3_goal',
    model='sphere',
    position=(-8, 0, 0),
    scale=0.25,
    color_val=Vec4(0.8, 0.8, 0.3, 0.3)
)

# CALF goal (RIGHT): x=+8, z=0
object_manager.create_object(
    name='calf_goal',
    model='sphere',
    position=(8, 0, 0),
    scale=0.25,
    color_val=Vec4(0.8, 0.8, 0.3, 0.3)
)

# Boundary boxes
# TD3 box (left)
object_manager.create_object(
    name='td3_boundary',
    model='cube',
    position=(-8, 0, 0),
    scale=(10, 0.1, 10),
    color_val=Vec4(0.2, 0.3, 0.8, 0.1)
)

# CALF box (right)
object_manager.create_object(
    name='calf_boundary',
    model='cube',
    position=(8, 0, 0),
    scale=(10, 0.1, 10),
    color_val=Vec4(0.2, 0.6, 0.3, 0.1)
)

# UI текст для статистики
stats_text = Text(
    text='',
    position=(-0.85, 0.45),
    scale=1.0,
    color=color.white
)

# Метки для групп
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

print()
print("="*70)
print("[OK] DUAL VISUALIZATION READY")
print("="*70)
print()

general_object_manager.print_stats()
simulation_engine.print_stats()

def update():
    """Обновление каждого кадра - вызывается автоматически Ursina"""
    global stats

    # 1. Обновление математики (симуляция физики)
    simulation_engine.update_all()

    # 2. Синхронизация визуализации с математикой
    general_object_manager.update_all()

    # 3. TD3 vs CALF: Update vectorized environments
    vec_env_td3.step()
    vec_env_calf.step()
    stats['step_counter'] += 1

    # Update TD3 group (LEFT, x=-8)
    for i in range(len(points_td3)):
        state = vec_env_td3.envs[i].state
        x, v = state[0], state[1]
        position = (x - 8, 0.1, v)  # LEFT side

        distance = np.linalg.norm(state)
        stats['td3_distances'].append(distance)

        # Reset on success
        if distance < 0.15:
            stats['td3_success'] += 1
            stats['td3_steps_to_goal'].append(stats['step_counter'])
            stats['td3_resets'] += 1
            trails_td3[i].clear()
            new_state = np.array([np.random.uniform(-2, 2), np.random.uniform(-0.5, 0.5)])
            vec_env_td3.envs[i].state = new_state

        points_td3[i].position = position
        trails_td3[i].add_point(position, mode='td3')

    # Update CALF group (RIGHT, x=+8)
    for i in range(len(points_calf)):
        state = vec_env_calf.envs[i].state
        x, v = state[0], state[1]
        position = (x + 8, 0.1, v)  # RIGHT side
        mode = vec_env_calf.policy.get_mode_for_env(i)

        # Color based on CALF mode
        if mode == 'td3':
            points_calf[i].color = Vec4(0.2, 0.3, 0.8, 1)  # Blue
        elif mode == 'relax':
            points_calf[i].color = Vec4(0.2, 0.6, 0.3, 1)  # Green
        elif mode == 'fallback':
            points_calf[i].color = Vec4(0.8, 0.4, 0.15, 1)  # Orange

        distance = np.linalg.norm(state)
        stats['calf_distances'].append(distance)

        # Reset on success
        if distance < 0.15:
            stats['calf_success'] += 1
            stats['calf_steps_to_goal'].append(stats['step_counter'])
            stats['calf_resets'] += 1
            trails_calf[i].clear()
            new_state = np.array([np.random.uniform(-2, 2), np.random.uniform(-0.5, 0.5)])
            vec_env_calf.envs[i].state = new_state

        points_calf[i].position = position
        trails_calf[i].add_point(position, mode=mode)

    # Update statistics display
    td3_avg_dist = np.mean(stats['td3_distances'][-100:]) if stats['td3_distances'] else 0
    calf_avg_dist = np.mean(stats['calf_distances'][-100:]) if stats['calf_distances'] else 0

    td3_success_rate = stats['td3_success'] / max(1, stats['td3_resets']) * 100
    calf_success_rate = stats['calf_success'] / max(1, stats['calf_resets']) * 100

    td3_avg_steps = np.mean(stats['td3_steps_to_goal']) if stats['td3_steps_to_goal'] else 0
    calf_avg_steps = np.mean(stats['calf_steps_to_goal']) if stats['calf_steps_to_goal'] else 0

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

>>> {better_policy} <<<'''

    # 4. Обновление менеджеров напрямую
    # Порядок важен: input → zoom → object → ui
    if hasattr(input_manager, 'update'):
        input_manager.update()

    if hasattr(zoom_manager, 'update'):
        zoom_manager.update()

    if hasattr(object_manager, 'update'):
        object_manager.update()

    ui_manager.update()

def input(key):
    """Глобальный обработчик ввода"""
    # Передаем управление в централизованный InputManager
    # InputManager сам обрабатывает все клавиши, включая q/escape и alt
    input_manager.handle_input(key)
        
        
app.run()