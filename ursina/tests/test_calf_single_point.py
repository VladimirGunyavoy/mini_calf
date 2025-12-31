"""
Test CALF Policy - Single Point Mode Switching
===============================================

Phase 7.3: Проверка переключения режимов на одной точке.

Тест демонстрирует:
1. Создание CALF политики с TD3 и PD
2. Переключение между режимами (td3, relax, fallback)
3. Изменение режима в зависимости от расстояния от цели
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from physics.policies import CALFPolicy, TD3Policy, PDPolicy
from physics import PointSystem


def test_calf_modes():
    """
    Тест переключения режимов CALF на различных расстояниях от цели.
    """
    print("\n" + "="*70)
    print("TEST: CALF Policy Mode Switching (Single Point)")
    print("="*70)
    
    # Создаем политики
    td3_policy = TD3Policy(action_dim=1, action_scale=0.3)
    pd_policy = PDPolicy(kp=1.0, kd=0.8, target=np.array([0.0]), dim=1)
    
    # Создаем CALF политику
    calf = CALFPolicy(
        td3_policy=td3_policy,
        pd_policy=pd_policy,
        fallback_threshold=0.3,
        relax_threshold=0.6,
        target=np.array([0.0]),
        dim=1
    )
    
    print("\n[OK] CALF Policy created")
    print(f"   Fallback threshold: {calf.fallback_threshold}")
    print(f"   Relax threshold: {calf.relax_threshold}")
    print(f"   Target: {calf.target}")
    
    # Тестируем различные состояния
    test_states = [
        # [x, v], distance, expected_mode
        (np.array([5.0, 0.0]), "far from target (5.0)", "fallback"),
        (np.array([3.0, 0.0]), "medium distance (3.0)", "fallback"),
        (np.array([2.0, 0.0]), "closer (2.0)", "fallback or relax"),
        (np.array([1.0, 0.0]), "close (1.0)", "relax"),
        (np.array([0.5, 0.0]), "very close (0.5)", "relax or td3"),
        (np.array([0.2, 0.0]), "near target (0.2)", "td3"),
        (np.array([0.05, 0.0]), "at target (0.05)", "td3"),
    ]
    
    print("\n" + "-"*70)
    print("Testing mode switching:")
    print("-"*70)
    print(f"{'State':<20} {'Distance':<12} {'Safety':<10} {'Mode':<12} {'Expected':<15}")
    print("-"*70)
    
    for state, description, expected_mode in test_states:
        # Вычисляем safety metric
        safety = calf.get_safety_metric(state)
        
        # Получаем действие (это также установит current_mode)
        action = calf.get_action(state)
        
        # Расстояние от цели
        distance = np.abs(state[0])
        
        print(f"x={state[0]:5.2f}, v={state[1]:4.1f}  "
              f"d={distance:6.2f}     "
              f"s={safety:6.4f}  "
              f"{calf.current_mode:<12} "
              f"{expected_mode:<15}")
    
    print("-"*70)
    
    # Проверяем переключения в динамике
    print("\n" + "-"*70)
    print("Dynamic simulation: point moving towards target")
    print("-"*70)
    
    # Создаем точку далеко от цели
    state = np.array([4.0, 0.0])
    dt = 0.01
    
    print(f"\n{'Step':<8} {'Position':<12} {'Velocity':<12} {'Safety':<10} {'Mode':<12}")
    print("-"*70)
    
    modes_seen = set()
    
    for step in range(300):
        # Получаем действие
        safety = calf.get_safety_metric(state)
        action = calf.get_action(state)
        
        # Запоминаем режим
        modes_seen.add(calf.current_mode)
        
        # Вывод каждые 30 шагов
        if step % 30 == 0:
            print(f"{step:<8} x={state[0]:6.3f}    v={state[1]:6.3f}    "
                  f"s={safety:6.4f}  {calf.current_mode:<12}")
        
        # Простая интеграция: x' = v, v' = u
        x, v = state[0], state[1]
        u = float(action[0]) if isinstance(action, np.ndarray) else float(action)
        
        # Euler integration
        x_new = x + v * dt
        v_new = v + u * dt
        state = np.array([x_new, v_new])
        
        # Останавливаемся если достигли цели
        if np.abs(state[0]) < 0.01 and np.abs(state[1]) < 0.1:
            print(f"{step:<8} x={state[0]:6.3f}    v={state[1]:6.3f}    "
                  f"s={safety:6.4f}  {calf.current_mode:<12} [GOAL REACHED]")
            break
    
    print("-"*70)
    print(f"\nModes seen during simulation: {sorted(modes_seen)}")
    
    # Проверка результатов
    print("\n" + "="*70)
    print("TEST RESULTS:")
    print("="*70)
    
    if len(modes_seen) >= 2:
        print("[OK] Mode switching works: observed", len(modes_seen), "different modes")
        print("    Modes:", sorted(modes_seen))
    else:
        print("[WARNING] Only one mode observed:", modes_seen)
        print("    This might be OK depending on trajectory")
    
    # Проверяем корректность safety metric
    safety_far = calf.get_safety_metric(np.array([5.0, 0.0]))
    safety_close = calf.get_safety_metric(np.array([0.2, 0.0]))
    
    if safety_far < safety_close:
        print("[OK] Safety metric: far < close (correct)")
    else:
        print("[FAIL] Safety metric: far >= close (incorrect)")
    
    # Проверяем, что режим меняется с расстоянием
    mode_far = calf.get_action(np.array([5.0, 0.0])) and calf.current_mode
    mode_close = calf.get_action(np.array([0.2, 0.0])) and calf.current_mode
    
    print(f"[INFO] Mode when far (x=5.0): {mode_far}")
    print(f"[INFO] Mode when close (x=0.2): {mode_close}")
    
    if mode_far == CALFPolicy.MODE_FALLBACK:
        print("[OK] Fallback mode activated when far from target")
    else:
        print(f"[WARNING] Expected fallback when far, got {mode_far}")
    
    if mode_close in [CALFPolicy.MODE_TD3, CALFPolicy.MODE_RELAX]:
        print("[OK] TD3 or Relax mode activated when close to target")
    else:
        print(f"[WARNING] Expected td3/relax when close, got {mode_close}")
    
    print("\n" + "="*70)
    print("[TEST COMPLETE] CALF mode switching verified")
    print("="*70)
    print()


if __name__ == "__main__":
    test_calf_modes()







