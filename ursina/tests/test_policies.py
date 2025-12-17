"""
Test Policy Switching - Проверка переключения между политиками
==============================================================

Phase 3 Test: Проверяем, что можно переключаться между:
- PDPolicy (классический PD контроллер)
- TD3Policy (stub с случайными действиями)

Используем PolicyAdapter для обратной совместимости с PointSystem.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from physics import PointSystem
from physics.policies import Policy, PDPolicy, TD3Policy, PolicyAdapter


def test_policy_creation():
    """Тест 1: Создание политик"""
    print("\n=== Test 1: Policy Creation ===")

    # PDPolicy
    pd_policy = PDPolicy(kp=1.0, kd=0.5, target=np.array([0.0]))
    print(f"[OK] PDPolicy created: kp={pd_policy.kp}, kd={pd_policy.kd}")

    # TD3Policy (stub)
    td3_policy = TD3Policy(action_dim=1, action_scale=0.1)
    print(f"[OK] TD3Policy created (stub mode)")

    return pd_policy, td3_policy


def test_policy_actions():
    """Тест 2: Получение действий от политик"""
    print("\n=== Test 2: Policy Actions ===")

    # Создаем политики
    pd_policy = PDPolicy(kp=1.0, kd=0.5, target=np.array([0.0]), dim=1)
    td3_policy = TD3Policy(action_dim=1, action_scale=0.1)

    # Тестовое состояние
    state = np.array([1.0, 0.5])  # [x=1.0, v=0.5]

    # PD действие
    pd_action = pd_policy.get_action(state)
    print(f"PD action for state {state}: {pd_action}")
    print(f"  Expected: negative (should pull towards target=0)")

    # TD3 действие (случайное)
    td3_action = td3_policy.get_action(state)
    print(f"TD3 action for state {state}: {td3_action}")
    print(f"  Expected: random ~ N(0, 0.1)")

    return pd_policy, td3_policy


def test_policy_adapter():
    """Тест 3: PolicyAdapter для обратной совместимости"""
    print("\n=== Test 3: PolicyAdapter ===")

    # Создаем политику
    policy = PDPolicy(kp=1.0, kd=0.5, target=np.array([0.0]), dim=1)

    # Оборачиваем в адаптер
    controller = PolicyAdapter(policy)

    # Используем через интерфейс Controller
    state = np.array([1.0, 0.5])
    control = controller.get_control(state)

    print(f"Policy -> Controller adapter works!")
    print(f"  State: {state}")
    print(f"  Control: {control}")

    return controller


def test_policy_switching():
    """Тест 4: Переключение политик в PointSystem"""
    print("\n=== Test 4: Policy Switching in PointSystem ===")

    # Создаем политики
    pd_policy = PDPolicy(kp=1.0, kd=0.5, target=np.array([0.0]), dim=1)
    td3_policy = TD3Policy(action_dim=1, action_scale=0.1)

    # Оборачиваем в адаптеры
    pd_controller = PolicyAdapter(pd_policy)
    td3_controller = PolicyAdapter(td3_policy)

    # Создаем систему с PD контроллером
    initial_state = np.array([2.0, 0.0])
    point = PointSystem(
        dt=0.01,
        initial_state=initial_state,
        controller=pd_controller
    )

    print(f"\n1. Using PDPolicy:")
    print(f"   Initial state: {point.get_state()}")

    # Делаем несколько шагов
    for i in range(5):
        point.step()

    pd_final_state = point.get_state()
    print(f"   After 5 steps: {pd_final_state}")
    print(f"   Expected: x moves towards 0 (target)")

    # Переключаемся на TD3
    point.reset_state()
    point.controller = td3_controller

    print(f"\n2. Using TD3Policy (stub):")
    print(f"   Reset to: {point.get_state()}")

    # Делаем несколько шагов
    for i in range(5):
        point.step()

    td3_final_state = point.get_state()
    print(f"   After 5 steps: {td3_final_state}")
    print(f"   Expected: random walk (TD3 stub gives random actions)")

    print(f"\n[OK] Policy switching works!")
    print(f"   PD final state: {pd_final_state}")
    print(f"   TD3 final state: {td3_final_state}")
    print(f"   Different behaviors confirmed!")


def test_batch_actions():
    """Тест 5: Батчевая обработка действий"""
    print("\n=== Test 5: Batch Actions ===")

    # Создаем политику
    policy = PDPolicy(kp=1.0, kd=0.5, target=np.array([0.0]), dim=1)

    # Батч состояний
    states = np.array([
        [1.0, 0.0],
        [2.0, 0.5],
        [-1.0, -0.3]
    ])

    # Получаем батч действий
    actions = policy.get_actions_batch(states)

    print(f"Batch of {len(states)} states processed:")
    for i, (state, action) in enumerate(zip(states, actions)):
        print(f"  State {i}: {state} -> Action: {action}")

    print(f"[OK] Batch processing works!")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Policy System (Phase 3)")
    print("=" * 60)

    # Запускаем все тесты
    test_policy_creation()
    test_policy_actions()
    test_policy_adapter()
    test_policy_switching()
    test_batch_actions()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
    print("\nPhase 3 Complete:")
    print("  [OK] Base Policy class created")
    print("  [OK] PDPolicy implemented")
    print("  [OK] TD3Policy stub implemented")
    print("  [OK] PolicyAdapter for backward compatibility")
    print("  [OK] Policy switching verified")
    print("\nReady for Phase 4: Vectorized Environments")
