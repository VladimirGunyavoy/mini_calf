"""
RL среда для точки с массой (Point Mass).

State: [position, velocity]
Action: acceleration (scalar)
Dynamics: x_dot = [velocity, acceleration]
"""

import numpy as np
from .base_env import BaseEnv


class PointMassEnv(BaseEnv):
    """
    Простая среда: точка в 2D фазовом пространстве [позиция, скорость]
    Цель: стабилизация в нуле

    State: [position, velocity]
    Action: acceleration (scalar)
    Dynamics: x_dot = [velocity, acceleration]
    Reward: -distance_to_goal - control_cost
    """

    def __init__(self, dt=0.01, max_action=5.0, goal_radius=0.1):
        """
        Parameters:
        -----------
        dt : float
            Шаг интегрирования
        max_action : float
            Максимальное ускорение
        goal_radius : float
            Радиус целевой области
        """
        super().__init__(dt, max_action, goal_radius)

    @property
    def state_dim(self) -> int:
        """Размерность состояния: [position, velocity]"""
        return 2

    @property
    def action_dim(self) -> int:
        """Размерность действия: acceleration"""
        return 1

    def reset(self, state=None):
        """Сбросить среду в начальное состояние"""
        if state is not None:
            self.state = np.array(state, dtype=np.float32)
        else:
            # Случайное начальное состояние
            self.state = np.random.uniform(-2, 2, size=2).astype(np.float32)
        self.steps = 0
        return self.state.copy()

    def dynamics(self, state, action):
        """Динамика системы: x_dot = [velocity, acceleration]"""
        position, velocity = state
        acceleration = np.clip(action, -self.max_action, self.max_action)
        # Преобразуем acceleration в скаляр если это массив
        if isinstance(acceleration, np.ndarray):
            acceleration = float(acceleration.item()) if acceleration.size == 1 else float(acceleration[0])
        return np.array([velocity, acceleration], dtype=np.float32)

    def step(self, action):
        """
        Сделать шаг в среде

        Returns:
        --------
        next_state : np.array
        reward : float
        done : bool
        info : dict
        """
        if isinstance(action, np.ndarray):
            action = action.flatten()[0] if action.size > 0 else action[0]

        # Интегрировать динамику (используем RK4 из базового класса)
        next_state = self.step_rk4(self.state, action)

        # Вычислить награду
        distance_to_goal = np.linalg.norm(next_state)
        control_cost = 0.01 * (action ** 2)
        reward = -distance_to_goal - control_cost

        # Проверить достижение цели
        in_goal = distance_to_goal <= self.goal_radius

        # Обновить состояние
        self.state = next_state
        self.steps += 1

        # Условие завершения эпизода
        done = self.steps >= self.max_steps or in_goal

        info = {
            'distance_to_goal': distance_to_goal,
            'in_goal': in_goal,
            'steps': self.steps
        }

        return self.state.copy(), reward, done, info


def pd_nominal_policy(max_action=5.0, kp=1.0, kd=1.0):
    """
    Создать номинальную безопасную политику π₀ на основе PD-контроллера

    Returns:
    --------
    policy : callable
        Функция π₀(state) -> action
    """
    def policy(state):
        position, velocity = state
        # PD контроль: a = -kp * position - kd * velocity
        acceleration = -kp * position - kd * velocity
        acceleration = np.clip(acceleration, -max_action, max_action)
        return np.array([acceleration])

    return policy


def test_env():
    """Тест среды"""
    env = PointMassEnv(dt=0.01, max_action=5.0, goal_radius=0.1)
    
    # Проверка state_dim и action_dim
    print(f"state_dim: {env.state_dim}, action_dim: {env.action_dim}")
    assert env.state_dim == 2, "state_dim должен быть 2"
    assert env.action_dim == 1, "action_dim должен быть 1"

    # Тест с PD-контроллером
    nominal_policy = pd_nominal_policy(max_action=5.0, kp=1.0, kd=1.0)

    state = env.reset(state=[2.0, 1.0])
    print(f"Initial state: {state}")

    for i in range(500):
        action = nominal_policy(state)
        state, reward, done, info = env.step(action)

        if i % 100 == 0:
            print(f"Step {i}: state = {state}, reward = {reward:.4f}, distance = {info['distance_to_goal']:.4f}")

        if done:
            print(f"\nDone at step {i}")
            print(f"Final state: {state}")
            print(f"Distance to goal: {info['distance_to_goal']:.4f}")
            print(f"In goal region: {info['in_goal']}")
            break
    
    print("\nТест пройден!")


if __name__ == "__main__":
    test_env()
