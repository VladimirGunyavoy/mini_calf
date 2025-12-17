import numpy as np


class PointMassEnv:
    """
    Простая среда: точка в 2D пространстве [позиция, скорость]
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
        self.dt = dt
        self.max_action = max_action
        self.goal_radius = goal_radius

        self.state_dim = 2
        self.action_dim = 1

        self.state = None
        self.steps = 0
        self.max_steps = 5000

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
        return np.array([velocity, acceleration])

    def step_rk4(self, state, action):
        """Один шаг метода Рунге-Кутта 4-го порядка"""
        k1 = self.dynamics(state, action)
        k2 = self.dynamics(state + self.dt * k1 / 2, action)
        k3 = self.dynamics(state + self.dt * k2 / 2, action)
        k4 = self.dynamics(state + self.dt * k3, action)
        return state + self.dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

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
            action = action[0]

        # Интегрировать динамику
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

    def distance_to_goal(self, state=None):
        """Расстояние до цели"""
        if state is None:
            state = self.state
        return np.linalg.norm(state)

    def render(self):
        """Вывести текущее состояние"""
        print(f"Step {self.steps}: state = {self.state}, distance = {self.distance_to_goal():.4f}")


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


if __name__ == "__main__":
    test_env()
