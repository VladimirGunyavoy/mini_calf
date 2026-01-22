# RL/differential_drive_env.py
"""
RL среда для дифференциального привода (Unicycle model).

State: [x, y, theta] - позиция и ориентация
Action: [v, omega] - линейная и угловая скорость

Dynamics:
    x_dot = v * cos(theta)
    y_dot = v * sin(theta)
    theta_dot = omega

Goal: стабилизация в [0, 0, 0]
"""

import numpy as np

try:
    from .base_env import BaseEnv
except ImportError:
    from base_env import BaseEnv


class DifferentialDriveEnv(BaseEnv):
    """
    Дифференциальный привод (unicycle model).
    
    State: [x, y, theta] - позиция + ориентация
    Action: [v, omega] - линейная + угловая скорость
    """
    
    def __init__(
        self,
        dt: float = 0.01,
        max_v: float = 1.0,
        max_omega: float = 2.0,
        goal_radius: float = 0.1,
        goal_angle_tolerance: float = 0.1,
        position_range: float = 2.0,
    ):
        """
        Parameters:
        -----------
        dt : float
            Шаг интегрирования
        max_v : float
            Максимальная линейная скорость
        max_omega : float
            Максимальная угловая скорость
        goal_radius : float
            Радиус целевой области по позиции
        goal_angle_tolerance : float
            Допуск по углу для достижения цели
        position_range : float
            Диапазон для случайной инициализации позиции
        """
        super().__init__(dt, max_v, goal_radius)  # max_action = max_v
        self.max_v = max_v
        self.max_omega = max_omega
        self.goal_angle_tolerance = goal_angle_tolerance
        self.position_range = position_range
    
    @property
    def state_dim(self) -> int:
        """Размерность состояния: [x, y, theta]"""
        return 3
    
    @property
    def action_dim(self) -> int:
        """Размерность действия: [v, omega]"""
        return 2
    
    def reset(self, state=None) -> np.ndarray:
        """
        Сбросить среду в начальное состояние.
        
        Parameters:
        -----------
        state : array-like, optional
            Начальное состояние [x, y, theta]. Если None, генерируется случайно.
        """
        if state is not None:
            self.state = np.array(state, dtype=np.float32)
        else:
            self.state = np.array([
                np.random.uniform(-self.position_range, self.position_range),
                np.random.uniform(-self.position_range, self.position_range),
                np.random.uniform(-np.pi, np.pi),
            ], dtype=np.float32)
        self.steps = 0
        return self.state.copy()
    
    def dynamics(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Динамика unicycle модели.
        
        x_dot = v * cos(theta)
        y_dot = v * sin(theta)
        theta_dot = omega
        """
        x, y, theta = state
        action = np.asarray(action, dtype=np.float32).flatten()
        
        # Клиппинг действий
        v = np.clip(action[0], -self.max_v, self.max_v)
        omega = np.clip(action[1], -self.max_omega, self.max_omega)
        
        return np.array([
            v * np.cos(theta),
            v * np.sin(theta),
            omega
        ], dtype=np.float32)
    
    @staticmethod
    def normalize_angle(angle: float) -> float:
        """Нормализовать угол в диапазон [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def step(self, action) -> tuple:
        """
        Выполнить один шаг в среде.
        
        Returns:
        --------
        next_state : np.ndarray
        reward : float
        done : bool
        info : dict
        """
        action = np.asarray(action, dtype=np.float32).flatten()
        
        # RK4 интегрирование
        next_state = self.step_rk4(self.state, action)
        
        # Нормализация угла
        next_state[2] = self.normalize_angle(next_state[2])
        
        # Reward
        position_dist = np.linalg.norm(next_state[:2])
        angle_dist = abs(next_state[2])
        control_cost = 0.01 * (action[0]**2 + action[1]**2)
        reward = -position_dist - 0.1 * angle_dist - control_cost
        
        # Goal check
        in_goal = (position_dist <= self.goal_radius and 
                   angle_dist <= self.goal_angle_tolerance)
        
        self.state = next_state
        self.steps += 1
        done = self.steps >= self.max_steps or in_goal
        
        info = {
            'position_distance': position_dist,
            'angle_distance': angle_dist,
            'distance_to_goal': position_dist + 0.1 * angle_dist,
            'in_goal': in_goal,
            'steps': self.steps
        }
        
        return self.state.copy(), reward, done, info
    
    def distance_to_goal(self, state=None) -> float:
        """
        Расстояние до цели (комбинация позиции и угла).
        """
        if state is None:
            state = self.state
        return np.linalg.norm(state[:2]) + 0.1 * abs(state[2])


def move_to_point_policy(
    max_v: float = 1.0,
    max_omega: float = 2.0,
    k_rho: float = 0.8,
    k_alpha: float = 3.0,
    k_theta: float = 2.0,
    position_threshold: float = 0.05,
):
    """
    Номинальная политика: едет к (0,0) с theta -> 0.
    
    Двухфазная стратегия:
    1. Если далеко от цели - едем к цели
    2. Если близко - только корректируем угол
    
    Parameters:
    -----------
    max_v : float
        Максимальная линейная скорость
    max_omega : float
        Максимальная угловая скорость
    k_rho : float
        Коэффициент для линейной скорости
    k_alpha : float
        Коэффициент для поворота к цели
    k_theta : float
        Коэффициент для финальной ориентации
    position_threshold : float
        Порог для переключения на режим коррекции угла
    
    Returns:
    --------
    policy : callable
        Функция policy(state) -> action
    """
    def normalize_angle(angle):
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def policy(state):
        x, y, theta = state
        rho = np.sqrt(x**2 + y**2)
        
        if rho < position_threshold:
            # Фаза 2: Близко к цели - только корректируем угол к 0
            v = 0.0
            omega = -k_theta * theta
        else:
            # Фаза 1: Едем к цели
            # Угол к цели (от робота к началу координат)
            angle_to_goal = np.arctan2(-y, -x)
            alpha = normalize_angle(angle_to_goal - theta)
            
            # Если смотрим сильно в сторону от цели - сначала поворачиваемся
            if abs(alpha) > np.pi / 2:
                # Едем назад или сильно поворачиваем
                v = -k_rho * rho * 0.3  # медленно назад
                omega = k_alpha * np.sign(alpha) * 0.5
            else:
                # Нормальное движение к цели
                v = k_rho * rho * np.cos(alpha)
                omega = k_alpha * alpha
        
        # Клиппинг
        v = np.clip(v, -max_v, max_v)
        omega = np.clip(omega, -max_omega, max_omega)
        
        return np.array([v, omega], dtype=np.float32)
    
    return policy


def test_env():
    """Тест среды с номинальной политикой"""
    env = DifferentialDriveEnv()
    policy = move_to_point_policy()
    
    # Проверка размерностей
    print(f"state_dim: {env.state_dim}, action_dim: {env.action_dim}")
    assert env.state_dim == 3, "state_dim should be 3"
    assert env.action_dim == 2, "action_dim should be 2"
    
    test_states = [
        [2.0, 0.0, 0.0],           # справа, смотрит вправо
        [0.0, 2.0, np.pi/2],       # сверху, смотрит вверх
        [-1.0, -1.0, -np.pi/4],    # снизу-слева
        [1.5, 1.5, np.pi],         # справа-сверху, смотрит влево
    ]
    
    all_passed = True
    max_steps_test = 2000
    
    for init_state in test_states:
        state = env.reset(state=init_state)
        print(f"\nStart: x={state[0]:.2f}, y={state[1]:.2f}, theta={np.degrees(state[2]):.1f} deg")
        
        reached_goal = False
        for i in range(max_steps_test):
            action = policy(state)
            state, reward, done, info = env.step(action)
            
            if i % 400 == 0:
                print(f"  Step {i}: pos={info['position_distance']:.3f}, angle={np.degrees(info['angle_distance']):.1f} deg")
            
            if info['in_goal']:
                print(f"  SUCCESS at step {i}: pos={info['position_distance']:.4f}, angle={np.degrees(info['angle_distance']):.2f} deg")
                reached_goal = True
                break
        
        if not reached_goal:
            print(f"  FAILED: pos={info['position_distance']:.4f}, angle={np.degrees(info['angle_distance']):.2f} deg")
            all_passed = False
    
    if all_passed:
        print("\n=== All tests PASSED! ===")
    else:
        print("\n=== Some tests FAILED ===")
    
    return all_passed


if __name__ == "__main__":
    test_env()
