# Stage 3: План реализации Differential Drive

**Дата начала**: 2026-01-22
**Приоритет**: RL → Физика → Визуализация → Интеграция

---

# Секция 1: Краткий чеклист

## Части реализации

- [x] **Part 1**: Базовые абстракции (BaseEnv, BaseDynamicalSystem)
- [x] **Part 2**: DifferentialDriveEnv (RL среда)
- [x] **Part 3**: Номинальная политика (move-to-point)
- [x] **Part 4**: DifferentialDriveSystem (физика для визуализации)
- [x] **Part 5**: OrientedAgent (визуализация cone)
- [x] **Part 6**: Интеграция VectorizedEnv + Config
- [x] **Part 7**: Интеграция main.py + CriticHeatmap
- [x] **Part 8**: Финальное тестирование

---

# Секция 2: Детальное ТЗ по частям

---

## Part 1: Базовые абстракции

**Приоритет**: КРИТИЧЕСКИЙ (блокирует Part 2-4)
**Статус**: ✅ ЗАВЕРШЁН
**Файлы**: `RL/base_env.py`, рефакторинг `RL/simple_env.py`

### Задачи

#### 1.1 Создать RL/base_env.py

```python
# RL/base_env.py

from abc import ABC, abstractmethod
import numpy as np


class BaseEnv(ABC):
    """Абстрактный базовый класс для RL сред."""
    
    def __init__(self, dt: float, max_action: float, goal_radius: float):
        self.dt = dt
        self.max_action = max_action
        self.goal_radius = goal_radius
        self.state = None
        self.steps = 0
        self.max_steps = 5000
    
    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Размерность пространства состояний"""
        pass
    
    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Размерность пространства действий"""
        pass
    
    @abstractmethod
    def reset(self, state=None) -> np.ndarray:
        """Сбросить среду"""
        pass
    
    @abstractmethod
    def step(self, action) -> tuple:
        """Шаг среды: (next_state, reward, done, info)"""
        pass
    
    @abstractmethod
    def dynamics(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Динамика: state_dot = f(state, action)"""
        pass
    
    def distance_to_goal(self, state=None) -> float:
        """Расстояние до цели (по умолчанию - норма)"""
        if state is None:
            state = self.state
        return np.linalg.norm(state)
    
    def step_rk4(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """RK4 интегрирование"""
        k1 = self.dynamics(state, action)
        k2 = self.dynamics(state + self.dt * k1 / 2, action)
        k3 = self.dynamics(state + self.dt * k2 / 2, action)
        k4 = self.dynamics(state + self.dt * k3, action)
        return state + self.dt * (k1 + 2*k2 + 2*k3 + k4) / 6
```

#### 1.2 Рефакторинг RL/simple_env.py

**Изменения**:
```python
# Добавить импорт
from .base_env import BaseEnv

# Изменить наследование
class PointMassEnv(BaseEnv):  # было: class PointMassEnv:

# В __init__ добавить super().__init__()
def __init__(self, dt=0.01, max_action=5.0, goal_radius=0.1):
    super().__init__(dt, max_action, goal_radius)
    # Убрать: self.state_dim = 2 и self.action_dim = 1

# Добавить @property для state_dim и action_dim
@property
def state_dim(self) -> int:
    return 2

@property
def action_dim(self) -> int:
    return 1
```

#### 1.3 Тест регрессии

```bash
py -3.12 -c "from RL.simple_env import PointMassEnv; e = PointMassEnv(); print(e.state_dim, e.action_dim)"
# Ожидание: 2 1
```

### Критерии успеха
- [x] BaseEnv создан
- [x] PointMassEnv наследуется от BaseEnv
- [x] Тест регрессии проходит

---

## Part 2: DifferentialDriveEnv

**Приоритет**: КРИТИЧЕСКИЙ
**Статус**: Pending
**Файл**: `RL/differential_drive_env.py`

### Математическая модель

```
State:  [x, y, θ]      - 3D
Action: [v, ω]         - 2D

Dynamics:
  ẋ = v · cos(θ)
  ẏ = v · sin(θ)
  θ̇ = ω

Цель: [0, 0, 0]
```

### Задачи

#### 2.1 Создать RL/differential_drive_env.py

```python
# RL/differential_drive_env.py

import numpy as np
from .base_env import BaseEnv


class DifferentialDriveEnv(BaseEnv):
    """
    Дифференциальный привод (unicycle model).
    
    State: [x, y, θ]
    Action: [v, ω]
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
        super().__init__(dt, max_v, goal_radius)  # max_action = max_v
        self.max_v = max_v
        self.max_omega = max_omega
        self.goal_angle_tolerance = goal_angle_tolerance
        self.position_range = position_range
    
    @property
    def state_dim(self) -> int:
        return 3
    
    @property
    def action_dim(self) -> int:
        return 2
    
    def reset(self, state=None) -> np.ndarray:
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
        x, y, theta = state
        action = np.asarray(action, dtype=np.float32).flatten()
        v = np.clip(action[0], -self.max_v, self.max_v)
        omega = np.clip(action[1], -self.max_omega, self.max_omega)
        
        return np.array([
            v * np.cos(theta),
            v * np.sin(theta),
            omega
        ], dtype=np.float32)
    
    @staticmethod
    def normalize_angle(angle: float) -> float:
        """Нормализовать угол в [-π, π]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def step(self, action) -> tuple:
        action = np.asarray(action, dtype=np.float32).flatten()
        
        # RK4 интегрирование
        next_state = self.step_rk4(self.state, action)
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
        if state is None:
            state = self.state
        return np.linalg.norm(state[:2]) + 0.1 * abs(state[2])
```

### Критерии успеха
- [ ] state_dim = 3, action_dim = 2
- [ ] Динамика корректна
- [ ] Угол нормализуется в [-π, π]

---

## Part 3: Номинальная политика

**Приоритет**: КРИТИЧЕСКИЙ
**Статус**: Pending
**Файл**: `RL/differential_drive_env.py` (в конце)

### Задачи

#### 3.1 Реализовать move_to_point_policy

```python
def move_to_point_policy(
    max_v: float = 1.0,
    max_omega: float = 2.0,
    k_rho: float = 1.0,
    k_alpha: float = 2.0,
    k_beta: float = -0.5,
):
    """
    Номинальная политика: едет к (0,0) с θ→0.
    """
    def policy(state):
        x, y, theta = state
        rho = np.sqrt(x**2 + y**2)
        
        if rho < 0.01:
            v = 0.0
            omega = -k_beta * theta
        else:
            angle_to_goal = np.arctan2(-y, -x)
            alpha = angle_to_goal - theta
            
            # Нормализация
            while alpha > np.pi:
                alpha -= 2 * np.pi
            while alpha < -np.pi:
                alpha += 2 * np.pi
            
            v = k_rho * rho * np.cos(alpha)
            omega = k_alpha * alpha + k_beta * theta
        
        v = np.clip(v, -max_v, max_v)
        omega = np.clip(omega, -max_omega, max_omega)
        
        return np.array([v, omega], dtype=np.float32)
    
    return policy
```

#### 3.2 Добавить тест в конец файла

```python
def test_env():
    """Тест среды с номинальной политикой"""
    env = DifferentialDriveEnv()
    policy = move_to_point_policy()
    
    test_states = [
        [2.0, 0.0, 0.0],
        [0.0, 2.0, np.pi/2],
        [-1.0, -1.0, -np.pi/4],
    ]
    
    for init_state in test_states:
        state = env.reset(state=init_state)
        print(f"\nStart: {state}")
        
        for i in range(1000):
            action = policy(state)
            state, reward, done, info = env.step(action)
            
            if done:
                print(f"Done at step {i}: in_goal={info['in_goal']}")
                break

if __name__ == "__main__":
    test_env()
```

#### 3.3 Тест

```bash
py -3.12 RL/differential_drive_env.py
# Ожидание: агент из каждого состояния достигает цели
```

### Критерии успеха
- [ ] Политика реализована
- [ ] Агент достигает [0, 0, 0] из разных начальных состояний
- [ ] Траектории плавные

---

## Part 4: DifferentialDriveSystem

**Приоритет**: Высокий
**Статус**: Pending
**Файлы**: `physics/base_system.py`, рефакторинг `physics/point_system.py`, новый `physics/differential_drive_system.py`

### Задачи

#### 4.1 Создать physics/base_system.py

```python
# ursina/physics/base_system.py

from abc import ABC, abstractmethod
import numpy as np


class BaseDynamicalSystem(ABC):
    """Абстрактный базовый класс для физических систем."""
    
    def __init__(self, dt: float, initial_state: np.ndarray = None):
        self.dt = dt
        self._initial_state = initial_state if initial_state is not None else self.default_state()
        self.state = self._initial_state.copy()
        self.u = None  # Внешнее управление
    
    @property
    @abstractmethod
    def state_dim(self) -> int:
        pass
    
    @abstractmethod
    def default_state(self) -> np.ndarray:
        pass
    
    @abstractmethod
    def compute_derivative(self, state: np.ndarray = None) -> np.ndarray:
        pass
    
    @abstractmethod
    def generate_random_state(self) -> np.ndarray:
        pass
    
    @abstractmethod
    def state_to_position_3d(self, state: np.ndarray = None) -> tuple:
        """Преобразование в (x, y, z) для Ursina"""
        pass
    
    def get_state(self) -> np.ndarray:
        return self.state.copy()
    
    def set_state(self, state: np.ndarray):
        self.state = np.array(state, dtype=np.float32).copy()
    
    def reset_state(self):
        self.state = self._initial_state.copy()
    
    def get_initial_state(self) -> np.ndarray:
        return self._initial_state.copy()
    
    def step(self):
        derivative = self.compute_derivative()
        self.state = self.state + derivative * self.dt
```

#### 4.2 Рефакторинг physics/point_system.py

**Добавить**:
- Импорт и наследование от BaseDynamicalSystem
- @property для state_dim
- Методы default_state(), generate_random_state(), state_to_position_3d()

**Убрать** дублирующие методы (уже в базовом классе).

#### 4.3 Создать physics/differential_drive_system.py

```python
# ursina/physics/differential_drive_system.py

import numpy as np
from .base_system import BaseDynamicalSystem


class DifferentialDriveSystem(BaseDynamicalSystem):
    """
    Дифференциальный привод для визуализации.
    State: [x, y, θ], Control: [v, ω]
    """
    
    def __init__(self, dt: float, initial_state: np.ndarray = None,
                 max_v: float = 1.0, max_omega: float = 2.0):
        self.max_v = max_v
        self.max_omega = max_omega
        super().__init__(dt, initial_state)
    
    @property
    def state_dim(self) -> int:
        return 3
    
    def default_state(self) -> np.ndarray:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    
    def generate_random_state(self) -> np.ndarray:
        return np.array([
            np.random.uniform(-2.0, 2.0),
            np.random.uniform(-2.0, 2.0),
            np.random.uniform(-np.pi, np.pi),
        ], dtype=np.float32)
    
    def state_to_position_3d(self, state: np.ndarray = None) -> tuple:
        if state is None:
            state = self.state
        x, y, theta = state
        return (x, 0.0, y)  # x→X, y→Z в Ursina
    
    def get_orientation(self, state: np.ndarray = None) -> float:
        if state is None:
            state = self.state
        return state[2]
    
    def compute_derivative(self, state: np.ndarray = None) -> np.ndarray:
        if state is None:
            state = self.state
        x, y, theta = state
        
        if self.u is not None:
            u = np.asarray(self.u, dtype=np.float32).flatten()
            v = np.clip(u[0], -self.max_v, self.max_v)
            omega = np.clip(u[1], -self.max_omega, self.max_omega)
        else:
            v, omega = 0.0, 0.0
        
        return np.array([v * np.cos(theta), v * np.sin(theta), omega], dtype=np.float32)
    
    def step(self):
        derivative = self.compute_derivative()
        self.state = self.state + derivative * self.dt
        # Нормализация угла
        while self.state[2] > np.pi:
            self.state[2] -= 2 * np.pi
        while self.state[2] < -np.pi:
            self.state[2] += 2 * np.pi
```

#### 4.4 Обновить physics/__init__.py

```python
from .base_system import BaseDynamicalSystem
from .point_system import PointSystem
from .differential_drive_system import DifferentialDriveSystem
# ... остальные импорты
```

### Критерии успеха
- [ ] BaseDynamicalSystem создан
- [ ] PointSystem рефакторинг завершён
- [ ] DifferentialDriveSystem создан
- [ ] Point Mass ещё работает (регрессия)

---

## Part 5: Визуализация (Cone)

**Приоритет**: Высокий
**Статус**: Pending
**Файл**: `visuals/oriented_agent.py`

### Задачи

#### 5.1 Создать visuals/oriented_agent.py

```python
# ursina/visuals/oriented_agent.py

from ursina import Entity, Vec3, color
import numpy as np


class OrientedAgent(Entity):
    """
    Визуальный агент с ориентацией (cone).
    """
    
    def __init__(self, position=(0,0,0), orientation=0.0, 
                 scale=0.15, agent_color=None, **kwargs):
        super().__init__(
            model='cone',
            color=agent_color or color.orange,
            scale=(scale, scale * 1.5, scale),
            position=position,
            **kwargs
        )
        
        # Cone по умолчанию смотрит вверх (Y+)
        # Поворачиваем чтобы смотрел вдоль Z+
        self.rotation_x = 90
        
        self._orientation = orientation
        self._update_rotation()
    
    def _update_rotation(self):
        angle_deg = np.degrees(self._orientation)
        self.rotation_y = -angle_deg
    
    def set_orientation(self, theta: float):
        self._orientation = theta
        self._update_rotation()
    
    def update_from_state(self, state: np.ndarray, height: float = 0.1):
        """Обновить из состояния [x, y, θ]"""
        x, y, theta = state[0], state[1], state[2]
        self.position = Vec3(x, height, y)  # x→X, y→Z
        self.set_orientation(theta)
```

#### 5.2 Обновить visuals/__init__.py

```python
from .oriented_agent import OrientedAgent
# ... добавить в __all__
```

### Критерии успеха
- [ ] Cone отображается
- [ ] Ориентация θ корректно отображается
- [ ] rotation_x = 90 (cone смотрит вдоль Z)

---

## Part 6: Интеграция (VectorizedEnv, Config)

**Приоритет**: Средний
**Статус**: Pending
**Файлы**: `physics/vectorized_env.py`, `config/training_config.py`, `RL/__init__.py`

### Задачи

#### 6.1 VectorizedEnv: добавить system_type

В `__init__` добавить параметр:
```python
system_type: str = 'point_mass'  # или 'differential_drive'
```

В цикле создания систем:
```python
if system_type == 'point_mass':
    system = PointSystem(dt=dt, initial_state=init_state)
elif system_type == 'differential_drive':
    system = DifferentialDriveSystem(dt=dt, initial_state=init_state)
```

#### 6.2 TrainingConfig: добавить параметры

```python
# После boundary_limit добавить:
system_type: str = 'point_mass'
max_v: float = 1.0
max_omega: float = 2.0
goal_angle_tolerance: float = 0.1
```

#### 6.3 Создать RL/__init__.py

```python
# RL/__init__.py

from .base_env import BaseEnv
from .simple_env import PointMassEnv, pd_nominal_policy
from .differential_drive_env import DifferentialDriveEnv, move_to_point_policy
from .td3 import TD3, ReplayBuffer
from .calf import CALFController

__all__ = [
    'BaseEnv', 'PointMassEnv', 'pd_nominal_policy',
    'DifferentialDriveEnv', 'move_to_point_policy',
    'TD3', 'ReplayBuffer', 'CALFController',
]
```

### Критерии успеха
- [ ] VectorizedEnv поддерживает system_type
- [ ] TrainingConfig содержит новые параметры
- [ ] RL/__init__.py создан

---

## Part 7: Интеграция (main.py, Heatmap)

**Приоритет**: Средний
**Статус**: Pending
**Файлы**: `main.py`, `visuals/critic_heatmap.py`

### Задачи

#### 7.1 main.py: добавить argparse

В начало после импортов:
```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--system', type=str, default='point_mass',
                    choices=['point_mass', 'differential_drive'])
args = parser.parse_args()
```

#### 7.2 main.py: условный выбор среды

```python
if args.system == 'point_mass':
    from RL.simple_env import PointMassEnv, pd_nominal_policy
    env = PointMassEnv(dt=0.01, max_action=5.0, goal_radius=config.training.goal_epsilon)
    nominal_policy = pd_nominal_policy(max_action=env.max_action)
elif args.system == 'differential_drive':
    from RL.differential_drive_env import DifferentialDriveEnv, move_to_point_policy
    env = DifferentialDriveEnv(
        dt=0.01, max_v=config.training.max_v,
        max_omega=config.training.max_omega,
        goal_radius=config.training.goal_epsilon
    )
    nominal_policy = move_to_point_policy(max_v=config.training.max_v, max_omega=config.training.max_omega)
```

#### 7.3 CriticHeatmap: поддержка 3D состояния

В `__init__` добавить:
```python
state_dim: int = 2
action_dim: int = 1
fixed_theta: float = 0.0
```

В `_compute_q_values` изменить создание состояний:
```python
if self.state_dim == 2:
    states = np.stack([self.x_grid.flatten(), self.v_grid.flatten()], axis=1)
elif self.state_dim == 3:
    n_points = self.x_grid.size
    states = np.stack([
        self.x_grid.flatten(),
        self.v_grid.flatten(),
        np.full(n_points, self.fixed_theta)
    ], axis=1)
```

### Критерии успеха
- [ ] `py -3.12 main.py` работает (по умолчанию point_mass)
- [ ] `py -3.12 main.py --system differential_drive` работает
- [ ] Heatmap показывает Q-values при θ=0

---

## Part 8: Тестирование

**Приоритет**: Обязательно
**Статус**: Pending

### Тесты

#### 8.1 Регрессия Point Mass
```bash
py -3.12 main.py
# Ожидание: работает как раньше
```

#### 8.2 Differential Drive визуализация
```bash
py -3.12 main.py --system differential_drive
# Ожидание: cone агенты, heatmap, обучение работает
```

#### 8.3 Обучение сходится
Запустить на 100+ эпизодов, убедиться что reward растёт.

### Критерии успеха
- [ ] Point Mass не сломан
- [ ] Differential Drive запускается
- [ ] Визуализация корректна (cone с ориентацией)
- [ ] Обучение сходится

---

## Порядок выполнения

```
Part 1 (BaseEnv)
    │
    ├── Part 2 (DifferentialDriveEnv)
    │       │
    │       └── Part 3 (Номинальная политика)
    │
    └── Part 4 (DifferentialDriveSystem)
            │
            └── Part 5 (OrientedAgent)

        ↓ После Part 1-5 ↓

Part 6 (VectorizedEnv, Config)
    │
    └── Part 7 (main.py, Heatmap)
            │
            └── Part 8 (Тестирование)
```

---

**Начинай с Part 1!**
