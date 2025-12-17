"""
Vectorized Environment - Векторизованная среда для параллельных симуляций
==========================================================================

Управляет N параллельными симуляциями PointSystem.
Позволяет эффективно обрабатывать множество агентов через batch операции.

Phase 4: Multi-agent vectorization
"""

import numpy as np
from typing import List, Dict, Optional, Any, Callable
from .point_system import PointSystem
from .policies.base_policy import Policy


class VectorizedEnvironment:
    """
    N параллельных симуляций PointSystem с единой политикой.
    
    Преимущества:
    - Batch вычисление действий через policy.get_actions_batch()
    - Единое хранилище состояний для всех агентов
    - Эффективная обработка множества сред
    
    Использование:
        policy = PDPolicy(kp=1.0, kd=0.5)
        vec_env = VectorizedEnvironment(
            n_envs=10,
            policy=policy,
            dt=0.01,
            initial_states=None  # Случайные начальные состояния
        )
        
        # Сброс всех сред
        states = vec_env.reset()
        
        # Шаг всех сред
        for _ in range(100):
            states = vec_env.step()
    """
    
    def __init__(
        self,
        n_envs: int,
        policy: Policy,
        dt: float = 0.01,
        initial_states: Optional[np.ndarray] = None,
        seed: Optional[int] = None
    ):
        """
        Инициализация векторизованной среды.
        
        Parameters:
        -----------
        n_envs : int
            Количество параллельных сред
        policy : Policy
            Единая политика для всех агентов (будет использовать get_actions_batch)
        dt : float
            Шаг интегрирования для физики
        initial_states : np.ndarray, optional
            Начальные состояния, shape (n_envs, state_dim)
            Если None, используются случайные состояния
        seed : int, optional
            Seed для генерации случайных начальных состояний
        """
        self.n_envs = n_envs
        self.policy = policy
        self.dt = dt
        
        # Seed для воспроизводимости
        if seed is not None:
            np.random.seed(seed)
        
        # Создаём N сред (PointSystem без контроллера, т.к. используем политику)
        self.envs: List[PointSystem] = []
        
        for i in range(n_envs):
            # Начальное состояние для этой среды
            if initial_states is not None:
                init_state = initial_states[i]
            else:
                # Случайное начальное состояние: x в [-2, 2], v в [-0.5, 0.5]
                init_state = np.array([
                    np.random.uniform(-2.0, 2.0),  # x
                    np.random.uniform(-0.5, 0.5)   # v
                ], dtype=np.float32)
            
            # Создаём PointSystem без контроллера (управляется через политику)
            env = PointSystem(
                dt=dt,
                initial_state=init_state,
                controller=None  # Контроллер не нужен, управляем через политику
            )
            self.envs.append(env)
        
        # Текущие состояния всех сред, shape (n_envs, state_dim)
        self.states = self._get_all_states()
        
        print(f"[OK] VectorizedEnvironment created: {n_envs} envs, policy={policy.__class__.__name__}")
    
    def _get_all_states(self) -> np.ndarray:
        """Собрать состояния всех сред в один массив."""
        return np.array([env.get_state() for env in self.envs])
    
    def reset(self, initial_states: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Сброс всех сред в начальное состояние.
        
        Parameters:
        -----------
        initial_states : np.ndarray, optional
            Новые начальные состояния, shape (n_envs, state_dim)
            Если None, используются случайные состояния
        
        Returns:
        --------
        states : np.ndarray
            Начальные состояния всех сред, shape (n_envs, state_dim)
        """
        for i, env in enumerate(self.envs):
            if initial_states is not None:
                new_state = initial_states[i]
            else:
                # Случайное начальное состояние
                new_state = np.array([
                    np.random.uniform(-2.0, 2.0),  # x
                    np.random.uniform(-0.5, 0.5)   # v
                ], dtype=np.float32)
            
            # Устанавливаем новое состояние
            env.state = new_state.copy()
        
        # Обновляем кэш состояний
        self.states = self._get_all_states()
        
        # Сбрасываем политику, если есть метод reset
        if hasattr(self.policy, 'reset'):
            self.policy.reset()
        
        return self.states.copy()
    
    def step(self) -> np.ndarray:
        """
        Выполнить один шаг симуляции для всех сред.
        
        Процесс:
        1. Получить действия от политики (batch)
        2. Применить действия к каждой среде
        3. Выполнить шаг интегрирования
        4. Обновить состояния
        
        Returns:
        --------
        states : np.ndarray
            Новые состояния всех сред, shape (n_envs, state_dim)
        """
        # 1. Получаем действия для всех сред через batch метод политики
        actions = self.policy.get_actions_batch(self.states)
        
        # 2. Применяем действия и делаем шаг для каждой среды
        for i, env in enumerate(self.envs):
            # Применяем действие
            env.u = actions[i]
            
            # Делаем шаг интегрирования
            env.step()
        
        # 3. Обновляем кэш состояний
        self.states = self._get_all_states()
        
        return self.states.copy()
    
    def get_states(self) -> np.ndarray:
        """
        Получить текущие состояния всех сред.
        
        Returns:
        --------
        states : np.ndarray
            Копия состояний всех сред, shape (n_envs, state_dim)
        """
        return self.states.copy()
    
    def get_positions(self) -> np.ndarray:
        """
        Получить только позиции всех агентов (без скоростей).
        
        Returns:
        --------
        positions : np.ndarray
            Позиции всех агентов, shape (n_envs,)
        """
        return self.states[:, 0]  # Первая компонента - позиция
    
    def get_velocities(self) -> np.ndarray:
        """
        Получить только скорости всех агентов.
        
        Returns:
        --------
        velocities : np.ndarray
            Скорости всех агентов, shape (n_envs,)
        """
        return self.states[:, 1]  # Вторая компонента - скорость
    
    def print_stats(self):
        """Print vectorized environment statistics."""
        positions = self.get_positions()
        velocities = self.get_velocities()
        
        print("\n--- VectorizedEnvironment Stats ---")
        print(f"  Number of envs: {self.n_envs}")
        print(f"  Policy: {self.policy.__class__.__name__}")
        print(f"  Time step: {self.dt}")
        print(f"  Positions (x):")
        print(f"    min={positions.min():.3f}, max={positions.max():.3f}, mean={positions.mean():.3f}")
        print(f"  Velocities (v):")
        print(f"    min={velocities.min():.3f}, max={velocities.max():.3f}, mean={velocities.mean():.3f}")
        print("-----------------------------------")
