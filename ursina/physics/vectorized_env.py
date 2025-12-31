"""
Vectorized Environment - Векторизованная среда для параллельных симуляций
==========================================================================

Управляет N параллельными симуляциями PointSystem.
Позволяет эффективно обрабатывать множество агентов через batch операции.

Phase 4: Multi-agent vectorization

ОБНОВЛЕНИЕ: Теперь поддерживает создание Agent объектов с встроенными траекториями.
Каждый Agent владеет своей траекторией (кольцевой буфер) и автоматически обновляет её.
"""

import numpy as np
from typing import List, Dict, Optional, Any, Callable
from ursina import Vec4
from .point_system import PointSystem
from .policies.base_policy import Policy
from .agent import Agent


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
        seed: Optional[int] = None,
        object_manager = None,
        group_name: str = "agents",
        offset: tuple = (0, 0, 0),
        color: Vec4 = None,
        trail_config: dict = None,
        create_agents: bool = False
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
        object_manager : ObjectManager, optional
            Менеджер для создания визуальных объектов (если create_agents=True)
        group_name : str
            Имя группы агентов (для идентификации)
        offset : tuple
            Смещение группы агентов в 3D пространстве (x, y, z)
        color : Vec4, optional
            Цвет агентов (если None, используется синий)
        trail_config : dict, optional
            Конфигурация траектории для агентов
        create_agents : bool
            Если True, создаёт Agent объекты с визуализацией и траекториями
            Если False, работает в старом режиме (только PointSystem)
        """
        self.n_envs = n_envs
        self.policy = policy
        self.dt = dt
        self.group_name = group_name
        self.offset = np.array(offset, dtype=float)
        self.create_agents = create_agents
        self.object_manager = object_manager

        # Цвет по умолчанию
        if color is None:
            color = Vec4(0.2, 0.3, 0.8, 1)  # Синий
        self.color = color

        # Seed для воспроизводимости
        if seed is not None:
            np.random.seed(seed)

        # Создаём N сред
        self.envs: List[PointSystem] = []
        self.agents: List[Agent] = []  # Список Agent объектов (если create_agents=True)

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
            point_system = PointSystem(
                dt=dt,
                initial_state=init_state,
                controller=None  # Контроллер не нужен, управляем через политику
            )
            self.envs.append(point_system)

            # Если включено создание агентов
            if create_agents and object_manager is not None:
                # Вычисляем начальную 3D позицию
                x, v = init_state[0], init_state[1]
                initial_position = (
                    x + self.offset[0],
                    0.1 + self.offset[1],
                    v + self.offset[2]
                )

                # Создаём Agent
                agent = Agent(
                    point_system=point_system,
                    object_manager=object_manager,
                    name=f'{group_name}_{i}',
                    initial_position=initial_position,
                    color=color,
                    offset=offset,
                    trail_config=trail_config
                )
                self.agents.append(agent)

        # Текущие состояния всех сред, shape (n_envs, state_dim)
        self.states = self._get_all_states()

        if create_agents:
            print(f"[OK] VectorizedEnvironment created: {n_envs} agents with trails, policy={policy.__class__.__name__}")
        else:
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
        5. Если есть агенты - обновить их визуализацию и траектории

        Returns:
        --------
        states : np.ndarray
            Новые состояния всех сред, shape (n_envs, state_dim)
        """
        # 1. Получаем действия для всех сред через batch метод политики
        actions = self.policy.get_actions_batch(self.states)

        # 2. Применяем действия и делаем шаг для каждой среды
        if self.create_agents and self.agents:
            # Режим с агентами: обновляем через Agent.step()
            for i, agent in enumerate(self.agents):
                action = actions[i]

                # Применяем действие к физике
                agent.point_system.u = action

                # Делаем шаг интегрирования
                agent.point_system.step()

                # Получаем режим (для CALF политики)
                mode = 'td3'  # По умолчанию
                if hasattr(self.policy, 'get_mode_for_env'):
                    mode = self.policy.get_mode_for_env(i)

                # Обновляем визуализацию и траекторию
                state = agent.point_system.get_state()
                agent.update_position(state, mode=mode)
        else:
            # Старый режим: только PointSystem
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
    
    def reset_agent(self, agent_idx: int, new_state: np.ndarray = None):
        """
        Сбросить конкретного агента в новое состояние.

        Parameters:
        -----------
        agent_idx : int
            Индекс агента
        new_state : np.ndarray, optional
            Новое состояние [x, v]. Если None, случайное.
        """
        if new_state is None:
            new_state = np.array([
                np.random.uniform(-2.0, 2.0),  # x
                np.random.uniform(-0.5, 0.5)   # v
            ], dtype=np.float32)

        # Обновляем состояние PointSystem
        self.envs[agent_idx].state = new_state.copy()

        # Если есть агент - обновляем его
        if self.create_agents and agent_idx < len(self.agents):
            self.agents[agent_idx].reset(new_state)

        # Обновляем кэш состояний
        self.states[agent_idx] = new_state.copy()

    def clear_agent_trail(self, agent_idx: int):
        """Очистить траекторию конкретного агента."""
        if self.create_agents and agent_idx < len(self.agents):
            self.agents[agent_idx].clear_trail()

    def clear_all_trails(self):
        """Очистить траектории всех агентов."""
        if self.create_agents:
            for agent in self.agents:
                agent.clear_trail()

    def apply_zoom_transform(self, a: float, b: np.ndarray):
        """
        Применить трансформацию зума ко всем агентам.

        Parameters:
        -----------
        a : float
            Масштаб
        b : np.ndarray
            Смещение [x, y, z]
        """
        if self.create_agents:
            for agent in self.agents:
                agent.apply_zoom_transform(a, b)

    def print_stats(self):
        """Print vectorized environment statistics."""
        positions = self.get_positions()
        velocities = self.get_velocities()

        print("\n--- VectorizedEnvironment Stats ---")
        print(f"  Number of envs: {self.n_envs}")
        print(f"  Policy: {self.policy.__class__.__name__}")
        print(f"  Time step: {self.dt}")
        if self.create_agents:
            print(f"  Agents with trails: {len(self.agents)}")
        print(f"  Positions (x):")
        print(f"    min={positions.min():.3f}, max={positions.max():.3f}, mean={positions.mean():.3f}")
        print(f"  Velocities (v):")
        print(f"    min={velocities.min():.3f}, max={velocities.max():.3f}, mean={velocities.mean():.3f}")
        print("-----------------------------------")
