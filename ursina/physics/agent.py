"""
Agent - агент с встроенной траекторией и визуализацией
========================================================

Каждый агент владеет:
- Физической моделью (PointSystem)
- Визуальным представлением (Entity шарик)
- Траекторией с кольцевым буфером (LineTrail или PointTrail)

Траектория интегрирована в агента - при каждом шаге агент
автоматически обновляет свою траекторию.
"""

import numpy as np
from ursina import Entity, Vec3, Vec4
from .point_system import PointSystem


class Agent:
    """
    Агент с встроенной траекторией и визуализацией.

    Принцип работы траектории:
    1. При создании агента создаётся пул точек для траектории (кольцевой буфер)
    2. При каждом шаге агент запоминает свою позицию
    3. Самая старая точка переиспользуется для отрисовки новой позиции
    4. Никаких destroy/create - только перемещение существующих объектов

    Features:
    - Физика: PointSystem для математики
    - Визуализация: Entity (шарик агента)
    - Траектория: кольцевой буфер с переиспользованием объектов
    - Автоматическое обновление траектории при шаге
    """

    # Цвета режимов для CALF
    MODE_COLORS = {
        'td3': Vec4(0.2, 0.4, 1.0, 1),      # Синий
        'relax': Vec4(0.2, 0.7, 0.3, 1),    # Зеленый
        'fallback': Vec4(1.0, 0.5, 0.1, 1)  # Оранжевый
    }

    def __init__(
        self,
        point_system: PointSystem,
        object_manager,
        name: str,
        initial_position: tuple,
        color: Vec4,
        offset: tuple = (0, 0, 0),
        trail_config: dict = None
    ):
        """
        Инициализация агента.

        Parameters:
        -----------
        point_system : PointSystem
            Физическая модель агента
        object_manager : ObjectManager
            Менеджер для создания визуальных объектов
        name : str
            Имя агента (для идентификации)
        initial_position : tuple
            Начальная 3D позиция (x, y, z)
        color : Vec4
            Цвет агента
        offset : tuple
            Смещение для группировки агентов (x, y, z)
        trail_config : dict, optional
            Конфигурация траектории:
            {
                'max_length': int,      # Размер кольцевого буфера
                'decimation': int,      # Каждую N-ую точку добавлять
                'point_size': float,    # Размер точек траектории
                'line_thickness': float # Толщина линии
            }
        """
        self.point_system = point_system
        self.object_manager = object_manager
        self.name = name
        self.offset = np.array(offset, dtype=float)
        self.color = color

        # Конфигурация траектории
        if trail_config is None:
            trail_config = {
                'max_length': 100,
                'decimation': 3,
                'point_size': 0.03
            }
        self.trail_config = trail_config

        # Визуальное представление агента (шарик)
        self.visual = object_manager.create_object(
            name=name,
            model='sphere',
            position=initial_position,
            scale=0.1,
            color_val=color
        )

        # КОЛЬЦЕВОЙ БУФЕР для траектории
        # Создаём пул точек один раз, потом только переиспользуем
        self.trail_points = []
        max_length = trail_config.get('max_length', 100)
        point_size = trail_config.get('point_size', 0.03)

        for i in range(max_length):
            point = object_manager.create_object(
                name=f'{name}_trail_{i}',
                model='sphere',
                position=(0, -1000, 0),  # Далеко за границами видимости
                scale=point_size,
                color_val=color
            )
            point.visible = False  # Скрываем до первого использования
            self.trail_points.append(point)

        # Индексы кольцевого буфера
        self.trail_head = 0  # Куда писать следующую точку
        self.trail_count = 0  # Сколько точек активно

        # Хранение реальных позиций (без зума) для траектории
        self.trail_positions = [None] * max_length

        # Счётчик для decimation
        self.step_counter = 0
        self.decimation = trail_config.get('decimation', 3)

        # Текущий режим (для CALF)
        self.current_mode = 'td3'

    def update_position(self, state: np.ndarray, mode: str = 'td3'):
        """
        Обновить позицию агента и траекторию.

        Parameters:
        -----------
        state : np.ndarray
            Состояние [x, v] из PointSystem
        mode : str
            Текущий режим: 'td3', 'relax', 'fallback'
        """
        # Извлекаем позицию из состояния
        x, v = state[0], state[1]

        # Вычисляем 3D позицию с учётом offset
        position_3d = (
            x + self.offset[0],
            0.1 + self.offset[1],
            v + self.offset[2]
        )

        # Обновляем визуальное представление агента
        self.visual.position = position_3d

        # Обновляем цвет в зависимости от режима (для CALF)
        self.current_mode = mode
        if mode in self.MODE_COLORS:
            self.visual.color = self.MODE_COLORS[mode]

        # Обновляем траекторию (с decimation)
        self._add_trail_point(position_3d, mode)

    def _add_trail_point(self, position: tuple, mode: str):
        """
        Добавить точку в траекторию (кольцевой буфер).

        Parameters:
        -----------
        position : tuple
            3D позиция (x, y, z)
        mode : str
            Режим для цвета точки
        """
        self.step_counter += 1

        # Decimation: добавляем только каждую N-ую точку
        if self.step_counter % self.decimation != 0:
            return

        # Сохраняем реальную позицию
        self.trail_positions[self.trail_head] = np.array(position)

        # Переиспользуем самую старую точку
        point = self.trail_points[self.trail_head]
        point.position = position
        point.color = self.MODE_COLORS.get(mode, self.MODE_COLORS['td3'])
        point.visible = True

        # Сдвигаем head (кольцевой буфер)
        self.trail_head = (self.trail_head + 1) % len(self.trail_points)

        # Увеличиваем count до максимума
        if self.trail_count < len(self.trail_points):
            self.trail_count += 1

    def clear_trail(self):
        """Очистить траекторию (скрыть все точки)."""
        for point in self.trail_points:
            point.visible = False
            point.position = (0, -1000, 0)  # Далеко за границами

        self.trail_head = 0
        self.trail_count = 0
        self.step_counter = 0
        self.trail_positions = [None] * len(self.trail_points)

    def step(self, action: float):
        """
        Выполнить шаг агента: применить действие, обновить физику, обновить визуализацию.

        Parameters:
        -----------
        action : float
            Управляющее воздействие
        """
        # Применяем действие к физике
        self.point_system.u = action

        # Делаем шаг интегрирования
        self.point_system.step()

        # Обновляем визуализацию (без mode, будет 'td3' по умолчанию)
        state = self.point_system.get_state()
        self.update_position(state, mode=self.current_mode)

    def get_state(self) -> np.ndarray:
        """Получить текущее состояние физики."""
        return self.point_system.get_state()

    def reset(self, new_state: np.ndarray = None):
        """
        Сбросить агента в новое состояние.

        Parameters:
        -----------
        new_state : np.ndarray, optional
            Новое состояние [x, v]. Если None, случайное.
        """
        if new_state is None:
            new_state = np.array([
                np.random.uniform(-2.0, 2.0),  # x
                np.random.uniform(-0.5, 0.5)   # v
            ], dtype=np.float32)

        self.point_system.state = new_state.copy()
        self.clear_trail()

        # Обновляем визуализацию
        self.update_position(new_state, mode=self.current_mode)

    def apply_zoom_transform(self, a: float, b: np.ndarray):
        """
        Применить трансформацию зума к траектории.

        Parameters:
        -----------
        a : float
            Масштаб
        b : np.ndarray
            Смещение [x, y, z]
        """
        # Обновляем позиции всех видимых точек траектории
        for i in range(len(self.trail_points)):
            if self.trail_positions[i] is not None and self.trail_points[i].visible:
                real_pos = self.trail_positions[i]
                transformed = real_pos * a + b
                self.trail_points[i].position = Vec3(transformed[0], transformed[1], transformed[2])

    def destroy(self):
        """Уничтожить агента и его траекторию."""
        # Визуальное представление будет уничтожено через ObjectManager
        # Траектория - тоже через ObjectManager (все точки созданы через него)
        pass

    def __repr__(self):
        return f"Agent(name={self.name}, state={self.get_state()}, mode={self.current_mode})"
