"""
Визуальный объект точки для отображения системы точки
"""

import numpy as np
from typing import Union, Tuple, Optional, TYPE_CHECKING
from utils.scalable import Scalable

if TYPE_CHECKING:
    from managers.zoom_manager import ZoomManager


class PointVisual(Scalable):
    """
    Визуальный объект точки для визуализации системы точки

    Наследуется от Scalable для поддержки масштабирования
    """

    def __init__(self, position: Union[Tuple[float, float, float], np.ndarray] = (0, 0, 0),
                 scale: Union[float, Tuple[float, float, float]] = 0.1,
                 color=None, **kwargs):
        """
        Инициализация визуальной точки

        Parameters:
        -----------
        position : tuple or np.ndarray
            Начальная позиция (x, y, z)
        scale : float or tuple
            Масштаб точки
        color : color
            Цвет точки
        **kwargs
            Дополнительные параметры для Scalable
        """
        # Если цвет не указан, используем красный по умолчанию
        if color is None:
            from ursina import color as ursina_color
            color = ursina_color.red

        # Создаем сферу для визуализации точки
        super().__init__(
            model='sphere',
            position=position,
            scale=scale,
            color=color,
            **kwargs
        )

        # Ссылка на ZoomManager (устанавливается при регистрации)
        self._zoom_manager: Optional['ZoomManager'] = None
    
    def set_position(self, position: Union[Tuple[float, float, float], np.ndarray]) -> None:
        """
        Установить позицию точки

        Обновляет real_position и сразу применяет текущую трансформацию.
        Это важно для корректной работы зума с движущимися объектами.

        Parameters:
        -----------
        position : tuple or np.ndarray
            Новая позиция (x, y, z). Если передано 2D (x, z), y будет установлен в 0
        """
        # Преобразуем в numpy array для единообразия
        if isinstance(position, (tuple, list)):
            position = np.array(position, dtype=np.float32)
        else:
            position = np.array(position, dtype=np.float32)
            # raise ValueError(f"Position must be a tuple or np.ndarray, got {type(position)}")

        # Если передана 2D позиция, добавляем y=0
        if position.shape == (2,):
            position = np.array([position[0], 0.0, position[1]], dtype=np.float32)
        elif position.shape != (3,):
            raise ValueError(f"Position must be 2D or 3D, got shape {position.shape}")

        # Обновляем реальную позицию
        self.real_position = position

        # ВАЖНО: Применяем текущую трансформацию СРАЗУ!
        # Это гарантирует, что зум работает правильно для движущихся объектов
        if self._zoom_manager is not None:
            # Если есть ссылка на ZoomManager, используем его трансформации
            self.apply_transform(
                self._zoom_manager.a_transformation,
                self._zoom_manager.b_translation
            )
        else:
            # Если нет ZoomManager, просто устанавливаем позицию напрямую
            # (это будет работать как раньше для незарегистрированных объектов)
            self.position = tuple(position)

    def set_zoom_manager(self, zoom_manager: 'ZoomManager') -> None:
        """
        Установить ссылку на ZoomManager для корректного применения трансформаций.

        Этот метод должен быть вызван при регистрации объекта в ZoomManager.

        Parameters:
        -----------
        zoom_manager : ZoomManager
            Менеджер масштабирования
        """
        self._zoom_manager = zoom_manager