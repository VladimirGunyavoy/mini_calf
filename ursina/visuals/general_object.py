"""
General Object - связывает математический и визуальный объекты
"""

import numpy as np
from typing import Any, Optional


class GeneralObject:
    """
    Объект, связывающий математический и визуальный объекты.
    
    Обновляет визуальный объект на основе состояния математического объекта.
    """
    
    def __init__(self, math_obj: Any, visual_obj: Any):
        """
        Инициализация GeneralObject
        
        Parameters:
        -----------
        math_obj : Any
            Математический объект (например, PointSystem)
        visual_obj : Any
            Визуальный объект (например, PointVisual)
        """
        self.math_obj = math_obj
        self.visual_obj = visual_obj
    
    def update_visual(self) -> None:
        """
        Обновить визуальный объект на основе состояния математического объекта.
        
        Извлекает позицию из состояния математического объекта и обновляет визуальный объект.
        """
        if self.math_obj is None or self.visual_obj is None:
            return
        
        # Получаем состояние математического объекта
        state = self.math_obj.get_state()
        
        # Для PointSystem состояние [x, v]
        # Используем x как координату по оси X, v как координату по оси Z
        # Это позволит визуализировать фазовую плоскость и получить движение по кругу
        if len(state) >= 2:
            x = state[0]
            v = state[1]
            # Создаем 3D позицию: (x, 0, v) для визуализации в плоскости XZ
            position = np.array([x, 0.0, v], dtype=np.float32)
        elif len(state) >= 1:
            x = state[0]
            # Если только одна координата, используем только x
            position = np.array([x, 0.0, 0.0], dtype=np.float32)
        else:
            position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        # Обновляем позицию визуального объекта
        if hasattr(self.visual_obj, 'set_position'):
            self.visual_obj.set_position(position)
        elif hasattr(self.visual_obj, 'position'):
            self.visual_obj.position = tuple(position)
