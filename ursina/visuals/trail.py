"""
Simple Trail Visualization
Одноцветная траектория для визуализации движения объектов
"""

from ursina import Entity, destroy, Vec3, Mesh, color
import numpy as np


class SimpleTrail:
    """
    Простая одноцветная траектория
    
    Features:
    - Максимальная длина (старые точки удаляются)
    - Decimation (скважность) - добавлять каждую N-ую точку
    - Цвет траектории
    - Очистка траектории
    """
    
    def __init__(self, trail_color=None, max_length=200, decimation=1, rebuild_frequency=5):
        """
        Parameters:
        -----------
        trail_color : ursina.color
            Цвет траектории (по умолчанию красный)
        max_length : int
            Максимальное количество точек в траектории
        decimation : int
            Каждая N-ая точка добавляется (1 = все точки, 2 = каждая вторая)
        rebuild_frequency : int
            Перестраивать mesh каждые N добавлений (для производительности)
        """
        self.trail_color = trail_color if trail_color is not None else color.red
        self.max_length = max_length
        self.decimation = decimation
        self.rebuild_frequency = rebuild_frequency
        
        # Хранение точек
        self.positions = []
        
        # Entity для визуализации
        self.trail_entity = None
        
        # Счетчик для decimation
        self.step_counter = 0
        
        # Счетчик для rebuild
        self.rebuild_counter = 0
        self.needs_rebuild = False
    
    def add_point(self, position):
        """
        Добавить точку в траекторию
        
        Parameters:
        -----------
        position : tuple/list/Vec3
            3D позиция точки [x, y, z] или Vec3
        """
        self.step_counter += 1
        
        # Decimation: добавляем только каждую N-ую точку
        if self.step_counter % self.decimation != 0:
            return
        
        # Конвертируем в Vec3
        if not isinstance(position, Vec3):
            position = Vec3(*position)
        
        # Добавляем точку
        self.positions.append(position)
        
        # Ограничение длины
        if len(self.positions) > self.max_length:
            self.positions.pop(0)
        
        # Отметить, что нужна перестройка
        self.needs_rebuild = True
        self.rebuild_counter += 1
        
        # Перестраивать только каждые N добавлений (для производительности)
        if self.rebuild_counter >= self.rebuild_frequency:
            self.rebuild()
            self.rebuild_counter = 0
    
    def rebuild(self):
        """
        Перестроить визуализацию траектории
        """
        # Удаляем старую визуализацию
        if self.trail_entity:
            destroy(self.trail_entity)
            self.trail_entity = None
        
        # Нужно минимум 2 точки для линии
        if len(self.positions) >= 2:
            # Простой способ: mode='line'
            self.trail_entity = Entity(
                model=Mesh(vertices=self.positions, mode='line', thickness=3),
                color=self.trail_color,
                alpha=0.8
            )
    
    def clear(self):
        """
        Очистить траекторию
        """
        self.positions = []
        self.step_counter = 0
        self.rebuild_counter = 0
        self.needs_rebuild = False
        
        if self.trail_entity:
            destroy(self.trail_entity)
            self.trail_entity = None
    
    def __del__(self):
        """Очистка при удалении объекта"""
        self.clear()
