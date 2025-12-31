"""
Point-based Trail Visualization
Траектория на основе шариков с кольцевым буфером (эффективное переиспользование)
"""

from ursina import Entity, Vec3, Vec4
import numpy as np


class PointTrail:
    """
    Траектория из шариков с кольцевым буфером.
    
    Принцип работы:
    - Фиксированный пул шариков создаётся один раз
    - При добавлении новой точки, самый старый шарик перемещается на новую позицию
    - Никаких destroy/create - только перемещение существующих Entity
    
    Цвета режимов CALF:
    - TD3: синий
    - Relax: зеленый  
    - Fallback: оранжевый
    """
    
    # Цвета режимов
    MODE_COLORS = {
        'td3': Vec4(0.2, 0.4, 1.0, 1),      # Синий
        'relax': Vec4(0.2, 0.7, 0.3, 1),    # Зеленый
        'fallback': Vec4(1.0, 0.5, 0.1, 1)  # Оранжевый
    }
    
    def __init__(self, max_points=100, point_size=0.03, decimation=3):
        """
        Parameters:
        -----------
        max_points : int
            Максимальное количество точек в траектории (размер пула)
        point_size : float
            Размер шариков
        decimation : int
            Добавлять каждую N-ую точку (1 = все, 3 = каждую третью)
        """
        self.max_points = max_points
        self.point_size = point_size
        self.decimation = decimation
        
        # Кольцевой буфер - пул шариков
        self.points = []
        for _ in range(max_points):
            point = Entity(
                model='sphere',
                scale=point_size,
                color=self.MODE_COLORS['td3'],
                visible=False,
                unlit=True
            )
            self.points.append(point)
        
        # Индексы кольцевого буфера
        self.head = 0  # Куда писать следующую точку
        self.count = 0  # Сколько точек активно
        
        # Счётчик для decimation
        self.step_counter = 0
        
        # Хранение реальных позиций (без зума)
        self.real_positions = [None] * max_points
        
        # Трансформации зума
        self.a_transformation = 1.0
        self.b_translation = np.array([0, 0, 0], dtype=float)
        self.zoom_manager = None
    
    def add_point(self, position, mode='td3'):
        """
        Добавить точку в траекторию.
        
        Parameters:
        -----------
        position : tuple/list/Vec3
            3D позиция [x, y, z]
        mode : str
            Режим: 'td3', 'relax', 'fallback'
        """
        self.step_counter += 1
        
        # Decimation
        if self.step_counter % self.decimation != 0:
            return
        
        # Конвертируем позицию
        if isinstance(position, Vec3):
            real_pos = np.array([position.x, position.y, position.z])
        else:
            real_pos = np.array(position)
        
        # Сохраняем реальную позицию
        self.real_positions[self.head] = real_pos
        
        # Применяем трансформацию зума
        transformed = real_pos * self.a_transformation + self.b_translation
        
        # Перемещаем шарик на новую позицию
        point = self.points[self.head]
        point.position = Vec3(transformed[0], transformed[1], transformed[2])
        point.color = self.MODE_COLORS.get(mode, self.MODE_COLORS['td3'])
        point.visible = True
        
        # Сдвигаем head (кольцевой буфер)
        self.head = (self.head + 1) % self.max_points
        
        # Увеличиваем count до max_points
        if self.count < self.max_points:
            self.count += 1
    
    def clear(self):
        """Очистить траекторию (скрыть все точки)"""
        for point in self.points:
            point.visible = False
        self.head = 0
        self.count = 0
        self.step_counter = 0
        self.real_positions = [None] * self.max_points
    
    def apply_transform(self, a, b, **kwargs):
        """
        Применить трансформацию зума ко всем точкам.
        
        Parameters:
        -----------
        a : float
            Масштаб
        b : np.ndarray
            Смещение [x, y, z]
        """
        self.a_transformation = a
        self.b_translation = b
        
        # Обновляем позиции всех видимых точек
        for i in range(self.max_points):
            if self.real_positions[i] is not None and self.points[i].visible:
                real_pos = self.real_positions[i]
                transformed = real_pos * a + b
                self.points[i].position = Vec3(transformed[0], transformed[1], transformed[2])
    
    def set_zoom_manager(self, zoom_manager):
        """Установить ссылку на ZoomManager"""
        self.zoom_manager = zoom_manager
    
    @property
    def enabled(self):
        """Для совместимости с ZoomManager"""
        return self.count > 0
    
    def destroy(self):
        """Уничтожить все Entity"""
        from ursina import destroy
        for point in self.points:
            try:
                destroy(point)
            except:
                pass
        self.points = []
    
    def __del__(self):
        """Cleanup"""
        try:
            import sys
            if sys.meta_path is not None:
                self.destroy()
        except:
            pass





