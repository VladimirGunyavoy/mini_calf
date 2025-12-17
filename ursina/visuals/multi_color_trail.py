"""
Multi-Color Trail Visualization
Траектория с разными цветами для визуализации переключений режимов CALF
"""

from ursina import Entity, destroy, Vec3, Vec4, Mesh
import numpy as np


class MultiColorTrail:
    """
    Мультицветная траектория для визуализации переключений режимов CALF.
    
    Каждый сегмент траектории окрашен в цвет соответствующего режима:
    - TD3: синий
    - Relax: зеленый
    - Fallback: оранжевый
    
    Features:
    - Автоматическая группировка последовательных точек по режиму
    - Максимальная длина (старые точки удаляются)
    - Decimation (скважность) - добавлять каждую N-ую точку
    - Оптимизированная перестройка
    """
    
    # Цвета режимов (Vec4 для надежности)
    MODE_COLORS = {
        'td3': Vec4(0.2, 0.3, 0.8, 1),      # Синий
        'relax': Vec4(0.2, 0.6, 0.3, 1),    # Зеленый
        'fallback': Vec4(0.8, 0.4, 0.15, 1) # Оранжевый
    }
    
    def __init__(self, max_length=200, decimation=1, rebuild_frequency=5):
        """
        Parameters:
        -----------
        max_length : int
            Максимальное количество точек в траектории
        decimation : int
            Каждая N-ая точка добавляется (1 = все точки)
        rebuild_frequency : int
            Перестраивать mesh каждые N добавлений
        """
        self.max_length = max_length
        self.decimation = decimation
        self.rebuild_frequency = rebuild_frequency
        
        # Хранение точек и режимов
        self.positions = []  # List[Vec3] - реальные позиции (без зума)
        self.modes = []      # List[str] - 'td3', 'relax', 'fallback'
        
        # Entity для каждого сегмента
        self.segments = []  # List[Entity]
        
        # Счетчики
        self.step_counter = 0
        self.rebuild_counter = 0
        
        # Трансформации зума
        self.a_transformation = 1.0
        self.b_translation = np.array([0, 0, 0], dtype=float)
        self.zoom_manager = None
    
    def add_point(self, position, mode):
        """
        Добавить точку в траекторию с указанием режима.
        
        Parameters:
        -----------
        position : tuple/list/Vec3
            3D позиция точки [x, y, z]
        mode : str
            Режим CALF: 'td3', 'relax', или 'fallback'
        """
        self.step_counter += 1
        
        # Decimation: добавляем только каждую N-ую точку
        if self.step_counter % self.decimation != 0:
            return
        
        # Конвертируем в Vec3
        if not isinstance(position, Vec3):
            position = Vec3(*position)
        
        # Добавляем точку и режим
        self.positions.append(position)
        self.modes.append(mode)
        
        # Ограничение длины
        if len(self.positions) > self.max_length:
            self.positions.pop(0)
            self.modes.pop(0)
        
        # Счетчик для rebuild
        self.rebuild_counter += 1
        
        # Перестраивать только каждые N добавлений
        if self.rebuild_counter >= self.rebuild_frequency:
            self.rebuild()
            self.rebuild_counter = 0
    
    def rebuild(self):
        """
        Перестроить визуализацию траектории.
        Создает отдельный Entity для каждого сегмента одного режима.
        """
        # Удаляем старые сегменты
        for seg in self.segments:
            destroy(seg)
        self.segments = []
        
        # Нужно минимум 2 точки
        if len(self.positions) < 2:
            return
        
        # Группируем по режимам
        groups = self._group_by_mode()
        
        # Применяем трансформацию зума к точкам
        transformed_groups = []
        for mode, points in groups:
            transformed_points = [self._apply_transform_to_point(p) for p in points]
            transformed_groups.append((mode, transformed_points))
        
        # Создаем Entity для каждой группы
        for mode, points in transformed_groups:
            if len(points) >= 2:
                # Цвет уже с alpha в Vec4(r, g, b, a)
                trail_color = self.MODE_COLORS.get(mode, Vec4(1, 1, 1, 1))
                seg = Entity(
                    model=Mesh(vertices=points, mode='line', thickness=3),
                    color=trail_color
                )
                self.segments.append(seg)
    
    def _apply_transform_to_point(self, point):
        """Применить трансформацию зума к точке"""
        p_array = np.array([point.x, point.y, point.z])
        transformed = p_array * self.a_transformation + self.b_translation
        return Vec3(transformed[0], transformed[1], transformed[2])
    
    def _group_by_mode(self):
        """
        Группировка последовательных точек по режиму.
        
        Returns:
        --------
        groups : List[Tuple[str, List[Vec3]]]
            Список групп: [(mode, [points]), ...]
            
        Example:
        --------
        positions = [p0, p1, p2, p3, p4]
        modes = ['td3', 'td3', 'relax', 'relax', 'td3']
        
        Result:
        [('td3', [p0, p1, p2]),      # Note: p2 включен для связи
         ('relax', [p2, p3, p4]),    # Note: p2 и p4 включены для связи
         ('td3', [p4])]              # Note: p4 включен для связи
        """
        if len(self.positions) < 2:
            return []
        
        groups = []
        current_mode = self.modes[0]
        current_points = [self.positions[0]]
        
        for i in range(1, len(self.positions)):
            pos = self.positions[i]
            mode = self.modes[i]
            
            if mode == current_mode:
                # Продолжаем текущую группу
                current_points.append(pos)
            else:
                # Режим изменился - завершаем текущую группу
                # Добавляем текущую точку для связи сегментов
                current_points.append(pos)
                groups.append((current_mode, current_points.copy()))
                
                # Начинаем новую группу (с текущей точкой для связи)
                current_mode = mode
                current_points = [pos]
        
        # Добавляем последнюю группу
        if len(current_points) >= 1:
            groups.append((current_mode, current_points))
        
        return groups
    
    def clear(self):
        """
        Очистить траекторию
        """
        self.positions = []
        self.modes = []
        self.step_counter = 0
        self.rebuild_counter = 0
        
        # Удаляем все сегменты
        for seg in self.segments:
            destroy(seg)
        self.segments = []
    
    def force_rebuild(self):
        """
        Принудительно перестроить траекторию (игнорируя rebuild_frequency)
        """
        self.rebuild()
        self.rebuild_counter = 0
    
    def apply_transform(self, a, b, **kwargs):
        """
        Применить трансформацию зума (для совместимости с ZoomManager)
        
        Parameters:
        -----------
        a : float
            Масштаб
        b : np.ndarray
            Смещение [x, y, z]
        """
        self.a_transformation = a
        self.b_translation = b
        # Перестроить с новой трансформацией
        self.rebuild()
    
    def set_zoom_manager(self, zoom_manager):
        """Установить ссылку на ZoomManager"""
        self.zoom_manager = zoom_manager
    
    @property
    def enabled(self):
        """Для совместимости с ZoomManager"""
        return len(self.segments) > 0
    
    def __del__(self):
        """Очистка при удалении объекта"""
        try:
            import sys
            # Проверяем что Python еще не завершается
            if sys.meta_path is not None:
                self.clear()
        except:
            # Игнорируем любые ошибки при shutdown
            pass


