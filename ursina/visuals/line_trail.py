"""
Line-based Trail with Ring Buffer and Multi-Color Segments
Траектория с кольцевым буфером и разноцветными сегментами по режимам
"""

from ursina import Entity, Vec3, Vec4, Mesh, destroy
import numpy as np


class LineTrail:
    """
    Траектория с кольцевым буфером и разноцветными сегментами.

    Принцип:
    - Кольцевой буфер хранит позиции и режимы
    - Пул Entity для сегментов разных цветов (макс 10)
    - Группировка последовательных точек по режиму
    - При обновлении обновляются только mesh'и (без destroy Entity)

    Эффективно: ~3-5 Entity на трейл (по числу смен режима)
    """

    # Цвета режимов
    MODE_COLORS = {
        'td3': Vec4(0.2, 0.4, 1.0, 1),      # Синий
        'relax': Vec4(0.2, 0.7, 0.3, 1),    # Зеленый
        'fallback': Vec4(1.0, 0.5, 0.1, 1)  # Оранжевый
    }

    MAX_SEGMENTS = 50  # Максимум сегментов (было 10, увеличено для частых переключений режима)

    def __init__(self, max_points=150, line_thickness=2, decimation=3, rebuild_freq=5):
        """
        Parameters:
        -----------
        max_points : int
            Максимальное количество точек в буфере
        line_thickness : float
            Толщина линии
        decimation : int
            Добавлять каждую N-ую точку
        rebuild_freq : int
            (deprecated) Не используется - линии перестраиваются каждый кадр
        """
        self.max_points = max_points
        self.line_thickness = line_thickness
        self.decimation = decimation
        
        # Кольцевой буфер позиций (реальных, без зума)
        self.positions = []  # List of np.array([x, y, z])
        self.modes = []      # List of mode strings
        
        # Пул Entity для сегментов (создаём один раз!)
        self.segment_pool = []
        for _ in range(50):  # Увеличен пул до 50
            seg = Entity(visible=False)
            self.segment_pool.append(seg)
        self.active_segments = 0
        
        # Счётчик для decimation
        self.step_counter = 0
        
        # Трансформации зума
        self.a_transformation = 1.0
        self.b_translation = np.array([0, 0, 0], dtype=float)
        self.zoom_manager = None
    
    def add_point(self, position, mode='td3', a_transform=None, b_translate=None):
        """Добавить точку в траекторию.

        Parameters:
        -----------
        position : array-like
            Реальная позиция точки
        mode : str
            Режим агента ('td3', 'relax', 'fallback')
        a_transform : float, optional
            Коэффициент трансформации (если None, берется из zoom_manager)
        b_translate : np.ndarray, optional
            Вектор сдвига (если None, берется из zoom_manager)
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

        # Добавляем в буфер
        self.positions.append(real_pos)
        self.modes.append(mode)

        # Ограничиваем размер буфера
        if len(self.positions) > self.max_points:
            self.positions.pop(0)
            self.modes.pop(0)

        # КРИТИЧНО: Используем явно переданные трансформации или берем из zoom_manager
        if a_transform is not None:
            self.a_transformation = a_transform
        elif self.zoom_manager is not None:
            self.a_transformation = self.zoom_manager.a_transformation

        if b_translate is not None:
            self.b_translation = b_translate
        elif self.zoom_manager is not None:
            self.b_translation = self.zoom_manager.b_translation

        # Перестраиваем линии каждый кадр (для синхронизации с агентами)
        self._rebuild_segments()
    
    def _group_by_mode(self):
        """
        Группировка последовательных точек по режиму.
        Возвращает список (mode, [indices]).
        """
        if len(self.positions) < 2:
            return []
        
        groups = []
        current_mode = self.modes[0]
        current_indices = [0]
        
        for i in range(1, len(self.positions)):
            mode = self.modes[i]
            
            if mode == current_mode:
                current_indices.append(i)
            else:
                # Добавляем текущую точку для связи
                current_indices.append(i)
                groups.append((current_mode, current_indices.copy()))
                # Новая группа начинается с текущей точки
                current_mode = mode
                current_indices = [i]
        
        # Последняя группа
        if len(current_indices) >= 1:
            groups.append((current_mode, current_indices))
        
        return groups
    
    def _rebuild_segments(self):
        """Перестроить сегменты линий по группам режимов."""
        # Сначала скрываем все сегменты
        for seg in self.segment_pool:
            seg.visible = False

        if len(self.positions) < 2:
            self.active_segments = 0
            return

        # Группируем по режимам
        groups = self._group_by_mode()

        # КРИТИЧНО: Берем ПОСЛЕДНИЕ сегменты, а не первые!
        # Это гарантирует что конец траектории (где агент сейчас) всегда рисуется
        if len(groups) > self.MAX_SEGMENTS:
            groups = groups[-self.MAX_SEGMENTS:]  # Берем последние MAX_SEGMENTS

        segment_idx = 0

        for mode, indices in groups:
            if len(indices) >= 2 and segment_idx < self.MAX_SEGMENTS:
                # Собираем точки для этого сегмента
                points = []
                for idx in indices:
                    pos = self.positions[idx]
                    t = pos * self.a_transformation + self.b_translation
                    points.append(Vec3(t[0], t[1], t[2]))

                # Обновляем Entity из пула
                seg = self.segment_pool[segment_idx]

                # КРИТИЧНО: Обнуляем старый mesh перед созданием нового
                # (Ursina автоматически очистит старый mesh при присвоении None)
                seg.model = None

                # Создаем новый mesh
                seg.model = Mesh(vertices=points, mode='line', thickness=self.line_thickness)
                seg.color = self.MODE_COLORS.get(mode, self.MODE_COLORS['td3'])
                seg.visible = True

                segment_idx += 1

        self.active_segments = segment_idx
    
    def clear(self):
        """Очистить траекторию."""
        self.positions = []
        self.modes = []
        self.step_counter = 0
        self.active_segments = 0
        for seg in self.segment_pool:
            seg.visible = False
            if seg.model is not None:
                try:
                    destroy(seg.model)
                except:
                    pass
                seg.model = None
    
    def apply_transform(self, a, b, **kwargs):
        """Применить трансформацию зума."""
        self.a_transformation = a
        self.b_translation = b
        self._rebuild_segments()
    
    def set_zoom_manager(self, zoom_manager):
        """Установить ссылку на ZoomManager."""
        self.zoom_manager = zoom_manager
    
    @property
    def enabled(self):
        """
        Для совместимости с ZoomManager.
        Всегда возвращает True чтобы получать обновления трансформаций
        даже когда трейл пуст (после clear()).
        """
        return True
    
    def destroy(self):
        """Уничтожить все Entity."""
        for seg in self.segment_pool:
            try:
                if seg.model is not None:
                    destroy(seg.model)
                destroy(seg)
            except:
                pass
        self.segment_pool = []
    
    def __del__(self):
        try:
            import sys
            if sys.meta_path is not None:
                self.destroy()
        except:
            pass





