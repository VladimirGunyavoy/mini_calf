from ursina import Entity, scene
from typing import Optional, List, Any
from utils.scalable import Scalable
from managers.color_manager import ColorManager

class Frame(Entity):
    """
    Класс для отображения локальной системы координат (frame) в виде трех
    цветных стрелок.
    """
    def __init__(self, position=(0, 0, 0), color_manager=None, origin_scale: float = 0.04, **kwargs):
        if color_manager is None:
            from managers.color_manager import ColorManager
            color_manager = ColorManager()
        self.color_manager = color_manager
        
        super().__init__(
            position=position,
            **kwargs
        )
        
        self.parent = scene
        self.collider = None
        self.texture = None

        self.origin_cube: Scalable = Scalable(
            parent=self,
            model='cube',
            color=self.color_manager.get_color('frame', 'origin'),
            scale=origin_scale
        )

        # Ursina ищет модели относительно папки assets/ или рабочей директории
        # Используем относительный путь, как в main.py
        arrow_model = 'assets/arrow.obj'
        
        self.x_axis: Scalable = Scalable(
            parent=self,
            model=arrow_model,
            color=self.color_manager.get_color('frame', 'x_axis'),
            rotation=(0, 0, 90),
            scale=(1, 1, 1),
            unlit=True
        )
        self.y_axis: Scalable = Scalable(
            parent=self,
            model=arrow_model,
            color=self.color_manager.get_color('frame', 'y_axis'),
            rotation=(0, 90, 0),
            scale=(1, 1, 1)
        )
        self.z_axis: Scalable = Scalable(
            parent=self,
            model=arrow_model,
            color=self.color_manager.get_color('frame', 'z_axis'),
            rotation=(0, -90, 90),
            scale=(1, 1, 1)
        )

        self.entities: List[Scalable] = [self.origin_cube, self.x_axis, self.y_axis, self.z_axis]

    def toggle_visibility(self) -> None:
        """Переключает видимость всех элементов Frame (оси и куб)."""
        current_state = self.origin_cube.enabled
        new_state = not current_state
        
        # Переключаем все элементы Frame
        self.origin_cube.enabled = new_state
        self.x_axis.enabled = new_state
        self.y_axis.enabled = new_state
        self.z_axis.enabled = new_state
        
        # Выводим статус
        status = "показан" if new_state else "скрыт"
        print(f"📐 Frame {status}")

    def hide_frame(self) -> None:
        """Скрывает все элементы Frame."""
        self.origin_cube.enabled = False
        self.x_axis.enabled = False
        self.y_axis.enabled = False
        self.z_axis.enabled = False

    def show_frame(self) -> None:
        """Показывает все элементы Frame."""
        self.origin_cube.enabled = True
        self.x_axis.enabled = True
        self.y_axis.enabled = True
        self.z_axis.enabled = True

    def is_visible(self) -> bool:
        """Проверяет видим ли Frame."""
        return self.origin_cube.enabled
    
    def register_in_object_manager(self, object_manager) -> None:
        """
        Регистрирует все части фрейма в ObjectManager
        
        Args:
            object_manager: ObjectManager для регистрации
        """
        object_manager.register_existing('frame_origin', self.origin_cube)
        object_manager.register_existing('frame_x', self.x_axis)
        object_manager.register_existing('frame_y', self.y_axis)
        object_manager.register_existing('frame_z', self.z_axis)
        print(f"[Frame] Registered in ObjectManager")

