"""
Настройка сцены - управление освещением, курсором, полом
Объединенный модуль для всех компонентов сцены
"""

from ursina import *
from typing import Optional, List, Tuple

from managers.color_manager import ColorManager
from managers.ui_manager import UIManager
from utils.scalable import Scalable
from .frame import Frame


# ============================================================================
# ФУНКЦИИ СОЗДАНИЯ СЦЕНЫ
# ============================================================================

def create_ground(color_manager: Optional[ColorManager] = None, 
                 object_manager=None):
    """
    Создание пола с сеткой через ObjectManager
    
    Args:
        color_manager: Менеджер цветов
        object_manager: Менеджер объектов для автоматической регистрации
    """
    if color_manager is None:
        from managers.color_manager import ColorManager
        color_manager = ColorManager()
    
    # Используем очень темный цвет в нормализованном формате (0-1)
    dark_floor_color = color.rgba(0.06, 0.06, 0.08, 1.0)
    grid_color = color.rgba(0.25, 0.25, 0.3, 0.7)
    
    if object_manager:
        # Создаем через ObjectManager - автоматическая регистрация
        ground = object_manager.create_object(
            name='ground',
            model='plane',
            position=(0, -0.01, 0),
            scale=(50, 1, 50),
            color_val=dark_floor_color,
            collider='box',
            unlit=True
        )
        
        grid = object_manager.create_object(
            name='grid',
            model='plane',
            position=(0, 0.00, 0),
            scale=(50, 1, 50),
            color_val=grid_color,
            texture='white_cube',
            texture_scale=(50, 50),
            unlit=True
        )
    else:
        # Fallback - создаем напрямую (для обратной совместимости)
        ground = Scalable(
            model='plane',
            scale=(50, 1, 50),
            collider='box',
            color=dark_floor_color,
            unlit=True,
            position=(0, -0.01, 0),
        )
        
        grid = Scalable(
            model='plane',
            scale=(50, 1, 50),
            texture='white_cube',
            texture_scale=(50, 50),
            position=(0, 0.00, 0),
            color=grid_color,
            unlit=True
        )
    
    return ground, grid


def setup_lighting(color_manager: Optional[ColorManager] = None) -> List[Light]:
    """
    Настройка расширенного освещения с несколькими источниками света
    
    Args:
        color_manager: Менеджер цветов для получения цветов освещения.
                      Если None, создается новый ColorManager.
    
    Returns:
        List[Light]: Список созданных источников света
    """
    # Используем переданный ColorManager или создаем новый
    if color_manager is None:
        from managers.color_manager import ColorManager
        color_manager = ColorManager()
    
    # Создаем расширенное освещение с несколькими источниками
    lights: List[Light] = [
        # Первый направленный свет под углом
        DirectionalLight(
            rotation=(-45, -45, 45), 
            color=color_manager.get_color('scene', 'directional_light'), 
            intensity=1.5
        ),
        # Второй направленный свет сверху
        DirectionalLight(
            rotation=(45, 0, 0), 
            color=color_manager.get_color('scene', 'directional_light'), 
            intensity=1.2
        ),
        # Окружающее освещение
        AmbientLight(
            color=color_manager.get_color('scene', 'ambient_light')
        )
    ]
    
    return lights


def setup_scene(color_manager: ColorManager, object_manager):
    """
    Полная настройка сцены - создает пол, сетку, освещение и координатный фрейм
    
    Args:
        color_manager: Менеджер цветов
        object_manager: Менеджер объектов для автоматической регистрации
        
    Returns:
        tuple: (ground, grid, lights, frame)
    """
    # 1. Создание пола и сетки (автоматическая регистрация через ObjectManager)
    ground, grid = create_ground(color_manager=color_manager, 
                                 object_manager=object_manager)
    
    # 2. Настройка освещения
    lights = setup_lighting(color_manager=color_manager)
    
    # 3. Координатный фрейм
    frame = Frame(position=(0, 0, 0), color_manager=color_manager)
    frame.register_in_object_manager(object_manager)
    
    print(f"[setup_scene] Scene setup complete: ground, grid, {len(lights)} lights, frame")
    
    return ground, grid, lights, frame


# ============================================================================
# КЛАСС SCENE SETUP (legacy, может быть удален если не используется)
# ============================================================================

class SceneSetup:
    """
    Класс для настройки сцены (legacy)
    Сейчас не используется в main.py, функции выше используются напрямую
    """
    
    def __init__(self, 
                 init_position: Tuple[float, float, float] = (1.5, -1, -2), 
                 init_rotation_x: float = 21, 
                 init_rotation_y: float = -35, 
                 color_manager: Optional[ColorManager] = None, 
                 ui_manager: Optional[UIManager] = None, 
                 **kwargs):
        # Используем переданный ColorManager или создаем новый
        if color_manager is None:
            from managers.color_manager import ColorManager
            color_manager = ColorManager()
        self.color_manager: ColorManager = color_manager
        
        # Используем переданный UIManager или создаем новый
        if ui_manager is None:
            from managers.ui_manager import UIManager
            ui_manager = UIManager(self.color_manager)
        self.ui_manager: UIManager = ui_manager
        
        self.lights: List[Light] = setup_lighting(self.color_manager)

        self.base_position: Tuple[float, float, float] = init_position
        self.base_speed: float = 2

        from ursina.prefabs.first_person_controller import FirstPersonController
        self.player: FirstPersonController = FirstPersonController(
            gravity=0,
            position=init_position,
            speed=self.base_speed
        )
        
        self.player.camera_pivot.rotation_x = init_rotation_x
        self.player.rotation_y = init_rotation_y
        
        # Новый флаг для "заморозки" ввода
        # input_frozen = False означает, что курсор захвачен (по умолчанию)
        # input_frozen = True означает, что курсор освобожден
        self.input_frozen: bool = False
        
        # Курсор заблокирован по умолчанию (захвачен в приложении)
        self.cursor_locked: bool = True
        
        # Принудительно захватываем курсор
        mouse.locked = True
        mouse.visible = False  # Скрываем курсор мыши
        
        window.color = self.color_manager.get_color('scene', 'window_background')
        
        # Принудительно устанавливаем состояние курсора в конце инициализации
        self._update_cursor_state()
        
    def _update_cursor_state(self) -> None:
        """Обновляет состояние курсора в соответствии с флагом input_frozen."""
        mouse.locked = not self.input_frozen
        mouse.visible = self.input_frozen
        print(f"[SceneSetup] Курсор установлен: locked={mouse.locked}, visible={mouse.visible}")
        
    def update_position_info(self) -> None:
        """Обновляется автоматически через UI Manager"""
        # Эта функция теперь вызывается автоматически через ui_manager.update_dynamic_elements()
        pass

    def toggle_freeze(self) -> None:
        """Переключает режим 'заморозки' всего ввода."""
        self.input_frozen = not self.input_frozen
        
        # Обновляем состояние курсора
        self._update_cursor_state()
        
        # Блокируем/разблокируем игрока
        self.player.enabled = not self.input_frozen
        
        status = "освобожден" if self.input_frozen else "захвачен"
        print(f"[SceneSetup] Курсор {status} (input_frozen={self.input_frozen})")

    def update(self, dt: float) -> None:
        """Updates additional parameters not included in FirstPersonController"""
        # Если ввод заморожен, остальную логику (движение) не выполняем
        if self.input_frozen:
            return
            
        self.player.y += (held_keys['space'] - held_keys['shift']) * self.player.speed * dt
    
    def set_visual_manager(self, visual_manager) -> None:
        """Позволяет передать visual_manager после инициализации."""
        self.visual_manager = visual_manager



