"""
UI Manager - Централизованное управление всеми UI элементами
============================================================

Класс для централизованного управления всеми текстовыми элементами в проекте:
- Статичные плашки (инструкции)
- Динамические плашки (статусы, счетчики)

Преимущества:
✅ Все UI элементы в одном месте
✅ Упрощенная структура (статичные/динамичные)
✅ Легкое обновление и управление
✅ Поддержка динамического обновления
"""

from ursina import color, destroy, Text, camera
from typing import Dict, Optional, Callable, Any, TYPE_CHECKING
from .color_manager import ColorManager
from config.ui_constants import UI_POSITIONS

if TYPE_CHECKING:
    from controls_window import ControlsWindow


class UIManager:
    """Централизованный менеджер UI элементов"""
    
    def __init__(self, color_manager: Optional[ColorManager] = None, 
                 player=None, zoom_manager=None):
        if color_manager is None:
            color_manager = ColorManager()
        self.color_manager: ColorManager = color_manager
        
        self.elements: Dict[str, Dict[str, Text]] = {'static': {}, 'dynamic': {}}
        self.category_visibility: Dict[str, bool] = {'static': True, 'dynamic': True}
        self.update_functions: Dict[str, Callable[[], None]] = {}
        
        # Флаг для автоматического отслеживания камеры
        self._camera_tracking_enabled: bool = False
        self._player = player  # Ссылка на Player для получения углов камеры
        
        # Ссылка на ZoomManager для отслеживания масштаба
        self._zoom_manager = zoom_manager
        self._scale_tracking_enabled: bool = False
        
        # Controls Window - отдельный компонент
        self.controls_window: Optional['ControlsWindow'] = None
        
        self.styles: Dict[str, Dict[str, Any]] = {
            'default': {
                'scale': 0.7, 'color': self.color_manager.get_color('ui', 'text_primary'),
                'font': 'VeraMono.ttf', 'has_background': True,
            },
            'instructions': {
                'scale': 0.7, 'color': self.color_manager.get_color('ui', 'text_primary'),
                'font': 'VeraMono.ttf', 'has_background': True,
            },
            'header': {
                'scale': 0.7, 'color': self.color_manager.get_color('ui', 'text_primary'),
                'font': 'VeraMono.ttf', 'has_background': True,
            },
            'status': {
                'scale': 0.7, 'color': self.color_manager.get_color('ui', 'text_secondary'),
                'font': 'VeraMono.ttf', 'has_background': False,
            },
            'counter': {
                'scale': 0.7, 'color': color.yellow,
                'font': 'VeraMono.ttf', 'has_background': True,
            },
            'debug': {
                'scale': 0.7, 'color': color.cyan,
                'font': 'VeraMono.ttf', 'has_background': True,
            }
        }
        
        # Автоматически настраиваем отслеживание, если переданы зависимости
        if self._player:
            self.setup_camera_tracking()
        
        if self._zoom_manager:
            self.setup_scale_tracking()
    
    def create_element(self, category: str, name: str, 
                      text: str = "", position: tuple = (0, 0), 
                      style: str = 'default', **kwargs) -> Text:
        
        style_info = self.styles.get(style, self.styles['default']).copy()
        
        # Получаем флаг для фона и удаляем его, чтобы он не попал в конструктор
        should_have_background: bool = style_info.pop('has_background', False)
        
        element_kwargs: Dict[str, Any] = style_info
        element_kwargs.update(kwargs)
        element_kwargs['text'] = text
        element_kwargs['position'] = position
        
        # 1. Создаем объект Text без фона
        element = Text(**element_kwargs)
        element.style = style
        element.enabled = self.category_visibility.get(category, True)
        
        # 2. Включаем фон отдельной командой, как вы и предложили
        if should_have_background:
            element.background = True
        
        self.elements[category][name] = element
        return element
    
    def update_text(self, name: str, text: str) -> None:
        """Обновляет текст и принудительно перерисовывает фон."""
        if name in self.elements['dynamic']:
            element = self.elements['dynamic'][name]
            element.text = text
            
            # При обновлении текста также принудительно перерисовываем фон
            style_info = self.styles.get(element.style, self.styles['default'])
            if style_info.get('has_background'):
                element.background = True
    
    def get_element(self, name: str) -> Optional[Text]:
        for category in self.elements.values():
            if name in category:
                return category[name]
        return None
    
    def show_element(self, name: str) -> None:
        element = self.get_element(name)
        if element:
            element.enabled = True
    
    def hide_element(self, name: str) -> None:
        element = self.get_element(name)
        if element:
            element.enabled = False
    
    def toggle_element(self, name: str) -> None:
        element = self.get_element(name)
        if element:
            element.enabled = not element.enabled
    
    def show_category(self, category: str) -> None:
        if category in self.elements:
            for element in self.elements[category].values():
                element.enabled = True

        if category in self.category_visibility:
            self.category_visibility[category] = True
            if category == 'static' and self.controls_window:
                self.controls_window.set_visibility(True)

    def hide_category(self, category: str) -> None:
        if category in self.elements:
            for element in self.elements[category].values():
                element.enabled = False

        if category in self.category_visibility:
            self.category_visibility[category] = False
            if category == 'static' and self.controls_window:
                self.controls_window.set_visibility(False)

    def toggle_category(self, category: str) -> None:
        new_state = None

        if category in self.elements:
            elements = list(self.elements[category].values())
            if elements:
                new_state = not elements[0].enabled
                for element in elements:
                    element.enabled = new_state
        elif category in self.category_visibility:
            new_state = not self.category_visibility[category]

        if new_state is not None and category in self.category_visibility:
            self.category_visibility[category] = new_state
            if category == 'static' and self.controls_window:
                self.controls_window.set_visibility(new_state)

    def register_update_function(self, key: str, func: Callable[[], None]) -> None:
        self.update_functions[key] = func
    
    def update_dynamic_elements(self) -> None:
        """Обновляет все динамические элементы через зарегистрированные функции"""
        for key, func in self.update_functions.items():
            try:
                func()
            except Exception as e:
                print(f"Ошибка обновления UI элемента {key}: {e}")
    
    def _update_camera_info(self) -> None:
        """Внутренний метод для обновления информации о камере"""
        if self._camera_tracking_enabled and 'camera' in self.elements['dynamic']:
            pos = camera.world_position
            
            # Получаем углы из player, если доступен
            if self._player:
                rot_x = self._player.camera_pivot.rotation_x if hasattr(self._player, 'camera_pivot') else 0
                rot_y = self._player.rotation_y if hasattr(self._player, 'rotation_y') else 0
                rot_z = self._player.rotation_z if hasattr(self._player, 'rotation_z') else 0
            else:
                # Fallback на camera.rotation
                rot = camera.rotation
                rot_x, rot_y, rot_z = rot.x, rot.y, rot.z
            
            self.update_text('camera', 
                f"Pos: {pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f}\nAng: {rot_x:.2f}, {rot_y:.2f}, {rot_z:.2f}")
    
    def _update_scale_info(self) -> None:
        """Внутренний метод для обновления информации о масштабе"""
        if self._scale_tracking_enabled and 'scale' in self.elements['dynamic'] and self._zoom_manager:
            a_scale = self._zoom_manager.a_transformation
            obj_scale = self._zoom_manager.objects_scale
            self.update_text('scale', 
                f"Scale: {a_scale:.3f}\nObj Scale: {obj_scale:.3f}")
    
    def enable_camera_tracking(self) -> None:
        """Включает автоматическое отслеживание камеры"""
        self._camera_tracking_enabled = True
    
    def disable_camera_tracking(self) -> None:
        """Выключает автоматическое отслеживание камеры"""
        self._camera_tracking_enabled = False
    
    def enable_scale_tracking(self, zoom_manager) -> None:
        """Включает автоматическое отслеживание масштаба
        
        Args:
            zoom_manager: Ссылка на ZoomManager для получения информации о масштабе
        """
        self._zoom_manager = zoom_manager
        self._scale_tracking_enabled = True
    
    def disable_scale_tracking(self) -> None:
        """Выключает автоматическое отслеживание масштаба"""
        self._scale_tracking_enabled = False
    
    def setup_camera_tracking(self) -> None:
        """Создает элемент камеры и включает его автоматическое отслеживание"""
        if not self._player:
            print("[WARNING] Player not provided to UIManager, camera angles may be incorrect")
        self.create_camera_info()
        self.enable_camera_tracking()
        print("[OK] Camera tracking enabled")
    
    def setup_scale_tracking(self) -> None:
        """Создает элемент масштаба и включает его автоматическое отслеживание"""
        if not self._zoom_manager:
            print("[WARNING] ZoomManager not provided to UIManager, scale tracking unavailable")
            return
        self.create_scale_info()
        self.enable_scale_tracking(self._zoom_manager)
        print("[OK] Scale tracking enabled")
    
    def update(self) -> None:
        """Основной метод обновления, вызываемый каждый кадр из VisualsUpdateManager"""
        # Обновляем информацию о камере, если включено отслеживание
        self._update_camera_info()
        
        # Обновляем информацию о масштабе, если включено отслеживание
        self._update_scale_info()
        
        # Обновляем все зарегистрированные динамические элементы
        self.update_dynamic_elements()
    
    def create_position_info(self, name: str = 'main', position: tuple = UI_POSITIONS.POSITION_INFO) -> Text:
        return self.create_element('dynamic', name,
            text="Position: 0.000, 0.000, 0.000\nRotation: 0.000, 0.000, 0.000",
            position=position, style='status')
    
    def create_camera_info(self, name: str = 'camera', position: tuple = UI_POSITIONS.CAMERA_INFO) -> Text:
        """Создает информацию о камере (позиция и углы) в правом нижнем углу"""
        return self.create_element('dynamic', name,
            text="Pos: 0.00, 0.00, 0.00\nAng: 0.00, 0.00, 0.00",
            position=position, style='status')
    
    def create_scale_info(self, name: str = 'scale', position: tuple = UI_POSITIONS.SCALE_INFO) -> Text:
        """Создает информацию о масштабе в верхнем правом углу"""
        return self.create_element('dynamic', name,
            text="Scale: 1.000\nObj Scale: 1.000",
            position=position, style='status')
    
    def create_instructions(self, name: str, instructions_text: str, position: tuple = UI_POSITIONS.DEFAULT_INSTRUCTIONS) -> Text:
        return self.create_element('static', name,
            text=instructions_text, position=position, style='instructions')
    
    def create_status_indicator(self, name: str, initial_text: str = "", position: tuple = UI_POSITIONS.CURSOR_STATUS) -> Text:
        return self.create_element('dynamic', name,
            text=initial_text, position=position, style='status', origin=(0, 0))
    
    def create_counter(self, name: str, initial_value: int = 0, position: tuple = UI_POSITIONS.DEFAULT_COUNTER, prefix: str = "") -> Text:
        text = f"{prefix}{initial_value}" if prefix else str(initial_value)
        return self.create_element('dynamic', name, text=text, position=position, style='counter')
    
    def update_counter(self, name: str, value: int, prefix: str = "") -> None:
        text = f"{prefix}{value}" if prefix else str(value)
        self.update_text(name, text)
    
    def create_info_block(self, name: str, title: str, content: str = "", position: tuple = UI_POSITIONS.DEFAULT_INFO_BLOCK, is_dynamic: bool = False) -> Text:
        category = 'dynamic' if is_dynamic else 'static'
        text = f"{title}\n{content}" if content else title
        return self.create_element(category, name, text=text, position=position, style='default')
    
    def update_info_block(self, name: str, title: str, content: str = "") -> None:
        text = f"{title}\n{content}" if content else title
        self.update_text(name, text)
    
    def create_debug_info(self, name: str, position: tuple = UI_POSITIONS.DEFAULT_DEBUG_INFO) -> Text:
        return self.create_element('dynamic', name, text="", position=position, style='debug')
    
    def remove_element(self, name: str) -> None:
        for category_elements in self.elements.values():
            if name in category_elements:
                destroy(category_elements[name])
                del category_elements[name]
                return
    
    def clear_category(self, category: str) -> None:
        if category in self.elements:
            for element in list(self.elements[category].values()):
                destroy(element)
            self.elements[category].clear()
    
    def clear_all(self) -> None:
        for category in list(self.elements.keys()):
            self.clear_category(category)
    
    def get_stats(self) -> Dict[str, int]:
        return {'static': len(self.elements['static']), 'dynamic': len(self.elements['dynamic']),
                'total': len(self.elements['static']) + len(self.elements['dynamic'])}
    
    def print_stats(self) -> None:
        stats = self.get_stats()
        print("\n--- UI Manager Stats ---")
        for category, count in stats.items():
            print(f"  {category.capitalize()}: {count}")
        print("------------------------")
    
    def create_controls_window(self, input_manager, position: tuple = UI_POSITIONS.CONTROLS_WINDOW, scale: float = 0.7) -> 'ControlsWindow':
        """
        Создает и настраивает окно управления.
        
        Args:
            input_manager: InputManager для получения команд
            position: Позиция окна
            scale: Масштаб текста
            
        Returns:
            Созданный экземпляр ControlsWindow
        """
        from .controls_window import ControlsWindow
        
        self.controls_window = ControlsWindow(
            input_manager=input_manager,
            color_manager=self.color_manager,
            position=position,
            scale=scale
        )
        
        print("📋 Controls window created via UIManager")
        return self.controls_window
    
    def toggle_controls_window(self) -> None:
        """Переключает видимость окна управления."""
        if self.controls_window:
            self.controls_window.toggle_visibility()
        else:
            print("⚠️ Controls window not initialized")
    
    def show_controls_window(self) -> None:
        """Показывает окно управления."""
        if self.controls_window and not self.controls_window.visible:
            self.controls_window.toggle_visibility()
    
    def hide_controls_window(self) -> None:
        """Скрывает окно управления."""
        if self.controls_window and self.controls_window.visible:
            self.controls_window.toggle_visibility()
    
    def update_controls_window(self) -> None:
        """Обновляет содержимое окна управления."""
        if self.controls_window:
            self.controls_window.update_commands()

def get_ui_manager(color_manager: Optional[ColorManager] = None) -> UIManager:
    return UIManager(color_manager)