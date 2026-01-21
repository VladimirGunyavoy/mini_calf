"""
Менеджер настройки окна приложения
"""

from ursina import *
from typing import Optional
from .color_manager import ColorManager


class WindowManager:
    """
    Класс для управления настройками окна Ursina с поддержкой нескольких мониторов.
    """
    
    # Настройки для разных мониторов
    MONITORS = {
        "local": {"size": (2500, 1500), "position": (0, 0)},
        "main":  {"size": (2700, 1580), "position": (100, 100)},
        "top":   {"size": (1920, 1080), "position": (0, -1080)},
        "left":  {"size": (1875, 970),  "position": (-1900, 220)},
        "down":  {"size": (3000, 1700), "position": (-500, 1500)}
    }
    
    def __init__(self, 
                 color_manager: Optional[ColorManager] = None,
                 title: str = "CALF Training Environment - Ursina",
                 monitor: str = "left"):
        """
        Инициализирует менеджер окна.
        
        Args:
            color_manager: Менеджер цветов для настройки фона.
            title: Заголовок окна.
            monitor: Тип монитора ("main", "top", "left", "down").
        """
        self.color_manager = color_manager
        self.current_monitor = monitor  # Сохраняем текущий монитор
        self.title = title
        self.setup_window()
        
    def setup_window(self):
        """Настройка параметров окна"""
        
        # Заголовок окна
        window.title = self.title
        
        # Применяем настройки монитора
        config = self.MONITORS.get(self.current_monitor, self.MONITORS["main"])
        
        # Устанавливаем размер и позицию ДО создания приложения Ursina
        # Это критически важно - эти параметры должны быть установлены до app = Ursina()
        window.size = config["size"]
        window.position = config["position"]
        
        # Размер окна
        window.borderless = True  # Убираем заголовок окна
        window.fullscreen = True
        window.exit_button.visible = False
        
        # Сохраняем позицию для возможной повторной установки
        self._pending_position = config["position"]
        self._pending_size = config["size"]
        
        # Курсор мыши (заблокирован для управления камерой)
        mouse.locked = True
        
        # FPS
        window.fps_counter.enabled = True
        window.fps_counter.position = (0.85, 0.47)
        window.fps_counter.color = color.yellow
        window.fps_counter.scale = 1.5
        
        # Цвет фона из ColorManager (если передан), иначе темный по умолчанию
        if self.color_manager:
            window.color = self.color_manager.get_color('scene', 'window_background')
        else:
            window.color = color.rgb(20, 20, 25)

        # Устанавливаем позицию окна (после всех настроек)
        self.apply_window_position()
        
        print(f"Window Manager initialized on monitor: {self.current_monitor}")
        print(f"  Window size: {window.size}")
        print(f"  Window position: {window.position}")
        print("Controls:")
        print("  WASD - movement")
        print("  Mouse - look around")
        print("  Space - up")
        print("  Shift - down")
        print("  Q - quit")
        print("  P - pause/resume training")
        print("  H - toggle heatmap")
        print("  G - toggle grid overlay")
        print("  +/- - add/remove agents")
        print("  F - fullscreen")
        print("  M - show/hide cursor")
        print("  Alt - toggle freeze (cursor + camera)")
        print("  Scroll - zoom in/out")
        print("  R - reset zoom")
        print("  1/2 - scale objects")
    
    def get_current_monitor(self) -> str:
        """Возвращает название текущего монитора."""
        return self.current_monitor
    
    def set_size(self, size: tuple) -> None:
        """Устанавливает размер окна."""
        window.size = size
    
    def set_position(self, position: tuple) -> None:
        """Устанавливает позицию окна."""
        window.position = position
    
    def set_background_color(self, a_color: color) -> None:
        """Устанавливает цвет фона."""
        window.color = a_color
    
    def apply_window_position(self) -> None:
        """Применяет сохраненную позицию и размер окна."""
        if hasattr(self, '_pending_position') and hasattr(self, '_pending_size'):
            try:
                pos = self._pending_position
                size = self._pending_size
                
                # Устанавливаем размер и позицию
                window.size = size
                window.position = pos
                
                # Попытка через Panda3D напрямую (Ursina использует Panda3D)
                if hasattr(window, 'win') and window.win:
                    props = window.win.getProperties()
                    props.setOrigin(pos[0], pos[1])
                    props.setSize(size[0], size[1])
                    window.win.requestProperties(props)
                    
                print(f"Window size set to: {size}")
                print(f"Window position set to: {pos}")
            except Exception as e:
                print(f"Warning: Could not set window size/position: {e}")
                print(f"  Attempted size: {self._pending_size}")
                print(f"  Attempted position: {self._pending_position}")
    
    @classmethod
    def setup_before_app(cls, monitor: str = "left"):
        """
        Статический метод для установки размера и позиции ДО создания app = Ursina()
        Используйте этот метод в main.py ПЕРЕД app = Ursina()
        
        Args:
            monitor: Тип монитора ("main", "top", "left", "down")
        """
        config = cls.MONITORS.get(monitor, cls.MONITORS["main"])
        window.size = config["size"]
        window.position = config["position"]
        print(f"[WindowManager] Setup for monitor '{monitor}': size={config['size']}, position={config['position']}")

