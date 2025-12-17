"""
Visuals Update Manager - Централизованное управление обновлениями
========================================================

Класс для централизованного вызова методов обновления каждый кадр.
Воспроизводит логику из стабильной версии проекта.
"""

from typing import Optional, TYPE_CHECKING
import time

if TYPE_CHECKING:
    from .ui_manager import UIManager
    from .input_manager import InputManager
    from .zoom_manager import ZoomManager
    from .object_manager import ObjectManager
    from .general_object_manager import GeneralObjectManager


class VisualsUpdateManager:
    """
    Централизованный класс для вызова методов обновления каждый кадр.
    """
    
    def __init__(self,
                 ui_manager: Optional['UIManager'] = None,
                 input_manager: Optional['InputManager'] = None,
                 zoom_manager: Optional['ZoomManager'] = None,
                 object_manager: Optional['ObjectManager'] = None,
                 general_object_manager: Optional['GeneralObjectManager'] = None):
        """
        Инициализация VisualsUpdateManager с менеджерами, которые нужно обновлять.
        
        Args:
            ui_manager: Менеджер UI элементов
            input_manager: Менеджер ввода
            zoom_manager: Менеджер зума
            object_manager: Менеджер объектов
            general_object_manager: Менеджер общих объектов (математика + визуализация)
        """
        self.ui_manager: Optional['UIManager'] = ui_manager
        self.input_manager: Optional['InputManager'] = input_manager
        self.zoom_manager: Optional['ZoomManager'] = zoom_manager
        self.object_manager: Optional['ObjectManager'] = object_manager
        self.general_object_manager: Optional['GeneralObjectManager'] = general_object_manager
        
        print("✅ VisualsUpdateManager initialized")
    
    def update_all(self) -> None:
        """
        Основной метод, который должен вызываться каждый кадр из главного цикла.
        Обновляет все зарегистрированные менеджеры в правильном порядке.
        """
        # 1. Input Manager - обрабатываем ввод первым
        if self.input_manager and hasattr(self.input_manager, 'update'):
            try:
                self.input_manager.update()
            except Exception as e:
                print(f"⚠️ Ошибка в input_manager.update(): {e}")
        
        # 2. Zoom Manager - обновляем зум и инвариантную точку
        if self.zoom_manager and hasattr(self.zoom_manager, 'update'):
            try:
                self.zoom_manager.update()
            except Exception as e:
                print(f"⚠️ Ошибка в zoom_manager.update(): {e}")
        
        # 3. Object Manager - обновляем объекты сцены
        if self.object_manager and hasattr(self.object_manager, 'update'):
            try:
                self.object_manager.update()
            except Exception as e:
                print(f"⚠️ Ошибка в object_manager.update(): {e}")
        
        # 4. UI Manager - обновляем UI элементы последними
        if self.ui_manager:
            try:
                self.ui_manager.update()
            except Exception as e:
                print(f"⚠️ Ошибка в ui_manager.update(): {e}")
    
    def register_ui_manager(self, ui_manager: 'UIManager') -> None:
        """Регистрирует UI Manager"""
        self.ui_manager = ui_manager
        print("📋 UI Manager зарегистрирован в VisualsUpdateManager")
    
    def register_input_manager(self, input_manager: 'InputManager') -> None:
        """Регистрирует Input Manager"""
        self.input_manager = input_manager
        print("⌨️ Input Manager зарегистрирован в VisualsUpdateManager")
    
    def register_zoom_manager(self, zoom_manager: 'ZoomManager') -> None:
        """Регистрирует Zoom Manager"""
        self.zoom_manager = zoom_manager
        print("🔍 Zoom Manager зарегистрирован в VisualsUpdateManager")
    
    def register_object_manager(self, object_manager: 'ObjectManager') -> None:
        """Регистрирует Object Manager"""
        self.object_manager = object_manager
        print("📦 Object Manager зарегистрирован в VisualsUpdateManager")
    
    def print_stats(self) -> None:
        """Выводит статистику зарегистрированных менеджеров"""
        print("\n--- Visuals Update Manager Stats ---")
        print(f"  UI Manager: {'✅' if self.ui_manager else '❌'}")
        print(f"  Input Manager: {'✅' if self.input_manager else '❌'}")
        print(f"  Zoom Manager: {'✅' if self.zoom_manager else '❌'}")
        print(f"  Object Manager: {'✅' if self.object_manager else '❌'}")
        print("----------------------------")
