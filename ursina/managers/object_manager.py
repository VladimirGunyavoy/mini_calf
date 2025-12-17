"""
ObjectManager - Централизованное управление объектами сцены
Автоматическая регистрация объектов в ZoomManager
"""

from typing import Dict, Optional, Any, List
from ursina import Entity, color, destroy
import numpy as np

from utils.scalable import Scalable
from .zoom_manager import ZoomManager


class ObjectManager:
    """
    Менеджер для управления всеми объектами сцены
    - Создание объектов через фабричные методы
    - Автоматическая регистрация в ZoomManager
    - Централизованное хранилище объектов
    """
    
    def __init__(self, zoom_manager: Optional[ZoomManager] = None):
        """
        Args:
            zoom_manager: ZoomManager для автоматической регистрации объектов
        """
        self.zoom_manager = zoom_manager
        self.objects: Dict[str, Entity] = {}
        
        print(f"[ObjectManager] Initialized with ZoomManager: {zoom_manager is not None}")
    
    def set_zoom_manager(self, zoom_manager: ZoomManager):
        """Установить ZoomManager (если не был передан в конструктор)"""
        self.zoom_manager = zoom_manager
        print(f"[ObjectManager] ZoomManager set")
    
    def create_object(self, name: str, model: str, position: tuple, 
                     scale: Any, color_val: Any, auto_register: bool = True,
                     **kwargs) -> Scalable:
        """
        Создать объект и автоматически зарегистрировать в ZoomManager
        
        Args:
            name: Уникальное имя объекта
            model: Модель ('cube', 'sphere', 'assets/arrow.obj' и т.д.)
            position: Позиция (x, y, z)
            scale: Масштаб (число или tuple/array)
            color_val: Цвет (color.red, color.blue и т.д.)
            auto_register: Автоматически регистрировать в ZoomManager
            **kwargs: Дополнительные параметры для Scalable
            
        Returns:
            Созданный объект Scalable
        """
        # Обработка масштаба
        if isinstance(scale, (list, np.ndarray)):
            scale = tuple(scale)
        
        # Создать объект
        obj = Scalable(
            model=model,
            position=position,
            scale=scale,
            color=color_val,
            **kwargs
        )
        
        # Сохранить в словарь
        self.objects[name] = obj
        
        # Автоматическая регистрация в ZoomManager
        if auto_register and self.zoom_manager:
            self.zoom_manager.register_object(obj, name)
            print(f"[ObjectManager] Created and registered '{name}'")
        else:
            print(f"[ObjectManager] Created '{name}' (not registered)")
        
        return obj
    
    def register_existing(self, name: str, obj: Entity) -> None:
        """
        Зарегистрировать существующий объект
        Полезно для объектов, созданных вне ObjectManager (ground, grid, frame и т.д.)
        
        Args:
            name: Имя объекта
            obj: Существующий объект Entity
        """
        self.objects[name] = obj
        
        if self.zoom_manager:
            self.zoom_manager.register_object(obj, name)
            print(f"[ObjectManager] Registered existing object '{name}'")
    
    def get_object(self, name: str) -> Optional[Entity]:
        """Получить объект по имени"""
        return self.objects.get(name)
    
    def remove_object(self, name: str) -> None:
        """Удалить объект"""
        if name in self.objects:
            obj = self.objects[name]
            destroy(obj)
            del self.objects[name]
            print(f"[ObjectManager] Removed '{name}'")
    
    def get_all_names(self) -> List[str]:
        """Получить список всех имен объектов"""
        return list(self.objects.keys())
    
    def count(self) -> int:
        """Получить количество объектов"""
        return len(self.objects)
    
    def show_all(self) -> None:
        """Показать все объекты"""
        for obj in self.objects.values():
            obj.enabled = True
        print(f"[ObjectManager] Showed all {len(self.objects)} objects")
    
    def hide_all(self) -> None:
        """Скрыть все объекты"""
        for obj in self.objects.values():
            obj.enabled = False
        print(f"[ObjectManager] Hid all {len(self.objects)} objects")
    
    def show_object(self, name: str) -> None:
        """Показать объект"""
        obj = self.get_object(name)
        if obj:
            obj.enabled = True
    
    def hide_object(self, name: str) -> None:
        """Скрыть объект"""
        obj = self.get_object(name)
        if obj:
            obj.enabled = False
    
    def clear_all(self) -> None:
        """Удалить все объекты"""
        for obj in list(self.objects.values()):
            destroy(obj)
        self.objects.clear()
        print(f"[ObjectManager] Cleared all objects")
    
    def print_stats(self) -> None:
        """Вывести статистику"""
        print(f"\n[ObjectManager] Stats:")
        print(f"  Total objects: {len(self.objects)}")
        print(f"  Objects: {', '.join(self.objects.keys())}")
        print(f"  ZoomManager: {'connected' if self.zoom_manager else 'not connected'}")
