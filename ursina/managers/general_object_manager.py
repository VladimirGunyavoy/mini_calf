"""
General Object Manager - создание и управление объектами, связывающими математику и визуализацию
"""

from typing import Dict, Optional, Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .object_manager import ObjectManager
    from .zoom_manager import ZoomManager

# Импорт модуля physics (физические системы)
from physics import SimulationEngine, PointSystem
from visuals.general_object import GeneralObject
from visuals.point_visual import PointVisual


class GeneralObjectManager:
    """
    Менеджер для создания и управления объектами, связывающими математические и визуальные объекты.

    Разделение ответственности с SimulationEngine:
    - SimulationEngine: управляет ТОЛЬКО математическими объектами (не знает про визуализацию)
    - GeneralObjectManager: связывает математику с визуализацией

    Создает математические объекты через SimulationEngine,
    визуальные объекты через ObjectManager,
    и связывает их в GeneralObject.
    """

    def __init__(self,
                 simulation_engine: 'SimulationEngine',
                 object_manager: 'ObjectManager',
                 zoom_manager: Optional['ZoomManager'] = None):
        """
        Инициализация GeneralObjectManager

        Parameters:
        -----------
        simulation_engine : SimulationEngine
            Engine для симуляции математических объектов
        object_manager : ObjectManager
            Менеджер для создания визуальных объектов
        zoom_manager : ZoomManager, optional
            Менеджер масштабирования (передается в object_manager)
        """
        self.simulation_engine = simulation_engine
        self.object_manager = object_manager
        self.zoom_manager = zoom_manager
        
        # Словарь для хранения созданных объектов
        self.objects: Dict[str, GeneralObject] = {}
        
        print("[OK] GeneralObjectManager initialized")
    
    def create_object(self,
                     name: str,
                     math_obj_type: type = PointSystem,
                     math_params: Optional[Dict[str, Any]] = None,
                     visual_obj_type: type = PointVisual,
                     visual_params: Optional[Dict[str, Any]] = None) -> GeneralObject:
        """
        Создать объект, связывающий математический и визуальный объекты.
        
        Parameters:
        -----------
        name : str
            Имя объекта
        math_obj_type : type
            Тип математического объекта (по умолчанию PointSystem)
        math_params : dict, optional
            Параметры для создания математического объекта
        visual_obj_type : type
            Тип визуального объекта (по умолчанию PointVisual)
        visual_params : dict, optional
            Параметры для создания визуального объекта
            
        Returns:
        --------
        GeneralObject
            Созданный объект, связывающий математику и визуализацию
        """
        if math_params is None:
            math_params = {}
        if visual_params is None:
            visual_params = {}
        
        # Создаем математический объект через SimulationEngine
        math_obj = self.simulation_engine.create_object(
            math_obj_type,
            name=f"{name}_math",
            **math_params
        )
        
        # Получаем начальное состояние для визуального объекта
        initial_state = math_obj.get_state()
        if len(initial_state) >= 1:
            x = initial_state[0]
            initial_position = np.array([x, 0.0, 0.0], dtype=np.float32)
        else:
            initial_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        # Устанавливаем начальную позицию в visual_params, если не указана
        if 'position' not in visual_params:
            visual_params['position'] = initial_position
        
        # Создаем визуальный объект
        visual_obj = visual_obj_type(**visual_params)
        
        # Регистрируем визуальный объект в ObjectManager
        self.object_manager.register_existing(f"{name}_visual", visual_obj)
        
        # Создаем GeneralObject
        general_obj = GeneralObject(math_obj, visual_obj)
        
        # Сохраняем объект
        self.objects[name] = general_obj
        
        print(f"[OK] GeneralObject '{name}' created and registered")
        
        return general_obj
    
    def update_all(self) -> None:
        """
        Синхронизировать визуальные объекты с математическими объектами.

        Вызывается каждый кадр:
        - Обновляет визуальные объекты на основе состояний математических объектов

        Note: Математические объекты обновляются SimulationEngine.update_all(),
              а не здесь. GeneralObjectManager только связывает math ↔ visual.
        """
        # Обновляем визуальные объекты на основе состояний математических объектов
        for name, obj in self.objects.items():
            try:
                obj.update_visual()
            except Exception as e:
                print(f"[ERROR] Error updating GeneralObject '{name}' visualization: {e}")
    
    def get_object(self, name: str) -> Optional[GeneralObject]:
        """
        Получить объект по имени.
        
        Parameters:
        -----------
        name : str
            Имя объекта
            
        Returns:
        --------
        GeneralObject, optional
            Объект с указанным именем или None, если не найден
        """
        return self.objects.get(name)
    
    def print_stats(self) -> None:
        """Print stats of created objects"""
        print("\n--- General Object Manager Stats ---")
        print(f"  Total objects: {len(self.objects)}")
        if self.objects:
            print("  Object names:")
            for name in self.objects.keys():
                print(f"    - {name}")
        print("----------------------------")
