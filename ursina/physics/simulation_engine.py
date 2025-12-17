"""
Simulation Engine - Движок симуляции физических систем
=======================================================

Engine для выполнения симуляции физических/математических систем.
Хранит математические объекты (PointSystem, контроллеры) и вызывает step() для физики.
НЕ знает про визуализацию - только математика и физика.

Разделение ответственности:
- SimulationEngine: управляет ТОЛЬКО математическими объектами
- GeneralObjectManager: связывает математику с визуализацией
"""

from typing import List, Dict, Optional, Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from core.state_buffer import StateBuffer


class SimulationEngine:
    """
    Engine для симуляции физических/математических систем.

    Отвечает ТОЛЬКО за математику:
    - Хранит математические объекты (PointSystem, контроллеры)
    - Вызывает step() для физических систем
    - НЕ знает про визуализацию

    GeneralObjectManager использует SimulationEngine для создания math объектов
    и связывает их с визуальными представлениями.
    """

    def __init__(self, state_buffer: Optional['StateBuffer'] = None):
        """
        Инициализация SimulationEngine.

        Parameters:
        -----------
        state_buffer : StateBuffer, optional
            Буфер для записи состояний (для развязки симуляции и визуализации).
            Если None, работает в обычном режиме (обратная совместимость).
        """
        # Список математических объектов для обновления
        self.math_objects: List[Any] = []

        # Словарь для хранения объектов с именами (опционально)
        self.named_objects: Dict[str, Any] = {}

        # Опциональный буфер состояний (Phase 2)
        self.state_buffer: Optional['StateBuffer'] = state_buffer

        print("[OK] SimulationEngine initialized")
    
    def create_object(self, obj_type: type, name: Optional[str] = None, **kwargs) -> Any:
        """
        Создать математический объект и зарегистрировать его для обновления.
        
        Parameters:
        -----------
        obj_type : type
            Тип объекта для создания (например, PointSystem)
        name : str, optional
            Имя объекта для идентификации. Если не указано, будет назначено автоматически
            в виде порядкового номера (obj_0, obj_1, ...)
        **kwargs
            Параметры для конструктора объекта
            
        Returns:
        --------
        obj : Any
            Созданный и зарегистрированный объект
        """
        # Создаем объект
        obj = obj_type(**kwargs)
        
        # Проверяем наличие метода step()
        if not hasattr(obj, 'step'):
            raise ValueError(f"Объект {obj_type.__name__} не имеет метода step()")
        
        # Если имя не указано, генерируем автоматически на основе порядкового номера
        if name is None:
            name = f"obj_{len(self.math_objects)}"
        
        # Регистрируем объект
        if name in self.named_objects:
            print(f"⚠️ Объект с именем '{name}' уже существует, будет перезаписан")
        
        self.math_objects.append(obj)
        self.named_objects[name] = obj
        
        print(f"📐 Математический объект '{name}' ({obj_type.__name__}) создан и зарегистрирован")
        
        return obj
    
    def register_object(self, obj: Any, name: Optional[str] = None) -> None:
        """
        Зарегистрировать математический объект для обновления.
        
        Объект должен иметь метод step() для выполнения шага интегрирования.
        
        Parameters:
        -----------
        obj : Any
            Математический объект (например, PointSystem)
        name : str, optional
            Имя объекта для идентификации. Если не указано, объект добавляется без имени.
        """
        if not hasattr(obj, 'step'):
            raise ValueError(f"Объект {type(obj).__name__} не имеет метода step()")
        
        self.math_objects.append(obj)
        
        if name is not None:
            if name in self.named_objects:
                print(f"[WARNING] Object with name '{name}' already exists, will be overwritten")
            self.named_objects[name] = obj
            print(f"[OK] Math object '{name}' registered in SimulationEngine")
        else:
            print(f"[OK] Math object {type(obj).__name__} registered in SimulationEngine")
    
    def unregister_object(self, obj: Any = None, name: Optional[str] = None) -> None:
        """
        Удалить математический объект из списка обновления.
        
        Можно указать либо объект, либо имя объекта.
        
        Parameters:
        -----------
        obj : Any, optional
            Объект для удаления
        name : str, optional
            Имя объекта для удаления
        """
        if name is not None:
            if name in self.named_objects:
                obj = self.named_objects.pop(name)
                if obj in self.math_objects:
                    self.math_objects.remove(obj)
                print(f"[OK] Math object '{name}' removed from SimulationEngine")
            else:
                print(f"[WARNING] Object with name '{name}' not found")
        elif obj is not None:
            if obj in self.math_objects:
                self.math_objects.remove(obj)
            # Удаляем из словаря, если есть
            names_to_remove = [name for name, o in self.named_objects.items() if o is obj]
            for name in names_to_remove:
                del self.named_objects[name]
            print(f"[OK] Math object {type(obj).__name__} removed from SimulationEngine")
        else:
            print("[WARNING] Must specify either obj or name")
    
    def update_all(self) -> None:
        """
        Обновить все зарегистрированные математические объекты.

        Вызывает метод step() для каждого объекта в списке.
        Если установлен state_buffer, записывает состояния в буфер после обновления.
        """
        for name, obj in self.named_objects.items():
            try:
                if hasattr(obj, 'step'):
                    obj.step()

                    # Если есть буфер - пишем состояние после step()
                    if self.state_buffer and hasattr(obj, 'get_state'):
                        state = obj.get_state()
                        self.state_buffer.write(name, state)

            except Exception as e:
                print(f"[ERROR] Error updating {type(obj).__name__}: {e}")
    
    def get_object(self, name: str) -> Optional[Any]:
        """
        Получить объект по имени.
        
        Parameters:
        -----------
        name : str
            Имя объекта
            
        Returns:
        --------
        obj : Any, optional
            Объект с указанным именем или None, если не найден
        """
        return self.named_objects.get(name)
    
    def print_stats(self) -> None:
        """Print stats of registered math objects"""
        print("\n--- Simulation Engine Stats ---")
        print(f"  Total objects: {len(self.math_objects)}")
        print(f"  Named objects: {len(self.named_objects)}")
        if self.named_objects:
            print("  Object names:")
            for name in self.named_objects.keys():
                obj = self.named_objects[name]
                print(f"    - {name}: {type(obj).__name__}")
        print("----------------------------")
