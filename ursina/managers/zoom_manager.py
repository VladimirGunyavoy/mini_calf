"""
Менеджер масштабирования и трансформаций
Управляет масштабированием объектов относительно точки взгляда камеры
"""

from ursina import *
import numpy as np
from typing import Dict, Optional, Tuple, List, Callable
from utils.scalable import Scalable


class ZoomManager:
    def __init__(self, player=None):
        """
        Args:
            player: Ссылка на объект игрока (для вычисления invariant point)
        """
        self.player = player

        # Параметры масштабирования
        self.zoom_fact: float = 1.125  # Множитель масштабирования (1 + 1/8)

        # Текущие трансформации
        self.a_transformation: float = 1.0  # Масштаб
        self.b_translation: np.ndarray = np.array([0, 0, 0], dtype=float)  # Смещение

        # Дополнительный масштаб для специальных объектов (например, спор)
        self.objects_scale: float = 1.0

        # Зарегистрированные объекты
        self.objects: Dict[str, Scalable] = {}

        # Инвариантная точка (точка, относительно которой происходит масштабирование)
        self.invariant_point: Tuple[float, float] = (0, 0)

        # Счетчик для уникальных ID
        self._global_counter = 0

        # Флаг вывода отладочной информации
        self.auto_print_enabled = False

        # Подписчики на изменения look_point
        self.look_point_subscribers: List[Callable[[float, float], None]] = []

    def register_object(self, obj: Scalable, name: Optional[str] = None) -> None:
        """
        Регистрирует объект в системе масштабирования

        Args:
            obj: Объект, наследующий Scalable
            name: Имя объекта (опционально)
        """
        if name is None:
            self._global_counter += 1
            name = f"obj_{self._global_counter}"

        self.objects[name] = obj

        # Устанавливаем ссылку на ZoomManager в объекте (если есть метод set_zoom_manager)
        if hasattr(obj, 'set_zoom_manager'):
            obj.set_zoom_manager(self)

        obj.apply_transform(self.a_transformation, self.b_translation, spores_scale=self.objects_scale)

        if self.auto_print_enabled:
            print(f"Зарегистрирован объект: {name} ({type(obj).__name__})")

    def unregister_object(self, name: str) -> None:
        """Удаляет объект из менеджера масштабирования"""
        if name in self.objects:
            del self.objects[name]
            if self.auto_print_enabled:
                print(f"Удален объект: {name}")

    def identify_invariant_point(self) -> Tuple[float, float]:
        """
        Вычисляет инвариантную точку - точку на земле, куда смотрит камера.
        Эта точка остается неподвижной при масштабировании.

        Returns:
            Координаты (x, z) точки взгляда на плоскости y=0
        """
        if not self.player:
            return (0, 0)

        # Углы камеры
        psi = np.radians(self.player.rotation_y)  # Горизонтальное вращение
        phi = np.radians(self.player.camera_pivot.rotation_x)  # Вертикальное вращение

        # Высота камеры
        h = self.player.camera_pivot.world_position.y

        # Избегаем деления на ноль
        if abs(np.tan(phi)) < 0.001:
            return (0, 0)

        # Расстояние до точки на земле
        d = h / np.tan(phi)

        # Смещение от позиции игрока
        dx = d * np.sin(psi)
        dz = d * np.cos(psi)

        # Координаты точки взгляда
        x_0 = self.player.camera_pivot.world_position.x + dx
        z_0 = self.player.camera_pivot.world_position.z + dz

        # Сохраняем для использования в change_zoom
        self.invariant_point = (x_0, z_0)

        # Уведомляем подписчиков
        self._notify_look_point_change(x_0, z_0)

        return (x_0, z_0)

    def update_transform(self) -> None:
        """Обновляет трансформации для всех зарегистрированных объектов"""
        for name, obj in list(self.objects.items()):
            try:
                if hasattr(obj, 'enabled') and obj.enabled:
                    obj.apply_transform(
                        self.a_transformation,
                        self.b_translation,
                        spores_scale=self.objects_scale
                    )
            except (AssertionError, AttributeError, RuntimeError, Exception) as e:
                # Объект невалиден - удаляем из списка
                if self.auto_print_enabled:
                    print(f"Удален невалидный объект: {name} (Error: {type(e).__name__})")
                del self.objects[name]

    def change_zoom(self, sign: int) -> None:
        """
        Изменяет масштаб относительно инвариантной точки

        Args:
            sign: Направление масштабирования (1 = приблизить, -1 = отдалить)
        """
        # Вычисляем инвариантную точку
        inv_2d = np.array(self.identify_invariant_point())
        inv_3d = np.array([inv_2d[0], 0, inv_2d[1]])

        # Множитель масштабирования
        zoom_multiplier = self.zoom_fact ** sign

        # Обновляем трансформации
        self.a_transformation *= zoom_multiplier
        self.b_translation = zoom_multiplier * self.b_translation + (1 - zoom_multiplier) * inv_3d

        # Применяем к объектам
        self.update_transform()

        if self.auto_print_enabled:
            print(f"Масштаб: {self.a_transformation:.3f}, Смещение: {self.b_translation}")

    def scale_objects(self, sign: int) -> None:
        """
        Изменяет дополнительный масштаб объектов (без изменения позиций)

        Args:
            sign: Направление масштабирования (1 = увеличить, -1 = уменьшить)
        """
        self.objects_scale *= self.zoom_fact ** sign
        self.update_transform()

        if self.auto_print_enabled:
            print(f"Масштаб объектов: {self.objects_scale:.3f}")

    def reset_all(self) -> None:
        """Сбрасывает все трансформации к исходному состоянию"""
        self.a_transformation = 1.0
        self.b_translation = np.array([0, 0, 0], dtype=float)
        self.objects_scale = 1.0
        self.update_transform()

        if self.auto_print_enabled:
            print("Трансформации сброшены")

    # ===== Удобные методы =====

    def zoom_in(self) -> None:
        """Приближает масштаб"""
        self.change_zoom(1)

    def zoom_out(self) -> None:
        """Отдаляет масштаб"""
        self.change_zoom(-1)

    def reset_zoom(self) -> None:
        """Сбрасывает масштаб"""
        self.reset_all()

    def increase_objects_scale(self) -> None:
        """Увеличивает масштаб объектов"""
        self.scale_objects(1)

    def decrease_objects_scale(self) -> None:
        """Уменьшает масштаб объектов"""
        self.scale_objects(-1)

    # ===== Система подписок =====

    def subscribe_look_point_change(self, callback: Callable[[float, float], None]) -> None:
        """
        Подписывается на изменения look_point

        Args:
            callback: Функция callback(x, z)
        """
        self.look_point_subscribers.append(callback)

    def unsubscribe_look_point_change(self, callback: Callable[[float, float], None]) -> None:
        """Отписывается от изменений look_point"""
        if callback in self.look_point_subscribers:
            self.look_point_subscribers.remove(callback)

    def _notify_look_point_change(self, x: float, z: float) -> None:
        """Уведомляет подписчиков об изменении look_point"""
        for callback in self.look_point_subscribers:
            try:
                callback(x, z)
            except Exception as e:
                if self.auto_print_enabled:
                    print(f"Ошибка в подписчике look_point: {e}")

    # ===== Утилиты =====

    def enable_auto_print(self) -> None:
        """Включает автоматический вывод отладочной информации"""
        self.auto_print_enabled = True

    def disable_auto_print(self) -> None:
        """Выключает автоматический вывод отладочной информации"""
        self.auto_print_enabled = False

    def print_stats(self) -> None:
        """Выводит статистику по зарегистрированным объектам"""
        print("\n=== ZoomManager Статистика ===")
        print(f"Объектов: {len(self.objects)}")
        print(f"Масштаб: {self.a_transformation:.3f}")
        print(f"Смещение: {self.b_translation}")
        print(f"Масштаб объектов: {self.objects_scale:.3f}")
        print(f"Инвариантная точка: {self.invariant_point}")

        if self.objects:
            print("\nТипы объектов:")
            types_count = {}
            for obj in self.objects.values():
                obj_type = type(obj).__name__
                types_count[obj_type] = types_count.get(obj_type, 0) + 1
            for obj_type, count in types_count.items():
                print(f"  {obj_type}: {count}")

        print("=" * 30 + "\n")
