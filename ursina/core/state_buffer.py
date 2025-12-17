"""
State Buffer - Буфер состояний для развязки симуляции и визуализации
====================================================================

Буфер для передачи состояний от симуляции к визуализации.
Пока простая реализация (dict), в Phase 12 станет thread-safe.

Архитектура:
- SimulationEngine пишет состояния в буфер после step()
- RenderEngine/Визуализация читает состояния из буфера
- Позволяет развязать симуляцию и рендеринг (подготовка к многопоточности)
"""

import numpy as np
from typing import Dict, Optional


class StateBuffer:
    """
    Буфер для передачи состояний от симуляции к визуализации.

    Пока простая реализация (dict), потом станет thread-safe (Phase 12).

    Usage:
    ------
    # Создание буфера
    buffer = StateBuffer()

    # Симуляция пишет (после step)
    buffer.write('point1', np.array([x, y, vx, vy]))

    # Визуализация читает
    state = buffer.read('point1')
    all_states = buffer.read_all()
    """

    def __init__(self):
        """Инициализация StateBuffer"""
        # Словарь {obj_id: state}
        # state - numpy array с состоянием объекта
        self._states: Dict[str, np.ndarray] = {}

        print("✅ StateBuffer initialized")

    def write(self, obj_id: str, state: np.ndarray) -> None:
        """
        Записать состояние объекта (вызывается из симуляции)

        Parameters:
        -----------
        obj_id : str
            Уникальный идентификатор объекта
        state : np.ndarray
            Состояние объекта (например, [x, y, vx, vy])
        """
        # Копируем состояние для безопасности
        self._states[obj_id] = state.copy()

    def read(self, obj_id: str) -> Optional[np.ndarray]:
        """
        Прочитать состояние объекта (для визуализации)

        Parameters:
        -----------
        obj_id : str
            Уникальный идентификатор объекта

        Returns:
        --------
        state : np.ndarray or None
            Состояние объекта или None если не найдено
        """
        return self._states.get(obj_id)

    def read_all(self) -> Dict[str, np.ndarray]:
        """
        Прочитать все состояния (для визуализации)

        Returns:
        --------
        states : dict
            Словарь {obj_id: state} со всеми состояниями
        """
        # Возвращаем копию для безопасности
        return self._states.copy()

    def clear(self) -> None:
        """Очистить все состояния"""
        self._states.clear()

    def remove(self, obj_id: str) -> None:
        """
        Удалить состояние объекта из буфера

        Parameters:
        -----------
        obj_id : str
            Идентификатор объекта для удаления
        """
        if obj_id in self._states:
            del self._states[obj_id]

    def has_state(self, obj_id: str) -> bool:
        """
        Проверить, есть ли состояние объекта в буфере

        Parameters:
        -----------
        obj_id : str
            Идентификатор объекта

        Returns:
        --------
        has : bool
            True если состояние есть в буфере
        """
        return obj_id in self._states

    def get_object_ids(self) -> list:
        """
        Получить список всех ID объектов в буфере

        Returns:
        --------
        ids : list
            Список идентификаторов объектов
        """
        return list(self._states.keys())

    def print_stats(self) -> None:
        """Выводит статистику буфера"""
        print("\n--- State Buffer Stats ---")
        print(f"  Объектов в буфере: {len(self._states)}")
        if self._states:
            print("  ID объектов:")
            for obj_id in self._states.keys():
                state = self._states[obj_id]
                print(f"    - {obj_id}: state shape {state.shape}")
        print("-------------------------")
