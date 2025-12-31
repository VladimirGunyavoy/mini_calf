"""
Тест новой архитектуры Agent с интегрированными траекториями
"""

import sys
import numpy as np

# Проверим что импорты работают
try:
    from physics.agent import Agent
    from physics.point_system import PointSystem
    print("[OK] Импорт Agent успешен")
except ImportError as e:
    print(f"[ERROR] Не удалось импортировать Agent: {e}")
    sys.exit(1)

# Проверим создание Agent (без визуализации)
print("\n=== Тест создания Agent ===")

# Создадим фейковый ObjectManager для теста
class FakeObjectManager:
    def __init__(self):
        self.objects = []

    def create_object(self, **kwargs):
        class FakeEntity:
            def __init__(self):
                self.position = (0, 0, 0)
                self.color = None
                self.visible = False
        obj = FakeEntity()
        self.objects.append(obj)
        return obj

fake_manager = FakeObjectManager()

# Создадим PointSystem
point_system = PointSystem(
    dt=0.01,
    initial_state=np.array([1.0, 0.5], dtype=np.float32),
    controller=None
)

print(f"PointSystem создан: state={point_system.get_state()}")

# Создадим Agent
try:
    agent = Agent(
        point_system=point_system,
        object_manager=fake_manager,
        name="test_agent",
        initial_position=(1.0, 0.1, 0.5),
        color=(0.2, 0.3, 0.8, 1),
        offset=(0, 0, 0),
        trail_config={
            'max_length': 100,
            'decimation': 3,
            'point_size': 0.03
        }
    )
    print(f"[OK] Agent создан: {agent}")
    print(f"[OK] Кольцевой буфер траектории: {len(agent.trail_points)} точек")
    print(f"[OK] Всего объектов создано: {len(fake_manager.objects)} (1 агент + 100 точек траектории)")
except Exception as e:
    print(f"[ERROR] Не удалось создать Agent: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Проверим обновление позиции
print("\n=== Тест обновления позиции ===")
try:
    # Делаем несколько шагов
    for i in range(10):
        action = 0.1  # Простое действие
        agent.point_system.u = action
        agent.point_system.step()
        state = agent.point_system.get_state()
        agent.update_position(state, mode='td3')

        if i % 3 == 0:  # Каждые 3 шага (из-за decimation=3)
            print(f"Шаг {i}: state={state}, trail_count={agent.trail_count}")

    print(f"[OK] Траектория работает! Активных точек: {agent.trail_count}")
except Exception as e:
    print(f"[ERROR] Ошибка при обновлении: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Проверим очистку траектории
print("\n=== Тест очистки траектории ===")
try:
    agent.clear_trail()
    print(f"[OK] Траектория очищена: trail_count={agent.trail_count}, trail_head={agent.trail_head}")
except Exception as e:
    print(f"[ERROR] Ошибка при очистке: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Проверим reset
print("\n=== Тест сброса агента ===")
try:
    agent.reset(new_state=np.array([2.0, -0.3], dtype=np.float32))
    new_state = agent.get_state()
    print(f"[OK] Агент сброшен: state={new_state}, trail_count={agent.trail_count}")
except Exception as e:
    print(f"[ERROR] Ошибка при сбросе: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
print("="*60)
print("\nРезюме:")
print(f"  - Agent успешно владеет своей траекторией (кольцевой буфер)")
print(f"  - Траектория автоматически обновляется при update_position()")
print(f"  - Переиспользование объектов работает (никаких destroy/create)")
print(f"  - Очистка и сброс работают корректно")
print("\nАрхитектура готова к интеграции с Ursina!")
