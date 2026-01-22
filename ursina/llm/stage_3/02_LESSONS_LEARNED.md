# Выученные уроки - Stage 3

**Формат записи**:
```
## Lesson N: [Дата YYYY-MM-DD] Краткое название

**Проблема**: Что случилось
**Решение**: Как исправили
**Вывод**: Что запомнить на будущее
```

---

# Уроки из Stage 2 (для reference)

Полный список см. в `../stage_2/02_LESSONS_LEARNED.md`.

**Ключевые для Stage 3**:

- **Lesson 4**: numpy array к скаляру - используй `float(u.item())` или `float(u[0])`
- **Lesson 10**: update() в Ursina должна быть глобальной функцией
- **Lesson 12**: Цвета через Vec4(0-1, 0-1, 0-1, 1)
- **Lesson 16**: Объекты для zoom регистрировать через ObjectManager
- **Lesson 20**: Dataclass для конфигураций
- **Lesson 21**: Избегать Unicode/emoji в консоли Windows

---

# Новые уроки из Stage 3

Записывай новые уроки ниже в том же формате.

---

## Lesson 28: [2026-01-22] add_visual_agent() создавал неправильный тип среды

**Проблема**: При нажатии `+` для добавления агента в режиме `differential_drive` создавался `PointMassEnv` вместо `DifferentialDriveEnv`. Это приводило к ошибке:
```
ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions.
```

**Решение**: В методе `add_visual_agent()` в `visualizer.py` добавить проверку `self.system_type`:
```python
if self.system_type == 'differential_drive':
    new_env = DifferentialDriveEnv(...)
else:
    new_env = PointMassEnv(...)
```
И передавать `system_type` в `VisualAgent`.

**Вывод**: При динамическом создании объектов всегда проверяй `system_type` для выбора правильного класса.

---

## Lesson 29: [2026-01-22] Скользящее среднее на графике - адаптивное окно

**Проблема**: При `window_size=10` и `update_frequency=5` скользящее среднее не появлялось первые 10 эпизодов, т.к. условие `len(data) >= window_size` не выполнялось.

**Решение**: Использовать адаптивное окно:
```python
if len(rewards) >= 2:
    effective_window = min(self.window_size, len(rewards))
    moving_avg = self._moving_average(rewards, effective_window)
```

**Вывод**: Для графиков в начале обучения используй адаптивные параметры чтобы визуализация была полезной сразу.

---

## Lesson 30: [2026-01-22] RewardPlotter - сохранение графиков в PNG

**Проблема**: Нужен способ отслеживать прогресс обучения без 3D визуализации в Ursina.

**Решение**: Создать `training/reward_plotter.py` который:
- Сохраняет reward/length/success в списки
- Периодически создаёт PNG график с matplotlib
- Использует `matplotlib.use('Agg')` для headless режима

**Вывод**: Для мониторинга обучения удобно сохранять PNG графики которые можно смотреть в отдельном окне.

---

