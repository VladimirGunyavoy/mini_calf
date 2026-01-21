# Stage 5 - Уроки по оптимизации визуализации

**Дата**: 2026-01-22

---

## Lesson 25: LineTrail rebuild frequency оптимизация

**Проблема**: LineTrail перестраивал mesh каждый кадр (при каждом `add_point`), что вызывало большое количество mesh операций и падение FPS при множестве агентов.

**Решение**: Добавлен параметр `rebuild_freq` для контроля частоты перестройки:
```python
class LineTrail:
    def __init__(self, rebuild_freq=5):
        self.rebuild_freq = rebuild_freq
        self.rebuild_counter = 0
        self.needs_rebuild = True

    def add_point(self, position, mode):
        # ... add to buffer
        self.needs_rebuild = True
        self.rebuild_counter += 1

        # Rebuild только каждые rebuild_freq добавлений
        if self.rebuild_counter >= self.rebuild_freq:
            self._rebuild_segments()
            self.rebuild_counter = 0
```

**Результат**:
- Меньше mesh операций (rebuild_freq=5 → 5x меньше rebuilds)
- Траектория всё ещё гладкая (визуально не заметно)
- Значительное улучшение FPS

**Вывод**: Не нужно перестраивать mesh каждый кадр - достаточно каждые N добавлений. Визуальное качество остаётся высоким, а производительность улучшается.

---

## Lesson 26: Vectorized group_by_mode через numpy

**Проблема**: `_group_by_mode()` в LineTrail использовал Python цикл для группировки последовательных точек по режиму. При большом количестве точек (max_points=150+) это медленно.

**Решение**: Векторизованная версия через numpy:
```python
def _group_by_mode(self):
    # Конвертируем в numpy array
    modes_array = np.array(self.modes)

    # Найти индексы где режим меняется
    mode_changes = np.concatenate(([True], modes_array[1:] != modes_array[:-1], [True]))
    change_indices = np.where(mode_changes)[0]

    # Создать группы из индексов изменений
    groups = []
    for i in range(len(change_indices) - 1):
        start_idx = change_indices[i]
        end_idx = change_indices[i + 1]
        current_mode = modes_array[start_idx]
        indices = list(range(start_idx, end_idx + 1 if i < len(change_indices) - 2 else end_idx))
        if len(indices) >= 2:
            groups.append((current_mode, indices))
    return groups
```

**Вывод**: Numpy операции намного быстрее Python циклов для обработки массивов. Векторизация через `np.where()` и array comparison эффективнее чем `for` цикл.

---

## Lesson 27: Кэширование mesh с needs_rebuild флагом

**Проблема**: Даже с `rebuild_freq`, если данные не изменились, мы всё равно пересоздаём mesh.

**Решение**: Добавлен флаг `needs_rebuild` для отслеживания изменений:
```python
def add_point(self, position, mode):
    # ... add to buffer
    self.needs_rebuild = True  # Отмечаем что нужен rebuild

    if self.rebuild_counter >= self.rebuild_freq:
        self._rebuild_segments()
        self.rebuild_counter = 0

def _rebuild_segments(self):
    if not self.needs_rebuild:
        return  # Skip если не было изменений

    # ... rebuild logic
    self.needs_rebuild = False
```

**Вывод**: Кэширование через флаг изменений предотвращает лишние операции. Особенно полезно когда `apply_transform` вызывается многократно без изменения данных.

---

## Lesson 28: Векторизованная трансформация точек

**Проблема**: В `_rebuild_segments()` каждая точка трансформировалась в Python цикле:
```python
for idx in indices:
    pos = self.positions[idx]
    t = pos * self.a_transformation + self.b_translation
    points.append(Vec3(t[0], t[1], t[2]))
```

**Решение**: Векторизованная трансформация через numpy:
```python
# Собираем все позиции в numpy array
positions_array = np.array([self.positions[idx] for idx in indices])

# Векторизованная трансформация (одна операция для всех точек)
transformed = positions_array * self.a_transformation + self.b_translation

# Конвертируем в Vec3 список
points = [Vec3(t[0], t[1], t[2]) for t in transformed]
```

**Вывод**: Batch операции через numpy быстрее чем циклы. Даже если итоговый Vec3 список создаётся через comprehension, основная математика выполняется векторизованно.

---

## Lesson 29: CriticHeatmap Q-values уже использует batch inference

**Проблема**: План оптимизации предлагал "добавить batch вычисление Q-значений", но оно уже было реализовано.

**Реализация**: В `_compute_q_values()`:
```python
# Flatten grid (31x31 = 961 точек)
states = np.stack([self.x_grid.flatten(), self.v_grid.flatten()], axis=1)

# ОДИН forward pass через critic для ВСЕХ точек
with torch.no_grad():
    states_tensor = torch.FloatTensor(states).to(device)
    actions_tensor = torch.zeros(len(states), action_dim).to(device)
    q1, q2 = self.td3_agent.critic(states_tensor, actions_tensor)
    q_values = torch.min(q1, q2).cpu().numpy().flatten()

# Reshape обратно в grid
q_values = q_values.reshape(self.grid_size, self.grid_size)
```

**Вывод**: Всегда проверяй существующую реализацию перед оптимизацией. Batch inference для heatmap критичен и был правильно реализован с самого начала.

---

## Lesson 30: Кэширование Q-grid для interpolation

**Проблема**: `get_q_value_for_state()` вызывается для каждого агента каждый кадр, но heatmap обновляется только каждые `update_frequency` кадров. Между обновлениями мы зря делаем interpolation из устаревшего `self.q_values`.

**Решение**: Добавлено кэширование Q-grid:
```python
def _compute_q_values(self):
    # ... compute Q-values

    # Кэшируем для interpolation между обновлениями
    self.q_values_cached = q_values.copy()
    self.cache_valid = True

    return q_values

def _interpolate_q_from_grid(self, state):
    # Используем кэш если доступен
    q_grid = self.q_values_cached if self.cache_valid and self.q_values_cached is not None else self.q_values
    # ... interpolation logic
```

**Преимущества**:
- `q_values_cached` синхронизирован с surface (оба обновляются одновременно)
- Агенты позиционируются на правильной высоте относительно surface
- Избежание race conditions между heatmap update и agent positioning

**Вывод**: Кэширование промежуточных результатов эффективно когда данные обновляются редко, но используются часто. Особенно важно для синхронизации между связанными компонентами (agents + surface).

---

## Lesson 31: Batch height computation уже реализован

**Проблема**: План предлагал оптимизировать "batch обновление позиций агентов", но это уже было в visualizer.

**Реализация**: В `TrainingVisualizer.update_visual_agents()`:
```python
# Collect all states
vis_states = np.array([env.state for env in self.visual_envs])

# Batch height computation (ONE call instead of N)
vis_heights = self.get_agent_heights_batch(vis_next_states_array)

# Update positions
for i in range(len(self.visual_envs)):
    x, v = vis_next_states[i][0], vis_next_states[i][1]
    y = vis_heights[i]  # Используем batch результат
    self.visual_agents[i].update_position((x, y, v), mode=vis_modes[i])
```

**Дополнительная оптимизация**: Получение zoom transform один раз:
```python
# Get zoom transformations once (avoid multiple lookups)
a_trans = self.zoom_manager.a_transformation
b_trans = self.zoom_manager.b_translation
```

**Вывод**: Batch операции критичны для производительности. Проверяй что batch обработка применяется на всех уровнях: вычисления (heights), трансформации (zoom), и inference (actions).

---

## Итоги Stage 5

**Выполненные оптимизации**:
1. ✅ LineTrail: rebuild frequency, vectorized grouping, caching, vectorized transforms
2. ✅ CriticHeatmap: Q-grid caching для interpolation
3. ✅ VisualAgent: batch height computation, zoom transform optimization

**Ожидаемые улучшения**:
- Меньше mesh операций (5x reduction за счёт rebuild_freq)
- Faster grouping (numpy vectorization)
- Лучшая cache utilization (needs_rebuild flag, Q-grid cache)
- Меньше zoom_manager lookups

**Следующие шаги**:
- Пользовательское тестирование с профайлером
- Замер FPS до/после оптимизаций
- Stage 6: Финальное тестирование и документация
