# Baseline Performance Metrics

## Цель
Установить baseline производительности перед оптимизацией для объективного сравнения результатов.

---

## Метрики для измерения

### 1. Frame Performance
- **FPS** (Frames Per Second) - основная метрика плавности
- **Frame Time** (ms) - время обработки одного кадра
- **Frame Time Breakdown**:
  - Physics simulation time
  - Visual updates time (agent positioning, trails)
  - Heatmap update time
  - UI update time
  - Input handling time

### 2. Resource Usage
- **CPU Usage %** - загрузка процессора
- **Memory Usage** (MB) - потребление RAM
- **GPU Queries Count** - количество обращений к critic network
- **Mesh Updates Count** - количество mesh rebuilds за кадр

### 3. Rendering Stats
- **Draw Calls** - количество draw calls за кадр
- **Vertices Count** - общее количество вершин в сцене
- **Active Entities** - количество активных Ursina объектов

### 4. Scalability
Измерить все метрики для разного количества агентов:
- N = 3 (low preset)
- N = 5 (medium preset)
- N = 10
- N = 20
- N = 30 (stress test)

---

## Методология измерения

### Setup
- **Preset**: `medium` (5 agents, 600 trail length)
- **Episode**: 200-500 steps (после warm-up)
- **System**: `point_mass` (проще для изоляции bottlenecks)
- **Measurement Duration**: 300 frames (5 секунд при 60 FPS)

### Tools
1. **Python `time.perf_counter()`** - для timing
2. **`psutil`** - для CPU/Memory usage
3. **Custom `PerformanceProfiler`** - для frame breakdown
4. **Ursina stats** - для draw calls (если доступно)

---

## Инструмент профилирования

Создать класс `PerformanceProfiler`:

```python
class PerformanceProfiler:
    def __init__(self):
        self.frame_times = []
        self.physics_times = []
        self.visual_times = []
        self.ui_times = []
        self.heatmap_times = []
        self.gpu_query_counts = []
        self.mesh_update_counts = []

    def start_frame(self):
        self.frame_start = time.perf_counter()

    def end_section(self, section_name):
        elapsed = time.perf_counter() - self.section_start
        getattr(self, f'{section_name}_times').append(elapsed)

    def get_stats(self):
        return {
            'fps_avg': 1.0 / np.mean(self.frame_times),
            'fps_min': 1.0 / np.max(self.frame_times),
            'frame_time_avg_ms': np.mean(self.frame_times) * 1000,
            'frame_time_p95_ms': np.percentile(self.frame_times, 95) * 1000,
            # ... и т.д.
        }
```

---

## Baseline Results Template

### Configuration
- **Date**: [YYYY-MM-DD]
- **Preset**: medium
- **System**: point_mass
- **Episodes**: 200-500
- **Hardware**: [CPU model, GPU model, RAM]

### Overall Performance

| N Agents | FPS Avg | FPS Min | Frame Time (ms) | Frame Time P95 (ms) |
|----------|---------|---------|-----------------|---------------------|
| 3        |         |         |                 |                     |
| 5        |         |         |                 |                     |
| 10       |         |         |                 |                     |
| 20       |         |         |                 |                     |
| 30       |         |         |                 |                     |

### Frame Time Breakdown (N=20)

| Component | Time (ms) | % of Frame |
|-----------|-----------|------------|
| Physics   |           |            |
| Visuals   |           |            |
| Heatmap   |           |            |
| UI        |           |            |
| Input     |           |            |
| Other     |           |            |
| **Total** |           | **100%**   |

### Resource Usage (N=20)

| Metric | Value |
|--------|-------|
| CPU Usage % |  |
| Memory (MB) |  |
| GPU Queries/frame |  |
| Mesh Updates/frame |  |

### Rendering Stats (N=20)

| Metric | Value |
|--------|-------|
| Draw Calls/frame |  |
| Total Vertices |  |
| Active Entities |  |

---

## Bottleneck Analysis

### Expected Bottlenecks

1. **Visual Updates** (~40-50% frame time)
   - Height computation for each agent
   - Per-agent interpolation loops
   - Trail mesh rebuilds

2. **UI Updates** (~10-20% frame time)
   - Q-value queries каждый кадр
   - Text entity updates

3. **Heatmap Updates** (spiked, но редкие)
   - Batch Q-value computation (961 points)
   - Mesh rebuild

### Profiling Points

Добавить timing в:
- `TrainingVisualizer.update_visual_agents()` - визуальные агенты
- `TrainingVisualizer.update_stats_display()` - UI обновление
- `CriticHeatmap.update()` - heatmap обновление
- `CriticHeatmap.get_q_value_for_states_batch()` - height computation

---

## Commands для запуска

### Baseline с N=5 (medium)
```bash
python ursina/main.py --system point_mass --profile --n_agents 5
```

### Baseline с N=20 (stress test)
```bash
python ursina/main.py --system point_mass --profile --n_agents 20
```

### Сохранение результатов
Результаты сохраняются в:
```
ursina/llm/stage_4/baseline_results_YYYYMMDD_HHMMSS.json
```

---

## Следующие шаги

1. ✅ Создать `PerformanceProfiler` класс
2. ✅ Интегрировать в `main.py` и `visualizer.py`
3. ✅ Запустить baseline tests для N=3,5,10,20,30
4. ✅ Записать результаты в JSON и эту таблицу
5. ➡️ Перейти к реализации оптимизаций

---

## Notes

- Warm-up первые 50 кадров перед началом измерения
- Измерение в стабильной части эпизода (не начало/конец)
- Повторить измерение 3 раза и взять median для стабильности
- Закрыть фоновые приложения для чистоты эксперимента
