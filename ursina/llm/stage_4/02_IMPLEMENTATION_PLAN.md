# Implementation Plan: Stage 4 Optimizations

## Overview

Поэтапный план реализации трёх критичных оптимизаций с детальными задачами и acceptance criteria.

---

## Phase 1: Performance Profiler Setup ⏱️

### Task 1.1: Create PerformanceProfiler class
**File**: `ursina/utils/performance_profiler.py`

**Requirements**:
- Context manager для timing секций
- Accumulation статистик за N кадров
- Export в JSON формате
- CPU/Memory monitoring с `psutil`

**Interface**:
```python
profiler = PerformanceProfiler()

with profiler.section('physics'):
    # physics code

with profiler.section('visuals'):
    # visual updates

stats = profiler.get_stats()  # dict с метриками
profiler.save_json('baseline.json')
```

**Acceptance Criteria**:
- ✅ Класс создан и протестирован
- ✅ Измеряет все метрики из `01_BASELINE_METRICS.md`
- ✅ Export в JSON работает

---

### Task 1.2: Integrate profiler into application
**Files**: `ursina/main.py`, `ursina/training/visualizer.py`

**Changes**:
- Добавить `--profile` flag в argparse
- Wrap ключевые секции в `profiler.section()`
- Save results после N кадров

**Acceptance Criteria**:
- ✅ `python main.py --profile` запускается
- ✅ JSON результаты сохраняются
- ✅ Minimal overhead (< 1% slowdown)

---

### Task 1.3: Run baseline measurements
**Command**: `python main.py --profile --n_agents N`

**Steps**:
1. Run для N=3, 5, 10, 20, 30
2. Записать результаты в `01_BASELINE_METRICS.md`
3. Идентифицировать top-3 bottlenecks

**Acceptance Criteria**:
- ✅ Все baseline таблицы заполнены
- ✅ Bottlenecks идентифицированы
- ✅ Baseline JSON files сохранены

---

## Phase 2: Vectorized Height Computation 🚀

### Task 2.1: Implement vectorized bilinear interpolation
**File**: `ursina/visuals/critic_heatmap.py`

**New Method**:
```python
def _interpolate_q_from_grid_vectorized(self, states: np.ndarray) -> np.ndarray:
    """
    Vectorized bilinear interpolation for batch of states

    Parameters:
    -----------
    states : np.ndarray, shape (N, 2)
        Batch of states [x, v]

    Returns:
    --------
    np.ndarray, shape (N,)
        Interpolated Q-values
    """
    # Full NumPy vectorization - no loops!
```

**Implementation Details**:
- Extract x, v from states[:, 0], states[:, 1]
- Vectorized clamping с `np.clip`
- Vectorized normalization
- Vectorized grid index computation
- Fancy indexing для bilinear weights
- Vectorized interpolation formula

**Acceptance Criteria**:
- ✅ Метод реализован
- ✅ Unit test: сравнение с loop версией (tolerance 1e-6)
- ✅ Benchmark: 5-10x быстрее loop версии

---

### Task 2.2: Update get_q_value_for_states_batch
**File**: `ursina/visuals/critic_heatmap.py`

**Changes**:
```python
def get_q_value_for_states_batch(self, states, use_cached=True):
    if use_cached:
        # Vectorized interpolation (NO loop!)
        q_values = self._interpolate_q_from_grid_vectorized(states)
        # Vectorized height computation
        heights = self._compute_height_from_q_vectorized(q_values)
    else:
        # Batch GPU query (already vectorized)
        ...
    return heights
```

**Acceptance Criteria**:
- ✅ Метод обновлён
- ✅ Все тесты пройдены
- ✅ Visual validation: агенты на правильной высоте

---

### Task 2.3: Measure performance improvement
**Command**: `python main.py --profile --n_agents 20`

**Metrics to compare**:
- Frame time for visual updates
- CPU usage
- FPS improvement

**Acceptance Criteria**:
- ✅ FPS improvement >= +20%
- ✅ CPU load снижение >= 30%
- ✅ Результаты записаны в `performance_report.md`

---

## Phase 3: Lazy Stats Caching 💾

### Task 3.1: Create StatsCache class
**File**: `ursina/utils/stats_cache.py`

**Interface**:
```python
class StatsCache:
    def __init__(self, update_interval: int = 10):
        self._cache = {}
        self._dirty_flags = {}
        self._frame_counter = 0

    def mark_dirty(self, key: str):
        self._dirty_flags[key] = True

    def get(self, key: str, compute_fn: Callable):
        if self._needs_update(key):
            self._cache[key] = compute_fn()
            self._dirty_flags[key] = False
        return self._cache[key]
```

**Acceptance Criteria**:
- ✅ Класс реализован
- ✅ Unit tests для dirty flags
- ✅ Configurable update intervals

---

### Task 3.2: Integrate into TrainingVisualizer
**File**: `ursina/training/visualizer.py`

**Changes in `update_stats_display()`**:
```python
def update_stats_display(self, step):
    # Mark dirty on episode end
    if self.trainer.episode_done:
        self.stats_cache.mark_dirty('episode_stats')

    # Get cached Q-value (update every 10 frames)
    q_value = self.stats_cache.get(
        'q_value',
        lambda: self._compute_current_q_value(),
        update_interval=10
    )

    # Update text только если changed
    if q_value != self._last_q_value:
        self.q_text.text = f'Q: {q_value:.3f}'
```

**Acceptance Criteria**:
- ✅ StatsCache интегрирован
- ✅ GPU queries снижены на 50-70%
- ✅ UI всё ещё обновляется корректно

---

### Task 3.3: Add config for update frequencies
**File**: `ursina/config/visualization_config.py`

**New fields**:
```python
@dataclass
class VisualizationConfig:
    # ... existing fields ...

    # Stats update frequencies (in frames)
    stats_q_value_update_freq: int = 10  # Q-value каждые 10 кадров
    stats_text_update_freq: int = 5      # Text каждые 5 кадров
    stats_profiler_update_freq: int = 30 # Profiler каждые 30 кадров
```

**Acceptance Criteria**:
- ✅ Config поля добавлены
- ✅ Presets обновлены
- ✅ Используются в `TrainingVisualizer`

---

### Task 3.4: Measure performance improvement
**Metrics**:
- GPU queries per frame
- UI update time (ms)
- Overall FPS improvement

**Acceptance Criteria**:
- ✅ GPU queries снижение >= 50%
- ✅ UI update time снижение >= 60%
- ✅ Результаты в `performance_report.md`

---

## Phase 4: Batched Trail Rendering 🎨

### Task 4.1: Design BatchedTrailRenderer
**File**: `ursina/visuals/batched_trail_renderer.py`

**Architecture**:
```python
class BatchedTrailRenderer:
    """
    Рендерит все trails агентов одним mesh/draw call
    """
    def __init__(self, max_agents: int, trail_length: int):
        self.max_agents = max_agents
        self.trail_length = trail_length

        # Один большой vertex buffer для всех trails
        self.all_vertices = np.zeros((max_agents * trail_length, 3))
        self.all_colors = np.zeros((max_agents * trail_length, 4))

        # Один Ursina Entity с большим mesh
        self.mesh_entity = None

    def update_trail(self, agent_id: int, positions: np.ndarray,
                    colors: np.ndarray):
        """Update trail для одного агента (no rebuild yet)"""
        start_idx = agent_id * self.trail_length
        end_idx = start_idx + len(positions)
        self.all_vertices[start_idx:end_idx] = positions
        self.all_colors[start_idx:end_idx] = colors

    def rebuild_mesh(self):
        """Один rebuild для всех trails (called once per frame)"""
        # Update mesh.vertices, mesh.colors, mesh.generate()
```

**Acceptance Criteria**:
- ✅ Класс спроектирован
- ✅ Interface определён
- ✅ Документация написана

---

### Task 4.2: Implement BatchedTrailRenderer
**File**: `ursina/visuals/batched_trail_renderer.py`

**Implementation Details**:
- Use `LineStrip` mode for trails (efficient)
- Partial vertex buffer updates (только изменённые агенты)
- Triangle topology generation для N trails
- Color per-vertex для mode switching

**Acceptance Criteria**:
- ✅ Класс реализован
- ✅ Визуально корректно (trails рисуются)
- ✅ No memory leaks

---

### Task 4.3: Refactor VisualAgent integration
**Files**:
- `ursina/training/visualizer.py`
- `ursina/visuals/oriented_agent.py` (если нужно)

**Changes**:
```python
class TrainingVisualizer:
    def __init__(self, ...):
        # Replace individual LineTrails with batched renderer
        self.batched_renderer = BatchedTrailRenderer(
            max_agents=50,
            trail_length=config.trail_max_length
        )

    def update_visual_agents(self, step):
        # Update all trails
        for i, agent in enumerate(self.visual_agents):
            agent.update_trajectory(...)  # Store in ring buffer

        # Rebuild batch mesh once
        self.batched_renderer.rebuild_mesh()
```

**Acceptance Criteria**:
- ✅ VisualAgent не создаёт свой LineTrail
- ✅ Все trails рендерятся через BatchedTrailRenderer
- ✅ Visual validation: trails выглядят идентично

---

### Task 4.4: Optimize partial updates
**File**: `ursina/visuals/batched_trail_renderer.py`

**Optimization**:
- Track dirty agents (которые двигались)
- Partial mesh update только для dirty regions
- Avoid full vertex buffer copy

```python
def rebuild_mesh(self, dirty_agent_ids: List[int]):
    """Update только для dirty агентов"""
    if not dirty_agent_ids:
        return

    # Update только затронутые регионы vertex buffer
    for agent_id in dirty_agent_ids:
        start = agent_id * self.trail_length
        end = start + self.trail_length
        # ... partial update
```

**Acceptance Criteria**:
- ✅ Partial updates работают
- ✅ Дополнительное ускорение ~20-30%

---

### Task 4.5: Measure performance improvement
**Metrics**:
- Draw calls per frame (до/после)
- Frame time for visual updates
- GPU overhead reduction

**Acceptance Criteria**:
- ✅ Draw calls: N → 1-3 (почти постоянно)
- ✅ Visual update time снижение >= 30%
- ✅ Масштабирование до 30+ агентов

---

## Phase 5: Final Validation & Reporting 📊

### Task 5.1: Run complete performance suite
**Commands**:
```bash
# Baseline (already done)
python main.py --profile --n_agents 20 --baseline

# Optimized (with all 3 optimizations)
python main.py --profile --n_agents 20 --optimized
```

**Compare**:
- All metrics from `01_BASELINE_METRICS.md`
- Side-by-side tables

**Acceptance Criteria**:
- ✅ All configurations tested
- ✅ Metrics collected
- ✅ Comparison tables generated

---

### Task 5.2: Visual regression testing
**Tests**:
1. Screenshot comparison (baseline vs optimized)
2. Agent positions on heatmap (numerically identical)
3. Trail colors (mode switching визуально correct)

**Tools**:
- `pytest-mpl` для screenshot comparison
- Manual visual inspection

**Acceptance Criteria**:
- ✅ No visual artifacts
- ✅ Numerically identical results (tolerance 1e-4)
- ✅ Screenshots match

---

### Task 5.3: Create performance report
**File**: `ursina/llm/stage_4/03_PERFORMANCE_REPORT.md`

**Contents**:
- Executive summary
- Baseline vs Optimized comparison tables
- Per-optimization breakdown
- Scalability charts (FPS vs N_agents)
- Bottleneck analysis (before/after)
- Recommendations for future work

**Acceptance Criteria**:
- ✅ Report complete
- ✅ All tables filled
- ✅ Graphs generated (if applicable)

---

### Task 5.4: Update documentation
**Files**:
- `README.md` - добавить Performance section
- `ursina/config/README.md` - документировать новые параметры
- `ursina/utils/README.md` - документировать profiler usage

**Acceptance Criteria**:
- ✅ Documentation updated
- ✅ Code examples added
- ✅ Performance tips included

---

## Timeline Estimate

| Phase | Tasks | Estimated Time | Complexity |
|-------|-------|----------------|------------|
| Phase 1 | 3 | 2-3 hours | Low |
| Phase 2 | 3 | 3-4 hours | Medium |
| Phase 3 | 4 | 3-4 hours | Medium |
| Phase 4 | 5 | 5-6 hours | High |
| Phase 5 | 4 | 2-3 hours | Low |
| **Total** | **19** | **15-20 hours** | - |

---

## Dependencies

```
Phase 1 (Profiler) → Phase 2, 3, 4 (need profiler for metrics)
Phase 2, 3, 4 (Optimizations) → Phase 5 (need results for report)
```

Phases 2, 3, 4 можно делать параллельно (независимы).

---

## Rollback Plan

Если оптимизация вызывает проблемы:

1. **Feature flag**: добавить config опцию для включения/выключения
   ```python
   use_vectorized_heights: bool = True
   use_stats_cache: bool = True
   use_batched_trails: bool = True
   ```

2. **Git branches**: каждая оптимизация в отдельной ветке
3. **Unit tests**: обязательны для rollback safety

---

## Success Criteria Summary

| Criteria | Target |
|----------|--------|
| FPS improvement (N=20) | >= +30% |
| CPU load reduction | >= -40% |
| GPU queries reduction | >= -50% |
| Draw calls reduction | N → 1-3 |
| No visual regressions | ✅ |
| All tests passing | ✅ |
| Documentation complete | ✅ |

---

## Next Steps

1. ➡️ Start with **Phase 1: Profiler Setup**
2. Create `ursina/utils/performance_profiler.py`
3. Integrate into `main.py`
4. Run baseline measurements
