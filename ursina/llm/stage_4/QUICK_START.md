# Stage 4 Quick Start Guide 🚀

## TL;DR

Stage 4 optimizes performance through:
1. **Vectorized height computation** → 10x faster
2. **Lazy stats caching** → 70% fewer GPU queries
3. **Batched trail rendering** → N draw calls → 1

**Target**: +30-50% FPS, support 30+ agents smoothly

---

## Quick Commands

### Run baseline profiling
```bash
cd c:\GitHub\Learn\CALF
python ursina/main.py --system point_mass --profile --n_agents 20
```

### Run with all optimizations (after implementation)
```bash
python ursina/main.py --system point_mass --optimized --n_agents 20
```

### Compare results
```bash
python ursina/utils/compare_profiles.py baseline.json optimized.json
```

---

## Implementation Checklist

### Phase 1: Setup (2-3 hours)
- [ ] Create `ursina/utils/performance_profiler.py`
- [ ] Integrate profiler into `main.py`
- [ ] Run baseline for N=5, 10, 20, 30
- [ ] Fill baseline tables in `01_BASELINE_METRICS.md`

### Phase 2: Vectorization (3-4 hours)
- [ ] Implement `_interpolate_q_from_grid_vectorized()` in `critic_heatmap.py`
- [ ] Update `get_q_value_for_states_batch()` to use vectorized version
- [ ] Add unit test: compare with loop version (tolerance 1e-6)
- [ ] Benchmark: verify 5-10x speedup
- [ ] Visual validation: agents at correct height

### Phase 3: Caching (3-4 hours)
- [ ] Create `ursina/utils/stats_cache.py`
- [ ] Integrate into `TrainingVisualizer.update_stats_display()`
- [ ] Add dirty flags for episode end
- [ ] Add config: `stats_q_value_update_freq = 10`
- [ ] Verify: GPU queries drop 50-70%

### Phase 4: Batched Trails (5-6 hours)
- [ ] Create `ursina/visuals/batched_trail_renderer.py`
- [ ] Design unified vertex buffer layout
- [ ] Implement `update_trail()` and `rebuild_mesh()`
- [ ] Refactor `VisualAgent` to use batched renderer
- [ ] Optimize: partial updates for dirty agents only
- [ ] Verify: Draw calls N → 1-3

### Phase 5: Validation (2-3 hours)
- [ ] Run complete performance suite (all N values)
- [ ] Screenshot comparison (baseline vs optimized)
- [ ] Fill `03_PERFORMANCE_REPORT.md` tables
- [ ] Update documentation

---

## Expected Results

| Metric | Baseline (N=20) | Target (N=20) | Status |
|--------|-----------------|---------------|--------|
| FPS | 25-30 | 40-50 | ⬜ |
| Frame Time | ~35ms | ~20ms | ⬜ |
| GPU Queries | 25-30 | 5-10 | ⬜ |
| Draw Calls | 25-30 | 1-3 | ⬜ |
| CPU Usage | 60-70% | 30-40% | ⬜ |

---

## Key Files

### To Create
```
ursina/utils/performance_profiler.py       # Profiling system
ursina/utils/stats_cache.py                # Caching system
ursina/visuals/batched_trail_renderer.py   # Batched rendering
```

### To Modify
```
ursina/visuals/critic_heatmap.py           # Add vectorized interpolation
ursina/training/visualizer.py              # Integrate cache
ursina/config/visualization_config.py      # Add cache config
ursina/main.py                             # Add --profile flag
```

---

## Testing Commands

### Unit tests
```bash
pytest ursina/tests/test_vectorized_interpolation.py -v
pytest ursina/tests/test_stats_cache.py -v
pytest ursina/tests/test_batched_trails.py -v
```

### Benchmark
```bash
python ursina/utils/benchmark_interpolation.py
# Expected: vectorized 5-10x faster than loop
```

### Visual regression
```bash
python ursina/main.py --screenshot baseline --n_agents 20
python ursina/main.py --screenshot optimized --n_agents 20 --optimized
diff baseline/ optimized/  # Should be identical
```

---

## Troubleshooting

### "FPS not improving after optimization"
1. Check config: `use_vectorized_heights = True`
2. Profile to confirm optimization is running
3. Verify baseline was measured correctly

### "Visual artifacts in trails"
1. Check vertex buffer alignment
2. Validate triangle indices
3. Compare screenshots with baseline

### "GPU memory error with many agents"
1. Reduce `max_agents` in BatchedTrailRenderer
2. Reduce `trail_max_length` in config
3. Reduce `grid_size` in heatmap

---

## Quick Architecture Overview

### Before
```
For each agent (loop):
  ├─ Interpolate Q-value (Python)  0.5ms
  ├─ Compute height                0.1ms
  ├─ Update trail mesh              1.0ms
  └─ Issue draw call                0.2ms
Total: 20 agents * 1.8ms = 36ms
```

### After
```
Batch all agents:
  ├─ Vectorized interpolation (NumPy)  1.0ms
  ├─ Vectorized height                 0.2ms
  ├─ Single mesh update                2.0ms
  └─ Single draw call                  0.3ms
Total: 3.5ms (10x faster!)
```

---

## Performance Targets by Hardware

### Low-end PC (4GB RAM, integrated GPU)
- Preset: `low`
- N agents: 3-5
- Target FPS: 30+
- Optimizations: Critical

### Mid-range PC (8GB RAM, dedicated GPU)
- Preset: `medium`
- N agents: 5-10
- Target FPS: 45+
- Optimizations: Recommended

### High-end PC (16GB+ RAM, strong GPU)
- Preset: `high`
- N agents: 20-30
- Target FPS: 60
- Optimizations: Enables more agents

---

## Code Snippets

### Vectorized Interpolation Template
```python
def _interpolate_q_from_grid_vectorized(self, states: np.ndarray) -> np.ndarray:
    """Vectorized bilinear interpolation (no loops!)"""
    x, v = states[:, 0], states[:, 1]

    # Normalize to grid space (vectorized)
    x_norm = (x - self.x_range[0]) / (self.x_range[1] - self.x_range[0])
    v_norm = (v - self.v_range[0]) / (self.v_range[1] - self.v_range[0])

    # Grid indices (vectorized)
    col = x_norm * (self.grid_size - 1)
    row = v_norm * (self.grid_size - 1)
    col0, row0 = np.floor(col).astype(int), np.floor(row).astype(int)
    col1 = np.minimum(col0 + 1, self.grid_size - 1)
    row1 = np.minimum(row0 + 1, self.grid_size - 1)

    # Weights (vectorized)
    wx, wv = col - col0, row - row0

    # Fetch corners (fancy indexing)
    q00 = self.q_grid[row0, col0]
    q01 = self.q_grid[row0, col1]
    q10 = self.q_grid[row1, col0]
    q11 = self.q_grid[row1, col1]

    # Bilinear interpolation (vectorized)
    q = (1-wx)*(1-wv)*q00 + wx*(1-wv)*q01 + (1-wx)*wv*q10 + wx*wv*q11

    return q
```

### Stats Cache Usage
```python
# In TrainingVisualizer
self.stats_cache = StatsCache()

def update_stats_display(self, step):
    # Cached Q-value (update every 10 frames)
    q_value = self.stats_cache.get(
        'q_value',
        lambda: self._compute_q_value(),
        update_interval=10
    )

    # Dirty on episode end
    if self.trainer.episode_done:
        self.stats_cache.mark_dirty('episode_reward')

    # Update text only if changed
    if q_value != self._last_q_value:
        self.q_text.text = f'Q: {q_value:.3f}'
```

### Batched Trail Renderer Usage
```python
# In TrainingVisualizer.__init__
self.batched_renderer = BatchedTrailRenderer(
    max_agents=50,
    trail_length=config.trail_max_length
)

# In update_visual_agents
for i, agent in enumerate(self.visual_agents):
    agent.step(action)
    self.batched_renderer.update_trail(
        agent_id=i,
        positions=agent.trail_positions,
        colors=agent.trail_colors
    )

# Single mesh rebuild (not per-agent!)
self.batched_renderer.rebuild_mesh()
```

---

## Metrics Collection

### Add to your update loop
```python
from ursina.utils.performance_profiler import PerformanceProfiler

profiler = PerformanceProfiler()

def update():
    profiler.start_frame()

    with profiler.section('physics'):
        simulation_engine.step()

    with profiler.section('visuals'):
        visualizer.update_visual_agents()

    with profiler.section('ui'):
        visualizer.update_stats_display()

    profiler.end_frame()

    # Save every 5 seconds
    if step % 300 == 0:
        stats = profiler.get_stats()
        print(f"FPS: {stats['fps_avg']:.1f}, Frame: {stats['frame_time_avg_ms']:.1f}ms")
        profiler.save_json(f'profile_step_{step}.json')
```

---

## Success Criteria

### Minimum (Must Have)
- ✅ FPS improvement >= +30% at N=20
- ✅ No visual regressions
- ✅ All unit tests passing

### Target (Should Have)
- ✅ FPS improvement >= +50% at N=20
- ✅ CPU usage reduction >= -40%
- ✅ GPU queries reduction >= -50%

### Stretch (Nice to Have)
- ✅ Support 30+ agents at 30+ FPS
- ✅ Memory usage < +20% overhead
- ✅ Code complexity increase < 30%

---

## Next Steps

1. **Read** `00_START_HERE.md` for full context
2. **Study** `ARCHITECTURE_DIAGRAM.md` for implementation details
3. **Follow** `02_IMPLEMENTATION_PLAN.md` for step-by-step tasks
4. **Measure** baseline with `01_BASELINE_METRICS.md`
5. **Implement** optimizations phase by phase
6. **Document** results in `03_PERFORMANCE_REPORT.md`
7. **Learn** lessons in `04_LESSONS_LEARNED.md`

---

## Support

### Issues?
- Check troubleshooting section above
- Review architecture diagram for understanding
- Profile to identify actual bottleneck
- Compare with baseline to isolate regression

### Questions?
- Read full documentation in stage_4/ folder
- Check Stage 3 docs for architecture background
- Review code comments in implementation files

---

**Good luck with the optimization!** 🚀

**Remember**: Profile first, optimize second, validate third.
