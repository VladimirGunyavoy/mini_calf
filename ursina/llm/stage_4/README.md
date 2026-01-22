# Stage 4: Performance Optimization

## Overview

Stage 4 focuses on **performance optimization** of the CALF visualization system to support more agents while maintaining smooth framerates.

---

## Goals

1. **Vectorized Height Computation** - Eliminate Python loops in critical paths
2. **Lazy Stats Caching** - Reduce unnecessary GPU queries
3. **Batched Trail Rendering** - Minimize draw calls with unified mesh

**Target Metrics**:
- +30-50% FPS improvement at N=20 agents
- -40-60% CPU load reduction
- -50-70% GPU query reduction
- Scale to 30+ agents at 30+ FPS

---

## Documents

1. **[00_START_HERE.md](00_START_HERE.md)** - Overview and context
2. **[01_BASELINE_METRICS.md](01_BASELINE_METRICS.md)** - Measurement methodology and baseline results
3. **[02_IMPLEMENTATION_PLAN.md](02_IMPLEMENTATION_PLAN.md)** - Detailed implementation tasks
4. **[03_PERFORMANCE_REPORT.md](03_PERFORMANCE_REPORT.md)** - Final results (template, filled after completion)

---

## Quick Start

### 1. Baseline Measurement
```bash
# Run baseline profiling
python ursina/main.py --system point_mass --profile --n_agents 20

# Results saved to:
# ursina/llm/stage_4/baseline_results_YYYYMMDD_HHMMSS.json
```

### 2. Implementation
Follow [02_IMPLEMENTATION_PLAN.md](02_IMPLEMENTATION_PLAN.md):
- Phase 1: Setup profiler
- Phase 2: Vectorize height computation
- Phase 3: Add stats caching
- Phase 4: Batch trail rendering
- Phase 5: Generate report

### 3. Validation
```bash
# Run optimized profiling
python ursina/main.py --system point_mass --profile --n_agents 20

# Compare results
python ursina/utils/compare_profiles.py \
  baseline_results.json \
  optimized_results.json
```

---

## Key Files

### New Files Created
```
ursina/utils/performance_profiler.py      # Profiling utilities
ursina/utils/stats_cache.py               # Stats caching system
ursina/visuals/batched_trail_renderer.py  # Batched trail rendering
```

### Modified Files
```
ursina/visuals/critic_heatmap.py          # Vectorized interpolation
ursina/training/visualizer.py             # Stats cache integration
ursina/config/visualization_config.py     # New config parameters
ursina/main.py                            # Profiler integration
```

---

## Architecture Changes

### Before: Per-Agent Processing
```
For each agent:
  1. Interpolate Q-value (Python loop)
  2. Compute height
  3. Update trail mesh
  4. Issue draw call
  → N draw calls, N mesh updates, N interpolations
```

### After: Batched Processing
```
1. Vectorized interpolation for ALL agents (NumPy)
2. Batch height computation
3. Single trail mesh update (all agents)
4. Single draw call
→ 1 draw call, 1 mesh update, 1 vectorized op
```

---

## Performance Metrics

### Baseline (N=20, medium preset)
| Metric | Value |
|--------|-------|
| FPS | ~25-30 |
| Frame Time | ~35ms |
| GPU Queries | ~25-30 |
| Draw Calls | ~25-30 |
| CPU Usage | ~60-70% |

### Target (N=20, optimized)
| Metric | Value | Improvement |
|--------|-------|-------------|
| FPS | ~40-50 | +60% |
| Frame Time | ~20ms | -43% |
| GPU Queries | ~5-10 | -70% |
| Draw Calls | ~1-3 | -90% |
| CPU Usage | ~30-40% | -50% |

---

## Configuration

### New Config Options

```python
# ursina/config/visualization_config.py

@dataclass
class VisualizationConfig:
    # Performance toggles (for A/B testing)
    use_vectorized_heights: bool = True
    use_stats_cache: bool = True
    use_batched_trails: bool = True

    # Cache update frequencies (in frames)
    stats_q_value_update_freq: int = 10
    stats_text_update_freq: int = 5
    stats_profiler_update_freq: int = 30
```

### Presets Updated
- **low**: 3 agents, conservative caching
- **medium**: 5 agents, balanced caching (recommended)
- **high**: 10 agents, aggressive caching

---

## Usage Examples

### Profile Your Current Setup
```python
from ursina.utils.performance_profiler import PerformanceProfiler

profiler = PerformanceProfiler()

def update():
    with profiler.section('physics'):
        # physics code

    with profiler.section('visuals'):
        # visual updates

    if step % 300 == 0:
        stats = profiler.get_stats()
        print(f"FPS: {stats['fps_avg']:.1f}")
```

### Compare Baseline vs Optimized
```python
from ursina.utils.compare_profiles import compare_results

compare_results(
    'baseline_results.json',
    'optimized_results.json',
    output='comparison.md'
)
```

### Toggle Optimizations
```python
# Disable specific optimization for debugging
config = VisualizationConfig.from_preset('medium')
config.use_batched_trails = False  # Use old per-agent trails

app = CALFApplication(config=config)
```

---

## Testing

### Unit Tests
```bash
# Test vectorized interpolation correctness
pytest ursina/tests/test_critic_heatmap.py::test_vectorized_interpolation

# Test stats cache behavior
pytest ursina/tests/test_stats_cache.py

# Test batched renderer
pytest ursina/tests/test_batched_trail_renderer.py
```

### Visual Regression
```bash
# Generate baseline screenshots
python ursina/main.py --screenshot --output baseline/

# Generate optimized screenshots
python ursina/main.py --screenshot --output optimized/

# Compare
python ursina/utils/compare_screenshots.py baseline/ optimized/
```

---

## Troubleshooting

### Issue: FPS not improving
**Possible causes**:
1. Optimizations not enabled - check config
2. Different bottleneck (e.g., physics) - profile to identify
3. Hardware limitations - try lower N agents

### Issue: Visual artifacts
**Possible causes**:
1. Interpolation error - check unit tests
2. Trail mesh topology wrong - validate triangle indices
3. Color buffer misalignment - check buffer sizes

### Issue: Crashes with many agents
**Possible causes**:
1. Buffer overflow - increase `max_agents` in BatchedTrailRenderer
2. Memory leak - check entity destruction
3. GPU out of memory - reduce grid_size or trail_length

---

## Contributing

When adding new optimizations:

1. **Profile first** - measure before optimizing
2. **Unit test** - ensure correctness
3. **Visual validate** - no regressions
4. **Document** - add to performance report
5. **Config option** - allow toggle for safety

---

## References

### Related Documents
- [Stage 3 Documentation](../stage_3/) - System architecture
- [Training Config](../../config/training_config.py) - Training parameters
- [Visualization Config](../../config/visualization_config.py) - Visualization parameters

### External Resources
- [NumPy Vectorization Guide](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- [Ursina Performance Tips](https://www.ursinaengine.org/performance.html)
- [PyTorch Performance Tuning](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

---

## File Structure

```
stage_4/
├── 00_START_HERE.md           # Overview and goals (read first!)
├── 01_BASELINE_METRICS.md     # Measurement methodology and results
├── 02_IMPLEMENTATION_PLAN.md  # Detailed task breakdown (19 tasks)
├── 03_PERFORMANCE_REPORT.md   # Final results (template)
├── 04_LESSONS_LEARNED.md      # Post-implementation insights
├── ARCHITECTURE_DIAGRAM.md    # Visual architecture and data flow
├── QUICK_START.md             # TL;DR and quick commands
└── README.md                  # This file
```

## Reading Order

1. **Start here**: [QUICK_START.md](QUICK_START.md) - Quick overview
2. **Full context**: [00_START_HERE.md](00_START_HERE.md) - Goals and motivation
3. **Architecture**: [ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md) - How it works
4. **Implementation**: [02_IMPLEMENTATION_PLAN.md](02_IMPLEMENTATION_PLAN.md) - Step-by-step
5. **Baseline**: [01_BASELINE_METRICS.md](01_BASELINE_METRICS.md) - Measure first
6. **Results**: [03_PERFORMANCE_REPORT.md](03_PERFORMANCE_REPORT.md) - After completion
7. **Insights**: [04_LESSONS_LEARNED.md](04_LESSONS_LEARNED.md) - After completion

---

## Status

- ✅ Stage 4 documentation complete (8 files)
- ⬜ Baseline measurements collected
- ⬜ Optimization 1 implemented (Vectorized Heights)
- ⬜ Optimization 2 implemented (Stats Cache)
- ⬜ Optimization 3 implemented (Batched Trails)
- ⬜ Performance report generated
- ⬜ Stage 4 complete

**Progress**: 0/19 tasks completed

---

**Last Updated**: 2026-01-22
**Stage Status**: 📋 Planning Complete - Ready for Implementation
