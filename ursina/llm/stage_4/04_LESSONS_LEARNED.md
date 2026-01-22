# Lessons Learned: Stage 4 Performance Optimization

> **Note**: Этот файл будет заполнен после реализации оптимизаций

---

## General Principles

### Performance Optimization Best Practices

1. **Measure First, Optimize Later**
   - Never optimize without profiling data
   - Identify actual bottlenecks, not assumed ones
   - Use `PerformanceProfiler` для объективных измерений

2. **Vectorize Hot Paths**
   - Python loops = slow
   - NumPy operations = fast (10-100x)
   - Особенно критично для операций над N агентами

3. **Cache Expensive Computations**
   - GPU queries дорогие (~1-5ms каждая)
   - Используйте dirty flags для invalidation
   - Configurable update frequency для гибкости

4. **Batch Rendering Operations**
   - N draw calls → 1 draw call = огромный выигрыш
   - Используйте unified mesh где возможно
   - Instance rendering для повторяющейся геометрии

---

## Technical Insights

### Vectorization Patterns

#### ❌ Bad: Python Loop
```python
heights = []
for state in states:
    q = self._interpolate_q_from_grid(state)  # Slow!
    h = self._compute_height(q)
    heights.append(h)
```

#### ✅ Good: Vectorized NumPy
```python
# Batch interpolation (no loops!)
q_values = self._interpolate_q_from_grid_vectorized(states)
heights = self._compute_height_vectorized(q_values)
```

**Key Insight**: Даже простая операция, повторённая N раз, становится узким местом.

---

### Caching Strategies

#### Pattern: Dirty Flag Cache
```python
class StatsCache:
    def __init__(self):
        self._cache = {}
        self._dirty = {}

    def mark_dirty(self, key):
        self._dirty[key] = True

    def get(self, key, compute_fn):
        if self._dirty.get(key, True):
            self._cache[key] = compute_fn()
            self._dirty[key] = False
        return self._cache[key]
```

**When to use**:
- Expensive computation (GPU queries)
- Infrequent changes (episode stats)
- Deterministic invalidation (episode end)

**When NOT to use**:
- Cheap computations (простые математические операции)
- Frequent changes (agent positions каждый кадр)
- Complex invalidation logic (слишком много зависимостей)

---

### Batched Rendering

#### Key Challenge: Buffer Management
```python
# One vertex buffer for N agents' trails
all_vertices = np.zeros((max_agents * trail_length, 3))

# Update only changed agents (partial update)
for agent_id in dirty_agents:
    start = agent_id * trail_length
    end = start + trail_length
    all_vertices[start:end] = agent.trail_positions
```

**Tradeoffs**:
- **Pro**: 1 draw call вместо N
- **Pro**: GPU memory contiguous
- **Con**: Сложнее управление (partial updates, indices)
- **Con**: Memory overhead для max_agents (даже если не все active)

---

## What Worked Well

### 1. Incremental Optimization
- Реализация по одной оптимизации за раз
- Измерение impact после каждой
- Возможность rollback при проблемах

### 2. Feature Flags для A/B Testing
```python
config.use_vectorized_heights = True  # Toggle on/off
```
- Лёгкое сравнение baseline vs optimized
- Debugging: изолировать проблемную оптимизацию

### 3. Visual Regression Testing
- Screenshot comparison catches артефакты
- Numerical validation (max error < 1e-4)
- Manual inspection для subtle issues

---

## Challenges Encountered

### Challenge 1: [To be filled]
**Description**: [What went wrong]

**Solution**: [How it was solved]

**Lesson**: [What we learned]

---

### Challenge 2: [To be filled]
**Description**: [What went wrong]

**Solution**: [How it was solved]

**Lesson**: [What we learned]

---

## Unexpected Findings

### Finding 1: [To be filled]
**Description**: [What was surprising]

**Impact**: [How it affected the optimization]

**Takeaway**: [What to remember for future]

---

## Performance Anti-Patterns Identified

### Anti-Pattern 1: Per-Frame GPU Queries
```python
# BAD: Query GPU every frame (60 FPS = 60 queries/sec)
def update():
    q_value = self.critic(state)  # Expensive!
    self.text.text = f'Q: {q_value}'
```

**Fix**: Cache с configurable update frequency
```python
# GOOD: Update every 10 frames
if frame % 10 == 0:
    self.cached_q = self.critic(state)
self.text.text = f'Q: {self.cached_q}'
```

---

### Anti-Pattern 2: Repeated Array Conversions
```python
# BAD: NumPy → Python list → NumPy
for i, state in enumerate(states):  # states is np.ndarray
    result = compute(state)  # state is np.ndarray[2]
    results.append(result)  # Python list (slow!)
results = np.array(results)  # Back to NumPy
```

**Fix**: Pre-allocate NumPy array
```python
# GOOD: Stay in NumPy
results = np.zeros(len(states))
results[:] = compute_vectorized(states)  # No loops!
```

---

### Anti-Pattern 3: Unnecessary Mesh Rebuilds
```python
# BAD: Rebuild mesh even if nothing changed
def update():
    self.trail.rebuild_mesh()  # Always rebuilds!
```

**Fix**: Dirty flag для conditional rebuild
```python
# GOOD: Rebuild only if changed
def update():
    if self.trail.dirty:
        self.trail.rebuild_mesh()
        self.trail.dirty = False
```

---

## Optimization Impact Matrix

| Optimization | Implementation Complexity | Performance Impact | Code Complexity Impact | Recommended? |
|--------------|--------------------------|-------------------|------------------------|--------------|
| Vectorized Heights | Medium | High (+40% FPS) | Low | ✅ Yes |
| Stats Caching | Low | Medium (+15% FPS) | Low | ✅ Yes |
| Batched Trails | High | High (+30% FPS) | Medium | ✅ Yes |
| GPU Compute Shaders | Very High | Very High (+100% heatmap) | High | ⚠️ Maybe |
| Async Physics | High | Medium (+20% FPS) | High | ⚠️ Maybe |

---

## Future Optimization Ideas

### High Priority
1. **Parallel Agent Updates** (multiprocessing)
   - Run visual agents in parallel processes
   - Estimated: +30-50% CPU utilization

2. **LOD (Level of Detail) for Trails**
   - Reduce trail resolution for distant agents
   - Estimated: +20% rendering performance

### Medium Priority
3. **Instanced Rendering for Agents**
   - Use GPU instancing for agent spheres/cones
   - Estimated: +15% draw call reduction

4. **Spatial Hashing for Culling**
   - Don't render off-screen agents
   - Estimated: +10-20% with many agents

### Low Priority (diminishing returns)
5. **SIMD Optimization** (numba/cython)
   - Hand-optimized critical loops
   - Estimated: +5-10% (marginal improvement)

---

## Recommendations for Stage 5+

### If focusing on scalability (100+ agents):
1. Implement parallel agent updates
2. Add LOD system for trails
3. GPU compute shaders for heatmap

### If focusing on visual quality:
1. Anti-aliasing for trails
2. Smooth camera transitions
3. Post-processing effects

### If focusing on flexibility:
1. Plugin system for custom systems
2. Config hot-reload
3. Real-time parameter tuning UI

---

## Key Metrics to Track

### Performance Health Indicators
- **FPS stability**: stddev < 5 FPS
- **Frame time P95**: < 50ms (20 FPS minimum)
- **GPU queries**: < 10 per frame
- **Draw calls**: < 5 per frame
- **Memory growth**: < 1 MB/minute (no leaks)

### When to Re-optimize
- FPS drops below 20 with recommended config
- New feature adds > 10ms frame time
- User reports sluggishness

---

## Testing Strategy

### Performance Tests
```python
@pytest.mark.performance
def test_vectorized_height_performance():
    """Vectorized version should be 5x faster"""
    states = generate_random_states(100)

    time_loop = benchmark(height_computation_loop, states)
    time_vec = benchmark(height_computation_vectorized, states)

    assert time_vec < time_loop / 5
```

### Regression Tests
```python
@pytest.mark.regression
def test_no_visual_regression():
    """Optimized version should produce identical results"""
    state = np.array([1.0, 0.5])

    height_baseline = compute_height_baseline(state)
    height_optimized = compute_height_optimized(state)

    assert abs(height_baseline - height_optimized) < 1e-6
```

---

## Documentation Updates Needed

### Code Comments
- Add performance notes in hot paths
- Document vectorization assumptions
- Explain cache invalidation logic

### User Documentation
- Performance tuning guide
- Hardware recommendations
- Config preset explanations

### Developer Documentation
- Profiling workflow
- Optimization checklist
- Benchmark suite usage

---

## Conclusion

### Key Takeaways

1. **Vectorization is king** for per-agent operations
2. **Cache expensive ops** with smart invalidation
3. **Batch rendering** for massive draw call reduction
4. **Always measure** - intuition can be wrong
5. **Feature flags** enable safe rollback

### Success Metrics Achieved
- ✅ [Metric 1]
- ✅ [Metric 2]
- ✅ [Metric 3]

### Future Work
- [ ] [Task 1]
- [ ] [Task 2]
- [ ] [Task 3]

---

**Document Status**: 🚧 Template - to be completed after Stage 4 implementation

**Last Updated**: 2026-01-22
