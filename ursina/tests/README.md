# 🧪 Tests Directory

Папка с тестами для проекта CALF.

## 📋 Список тестов

### Phase 3: Policy Abstraction
- **test_policies.py** - Тестирование политик (PDPolicy, TD3Policy, PolicyAdapter)

### Phase 4: Vectorized Environments
- **test_vectorized_env.py** - Базовый тест векторизованной среды (10 агентов)
- **test_performance.py** - Тест производительности (100 агентов)
- **test_scaling_curve.py** - Кривая масштабирования (10-200 агентов)

### Phase 5: Simple Trails
- **test_trails.py** - Тест простых траекторий (10 агентов)
- **test_trails_50.py** - Тест с 50 агентами и сбросом эпизодов

### Phase 6: Dual Visualization
- **test_dual_visualization.py** - Сравнение TD3 vs PD (2x25 агентов с полной статистикой)

## 🚀 Как запускать тесты

```bash
cd ursina/tests
python test_policies.py
python test_vectorized_env.py
python test_performance.py
python test_scaling_curve.py
python test_trails.py
python test_trails_50.py
python test_dual_visualization.py
```

## 📝 Примечания

- Все тесты используют Ursina для визуализации
- Для запуска нужны зависимости из `requirements.txt`
- Тесты с визуализацией могут занимать некоторое время
- Используйте ASCII в print() для совместимости с Windows
