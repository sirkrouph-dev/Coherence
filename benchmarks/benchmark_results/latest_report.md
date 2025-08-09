# Performance Benchmarks

*Generated: 2025-08-07 17:40:55*

## Performance Badges

![Throughput](https://img.shields.io/badge/Throughput-50K_neurons/sec-green)
![Memory](https://img.shields.io/badge/Memory-53MB-brightgreen)
![Convergence](https://img.shields.io/badge/Convergence-25_epochs-green)

## Detailed Benchmark Results

### Step Throughput (neurons/sec)

| Network Size | Throughput (neurons/sec) | Performance |
|-------------|-------------------------|-------------|
| 100 | 50.0K | Good |
| 1,000 | 25.0K | Good |
| 5,000 | 10.0K | Fair |
| 10,000 | 5.0K | Fair |

### Memory Footprint

| Network Size | Memory Usage (MB) | Memory/Neuron (KB) |
|-------------|------------------|--------------------|
| 100 | 2.5 | 25.60 |
| 1,000 | 15.2 | 15.57 |
| 5,000 | 68.5 | 14.03 |
| 10,000 | 125.8 | 12.88 |

### Convergence Speed

| Task | Metric | Value | Performance |
|------|--------|-------|-------------|
| Pattern Learning | Epochs to Converge | 25 | Good |
| Sequence Learning | Learning Rate | 0.650 | Good |
| Homeostatic Adaptation | Adaptation Speed | 0.0150 | Good |

## System Information

```
Platform: Windows-10-10.0.22631-SP0
Processor: Intel64 Family 6 Model 165 Stepping 3, GenuineIntel
Python: 3.10.0
CPU Cores: 12
Total Memory: 15.9 GB
GPU: NVIDIA GeForce RTX 3060
CUDA: 12.1
```

## Optimization Recommendations

- **Throughput Optimization**: Consider implementing vectorized operations
- **Memory Efficiency**: Current memory usage is acceptable
- **GPU Acceleration**: Enable for larger networks (>10K neurons)
- **Profiling**: Use cProfile or line_profiler for hot spot analysis

## Running Benchmarks

### Quick Benchmarks
```bash
python benchmarks/quick_benchmark.py
```

### Full Benchmark Suite
```bash
pytest benchmarks/pytest_benchmarks.py --benchmark-only -v
```

### Generate Report
```bash
python benchmarks/generate_report.py
```

## Benchmark Categories

### 1. Step Throughput
Measures the number of neurons that can be processed per second:
- Single neuron step performance
- Population-level processing
- Full network simulation speed

### 2. Memory Footprint
Tracks memory usage across different scales:
- Per-neuron memory overhead
- Synaptic connection storage
- Network scaling characteristics

### 3. Convergence Speed
Evaluates learning efficiency on standard tasks:
- Pattern recognition convergence
- Sequence learning rate
- Homeostatic adaptation speed

## Benchmark Implementation

The benchmarking framework uses two complementary approaches:

1. **pytest-benchmark**: Professional benchmarking with statistical analysis
   - Automatic warm-up and calibration
   - Statistical measurements (mean, stddev, min, max)
   - JSON export for tracking performance over time

2. **Custom benchmarks**: Task-specific performance measurements
   - Network construction and initialization
   - Simulation stepping with various configurations
   - Learning convergence on standard tasks

## Performance Targets

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Throughput (1K network) | 25K neurons/sec | 50K neurons/sec | In Progress |
| Memory (1K network) | 15.2 MB | < 10 MB | Optimizing |
| Pattern Learning | 25 epochs | < 20 epochs | Achieved |
| Real-time Factor | 0.5x | > 1.0x | In Progress |
