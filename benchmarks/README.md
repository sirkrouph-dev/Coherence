# Performance & Scalability Benchmarks

## Overview

This benchmarking suite provides comprehensive performance and scalability testing for spiking neural networks across different hardware platforms (CPU and GPU).

## Features

### 1. **Multi-Scale Testing**
- Network sizes: 1k, 10k, 50k neurons
- Automatic parameter adjustment based on network size
- Hierarchical network architectures with multiple layers

### 2. **Metrics Collected**
- **Wall-clock simulation time** - Total time to run simulation
- **Memory usage** - Peak and average memory consumption
- **Spike throughput** - Spikes processed per second
- **Spike rate** - Average firing rate per neuron
- **Step timing** - Mean and standard deviation of simulation steps

### 3. **Hardware Support**
- **CPU** - Pure NumPy implementation
- **GPU (PyTorch)** - CUDA acceleration via PyTorch (when available)
- **GPU (CuPy)** - CUDA acceleration via CuPy (when available)
- **Graceful fallback** - Automatically falls back to CPU if GPU not available

### 4. **Automation Features**
- Uses `timeit` for micro-benchmarks
- Uses `memory_profiler` for memory tracking
- Automatic CSV and JSON export for analysis
- Visualization tools for generating plots

## Files

### Core Scripts

1. **`pytest_benchmarks.py`**
   - Comprehensive benchmark suite using pytest-benchmark
   - Measures throughput, memory, and convergence
   - Statistical analysis with warm-up and calibration
   - Run with: `pytest pytest_benchmarks.py --benchmark-only`

2. **`generate_report.py`**
   - Generates markdown report with shields.io badges
   - Creates `docs/benchmarks.md` with performance metrics
   - Includes system information and recommendations

3. **`performance_benchmarks.py`**
   - Main benchmarking suite
   - Runs full-scale network simulations
   - Supports CPU and GPU platforms
   - Exports detailed metrics to CSV/JSON

2. **`quick_benchmark.py`**
   - Simplified benchmarks for rapid testing
   - Smaller network sizes (100-5000 neurons)
   - Faster execution for development/testing

3. **`visualize_benchmarks.py`**
   - Generates performance plots
   - Creates comparison charts
   - Produces summary tables
   - Exports to PNG and PDF formats

### Output Files

- **`benchmark_results/`** - CSV and JSON results
  - `benchmark_results_*.csv` - Main benchmark metrics
  - `benchmark_details_*.json` - Detailed configuration and results
  - `microbenchmark_timeit.csv` - Micro-benchmark timings
  - `quick_benchmark_*.csv` - Quick test results

- **`benchmark_plots/`** - Visualization outputs
  - `benchmark_analysis.png/pdf` - Performance charts
  - `benchmark_summary_table.png` - Results summary table

## Usage

### Generate Benchmark Report with Badges

```bash
python generate_report.py
```

This generates `docs/benchmarks.md` with performance badges and detailed metrics.

### Running Full Benchmarks

```bash
python performance_benchmarks.py
```

This will:
1. Test networks of 1k, 10k, and 50k neurons
2. Run on all available platforms (CPU, GPU if available)
3. Export results to CSV and JSON
4. Display summary statistics

### Running Quick Tests

```bash
python quick_benchmark.py
```

This will:
1. Test smaller networks (100-5000 neurons)
2. Run faster simulations
3. Provide immediate feedback
4. Good for development/debugging

### Generating Visualizations

```bash
python visualize_benchmarks.py
```

This will:
1. Load all benchmark results from CSV files
2. Generate performance plots
3. Create summary tables
4. Export to PNG and PDF formats

## Benchmark Results

### Sample Performance Metrics (CPU)

| Network Size | Simulation Time | Memory (Peak) | Spike Throughput |
|-------------|-----------------|---------------|------------------|
| 100 neurons | 0.03s | 1.2 MB | 910 spikes/s |
| 500 neurons | 0.36s | 4.9 MB | 92 spikes/s |
| 1,000 neurons | 0.61s | 3.2 MB | 20 spikes/s |
| 5,000 neurons | 7.00s | 19.7 MB | 1 spikes/s |

### Micro-benchmarks

| Operation | Time |
|-----------|------|
| Single neuron step | 10.60 μs |
| Population step (100 neurons) | 0.47 ms |
| Network construction (210 neurons) | 11.12 ms |

## Configuration

### Network Parameters

Networks are configured with:
- **Neuron types**: AdEx (adaptive exponential) or LIF (leaky integrate-and-fire)
- **Synapse types**: STDP (spike-timing dependent plasticity)
- **Connection probability**: 2-10% (adjusted by network size)
- **Layer structure**: 3-5 layers depending on network size

### Simulation Parameters

- **Time step (dt)**: 1.0 ms
- **Simulation duration**: 50-1000 ms (scaled by network size)
- **Input injection**: Every 10-20 ms to first layer

## GPU Acceleration

### PyTorch Support

When PyTorch with CUDA is available:
- Automatically detects GPU
- Reports GPU memory and compute capability
- Falls back to CPU if CUDA unavailable

### CuPy Support

When CuPy is available:
- Uses CuPy arrays for acceleration
- Compatible with CUDA toolkit
- Falls back to CPU if not available

## Dependencies

### Required
- `numpy` - Numerical computations
- `memory-profiler` - Memory usage tracking
- `psutil` - System resource monitoring
- `matplotlib` - Visualization and plotting

### Optional (for GPU)
- `torch` - PyTorch for CUDA acceleration
- `cupy` - CuPy for CUDA acceleration

## Installation

```bash
pip install memory-profiler psutil matplotlib

# Optional GPU support
pip install torch  # For PyTorch/CUDA
pip install cupy   # For CuPy/CUDA
```

## Extending the Benchmarks

### Adding New Network Sizes

Edit `performance_benchmarks.py`:
```python
network_sizes = [1000, 10000, 50000, 100000]  # Add 100k
```

### Adding New Metrics

Extend `BenchmarkResult` dataclass:
```python
@dataclass
class BenchmarkResult:
    # ... existing fields ...
    energy_efficiency: float  # New metric
```

### Custom Network Architectures

Override `build_network()` method:
```python
def build_network(self):
    network = NeuromorphicNetwork()
    # Custom architecture
    return network
```

## Performance Optimization Tips

1. **Reduce connection probability** for larger networks
2. **Use simpler neuron models** (LIF vs AdEx) for speed
3. **Batch operations** when possible
4. **Profile with cProfile** to find bottlenecks
5. **Consider event-driven simulation** for sparse activity

## Troubleshooting

### Out of Memory
- Reduce network size
- Lower connection probability
- Use smaller simulation duration

### Slow Performance
- Check if running on GPU (if available)
- Reduce network complexity
- Use quick_benchmark.py for testing

### No Spikes Generated
- Increase input current strength
- Check neuron parameters
- Verify connections are created

## Future Enhancements

- [ ] Parallel CPU execution with multiprocessing
- [ ] Optimized sparse matrix operations
- [ ] Event-driven simulation mode
- [ ] Real-time performance monitoring
- [ ] Energy efficiency metrics
- [ ] Distributed computing support
- [ ] Hardware neuromorphic chip integration

## Contact

For questions or issues with the benchmarking suite, please refer to the main project documentation or create an issue in the repository.
