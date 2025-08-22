# Gemini Code Insight: Coherence Neuromorphic Framework

## Project Overview

This project, named "Coherence," is an experimental neuromorphic computing framework written in Python. Its primary goal is to demonstrate stable concept representation through balanced competitive learning, addressing the "binding problem" in neuromorphic computing. The framework is designed for simulating large-scale, brain-inspired neural networks with a focus on computational efficiency and biological plausibility.

**Key Features:**

*   **Large-Scale Simulation:** Supports networks of 10,000 to over 1 million neurons, with GPU acceleration (via PyTorch and CuPy) for handling the computational load.
*   **Advanced Plasticity:** Implements a sophisticated multi-plasticity system that includes STDP, homeostatic plasticity, metaplasticity, reward-modulated learning, and synaptic competition. This is the core of the "balanced competitive learning" approach.
*   **Biologically Inspired Architecture:** Features diverse neuron models (e.g., AdEx, Izhikevich), brain-inspired network topologies (e.g., distance-dependent connectivity, E/I balance), and neuromodulation systems (e.g., dopamine-based reward prediction).
*   **Event-Driven and Efficient:** Utilizes an event-driven simulation engine for efficiency, which is crucial for sparse spiking networks. The codebase is optimized for performance, with a dedicated benchmarking suite.
*   **Edge Computing Ready:** The architecture is designed with edge deployment in mind, with specific optimizations for NVIDIA Jetson platforms.

**Technology Stack:**

*   **Core:** Python 3.9+
*   **Scientific Computing:** NumPy, SciPy
*   **Data Handling & Analysis:** Pandas, scikit-learn
*   **GPU Acceleration (Optional):** PyTorch, CuPy
*   **Development & Tooling:** `pyproject.toml` (setuptools), `black`, `ruff`, `isort`, `mypy`, `pytest`

## Building and Running

### Installation

The project uses a `pyproject.toml` file, so it can be installed in editable mode using pip.

**Basic Installation:**
```bash
pip install -e .
```

**Optional Dependencies:**
For development, GPU support, or other extras, install the optional dependencies:
```bash
# For development tools (pytest, black, ruff, etc.)
pip install -e ".[dev]"

# For NVIDIA GPU acceleration with CUDA
pip install -e ".[gpu]"

# For NVIDIA Jetson platform optimizations
pip install -e ".[jetson]"

# To install all optional dependencies
pip install -e ".[all]"
```

### Running Demos

The project includes several demonstration scripts to showcase its capabilities.

**Main Demo:**
```bash
python tools/agi_testbed_complete.py
```

**Sensorimotor Demo:**
```bash
python demo/sensorimotor_demo.py
```

### Running Tests

The project has a comprehensive test suite using `pytest`.

```bash
python -m pytest tests/
```

### Running Benchmarks

A dedicated performance benchmarking suite is available to evaluate the framework's scalability and speed.

```bash
python benchmarks/performance_benchmarks.py
```
The results are saved in the `benchmark_results/` directory.

## Development Conventions

The codebase follows modern Python development practices, enforced by a suite of linters and formatters.

*   **Code Formatting:** `black` is used for consistent code formatting.
*   **Linting:** `ruff` and `flake8` are used for identifying potential errors and style issues.
*   **Import Sorting:** `isort` is used to organize imports.
*   **Type Checking:** `mypy` is used for static type analysis.
*   **Testing:** Unit and integration tests are written using the `pytest` framework and are located in the `tests/` directory.
*   **Configuration:** All tool configurations are centralized in the `pyproject.toml` file.
