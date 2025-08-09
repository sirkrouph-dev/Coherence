# Test Suite Documentation

## Overview

This directory contains comprehensive unit and integration tests for the neuromorphic programming system. The test suite enforces a minimum code coverage of 90% and includes tests for neurons, synapses, learning rules, and end-to-end learning tasks.

## Test Structure

```
tests/
├── test_neurons.py      # Unit tests for neuron models
├── test_synapses.py     # Unit tests for synapse models
├── test_learning.py     # Unit tests for learning rules
├── test_integration.py  # Integration tests for end-to-end scenarios
└── README.md           # This file
```

## Test Categories

### Unit Tests

#### test_neurons.py
- **NeuronModel**: Base neuron functionality
- **AdaptiveExponentialIntegrateAndFire**: AdEx neuron dynamics
- **HodgkinHuxleyNeuron**: HH neuron with ion channels
- **LeakyIntegrateAndFire**: LIF neuron model
- **NeuronPopulation**: Population-level operations
- **NumericalStability**: Numerical stability tests

#### test_synapses.py
- **SynapseModel**: Base synapse functionality
- **STDP_Synapse**: Spike-timing-dependent plasticity
- **ShortTermPlasticitySynapse**: STP with depression/facilitation
- **NeuromodulatorySynapse**: Neuromodulatory learning
- **RSTDP_Synapse**: Reward-modulated STDP
- **SynapseFactory**: Factory pattern tests
- **SynapsePopulation**: Population-level synaptic operations

#### test_learning.py
- **PlasticityConfig**: Configuration management
- **STDPRule**: STDP learning rule
- **HebbianRule**: Classical Hebbian learning
- **BCMRule**: BCM learning with sliding threshold
- **RewardModulatedSTDP**: Reward-based learning
- **TripletSTDP**: Triplet-based STDP
- **HomeostaticPlasticity**: Homeostatic regulation
- **CustomPlasticityRule**: User-defined rules
- **PlasticityManager**: Multi-rule management

### Integration Tests

#### test_integration.py
- **PatternLearning**: Pattern recognition and selectivity
- **SequenceLearning**: Temporal sequence learning
- **RewardModulatedLearning**: Reinforcement learning
- **MultiLayerLearning**: Hierarchical feature learning
- **HomeostaticRegulation**: Firing rate homeostasis
- **SynapticScaling**: Weight distribution stability
- **LearningPerformance**: Performance benchmarks
- **LearningRobustness**: Noise robustness and catastrophic forgetting

## Running Tests

### Quick Start

```bash
# Run all tests with coverage
make test

# Run only unit tests
make test-unit

# Run only integration tests
make test-integration

# Run tests without linting
make test-fast

# Run tests in verbose mode
make test-verbose
```

### Using pytest directly

```bash
# Run all tests with coverage enforcement
pytest --cov=core --cov=api --cov-fail-under=90

# Run specific test file
pytest tests/test_neurons.py -v

# Run specific test class
pytest tests/test_learning.py::TestSTDPRule -v

# Run specific test method
pytest tests/test_integration.py::TestPatternLearning::test_pattern_recognition -v

# Run tests with specific markers
pytest -m "not slow"     # Skip slow tests
pytest -m "not gpu"       # Skip GPU tests
pytest -m integration     # Run only integration tests
```

### Using the test runner script

```bash
# Run all tests with default settings
python scripts/run_tests.py

# Run only unit tests
python scripts/run_tests.py --type unit

# Run with custom coverage threshold
python scripts/run_tests.py --coverage 95

# Skip linting checks
python scripts/run_tests.py --no-lint

# Generate coverage report only
python scripts/run_tests.py --report-only
```

## Coverage Requirements

The test suite enforces a **minimum coverage of 90%** for the `core` and `api` modules. This is configured in:

1. **pyproject.toml**: `--cov-fail-under=90` in pytest options
2. **GitHub Actions CI**: Coverage check in `.github/workflows/test.yml`
3. **Pre-commit hooks**: Coverage enforcement in `.pre-commit-config.yaml`

### Viewing Coverage Reports

After running tests, coverage reports are available in multiple formats:

- **Terminal**: Displayed automatically after test run
- **HTML**: Open `htmlcov/index.html` in a browser
- **XML**: `coverage.xml` for CI integration

```bash
# Generate HTML coverage report
coverage html

# Open HTML report (Windows)
start htmlcov/index.html

# Open HTML report (Mac/Linux)
open htmlcov/index.html
```

## CI/CD Integration

### GitHub Actions

The CI pipeline runs on:
- **Push**: to `main` and `develop` branches
- **Pull Requests**: targeting `main` and `develop`

Test matrix:
- **OS**: Ubuntu, Windows, macOS
- **Python**: 3.8, 3.9, 3.10, 3.11

### Pre-commit Hooks

Install pre-commit hooks to run tests automatically:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Test Markers

Tests can be marked with specific attributes:

```python
@pytest.mark.slow
def test_long_running_simulation():
    # Test that takes > 5 seconds
    pass

@pytest.mark.gpu
def test_gpu_acceleration():
    # Test requiring GPU
    pass

@pytest.mark.integration
def test_end_to_end():
    # Integration test
    pass
```

## Writing New Tests

### Guidelines

1. **Naming**: Test files must start with `test_`
2. **Classes**: Test classes must start with `Test`
3. **Methods**: Test methods must start with `test_`
4. **Assertions**: Use clear, descriptive assertions
5. **Fixtures**: Use pytest fixtures for setup/teardown
6. **Mocking**: Use `unittest.mock` for external dependencies
7. **Coverage**: Ensure new code has ≥90% coverage

### Example Test Structure

```python
import unittest
import numpy as np
from core.neurons import NeuronModel

class TestNewFeature(unittest.TestCase):
    """Test suite for new feature."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = NeuronModel(neuron_id=1)
    
    def tearDown(self):
        """Clean up after tests."""
        self.model.reset()
    
    def test_feature_behavior(self):
        """Test expected behavior."""
        result = self.model.some_method()
        self.assertEqual(result, expected_value)
    
    def test_edge_case(self):
        """Test edge cases."""
        with self.assertRaises(ValueError):
            self.model.invalid_operation()
```

## Performance Testing

For performance-critical code, use benchmarking:

```python
import time

def test_performance():
    start = time.time()
    # Run operation
    elapsed = time.time() - start
    assert elapsed < 1.0  # Should complete in < 1 second
```

Run benchmarks:
```bash
make benchmark
```

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure project is installed: `pip install -e .`
2. **Coverage too low**: Check uncovered lines: `coverage report -m`
3. **Slow tests**: Use markers to skip: `pytest -m "not slow"`
4. **Flaky tests**: Check for randomness, add seeds: `np.random.seed(42)`

### Debug Mode

Run tests with debugging:
```bash
# Drop into debugger on failure
pytest --pdb

# Show local variables on failure
pytest -l

# Verbose output with full tracebacks
pytest -vvv --tb=long
```

## Contributing

When adding new features:

1. Write tests FIRST (TDD approach)
2. Ensure all tests pass
3. Check coverage: `make coverage`
4. Run linting: `make lint`
5. Update this README if needed

## License

Tests are part of the neuromorphic system project and follow the same MIT license.
