"""
Sleep/Rest Phase Integration Tests
----------------------------------

Verifies that the optional sleep phase integrates with learning and can
produce non-identical post-sleep responses under deterministic conditions.
"""

import os
import sys
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.network import NeuromorphicNetwork


def _build_network(input_size: int = 10, output_size: int = 5) -> NeuromorphicNetwork:
    net = NeuromorphicNetwork()
    net.add_layer("input", input_size, "lif")
    net.add_layer("output", output_size, "lif")
    # Fully connect for determinism in small test
    net.connect_layers("input", "output", "stdp", connection_probability=1.0)
    return net


def _patterns(n: int):
    p0 = np.zeros(n); p0[0:3] = 50.0
    p1 = np.zeros(n); p1[3:6] = 50.0
    p2 = np.zeros(n); p2[6:9] = 50.0
    return [p0, p1, p2]


def _boost_input(net: NeuromorphicNetwork, pattern: np.ndarray, scale: float = 0.1):
    layer = net.layers["input"].neuron_population
    for i, neuron in enumerate(layer.neurons):
        neuron.membrane_potential += float(pattern[i]) * scale


def _train_epoch(net: NeuromorphicNetwork, patterns, dt=0.1, present_ms=20.0):
    steps = int(present_ms / dt)
    for pattern in patterns:
        net.layers["input"].reset(); net.layers["output"].reset()
        for _ in range(steps):
            _boost_input(net, pattern, scale=0.1)
            net.step(dt)


def _measure(net: NeuromorphicNetwork, patterns, dt=0.1, present_ms=20.0):
    steps = int(present_ms / dt)
    responses = []
    for pattern in patterns:
        net.layers["input"].reset(); net.layers["output"].reset()
        spike_counts = np.zeros(net.layers["output"].size)
        for _ in range(steps):
            _boost_input(net, pattern, scale=0.1)
            net.step(dt)
            spikes = [n.is_spiking for n in net.layers["output"].neuron_population.neurons]
            spike_counts += np.array(spikes, dtype=float)
        responses.append(spike_counts)
    return responses


def test_sleep_phase_induces_selectivity():
    """Sleep phase with replay/noise should yield non-identical post-sleep responses."""
    np.random.seed(42)

    dt = 0.1
    net = _build_network()
    pats = _patterns(net.layers["input"].size)

    # Brief training
    for _ in range(3):
        _train_epoch(net, pats, dt=dt, present_ms=20.0)

    before = _measure(net, pats, dt=dt, present_ms=20.0)

    # Sleep/rest with replay and mild noise/downscaling
    net.run_sleep_phase(
        duration=30.0,
        dt=dt,
        downscale_factor=0.99,
        normalize_incoming=True,
        replay={"input": pats[0]},
        noise_std=0.02,
    )

    after = _measure(net, pats, dt=dt, present_ms=20.0)

    # Assert that not all post-sleep responses are identical across patterns
    all_equal = True
    for i in range(len(after)):
        for j in range(i + 1, len(after)):
            if not np.array_equal(after[i], after[j]):
                all_equal = False
                break
        if not all_equal:
            break

    assert not all_equal, "Post-sleep responses should not be identical across all patterns"


