import os
import sys
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.network import NeuromorphicNetwork


def build_network(input_size: int = 10, output_size: int = 5) -> NeuromorphicNetwork:
    net = NeuromorphicNetwork()
    net.add_layer("input", input_size, "lif")
    net.add_layer("output", output_size, "lif")
    net.connect_layers("input", "output", "stdp", connection_probability=1.0)
    return net


def generate_patterns(input_size: int):
    patterns = []
    p0 = np.zeros(input_size); p0[0:3] = 50.0; patterns.append(p0)
    p1 = np.zeros(input_size); p1[3:6] = 50.0; patterns.append(p1)
    p2 = np.zeros(input_size); p2[6:9] = 50.0; patterns.append(p2)
    return patterns


def boost_input(net: NeuromorphicNetwork, pattern: np.ndarray, scale: float = 0.1):
    layer = net.layers["input"].neuron_population
    for i, neuron in enumerate(layer.neurons):
        neuron.membrane_potential += float(pattern[i]) * scale


def run_train_epoch(net: NeuromorphicNetwork, patterns, dt=0.1, present_ms=20.0):
    steps = int(present_ms / dt)
    for pattern in patterns:
        net.layers["input"].reset(); net.layers["output"].reset()
        for _ in range(steps):
            boost_input(net, pattern, scale=0.1)
            net.step(dt)


def measure_responses(net: NeuromorphicNetwork, patterns, dt=0.1, present_ms=20.0):
    steps = int(present_ms / dt)
    responses = []
    for pattern in patterns:
        net.layers["input"].reset(); net.layers["output"].reset()
        spike_counts = np.zeros(net.layers["output"].size)
        for _ in range(steps):
            boost_input(net, pattern, scale=0.1)
            net.step(dt)
            spikes = [n.is_spiking for n in net.layers["output"].neuron_population.neurons]
            spike_counts += np.array(spikes, dtype=float)
        responses.append(spike_counts)
    return responses


def main():
    dt = 0.1
    net = build_network()
    patterns = generate_patterns(net.layers["input"].size)

    # Train briefly
    for _ in range(5):
        run_train_epoch(net, patterns, dt=dt, present_ms=20.0)

    # Measure before sleep
    before = measure_responses(net, patterns, dt=dt, present_ms=20.0)

    # Sleep/rest with replay and noise
    replay_current = {"input": patterns[0]}  # replay one pattern
    net.run_sleep_phase(
        duration=50.0,
        dt=dt,
        downscale_factor=0.98,
        normalize_incoming=True,
        replay=replay_current,
        noise_std=0.05,
    )

    # Measure after sleep
    after = measure_responses(net, patterns, dt=dt, present_ms=20.0)

    # Print summary
    def fmt(vec):
        return ", ".join(f"{v:.0f}" for v in vec)

    print("Responses before sleep:")
    for i, r in enumerate(before):
        print(f"  Pattern {i}: [{fmt(r)}]")
    print("Responses after sleep:")
    for i, r in enumerate(after):
        print(f"  Pattern {i}:  [{fmt(r)}]")

    # Weight change summary
    conn = net.connections[("input", "output")]
    W = conn.get_weight_matrix()
    print(f"Non-zero weights: {int(np.sum(W>0))}, mean: {float(np.mean(W[W>0])) if np.any(W>0) else 0:.3f}")


if __name__ == "__main__":
    main()
