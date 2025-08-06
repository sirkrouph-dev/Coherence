"""
Advanced sensorimotor training system with adaptive learning.
"""

from core.neuromodulation import AdaptiveLearningController, NeuromodulatorType
from core.encoding import CochlearEncoder, RetinalEncoder, SomatosensoryEncoder
from api.neuromorphic_system import NeuromorphicSystem
import os
import sys
from typing import Any, Dict, List

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_sensorimotor_system():
    """Create a simplified sensorimotor system for faster training"""
    system = NeuromorphicSystem(backend="time_driven")

    # Add smaller encoders
    system.add_sensory_encoder("vision", RetinalEncoder(resolution=(8, 8)))
    system.add_sensory_encoder("auditory", CochlearEncoder(frequency_bands=16))
    system.add_sensory_encoder("tactile", SomatosensoryEncoder(sensor_grid=(4, 4)))

    # Use much smaller network configuration
    network_config = {
        "layers": [
            {
                "name": "sensory",
                "type": "sensory",
                "size": 64,
                "encoding_type": "multimodal",
            },
            {
                "name": "integration",
                "type": "processing",
                "size": 32,
                "neuron_type": "adex",
            },
            {
                "name": "decision",
                "type": "processing",
                "size": 16,
                "neuron_type": "adex",
            },
            {"name": "motor", "type": "motor", "size": 8},
        ],
        "connections": [
            {
                "pre_layer": "sensory",
                "post_layer": "integration",
                "connection_type": "feedforward",
                "synapse_type": "rstdp",
                "probability": 0.3,
                "params": {"learning_rate": 0.05},
            },
            {
                "pre_layer": "integration",
                "post_layer": "decision",
                "connection_type": "feedforward",
                "synapse_type": "rstdp",
                "probability": 0.2,
                "params": {"learning_rate": 0.03},
            },
            {
                "pre_layer": "decision",
                "post_layer": "motor",
                "connection_type": "feedforward",
                "synapse_type": "rstdp",
                "probability": 0.4,
                "params": {"learning_rate": 0.02},
            },
            {
                "pre_layer": "motor",
                "post_layer": "integration",
                "connection_type": "feedback",
                "synapse_type": "stdp",
                "probability": 0.1,
            },
        ],
    }

    system.build_network(network_config)
    system.modulatory_controller = AdaptiveLearningController()
    return system


def train_sensorimotor_system(system, training_data, epochs=10):
    """Train the sensorimotor system with simplified parameters"""
    for epoch in range(epochs):
        total_reward = 0
        for i, trial in enumerate(training_data):
            # Use shorter simulation duration
            results = system.run_simulation(
                duration=50.0,  # Reduced from 100.0
                inputs={
                    "vision": trial["image"],
                    "auditory": trial["sound"],
                    "tactile": trial["touch"],
                },
            )

            motor_spikes = results["layer_spike_times"].get("motor", [])
            action = determine_action(motor_spikes)
            reward = calculate_reward(action, trial["target"])
            total_reward += reward

            system.update_learning(
                reward=reward,
                expected_reward=system.reward_system.expected_reward,
                positive_events=1 if reward > 0.5 else 0,
                negative_events=1 if reward < 0.2 else 0,
                threat_signals=trial.get("threat", 0),
                task_difficulty=trial.get("difficulty", 0.5),
            )

            # Apply adaptive learning
            network_info = system.network.get_network_info()
            system.modulatory_controller.update_learning_rates(network_info)
            system.modulatory_controller.apply_learning(system.network)

            # Add progress indicator
            if i % 5 == 0:
                print(f"  Epoch {epoch}, Trial {i}/{len(training_data)}")

        print(f"Epoch {epoch}: Avg Reward = {total_reward/len(training_data):.2f}")


def determine_action(motor_spikes):
    """Determine action from motor layer spiking pattern"""
    if not motor_spikes:
        return 0
    spike_counts = [len(spikes) for spikes in motor_spikes]
    return np.argmax(spike_counts)


def calculate_reward(action, target):
    """Calculate reward based on action accuracy"""
    return 1.0 if action == target else -0.2


def create_training_data(num_trials=10):
    """Generate simplified training data with smaller inputs"""
    training_data = []

    for i in range(num_trials):
        # Create much smaller synthetic sensory inputs
        image = np.random.rand(8, 8)  # Reduced from 32x32
        sound = np.random.randn(100)  # Reduced from 44100
        touch = np.random.rand(4, 4)  # Reduced from 16x16

        # Random target action (0-7 for 8 motor neurons)
        target = np.random.randint(0, 8)

        # Add some structure to make learning possible
        if i % 3 == 0:  # Every 3rd trial has a pattern
            # Create a simple pattern: bright spot in image correlates with target
            image[target // 4, target % 4] = 1.0  # Bright spot at target location

        training_data.append(
            {
                "image": image,
                "sound": sound,
                "touch": touch,
                "target": target,
                "difficulty": 0.3 + 0.4 * np.random.random(),
                "threat": 0.1 * np.random.random(),
            }
        )

    return training_data


def demonstrate_adaptive_learning():
    """Demonstrate the adaptive sensorimotor learning system"""
    print("\n=== Adaptive Sensorimotor Learning Demonstration ===")

    # Create system
    system = create_sensorimotor_system()
    print("Created adaptive sensorimotor system")

    # Generate training data
    training_data = create_training_data(num_trials=10)  # Reduced from 20
    print(f"Generated {len(training_data)} training trials")

    # Train the system
    print("Training sensorimotor system...")
    train_sensorimotor_system(system, training_data, epochs=5)  # Reduced from 30

    # Test the trained system
    print("\nTesting trained system...")
    test_trials = create_training_data(num_trials=5)  # Reduced from 10

    correct_actions = 0
    for trial in test_trials:
        results = system.run_simulation(
            duration=50.0,  # Reduced from 100.0
            inputs={
                "vision": trial["image"],
                "auditory": trial["sound"],
                "tactile": trial["touch"],
            },
        )

        motor_spikes = results["layer_spike_times"].get("motor", [])
        action = determine_action(motor_spikes)

        if action == trial["target"]:
            correct_actions += 1

    accuracy = correct_actions / len(test_trials)
    print(f"Test Accuracy: {accuracy:.2f} ({correct_actions}/{len(test_trials)})")

    # Show network state
    state = system.get_network_state()
    print(f"\nNetwork State:")
    print(f"  Dopamine Level: {state['modulators'][NeuromodulatorType.DOPAMINE]:.3f}")
    print(f"  Learning Rate: {system.get_learning_rate():.4f}")
    print(f"  Reward Prediction Error: {state['reward_prediction_error']:.3f}")

    return system


if __name__ == "__main__":
    demonstrate_adaptive_learning()
