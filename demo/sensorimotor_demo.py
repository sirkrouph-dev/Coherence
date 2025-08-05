"""
Demonstration of the neuromorphic programming system.
Shows sensorimotor control with adaptive learning.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.neuromorphic_api import SensorimotorSystem, NeuromorphicAPI
from core.encoding import RetinalEncoder, CochlearEncoder, SomatosensoryEncoder


def create_training_data(num_trials: int = 50) -> List[Dict[str, Any]]:
    """
    Create synthetic training data for sensorimotor learning.
    
    Args:
        num_trials: Number of training trials
        
    Returns:
        List of training trials
    """
    training_data = []
    
    for trial in range(num_trials):
        # Create synthetic sensory inputs
        visual_input = create_visual_input()
        auditory_input = create_auditory_input()
        tactile_input = create_tactile_input()
        
        # Define target motor output (simple mapping)
        target = np.array([trial % 5, (trial + 1) % 3, trial % 2])  # 3D motor output
        
        # Create trial data
        trial_data = {
            'visual': visual_input,
            'auditory': auditory_input,
            'tactile': tactile_input,
            'target': target,
            'outcome': target,  # Perfect outcome for demonstration
            'trial_id': trial
        }
        
        training_data.append(trial_data)
        
    return training_data


def create_visual_input() -> np.ndarray:
    """Create synthetic visual input (32x32 image)."""
    # Create a simple pattern (circle in center)
    image = np.zeros((32, 32))
    
    # Add a circle in the center
    center_x, center_y = 16, 16
    radius = 8
    
    for y in range(32):
        for x in range(32):
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            if distance <= radius:
                image[y, x] = 1.0
                
    # Add some noise
    noise = np.random.normal(0, 0.1, (32, 32))
    image = np.clip(image + noise, 0, 1)
    
    return image


def create_auditory_input() -> np.ndarray:
    """Create synthetic auditory input (1 second of audio)."""
    # Create a simple tone with some harmonics
    sample_rate = 44100
    duration = 1.0  # 1 second
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create a tone at 440 Hz (A4) with harmonics
    frequency = 440.0
    audio = np.sin(2 * np.pi * frequency * t)
    audio += 0.5 * np.sin(2 * np.pi * 2 * frequency * t)  # Second harmonic
    audio += 0.25 * np.sin(2 * np.pi * 3 * frequency * t)  # Third harmonic
    
    # Add some noise
    noise = np.random.normal(0, 0.1, len(audio))
    audio = audio + noise
    
    # Normalize
    audio = audio / np.max(np.abs(audio))
    
    return audio


def create_tactile_input() -> np.ndarray:
    """Create synthetic tactile input (16x16 pressure map)."""
    # Create a pressure map with a central touch
    pressure_map = np.zeros((16, 16))
    
    # Add a central touch
    center_x, center_y = 8, 8
    for y in range(16):
        for x in range(16):
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            if distance <= 4:
                pressure_map[y, x] = 1.0 - (distance / 4.0)
                
    # Add some noise
    noise = np.random.normal(0, 0.05, (16, 16))
    pressure_map = np.clip(pressure_map + noise, 0, 1)
    
    return pressure_map


def demonstrate_basic_network():
    """Demonstrate basic network functionality."""
    print("\n=== Basic Network Demonstration ===")
    
    # Create API and network
    api = NeuromorphicAPI()
    api.create_network()
    
    # Add layers
    api.add_sensory_layer("input", 50, "rate")
    api.add_processing_layer("hidden", 25, "adex")
    api.add_motor_layer("output", 10)
    
    # Connect layers
    api.connect_layers("input", "hidden", "feedforward", synapse_type="stdp")
    api.connect_layers("hidden", "output", "feedforward", synapse_type="stdp")
    
    # Create input spikes
    input_spikes = [(i, i * 2.0) for i in range(20)]
    
    # Run simulation
    results = api.run_simulation(100.0, external_inputs={"input": input_spikes})
    
    print(f"Simulation completed in {results['duration']} ms")
    
    # Get network info from the API object
    network_info = api.network.get_network_info()
    print(f"Network has {network_info['total_neurons']} neurons")
    print(f"Network has {network_info['total_synapses']} synapses")
    
    # Show spike statistics
    for layer_name, spike_times in results['layer_spike_times'].items():
        print(f"{layer_name} layer: {len(spike_times)} spikes")
    
    return results


def demonstrate_sensorimotor_learning():
    """Demonstrate sensorimotor learning."""
    print("\n=== Sensorimotor Learning Demonstration ===")
    
    # Create sensorimotor system
    system = SensorimotorSystem()
    
    # Get network info
    network_info = system.get_network_info()
    print(f"Created sensorimotor network with {network_info['total_neurons']} neurons")
    print(f"Network has {network_info['total_synapses']} synapses")
    
    # Create training data
    print("Creating training data...")
    training_data = create_training_data(num_trials=20)
    
    # Train the system
    print("Training sensorimotor system...")
    training_results = system.train(training_data, epochs=50)
    
    # Plot learning curves
    training_history = training_results['training_history']
    epochs = [d['epoch'] for d in training_history]
    rewards = [d['average_reward'] for d in training_history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, rewards, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Average Reward')
    plt.title('Sensorimotor Learning Curve')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Plot neuromodulator levels
    neuromodulator_data = training_history
    if neuromodulator_data:
        epochs = [d['epoch'] for d in neuromodulator_data]
        modulator_levels = neuromodulator_data[0]['neuromodulator_levels']
        modulators = list(modulator_levels.keys())
        
        plt.figure(figsize=(12, 6))
        for modulator in modulators:
            levels = [d['neuromodulator_levels'][modulator] for d in neuromodulator_data]
            plt.plot(epochs, levels, label=modulator.value, linewidth=2)
            
        plt.xlabel('Epoch')
        plt.ylabel('Level')
        plt.title('Neuromodulator Levels During Training')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    return training_results


def demonstrate_sensory_encoding():
    """Demonstrate sensory encoding."""
    print("\n=== Sensory Encoding Demonstration ===")
    
    # Test visual encoding
    print("Testing visual encoding...")
    visual_encoder = RetinalEncoder()
    visual_input = create_visual_input()
    visual_spikes = visual_encoder.encode(visual_input)
    print(f"Visual encoding: {len(visual_spikes)} spikes generated")
    
    # Test auditory encoding
    print("Testing auditory encoding...")
    auditory_encoder = CochlearEncoder()
    auditory_input = create_auditory_input()
    auditory_spikes = auditory_encoder.encode(auditory_input)
    print(f"Auditory encoding: {len(auditory_spikes)} spikes generated")
    
    # Test tactile encoding
    print("Testing tactile encoding...")
    tactile_encoder = SomatosensoryEncoder()
    tactile_input = create_tactile_input()
    tactile_spikes = tactile_encoder.encode(tactile_input)
    print(f"Tactile encoding: {len(tactile_spikes)} spikes generated")
    
    # Visualize spike distributions
    plt.figure(figsize=(15, 5))
    
    # Visual spikes
    plt.subplot(1, 3, 1)
    if visual_spikes:
        spike_times = [spike[1] for spike in visual_spikes]
        plt.hist(spike_times, bins=20, alpha=0.7, color='red')
        plt.xlabel('Spike Time (ms)')
        plt.ylabel('Count')
        plt.title('Visual Spike Distribution')
    
    # Auditory spikes
    plt.subplot(1, 3, 2)
    if auditory_spikes:
        spike_times = [spike[1] for spike in auditory_spikes]
        plt.hist(spike_times, bins=20, alpha=0.7, color='blue')
        plt.xlabel('Spike Time (ms)')
        plt.ylabel('Count')
        plt.title('Auditory Spike Distribution')
    
    # Tactile spikes
    plt.subplot(1, 3, 3)
    if tactile_spikes:
        spike_times = [spike[1] for spike in tactile_spikes]
        plt.hist(spike_times, bins=20, alpha=0.7, color='green')
        plt.xlabel('Spike Time (ms)')
        plt.ylabel('Count')
        plt.title('Tactile Spike Distribution')
    
    plt.tight_layout()
    plt.show()


def demonstrate_adaptive_behavior():
    """Demonstrate adaptive behavior with changing inputs."""
    print("\n=== Adaptive Behavior Demonstration ===")
    
    # Create a simple adaptive network
    api = NeuromorphicAPI()
    api.create_network()
    
    # Add layers
    api.add_sensory_layer("sensory", 50, "rate")
    api.add_processing_layer("processing", 25, "adex")
    api.add_motor_layer("motor", 10)
    
    # Connect with STDP learning
    api.connect_layers("sensory", "processing", "feedforward", synapse_type="stdp")
    api.connect_layers("processing", "motor", "feedforward", synapse_type="stdp")
    
    # Run multiple trials with different inputs
    trials = []
    for trial in range(5):
        # Create different input patterns
        input_spikes = [(i, i * 5.0 + trial * 20.0) for i in range(10)]
        
        # Run simulation
        results = api.run_simulation(50.0, external_inputs={"sensory": input_spikes})
        trials.append(results)
        
        print(f"Trial {trial + 1}: {len(results['layer_spike_times'].get('motor', [[]]))} motor spikes")
    
    # Show weight evolution
    print("Network adapted to different input patterns")
    
    return trials


def main():
    """Run all demonstrations."""
    print("Neuromorphic Programming System Demonstration")
    print("=" * 50)
    
    try:
        # Basic network demonstration
        basic_results = demonstrate_basic_network()
        
        # Sensory encoding demonstration
        demonstrate_sensory_encoding()
        
        # Adaptive behavior demonstration
        adaptive_results = demonstrate_adaptive_behavior()
        
        # Sensorimotor learning demonstration
        learning_results = demonstrate_sensorimotor_learning()
        
        print("\n=== Demonstration Complete ===")
        print("All demonstrations completed successfully!")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 