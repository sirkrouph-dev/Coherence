#!/usr/bin/env python3
"""
Image Encoding and Network Processing Example
==============================================

This example demonstrates how to:
1. Load and preprocess an image
2. Encode it using retinal encoding
3. Create a simple visual processing network
4. Run the image through the network
5. Visualize the results

Requirements:
    - numpy
    - matplotlib
    - opencv-python (cv2)
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

# Add parent directory to path to import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.encoding import RetinalEncoder, RateEncoder
from core.network import NeuromorphicNetwork, NetworkBuilder


def create_sample_image(size=(64, 64)):
    """Create a sample image with simple patterns for testing."""
    image = np.zeros(size, dtype=np.uint8)
    
    # Add some patterns
    # Vertical line
    image[:, size[1]//2-2:size[1]//2+2] = 255
    
    # Horizontal line
    image[size[0]//2-2:size[0]//2+2, :] = 255
    
    # Circle in the center
    center = (size[0]//2, size[1]//2)
    radius = min(size) // 4
    y, x = np.ogrid[:size[0], :size[1]]
    mask = (x - center[1])**2 + (y - center[0])**2 <= radius**2
    image[mask] = 200
    
    return image


def encode_image(image: np.ndarray) -> Dict[str, Any]:
    """
    Encode an image using retinal encoding.
    
    Args:
        image: Input image (grayscale or RGB)
        
    Returns:
        Dictionary containing encoded representations
    """
    print("Encoding image with retinal encoder...")
    
    # Initialize retinal encoder
    encoder = RetinalEncoder(resolution=(32, 32))
    
    # Encode the image
    encoded = encoder.encode(image)
    
    print(f"  - ON-center response shape: {encoded['on_center'].shape}")
    print(f"  - OFF-center response shape: {encoded['off_center'].shape}")
    
    return encoded


def convert_to_spike_trains(encoded_image: Dict[str, Any], duration: float = 100.0):
    """
    Convert encoded image to spike trains.
    
    Args:
        encoded_image: Output from retinal encoder
        duration: Simulation duration in ms
        
    Returns:
        Spike trains for ON and OFF channels
    """
    print("\nConverting to spike trains...")
    
    rate_encoder = RateEncoder(max_rate=100.0)
    
    # Normalize images to 0-1 range
    on_normalized = encoded_image['on_center'] / 255.0
    off_normalized = encoded_image['off_center'] / 255.0
    
    # Generate spike trains
    on_spikes = rate_encoder.encode_array(on_normalized, duration=duration)
    off_spikes = rate_encoder.encode_array(off_normalized, duration=duration)
    
    print(f"  - Generated {len(on_spikes)} ON-channel spikes")
    print(f"  - Generated {len(off_spikes)} OFF-channel spikes")
    
    return on_spikes, off_spikes


def create_visual_network():
    """
    Create a simple visual processing network.
    
    Network structure:
        - Input layer (retinal): 1024 neurons (32x32)
        - V1 layer (edge detection): 256 neurons
        - V2 layer (feature integration): 64 neurons
        - Output layer: 10 neurons
    """
    print("\nCreating visual processing network...")
    
    builder = NetworkBuilder()
    
    # Add layers
    builder.add_sensory_layer("retinal", size=1024)  # 32x32 input
    builder.add_processing_layer("V1", size=256, neuron_type="lif")
    builder.add_processing_layer("V2", size=64, neuron_type="adex")
    builder.add_motor_layer("output", size=10)
    
    # Connect layers with different patterns
    builder.connect_layers("retinal", "V1", 
                          connection_type="feedforward",
                          synapse_type="stdp",
                          connection_probability=0.1)
    
    builder.connect_layers("V1", "V2",
                          connection_type="feedforward", 
                          synapse_type="stdp",
                          connection_probability=0.2)
    
    builder.connect_layers("V2", "output",
                          connection_type="feedforward",
                          synapse_type="stdp",
                          connection_probability=0.3)
    
    # Add lateral connections in V1 for competition
    builder.connect_layers("V1", "V1",
                          connection_type="lateral",
                          synapse_type="stp",
                          connection_probability=0.05)
    
    network = builder.build()
    
    # Print network info
    info = network.get_network_info()
    print(f"  - Total neurons: {info['total_neurons']}")
    print(f"  - Total synapses: {info['total_synapses']}")
    print(f"  - Layers: {list(info['layers'].keys())}")
    
    return network


def inject_spikes_to_network(network: NeuromorphicNetwork, spike_trains, layer_name: str = "retinal"):
    """
    Inject spike trains into the network's input layer.
    
    Args:
        network: The neuromorphic network
        spike_trains: List of (neuron_id, spike_time) tuples
        layer_name: Name of the input layer
    """
    print(f"\nInjecting {len(spike_trains)} spikes into '{layer_name}' layer...")
    
    # Create external input currents for the simulation
    # This is a simplified approach - in reality, you'd use an event-driven simulator
    
    # Group spikes by time bins
    dt = 0.1  # ms
    duration = 100.0  # ms
    time_bins = int(duration / dt)
    
    input_currents = np.zeros((time_bins, network.layers[layer_name].size))
    
    for neuron_id, spike_time in spike_trains:
        if neuron_id < network.layers[layer_name].size:
            time_bin = int(spike_time / dt)
            if 0 <= time_bin < time_bins:
                # Add a brief current pulse
                input_currents[time_bin, neuron_id] += 10.0  # pA
    
    return input_currents


def run_simulation(network: NeuromorphicNetwork, input_currents: np.ndarray, duration: float = 100.0):
    """
    Run the network simulation with input.
    
    Args:
        network: The neuromorphic network
        input_currents: Input current matrix (time x neurons)
        duration: Simulation duration in ms
        
    Returns:
        Simulation results
    """
    print("\nRunning network simulation...")
    
    dt = 0.1  # ms
    num_steps = int(duration / dt)
    
    # Note: This is a simplified simulation approach
    # The actual network.run_simulation() method would be used in production
    
    results = network.run_simulation(duration=duration, dt=dt)
    
    print(f"  - Simulation completed: {results['final_time']} ms")
    
    # Count spikes per layer
    for layer_name, spike_times in results['layer_spike_times'].items():
        total_spikes = sum(len(times) for times in spike_times)
        print(f"  - {layer_name} layer: {total_spikes} total spikes")
    
    return results


def visualize_results(image: np.ndarray, encoded: Dict[str, Any], results: Dict[str, Any]):
    """
    Visualize the encoding and simulation results.
    
    Args:
        image: Original input image
        encoded: Encoded image representations
        results: Simulation results
    """
    print("\nVisualizing results...")
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # ON-center response
    axes[0, 1].imshow(encoded['on_center'], cmap='hot')
    axes[0, 1].set_title('ON-Center Response')
    axes[0, 1].axis('off')
    
    # OFF-center response
    axes[0, 2].imshow(encoded['off_center'], cmap='hot')
    axes[0, 2].set_title('OFF-Center Response')
    axes[0, 2].axis('off')
    
    # Spike raster plot for each layer
    layer_names = ['retinal', 'V1', 'V2']
    for idx, layer_name in enumerate(layer_names):
        if layer_name in results['layer_spike_times']:
            spike_times = results['layer_spike_times'][layer_name]
            
            # Create raster plot
            for neuron_id, times in enumerate(spike_times[:50]):  # Show first 50 neurons
                if times:
                    axes[1, idx].scatter(times, [neuron_id] * len(times), s=1, c='black')
            
            axes[1, idx].set_xlabel('Time (ms)')
            axes[1, idx].set_ylabel('Neuron ID')
            axes[1, idx].set_title(f'{layer_name} Layer Spikes')
            axes[1, idx].set_xlim(0, 100)
    
    plt.suptitle('Image Encoding and Network Processing Example')
    plt.tight_layout()
    plt.show()


def main():
    """Main function to run the complete example."""
    print("=" * 60)
    print("Image Encoding and Network Processing Example")
    print("=" * 60)
    
    # Step 1: Create or load an image
    print("\n1. Creating sample image...")
    image = create_sample_image(size=(64, 64))
    print(f"   Image shape: {image.shape}")
    
    # Step 2: Encode the image
    print("\n2. Encoding image...")
    encoded = encode_image(image)
    
    # Step 3: Convert to spike trains
    print("\n3. Converting to spike trains...")
    on_spikes, off_spikes = convert_to_spike_trains(encoded)
    
    # Combine ON and OFF spikes
    all_spikes = on_spikes + [(n + 512, t) for n, t in off_spikes]  # Offset OFF neurons
    
    # Step 4: Create the network
    print("\n4. Creating visual processing network...")
    network = create_visual_network()
    
    # Step 5: Inject spikes and run simulation
    print("\n5. Processing image through network...")
    input_currents = inject_spikes_to_network(network, all_spikes)
    results = run_simulation(network, input_currents)
    
    # Step 6: Visualize results
    print("\n6. Visualizing results...")
    visualize_results(image, encoded, results)
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
    
    return network, results


if __name__ == "__main__":
    network, results = main()
    
    # Additional analysis
    print("\n\nAdditional Analysis:")
    print("-" * 40)
    
    # Analyze weight changes (if STDP was active)
    for conn_name, weight_matrix in results['weight_matrices'].items():
        if weight_matrix is not None:
            print(f"\nConnection {conn_name}:")
            print(f"  - Weight matrix shape: {weight_matrix.shape}")
            print(f"  - Mean weight: {np.mean(weight_matrix):.4f}")
            print(f"  - Weight range: [{np.min(weight_matrix):.4f}, {np.max(weight_matrix):.4f}]")
    
    print("\nTip: Try modifying the image patterns or network parameters to see different responses!")
