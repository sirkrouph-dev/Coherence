#!/usr/bin/env python3
"""
Debug Learning Demo - Find out why the network isn't learning!
Let's diagnose step by step what's happening.
"""

import numpy as np
import matplotlib.pyplot as plt
from api.neuromorphic_api import NeuromorphicAPI
from core.encoding import RetinalEncoder

class LearningDebugger:
    def __init__(self):
        print("🔍 LEARNING DIAGNOSTIC TOOL")
        print("=" * 50)
        
        # Create minimal network for debugging
        self.api = NeuromorphicAPI()
        self.api.create_network()
        
        # Very simple network
        self.api.add_sensory_layer("input", 64, "lif")      # 8x8 input
        self.api.add_processing_layer("hidden", 20, "lif")  # Small hidden layer
        self.api.add_motor_layer("output", 4)               # 4 shape classes
        
        # Connect with standard parameters first
        self.api.connect_layers("input", "hidden", "stdp", 
                               connection_probability=0.3)
        self.api.connect_layers("hidden", "output", "stdp", 
                               connection_probability=0.5)
        
        print(f"🧠 Debug Network Created:")
        print(f"   Input: 64 neurons")
        print(f"   Hidden: 20 neurons") 
        print(f"   Output: 4 neurons")
        
        # Create simple encoder
        self.encoder = RetinalEncoder(resolution=(8, 8))
    
    def create_simple_pattern(self, pattern_type, size=8):
        """Create very simple, high-contrast patterns"""
        image = np.zeros((size, size))
        
        if pattern_type == "circle":
            # Simple filled circle
            center = size // 2
            for i in range(size):
                for j in range(size):
                    if (i - center)**2 + (j - center)**2 <= (size//3)**2:
                        image[i, j] = 1.0
        
        elif pattern_type == "square":
            # Simple filled square
            start = size // 4
            end = 3 * size // 4
            image[start:end, start:end] = 1.0
        
        elif pattern_type == "line":
            # Diagonal line
            for i in range(size):
                if i < size:
                    image[i, i] = 1.0
        
        elif pattern_type == "corner":
            # L-shape
            image[0:size//2, 0:2] = 1.0  # Vertical line
            image[0:2, 0:size//2] = 1.0  # Horizontal line
        
        return image
    
    def test_input_encoding(self):
        """Test if input patterns generate spikes"""
        print(f"\n🔍 TESTING INPUT ENCODING")
        print("-" * 30)
        
        patterns = ["circle", "square", "line", "corner"]
        
        for pattern_name in patterns:
            pattern = self.create_simple_pattern(pattern_name)
            
            # Convert to spike trains
            spike_data = self.encoder.encode(pattern)
            
            print(f"\n{pattern_name.upper()}:")
            print(f"  Pattern sum: {np.sum(pattern):.1f}")
            print(f"  Pattern max: {np.max(pattern):.1f}")
            
            if isinstance(spike_data, dict) and 'spike_times' in spike_data:
                spike_times = spike_data['spike_times']
                total_spikes = len(spike_times) if spike_times else 0
                print(f"  Total input spikes: {total_spikes}")
                
                if total_spikes > 0:
                    print(f"  First few spikes: {spike_times[:5] if len(spike_times) >= 5 else spike_times}")
                else:
                    print(f"  ❌ NO SPIKES GENERATED!")
            else:
                print(f"  ❌ Invalid spike data format: {type(spike_data)}")
                print(f"  Data content: {spike_data}")  # Debug what we actually get
    
    def test_network_response(self):
        """Test if network responds to direct stimulation"""
        print(f"\n🔍 TESTING NETWORK RESPONSE") 
        print("-" * 30)
        
        # Create strong direct input
        strong_input = {}
        
        # Stimulate first 10 input neurons strongly
        for neuron_id in range(10):
            strong_input[f"input_{neuron_id}"] = [
                (5.0, 20.0),   # Strong current from 5ms to 20ms
                (25.0, 40.0),  # Another pulse
                (45.0, 60.0),  # Third pulse
            ]
        
        print(f"Direct stimulation: 10 neurons with strong pulses")
        
        # Run simulation
        results = self.api.run_simulation(duration=100.0, external_inputs=strong_input)
        
        # Check results
        if 'layer_spike_times' in results:
            for layer, spikes in results['layer_spike_times'].items():
                print(f"\n{layer.upper()} LAYER:")
                if spikes:
                    print(f"  Total spikes: {len(spikes)}")
                    print(f"  Spike times: {spikes[:10] if len(spikes) >= 10 else spikes}")
                else:
                    print(f"  ❌ NO SPIKES!")
        else:
            print(f"❌ No spike data in results: {list(results.keys())}")
    
    def test_neuron_parameters(self):
        """Check if neuron parameters allow spiking"""
        print(f"\n🔍 TESTING NEURON PARAMETERS")
        print("-" * 30)
        
        print("Checking if network was created successfully...")
        if self.api.network:
            print("✅ Network exists")
            layers = list(self.api.network.layers.keys())
            print(f"  Layers: {layers}")
        else:
            print("❌ No network found!")
            return
        
        # Try to get neuron parameters (this might not work depending on API)
        try:
            # Check what neuron types we can create
            from core.neurons import NeuronFactory
            
            print("Testing LIF neuron parameters...")
            test_neuron = NeuronFactory.create_neuron('lif', neuron_id=0)
            
            print(f"  Neuron type: {type(test_neuron)}")
            print(f"  Has threshold: {hasattr(test_neuron, 'threshold')}")
            print(f"  Has membrane potential: {hasattr(test_neuron, 'v')}")
            
            # Check available attributes safely
            attrs = [attr for attr in dir(test_neuron) if not attr.startswith('_')]
            print(f"  Available attributes: {attrs[:10]}...")  # First 10 attrs
            
            # Try to access common neuron properties safely using getattr
            try:
                threshold = getattr(test_neuron, 'threshold', 'Not available')
                print(f"  Threshold: {threshold}")
                
                v_rest = getattr(test_neuron, 'v_rest', 'Not available')
                print(f"  Resting potential: {v_rest}")
                
                # Check membrane potential
                v_membrane = getattr(test_neuron, 'v', 'Not available')
                print(f"  Membrane potential: {v_membrane}")
                
            except Exception as attr_error:
                print(f"  Attribute access error: {attr_error}")
            
            # Test if neuron can spike with strong input
            test_neuron.reset()
            strong_current = 50.0  # Very strong input
            
            for step in range(100):
                dt = 0.1
                spiked = test_neuron.step(dt, strong_current)
                if spiked:
                    print(f"  ✅ Neuron spikes at step {step} with {strong_current}nA input")
                    break
            else:
                print(f"  ❌ Neuron doesn't spike even with {strong_current}nA!")
                
        except Exception as e:
            print(f"  Error testing neuron: {e}")
    
    def run_full_diagnosis(self):
        """Run complete diagnostic sequence"""
        print("🚀 STARTING COMPLETE LEARNING DIAGNOSIS")
        print("=" * 50)
        
        self.test_input_encoding()
        self.test_network_response() 
        self.test_neuron_parameters()
        
        print(f"\n{'='*50}")
        print(f"📋 DIAGNOSIS COMPLETE")
        print(f"{'='*50}")
        print(f"🔍 Check results above to identify the learning bottleneck!")

def main():
    debugger = LearningDebugger()
    debugger.run_full_diagnosis()

if __name__ == "__main__":
    main()
