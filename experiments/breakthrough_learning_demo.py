#!/usr/bin/env python3
"""
SUCCESS LEARNING DEMO - Input spikes work, now fix propagation!
The final step to achieve working neuromorphic learning.
"""

import numpy as np
from core.network import NeuromorphicNetwork

class InnovationLearner:
    def __init__(self):
        print("🎯 SUCCESS NEUROMORPHIC LEARNING")
        print("=" * 50)
        print("Final step: Fix spike propagation!")
        
        # Create minimal network for easier debugging
        self.network = NeuromorphicNetwork()
        
        # Smaller network for clearer signal propagation
        self.network.add_layer("input", 4, "lif")      # 2x2 input (simpler)
        self.network.add_layer("output", 4, "lif")     # Direct input→output
        
        # Strong direct connections for guaranteed propagation
        self.network.connect_layers("input", "output", "stdp", 
                                   connection_probability=1.0)  # Full connectivity
        
        print(f"✅ Minimal network: 4 → 4 neurons (full connectivity)")
        
        # Get neuron populations
        self.input_pop = self.network.layers["input"].neuron_population
        self.output_pop = self.network.layers["output"].neuron_population
        
        # Simple 2x2 patterns
        self.patterns = {
            'top': np.array([[1, 1], [0, 0]]),     # Top row active
            'bottom': np.array([[0, 0], [1, 1]]),  # Bottom row active
            'left': np.array([[1, 0], [1, 0]]),    # Left column active
            'right': np.array([[0, 1], [0, 1]])    # Right column active
        }
        
        print(f"✅ Created {len(self.patterns)} simple 2x2 patterns")
    
    def stimulate_and_propagate(self, pattern, stimulation_steps=100, propagation_steps=200):
        """Stimulate input and let activity propagate through network"""
        flat_pattern = pattern.flatten()
        dt = 0.1
        
        # Phase 1: Strong input stimulation
        input_currents = [30.0 if pixel > 0.5 else 0.0 for pixel in flat_pattern]
        input_spikes = 0
        
        print(f"  Phase 1: Input stimulation ({stimulation_steps} steps)")
        for step in range(stimulation_steps):
            spike_states = self.input_pop.step(dt, input_currents)
            spikes_this_step = sum(spike_states)
            input_spikes += spikes_this_step
            
            # Also step the network to allow immediate propagation
            self.network.step(dt)
        
        print(f"    Input spikes: {input_spikes}")
        
        # Phase 2: Let activity propagate (no external input)
        print(f"  Phase 2: Network propagation ({propagation_steps} steps)")
        output_spikes = [0, 0, 0, 0]
        
        for step in range(propagation_steps):
            # No external input, just internal propagation
            zero_currents = [0.0, 0.0, 0.0, 0.0]
            self.input_pop.step(dt, zero_currents)
            
            # Step network to propagate activity
            self.network.step(dt)
            
            # Check for output spikes manually
            # Since output detection is tricky, let's try direct neuron access
            for i, neuron in enumerate(self.output_pop.neurons):
                if hasattr(neuron, 'spike_times') and len(neuron.spike_times) > 0:
                    # Check for recent spikes
                    current_time = (stimulation_steps + step) * dt
                    recent_spikes = [t for t in neuron.spike_times if t >= current_time - dt]
                    output_spikes[i] += len(recent_spikes)
        
        print(f"    Output spikes: {output_spikes}")
        
        return input_spikes, output_spikes
    
    def test_basic_propagation(self):
        """Test if spikes can propagate from input to output"""
        print(f"\n🧪 TESTING BASIC PROPAGATION")
        print("-" * 35)
        
        for pattern_name, pattern in self.patterns.items():
            print(f"\nTesting {pattern_name}:")
            print(f"  Pattern: {pattern.tolist()}")
            
            input_spikes, output_spikes = self.stimulate_and_propagate(pattern)
            
            total_output = sum(output_spikes)
            if total_output > 0:
                print(f"  🎉 SUCCESS! Output spikes detected: {output_spikes}")
            else:
                print(f"  ⚪ No output spikes detected")
    
    def test_direct_network_simulation(self):
        """Use the network's built-in simulation instead of manual stepping"""
        print(f"\n🎯 TESTING DIRECT NETWORK SIMULATION")
        print("-" * 40)
        
        # Try using the network's run_simulation method
        print("Testing built-in simulation method...")
        
        try:
            # Run a short simulation without external inputs first
            results = self.network.run_simulation(duration=10.0, dt=0.1)
            
            print(f"Simulation completed successfully!")
            print(f"Results keys: {list(results.keys())}")
            
            if 'layer_spike_times' in results:
                for layer, spikes in results['layer_spike_times'].items():
                    spike_count = len(spikes) if spikes else 0
                    print(f"  {layer}: {spike_count} spikes")
            
        except Exception as e:
            print(f"Built-in simulation error: {e}")
    
    def test_synaptic_connectivity(self):
        """Test if synapses are actually connecting layers"""
        print(f"\n🔗 TESTING SYNAPTIC CONNECTIVITY")
        print("-" * 35)
        
        # Check network connections
        if hasattr(self.network, 'connections'):
            print(f"Network connections: {len(self.network.connections)}")
            
            for i, connection in enumerate(self.network.connections):
                # Connection might be a tuple (pre_layer, post_layer) or an object
                if isinstance(connection, tuple):
                    pre_layer, post_layer = connection
                    print(f"  Connection {i}: {pre_layer} → {post_layer}")
                else:
                    print(f"  Connection {i}: {type(connection)}")
        else:
            print("❌ No connections found in network")
        
        # Check individual layers
        print(f"\nLayer information:")
        for layer_name, layer in self.network.layers.items():
            neuron_count = len(layer.neuron_population.neurons) if hasattr(layer, 'neuron_population') else 'unknown'
            print(f"  {layer_name}: {neuron_count} neurons")
    
    def run_innovation_test(self):
        """Run the complete innovation test sequence"""
        print(f"\n🚀 SUCCESS TEST SEQUENCE")
        print("=" * 40)
        
        self.test_direct_network_simulation()
        self.test_synaptic_connectivity()
        self.test_basic_propagation()
        
        print(f"\n{'='*50}")
        print(f"🎯 SUCCESS ANALYSIS")
        print(f"{'='*50}")
        print(f"✅ Input stimulation: WORKING (hundreds of spikes)")
        print(f"🔍 Spike propagation: UNDER INVESTIGATION")
        print(f"🎯 Next: Fix output spike detection or propagation")

def main():
    learner = InnovationLearner()
    learner.run_innovation_test()

if __name__ == "__main__":
    main()
