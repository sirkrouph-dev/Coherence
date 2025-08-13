#!/usr/bin/env python3
"""
NEUROMORPHIC LEARNING DIAGNOSTIC TOOL
Deep investigation of learning mechanisms and synaptic behavior
"""

import numpy as np
from core.network import NeuromorphicNetwork
import json
from datetime import datetime

class LearningDiagnostic:
    def __init__(self):
        print("🔬 NEUROMORPHIC LEARNING DIAGNOSTIC")
        print("=" * 40)
        print("Deep investigation of learning mechanisms")
        
        self.network = NeuromorphicNetwork()
        self.setup_diagnostic_network()
        
    def setup_diagnostic_network(self):
        """Create minimal network for detailed analysis"""
        # Very small network for detailed observation
        self.network.add_layer("input", 4, "lif")    # 2x2 input
        self.network.add_layer("output", 2, "lif")   # 2 outputs
        
        # Single connection for detailed analysis
        self.network.connect_layers("input", "output", "stdp",
                                  connection_probability=1.0,  # Full connectivity
                                  weight=1.0,                  # Standard weight
                                  A_plus=0.2,                 # Very strong LTP
                                  A_minus=0.1,                # Strong LTD
                                  tau_stdp=30.0)              # Learning window
        
        print(f"✅ Diagnostic network: 4 → 2 neurons (full connectivity)")
        print(f"✅ Very strong STDP: A+ = 0.2, A- = 0.1")
        
    def create_simple_patterns(self):
        """Create minimal patterns for clear analysis"""
        patterns = {
            'pattern_A': np.array([1, 0, 0, 1], dtype=float),  # Corners
            'pattern_B': np.array([0, 1, 1, 0], dtype=float)   # Center
        }
        
        targets = {
            'pattern_A': [1, 0],  # Output neuron 0
            'pattern_B': [0, 1]   # Output neuron 1
        }
        
        return patterns, targets
    
    def analyze_neuron_behavior(self):
        """Analyze individual neuron firing behavior"""
        print(f"\n🧪 NEURON BEHAVIOR ANALYSIS")
        print("-" * 30)
        
        # Test neuron responsiveness to different currents
        input_pop = self.network.layers["input"].neuron_population
        output_pop = self.network.layers["output"].neuron_population
        
        current_levels = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0]
        
        for current in current_levels:
            input_pop.reset()
            output_pop.reset()
            
            # Test current level
            input_currents = [current, 0.0, 0.0, 0.0]
            output_currents = [current, 0.0]
            
            spikes_in = 0
            spikes_out = 0
            
            for step in range(20):
                in_states = input_pop.step(0.1, input_currents)
                out_states = output_pop.step(0.1, output_currents)
                spikes_in += sum(in_states)
                spikes_out += sum(out_states)
            
            print(f"Current {current:4.0f}mA: Input spikes={spikes_in}, Output spikes={spikes_out}")
    
    def analyze_synapse_weights(self):
        """Analyze synaptic weight changes"""
        print(f"\n🔗 SYNAPTIC WEIGHT ANALYSIS")
        print("-" * 30)
        
        # Get initial weights
        initial_weights = self.capture_all_weights()
        print(f"Initial weights: {initial_weights}")
        
        # Perform one training episode
        patterns, targets = self.create_simple_patterns()
        
        print(f"\nTraining pattern A...")
        self.train_diagnostic_pattern(patterns['pattern_A'], targets['pattern_A'])
        
        # Check weight changes
        post_training_weights = self.capture_all_weights()
        print(f"After training: {post_training_weights}")
        
        # Calculate weight changes
        weight_changes = []
        for i, (initial, final) in enumerate(zip(initial_weights, post_training_weights)):
            change = final - initial
            weight_changes.append(change)
            print(f"Synapse {i}: {initial:.3f} → {final:.3f} (Δ = {change:+.3f})")
        
        return weight_changes
    
    def capture_all_weights(self):
        """Capture all synaptic weights"""
        weights = []
        for (pre_layer, post_layer), connection in self.network.connections.items():
            if connection.synapse_population and hasattr(connection.synapse_population, 'synapses'):
                for (pre_idx, post_idx), synapse in connection.synapse_population.synapses.items():
                    weights.append(synapse.weight)
        return weights
    
    def train_diagnostic_pattern(self, pattern, target):
        """Train single pattern with detailed monitoring"""
        input_currents = [60.0 if pixel > 0.5 else 0.0 for pixel in pattern]
        target_currents = [40.0 if target[i] == 1 else 0.0 for i in range(2)]
        
        input_pop = self.network.layers["input"].neuron_population
        output_pop = self.network.layers["output"].neuron_population
        
        dt = 0.1
        steps = 30
        
        spike_times = {'input': [], 'output': []}
        
        for step in range(steps):
            time = step * dt
            
            # Input stimulation
            input_states = input_pop.step(dt, input_currents)
            output_states = output_pop.step(dt, target_currents)
            
            # Record spike times
            for i, spiked in enumerate(input_states):
                if spiked:
                    spike_times['input'].append((i, time))
            
            for i, spiked in enumerate(output_states):
                if spiked:
                    spike_times['output'].append((i, time))
            
            # Critical: Network step for STDP
            self.network.step(dt)
        
        print(f"Input spikes: {spike_times['input']}")
        print(f"Output spikes: {spike_times['output']}")
        
        return spike_times
    
    def test_synaptic_transmission(self):
        """Test if synaptic transmission is working"""
        print(f"\n⚡ SYNAPTIC TRANSMISSION TEST")
        print("-" * 30)
        
        # Force input spikes and see if they reach output
        input_pop = self.network.layers["input"].neuron_population
        output_pop = self.network.layers["output"].neuron_population
        
        # Reset network
        input_pop.reset()
        output_pop.reset()
        
        # Strong input to trigger spikes
        strong_input = [100.0, 0.0, 0.0, 0.0]  # Very strong current
        
        print("Testing synaptic transmission...")
        for step in range(50):
            time = step * 0.1
            
            # Input stimulation only
            input_states = input_pop.step(0.1, strong_input)
            
            # No external current to output - test pure synaptic transmission
            output_states = output_pop.step(0.1, [0.0, 0.0])
            
            if any(input_states):
                print(f"Step {step}: Input spike detected")
            
            if any(output_states):
                print(f"Step {step}: Output spike detected - SYNAPTIC TRANSMISSION WORKING!")
            
            # Network step
            self.network.step(0.1)
    
    def comprehensive_diagnostic(self):
        """Run complete diagnostic analysis"""
        print("Starting comprehensive neuromorphic diagnostic...")
        
        # 1. Neuron behavior analysis
        self.analyze_neuron_behavior()
        
        # 2. Synaptic weight analysis
        weight_changes = self.analyze_synapse_weights()
        
        # 3. Synaptic transmission test
        self.test_synaptic_transmission()
        
        # 4. Summary
        print(f"\n📊 DIAGNOSTIC SUMMARY")
        print("=" * 25)
        
        # Check if neurons can fire
        neuron_responsive = True  # We'll determine this from behavior analysis
        
        # Check if weights change
        weights_changing = any(abs(change) > 0.001 for change in weight_changes)
        
        # Check synaptic connectivity
        total_synapses = len(weight_changes)
        
        print(f"✅ Network structure: {total_synapses} synapses created")
        print(f"{'✅' if weights_changing else '❌'} Weight plasticity: {'Working' if weights_changing else 'Not working'}")
        print(f"Weight changes: {[f'{w:+.3f}' for w in weight_changes]}")
        
        if weights_changing:
            print(f"\n🎯 LEARNING MECHANISM STATUS: PARTIALLY FUNCTIONAL")
            print(f"   - Synaptic plasticity is working")
            print(f"   - Issue may be in transmission or neuron thresholds")
        else:
            print(f"\n❌ LEARNING MECHANISM STATUS: NOT FUNCTIONAL")
            print(f"   - Synaptic plasticity is not working")
            print(f"   - Need to investigate STDP implementation")
        
        return {
            'weights_changing': weights_changing,
            'weight_changes': weight_changes,
            'total_synapses': total_synapses
        }

if __name__ == "__main__":
    diagnostic = LearningDiagnostic()
    diagnostic.comprehensive_diagnostic()
