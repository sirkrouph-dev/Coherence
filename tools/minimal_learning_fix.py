#!/usr/bin/env python3
"""
MINIMAL LEARNING FIX
Direct implementation of learning with proper timing
"""

import numpy as np
from core.network import NeuromorphicNetwork

class MinimalLearningFix:
    def __init__(self):
        print("🔧 MINIMAL LEARNING FIX")
        print("=" * 25)
        
        self.network = NeuromorphicNetwork()
        self.setup_minimal_network()
        
    def setup_minimal_network(self):
        """Create minimal network for learning fix"""
        self.network.add_layer("input", 4, "lif")
        self.network.add_layer("output", 2, "lif")
        
        self.network.connect_layers("input", "output", "stdp",
                                  connection_probability=1.0,
                                  weight=1.0,
                                  A_plus=0.3,  # Strong learning
                                  A_minus=0.15)
        
        print("✅ Minimal network: 4 → 2 neurons")
        
    def manual_stdp_training(self, input_pattern, target_neuron):
        """Manually implement STDP training with proper timing"""
        print(f"\n🧪 Manual STDP Training")
        print(f"Pattern: {input_pattern}, Target: {target_neuron}")
        
        # Get populations
        input_pop = self.network.layers["input"].neuron_population
        output_pop = self.network.layers["output"].neuron_population
        
        # Convert pattern to currents
        input_currents = [80.0 if x > 0.5 else 0.0 for x in input_pattern]
        target_currents = [60.0 if i == target_neuron else 0.0 for i in range(2)]
        
        dt = 0.1
        time = 0.0
        
        # Record initial weights
        initial_weights = self.get_all_weights()
        print(f"Initial weights: {[f'{w:.3f}' for w in initial_weights]}")
        
        # Training sequence with proper timing for STDP
        for step in range(50):
            time = step * dt
            
            # Input stimulation (early phase)
            if step < 20:
                input_spikes = input_pop.step(dt, input_currents)
            else:
                input_spikes = input_pop.step(dt, [0.0] * 4)
            
            # Target output (overlapping with input for causality)
            if 5 <= step < 25:
                output_spikes = output_pop.step(dt, target_currents)
            else:
                output_spikes = output_pop.step(dt, [0.0] * 2)
            
            # Manual STDP update - directly call synapse methods
            if any(input_spikes) or any(output_spikes):
                print(f"Step {step}: Input={input_spikes}, Output={output_spikes}")
                self.manual_weight_update(input_spikes, output_spikes, time)
            
            # Network step for other processing
            self.network.step(dt)
        
        # Check final weights
        final_weights = self.get_all_weights()
        print(f"Final weights: {[f'{w:.3f}' for w in final_weights]}")
        
        changes = [final - initial for initial, final in zip(initial_weights, final_weights)]
        print(f"Weight changes: {[f'{w:+.3f}' for w in changes]}")
        
        return any(abs(change) > 0.01 for change in changes)
    
    def manual_weight_update(self, input_spikes, output_spikes, current_time):
        """Manually update weights using direct STDP calls"""
        for (pre_layer, post_layer), connection in self.network.connections.items():
            if connection.synapse_population and hasattr(connection.synapse_population, 'synapses'):
                for (pre_idx, post_idx), synapse in connection.synapse_population.synapses.items():
                    # Direct STDP updates
                    if input_spikes[pre_idx]:
                        synapse.pre_spike(current_time)
                    if output_spikes[post_idx]:
                        synapse.post_spike(current_time)
    
    def get_all_weights(self):
        """Get all synaptic weights"""
        weights = []
        for (pre_layer, post_layer), connection in self.network.connections.items():
            if connection.synapse_population and hasattr(connection.synapse_population, 'synapses'):
                for (pre_idx, post_idx), synapse in connection.synapse_population.synapses.items():
                    weights.append(synapse.weight)
        return weights
    
    def test_learning_patterns(self):
        """Test learning with different patterns"""
        print("\n🎯 PATTERN LEARNING TEST")
        print("-" * 25)
        
        patterns = [
            ([1, 0, 0, 1], 0),  # Pattern A → Output 0
            ([0, 1, 1, 0], 1),  # Pattern B → Output 1
        ]
        
        for i, (pattern, target) in enumerate(patterns):
            print(f"\nTraining pattern {i+1}: {pattern} → {target}")
            success = self.manual_stdp_training(pattern, target)
            print(f"Learning success: {'✅' if success else '❌'}")
        
        return True
    
    def test_learned_behavior(self):
        """Test if network learned the patterns"""
        print(f"\n🧪 TESTING LEARNED BEHAVIOR")
        print("-" * 30)
        
        patterns = [
            [1, 0, 0, 1],  # Should activate output 0
            [0, 1, 1, 0],  # Should activate output 1
        ]
        
        input_pop = self.network.layers["input"].neuron_population
        output_pop = self.network.layers["output"].neuron_population
        
        for i, pattern in enumerate(patterns):
            print(f"\nTesting pattern {i+1}: {pattern}")
            
            # Reset
            input_pop.reset()
            output_pop.reset()
            
            # Test input
            input_currents = [60.0 if x > 0.5 else 0.0 for x in pattern]
            
            output_activity = [0, 0]
            
            for step in range(30):
                input_spikes = input_pop.step(0.1, input_currents)
                output_spikes = output_pop.step(0.1, [0.0, 0.0])  # No external output current
                
                for j, spike in enumerate(output_spikes):
                    if spike:
                        output_activity[j] += 1
                
                self.network.step(0.1)
            
            print(f"Output activity: {output_activity}")
            
            if max(output_activity) > 0:
                winner = np.argmax(output_activity)
                print(f"Winner: Output {winner} ({'✅ Correct' if winner == i else '❌ Wrong'})")
            else:
                print("❌ No output activity")
    
    def run_learning_fix(self):
        """Run complete learning fix test"""
        print("Starting minimal learning fix test...")
        
        # Test pattern learning
        self.test_learning_patterns()
        
        # Test learned behavior
        self.test_learned_behavior()
        
        print(f"\n🎉 LEARNING FIX COMPLETE")
        return True

if __name__ == "__main__":
    fix = MinimalLearningFix()
    fix.run_learning_fix()
