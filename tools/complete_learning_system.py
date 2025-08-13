#!/usr/bin/env python3
"""
COMPLETE LEARNING SYSTEM
Full implementation with learning AND proper testing
"""

import numpy as np
from core.network import NeuromorphicNetwork

class CompleteLearningSystem:
    def __init__(self):
        print("🎯 COMPLETE NEUROMORPHIC LEARNING SYSTEM")
        print("=" * 42)
        print("Implementing full learning with transfer verification")
        
        self.network = NeuromorphicNetwork()
        self.setup_learning_network()
        
    def setup_learning_network(self):
        """Create network optimized for learning and testing"""
        self.network.add_layer("input", 4, "lif")
        self.network.add_layer("output", 2, "lif")
        
        self.network.connect_layers("input", "output", "stdp",
                                  connection_probability=1.0,
                                  weight=1.0,
                                  A_plus=0.3,   # Strong learning
                                  A_minus=0.15,
                                  tau_stdp=20.0,
                                  tau_syn=5.0)  # Fast synaptic transmission
        
        print("✅ Learning network: 4 → 2 neurons (full connectivity)")
        print("✅ Strong STDP: A+ = 0.3, A- = 0.15")
        
    def train_pattern_with_stdp(self, input_pattern, target_neuron, epochs=10):
        """Train single pattern with proper STDP timing"""
        print(f"\n🎓 Training: {input_pattern} → {target_neuron}")
        
        input_currents = [80.0 if x > 0.5 else 0.0 for x in input_pattern]
        target_currents = [50.0 if i == target_neuron else 0.0 for i in range(2)]
        
        initial_weights = self.get_weights()
        print(f"Initial weights: {[f'{w:.2f}' for w in initial_weights]}")
        
        for epoch in range(epochs):
            # Reset populations
            input_pop = self.network.layers["input"].neuron_population
            output_pop = self.network.layers["output"].neuron_population
            
            time = 0.0
            dt = 0.1
            
            # Training episode with proper timing
            for step in range(40):
                time = step * dt
                
                # Input phase (first 20 steps)
                if step < 20:
                    input_spikes = input_pop.step(dt, input_currents)
                else:
                    input_spikes = input_pop.step(dt, [0.0] * 4)
                
                # Target phase (overlapping, steps 5-25)
                if 5 <= step < 25:
                    output_spikes = output_pop.step(dt, target_currents)
                else:
                    output_spikes = output_pop.step(dt, [0.0] * 2)
                
                # Manual STDP update when spikes occur
                if any(input_spikes) or any(output_spikes):
                    self.apply_stdp_update(input_spikes, output_spikes, time)
                
                # Network step
                self.network.step(dt)
        
        final_weights = self.get_weights()
        print(f"Final weights: {[f'{w:.2f}' for w in final_weights]}")
        
        changes = [final - initial for initial, final in zip(initial_weights, final_weights)]
        print(f"Changes: {[f'{w:+.2f}' for w in changes]}")
        
        return max(abs(c) for c in changes) > 0.5  # Significant change threshold
    
    def apply_stdp_update(self, input_spikes, output_spikes, current_time):
        """Apply STDP updates directly to synapses"""
        for (pre_layer, post_layer), connection in self.network.connections.items():
            if connection.synapse_population and hasattr(connection.synapse_population, 'synapses'):
                for (pre_idx, post_idx), synapse in connection.synapse_population.synapses.items():
                    if input_spikes[pre_idx]:
                        synapse.pre_spike(current_time)
                    if output_spikes[post_idx]:
                        synapse.post_spike(current_time)
    
    def test_pattern_recognition(self, input_pattern, expected_output):
        """Test pattern recognition with synaptic transmission"""
        print(f"\n🧪 Testing: {input_pattern} (expect output {expected_output})")
        
        # Reset network
        input_pop = self.network.layers["input"].neuron_population
        output_pop = self.network.layers["output"].neuron_population
        input_pop.reset()
        output_pop.reset()
        
        # Test currents
        input_currents = [60.0 if x > 0.5 else 0.0 for x in input_pattern]
        
        # Track activity
        output_activity = [0, 0]
        synaptic_currents_received = [0.0, 0.0]
        
        dt = 0.1
        
        for step in range(50):
            # Input stimulation
            input_spikes = input_pop.step(dt, input_currents)
            
            # Calculate synaptic currents manually
            current_time = step * dt
            for (pre_layer, post_layer), connection in self.network.connections.items():
                if connection.synapse_population:
                    # Get synaptic currents from spikes
                    currents = connection.get_synaptic_currents(input_spikes, current_time)
                    for i, current in enumerate(currents):
                        synaptic_currents_received[i] += current
            
            # Output with NO external current (pure synaptic transmission)
            output_spikes = output_pop.step(dt, [0.0, 0.0])
            
            # Count output spikes
            for i, spike in enumerate(output_spikes):
                if spike:
                    output_activity[i] += 1
            
            # Network step
            self.network.step(dt)
            
            # Debug: Show synaptic transmission
            if step < 10 and any(input_spikes):
                currents = connection.get_synaptic_currents(input_spikes, current_time)
                print(f"  Step {step}: Input={input_spikes}, Synaptic currents={[f'{c:.2f}' for c in currents]}")
        
        print(f"Output activity: {output_activity}")
        print(f"Total synaptic current received: {[f'{c:.2f}' for c in synaptic_currents_received]}")
        
        if max(output_activity) > 0:
            winner = np.argmax(output_activity)
            correct = winner == expected_output
            print(f"Winner: Output {winner} ({'✅ CORRECT' if correct else '❌ Wrong'})")
            return correct
        else:
            print("❌ NO OUTPUT - Learning transfer failed")
            return False
    
    def get_weights(self):
        """Get all synaptic weights"""
        weights = []
        for (pre_layer, post_layer), connection in self.network.connections.items():
            if connection.synapse_population and hasattr(connection.synapse_population, 'synapses'):
                for (pre_idx, post_idx), synapse in connection.synapse_population.synapses.items():
                    weights.append(synapse.weight)
        return weights
    
    def run_complete_learning_system(self):
        """Run complete learning and testing system"""
        print("Starting complete neuromorphic learning system...")
        
        # Define patterns
        patterns = [
            ([1, 0, 0, 1], 0),  # Corners → Output 0
            ([0, 1, 1, 0], 1),  # Center → Output 1
        ]
        
        print(f"\n🎯 TRAINING PHASE")
        print("-" * 20)
        
        learning_success = []
        for pattern, target in patterns:
            success = self.train_pattern_with_stdp(pattern, target, epochs=5)
            learning_success.append(success)
        
        print(f"\n🧪 TESTING PHASE")
        print("-" * 17)
        
        recognition_success = []
        for pattern, expected in patterns:
            success = self.test_pattern_recognition(pattern, expected)
            recognition_success.append(success)
        
        # Final results
        print(f"\n📊 COMPLETE LEARNING SYSTEM RESULTS")
        print("=" * 37)
        
        learning_rate = sum(learning_success) / len(learning_success)
        recognition_rate = sum(recognition_success) / len(recognition_success)
        
        print(f"Learning Success: {sum(learning_success)}/{len(learning_success)} ({learning_rate:.1%})")
        print(f"Recognition Success: {sum(recognition_success)}/{len(recognition_success)} ({recognition_rate:.1%})")
        
        if learning_rate > 0.5 and recognition_rate > 0.5:
            print(f"\n🎉 MEANINGFUL LEARNING TRANSFER ACHIEVED!")
            print(f"   ✅ Synaptic plasticity working")
            print(f"   ✅ Pattern recognition working")
            print(f"   ✅ Learning transfer verified")
        elif learning_rate > 0.5:
            print(f"\n🟡 PARTIAL SUCCESS - Learning works, transfer needs improvement")
        else:
            print(f"\n❌ LEARNING SYSTEM NEEDS MORE WORK")
        
        return learning_rate > 0.5 and recognition_rate > 0.5

if __name__ == "__main__":
    system = CompleteLearningSystem()
    system.run_complete_learning_system()
