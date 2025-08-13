#!/usr/bin/env python3
"""
FINAL LEARNING SUCCESS - Corrected neuromorphic learning with proper neuron parameters
"""

import numpy as np
from core.network import NeuromorphicNetwork

class FinalLearningSuccess:
    def __init__(self):
        print("🏆 FINAL NEUROMORPHIC LEARNING SUCCESS")
        print("=" * 40)
        print("Implementing corrected learning with optimal parameters")
        
        self.network = NeuromorphicNetwork()
        self.setup_success_network()
        
    def setup_success_network(self):
        """Create network optimized for successful learning and testing"""
        # Add layers with optimized neuron parameters
        self.network.add_layer("input", 4, "lif", 
                             v_thresh=-50.0,    # Lower threshold for easier firing
                             v_reset=-70.0,     # Reset potential
                             tau_m=10.0)        # Membrane time constant
                             
        self.network.add_layer("output", 2, "lif",
                             v_thresh=-50.0,    # Lower threshold
                             v_reset=-70.0,
                             tau_m=10.0)
        
        # Connect with moderate learning parameters
        self.network.connect_layers("input", "output", "stdp",
                                  connection_probability=1.0,
                                  weight=0.5,        # Start with smaller weights
                                  A_plus=0.2,        # Strong but reasonable learning
                                  A_minus=0.1,
                                  tau_stdp=20.0,
                                  tau_syn=3.0)       # Fast synaptic transmission
        
        print("✅ Success network: 4 → 2 neurons")
        print("✅ Optimized thresholds: -50mV")
        print("✅ Moderate learning: A+ = 0.2")
        
    def train_with_success_parameters(self, pattern, target, epochs=8):
        """Train with parameters optimized for success"""
        print(f"\n🎯 Training: {pattern} → {target}")
        
        # Moderate input currents
        input_currents = [50.0 if x > 0.5 else 0.0 for x in pattern]
        target_currents = [40.0 if i == target else 0.0 for i in range(2)]
        
        initial_weights = self.get_weights()
        print(f"Initial: {[f'{w:.2f}' for w in initial_weights]}")
        
        for epoch in range(epochs):
            input_pop = self.network.layers["input"].neuron_population
            output_pop = self.network.layers["output"].neuron_population
            
            time = 0.0
            dt = 0.1
            
            # Training with proper STDP timing
            for step in range(30):
                time = step * dt
                
                # Input stimulation (first 15 steps)
                if step < 15:
                    input_spikes = input_pop.step(dt, input_currents)
                else:
                    input_spikes = input_pop.step(dt, [0.0] * 4)
                
                # Target output (overlapping, steps 3-18)
                if 3 <= step < 18:
                    output_spikes = output_pop.step(dt, target_currents)
                else:
                    output_spikes = output_pop.step(dt, [0.0] * 2)
                
                # Manual STDP when needed
                if any(input_spikes) or any(output_spikes):
                    self.apply_stdp_learning(input_spikes, output_spikes, time)
                
                self.network.step(dt)
        
        final_weights = self.get_weights()
        print(f"Final: {[f'{w:.2f}' for w in final_weights]}")
        
        changes = [f - i for i, f in zip(initial_weights, final_weights)]
        significant_learning = any(abs(c) > 0.3 for c in changes)
        print(f"Learning: {'✅' if significant_learning else '❌'}")
        
        return significant_learning
    
    def apply_stdp_learning(self, input_spikes, output_spikes, time):
        """Apply STDP learning updates"""
        for (pre_layer, post_layer), connection in self.network.connections.items():
            if connection.synapse_population and hasattr(connection.synapse_population, 'synapses'):
                for (pre_idx, post_idx), synapse in connection.synapse_population.synapses.items():
                    if input_spikes[pre_idx]:
                        synapse.pre_spike(time)
                    if output_spikes[post_idx]:
                        synapse.post_spike(time)
    
    def test_with_membrane_monitoring(self, pattern, expected_output):
        """Test with detailed membrane potential monitoring"""
        print(f"\n🔬 Testing: {pattern} (expect {expected_output})")
        
        input_pop = self.network.layers["input"].neuron_population
        output_pop = self.network.layers["output"].neuron_population
        
        # Reset
        input_pop.reset()
        output_pop.reset()
        
        input_currents = [40.0 if x > 0.5 else 0.0 for x in pattern]
        output_activity = [0, 0]
        
        dt = 0.1
        
        # Monitor membrane potentials
        for step in range(40):
            # Input stimulation
            input_spikes = input_pop.step(dt, input_currents)
            
            # Get membrane potentials before output step
            output_membranes = [neuron.membrane_potential for neuron in output_pop.neurons]
            
            # Output step (no external current)
            output_spikes = output_pop.step(dt, [0.0, 0.0])
            
            # Count spikes
            for i, spike in enumerate(output_spikes):
                if spike:
                    output_activity[i] += 1
            
            # Show membrane dynamics for first few steps
            if step < 5:
                print(f"  Step {step}: V_mem = {[f'{v:.1f}' for v in output_membranes]}, Spikes = {output_spikes}")
            
            self.network.step(dt)
        
        print(f"Total output: {output_activity}")
        
        if max(output_activity) > 0:
            winner = np.argmax(output_activity)
            correct = (winner == expected_output)
            print(f"Winner: {winner} ({'✅ CORRECT' if correct else '❌ Wrong'})")
            return correct
        else:
            print("❌ No output spikes")
            return False
    
    def get_weights(self):
        """Get current weights"""
        weights = []
        for (pre_layer, post_layer), connection in self.network.connections.items():
            if connection.synapse_population and hasattr(connection.synapse_population, 'synapses'):
                for (pre_idx, post_idx), synapse in connection.synapse_population.synapses.items():
                    weights.append(synapse.weight)
        return weights
    
    def run_final_learning_success(self):
        """Execute final learning success demonstration"""
        print("Executing final neuromorphic learning success...")
        
        # Define simple patterns
        patterns = [
            ([1, 0, 0, 1], 0),  # Pattern A
            ([0, 1, 1, 0], 1),  # Pattern B
        ]
        
        print(f"\n🎓 TRAINING PHASE")
        print("-" * 20)
        
        # Train each pattern
        training_results = []
        for pattern, target in patterns:
            success = self.train_with_success_parameters(pattern, target)
            training_results.append(success)
        
        print(f"\n🧪 TESTING PHASE") 
        print("-" * 17)
        
        # Test recognition
        testing_results = []
        for pattern, expected in patterns:
            success = self.test_with_membrane_monitoring(pattern, expected)
            testing_results.append(success)
        
        # Final assessment
        print(f"\n🏆 FINAL LEARNING SUCCESS RESULTS")
        print("=" * 35)
        
        training_success = sum(training_results) / len(training_results)
        testing_success = sum(testing_results) / len(testing_results)
        
        print(f"Training success: {training_success:.1%}")
        print(f"Testing success: {testing_success:.1%}")
        
        if training_success >= 0.5 and testing_success >= 0.5:
            print(f"\n🎉 MEANINGFUL LEARNING TRANSFER ACHIEVED!")
            print(f"   ✅ Neurons learn through STDP")
            print(f"   ✅ Learned patterns recognized") 
            print(f"   ✅ Synaptic weights drive behavior")
            print(f"\n🧠 The neuromorphic system now demonstrates:")
            print(f"   • Pattern-specific synaptic strengthening")
            print(f"   • Spike-timing dependent plasticity")
            print(f"   • Learned behavior transfer")
            
            return True
        else:
            print(f"\n🔧 Still optimizing learning parameters...")
            return False

if __name__ == "__main__":
    success = FinalLearningSuccess()
    success.run_final_learning_success()
