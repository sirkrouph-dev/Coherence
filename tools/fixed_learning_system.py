#!/usr/bin/env python3
"""
FIXED NEUROMORPHIC LEARNING SYSTEM
Bypass network integration issues with manual synaptic coordination
"""

import numpy as np
from core.network import NeuromorphicNetwork
import json
from datetime import datetime

class FixedNeuromorphicLearning:
    def __init__(self):
        print("🔧 FIXED NEUROMORPHIC LEARNING SYSTEM")
        print("=" * 50)
        print("Manual synaptic coordination for reliable learning")
        
        self.network = NeuromorphicNetwork()
        self.setup_learning_network()
        self.learning_results = []
        
    def setup_learning_network(self):
        """Create network with manual coordination"""
        # Simple but effective architecture
        self.network.add_layer("input", 4, "lif")      # 4 clear inputs
        self.network.add_layer("output", 3, "lif")     # 3 distinct outputs
        
        # Strong learning connection
        self.network.connect_layers("input", "output", "stdp",
                                  connection_probability=1.0,
                                  weight=2.0,
                                  A_plus=0.3,
                                  A_minus=0.12,
                                  tau_stdp=20.0)
        
        print("✅ Fixed network: 4 → 3 neurons with manual coordination")
        
    def create_learning_patterns(self):
        """Create simple, distinct learning patterns"""
        patterns = {
            'pattern_1': [1, 1, 0, 0],  # High-Low
            'pattern_2': [0, 0, 1, 1],  # Low-High
            'pattern_3': [1, 0, 1, 0]   # Alternating
        }
        
        targets = {
            'pattern_1': [1, 0, 0],
            'pattern_2': [0, 1, 0],
            'pattern_3': [0, 0, 1]
        }
        
        return patterns, targets
    
    def manual_learning_session(self, pattern_name, pattern, target, session_num):
        """Learning session with manual synaptic coordination"""
        print(f"\n🎯 LEARNING SESSION {session_num}: {pattern_name}")
        print(f"Input pattern: {pattern}")
        print(f"Target output: {target}")
        
        # Reset network
        for layer in self.network.layers.values():
            layer.neuron_population.reset()
        
        # Get populations and connections
        input_pop = self.network.layers["input"].neuron_population
        output_pop = self.network.layers["output"].neuron_population
        connection = self.network.connections[("input", "output")]
        
        # Capture initial weights
        initial_weights = self.capture_synaptic_weights(connection)
        
        # Manual learning protocol
        learning_rounds = 30
        total_spikes = {'input': 0, 'output': 0}
        successful_stdp_applications = 0
        
        dt = 0.1
        
        for round_num in range(learning_rounds):
            round_spikes = {'input': 0, 'output': 0}
            
            # PHASE 1: Input stimulation (20 steps)
            for step in range(20):
                time = round_num * 30 * dt + step * dt
                
                # Strong input currents
                input_currents = [150.0 if p == 1 else 5.0 for p in pattern]
                input_states = input_pop.step(dt, input_currents)
                
                # Calculate synaptic currents for output layer
                synaptic_currents = self.calculate_synaptic_currents(connection, input_states)
                output_states = output_pop.step(dt, synaptic_currents)
                
                # Track spikes
                round_spikes['input'] += sum(input_states)
                round_spikes['output'] += sum(output_states)
                
                # Apply STDP when spikes occur
                if any(input_states) or any(output_states):
                    self.apply_manual_stdp(connection, input_states, output_states, time)
                    successful_stdp_applications += 1
                
                # Mini network step (for any internal updates)
                self.network.step(dt)
            
            # PHASE 2: Target teaching (20 steps) 
            for step in range(20):
                time = round_num * 30 * dt + (20 + step) * dt
                
                # Continue input + force target output
                input_currents = [100.0 if p == 1 else 0.0 for p in pattern]
                target_currents = [120.0 if t == 1 else 0.0 for t in target]
                
                input_states = input_pop.step(dt, input_currents)
                output_states = output_pop.step(dt, target_currents)
                
                round_spikes['input'] += sum(input_states)
                round_spikes['output'] += sum(output_states)
                
                # Strong STDP during teaching
                if any(input_states) or any(output_states):
                    self.apply_manual_stdp(connection, input_states, output_states, time, strength=1.2)
                    successful_stdp_applications += 1
                
                self.network.step(dt)
            
            total_spikes['input'] += round_spikes['input']
            total_spikes['output'] += round_spikes['output']
            
            # Progress check every 10 rounds
            if (round_num + 1) % 10 == 0:
                test_result = self.quick_test_learned_pattern(pattern, target)
                print(f"  Round {round_num + 1}: Input={round_spikes['input']} spikes, Output={round_spikes['output']} spikes, Test={'✅' if test_result['success'] else '❌'}")
        
        # Final weight capture
        final_weights = self.capture_synaptic_weights(connection)
        weight_changes = [final - initial for initial, final in zip(initial_weights, final_weights)]
        total_weight_change = sum(abs(change) for change in weight_changes)
        
        # Comprehensive final test
        final_test = self.comprehensive_test_pattern(pattern, target, pattern_name)
        
        # Session summary
        session_data = {
            'session': session_num,
            'pattern_name': pattern_name,
            'pattern': pattern,
            'target': target,
            'learning_rounds': learning_rounds,
            'total_spikes': total_spikes,
            'stdp_applications': successful_stdp_applications,
            'weight_changes': weight_changes,
            'total_weight_change': total_weight_change,
            'final_test': final_test
        }
        
        self.learning_results.append(session_data)
        
        print(f"  📊 Total spikes: Input={total_spikes['input']}, Output={total_spikes['output']}")
        print(f"  ⚡ STDP applications: {successful_stdp_applications}")
        print(f"  💪 Weight change: {total_weight_change:.2f}")
        print(f"  🎯 Final test: {'✅ LEARNED' if final_test['recognized'] else '❌ Not learned'}")
        
        if final_test['recognized']:
            print(f"    Confidence: {final_test['confidence']:.2f}")
        
        return session_data
    
    def calculate_synaptic_currents(self, connection, pre_spikes):
        """Calculate synaptic currents from pre-synaptic spikes"""
        post_layer_size = len(self.network.layers["output"].neuron_population.neurons)
        synaptic_currents = [0.0] * post_layer_size
        
        if connection.synapse_population and hasattr(connection.synapse_population, 'synapses'):
            for (pre_idx, post_idx), synapse in connection.synapse_population.synapses.items():
                if pre_idx < len(pre_spikes) and pre_spikes[pre_idx]:
                    # Convert synaptic weight to current (amplified)
                    current = synapse.weight * 15.0  # Amplification factor
                    synaptic_currents[post_idx] += current
        
        return synaptic_currents
    
    def apply_manual_stdp(self, connection, pre_spikes, post_spikes, time, strength=1.0):
        """Apply STDP manually with coordination"""
        if connection.synapse_population and hasattr(connection.synapse_population, 'synapses'):
            for (pre_idx, post_idx), synapse in connection.synapse_population.synapses.items():
                if pre_idx < len(pre_spikes) and post_idx < len(post_spikes):
                    
                    # Enhanced STDP application
                    if pre_spikes[pre_idx]:
                        synapse.pre_spike(time)
                        
                        # Potentiation when both pre and post spike
                        if post_spikes[post_idx]:
                            weight_increase = synapse.A_plus * 0.2 * strength
                            synapse.weight += weight_increase
                    
                    if post_spikes[post_idx]:
                        synapse.post_spike(time)
                        
                        # Depression when only post spikes
                        if not pre_spikes[pre_idx]:
                            weight_decrease = synapse.A_minus * 0.1 * strength
                            synapse.weight -= weight_decrease
                    
                    # Keep weights in reasonable bounds
                    synapse.weight = np.clip(synapse.weight, 0.1, 8.0)
    
    def quick_test_learned_pattern(self, pattern, target):
        """Quick test of learning progress"""
        # Reset for testing
        for layer in self.network.layers.values():
            layer.neuron_population.reset()
        
        input_pop = self.network.layers["input"].neuron_population
        output_pop = self.network.layers["output"].neuron_population
        connection = self.network.connections[("input", "output")]
        
        output_activity = [0] * 3
        
        # Test with learned weights
        for step in range(40):
            input_currents = [80.0 if p == 1 else 0.0 for p in pattern]
            input_states = input_pop.step(0.1, input_currents)
            
            # Use learned synaptic weights
            synaptic_currents = self.calculate_synaptic_currents(connection, input_states)
            output_states = output_pop.step(0.1, synaptic_currents)
            
            for i, spike in enumerate(output_states):
                if spike:
                    output_activity[i] += 1
            
            self.network.step(0.1)
        
        # Check if correct output is most active
        expected_idx = target.index(1)
        if max(output_activity) > 0:
            actual_idx = np.argmax(output_activity)
            success = (actual_idx == expected_idx)
        else:
            success = False
            actual_idx = -1
        
        return {
            'success': success,
            'expected': expected_idx,
            'actual': actual_idx,
            'activity': output_activity
        }
    
    def comprehensive_test_pattern(self, pattern, target, pattern_name):
        """Comprehensive test with detailed metrics"""
        # Complete reset
        for layer in self.network.layers.values():
            layer.neuron_population.reset()
        
        input_pop = self.network.layers["input"].neuron_population
        output_pop = self.network.layers["output"].neuron_population
        connection = self.network.connections[("input", "output")]
        
        output_activity = [0] * 3
        
        # Extended test (80 steps)
        for step in range(80):
            input_currents = [70.0 if p == 1 else 0.0 for p in pattern]
            input_states = input_pop.step(0.1, input_currents)
            
            synaptic_currents = self.calculate_synaptic_currents(connection, input_states)
            output_states = output_pop.step(0.1, synaptic_currents)
            
            for i, spike in enumerate(output_states):
                if spike:
                    output_activity[i] += 1
            
            self.network.step(0.1)
        
        # Analysis
        expected_idx = target.index(1)
        total_output = sum(output_activity)
        
        if total_output > 0:
            actual_idx = np.argmax(output_activity)
            recognized = (actual_idx == expected_idx)
            confidence = output_activity[actual_idx] / total_output
        else:
            actual_idx = -1
            recognized = False
            confidence = 0.0
        
        return {
            'pattern_name': pattern_name,
            'expected_idx': expected_idx,
            'actual_idx': actual_idx,
            'recognized': recognized,
            'confidence': confidence,
            'output_activity': output_activity,
            'total_spikes': total_output
        }
    
    def capture_synaptic_weights(self, connection):
        """Capture all synaptic weights"""
        weights = []
        if connection.synapse_population and hasattr(connection.synapse_population, 'synapses'):
            for (pre_idx, post_idx), synapse in connection.synapse_population.synapses.items():
                weights.append(synapse.weight)
        return weights
    
    def run_fixed_learning_curriculum(self):
        """Run complete fixed learning curriculum"""
        print("Starting fixed neuromorphic learning curriculum...")
        
        patterns, targets = self.create_learning_patterns()
        print(f"✅ Created {len(patterns)} learning patterns")
        
        # Learning phase
        print(f"\n🎓 LEARNING PHASE")
        print("-" * 25)
        
        for i, (pattern_name, pattern) in enumerate(patterns.items(), 1):
            target = targets[pattern_name]
            session_data = self.manual_learning_session(pattern_name, pattern, target, i)
        
        # Final assessment
        print(f"\n🏆 FINAL LEARNING ASSESSMENT")
        print("=" * 35)
        
        learned_patterns = 0
        total_patterns = len(patterns)
        confidence_scores = []
        
        for pattern_name, pattern in patterns.items():
            target = targets[pattern_name]
            test_result = self.comprehensive_test_pattern(pattern, target, pattern_name)
            
            if test_result['recognized']:
                learned_patterns += 1
                confidence_scores.append(test_result['confidence'])
                print(f"✅ {pattern_name}: LEARNED (confidence {test_result['confidence']:.2f})")
            else:
                print(f"❌ {pattern_name}: Not learned (output: {test_result['output_activity']})")
        
        # Performance metrics
        learning_rate = learned_patterns / total_patterns
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        total_weight_changes = sum(result['total_weight_change'] for result in self.learning_results)
        total_stdp_applications = sum(result['stdp_applications'] for result in self.learning_results)
        
        print(f"\n📊 LEARNING PERFORMANCE")
        print("-" * 25)
        print(f"Patterns learned: {learned_patterns}/{total_patterns} ({learning_rate:.1%})")
        print(f"Average confidence: {avg_confidence:.3f}")
        print(f"Total weight changes: {total_weight_changes:.2f}")
        print(f"STDP applications: {total_stdp_applications}")
        
        # Success evaluation
        if learning_rate >= 0.8 and avg_confidence >= 0.6:
            print(f"\n🌟 FIXED LEARNING: HIGHLY SUCCESSFUL!")
            print(f"✅ Excellent pattern learning")
            print(f"✅ Strong recognition confidence")
            print(f"✅ Effective synaptic plasticity")
            success_status = "HIGHLY_SUCCESSFUL"
        elif learning_rate >= 0.6:
            print(f"\n✅ FIXED LEARNING: SUCCESSFUL!")
            print(f"✅ Good pattern learning")
            success_status = "SUCCESSFUL"
        elif learning_rate > 0:
            print(f"\n🟡 FIXED LEARNING: PARTIAL SUCCESS")
            print(f"🔄 Some learning achieved")
            success_status = "PARTIAL_SUCCESS"
        else:
            print(f"\n🔄 FIXED LEARNING: NEEDS REFINEMENT")
            success_status = "NEEDS_REFINEMENT"
        
        # Save comprehensive report
        report = {
            'timestamp': datetime.now().isoformat(),
            'method': 'fixed_manual_coordination',
            'total_patterns': total_patterns,
            'patterns_learned': learned_patterns,
            'learning_rate': learning_rate,
            'average_confidence': avg_confidence,
            'total_weight_changes': total_weight_changes,
            'total_stdp_applications': total_stdp_applications,
            'success_status': success_status,
            'learning_sessions': self.learning_results
        }
        
        with open('fixed_learning_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n💾 Fixed learning report saved: fixed_learning_report.json")
        
        return learning_rate >= 0.5

if __name__ == "__main__":
    fixed_learning = FixedNeuromorphicLearning()
    success = fixed_learning.run_fixed_learning_curriculum()
    
    if success:
        print(f"\n🚀 FIXED NEUROMORPHIC LEARNING: SUCCESS!")
        print(f"Manual synaptic coordination enables reliable learning.")
    else:
        print(f"\n🔧 Continuing to refine learning methods...")
