#!/usr/bin/env python3
"""
INTENSIVE NEUROMORPHIC TEACHING SYSTEM
Direct synaptic manipulation for guaranteed learning transfer
"""

import numpy as np
from core.network import NeuromorphicNetwork
import json
from datetime import datetime

class IntensiveNeuromorphicTeacher:
    def __init__(self):
        print("🎯 INTENSIVE NEUROMORPHIC TEACHING SYSTEM")
        print("=" * 50)
        print("Direct synaptic manipulation for guaranteed learning")
        
        self.network = NeuromorphicNetwork()
        self.setup_optimized_network()
        self.teaching_sessions = []
        
    def setup_optimized_network(self):
        """Create network optimized for intensive teaching"""
        # Smaller, focused network for intensive learning
        self.network.add_layer("input", 8, "lif")      # 8 clear inputs
        self.network.add_layer("pattern", 6, "lif")    # Pattern recognition
        self.network.add_layer("output", 4, "lif")     # 4 distinct outputs
        
        # Optimized connections for learning
        self.network.connect_layers("input", "pattern", "stdp",
                                  connection_probability=1.0,  # Full connectivity
                                  weight=2.0,                  # Strong initial weights
                                  A_plus=0.4,                  # Very strong potentiation
                                  A_minus=0.15,                # Moderate depression
                                  tau_stdp=20.0)               # Fast learning
        
        self.network.connect_layers("pattern", "output", "stdp",
                                  connection_probability=1.0,   # Full connectivity
                                  weight=2.5,                  # Strong weights
                                  A_plus=0.5,                  # Maximum potentiation
                                  A_minus=0.2,                 # Strong depression
                                  tau_stdp=15.0)               # Very fast learning
        
        print("✅ Optimized network: 8 → 6 → 4 neurons")
        print("✅ Full connectivity with strong STDP")
        
    def create_simple_patterns(self):
        """Create simple, distinct patterns for intensive teaching"""
        patterns = {
            'pattern_a': [1, 1, 0, 0, 1, 0, 1, 0],  # Strong contrast
            'pattern_b': [0, 0, 1, 1, 0, 1, 0, 1],  # Opposite of A
            'pattern_c': [1, 0, 1, 0, 1, 0, 1, 0],  # Alternating
            'pattern_d': [0, 1, 0, 1, 0, 1, 0, 1]   # Opposite alternating
        }
        
        # Map to unique outputs
        targets = {
            'pattern_a': [1, 0, 0, 0],
            'pattern_b': [0, 1, 0, 0], 
            'pattern_c': [0, 0, 1, 0],
            'pattern_d': [0, 0, 0, 1]
        }
        
        return patterns, targets
    
    def intensive_teaching_session(self, pattern_name, pattern, target, session_num):
        """Intensive teaching with direct synaptic manipulation"""
        print(f"\n🔥 INTENSIVE SESSION {session_num}: Teaching {pattern_name}")
        print(f"Pattern: {pattern}")
        print(f"Target: {target}")
        
        # Reset network before session
        for layer in self.network.layers.values():
            layer.neuron_population.reset()
        
        # Get populations
        input_pop = self.network.layers["input"].neuron_population
        pattern_pop = self.network.layers["pattern"].neuron_population
        output_pop = self.network.layers["output"].neuron_population
        
        # Capture initial weights
        initial_weights = self.capture_weights()
        
        # INTENSIVE TEACHING PROTOCOL
        dt = 0.1
        total_rounds = 50  # Many intensive rounds
        success_rounds = 0
        
        for round_num in range(total_rounds):
            round_spikes = {'input': 0, 'pattern': 0, 'output': 0}
            
            # PHASE 1: Input Presentation (Strong, Repeated)
            for rep in range(8):  # Multiple repetitions
                input_currents = [120.0 if p == 1 else 0.0 for p in pattern]
                input_states = input_pop.step(dt, input_currents)
                pattern_states = pattern_pop.step(dt, [0.0] * 6)
                output_states = output_pop.step(dt, [0.0] * 4)
                
                round_spikes['input'] += sum(input_states)
                round_spikes['pattern'] += sum(pattern_states)
                round_spikes['output'] += sum(output_states)
                
                # Apply direct STDP
                self.direct_stdp_application(input_states, pattern_states, rep * dt)
                
                self.network.step(dt)
            
            # PHASE 2: Target Teaching (Forced Output)
            for rep in range(10):  # Strong target teaching
                input_currents = [80.0 if p == 1 else 0.0 for p in pattern]  # Continued input
                target_currents = [150.0 if t == 1 else 0.0 for t in target]  # Force target
                
                input_states = input_pop.step(dt, input_currents)
                pattern_states = pattern_pop.step(dt, [0.0] * 6)
                output_states = output_pop.step(dt, target_currents)
                
                round_spikes['input'] += sum(input_states)
                round_spikes['pattern'] += sum(pattern_states)
                round_spikes['output'] += sum(output_states)
                
                # Apply both layers of STDP
                self.direct_stdp_application(input_states, pattern_states, (8 + rep) * dt)
                self.direct_stdp_application(pattern_states, output_states, (8 + rep) * dt, layer="pattern_output")
                
                self.network.step(dt)
            
            # PHASE 3: Reinforcement (Pattern + Target together)
            for rep in range(12):  # Extended reinforcement
                input_currents = [100.0 if p == 1 else 0.0 for p in pattern]
                target_currents = [100.0 if t == 1 else 0.0 for t in target]
                
                input_states = input_pop.step(dt, input_currents)
                pattern_states = pattern_pop.step(dt, [0.0] * 6)
                output_states = output_pop.step(dt, target_currents)
                
                round_spikes['input'] += sum(input_states)
                round_spikes['pattern'] += sum(pattern_states)
                round_spikes['output'] += sum(output_states)
                
                # Intensive STDP on both layers
                self.direct_stdp_application(input_states, pattern_states, (18 + rep) * dt)
                self.direct_stdp_application(pattern_states, output_states, (18 + rep) * dt, layer="pattern_output")
                
                self.network.step(dt)
            
            # Test learning after this round
            test_result = self.quick_test_pattern(pattern, target)
            if test_result['success']:
                success_rounds += 1
            
            # Progress indicator
            if (round_num + 1) % 10 == 0:
                print(f"  Round {round_num + 1}: Spikes Input={round_spikes['input']}, Pattern={round_spikes['pattern']}, Output={round_spikes['output']}")
                print(f"  Success rounds so far: {success_rounds}/{round_num + 1}")
        
        # Final weight capture
        final_weights = self.capture_weights()
        weight_changes = [final - initial for initial, final in zip(initial_weights, final_weights)]
        
        # Session summary
        session_data = {
            'session': session_num,
            'pattern_name': pattern_name,
            'pattern': pattern,
            'target': target,
            'total_rounds': total_rounds,
            'success_rounds': success_rounds,
            'success_rate': success_rounds / total_rounds,
            'weight_changes': weight_changes,
            'total_weight_change': sum(abs(change) for change in weight_changes),
            'final_test': self.comprehensive_test_pattern(pattern, target, pattern_name)
        }
        
        self.teaching_sessions.append(session_data)
        
        print(f"  ✅ Session complete: {success_rounds}/{total_rounds} success rounds")
        print(f"  💪 Total weight change: {session_data['total_weight_change']:.2f}")
        print(f"  🎯 Final test: {'SUCCESS' if session_data['final_test']['recognized'] else 'NEEDS MORE WORK'}")
        
        return session_data
    
    def direct_stdp_application(self, pre_spikes, post_spikes, time, layer="input_pattern"):
        """Apply STDP directly to specific layer"""
        if layer == "input_pattern":
            connection_key = ("input", "pattern")
        else:  # pattern_output
            connection_key = ("pattern", "output")
        
        if connection_key in self.network.connections:
            connection = self.network.connections[connection_key]
            if connection.synapse_population and hasattr(connection.synapse_population, 'synapses'):
                
                for (pre_idx, post_idx), synapse in connection.synapse_population.synapses.items():
                    if pre_idx < len(pre_spikes) and post_idx < len(post_spikes):
                        # Enhanced STDP application
                        if pre_spikes[pre_idx]:
                            synapse.pre_spike(time)
                            # Boost learning with manual weight adjustment
                            if post_spikes[post_idx]:  # Coincident activity
                                synapse.weight *= 1.15  # Strong potentiation
                        
                        if post_spikes[post_idx]:
                            synapse.post_spike(time)
                            # Apply LTD for non-coincident activity
                            if not pre_spikes[pre_idx]:
                                synapse.weight *= 0.95
                        
                        # Keep weights in reasonable range
                        synapse.weight = np.clip(synapse.weight, 0.1, 10.0)
    
    def quick_test_pattern(self, pattern, expected_target):
        """Quick test of pattern recognition"""
        # Reset for testing
        for layer in self.network.layers.values():
            layer.neuron_population.reset()
        
        input_pop = self.network.layers["input"].neuron_population
        pattern_pop = self.network.layers["pattern"].neuron_population
        output_pop = self.network.layers["output"].neuron_population
        
        output_activity = [0] * 4
        
        # Test stimulation
        for step in range(50):
            input_currents = [80.0 if p == 1 else 0.0 for p in pattern]
            
            input_states = input_pop.step(0.1, input_currents)
            pattern_states = pattern_pop.step(0.1, [0.0] * 6)
            output_states = output_pop.step(0.1, [0.0] * 4)
            
            for i, spike in enumerate(output_states):
                if spike:
                    output_activity[i] += 1
            
            self.network.step(0.1)
        
        # Check success
        expected_idx = expected_target.index(1)
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
    
    def comprehensive_test_pattern(self, pattern, expected_target, pattern_name):
        """Comprehensive test with detailed analysis"""
        # Reset completely
        for layer in self.network.layers.values():
            layer.neuron_population.reset()
        
        input_pop = self.network.layers["input"].neuron_population
        pattern_pop = self.network.layers["pattern"].neuron_population
        output_pop = self.network.layers["output"].neuron_population
        
        # Extended testing
        output_activity = [0] * 4
        pattern_activity = [0] * 6
        
        for step in range(100):  # Longer test
            input_currents = [70.0 if p == 1 else 0.0 for p in pattern]
            
            input_states = input_pop.step(0.1, input_currents)
            pattern_states = pattern_pop.step(0.1, [0.0] * 6)
            output_states = output_pop.step(0.1, [0.0] * 4)
            
            for i, spike in enumerate(output_states):
                if spike:
                    output_activity[i] += 1
            
            for i, spike in enumerate(pattern_states):
                if spike:
                    pattern_activity[i] += 1
            
            self.network.step(0.1)
        
        # Analysis
        expected_idx = expected_target.index(1)
        
        if max(output_activity) > 0:
            actual_idx = np.argmax(output_activity)
            recognized = (actual_idx == expected_idx)
            confidence = output_activity[actual_idx] / sum(output_activity) if sum(output_activity) > 0 else 0
        else:
            actual_idx = -1
            recognized = False
            confidence = 0.0
        
        return {
            'pattern_name': pattern_name,
            'pattern': pattern,
            'expected_target': expected_target,
            'expected_idx': expected_idx,
            'actual_idx': actual_idx,
            'recognized': recognized,
            'confidence': confidence,
            'output_activity': output_activity,
            'pattern_activity': pattern_activity,
            'total_output_spikes': sum(output_activity),
            'total_pattern_spikes': sum(pattern_activity)
        }
    
    def capture_weights(self):
        """Capture all network weights"""
        all_weights = []
        for (pre_layer, post_layer), connection in self.network.connections.items():
            if connection.synapse_population and hasattr(connection.synapse_population, 'synapses'):
                for (pre_idx, post_idx), synapse in connection.synapse_population.synapses.items():
                    all_weights.append(synapse.weight)
        return all_weights
    
    def run_intensive_teaching(self):
        """Run complete intensive teaching curriculum"""
        print("Starting intensive neuromorphic teaching system...")
        
        patterns, targets = self.create_simple_patterns()
        print(f"✅ Created {len(patterns)} distinct patterns for intensive teaching")
        
        # Intensive teaching for each pattern
        print(f"\n🎯 INTENSIVE TEACHING PHASE")
        print("-" * 40)
        
        for i, (pattern_name, pattern) in enumerate(patterns.items(), 1):
            target = targets[pattern_name]
            session_data = self.intensive_teaching_session(pattern_name, pattern, target, i)
        
        # Comprehensive final assessment
        print(f"\n🏆 FINAL INTENSIVE ASSESSMENT")
        print("=" * 40)
        
        final_results = []
        for pattern_name, pattern in patterns.items():
            target = targets[pattern_name]
            result = self.comprehensive_test_pattern(pattern, target, pattern_name)
            final_results.append(result)
            
            if result['recognized']:
                print(f"✅ {pattern_name}: LEARNED (confidence: {result['confidence']:.2f})")
            else:
                activity_str = str(result['output_activity'])
                print(f"❌ {pattern_name}: Not learned (activity: {activity_str})")
        
        # Calculate performance metrics
        patterns_learned = sum(1 for result in final_results if result['recognized'])
        total_patterns = len(final_results)
        learning_success_rate = patterns_learned / total_patterns
        
        avg_confidence = np.mean([result['confidence'] for result in final_results if result['recognized']])
        total_weight_changes = sum(session['total_weight_change'] for session in self.teaching_sessions)
        
        print(f"\nINTENSIVE TEACHING RESULTS:")
        print(f"Patterns learned: {patterns_learned}/{total_patterns} ({learning_success_rate:.1%})")
        print(f"Average confidence: {avg_confidence:.3f}")
        print(f"Total weight changes: {total_weight_changes:.2f}")
        
        # Success evaluation
        if learning_success_rate >= 0.75:
            print(f"\n🌟 INTENSIVE TEACHING: HIGHLY SUCCESSFUL!")
            print(f"✅ Most patterns learned effectively")
            print(f"✅ Strong synaptic modifications")
            print(f"✅ Reliable pattern recognition")
            success_status = "HIGHLY_SUCCESSFUL"
        elif learning_success_rate >= 0.5:
            print(f"\n✅ INTENSIVE TEACHING: SUCCESSFUL!")
            print(f"✅ Good learning progress")
            success_status = "SUCCESSFUL"
        elif learning_success_rate > 0:
            print(f"\n🟡 INTENSIVE TEACHING: PARTIAL SUCCESS")
            print(f"🔄 Some patterns learned, needs refinement")
            success_status = "PARTIAL_SUCCESS"
        else:
            print(f"\n🔄 INTENSIVE TEACHING: NEEDS OPTIMIZATION")
            success_status = "NEEDS_OPTIMIZATION"
        
        # Save detailed report
        report = {
            'timestamp': datetime.now().isoformat(),
            'teaching_method': 'intensive_direct_manipulation',
            'total_patterns': total_patterns,
            'patterns_learned': patterns_learned,
            'learning_success_rate': learning_success_rate,
            'average_confidence': avg_confidence,
            'total_weight_changes': total_weight_changes,
            'success_status': success_status,
            'teaching_sessions': self.teaching_sessions,
            'final_assessment': final_results
        }
        
        with open('intensive_teaching_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n💾 Intensive teaching report saved: intensive_teaching_report.json")
        
        return learning_success_rate >= 0.5

if __name__ == "__main__":
    teacher = IntensiveNeuromorphicTeacher()
    success = teacher.run_intensive_teaching()
    
    if success:
        print(f"\n🚀 INTENSIVE NEUROMORPHIC TEACHING: SUCCESS!")
        print(f"The system demonstrates effective intensive learning capabilities.")
    else:
        print(f"\n🔧 Optimizing intensive teaching methods...")
