#!/usr/bin/env python3
"""
ADVANCED NEUROMORPHIC LEARNING CURRICULUM
Progressive teaching system for sophisticated pattern recognition
"""

import numpy as np
from core.network import NeuromorphicNetwork
import json
from datetime import datetime

class AdvancedLearningCurriculum:
    def __init__(self):
        print("🎓 ADVANCED NEUROMORPHIC LEARNING CURRICULUM")
        print("=" * 50)
        print("Progressive teaching for sophisticated pattern recognition")
        
        self.network = NeuromorphicNetwork()
        self.setup_advanced_network()
        self.learning_history = []
        
    def setup_advanced_network(self):
        """Create larger network for complex learning"""
        # Expanded architecture for complex patterns
        self.network.add_layer("input", 16, "lif")      # 4x4 input grid
        self.network.add_layer("hidden1", 12, "lif")    # First hidden layer
        self.network.add_layer("hidden2", 8, "lif")     # Second hidden layer  
        self.network.add_layer("output", 6, "lif")      # 6 pattern categories
        
        # Multi-layer learning with progressive complexity
        self.network.connect_layers("input", "hidden1", "stdp",
                                  connection_probability=0.8,
                                  weight=0.8,
                                  A_plus=0.15,
                                  A_minus=0.075,
                                  tau_stdp=25.0)
                                  
        self.network.connect_layers("hidden1", "hidden2", "stdp",
                                  connection_probability=0.9,
                                  weight=1.0,
                                  A_plus=0.2,
                                  A_minus=0.1,
                                  tau_stdp=20.0)
                                  
        self.network.connect_layers("hidden2", "output", "stdp",
                                  connection_probability=1.0,
                                  weight=1.2,
                                  A_plus=0.25,
                                  A_minus=0.125,
                                  tau_stdp=15.0)
        
        print("✅ Advanced network: 16 → 12 → 8 → 6 neurons")
        print("✅ Multi-layer STDP learning")
        
    def create_advanced_patterns(self):
        """Create sophisticated pattern curriculum"""
        # Level 1: Basic shapes
        basic_patterns = {
            'vertical_line': np.array([
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0]
            ], dtype=float),
            
            'horizontal_line': np.array([
                [1, 1, 1, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]
            ], dtype=float),
            
            'diagonal': np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], dtype=float)
        }
        
        # Level 2: Complex shapes
        complex_patterns = {
            'cross': np.array([
                [0, 1, 1, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [0, 1, 1, 0]
            ], dtype=float),
            
            'corner_l': np.array([
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 1, 1, 1]
            ], dtype=float),
            
            'checkerboard': np.array([
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 1]
            ], dtype=float)
        }
        
        # Combine all patterns
        all_patterns = {**basic_patterns, **complex_patterns}
        
        # Create target mapping
        targets = {}
        for i, pattern_name in enumerate(all_patterns.keys()):
            target = [0] * 6
            target[i] = 1
            targets[pattern_name] = target
        
        return all_patterns, targets
    
    def progressive_teaching(self, patterns, targets):
        """Teach patterns progressively with increasing complexity"""
        print(f"\n🎯 PROGRESSIVE TEACHING CURRICULUM")
        print("-" * 40)
        
        # Curriculum phases
        phases = [
            {
                'name': 'Basic Shapes',
                'patterns': ['vertical_line', 'horizontal_line', 'diagonal'],
                'epochs': 15,
                'intensity': 'moderate'
            },
            {
                'name': 'Complex Shapes', 
                'patterns': ['cross', 'corner_l', 'checkerboard'],
                'epochs': 20,
                'intensity': 'strong'
            },
            {
                'name': 'Mixed Review',
                'patterns': list(patterns.keys()),
                'epochs': 25,
                'intensity': 'adaptive'
            }
        ]
        
        total_learning_progress = []
        
        for phase_num, phase in enumerate(phases, 1):
            print(f"\n📚 PHASE {phase_num}: {phase['name']}")
            print(f"Teaching {len(phase['patterns'])} patterns for {phase['epochs']} epochs")
            print("-" * 50)
            
            phase_progress = self.teach_phase(
                patterns, targets, 
                phase['patterns'], 
                phase['epochs'],
                phase['intensity']
            )
            
            total_learning_progress.extend(phase_progress)
            
            # Test phase understanding
            self.test_phase_understanding(patterns, targets, phase['patterns'], phase_num)
        
        return total_learning_progress
    
    def teach_phase(self, patterns, targets, pattern_names, epochs, intensity):
        """Teach specific phase with adaptive parameters"""
        phase_progress = []
        
        # Adaptive training parameters based on intensity
        if intensity == 'moderate':
            input_strength = 70.0
            target_strength = 50.0
            steps = 40
        elif intensity == 'strong':
            input_strength = 90.0
            target_strength = 70.0
            steps = 60
        else:  # adaptive
            input_strength = 80.0
            target_strength = 60.0
            steps = 50
        
        for epoch in range(epochs):
            epoch_weights_before = self.capture_network_weights()
            epoch_results = {}
            
            # Train each pattern in the phase
            for pattern_name in pattern_names:
                pattern = patterns[pattern_name]
                target = targets[pattern_name]
                
                # Multi-repetition training for better learning
                for rep in range(3):
                    learning_result = self.advanced_pattern_training(
                        pattern, target, pattern_name,
                        input_strength, target_strength, steps
                    )
                    
                    if pattern_name not in epoch_results:
                        epoch_results[pattern_name] = learning_result
                    else:
                        # Accumulate learning metrics
                        for key in ['input_spikes', 'hidden1_spikes', 'hidden2_spikes', 'output_spikes']:
                            epoch_results[pattern_name][key] += learning_result[key]
            
            epoch_weights_after = self.capture_network_weights()
            
            # Calculate learning progress
            weight_changes = self.calculate_weight_changes(epoch_weights_before, epoch_weights_after)
            learning_magnitude = sum(abs(change) for change in weight_changes)
            
            phase_progress.append({
                'epoch': epoch + 1,
                'patterns': pattern_names,
                'learning_magnitude': learning_magnitude,
                'results': epoch_results
            })
            
            # Progress reporting
            if (epoch + 1) % 5 == 0:
                avg_learning = np.mean([entry['learning_magnitude'] for entry in phase_progress[-5:]])
                print(f"  Epoch {epoch + 1}/{epochs}: Learning magnitude = {avg_learning:.3f}")
                
                # Show pattern-specific progress
                for pattern_name in pattern_names:
                    spikes = epoch_results[pattern_name]['output_spikes']
                    print(f"    {pattern_name}: {spikes} output spikes")
        
        return phase_progress
    
    def advanced_pattern_training(self, pattern, target, pattern_name, 
                                input_strength, target_strength, steps):
        """Advanced training with multi-layer coordination"""
        # Flatten 4x4 pattern to 16 inputs
        flat_pattern = pattern.flatten()
        input_currents = [input_strength if pixel > 0.5 else 0.0 for pixel in flat_pattern]
        target_currents = [target_strength if target[i] == 1 else 0.0 for i in range(6)]
        
        # Get all populations
        input_pop = self.network.layers["input"].neuron_population
        hidden1_pop = self.network.layers["hidden1"].neuron_population
        hidden2_pop = self.network.layers["hidden2"].neuron_population
        output_pop = self.network.layers["output"].neuron_population
        
        dt = 0.1
        time = 0.0
        
        # Spike counters
        spike_counts = {
            'input_spikes': 0,
            'hidden1_spikes': 0, 
            'hidden2_spikes': 0,
            'output_spikes': 0
        }
        
        # Advanced training sequence with proper timing
        for step in range(steps):
            time = step * dt
            
            # Input phase (first 60% of steps)
            if step < int(steps * 0.6):
                input_states = input_pop.step(dt, input_currents)
                spike_counts['input_spikes'] += sum(input_states)
            else:
                input_states = input_pop.step(dt, [0.0] * 16)
            
            # Hidden layer processing
            hidden1_states = hidden1_pop.step(dt, [0.0] * 12)
            hidden2_states = hidden2_pop.step(dt, [0.0] * 8)
            
            spike_counts['hidden1_spikes'] += sum(hidden1_states)
            spike_counts['hidden2_spikes'] += sum(hidden2_states)
            
            # Target guidance (overlapping with input, middle 60% of steps)
            if int(steps * 0.2) <= step < int(steps * 0.8):
                output_states = output_pop.step(dt, target_currents)
                spike_counts['output_spikes'] += sum(output_states)
            else:
                output_states = output_pop.step(dt, [0.0] * 6)
            
            # Apply multi-layer STDP learning
            self.apply_multilayer_stdp(
                input_states, hidden1_states, hidden2_states, output_states, time
            )
            
            # Network coordination step
            self.network.step(dt)
        
        spike_counts['pattern_name'] = pattern_name
        spike_counts['target_neuron'] = target.index(1) if 1 in target else -1
        
        return spike_counts
    
    def apply_multilayer_stdp(self, input_spikes, hidden1_spikes, hidden2_spikes, output_spikes, time):
        """Apply STDP across all network layers"""
        # Input → Hidden1 STDP
        for (pre_layer, post_layer), connection in self.network.connections.items():
            if pre_layer == "input" and post_layer == "hidden1":
                self.apply_layer_stdp(connection, input_spikes, hidden1_spikes, time)
            elif pre_layer == "hidden1" and post_layer == "hidden2":
                self.apply_layer_stdp(connection, hidden1_spikes, hidden2_spikes, time)
            elif pre_layer == "hidden2" and post_layer == "output":
                self.apply_layer_stdp(connection, hidden2_spikes, output_spikes, time)
    
    def apply_layer_stdp(self, connection, pre_spikes, post_spikes, time):
        """Apply STDP to specific layer connection"""
        if connection.synapse_population and hasattr(connection.synapse_population, 'synapses'):
            for (pre_idx, post_idx), synapse in connection.synapse_population.synapses.items():
                if pre_idx < len(pre_spikes) and post_idx < len(post_spikes):
                    if pre_spikes[pre_idx]:
                        synapse.pre_spike(time)
                    if post_spikes[post_idx]:
                        synapse.post_spike(time)
    
    def test_phase_understanding(self, patterns, targets, pattern_names, phase_num):
        """Test understanding of patterns taught in this phase"""
        print(f"\n🧪 PHASE {phase_num} UNDERSTANDING TEST")
        print("-" * 35)
        
        correct_recognitions = 0
        total_patterns = len(pattern_names)
        
        for pattern_name in pattern_names:
            pattern = patterns[pattern_name]
            expected_output = targets[pattern_name].index(1)
            
            recognition_result = self.test_advanced_pattern(pattern, expected_output, pattern_name)
            
            if recognition_result['recognized']:
                correct_recognitions += 1
                print(f"  ✅ {pattern_name}: Correctly recognized (neuron {recognition_result['winner']})")
            else:
                activity = recognition_result['output_activity']
                if max(activity) > 0:
                    winner = np.argmax(activity)
                    print(f"  🟡 {pattern_name}: Confused (neuron {winner}, expected {expected_output})")
                else:
                    print(f"  ❌ {pattern_name}: No response")
        
        accuracy = correct_recognitions / total_patterns
        print(f"\nPhase {phase_num} Accuracy: {correct_recognitions}/{total_patterns} ({accuracy:.1%})")
        
        if accuracy >= 0.8:
            print("🎉 Excellent phase mastery!")
        elif accuracy >= 0.5:
            print("🟡 Good progress, needs reinforcement")
        else:
            print("🔄 Requires additional teaching")
        
        return accuracy
    
    def test_advanced_pattern(self, pattern, expected_output, pattern_name):
        """Test recognition of advanced pattern"""
        # Reset network
        for layer in self.network.layers.values():
            layer.neuron_population.reset()
        
        # Test input
        flat_pattern = pattern.flatten()
        input_currents = [60.0 if pixel > 0.5 else 0.0 for pixel in flat_pattern]
        
        # Get populations
        input_pop = self.network.layers["input"].neuron_population
        hidden1_pop = self.network.layers["hidden1"].neuron_population
        hidden2_pop = self.network.layers["hidden2"].neuron_population
        output_pop = self.network.layers["output"].neuron_population
        
        output_activity = [0] * 6
        
        # Extended testing for multi-layer propagation
        for step in range(80):
            # Input stimulation
            input_states = input_pop.step(0.1, input_currents)
            
            # Layer processing (no external currents)
            hidden1_states = hidden1_pop.step(0.1, [0.0] * 12)
            hidden2_states = hidden2_pop.step(0.1, [0.0] * 8)
            output_states = output_pop.step(0.1, [0.0] * 6)
            
            # Count output activity
            for i, spike in enumerate(output_states):
                if spike:
                    output_activity[i] += 1
            
            self.network.step(0.1)
        
        # Determine recognition
        if max(output_activity) > 0:
            winner = np.argmax(output_activity)
            recognized = (winner == expected_output)
        else:
            winner = -1
            recognized = False
        
        return {
            'pattern_name': pattern_name,
            'expected_output': expected_output,
            'winner': winner,
            'output_activity': output_activity,
            'recognized': recognized
        }
    
    def capture_network_weights(self):
        """Capture all network weights for learning analysis"""
        all_weights = []
        for (pre_layer, post_layer), connection in self.network.connections.items():
            if connection.synapse_population and hasattr(connection.synapse_population, 'synapses'):
                for (pre_idx, post_idx), synapse in connection.synapse_population.synapses.items():
                    all_weights.append(synapse.weight)
        return all_weights
    
    def calculate_weight_changes(self, weights_before, weights_after):
        """Calculate weight changes for learning measurement"""
        return [after - before for before, after in zip(weights_before, weights_after)]
    
    def run_advanced_curriculum(self):
        """Execute complete advanced learning curriculum"""
        print("Starting advanced neuromorphic learning curriculum...")
        
        # Create sophisticated patterns
        patterns, targets = self.create_advanced_patterns()
        print(f"✅ Created {len(patterns)} advanced patterns")
        
        # Progressive teaching
        learning_progress = self.progressive_teaching(patterns, targets)
        
        # Final comprehensive test
        print(f"\n🏆 FINAL COMPREHENSIVE ASSESSMENT")
        print("=" * 40)
        
        final_results = []
        for pattern_name, pattern in patterns.items():
            expected = targets[pattern_name].index(1)
            result = self.test_advanced_pattern(pattern, expected, pattern_name)
            final_results.append(result)
        
        # Calculate final performance
        total_correct = sum(1 for result in final_results if result['recognized'])
        total_patterns = len(final_results)
        final_accuracy = total_correct / total_patterns
        
        print(f"\nFinal Performance: {total_correct}/{total_patterns} ({final_accuracy:.1%})")
        
        # Advanced analysis
        learning_magnitude_trend = [entry['learning_magnitude'] for entry in learning_progress]
        avg_learning = np.mean(learning_magnitude_trend)
        
        print(f"Average learning magnitude: {avg_learning:.3f}")
        print(f"Total learning epochs: {len(learning_progress)}")
        
        # Success criteria
        if final_accuracy >= 0.8 and avg_learning > 1.0:
            print(f"\n🎉 ADVANCED LEARNING SUCCESS!")
            print(f"✅ High pattern recognition accuracy")
            print(f"✅ Strong synaptic plasticity")
            print(f"✅ Multi-layer learning coordination")
            success_status = "ADVANCED_SUCCESS"
        elif final_accuracy >= 0.5:
            print(f"\n🟡 GOOD PROGRESS - Continuing development")
            success_status = "PROGRESSING"
        else:
            print(f"\n🔄 LEARNING IN PROGRESS - More teaching needed")
            success_status = "DEVELOPING"
        
        # Save advanced learning report
        report = {
            'timestamp': datetime.now().isoformat(),
            'curriculum_type': 'advanced_progressive',
            'total_patterns': total_patterns,
            'patterns_recognized': total_correct,
            'final_accuracy': final_accuracy,
            'average_learning_magnitude': avg_learning,
            'total_epochs': len(learning_progress),
            'success_status': success_status,
            'detailed_results': final_results,
            'learning_progress': learning_progress[-10:]  # Last 10 epochs
        }
        
        with open('advanced_learning_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n💾 Advanced learning report saved: advanced_learning_report.json")
        
        return final_accuracy >= 0.5

if __name__ == "__main__":
    curriculum = AdvancedLearningCurriculum()
    success = curriculum.run_advanced_curriculum()
    
    if success:
        print(f"\n🌟 ADVANCED NEUROMORPHIC LEARNING: SUCCESSFUL!")
        print(f"The system demonstrates sophisticated pattern learning capabilities.")
    else:
        print(f"\n📚 Continuing the advanced learning journey...")
