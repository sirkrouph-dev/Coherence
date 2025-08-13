#!/usr/bin/env python3
"""
NEUROMORPHIC LEARNING ASSESSMENT TOOL
Tangible verification of what the network has actually learned
"""

import numpy as np
from core.network import NeuromorphicNetwork
import json
import time
from datetime import datetime

class LearningAssessment:
    def __init__(self):
        print("📊 NEUROMORPHIC LEARNING ASSESSMENT")
        print("=" * 40)
        print("Tangible verification of learned knowledge")
        
        # Create assessment network
        self.network = NeuromorphicNetwork()
        self.setup_assessment_network()
        
        # Learning metrics
        self.learning_metrics = {
            'pattern_recognition': {},
            'association_strength': {},
            'memory_retention': {},
            'discrimination_ability': {},
            'generalization': {}
        }
        
    def setup_assessment_network(self):
        """Create simple network for learning assessment"""
        # Small network for clear learning observation
        self.network.add_layer("input", 16, "lif")    # 4x4 patterns
        self.network.add_layer("hidden", 8, "lif")    # Processing
        self.network.add_layer("output", 4, "lif")    # 4 categories
        
        # Learning connections
        self.network.connect_layers("input", "hidden", "stdp", connection_probability=0.7)
        self.network.connect_layers("hidden", "output", "stdp", connection_probability=0.8)
        
        print(f"✅ Assessment network: 16 → 8 → 4 neurons")
    
    def create_training_patterns(self):
        """Create distinct patterns for learning assessment"""
        patterns = {
            'vertical_lines': np.array([
                [1, 0, 1, 0],
                [1, 0, 1, 0], 
                [1, 0, 1, 0],
                [1, 0, 1, 0]
            ]),
            'horizontal_lines': np.array([
                [1, 1, 1, 1],
                [0, 0, 0, 0],
                [1, 1, 1, 1], 
                [0, 0, 0, 0]
            ]),
            'diagonal': np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]),
            'center_cross': np.array([
                [0, 1, 1, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [0, 1, 1, 0]
            ])
        }
        
        # Target outputs (one-hot encoding)
        targets = {
            'vertical_lines': [1, 0, 0, 0],
            'horizontal_lines': [0, 1, 0, 0],
            'diagonal': [0, 0, 1, 0],
            'center_cross': [0, 0, 0, 1]
        }
        
        return patterns, targets
    
    def train_network(self, patterns, targets, epochs=10):
        """Train network and record learning progress"""
        print(f"\n🎓 TRAINING PHASE: {epochs} epochs")
        print("-" * 30)
        
        training_history = []
        
        for epoch in range(epochs):
            epoch_results = {}
            print(f"Epoch {epoch + 1}/{epochs}")
            
            for pattern_name, pattern in patterns.items():
                target = targets[pattern_name]
                
                # Train on this pattern
                result = self.train_single_pattern(pattern, target)
                epoch_results[pattern_name] = result
                
                print(f"  {pattern_name}: Input={result['input_spikes']}, Output={result['output_spikes']}")
            
            training_history.append({
                'epoch': epoch + 1,
                'timestamp': datetime.now().isoformat(),
                'results': epoch_results
            })
        
        return training_history
    
    def train_single_pattern(self, pattern, target):
        """Train on a single pattern and return learning metrics"""
        flat_pattern = pattern.flatten()
        
        # Input stimulation
        input_currents = [40.0 if pixel > 0.5 else 0.0 for pixel in flat_pattern]
        
        # Target encouragement (for supervised learning)
        target_currents = [20.0 if target[i] == 1 else 0.0 for i in range(4)]
        
        # Get populations
        input_pop = self.network.layers["input"].neuron_population
        output_pop = self.network.layers["output"].neuron_population
        
        dt = 0.1
        steps = 30
        
        input_spikes = 0
        output_spikes = 0
        
        for step in range(steps):
            # Stimulate
            input_states = input_pop.step(dt, input_currents)
            output_states = output_pop.step(dt, target_currents)
            
            input_spikes += sum(input_states)
            output_spikes += sum(output_states)
            
            # Network learning step
            self.network.step(dt)
        
        return {
            'input_spikes': input_spikes,
            'output_spikes': output_spikes,
            'pattern_pixels': int(np.sum(pattern)),
            'target_neuron': target.index(1) if 1 in target else -1
        }
    
    def test_learned_knowledge(self, patterns, targets):
        """Test what the network has actually learned"""
        print(f"\n🧪 KNOWLEDGE TESTING")
        print("-" * 25)
        
        test_results = {}
        
        for pattern_name, pattern in patterns.items():
            print(f"\nTesting: {pattern_name}")
            
            # Show pattern
            print("Pattern:")
            for row in pattern:
                line = "  "
                for pixel in row:
                    line += "██" if pixel > 0.5 else "  "
                print(line)
            
            # Test recognition
            result = self.test_single_pattern(pattern)
            test_results[pattern_name] = result
            
            # Analyze response
            expected_neuron = targets[pattern_name].index(1)
            output_activity = result['output_activity']
            
            print(f"Expected neuron: {expected_neuron}")
            print(f"Output activity: {output_activity}")
            
            if max(output_activity) > 0:
                predicted_neuron = output_activity.index(max(output_activity))
                correct = predicted_neuron == expected_neuron
                
                print(f"Predicted neuron: {predicted_neuron}")
                print(f"Accuracy: {'✅ CORRECT' if correct else '❌ WRONG'}")
                
                # Calculate confidence
                total_activity = sum(output_activity)
                confidence = (max(output_activity) / total_activity * 100) if total_activity > 0 else 0
                print(f"Confidence: {confidence:.1f}%")
                
                result['predicted_neuron'] = predicted_neuron
                result['correct'] = correct
                result['confidence'] = confidence
            else:
                print("❓ NO RESPONSE - Network needs more training")
                result['predicted_neuron'] = -1
                result['correct'] = False
                result['confidence'] = 0
        
        return test_results
    
    def test_single_pattern(self, pattern):
        """Test network response to a single pattern"""
        flat_pattern = pattern.flatten()
        input_currents = [30.0 if pixel > 0.5 else 0.0 for pixel in flat_pattern]
        
        # Get populations
        input_pop = self.network.layers["input"].neuron_population
        output_pop = self.network.layers["output"].neuron_population
        
        dt = 0.1
        steps = 40
        
        output_activity = [0, 0, 0, 0]
        
        for step in range(steps):
            # Input only (no target assistance)
            input_pop.step(dt, input_currents)
            
            # Monitor output
            output_states = output_pop.step(dt, [0.0, 0.0, 0.0, 0.0])
            for i, fired in enumerate(output_states):
                if fired:
                    output_activity[i] += 1
            
            # Network step
            self.network.step(dt)
        
        return {
            'output_activity': output_activity,
            'total_output': sum(output_activity),
            'pattern_pixels': int(np.sum(pattern))
        }
    
    def calculate_learning_metrics(self, training_history, test_results):
        """Calculate tangible learning metrics"""
        print(f"\n📊 LEARNING METRICS ANALYSIS")
        print("-" * 35)
        
        metrics = {}
        
        # 1. Training Progression
        first_epoch = training_history[0]['results']
        last_epoch = training_history[-1]['results']
        
        training_improvement = {}
        for pattern_name in first_epoch.keys():
            first_output = first_epoch[pattern_name]['output_spikes']
            last_output = last_epoch[pattern_name]['output_spikes']
            improvement = ((last_output - first_output) / max(first_output, 1)) * 100
            training_improvement[pattern_name] = improvement
        
        metrics['training_improvement'] = training_improvement
        
        # 2. Pattern Recognition Accuracy
        correct_predictions = sum(1 for result in test_results.values() if result.get('correct', False))
        total_patterns = len(test_results)
        accuracy = (correct_predictions / total_patterns) * 100 if total_patterns > 0 else 0
        
        metrics['recognition_accuracy'] = accuracy
        metrics['correct_patterns'] = correct_predictions
        metrics['total_patterns'] = total_patterns
        
        # 3. Response Strength
        avg_confidence = np.mean([result.get('confidence', 0) for result in test_results.values()])
        metrics['average_confidence'] = avg_confidence
        
        # 4. Network Activity
        total_responses = sum(result['total_output'] for result in test_results.values())
        active_patterns = sum(1 for result in test_results.values() if result['total_output'] > 0)
        
        metrics['total_neural_responses'] = total_responses
        metrics['active_response_rate'] = (active_patterns / total_patterns) * 100
        
        return metrics
    
    def generate_learning_report(self, metrics, training_history, test_results):
        """Generate comprehensive learning report"""
        print(f"\n📋 COMPREHENSIVE LEARNING REPORT")
        print("=" * 40)
        
        # Overall Performance
        print(f"🎯 OVERALL PERFORMANCE:")
        print(f"  Recognition Accuracy: {metrics['recognition_accuracy']:.1f}%")
        print(f"  Correct Patterns: {metrics['correct_patterns']}/{metrics['total_patterns']}")
        print(f"  Average Confidence: {metrics['average_confidence']:.1f}%")
        print(f"  Neural Response Rate: {metrics['active_response_rate']:.1f}%")
        
        # Training Progress
        print(f"\n📈 TRAINING PROGRESS:")
        for pattern, improvement in metrics['training_improvement'].items():
            direction = "↗️" if improvement > 0 else "↘️" if improvement < 0 else "→"
            print(f"  {pattern}: {improvement:+.1f}% {direction}")
        
        # Pattern-by-Pattern Analysis
        print(f"\n🔍 DETAILED PATTERN ANALYSIS:")
        for pattern_name, result in test_results.items():
            status = "✅" if result.get('correct', False) else "❌"
            confidence = result.get('confidence', 0)
            activity = result['total_output']
            print(f"  {pattern_name}: {status} (confidence: {confidence:.1f}%, activity: {activity})")
        
        # Learning Evidence
        print(f"\n🧠 EVIDENCE OF LEARNING:")
        evidence_count = 0
        
        if metrics['recognition_accuracy'] > 0:
            print(f"  ✅ Network recognizes {metrics['correct_patterns']} patterns correctly")
            evidence_count += 1
        
        if metrics['average_confidence'] > 50:
            print(f"  ✅ High confidence responses ({metrics['average_confidence']:.1f}%)")
            evidence_count += 1
        
        if metrics['active_response_rate'] > 75:
            print(f"  ✅ Consistent neural responses ({metrics['active_response_rate']:.1f}%)")
            evidence_count += 1
        
        # Check for learning improvement
        positive_improvements = sum(1 for imp in metrics['training_improvement'].values() if imp > 0)
        if positive_improvements >= len(metrics['training_improvement']) / 2:
            print(f"  ✅ Training shows improvement in {positive_improvements} patterns")
            evidence_count += 1
        
        print(f"\n🎖️ LEARNING VERIFICATION: {evidence_count}/4 criteria met")
        
        if evidence_count >= 3:
            print(f"🎉 STRONG EVIDENCE OF LEARNING!")
        elif evidence_count >= 2:
            print(f"📈 MODERATE LEARNING DETECTED")
        elif evidence_count >= 1:
            print(f"📚 EARLY LEARNING SIGNS")
        else:
            print(f"🔧 LEARNING IN PROGRESS")
        
        return {
            'overall_metrics': metrics,
            'evidence_score': evidence_count,
            'learning_verified': evidence_count >= 2
        }
    
    def run_learning_assessment(self):
        """Run complete learning assessment"""
        print(f"Starting comprehensive learning assessment...")
        
        # Create patterns
        patterns, targets = self.create_training_patterns()
        
        # Train network
        training_history = self.train_network(patterns, targets, epochs=5)
        
        # Test learned knowledge
        test_results = self.test_learned_knowledge(patterns, targets)
        
        # Calculate metrics
        metrics = self.calculate_learning_metrics(training_history, test_results)
        
        # Generate report
        final_report = self.generate_learning_report(metrics, training_history, test_results)
        
        # Save results
        assessment_data = {
            'timestamp': datetime.now().isoformat(),
            'training_history': training_history,
            'test_results': test_results,
            'metrics': metrics,
            'final_report': final_report
        }
        
        return assessment_data

def main():
    assessment = LearningAssessment()
    results = assessment.run_learning_assessment()
    
    # Save to file for future reference
    with open('learning_assessment_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: learning_assessment_results.json")

if __name__ == "__main__":
    main()
