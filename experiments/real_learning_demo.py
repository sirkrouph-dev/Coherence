#!/usr/bin/env python3
"""
FIXED LEARNING DEMO v2 - With correct input format!
"""

import numpy as np
from api.neuromorphic_api import NeuromorphicAPI

class RealShapeLearner:
    def __init__(self):
        print("🧠 REAL SHAPE LEARNING - CORRECT INPUT FORMAT")
        print("=" * 55)
        
        # Create network
        self.api = NeuromorphicAPI()
        self.api.create_network()
        
        # Simple but effective network
        self.api.add_sensory_layer("input", 64, "lif")    # 8x8 = 64 neurons
        self.api.add_processing_layer("hidden", 20, "lif") # Hidden layer
        self.api.add_motor_layer("output", 4)              # 4 shape classes
        
        # Connect with good connectivity
        self.api.connect_layers("input", "hidden", "stdp", connection_probability=0.3)
        self.api.connect_layers("hidden", "output", "stdp", connection_probability=0.5)
        
        print(f"✅ Network: 64 input → 20 hidden → 4 output")
        
        # Create shape patterns
        self.shapes = {
            'circle': self.create_circle_pattern(),
            'square': self.create_square_pattern(),
            'triangle': self.create_triangle_pattern(),
            'line': self.create_line_pattern()
        }
        
        print(f"✅ Created {len(self.shapes)} shape patterns")
    
    def create_circle_pattern(self):
        """Simple circle on 8x8 grid"""
        pattern = np.zeros((8, 8))
        center = 3.5
        radius = 2.5
        for i in range(8):
            for j in range(8):
                if (i - center)**2 + (j - center)**2 <= radius**2:
                    pattern[i, j] = 1.0
        return pattern
    
    def create_square_pattern(self):
        """Simple square"""
        pattern = np.zeros((8, 8))
        pattern[2:6, 2:6] = 1.0
        return pattern
    
    def create_triangle_pattern(self):
        """Simple triangle"""
        pattern = np.zeros((8, 8))
        # Triangle pointing up
        for i in range(8):
            width = max(0, 7 - i)
            if width > 0:
                start = (8 - width) // 2
                end = start + width
                pattern[i, start:end] = 1.0
        return pattern
    
    def create_line_pattern(self):
        """Diagonal line"""
        pattern = np.zeros((8, 8))
        for i in range(8):
            pattern[i, i] = 1.0
        return pattern
    
    def pattern_to_spikes(self, pattern, duration=100.0):
        """Convert pattern to spike format: [(neuron_id, spike_time), ...]"""
        spikes = []
        flat_pattern = pattern.flatten()
        
        for neuron_id in range(64):  # 8x8 = 64 neurons
            activation = flat_pattern[neuron_id]
            
            if activation > 0.5:  # Active pixel
                # Generate multiple spikes for active neurons
                spike_times = [10, 25, 40, 55, 70, 85]  # Regular firing
                for spike_time in spike_times:
                    if spike_time < duration:
                        spikes.append((neuron_id, float(spike_time)))
        
        return spikes
    
    def measure_output_spikes(self, results):
        """Count spikes in each output neuron"""
        if 'layer_spike_times' not in results:
            return [0, 0, 0, 0]
        
        output_spikes = results['layer_spike_times'].get('output', [])
        spike_counts = [0, 0, 0, 0]
        
        # Count spikes per output neuron
        if isinstance(output_spikes, list) and len(output_spikes) >= 4:
            for i in range(4):
                if i < len(output_spikes) and isinstance(output_spikes[i], list):
                    spike_counts[i] = len(output_spikes[i])
        
        return spike_counts
    
    def classify_response(self, spike_counts, target_shape):
        """Determine predicted class from spike counts"""
        shape_names = ['circle', 'square', 'triangle', 'line']
        total_spikes = sum(spike_counts)
        
        if total_spikes == 0:
            return 0.0, "No spikes", False
        
        # Find most active output neuron
        max_spikes = max(spike_counts)
        predicted_idx = spike_counts.index(max_spikes)
        predicted_shape = shape_names[predicted_idx]
        
        # Calculate confidence
        confidence = max_spikes / total_spikes
        
        # Check correctness
        target_idx = shape_names.index(target_shape)
        correct = (predicted_idx == target_idx)
        
        status = "✅" if correct else "❌"
        return confidence, f"{status} {predicted_shape}", correct
    
    def train_on_shape(self, shape_name, pattern):
        """Train network on one shape with supervised learning"""
        # Convert pattern to spike input
        input_spikes = self.pattern_to_spikes(pattern)
        
        # Add target teaching signal
        shape_names = ['circle', 'square', 'triangle', 'line']
        target_idx = shape_names.index(shape_name)
        
        # Create external inputs with target signal
        external_inputs = {
            "input": input_spikes,
            # Add teaching signal to correct output neuron
            "output": [(target_idx, 50.0), (target_idx, 70.0)]  # Teacher spikes
        }
        
        # Run training simulation
        results = self.api.run_simulation(duration=100.0, external_inputs=external_inputs)
        
        # Measure response
        spike_counts = self.measure_output_spikes(results)
        confidence, status, correct = self.classify_response(spike_counts, shape_name)
        
        return spike_counts, confidence, status, correct
    
    def test_shape(self, shape_name, pattern):
        """Test network on shape without teaching signal"""
        input_spikes = self.pattern_to_spikes(pattern)
        
        # No teaching signal - pure test
        external_inputs = {"input": input_spikes}
        
        results = self.api.run_simulation(duration=100.0, external_inputs=external_inputs)
        
        spike_counts = self.measure_output_spikes(results)
        confidence, status, correct = self.classify_response(spike_counts, shape_name)
        
        return spike_counts, confidence, status, correct
    
    def run_learning_experiment(self, epochs=5):
        """Run complete learning experiment"""
        print(f"\n🎓 LEARNING EXPERIMENT - {epochs} EPOCHS")
        print("=" * 50)
        
        # Test before training
        print(f"\n🧪 PRE-TRAINING TEST:")
        pre_accuracy = self.test_all_shapes()
        
        # Training loop
        epoch_accuracies = []
        
        for epoch in range(epochs):
            print(f"\n📚 EPOCH {epoch+1}/{epochs}")
            print("-" * 30)
            
            correct_count = 0
            total_count = 0
            
            for shape_name, pattern in self.shapes.items():
                spike_counts, confidence, status, correct = self.train_on_shape(shape_name, pattern)
                
                print(f"  {shape_name:8s}: {status} | spikes: {spike_counts} | conf: {confidence:.2f}")
                
                if correct:
                    correct_count += 1
                total_count += 1
            
            accuracy = (correct_count / total_count) * 100
            epoch_accuracies.append(accuracy)
            print(f"  📊 Epoch accuracy: {accuracy:.1f}%")
        
        # Test after training
        print(f"\n🧪 POST-TRAINING TEST:")
        post_accuracy = self.test_all_shapes()
        
        return pre_accuracy, epoch_accuracies, post_accuracy
    
    def test_all_shapes(self):
        """Test network on all shapes"""
        correct_count = 0
        total_count = 0
        
        for shape_name, pattern in self.shapes.items():
            spike_counts, confidence, status, correct = self.test_shape(shape_name, pattern)
            
            print(f"  {shape_name:8s}: {status} | spikes: {spike_counts} | conf: {confidence:.2f}")
            
            if correct:
                correct_count += 1
            total_count += 1
        
        accuracy = (correct_count / total_count) * 100
        print(f"  📊 Test accuracy: {accuracy:.1f}%")
        return accuracy

def main():
    # Create learner
    learner = RealShapeLearner()
    
    # Run learning experiment
    pre_acc, epoch_accs, post_acc = learner.run_learning_experiment(epochs=3)
    
    # Summary
    print(f"\n{'='*50}")
    print(f"📊 LEARNING SUMMARY")
    print(f"{'='*50}")
    print(f"Pre-training accuracy:  {pre_acc:.1f}%")
    for i, acc in enumerate(epoch_accs):
        print(f"Epoch {i+1} accuracy:     {acc:.1f}%")
    print(f"Post-training accuracy: {post_acc:.1f}%")
    
    improvement = post_acc - pre_acc
    print(f"\nLearning improvement: {improvement:+.1f}%")
    
    if improvement > 25:
        print("🎉 EXCELLENT LEARNING!")
    elif improvement > 10:
        print("📈 Good learning progress!")
    elif improvement > 0:
        print("📊 Some learning detected")
    else:
        print("🤔 Learning needs optimization")

if __name__ == "__main__":
    main()
