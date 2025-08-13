#!/usr/bin/env python3
"""
FIXED LEARNING DEMO - Now with proper spike generation and learning!
Based on diagnostic findings, this should actually learn.
"""

import numpy as np
import matplotlib.pyplot as plt
from api.neuromorphic_api import NeuromorphicAPI

class WorkingShapeLearner:
    def __init__(self):
        print("🧠 WORKING SHAPE LEARNING DEMO")
        print("=" * 50)
        print("Fixed based on diagnostic findings!")
        
        # Create network with proper API
        self.api = NeuromorphicAPI()
        self.api.create_network()
        
        # Simple network
        self.api.add_sensory_layer("input", 64, "lif")    # 8x8 = 64 neurons
        self.api.add_processing_layer("hidden", 16, "lif") # Hidden layer
        self.api.add_motor_layer("output", 4)              # 4 shape classes
        
        # Connect layers
        self.api.connect_layers("input", "hidden", "stdp", connection_probability=0.4)
        self.api.connect_layers("hidden", "output", "stdp", connection_probability=0.6)
        
        print(f"✅ Network Created: 64→16→4 neurons")
        
        # Shape patterns (8x8 grid)
        self.shapes = {
            'circle': self.create_circle(),
            'square': self.create_square(), 
            'line': self.create_line(),
            'cross': self.create_cross()
        }
        
        print(f"✅ Created {len(self.shapes)} shape patterns")
    
    def create_circle(self):
        """Create 8x8 circle pattern"""
        pattern = np.zeros((8, 8))
        center = 3.5
        for i in range(8):
            for j in range(8):
                if (i - center)**2 + (j - center)**2 <= 9:  # radius ~3
                    pattern[i, j] = 1.0
        return pattern
    
    def create_square(self):
        """Create 8x8 square pattern"""
        pattern = np.zeros((8, 8))
        pattern[2:6, 2:6] = 1.0  # 4x4 square in center
        return pattern
    
    def create_line(self):
        """Create diagonal line"""
        pattern = np.zeros((8, 8))
        for i in range(8):
            pattern[i, i] = 1.0  # Main diagonal
        return pattern
    
    def create_cross(self):
        """Create cross/plus pattern"""
        pattern = np.zeros((8, 8))
        pattern[3, :] = 1.0  # Horizontal line
        pattern[:, 3] = 1.0  # Vertical line
        return pattern
    
    def pattern_to_spikes(self, pattern, duration=100.0, intensity=30.0):
        """Convert 2D pattern to spike trains for each input neuron"""
        external_inputs = {}
        
        flat_pattern = pattern.flatten()  # Convert 8x8 to 64 values
        
        for neuron_idx in range(64):
            activation = flat_pattern[neuron_idx]
            
            if activation > 0.5:  # Active pixel
                # Generate spike train for this neuron
                spike_times = []
                
                # Strong regular spiking for active pixels
                for t in np.arange(10, duration-10, 20):  # Every 20ms
                    spike_times.append((t, t + 2))  # 2ms current pulse
                
                if spike_times:
                    external_inputs[f"input_{neuron_idx}"] = spike_times
        
        return external_inputs
    
    def measure_response(self, results, target_shape):
        """Measure network response to determine classification"""
        if 'layer_spike_times' not in results:
            return 0.0, "No spike data"
        
        output_spikes = results['layer_spike_times'].get('output', [])
        
        # Count spikes per output neuron
        spike_counts = [0, 0, 0, 0]
        shape_names = ['circle', 'square', 'line', 'cross']
        
        for neuron_spikes in output_spikes:
            if isinstance(neuron_spikes, list):
                for i, spikes in enumerate(neuron_spikes):
                    if i < 4:
                        spike_counts[i] += len(spikes) if spikes else 0
        
        total_spikes = sum(spike_counts)
        if total_spikes == 0:
            return 0.0, "No output spikes"
        
        # Find most active neuron
        max_spikes = max(spike_counts)
        predicted_idx = spike_counts.index(max_spikes)
        predicted_shape = shape_names[predicted_idx]
        
        # Calculate confidence
        confidence = max_spikes / total_spikes if total_spikes > 0 else 0.0
        
        # Check if correct
        correct = (predicted_shape == target_shape)
        status = "✅" if correct else "❌"
        
        return confidence, f"{status} {predicted_shape} (spikes: {spike_counts})"
    
    def train_shape(self, shape_name, pattern, training_rounds=3):
        """Train network on a specific shape"""
        results_summary = []
        
        for round_num in range(training_rounds):
            # Convert pattern to spike inputs
            spike_inputs = self.pattern_to_spikes(pattern, duration=100.0)
            
            if not spike_inputs:
                results_summary.append((0.0, "No input spikes generated"))
                continue
            
            # Provide target signal for this shape (reward-based learning)
            shape_idx = ['circle', 'square', 'line', 'cross'].index(shape_name)
            
            # Add target neuron stimulation for supervised learning
            target_neuron = f"output_{shape_idx}"
            if target_neuron not in spike_inputs:
                spike_inputs[target_neuron] = []
            # Add reward signal
            spike_inputs[target_neuron].extend([(50, 52), (70, 72), (90, 92)])
            
            # Run training simulation
            results = self.api.run_simulation(duration=100.0, external_inputs=spike_inputs)
            
            # Measure response
            confidence, status = self.measure_response(results, shape_name)
            results_summary.append((confidence, status))
            
            print(f"    Round {round_num+1}: {status} (conf: {confidence:.2f})")
        
        return results_summary
    
    def run_learning_experiment(self, epochs=5):
        """Run complete learning experiment"""
        print(f"\n🎓 STARTING LEARNING EXPERIMENT")
        print(f"Training for {epochs} epochs...")
        
        epoch_accuracies = []
        
        for epoch in range(epochs):
            print(f"\n📚 Epoch {epoch+1}/{epochs}")
            
            correct_predictions = 0
            total_predictions = 0
            
            for shape_name, pattern in self.shapes.items():
                print(f"\n  Training on {shape_name}:")
                
                # Train on this shape
                training_results = self.train_shape(shape_name, pattern)
                
                # Get final result
                final_confidence, final_status = training_results[-1]
                if "✅" in final_status:
                    correct_predictions += 1
                total_predictions += 1
                
                print(f"  Final: {final_status}")
            
            # Calculate epoch accuracy
            epoch_accuracy = (correct_predictions / total_predictions) * 100
            epoch_accuracies.append(epoch_accuracy)
            
            print(f"\n  📊 Epoch {epoch+1} Accuracy: {epoch_accuracy:.1f}%")
        
        return epoch_accuracies
    
    def test_network(self):
        """Test network on all shapes without training"""
        print(f"\n🧪 TESTING NETWORK (No Training)")
        print("-" * 40)
        
        for shape_name, pattern in self.shapes.items():
            spike_inputs = self.pattern_to_spikes(pattern, duration=100.0)
            
            if spike_inputs:
                results = self.api.run_simulation(duration=100.0, external_inputs=spike_inputs)
                confidence, status = self.measure_response(results, shape_name)
                print(f"  {shape_name:8s}: {status} (conf: {confidence:.2f})")
            else:
                print(f"  {shape_name:8s}: ❌ No input spikes")

def main():
    # Create learner
    learner = WorkingShapeLearner()
    
    # First test without training
    learner.test_network()
    
    # Run learning experiment
    accuracies = learner.run_learning_experiment(epochs=3)
    
    # Show results
    print(f"\n{'='*50}")
    print(f"📊 LEARNING RESULTS")
    print(f"{'='*50}")
    
    for i, accuracy in enumerate(accuracies):
        print(f"Epoch {i+1}: {accuracy:.1f}% accuracy")
    
    if len(accuracies) > 1:
        improvement = accuracies[-1] - accuracies[0]
        print(f"\nLearning improvement: {improvement:+.1f}%")
        
        if improvement > 10:
            print("🎉 Significant learning detected!")
        elif improvement > 0:
            print("📈 Some learning progress!")
        else:
            print("🤔 Learning needs optimization...")

if __name__ == "__main__":
    main()
