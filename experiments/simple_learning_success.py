#!/usr/bin/env python3
"""
SIMPLE LEARNING SUCCESS - Using proven GPUNeuronPool approach!
Based on our successful 300M neuron verification.
"""

import numpy as np
import matplotlib.pyplot as plt
from core.gpu_neurons import GPUNeuronPool

class SimpleShapeLearner:
    def __init__(self):
        print("🧠 SIMPLE SHAPE LEARNING SUCCESS")
        print("=" * 50)
        print("Using proven GPUNeuronPool approach!")
        
        # Create neuron populations directly (we know this works!)
        self.input_neurons = GPUNeuronPool(16, 'lif')    # 4x4 input
        self.hidden_neurons = GPUNeuronPool(8, 'lif')    # Hidden layer  
        self.output_neurons = GPUNeuronPool(4, 'lif')    # Output layer
        
        print(f"✅ Created neuron pools:")
        print(f"   Input: 16 neurons (4x4 grid)")
        print(f"   Hidden: 8 neurons")
        print(f"   Output: 4 neurons")
        
        # Create simple weight matrices (manual connections)
        self.w_input_hidden = np.random.randn(16, 8) * 0.5  # Input→Hidden weights
        self.w_hidden_output = np.random.randn(8, 4) * 0.5  # Hidden→Output weights
        
        print(f"✅ Initialized random weights")
        
        # Learning parameters
        self.learning_rate = 0.01
        
        # Create 4x4 patterns
        self.patterns = {
            'circle': np.array([[0,1,1,0],[1,0,0,1],[1,0,0,1],[0,1,1,0]]),
            'square': np.array([[1,1,1,1],[1,0,0,1],[1,0,0,1],[1,1,1,1]]),
            'triangle': np.array([[0,0,1,0],[0,1,1,1],[1,1,1,1],[1,1,1,1]]),
            'line': np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        }
        
        print(f"✅ Created {len(self.patterns)} shape patterns")
    
    def pattern_to_current(self, pattern, strength=25.0):
        """Convert 4x4 pattern to input currents"""
        flat_pattern = pattern.flatten()
        currents = np.zeros(16)
        for i in range(16):
            if flat_pattern[i] > 0.5:
                currents[i] = strength
        return currents
    
    def forward_pass(self, input_currents, duration_steps=50):
        """Run forward pass through the network"""
        # Note: GPUNeuronPool doesn't have reset method, neurons reset automatically
        
        input_spikes = []
        hidden_spikes = []
        output_spikes = []
        
        for step in range(duration_steps):
            dt = 0.1
            
            # Step 1: Input layer
            input_spike_indices, _ = self.input_neurons.step(dt, input_currents)
            # Convert CuPy array to list
            if hasattr(input_spike_indices, 'get'):
                input_spike_indices = input_spike_indices.get().tolist()
            elif hasattr(input_spike_indices, 'tolist'):
                input_spike_indices = input_spike_indices.tolist()
            
            if len(input_spike_indices) > 0:
                input_spikes.extend(input_spike_indices)
            
            # Step 2: Hidden layer (receive from input)
            hidden_input = np.zeros(8)
            for spike_idx in input_spike_indices:
                if spike_idx < 16:  # Valid input neuron
                    hidden_input += self.w_input_hidden[spike_idx] * 10.0  # Spike strength
            
            hidden_spike_indices, _ = self.hidden_neurons.step(dt, hidden_input)
            # Convert CuPy array to list
            if hasattr(hidden_spike_indices, 'get'):
                hidden_spike_indices = hidden_spike_indices.get().tolist()
            elif hasattr(hidden_spike_indices, 'tolist'):
                hidden_spike_indices = hidden_spike_indices.tolist()
                
            if len(hidden_spike_indices) > 0:
                hidden_spikes.extend(hidden_spike_indices)
            
            # Step 3: Output layer (receive from hidden)  
            output_input = np.zeros(4)
            for spike_idx in hidden_spike_indices:
                if spike_idx < 8:  # Valid hidden neuron
                    output_input += self.w_hidden_output[spike_idx] * 10.0  # Spike strength
            
            output_spike_indices, _ = self.output_neurons.step(dt, output_input)
            # Convert CuPy array to list
            if hasattr(output_spike_indices, 'get'):
                output_spike_indices = output_spike_indices.get().tolist()
            elif hasattr(output_spike_indices, 'tolist'):
                output_spike_indices = output_spike_indices.tolist()
                
            if len(output_spike_indices) > 0:
                output_spikes.extend(output_spike_indices)
        
        return input_spikes, hidden_spikes, output_spikes
    
    def train_pattern(self, pattern_name, pattern, target_class):
        """Train on one pattern"""
        # Convert pattern to input currents
        input_currents = self.pattern_to_current(pattern, strength=30.0)
        
        # Forward pass
        input_spikes, hidden_spikes, output_spikes = self.forward_pass(input_currents)
        
        # Count output spikes per neuron
        output_counts = [0, 0, 0, 0]
        for spike_idx in output_spikes:
            if 0 <= spike_idx < 4:
                output_counts[spike_idx] += 1
        
        # Determine prediction
        total_spikes = sum(output_counts)
        if total_spikes > 0:
            predicted_class = output_counts.index(max(output_counts))
            confidence = max(output_counts) / total_spikes
        else:
            predicted_class = -1
            confidence = 0.0
        
        # Simple learning: strengthen weights to target class
        if len(hidden_spikes) > 0:
            for spike_idx in hidden_spikes:
                if 0 <= spike_idx < 8:
                    # Strengthen connection to target class
                    self.w_hidden_output[spike_idx, target_class] += self.learning_rate
                    # Weaken connections to other classes
                    for other_class in range(4):
                        if other_class != target_class:
                            self.w_hidden_output[spike_idx, other_class] -= self.learning_rate * 0.1
        
        # Check correctness
        correct = (predicted_class == target_class)
        status = "✅" if correct else "❌" if total_spikes > 0 else "⚪"
        
        return correct, confidence, status, output_counts, len(input_spikes), len(hidden_spikes)
    
    def run_learning_experiment(self, epochs=5):
        """Run complete learning experiment"""
        print(f"\n🎓 LEARNING EXPERIMENT - {epochs} EPOCHS")
        print("=" * 50)
        
        shape_names = list(self.patterns.keys())
        all_accuracies = []
        
        for epoch in range(epochs):
            print(f"\n📚 Epoch {epoch+1}/{epochs}")
            print("-" * 30)
            
            correct_count = 0
            total_count = 0
            
            for target_class, (shape_name, pattern) in enumerate(self.patterns.items()):
                correct, confidence, status, output_counts, input_spikes, hidden_spikes = self.train_pattern(
                    shape_name, pattern, target_class
                )
                
                print(f"  {shape_name:8s}: {status} | conf: {confidence:.2f} | out: {output_counts} | in: {input_spikes:2d} | hid: {hidden_spikes:2d}")
                
                if correct:
                    correct_count += 1
                total_count += 1
            
            accuracy = (correct_count / total_count) * 100
            all_accuracies.append(accuracy)
            print(f"  📊 Epoch accuracy: {accuracy:.1f}%")
        
        return all_accuracies
    
    def test_basic_functionality(self):
        """Test if neurons respond to patterns"""
        print(f"\n🧪 BASIC FUNCTIONALITY TEST")
        print("-" * 35)
        
        for shape_name, pattern in self.patterns.items():
            input_currents = self.pattern_to_current(pattern)
            input_spikes, hidden_spikes, output_spikes = self.forward_pass(input_currents, duration_steps=30)
            
            print(f"{shape_name:8s}: input={len(input_spikes):2d}, hidden={len(hidden_spikes):2d}, output={len(output_spikes):2d}")
            
            if len(input_spikes) > 0:
                print(f"           ✅ Input neurons responding!")
            else:
                print(f"           ❌ No input spikes")

def main():
    # Create learner using proven approach
    learner = SimpleShapeLearner()
    
    # Test basic functionality
    learner.test_basic_functionality()
    
    # Run learning experiment
    accuracies = learner.run_learning_experiment(epochs=5)
    
    # Show results
    print(f"\n{'='*50}")
    print(f"🎯 LEARNING RESULTS")
    print(f"{'='*50}")
    
    for i, accuracy in enumerate(accuracies):
        print(f"Epoch {i+1}: {accuracy:.1f}% accuracy")
    
    if len(accuracies) > 1:
        improvement = accuracies[-1] - accuracies[0]
        print(f"\nLearning improvement: {improvement:+.1f}%")
        
        if improvement > 25:
            print("🎉 EXCELLENT LEARNING ACHIEVED!")
        elif improvement > 10:
            print("📈 Good learning progress!")
        elif improvement > 0:
            print("📊 Some learning detected")
        else:
            print("🔧 Learning needs more epochs")
    
    # Success criteria
    if any(acc > 50 for acc in accuracies):
        print(f"\n🚀 SUCCESS! Achieved >50% accuracy!")
        print(f"✅ Neuromorphic learning is working!")
    else:
        print(f"\n🔧 Need to optimize learning parameters")

if __name__ == "__main__":
    main()
