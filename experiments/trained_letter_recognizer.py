#!/usr/bin/env python3
"""
TRAINED LETTER RECOGNITION - PROPER STDP LEARNING
This version implements proper training to distinguish letters
"""

import numpy as np
from core.network import NeuromorphicNetwork
import time

class TrainedLetterRecognizer:
    def __init__(self):
        print("🎓 TRAINED LETTER RECOGNITION - PROPER STDP LEARNING")
        print("=" * 60)
        print("Goal: Train network to distinguish between letters!")
        
        # Create network optimized for learning
        self.network = NeuromorphicNetwork()
        
        input_size = 16    # 4x4 pixels
        hidden_size = 10   # Slightly larger hidden layer
        output_size = 5    # A, E, I, O, U
        
        self.network.add_layer("input", input_size, "lif")
        self.network.add_layer("hidden", hidden_size, "lif")
        self.network.add_layer("output", output_size, "lif")
        
        # Moderate connectivity for learning
        self.network.connect_layers("input", "hidden", "stdp", connection_probability=0.6)
        self.network.connect_layers("hidden", "output", "stdp", connection_probability=0.7)
        
        print(f"✅ Network: {input_size} → {hidden_size} → {output_size}")
        print(f"   Training-optimized connectivity")
        
        # Create distinct letters
        self.create_maximally_distinct_letters()
        
        self.alphabet = ['A', 'E', 'I', 'O', 'U']
        self.letter_to_index = {letter: i for i, letter in enumerate(self.alphabet)}
        
        print(f"✅ Created {len(self.letters)} maximally distinct patterns")
    
    def create_maximally_distinct_letters(self):
        """Create letters that are as different as possible"""
        
        # Make each letter have a unique pattern signature
        
        # A: Top triangle (diagonal emphasis)
        A = np.array([
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 1, 1, 1],
            [1, 0, 0, 1]
        ])
        
        # E: Left side emphasis
        E = np.array([
            [1, 1, 1, 1],
            [1, 0, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1]
        ])
        
        # I: Center emphasis
        I = np.array([
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0]
        ])
        
        # O: Ring/border emphasis
        O = np.array([
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1]
        ])
        
        # U: Bottom emphasis
        U = np.array([
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1]
        ])
        
        self.letters = {'A': A, 'E': E, 'I': I, 'O': O, 'U': U}
        
        # Show distinctiveness
        for letter, pattern in self.letters.items():
            flat = pattern.flatten()
            signature = f"pixels={sum(flat)}, top={sum(flat[:8])}, center={sum(flat[6:10])}"
            print(f"   {letter}: {signature}")
    
    def visualize_letter(self, letter):
        """Show letter"""
        pattern = self.letters[letter]
        print(f"\nLetter {letter}:")
        for row in pattern:
            line = "  "
            for pixel in row:
                line += "██" if pixel > 0.5 else "  "
            print(line)
    
    def train_on_letter(self, letter, target_output_neuron, training_steps=50):
        """Train network to associate letter with specific output neuron"""
        pattern = self.letters[letter]
        flat_pattern = pattern.flatten()
        
        # Strong input for this letter
        input_currents = [40.0 if pixel > 0.5 else 0.0 for pixel in flat_pattern]
        
        # Target output: strong current for target neuron, weak for others
        target_currents = [0.0] * 5
        target_currents[target_output_neuron] = 20.0  # Encourage this neuron
        
        input_pop = self.network.layers["input"].neuron_population
        output_pop = self.network.layers["output"].neuron_population
        
        dt = 0.1
        total_input_spikes = 0
        total_output_spikes = 0
        
        for step in range(training_steps):
            # Stimulate input
            input_states = input_pop.step(dt, input_currents)
            total_input_spikes += sum(input_states)
            
            # Encourage target output  
            output_states = output_pop.step(dt, target_currents)
            total_output_spikes += sum(output_states)
            
            # Network step for STDP learning
            self.network.step(dt)
        
        return total_input_spikes, total_output_spikes
    
    def train_all_letters(self, epochs=3):
        """Train the network on all letters"""
        print(f"\n🎓 TRAINING PHASE - {epochs} epochs")
        print("=" * 35)
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            for i, letter in enumerate(self.alphabet):
                print(f"  Training {letter} → output neuron {i}")
                
                # Train this letter to activate output neuron i
                input_spikes, output_spikes = self.train_on_letter(letter, i)
                print(f"    Input: {input_spikes}, Output: {output_spikes}")
        
        print(f"✅ Training completed!")
    
    def test_letter(self, letter):
        """Test recognition of a single letter"""
        pattern = self.letters[letter]
        flat_pattern = pattern.flatten()
        
        # Test input only (no target encouragement)
        input_currents = [35.0 if pixel > 0.5 else 0.0 for pixel in flat_pattern]
        
        input_pop = self.network.layers["input"].neuron_population
        output_pop = self.network.layers["output"].neuron_population
        
        dt = 0.1
        test_steps = 40
        
        # Track output neuron activity
        output_activity = [0] * 5
        
        for step in range(test_steps):
            # Input only
            input_pop.step(dt, input_currents)
            
            # Monitor output (no external stimulation)
            output_states = output_pop.step(dt, [0.0] * 5)
            for i, fired in enumerate(output_states):
                if fired:
                    output_activity[i] += 1
            
            # Network propagation
            self.network.step(dt)
        
        # Determine prediction
        total_output = sum(output_activity)
        if total_output > 0:
            predicted_neuron = output_activity.index(max(output_activity))
            predicted_letter = self.alphabet[predicted_neuron]
            confidence = max(output_activity) / total_output * 100
        else:
            predicted_letter = 'None'
            confidence = 0
        
        return predicted_letter, output_activity, confidence
    
    def test_all_letters(self):
        """Test recognition on all letters"""
        print(f"\n🧪 TESTING PHASE")
        print("=" * 20)
        
        results = {}
        
        for letter in self.alphabet:
            print(f"\nTesting {letter}:")
            self.visualize_letter(letter)
            
            prediction, activity, confidence = self.test_letter(letter)
            
            correct = prediction == letter
            results[letter] = {
                'prediction': prediction,
                'activity': activity,
                'confidence': confidence,
                'correct': correct
            }
            
            print(f"  Activity: {activity}")
            print(f"  Prediction: {prediction} ({confidence:.1f}% confidence)")
            
            if correct:
                print(f"  ✅ CORRECT!")
            else:
                print(f"  ❌ Wrong (expected {letter})")
        
        return results
    
    def demonstrate_trained_recognition(self):
        """Full demonstration with training and testing"""
        
        # Step 1: Train the network
        self.train_all_letters(epochs=5)
        
        # Step 2: Test recognition
        results = self.test_all_letters()
        
        # Step 3: Analysis
        correct_count = sum(1 for r in results.values() if r['correct'])
        accuracy = (correct_count / len(self.alphabet)) * 100
        
        print(f"\n{'='*50}")
        print(f"🎯 TRAINED RECOGNITION RESULTS")
        print(f"{'='*50}")
        print(f"Total letters: {len(self.alphabet)}")
        print(f"Correct predictions: {correct_count}")
        print(f"Accuracy: {accuracy:.1f}%")
        
        print(f"\nDetailed Results:")
        for letter, result in results.items():
            status = "✅" if result['correct'] else "❌"
            conf = result['confidence']
            print(f"  {letter} → {result['prediction']} ({conf:.1f}%) {status}")
        
        if accuracy >= 80:
            print(f"\n🎉 EXCELLENT! STDP learning is working!")
            print(f"🚀 Ready for full alphabet and word processing!")
        elif accuracy >= 60:
            print(f"\n📈 GOOD! Network shows strong learning capability!")
        elif accuracy >= 40:
            print(f"\n📚 PROGRESS! STDP is creating letter associations!")
        else:
            print(f"\n🔧 DEVELOPMENT! Network is learning but needs refinement!")
        
        print(f"\n🎓 TRAINING SUCCESS INDICATORS:")
        any_learning = any(r['correct'] for r in results.values())
        varied_predictions = len(set(r['prediction'] for r in results.values())) > 1
        output_activity = any(sum(r['activity']) > 0 for r in results.values())
        
        print(f"  ✅ Any correct predictions: {any_learning}")
        print(f"  ✅ Varied predictions: {varied_predictions}")
        print(f"  ✅ Output neuron activity: {output_activity}")
        
        return results

def main():
    recognizer = TrainedLetterRecognizer()
    recognizer.demonstrate_trained_recognition()

if __name__ == "__main__":
    main()
