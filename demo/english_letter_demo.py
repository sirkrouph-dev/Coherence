#!/usr/bin/env python3
"""
ENGLISH LETTER RECOGNITION DEMO
Phase 1 of neuromorphic language processing - learning the alphabet!
"""

import numpy as np
from core.network import NeuromorphicNetwork
import time

class EnglishLetterLearner:
    def __init__(self):
        print("🔤 ENGLISH LETTER RECOGNITION - NEUROMORPHIC LEARNING")
        print("=" * 60)
        print("Phase 1: Teaching the network to read letters A-Z!")
        
        # Create letter recognition network
        self.network = NeuromorphicNetwork()
        
        # Network architecture for letter recognition
        # 28x28 input (like MNIST) → hidden layer → output neurons (one per implemented letter)
        input_size = 28 * 28  # 784 neurons for 28x28 pixels
        hidden_size = 128     # Hidden processing layer
        output_size = 10      # One neuron per implemented letter (not all 26)
        
        self.network.add_layer("visual_input", input_size, "lif")
        self.network.add_layer("letter_processing", hidden_size, "lif") 
        self.network.add_layer("letter_output", output_size, "lif")
        
        # Connect layers with STDP learning
        self.network.connect_layers("visual_input", "letter_processing", "stdp", 
                                   connection_probability=0.1)  # Sparse connectivity
        self.network.connect_layers("letter_processing", "letter_output", "stdp",
                                   connection_probability=0.3)  # Denser for classification
        
        print(f"✅ Network: {input_size} → {hidden_size} → {output_size} neurons")
        print(f"   Total neurons: {input_size + hidden_size + output_size:,}")
        print(f"   Task: Learn to recognize {output_size} English letters")
        
        # Letter patterns - simple bitmap representations
        self.letters = {}
        self.create_letter_patterns()
        
        print(f"✅ Created {len(self.letters)} letter patterns")
    
    def create_letter_patterns(self):
        """Create simple bitmap patterns for letters A-Z"""
        # For demo purposes, create simple 28x28 patterns for key letters
        # In practice, these would be from font rendering or handwriting data
        
        size = 28
        patterns = {}
        
        # Letter A - Triangle with crossbar
        A = np.zeros((size, size))
        # Vertical lines
        A[5:23, 10] = 1  # Left line
        A[5:23, 17] = 1  # Right line
        # Top point
        A[5, 11:17] = 1
        # Crossbar
        A[14, 11:17] = 1
        patterns['A'] = A
        
        # Letter B - Two bumps
        B = np.zeros((size, size))
        B[5:23, 8] = 1      # Vertical line
        B[5, 8:16] = 1      # Top horizontal
        B[13, 8:15] = 1     # Middle horizontal  
        B[22, 8:16] = 1     # Bottom horizontal
        B[8, 15] = 1        # Top right curve
        B[17, 15] = 1       # Bottom right curve
        patterns['B'] = B
        
        # Letter C - Curved opening
        C = np.zeros((size, size))
        C[8:20, 10] = 1     # Left vertical
        C[8, 11:17] = 1     # Top horizontal
        C[19, 11:17] = 1    # Bottom horizontal
        patterns['C'] = C
        
        # Letter E - Three horizontal lines
        E = np.zeros((size, size))
        E[5:23, 8] = 1      # Vertical line
        E[5, 8:18] = 1      # Top horizontal
        E[13, 8:15] = 1     # Middle horizontal
        E[22, 8:18] = 1     # Bottom horizontal
        patterns['E'] = E
        
        # Letter H - Two verticals with crossbar
        H = np.zeros((size, size))
        H[5:23, 8] = 1      # Left vertical
        H[5:23, 18] = 1     # Right vertical
        H[13, 8:19] = 1     # Crossbar
        patterns['H'] = H
        
        # Letter I - Vertical line with serifs
        I = np.zeros((size, size))
        I[5:23, 13] = 1     # Vertical line
        I[5, 11:16] = 1     # Top serif
        I[22, 11:16] = 1    # Bottom serif
        patterns['I'] = I
        
        # Letter L - Vertical with bottom horizontal
        L = np.zeros((size, size))
        L[5:23, 8] = 1      # Vertical line
        L[22, 8:18] = 1     # Bottom horizontal
        patterns['L'] = L
        
        # Letter O - Circle/oval
        O = np.zeros((size, size))
        # Create oval shape
        for i in range(8, 20):
            for j in range(10, 18):
                if (i-14)**2/36 + (j-14)**2/16 <= 1:
                    if (i-14)**2/25 + (j-14)**2/9 >= 1:  # Hollow center
                        O[i, j] = 1
        patterns['O'] = O
        
        # Letter T - Horizontal top with vertical center
        T = np.zeros((size, size))
        T[5, 6:22] = 1      # Top horizontal
        T[5:23, 13] = 1     # Vertical line
        patterns['T'] = T
        
        # Letter X - Two diagonal lines
        X = np.zeros((size, size))
        for i in range(18):
            X[5+i, 8+i] = 1   # Diagonal \
            X[5+i, 20-i] = 1  # Diagonal /
        patterns['X'] = X
        
        self.letters = patterns
        
        # Also create alphabet mapping
        self.alphabet = list('ABCEHILOTX')  # Letters we implemented
        self.letter_to_index = {letter: i for i, letter in enumerate(self.alphabet)}
        
        print(f"   Implemented letters: {', '.join(self.alphabet)}")
    
    def visualize_letter(self, letter):
        """Print ASCII visualization of a letter pattern"""
        if letter not in self.letters:
            print(f"Letter {letter} not implemented")
            return
        
        pattern = self.letters[letter]
        print(f"\nLetter {letter} (28x28 pattern):")
        print("+" + "-" * 28 + "+")
        
        for row in pattern:
            line = "|"
            for pixel in row:
                line += "█" if pixel > 0.5 else " "
            line += "|"
            print(line)
        print("+" + "-" * 28 + "+")
    
    def train_letter_recognition(self, epochs_per_letter=3):
        """Train the network to recognize letters"""
        print(f"\n🧠 TRAINING LETTER RECOGNITION")
        print("=" * 40)
        
        total_letters = len(self.alphabet)
        total_training = total_letters * epochs_per_letter
        
        print(f"Training on {total_letters} letters × {epochs_per_letter} epochs = {total_training} presentations")
        
        training_results = []
        start_time = time.time()
        
        for epoch in range(epochs_per_letter):
            print(f"\n📚 Epoch {epoch + 1}/{epochs_per_letter}")
            epoch_results = []
            
            for i, letter in enumerate(self.alphabet):
                print(f"  Training letter {letter} ({i+1}/{total_letters})")
                
                # Get letter pattern and flatten it
                pattern = self.letters[letter]
                flat_pattern = pattern.flatten()
                
                # Stimulate the network with this letter
                result = self.stimulate_with_letter(flat_pattern, letter)
                epoch_results.append(result)
                
                # Quick progress indicator
                if result['output_spikes'] > 0:
                    print(f"    ✅ {result['input_spikes']} input → {result['output_spikes']} output spikes")
                else:
                    print(f"    ⚪ {result['input_spikes']} input → no output")
            
            training_results.append(epoch_results)
        
        training_time = time.time() - start_time
        print(f"\n🎯 Training completed in {training_time:.1f} seconds")
        
        return training_results
    
    def stimulate_with_letter(self, flat_pattern, letter_name):
        """Stimulate network with a specific letter pattern"""
        # Convert pattern to neural stimulation
        input_currents = []
        for pixel in flat_pattern:
            if pixel > 0.5:
                input_currents.append(25.0)  # Strong current for active pixels
            else:
                input_currents.append(0.0)   # No current for inactive pixels
        
        # Get neuron populations
        input_pop = self.network.layers["visual_input"].neuron_population
        output_pop = self.network.layers["letter_output"].neuron_population
        
        # Stimulation parameters
        dt = 0.1
        stimulation_steps = 50
        propagation_steps = 100
        
        # Phase 1: Input stimulation
        input_spikes = 0
        for step in range(stimulation_steps):
            spike_states = input_pop.step(dt, input_currents)
            input_spikes += sum(spike_states)
            # Also step the network for immediate propagation
            self.network.step(dt)
        
        # Phase 2: Let activity propagate
        output_spikes = 0
        output_pattern = [0] * len(self.alphabet)
        
        for step in range(propagation_steps):
            # No external input, just propagation
            zero_currents = [0.0] * len(input_currents)
            input_pop.step(dt, zero_currents)
            self.network.step(dt)
            
            # Check for output spikes (use len(self.alphabet) since that's our actual output size)
            output_spike_states = output_pop.step(dt, [0.0] * len(self.alphabet))
            step_spikes = sum(output_spike_states)
            output_spikes += step_spikes
            
            # Record which output neurons fired
            for i, fired in enumerate(output_spike_states):
                if fired:
                    output_pattern[i] += 1
        
        return {
            'letter': letter_name,
            'input_spikes': input_spikes,
            'output_spikes': output_spikes,
            'output_pattern': output_pattern,
            'target_neuron': self.letter_to_index.get(letter_name, -1)
        }
    
    def test_letter_recognition(self):
        """Test the trained network on letter recognition"""
        print(f"\n🧪 TESTING LETTER RECOGNITION")
        print("=" * 35)
        
        test_results = []
        correct_predictions = 0
        
        for letter in self.alphabet:
            print(f"\nTesting letter {letter}:")
            
            # Visualize the letter
            self.visualize_letter(letter)
            
            # Test recognition
            pattern = self.letters[letter]
            flat_pattern = pattern.flatten()
            result = self.stimulate_with_letter(flat_pattern, letter)
            
            # Analyze output
            output_pattern = result['output_pattern']
            target_index = result['target_neuron']
            
            print(f"Output neuron activities: {output_pattern}")
            print(f"Target neuron (index {target_index}): {output_pattern[target_index] if target_index >= 0 else 'N/A'}")
            
            # Determine predicted letter
            max_activity = max(output_pattern)
            if max_activity > 0:
                predicted_index = output_pattern.index(max_activity)
                predicted_letter = self.alphabet[predicted_index] if predicted_index < len(self.alphabet) else '?'
                
                if predicted_letter == letter:
                    print(f"✅ CORRECT! Predicted: {predicted_letter}")
                    correct_predictions += 1
                else:
                    print(f"❌ WRONG! Predicted: {predicted_letter}, Expected: {letter}")
            else:
                print(f"⚪ NO PREDICTION (no output spikes)")
            
            test_results.append(result)
        
        # Summary
        accuracy = (correct_predictions / len(self.alphabet)) * 100
        print(f"\n{'='*50}")
        print(f"🎯 LETTER RECOGNITION RESULTS")
        print(f"{'='*50}")
        print(f"Letters tested: {len(self.alphabet)}")
        print(f"Correct predictions: {correct_predictions}")
        print(f"Accuracy: {accuracy:.1f}%")
        
        if accuracy > 70:
            print(f"🎉 EXCELLENT! The network is learning to read!")
        elif accuracy > 30:
            print(f"📈 GOOD PROGRESS! Network shows letter discrimination")
        else:
            print(f"📚 LEARNING PHASE! Network needs more training")
        
        return test_results
    
    def demonstrate_english_learning(self):
        """Run the complete English letter learning demonstration"""
        print(f"\n🚀 ENGLISH LEARNING DEMONSTRATION")
        print("=" * 45)
        
        # Show a few letter patterns
        for letter in ['A', 'E', 'I', 'O']:
            if letter in self.letters:
                self.visualize_letter(letter)
        
        # Train the network
        training_results = self.train_letter_recognition(epochs_per_letter=2)
        
        # Test recognition
        test_results = self.test_letter_recognition()
        
        print(f"\n{'='*60}")
        print(f"🔤 ENGLISH NEUROMORPHIC LEARNING STATUS")
        print(f"{'='*60}")
        print(f"✅ Network size: {784 + 128 + 26:,} neurons")
        print(f"✅ Letters implemented: {len(self.alphabet)}")
        print(f"✅ Training completed: {len(self.alphabet) * 2} presentations")
        print(f"✅ STDP learning: Active across all layers")
        print(f"🎯 Next phase: Word recognition and language understanding!")

def main():
    learner = EnglishLetterLearner()
    learner.demonstrate_english_learning()

if __name__ == "__main__":
    main()
