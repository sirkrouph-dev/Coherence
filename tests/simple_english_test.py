#!/usr/bin/env python3
"""
Simple English Learning Test
Testing basic functionality of English learning in neuromorphic system
"""

import numpy as np
import json
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.network import NeuromorphicNetwork


class SimpleEnglishTest:
    """Simple test of English learning"""
    
    def __init__(self):
        """Initialize simple English test"""
        print("🧪 SIMPLE ENGLISH LEARNING TEST")
        print("=" * 40)
        
        # Create simple 3-layer network
        self.network = NeuromorphicNetwork()
        self.network.add_layer("input", 26, "lif")    # 26 letters
        self.network.add_layer("hidden", 16, "adex")   # Hidden processing  
        self.network.add_layer("output", 8, "adex")    # Semantic categories
        
        # Connect layers
        self.network.connect_layers("input", "hidden", "stdp",
                                   connection_probability=0.5, weight=1.5)
        self.network.connect_layers("hidden", "output", "stdp", 
                                   connection_probability=0.6, weight=2.0)
        
        print(f"✅ Network created: {len(self.network.layers)} layers")
        
        # Simple character encoding
        self.chars = "abcdefghijklmnopqrstuvwxyz"
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        
    def encode_letter(self, letter: str) -> np.ndarray:
        """Encode a single letter"""
        encoding = np.zeros(26)
        if letter.lower() in self.char_to_idx:
            idx = self.char_to_idx[letter.lower()]
            encoding[idx] = 1.0
            # Add some neighboring activation
            if idx > 0:
                encoding[idx-1] = 0.3
            if idx < 25:
                encoding[idx+1] = 0.3
        return encoding
        
    def test_letter_learning(self, letter: str, target_category: int, rounds: int = 20):
        """Test learning of a single letter"""
        print(f"\\n📖 Testing letter '{letter}' -> category {target_category}")
        
        encoding = self.encode_letter(letter)
        successful_rounds = 0
        
        for round_num in range(rounds):
            # Reset network
            for layer in self.network.layers.values():
                layer.neuron_population.reset()
            
            # Apply input stimulation
            input_currents = [40.0 if e > 0.5 else 0.0 for e in encoding]
            output_activity = np.zeros(8)
            
            # Run simulation
            for step in range(20):
                # Stimulate input layer
                input_spikes = self.network.layers["input"].neuron_population.step(0.1, input_currents)
                
                # Let network process
                self.network.step(0.1)
                
                # Collect output activity  
                output_spikes = self.network.layers["output"].neuron_population.get_spike_states()
                for i, spike in enumerate(output_spikes):
                    if spike:
                        output_activity[i] += 1
            
            # Check if target category activated most
            if len(output_activity) > target_category and output_activity[target_category] > 0:
                max_activity = np.max(output_activity)
                if output_activity[target_category] == max_activity:
                    successful_rounds += 1
                    
        success_rate = successful_rounds / rounds
        print(f"   Success: {successful_rounds}/{rounds} rounds ({success_rate:.1%})")
        
        return success_rate > 0.3  # Consider learned if >30% success
        
    def run_simple_curriculum(self):
        """Run simple learning curriculum"""
        print("\\n🚀 RUNNING SIMPLE CURRICULUM")
        print("-" * 30)
        
        # Simple vowel/consonant categorization
        curriculum = [
            ('a', 0),  # vowel category
            ('e', 0),  # vowel category  
            ('i', 0),  # vowel category
            ('b', 1),  # consonant category
            ('c', 1),  # consonant category
            ('d', 1),  # consonant category
        ]
        
        results = []
        learned_count = 0
        
        for letter, category in curriculum:
            learned = self.test_letter_learning(letter, category)
            results.append({
                'letter': letter,
                'category': category,
                'learned': learned
            })
            if learned:
                learned_count += 1
                
        overall_success = learned_count / len(curriculum)
        
        print(f"\\n📊 RESULTS:")
        print(f"   Letters learned: {learned_count}/{len(curriculum)}")
        print(f"   Success rate: {overall_success:.1%}")
        
        if overall_success > 0.5:
            print("✅ GOOD: Basic letter learning working!")
        elif overall_success > 0.2:
            print("🟡 PARTIAL: Some learning detected")
        else:
            print("❌ NEEDS WORK: No significant learning")
            
        return {
            'success_rate': overall_success,
            'results': results,
            'learned_count': learned_count,
            'total_tested': len(curriculum)
        }


def main():
    """Run simple English learning test"""
    tester = SimpleEnglishTest()
    results = tester.run_simple_curriculum()
    
    # Save results
    with open('simple_english_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\\n💾 Results saved to: simple_english_test_results.json")
    return results


if __name__ == "__main__":
    main()
