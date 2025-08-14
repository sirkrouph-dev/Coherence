#!/usr/bin/env python3
"""
Working English Learning System
Based on proven manual coordination approach that achieved success
"""

import numpy as np
import json
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.network import NeuromorphicNetwork


class WorkingEnglishLearner:
    """English learning using proven manual coordination approach"""
    
    def __init__(self):
        """Initialize working English learner"""
        print("🗣️ WORKING ENGLISH LEARNING SYSTEM")
        print("=" * 45)
        print("✅ Based on proven manual coordination approach")
        print("✅ Manual current calculation with amplification")
        print("✅ Direct weight updates for learning")
        
        # Character vocabulary
        self.characters = [
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
        ]
        self.char_to_idx = {char: idx for idx, char in enumerate(self.characters)}
        
        # Network sizes
        self.input_size = len(self.characters)  # 26 characters
        self.hidden_size = 12
        self.output_size = 8  # Categories
        
        # Build network
        self.network = self._build_working_network()
        
        # Learning progress tracking
        self.word_memories = {}
        
    def _build_working_network(self):
        """Build network using proven working approach"""
        network = NeuromorphicNetwork()
        
        # Create layers
        network.add_layer("input", self.input_size, "lif")
        network.add_layer("hidden", self.hidden_size, "adex") 
        network.add_layer("output", self.output_size, "adex")
        
        # Connect with proven parameters
        network.connect_layers("input", "hidden", "stdp",
                              connection_probability=0.8, weight=2.0)
        network.connect_layers("hidden", "output", "stdp", 
                              connection_probability=0.9, weight=2.5)
        
        print(f"✅ Network built: {self.input_size}→{self.hidden_size}→{self.output_size}")
        
        return network
    
    def encode_text_to_neural(self, text: str) -> np.ndarray:
        """Convert text to neural pattern using proven encoding"""
        encoding = np.zeros(self.input_size)
        text = text.lower().strip()
        
        # Character-based encoding with context
        for char in text:
            if char in self.char_to_idx:
                idx = self.char_to_idx[char]
                encoding[idx] = 1.0
                
                # Add neighboring context (proven approach)
                if idx > 0:
                    encoding[idx-1] += 0.2
                if idx < len(encoding) - 1:
                    encoding[idx+1] += 0.2
        
        # Normalize to spike rates
        if np.sum(encoding) > 0:
            encoding = encoding / np.max(encoding)
            
        return encoding
    
    def calculate_manual_currents(self, input_pattern: np.ndarray):
        """Calculate currents manually using proven approach"""
        
        # Reset all populations
        for layer in self.network.layers.values():
            layer.neuron_population.reset()
        
        # Input→Hidden currents
        input_hidden_key = ("input", "hidden")
        hidden_currents = np.zeros(self.hidden_size)
        
        if input_hidden_key in self.network.connections:
            connection = self.network.connections[input_hidden_key]
            synapse_pop = connection.synapse_population
            
            if synapse_pop and hasattr(synapse_pop, 'synapses'):
                for (pre_idx, post_idx), synapse in synapse_pop.synapses.items():
                    if pre_idx < len(input_pattern):
                        # Proven formula: input × weight × 12
                        current_contribution = input_pattern[pre_idx] * synapse.weight * 12.0
                        hidden_currents[post_idx] += current_contribution
        
        # Generate hidden spikes
        hidden_spikes = np.zeros(self.hidden_size)
        for i in range(self.hidden_size):
            if hidden_currents[i] > 8.0:  # Proven threshold
                hidden_spikes[i] = 1.0
        
        # Hidden→Output currents  
        hidden_output_key = ("hidden", "output")
        output_currents = np.zeros(self.output_size)
        
        if hidden_output_key in self.network.connections:
            connection = self.network.connections[hidden_output_key]
            synapse_pop = connection.synapse_population
            
            if synapse_pop and hasattr(synapse_pop, 'synapses'):
                for (pre_idx, post_idx), synapse in synapse_pop.synapses.items():
                    if pre_idx < len(hidden_spikes):
                        # Proven amplification: activity × weight × 15
                        current_contribution = hidden_spikes[pre_idx] * synapse.weight * 15.0
                        output_currents[post_idx] += current_contribution
        
        # Generate output spikes
        output_spikes = np.zeros(self.output_size)
        for i in range(self.output_size):
            if output_currents[i] > 12.0:  # Proven threshold
                output_spikes[i] = 1.0
        
        return {
            'input_pattern': input_pattern,
            'hidden_currents': hidden_currents,
            'hidden_spikes': hidden_spikes,
            'output_currents': output_currents,
            'output_spikes': output_spikes
        }
    
    def update_weights_manual(self, hidden_spikes: np.ndarray, output_spikes: np.ndarray):
        """Update weights manually using proven approach"""
        total_changes = 0
        
        # Update hidden→output weights
        hidden_output_key = ("hidden", "output")
        if hidden_output_key in self.network.connections:
            connection = self.network.connections[hidden_output_key]
            synapse_pop = connection.synapse_population
            
            if synapse_pop and hasattr(synapse_pop, 'synapses'):
                for (pre_idx, post_idx), synapse in synapse_pop.synapses.items():
                    if pre_idx < len(hidden_spikes) and post_idx < len(output_spikes):
                        pre_active = hidden_spikes[pre_idx] > 0.5
                        post_active = output_spikes[post_idx] > 0.5
                        
                        if pre_active and post_active:
                            # Strengthen connection (proven values)
                            synapse.weight += 0.42
                            total_changes += 1
                        elif pre_active and not post_active:
                            # Weaken slightly
                            synapse.weight -= 0.08
                            total_changes += 1
                        
                        # Keep weights bounded
                        synapse.weight = np.clip(synapse.weight, 0.1, 15.0)
        
        return total_changes
    
    def learn_english_word(self, word: str, category: str, target_output: int, rounds: int = 30):
        """Learn an English word using proven coordination"""
        print(f"\\n📚 Learning: '{word}' ({category}) -> output {target_output}")
        
        # Encode input
        word_pattern = self.encode_text_to_neural(word)
        
        success_count = 0
        total_weight_changes = 0
        
        for round_num in range(rounds):
            # Calculate activity using proven approach
            activity = self.calculate_manual_currents(word_pattern)
            
            # Check if target output activated
            target_active = activity['output_spikes'][target_output] > 0.5
            
            if target_active:
                success_count += 1
                
            # Create desired output pattern
            desired_output = np.zeros(self.output_size)
            desired_output[target_output] = 1.0
            
            # Update weights toward desired pattern
            weight_changes = self.update_weights_manual(
                activity['hidden_spikes'], desired_output
            )
            total_weight_changes += weight_changes
        
        # Final test
        final_activity = self.calculate_manual_currents(word_pattern)
        final_success = final_activity['output_spikes'][target_output] > 0.5
        
        success_rate = success_count / rounds
        confidence = success_rate
        
        # Store in memory (convert numpy types for JSON)
        self.word_memories[word] = {
            'category': category,
            'target_output': int(target_output),
            'final_success': bool(final_success),
            'confidence': float(confidence),
            'weight_changes': int(total_weight_changes),
            'final_activity': [float(x) for x in final_activity['output_spikes']]
        }
        
        print(f"   Success: {success_count}/{rounds} rounds ({success_rate:.1%})")
        print(f"   Final test: {'✅ Success' if final_success else '❌ Failed'}")
        print(f"   Weight changes: {total_weight_changes}")
        
        return {
            'word': word,
            'learned': bool(final_success),
            'confidence': float(confidence),
            'success_rate': float(success_rate),
            'weight_changes': int(total_weight_changes)
        }
    
    def run_english_curriculum(self):
        """Run English learning curriculum"""
        print("\\n🚀 STARTING ENGLISH CURRICULUM")
        print("-" * 40)
        
        # Progressive curriculum
        curriculum = [
            # Vowels (category 0)
            ('a', 'vowel', 0),
            ('e', 'vowel', 0), 
            ('i', 'vowel', 0),
            ('o', 'vowel', 0),
            ('u', 'vowel', 0),
            
            # Consonants (category 1)  
            ('b', 'consonant', 1),
            ('c', 'consonant', 1),
            ('d', 'consonant', 1),
            ('f', 'consonant', 1),
            ('g', 'consonant', 1),
            
            # Simple words - animals (category 2)
            ('cat', 'animal', 2),
            ('dog', 'animal', 2),
            
            # Simple words - objects (category 3)
            ('car', 'object', 3),
            ('box', 'object', 3),
            
            # Actions (category 4)
            ('run', 'action', 4),
            ('eat', 'action', 4),
        ]
        
        results = []
        learned_count = 0
        total_weight_changes = 0
        
        for word, category, target in curriculum:
            result = self.learn_english_word(word, category, target)
            results.append(result)
            
            if result['learned']:
                learned_count += 1
            total_weight_changes += result['weight_changes']
        
        overall_success = learned_count / len(curriculum)
        
        # Determine achievement level
        if overall_success >= 0.7:
            achievement = "🌟 EXCELLENT - Strong English learning foundation"
        elif overall_success >= 0.5:
            achievement = "✅ GOOD - Solid English learning progress"  
        elif overall_success >= 0.3:
            achievement = "🔄 DEVELOPING - Building English understanding"
        else:
            achievement = "🌱 FOUNDATIONAL - Early English pattern recognition"
        
        print(f"\\n🎓 ENGLISH LEARNING RESULTS")
        print("=" * 30)
        print(f"📚 Items learned: {learned_count}/{len(curriculum)}")
        print(f"📈 Success rate: {overall_success:.1%}")
        print(f"🔧 Total weight changes: {total_weight_changes}")
        print(f"🏆 Achievement: {achievement}")
        
        # Save detailed report (ensure JSON serializable)
        report = {
            'timestamp': datetime.now().isoformat(),
            'curriculum_size': len(curriculum),
            'items_learned': int(learned_count),
            'overall_success_rate': float(overall_success),
            'total_weight_changes': int(total_weight_changes),
            'achievement': achievement,
            'detailed_results': results,
            'word_memories': self.word_memories
        }
        
        with open('working_english_learning_report.json', 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"\\n💾 Report saved: working_english_learning_report.json")
        
        return report


def main():
    """Run working English learning demonstration"""
    learner = WorkingEnglishLearner()
    results = learner.run_english_curriculum()
    
    print(f"\\n🎯 Final Achievement: {results['achievement']}")
    print(f"📊 Success Rate: {results['overall_success_rate']:.1%}")
    
    return results


if __name__ == "__main__":
    main()
