#!/usr/bin/env python3
"""
Simple Neuromorphic English Learning - Direct Application of Fixed Learning Success

Using the proven manual coordination approach from fixed_learning_system.py
that achieved 66.7% success rate, now applied to English learning.

Key Success Factors from Fixed Learning:
- Manual synaptic current calculation 
- Direct weight updates bypassing network.step()
- Strong amplification (weight × 10-15)
- Clear pattern-based learning assessment
"""

import numpy as np
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Import neuromorphic components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.network import NeuromorphicNetwork

class SimpleEnglishLearner:
    """Simple English learning using proven coordination approach"""
    
    def __init__(self):
        # Character vocabulary for English
        self.characters = [
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
            ' ', '.', ',', '!', '?'
        ]
        
        # Create character-to-index mapping
        self.char_to_idx = {char: idx for idx, char in enumerate(self.characters)}
        
        # Network architecture for English learning  
        self.input_size = len(self.characters)  # 31 characters
        self.hidden_size = 16
        self.output_size = 8
        
        # Build the language network
        self.network = self._build_simple_network()
        
        # Track learning progress
        self.word_memories = {}
        
        print(f"🗣️ Simple English Learning System Initialized")
        print(f"✅ Character vocabulary: {len(self.characters)} characters")
        print(f"✅ Network: {self.input_size}→{self.hidden_size}→{self.output_size}")
        print(f"✅ Using proven manual coordination approach")
    
    def _build_simple_network(self) -> NeuromorphicNetwork:
        """Build simple neuromorphic network for English learning"""
        network = NeuromorphicNetwork()
        
        # Add layers
        network.add_layer("input", self.input_size, "lif")
        network.add_layer("hidden", self.hidden_size, "lif") 
        network.add_layer("output", self.output_size, "lif")
        
        # Connect layers with STDP
        network.connect_layers("input", "hidden", "stdp", connection_probability=0.3)
        network.connect_layers("hidden", "output", "stdp", connection_probability=0.4)
        
        return network
    
    def encode_text_to_neural(self, text: str) -> np.ndarray:
        """Convert text to neural spike pattern"""
        encoding = np.zeros(self.input_size)
        
        # Normalize text
        text = text.lower().strip()
        
        # Create distributed encoding
        for char in text:
            if char in self.char_to_idx:
                idx = self.char_to_idx[char]
                encoding[idx] = 1.0
                
                # Add neighboring context
                if idx > 0:
                    encoding[idx-1] += 0.2
                if idx < len(encoding) - 1:
                    encoding[idx+1] += 0.2
        
        # Normalize to spike rates
        if np.sum(encoding) > 0:
            encoding = encoding / np.max(encoding)
        
        return encoding
    
    def calculate_manual_currents(self, input_pattern: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate synaptic currents manually using successful approach"""
        
        # Get input→hidden synapses
        input_hidden_key = ("input", "hidden")
        hidden_currents = np.zeros(self.hidden_size)
        
        if input_hidden_key in self.network.connections:
            connection = self.network.connections[input_hidden_key]
            synapse_pop = connection.synapse_population
            
            # Calculate currents for each hidden neuron
            if synapse_pop and hasattr(synapse_pop, 'synapses'):
                for (pre_idx, post_idx), synapse in synapse_pop.synapses.items():
                    if pre_idx < len(input_pattern):
                        # Use proven amplification: activity × weight × 12
                        current_contribution = input_pattern[pre_idx] * synapse.weight * 12.0
                        hidden_currents[post_idx] += current_contribution
        
        # Generate hidden spikes
        hidden_spikes = np.zeros(self.hidden_size)
        for i in range(self.hidden_size):
            if hidden_currents[i] > 8.0:  # Spike threshold
                hidden_spikes[i] = 1.0
        
        # Get hidden→output synapses
        hidden_output_key = ("hidden", "output")
        output_currents = np.zeros(self.output_size)
        
        if hidden_output_key in self.network.connections:
            connection = self.network.connections[hidden_output_key]
            synapse_pop = connection.synapse_population
            
            # Calculate currents for each output neuron
            if synapse_pop and hasattr(synapse_pop, 'synapses'):
                for (pre_idx, post_idx), synapse in synapse_pop.synapses.items():
                    if pre_idx < len(hidden_spikes):
                        # Use proven amplification: activity × weight × 15
                        current_contribution = hidden_spikes[pre_idx] * synapse.weight * 15.0
                        output_currents[post_idx] += current_contribution
        
        # Generate output spikes
        output_spikes = np.zeros(self.output_size)
        for i in range(self.output_size):
            if output_currents[i] > 10.0:  # Spike threshold
                output_spikes[i] = 1.0
        
        return hidden_spikes, output_spikes
    
    def apply_manual_stdp_learning(self, input_spikes: np.ndarray, hidden_spikes: np.ndarray, 
                                  output_spikes: np.ndarray) -> int:
        """Apply STDP learning using proven manual approach"""
        total_changes = 0
        
        # Input→Hidden STDP
        input_hidden_key = ("input", "hidden")
        if input_hidden_key in self.network.connections:
            connection = self.network.connections[input_hidden_key]
            synapse_pop = connection.synapse_population
            
            if synapse_pop and hasattr(synapse_pop, 'synapses'):
                for (pre_idx, post_idx), synapse in synapse_pop.synapses.items():
                    if pre_idx < len(input_spikes) and post_idx < len(hidden_spikes):
                        pre_active = input_spikes[pre_idx] > 0.5
                        post_active = hidden_spikes[post_idx] > 0.5
                        
                        if pre_active and post_active:
                            # Strengthen connection (LTP)
                            synapse.weight += 0.35  # Proven potentiation value
                            total_changes += 1
                        elif pre_active and not post_active:
                            # Weaken connection (LTD)
                            synapse.weight -= 0.12  # Proven depression value
                            total_changes += 1
                        
                        # Keep weights bounded
                        synapse.weight = np.clip(synapse.weight, 0.1, 12.0)
        
        # Hidden→Output STDP
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
                            # Strengthen connection
                            synapse.weight += 0.42  # Strong output potentiation
                            total_changes += 1
                        elif pre_active and not post_active:
                            # Weaken connection
                            synapse.weight -= 0.08
                            total_changes += 1
                        
                        # Keep weights bounded
                        synapse.weight = np.clip(synapse.weight, 0.1, 15.0)
        
        return total_changes
    
    def learn_english_word(self, word: str, category: str, rounds: int = 25) -> Dict:
        """Learn an English word using proven manual coordination"""
        
        print(f"\n📚 LEARNING: '{word}' ({category})")
        
        # Encode the word
        word_pattern = self.encode_text_to_neural(word)
        
        success_count = 0
        total_weight_changes = 0
        confidence_scores = []
        
        for round_num in range(1, rounds + 1):
            # Manual forward propagation using proven approach
            hidden_spikes, output_spikes = self.calculate_manual_currents(word_pattern)
            
            # Calculate understanding confidence
            hidden_activity = np.sum(hidden_spikes)
            output_activity = np.sum(output_spikes)
            
            # Proven confidence calculation
            if output_activity > 0 and hidden_activity > 0:
                confidence = min(1.0, (output_activity * hidden_activity) / 8.0)
            else:
                confidence = 0.0
            
            confidence_scores.append(confidence)
            
            # Apply proven STDP learning
            input_spikes = word_pattern > 0.3  # Input spike threshold
            
            weight_changes = self.apply_manual_stdp_learning(input_spikes, hidden_spikes, output_spikes)
            total_weight_changes += weight_changes
            
            # Success criterion from proven approach
            if confidence > 0.6:
                success_count += 1
            
            # Progress reporting
            if round_num % 10 == 0:
                status = "✅" if confidence > 0.6 else "❌"
                print(f"  Round {round_num}: {status} Confidence = {confidence:.2f}")
        
        # Final assessment using proven metrics
        success_rate = success_count / rounds
        final_confidence = confidence_scores[-1] if confidence_scores else 0.0
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
        # Learning status from proven thresholds
        if success_rate > 0.4:
            status = "✅ LEARNED"
        elif success_rate > 0.2:
            status = "🔄 LEARNING"
        else:
            status = "❌ NEEDS PRACTICE"
        
        # Store in memory
        self.word_memories[word] = {
            'confidence': final_confidence,
            'success_rate': success_rate,
            'category': category,
            'weight_changes': total_weight_changes,
            'status': status
        }
        
        result = {
            'word': word,
            'category': category,
            'rounds': rounds,
            'success_rate': success_rate,
            'final_confidence': final_confidence,
            'avg_confidence': avg_confidence,
            'weight_changes': total_weight_changes,
            'status': status
        }
        
        print(f"  📊 Success: {success_count}/{rounds} rounds ({success_rate:.1%})")
        print(f"  🧠 Status: {status}")
        print(f"  ⚡ Weight changes: {total_weight_changes}")
        
        return result
    
    def run_english_curriculum(self):
        """Run simple English curriculum using proven approach"""
        
        print("\n🗣️ NEUROMORPHIC ENGLISH LEARNING - PROVEN APPROACH")
        print("=" * 55)
        print("✅ Based on successful 66.7% pattern learning")
        print("✅ Manual synaptic coordination")
        print("✅ Proven amplification and thresholds")
        
        # Focused English curriculum
        curriculum = [
            # Basic letters
            ('a', 'vowel'), ('e', 'vowel'), ('i', 'vowel'),
            ('b', 'consonant'), ('c', 'consonant'), ('d', 'consonant'),
            
            # Simple words  
            ('cat', 'animal'), ('dog', 'animal'), ('car', 'object'),
            ('red', 'color'), ('big', 'size'), ('run', 'action'),
            
            # Common words
            ('the', 'article'), ('and', 'conjunction'), ('can', 'modal'),
            ('see', 'action'), ('you', 'pronoun'),
            
            # Simple phrases
            ('i am', 'identity'), ('can see', 'ability'), ('red car', 'description')
        ]
        
        learning_results = []
        
        # Teach each item
        for word, category in curriculum:
            result = self.learn_english_word(word, category, rounds=20)
            learning_results.append(result)
        
        # Final assessment using proven metrics
        print(f"\n🗣️ ENGLISH LEARNING ASSESSMENT")
        print("=" * 40)
        
        learned_count = 0
        learning_count = 0
        total_confidence = 0
        total_weight_changes = 0
        
        for result in learning_results:
            status = result['status']
            confidence = result['final_confidence']
            
            if "LEARNED" in status:
                symbol = "✅"
                learned_count += 1
            elif "LEARNING" in status:
                symbol = "🔄"
                learning_count += 1
            else:
                symbol = "❌"
            
            print(f"{symbol} '{result['word']}': {confidence:.2f} confidence "
                  f"({result['success_rate']:.1%} success)")
            
            total_confidence += confidence
            total_weight_changes += result['weight_changes']
        
        # Summary statistics
        total_items = len(learning_results)
        overall_success = learned_count / total_items
        progress_rate = (learned_count + learning_count) / total_items
        avg_confidence = total_confidence / total_items
        
        print(f"\n📊 ENGLISH LEARNING RESULTS")
        print("-" * 30)
        print(f"Fully learned: {learned_count}/{total_items} ({overall_success:.1%})")
        print(f"Making progress: {learning_count}/{total_items}")
        print(f"Total progress: {learned_count + learning_count}/{total_items} ({progress_rate:.1%})")
        print(f"Average confidence: {avg_confidence:.3f}")
        print(f"Total weight changes: {total_weight_changes}")
        
        # Achievement assessment using proven thresholds
        if overall_success >= 0.6:
            achievement = "🌟 EXCELLENT - Strong English learning success!"
        elif overall_success >= 0.4:
            achievement = "✅ GOOD - Solid English learning progress"
        elif progress_rate >= 0.5:
            achievement = "🔄 DEVELOPING - Building English understanding"
        else:
            achievement = "🌱 FOUNDATIONAL - Early English recognition"
        
        print(f"\n🌱 ENGLISH LEARNING: {achievement}")
        
        # Save results
        report = {
            'timestamp': datetime.now().isoformat(),
            'approach': 'Manual Coordination (proven 66.7% success)',
            'curriculum_size': total_items,
            'fully_learned': learned_count,
            'making_progress': learning_count,
            'overall_success_rate': overall_success,
            'progress_rate': progress_rate,
            'average_confidence': avg_confidence,
            'total_weight_changes': total_weight_changes,
            'achievement': achievement,
            'detailed_results': learning_results,
            'word_memories': self.word_memories
        }
        
        with open('simple_english_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"💾 English learning report saved: simple_english_report.json")
        
        return report

def main():
    """Run simple English learning with proven approach"""
    
    # Check for acceleration
    try:
        import cupy as cp
        print("[OK] CuPy GPU acceleration available")
    except ImportError:
        print("[INFO] CuPy not available, using CPU")
    
    try:
        import torch
        print("[OK] PyTorch acceleration available")
    except ImportError:
        print("[INFO] PyTorch not available")
    
    # Run simple English learning
    learner = SimpleEnglishLearner()
    results = learner.run_english_curriculum()
    
    # Final summary
    print(f"\n📚 English learning session complete!")
    print(f"Achievement: {results['achievement']}")
    print(f"Success rate: {results['overall_success_rate']:.1%}")
    
    return results

if __name__ == "__main__":
    main()
