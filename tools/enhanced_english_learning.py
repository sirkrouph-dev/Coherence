#!/usr/bin/env python3
"""
Enhanced Neuromorphic English Learning System

Building on the successful manual synaptic coordination from fixed_learning_system.py,
this system applies proven techniques to English language learning.

Features:
- Manual synaptic current calculation (bypassing network.step() issues)
- Character-to-neural encoding with distributed representations
- Progressive English curriculum (letters → words → phrases)
- STDP plasticity with manual coordination
- Real-time learning assessment and feedback
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

from core.neurons import AdaptiveExponentialIntegrateAndFire, LeakyIntegrateAndFire
from core.synapses import STDP_Synapse
from core.network import NeuromorphicNetwork, NetworkLayer
from core.logging_utils import neuromorphic_logger

def setup_english_logging():
    """Setup enhanced logging for English learning"""
    logger = logging.getLogger('english_learning')
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '[%(levelname)s] %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

class EnhancedEnglishLearner:
    """Enhanced English learning with manual synaptic coordination"""
    
    def __init__(self):
        self.logger = setup_english_logging()
        
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
        self.hidden_size = 20
        self.output_size = 10
        
        # Build the language network
        self.network = self._build_language_network()
        
        # Track learning progress
        self.learning_history = []
        self.word_memories = {}
        
        self.logger.info("🗣️ Enhanced English Learning System Initialized")
        self.logger.info(f"✅ Character vocabulary: {len(self.characters)} characters")
        self.logger.info(f"✅ Network: {self.input_size}→{self.hidden_size}→{self.output_size}")
    
    def _build_language_network(self) -> NeuromorphicNetwork:
        """Build neuromorphic network for English learning with manual coordination"""
        network = NeuromorphicNetwork()
        
        # Input layer - character encoding
        network.add_layer("input", self.input_size, "lif")
        
        # Hidden layer - pattern processing
        network.add_layer("hidden", self.hidden_size, "adex")
        
        # Output layer - understanding/meaning
        network.add_layer("output", self.output_size, "lif")
        
        # Connect layers with STDP synapses
        network.connect_layers("input", "hidden", "stdp", 
                             connection_probability=0.3)
        
        network.connect_layers("hidden", "output", "stdp",
                             connection_probability=0.4)
        
        return network
    
    def encode_text_to_neural(self, text: str) -> np.ndarray:
        """Convert text to neural spike pattern"""
        # Create distributed encoding for each character
        encoding = np.zeros(self.input_size)
        
        # Normalize text
        text = text.lower().strip()
        
        for char in text:
            if char in self.char_to_idx:
                idx = self.char_to_idx[char]
                # Create distributed representation
                encoding[idx] = 1.0
                
                # Add neighboring activations for context
                if idx > 0:
                    encoding[idx-1] += 0.3
                if idx < len(encoding) - 1:
                    encoding[idx+1] += 0.3
        
        # Normalize to spike rates
        if np.sum(encoding) > 0:
            encoding = encoding / np.max(encoding)
        
        return encoding
    
    def calculate_synaptic_currents(self, pre_activity: np.ndarray, layer_name: str) -> Dict[int, float]:
        """Calculate synaptic currents manually (bypassing network.step() issues)"""
        currents = {}
        
        if layer_name == "hidden":
            # Input → Hidden connections
            for post_idx in range(self.hidden_size):
                total_current = 0.0
                
                for pre_idx in range(self.input_size):
                    # Find synapses from input[pre_idx] to hidden[post_idx]
                    input_layer = self.network.layers["input"]
                    hidden_layer = self.network.layers["hidden"]
                    
                    # Check for connections
                    for synapse in input_layer.output_synapses:
                        if (hasattr(synapse, 'pre_neuron_id') and hasattr(synapse, 'post_neuron_id') and
                            synapse.pre_neuron_id == pre_idx and synapse.post_neuron_id == post_idx):
                            
                            # Calculate current: activity * weight * amplification
                            current_contribution = pre_activity[pre_idx] * synapse.weight * 12.0
                            total_current += current_contribution
                
                currents[post_idx] = total_current
        
        elif layer_name == "output":
            # Hidden → Output connections
            hidden_layer = self.network.layers["hidden"]
            hidden_activity = np.array([neuron.membrane_potential for neuron in hidden_layer.neurons])
            
            for post_idx in range(self.output_size):
                total_current = 0.0
                
                for pre_idx in range(self.hidden_size):
                    # Find synapses from hidden[pre_idx] to output[post_idx]
                    output_layer = self.network.layers["output"]
                    
                    for synapse in hidden_layer.output_synapses:
                        if (hasattr(synapse, 'pre_neuron_id') and hasattr(synapse, 'post_neuron_id') and
                            synapse.pre_neuron_id == pre_idx and synapse.post_neuron_id == post_idx):
                            
                            # Calculate current
                            current_contribution = hidden_activity[pre_idx] * synapse.weight * 15.0
                            total_current += current_contribution
                
                currents[post_idx] = total_current
        
        return currents
    
    def apply_manual_stdp(self, pre_spikes: np.ndarray, post_spikes: np.ndarray, 
                         layer_connection: str) -> int:
        """Apply STDP learning rules manually"""
        changes_made = 0
        
        if layer_connection == "input_to_hidden":
            input_layer = self.network.layers["input"]
            
            for synapse in input_layer.output_synapses:
                if hasattr(synapse, 'pre_neuron_id') and hasattr(synapse, 'post_neuron_id'):
                    pre_idx = synapse.pre_neuron_id
                    post_idx = synapse.post_neuron_id
                    
                    if pre_idx < len(pre_spikes) and post_idx < len(post_spikes):
                        # Apply STDP based on spike timing
                        pre_spike = pre_spikes[pre_idx] > 0.5
                        post_spike = post_spikes[post_idx] > 0.5
                        
                        old_weight = synapse.weight
                        
                        if pre_spike and post_spike:
                            # Strengthen connection (LTP)
                            synapse.weight += 0.4  # Strong potentiation
                            changes_made += 1
                        elif pre_spike and not post_spike:
                            # Weaken connection slightly (LTD)
                            synapse.weight -= 0.1
                            changes_made += 1
                        
                        # Keep weights in reasonable bounds
                        synapse.weight = np.clip(synapse.weight, 0.1, 15.0)
        
        elif layer_connection == "hidden_to_output":
            hidden_layer = self.network.layers["hidden"]
            
            for synapse in hidden_layer.output_synapses:
                if hasattr(synapse, 'pre_neuron_id') and hasattr(synapse, 'post_neuron_id'):
                    pre_idx = synapse.pre_neuron_id
                    post_idx = synapse.post_neuron_id
                    
                    if pre_idx < len(pre_spikes) and post_idx < len(post_spikes):
                        pre_spike = pre_spikes[pre_idx] > 0.5
                        post_spike = post_spikes[post_idx] > 0.5
                        
                        old_weight = synapse.weight
                        
                        if pre_spike and post_spike:
                            synapse.weight += 0.5  # Output layer potentiation
                            changes_made += 1
                        elif pre_spike and not post_spike:
                            synapse.weight -= 0.08
                            changes_made += 1
                        
                        synapse.weight = np.clip(synapse.weight, 0.1, 20.0)
        
        return changes_made
    
    def learn_english_word(self, word: str, category: str, rounds: int = 30) -> Dict:
        """Learn an English word with manual coordination"""
        
        self.logger.info(f"📚 LEARNING WORD: '{word}' (category: {category})")
        
        # Encode the word
        word_encoding = self.encode_text_to_neural(word)
        
        learning_success = 0
        total_weight_changes = 0
        confidence_scores = []
        
        for round_num in range(1, rounds + 1):
            # Apply input pattern
            input_layer = self.network.layers["input"]
            for i, neuron in enumerate(input_layer.neurons):
                if i < len(word_encoding):
                    neuron.membrane_potential = word_encoding[i] * 0.8
            
            # Manual forward propagation
            # Step 1: Input → Hidden
            hidden_currents = self.calculate_synaptic_currents(word_encoding, "hidden")
            
            hidden_layer = self.network.layers["hidden"]
            hidden_spikes = np.zeros(self.hidden_size)
            
            for idx, neuron in enumerate(hidden_layer.neurons):
                if idx in hidden_currents:
                    neuron.membrane_potential = hidden_currents[idx] * 0.6
                    
                    # Spike if above threshold
                    if neuron.membrane_potential > 0.7:
                        hidden_spikes[idx] = 1.0
                        neuron.membrane_potential = 0.0  # Reset after spike
            
            # Step 2: Hidden → Output
            output_currents = self.calculate_synaptic_currents(hidden_spikes, "output")
            
            output_layer = self.network.layers["output"]
            output_spikes = np.zeros(self.output_size)
            
            for idx, neuron in enumerate(output_layer.neurons):
                if idx in output_currents:
                    neuron.membrane_potential = output_currents[idx] * 0.5
                    
                    if neuron.membrane_potential > 0.6:
                        output_spikes[idx] = 1.0
                        neuron.membrane_potential = 0.0
            
            # Calculate understanding confidence
            output_activity = np.sum(output_spikes)
            hidden_activity = np.sum(hidden_spikes)
            
            # Confidence based on coordinated activity
            if output_activity > 0 and hidden_activity > 0:
                confidence = min(1.0, (output_activity * hidden_activity) / (self.output_size * 0.5))
            else:
                confidence = 0.0
            
            confidence_scores.append(confidence)
            
            # Apply learning (STDP)
            input_spikes = word_encoding > 0.5
            
            changes_1 = self.apply_manual_stdp(input_spikes, hidden_spikes, "input_to_hidden")
            changes_2 = self.apply_manual_stdp(hidden_spikes, output_spikes, "hidden_to_output")
            
            total_weight_changes += changes_1 + changes_2
            
            # Check for learning success
            if confidence > 0.7:
                learning_success += 1
            
            # Progress report every 10 rounds
            if round_num % 10 == 0:
                status = "✅" if confidence > 0.7 else "❌"
                self.logger.info(f"  Round {round_num}: {status} Confidence = {confidence:.2f}")
        
        # Final assessment
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        final_confidence = confidence_scores[-1] if confidence_scores else 0.0
        
        success_rate = learning_success / rounds
        learning_status = "✅ LEARNED" if success_rate > 0.3 else "❌ NEEDS PRACTICE"
        
        # Store word memory
        self.word_memories[word] = {
            'confidence': final_confidence,
            'success_rate': success_rate,
            'category': category,
            'weight_changes': total_weight_changes
        }
        
        result = {
            'word': word,
            'category': category,
            'rounds': rounds,
            'success_rate': success_rate,
            'avg_confidence': avg_confidence,
            'final_confidence': final_confidence,
            'weight_changes': total_weight_changes,
            'status': learning_status
        }
        
        self.logger.info(f"  📊 Success: {learning_success}/{rounds} rounds")
        self.logger.info(f"  🧠 Final confidence: {learning_status}")
        self.logger.info(f"  ⚡ Weight changes: {total_weight_changes}")
        
        return result
    
    def run_english_curriculum(self):
        """Run comprehensive English learning curriculum"""
        
        print("\n🗣️ ENHANCED NEUROMORPHIC ENGLISH LEARNING")
        print("=" * 55)
        print("Building on successful manual synaptic coordination")
        print("✅ Manual current calculation (bypassing network issues)")
        print("✅ Progressive English curriculum")
        print("✅ Real-time learning assessment")
        
        # Progressive English curriculum
        curriculum = [
            # Stage 1: Letters
            ('a', 'vowel'), ('e', 'vowel'), ('i', 'vowel'), ('o', 'vowel'),
            ('b', 'consonant'), ('c', 'consonant'), ('d', 'consonant'),
            
            # Stage 2: Simple words
            ('cat', 'animal'), ('dog', 'animal'), ('car', 'object'), 
            ('red', 'color'), ('big', 'size'),
            
            # Stage 3: Common words
            ('the', 'article'), ('and', 'conjunction'), ('can', 'modal'), 
            ('see', 'action'), ('run', 'action'),
            
            # Stage 4: Basic phrases (simplified)
            ('i am', 'identity'), ('you are', 'identity'), ('can see', 'ability')
        ]
        
        learning_results = []
        stage_names = ["Individual letters", "Simple words", "Common words", "Basic phrases"]
        stage_boundaries = [7, 12, 17, 20]
        current_stage = 0
        
        for lesson_num, (item, category) in enumerate(curriculum, 1):
            # Check for stage transition
            if current_stage < len(stage_boundaries) and lesson_num > stage_boundaries[current_stage]:
                current_stage += 1
            
            if current_stage < len(stage_names):
                if lesson_num == 1 or lesson_num in [stage_boundaries[i]+1 for i in range(len(stage_boundaries))]:
                    print(f"\n🎓 STAGE: {stage_names[current_stage]}")
                    print("-" * 50)
            
            result = self.learn_english_word(item, category, rounds=20)
            learning_results.append(result)
        
        # Final assessment
        print(f"\n🗣️ ENGLISH LEARNING ASSESSMENT")
        print("=" * 40)
        
        learned_count = 0
        total_confidence = 0
        total_weight_changes = 0
        
        for result in learning_results:
            status_symbol = "✅" if "LEARNED" in result['status'] else "❌"
            print(f"{status_symbol} '{result['word']}': {result['final_confidence']:.2f} confidence "
                  f"({result['success_rate']:.1%} success)")
            
            if "LEARNED" in result['status']:
                learned_count += 1
            
            total_confidence += result['final_confidence']
            total_weight_changes += result['weight_changes']
        
        # Summary statistics
        overall_success = learned_count / len(learning_results)
        avg_confidence = total_confidence / len(learning_results)
        
        print(f"\n📊 ENGLISH LEARNING RESULTS")
        print("-" * 30)
        print(f"Words/phrases learned: {learned_count}/{len(learning_results)} ({overall_success:.1%})")
        print(f"Average confidence: {avg_confidence:.3f}")
        print(f"Total weight changes: {total_weight_changes}")
        
        # Learning achievement level
        if overall_success >= 0.7:
            achievement = "🌟 EXCELLENT - Strong English learning foundation"
        elif overall_success >= 0.5:
            achievement = "✅ GOOD - Making progress with English"
        elif overall_success >= 0.3:
            achievement = "🔄 DEVELOPING - Beginning to understand English"
        else:
            achievement = "🌱 FOUNDATIONAL - Building English recognition"
        
        print(f"\n🌱 ENGLISH LEARNING: {achievement}")
        
        # Save detailed report
        report = {
            'timestamp': datetime.now().isoformat(),
            'curriculum_items': len(learning_results),
            'words_learned': learned_count,
            'overall_success_rate': overall_success,
            'average_confidence': avg_confidence,
            'total_weight_changes': total_weight_changes,
            'achievement_level': achievement,
            'detailed_results': learning_results,
            'word_memories': self.word_memories
        }
        
        with open('enhanced_english_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"💾 Enhanced English report saved: enhanced_english_report.json")
        
        return report

def main():
    """Run enhanced English learning demonstration"""
    
    # Check for GPU acceleration
    try:
        import cupy as cp
        print("[OK] CuPy GPU acceleration available for neuromorphic computing")
    except ImportError:
        print("[INFO] CuPy not available, using CPU processing")
    
    try:
        import torch
        print("[OK] PyTorch acceleration available")
    except ImportError:
        print("[INFO] PyTorch not available")
    
    # Initialize enhanced English learning system
    learner = EnhancedEnglishLearner()
    
    # Run the English curriculum
    results = learner.run_english_curriculum()
    
    # Show key achievements
    print(f"\n📚 English learning complete!")
    print(f"Achievement: {results['achievement_level']}")
    
    return results

if __name__ == "__main__":
    main()
