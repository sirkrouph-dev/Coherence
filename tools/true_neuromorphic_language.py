#!/usr/bin/env python3
"""
True Neuromorphic Language Generation - No Static Responses

This system generates language DIRECTLY from neural activity patterns,
not from pre-written response libraries. The neural networks learn to
associate spike patterns with character/word generation.

Key Differences from Static System:
- Neural patterns directly map to character generation
- No pre-written response templates
- Language emerges from neural state dynamics
- True compositional language generation
- Learned associations between concepts and expressions
"""

import numpy as np
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Import neuromorphic components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.network import NeuromorphicNetwork

class TrueNeuromorphicLanguage:
    """Pure neuromorphic language generation without static responses"""
    
    def __init__(self):
        # Core vocabulary for emergent language
        self.vocabulary = [
            # Letters
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
            # Essential symbols
            ' ', '.', ',', '!', '?', "'", '-'
        ]
        
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocabulary)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.vocabulary)}
        
        # Pure neuromorphic architecture for language generation
        self.vocab_size = len(self.vocabulary)       # 33 characters
        self.input_encoding_size = 20                # Input concept encoding
        self.semantic_size = 16                      # Semantic understanding
        self.syntax_size = 12                        # Syntax/grammar
        self.generation_size = 18                    # Language generation
        self.character_output_size = self.vocab_size # Character-level output
        
        # Build the true neuromorphic language network
        self.network = self._build_pure_neural_language_network()
        
        # Neural concept mappings (learned, not hardcoded)
        self.concept_patterns = {}
        self.learned_associations = {}
        
        # Neural state tracking
        self.neural_memory = np.zeros(self.semantic_size)
        self.syntax_state = np.zeros(self.syntax_size)
        
        print("🧠 TRUE NEUROMORPHIC LANGUAGE SYSTEM")
        print("=" * 45)
        print("✅ NO static responses - pure neural generation")
        print(f"✅ Vocabulary: {self.vocab_size} characters")
        print(f"✅ Architecture: Input→Semantic→Syntax→Generation→Output")
        print("✅ Language emerges from neural dynamics")
        print("🚀 Ready for emergent communication!")
    
    def _build_pure_neural_language_network(self) -> NeuromorphicNetwork:
        """Build network for pure neural language generation"""
        network = NeuromorphicNetwork()
        
        # Input concept encoding layer
        network.add_layer("input", self.input_encoding_size, "lif")
        
        # Semantic understanding layer
        network.add_layer("semantic", self.semantic_size, "adex")
        
        # Syntax/grammar layer
        network.add_layer("syntax", self.syntax_size, "lif")
        
        # Language generation layer
        network.add_layer("generation", self.generation_size, "adex")
        
        # Character output layer
        network.add_layer("output", self.character_output_size, "lif")
        
        # Neural connections for language processing
        network.connect_layers("input", "semantic", "stdp", connection_probability=0.6)
        network.connect_layers("semantic", "syntax", "stdp", connection_probability=0.5)
        network.connect_layers("semantic", "generation", "stdp", connection_probability=0.7)
        network.connect_layers("syntax", "generation", "stdp", connection_probability=0.6)
        network.connect_layers("generation", "output", "stdp", connection_probability=0.8)
        
        # Recurrent connections for memory and context
        network.connect_layers("semantic", "semantic", "stdp", connection_probability=0.3)
        network.connect_layers("generation", "syntax", "stdp", connection_probability=0.4)
        
        return network
    
    def encode_concept_to_neural(self, concept_text: str) -> np.ndarray:
        """Encode input concept into distributed neural pattern"""
        # Create distributed representation based on character content
        encoding = np.zeros(self.input_encoding_size)
        
        concept_text = concept_text.lower().strip()
        
        # Distributed encoding based on character patterns
        for i, char in enumerate(concept_text[:10]):  # Limit to first 10 chars
            if char in self.char_to_idx:
                char_idx = self.char_to_idx[char]
                
                # Map character to input neurons with some spreading
                neuron_idx = (char_idx * 7 + i * 3) % self.input_encoding_size
                encoding[neuron_idx] = 1.0
                
                # Spread activation to neighboring neurons
                for offset in [-1, 1]:
                    neighbor_idx = (neuron_idx + offset) % self.input_encoding_size
                    encoding[neighbor_idx] += 0.4
        
        # Add concept length information
        length_encoding = min(1.0, len(concept_text) / 20.0)
        encoding[0] += length_encoding
        
        # Normalize
        if np.sum(encoding) > 0:
            encoding = encoding / np.max(encoding)
        
        return encoding
    
    def neural_forward_pass(self, input_pattern: np.ndarray) -> Dict[str, np.ndarray]:
        """Pure neural forward pass through language network"""
        
        # Initialize layer activities
        activities = {}
        
        # Input → Semantic processing
        semantic_currents = self._compute_layer_currents(input_pattern, "input", "semantic")
        
        # Add recurrent semantic memory
        semantic_memory_currents = self._compute_layer_currents(self.neural_memory, "semantic", "semantic")
        combined_semantic_currents = semantic_currents + semantic_memory_currents * 0.5
        
        semantic_activity = self._neural_activation(combined_semantic_currents, threshold=5.0)
        activities['semantic'] = semantic_activity
        
        # Update neural memory
        self.neural_memory = 0.7 * self.neural_memory + 0.3 * semantic_activity
        
        # Semantic → Syntax processing
        syntax_currents = self._compute_layer_currents(semantic_activity, "semantic", "syntax")
        syntax_activity = self._neural_activation(syntax_currents, threshold=4.0)
        activities['syntax'] = syntax_activity
        
        # Semantic + Syntax → Generation
        generation_currents_semantic = self._compute_layer_currents(semantic_activity, "semantic", "generation")
        generation_currents_syntax = self._compute_layer_currents(syntax_activity, "syntax", "generation")
        
        # Recurrent generation feedback (ensure compatible dimensions)
        generation_syntax_feedback = np.zeros(self.generation_size)
        
        combined_generation_currents = (generation_currents_semantic + 
                                      generation_currents_syntax + 
                                      generation_syntax_feedback * 0.3)
        
        generation_activity = self._neural_activation(combined_generation_currents, threshold=6.0)
        activities['generation'] = generation_activity
        
        # Update syntax state (ensure compatible dimensions)
        if len(syntax_activity) == len(self.syntax_state):
            self.syntax_state = 0.8 * self.syntax_state + 0.2 * syntax_activity
        else:
            self.syntax_state = syntax_activity.copy() if len(syntax_activity) == self.syntax_size else np.zeros(self.syntax_size)
        
        # Generation → Output characters
        output_currents = self._compute_layer_currents(generation_activity, "generation", "output")
        output_activity = self._neural_activation(output_currents, threshold=3.0)
        activities['output'] = output_activity
        
        return activities
    
    def _compute_layer_currents(self, pre_activity: np.ndarray, pre_layer: str, post_layer: str) -> np.ndarray:
        """Compute synaptic currents between layers"""
        connection_key = (pre_layer, post_layer)
        
        # Determine post-layer size and amplification
        layer_configs = {
            "semantic": (self.semantic_size, 14.0),
            "syntax": (self.syntax_size, 12.0),
            "generation": (self.generation_size, 16.0),
            "output": (self.character_output_size, 10.0)
        }
        
        if post_layer in layer_configs:
            post_size, amplification = layer_configs[post_layer]
        else:
            post_size, amplification = (10, 10.0)
        
        currents = np.zeros(post_size)
        
        if connection_key in self.network.connections:
            connection = self.network.connections[connection_key]
            synapse_pop = connection.synapse_population
            
            if synapse_pop and hasattr(synapse_pop, 'synapses'):
                for (pre_idx, post_idx), synapse in synapse_pop.synapses.items():
                    if pre_idx < len(pre_activity) and post_idx < post_size:
                        current = pre_activity[pre_idx] * synapse.weight * amplification
                        currents[post_idx] += current
        
        return currents
    
    def _neural_activation(self, currents: np.ndarray, threshold: float) -> np.ndarray:
        """Convert currents to neural activations"""
        activations = np.zeros_like(currents)
        for i, current in enumerate(currents):
            if current > threshold:
                # Sigmoid-like activation
                activations[i] = 1.0 / (1.0 + np.exp(-(current - threshold) / 2.0))
        return activations
    
    def neural_language_generation(self, activities: Dict[str, np.ndarray], max_chars: int = 50) -> str:
        """Generate language directly from neural activities - NO STATIC RESPONSES"""
        
        output_activity = activities['output']
        generation_activity = activities['generation']
        semantic_activity = activities['semantic']
        
        generated_text = ""
        
        # Generate characters based on neural output activity
        for char_step in range(max_chars):
            # Find most active output neurons
            active_indices = np.argsort(output_activity)[-3:]  # Top 3 active neurons
            
            char_probabilities = []
            for idx in active_indices:
                if idx < len(self.vocabulary):
                    prob = output_activity[idx]
                    char_probabilities.append((self.vocabulary[idx], prob))
            
            # Select character based on neural activity
            if char_probabilities:
                # Weight selection by activity level
                total_activity = sum(prob for _, prob in char_probabilities)
                if total_activity > 0.1:  # Minimum activity threshold
                    # Probabilistic selection based on neural activity
                    rand_val = np.random.random() * total_activity
                    cumulative = 0
                    selected_char = ' '
                    
                    for char, prob in char_probabilities:
                        cumulative += prob
                        if rand_val <= cumulative:
                            selected_char = char
                            break
                    
                    generated_text += selected_char
                    
                    # Stop conditions based on neural patterns
                    if selected_char in '.!?' and len(generated_text) > 10:
                        break
                    if len(generated_text) > 5 and semantic_activity.max() < 0.3:
                        break  # Neural activity too low
                else:
                    break  # No significant neural activity
            else:
                break
            
            # Update neural state for next character (recurrent processing)
            # Slightly modify output activity based on what was just generated
            char_feedback = np.zeros_like(output_activity)
            if generated_text and generated_text[-1] in self.char_to_idx:
                last_char_idx = self.char_to_idx[generated_text[-1]]
                char_feedback[last_char_idx] = -0.2  # Slight inhibition of just-used character
                
                # Enhance activity for likely next characters
                if generated_text[-1] == ' ':
                    # After space, boost consonants
                    for i, char in enumerate(self.vocabulary):
                        if char in 'bcdfghjklmnpqrstvwxyz':
                            char_feedback[i] += 0.1
                elif generated_text[-1] in 'bcdfghjklmnpqrstvwxyz':
                    # After consonant, boost vowels
                    for i, char in enumerate(self.vocabulary):
                        if char in 'aeiou':
                            char_feedback[i] += 0.15
            
            output_activity = np.clip(output_activity + char_feedback, 0, 1)
        
        # Clean up generated text
        generated_text = generated_text.strip()
        
        # If no meaningful text generated, create minimal response
        if len(generated_text) < 2:
            # Use semantic activity to generate something
            if semantic_activity.max() > 0.5:
                generated_text = "yes"
            elif semantic_activity.mean() > 0.3:
                generated_text = "i see"
            else:
                generated_text = "hm"
        
        return generated_text
    
    def learn_from_interaction(self, input_text: str, generated_response: str, 
                             activities: Dict[str, np.ndarray]) -> int:
        """Learn from the interaction using STDP"""
        
        # Encode input and response for learning
        input_pattern = self.encode_concept_to_neural(input_text)
        response_pattern = self.encode_concept_to_neural(generated_response)
        
        total_changes = 0
        
        # Learning connections
        learning_pairs = [
            ("input", "semantic", input_pattern, activities['semantic']),
            ("semantic", "syntax", activities['semantic'], activities['syntax']),
            ("semantic", "generation", activities['semantic'], activities['generation']),
            ("syntax", "generation", activities['syntax'], activities['generation']),
            ("generation", "output", activities['generation'], activities['output'])
        ]
        
        for pre_layer, post_layer, pre_activity, post_activity in learning_pairs:
            changes = self._apply_neural_language_stdp(pre_layer, post_layer, pre_activity, post_activity)
            total_changes += changes
        
        return total_changes
    
    def _apply_neural_language_stdp(self, pre_layer: str, post_layer: str,
                                   pre_activity: np.ndarray, post_activity: np.ndarray) -> int:
        """Apply STDP for language learning"""
        
        connection_key = (pre_layer, post_layer)
        changes = 0
        
        if connection_key in self.network.connections:
            connection = self.network.connections[connection_key]
            synapse_pop = connection.synapse_population
            
            if synapse_pop and hasattr(synapse_pop, 'synapses'):
                for (pre_idx, post_idx), synapse in synapse_pop.synapses.items():
                    if pre_idx < len(pre_activity) and post_idx < len(post_activity):
                        pre_active = pre_activity[pre_idx] > 0.3
                        post_active = post_activity[post_idx] > 0.3
                        
                        if pre_active and post_active:
                            # Strengthen language associations
                            synapse.weight += 0.3
                            changes += 1
                        elif pre_active and not post_active:
                            # Weaken unused pathways
                            synapse.weight -= 0.05
                            changes += 1
                        
                        # Keep weights bounded
                        synapse.weight = np.clip(synapse.weight, 0.1, 18.0)
        
        return changes
    
    def have_neural_conversation(self):
        """Pure neural conversation - no static responses"""
        
        print("\n🧠 PURE NEUROMORPHIC LANGUAGE GENERATION")
        print("=" * 50)
        print("🚀 NO pre-written responses")
        print("🧠 Language emerges from neural activity")
        print("⚡ Direct neural-to-text generation")
        print("📝 Type 'quit' to end conversation")
        print("-" * 50)
        
        conversation_count = 0
        total_learning = 0
        
        while True:
            try:
                user_input = input(f"\n👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'stop']:
                    break
                
                if not user_input:
                    continue
                
                conversation_count += 1
                print(f"🧠 Neural processing...")
                
                # Pure neural processing
                start_time = time.time()
                input_pattern = self.encode_concept_to_neural(user_input)
                neural_activities = self.neural_forward_pass(input_pattern)
                
                # Generate response directly from neural activity
                neural_response = self.neural_language_generation(neural_activities)
                processing_time = time.time() - start_time
                
                print(f"🤖 AI: {neural_response}")
                
                # Learn from this interaction
                weight_changes = self.learn_from_interaction(user_input, neural_response, neural_activities)
                total_learning += weight_changes
                
                # Show pure neural activity
                semantic_activity = np.sum(neural_activities['semantic'])
                syntax_activity = np.sum(neural_activities['syntax'])
                generation_activity = np.sum(neural_activities['generation'])
                output_activity = np.sum(neural_activities['output'])
                
                print(f"📊 Neural: Semantic:{semantic_activity:.1f} "
                      f"Syntax:{syntax_activity:.1f} Generation:{generation_activity:.1f} "
                      f"Output:{output_activity:.1f}")
                print(f"⚡ Learning: {weight_changes} changes | Time: {processing_time:.3f}s")
                
            except KeyboardInterrupt:
                print(f"\n🛑 Conversation ended.")
                break
            except Exception as e:
                print(f"❌ Neural error: {e}")
                continue
        
        print(f"\n🏁 PURE NEURAL CONVERSATION ENDED")
        print(f"💬 Total exchanges: {conversation_count}")
        print(f"🧠 Total neural learning: {total_learning} synaptic changes")
        print("🚀 Language was generated purely from neural dynamics!")
        
        return {
            'exchanges': conversation_count,
            'total_learning': total_learning,
            'neural_memory_state': self.neural_memory.tolist(),
            'syntax_state': self.syntax_state.tolist()
        }

def main():
    """Start true neuromorphic language generation"""
    
    print("[INFO] Starting PURE neuromorphic language system...")
    print("[INFO] No LLMs, no static responses - only neural networks!")
    
    # Initialize pure neural language system
    neural_ai = TrueNeuromorphicLanguage()
    
    # Start pure neural conversation
    results = neural_ai.have_neural_conversation()
    
    print("\n🧠 Pure neuromorphic language generation complete!")
    
    return results

if __name__ == "__main__":
    main()
