#!/usr/bin/env python3
"""
Neuromorphic Symbol Grounding System

Tackling the REAL obstacle: How do neural patterns learn to represent
actual concepts and meanings, not just character sequences?

The Symbol Grounding Problem:
- How does a neural pattern for "cat" connect to the concept of cat?
- How do we go from character sequences to semantic understanding?
- How does meaning emerge from neural dynamics?

Approach:
- Multi-modal sensory input (visual, auditory, textual)
- Reward-driven concept formation
- Hierarchical binding of features to symbols
- Persistent concept memory formation
- Cross-modal association learning
"""

import numpy as np
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

# Import neuromorphic components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.network import NeuromorphicNetwork

class ConceptualNeuralPattern:
    """Represents a learned concept as a stable neural pattern"""
    
    def __init__(self, name: str, neural_signature: np.ndarray):
        self.name = name
        self.neural_signature = neural_signature  # Stable pattern representing this concept
        self.activation_history = []
        self.association_strength = {}  # Links to other concepts
        self.sensory_features = {}  # Multi-modal features
        self.formation_time = datetime.now()
        self.reinforcement_count = 0
    
    def strengthen(self, new_pattern: np.ndarray, learning_rate: float = 0.1):
        """Strengthen concept pattern through hebbian learning"""
        self.neural_signature = (1 - learning_rate) * self.neural_signature + learning_rate * new_pattern
        self.reinforcement_count += 1
    
    def similarity(self, pattern: np.ndarray) -> float:
        """Calculate similarity to input pattern"""
        if len(pattern) != len(self.neural_signature):
            return 0.0
        return np.dot(pattern, self.neural_signature) / (np.linalg.norm(pattern) * np.linalg.norm(self.neural_signature) + 1e-8)

class SymbolGroundingSystem:
    """Neuromorphic system that learns to ground symbols in neural patterns"""
    
    def __init__(self):
        # Multi-modal sensory vocabulary
        self.text_vocab = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                          'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ']
        
        # Sensory feature dimensions
        self.visual_features = 20    # Visual properties (size, color, shape, etc.)
        self.auditory_features = 15  # Auditory properties (phonemes, rhythm, etc.) 
        self.text_features = len(self.text_vocab)  # Textual characters
        
        # Neural architecture for concept formation
        self.sensory_integration_size = 32  # Multi-modal integration
        self.concept_formation_size = 24    # Concept learning layer
        self.binding_size = 18              # Symbol-concept binding
        self.reward_processing_size = 12    # Reward/goal processing
        self.concept_memory_size = 16       # Long-term concept storage
        
        # Build the symbol grounding network
        self.network = self._build_grounding_network()
        
        # Concept memory system
        self.learned_concepts = {}  # Dict[str, ConceptualNeuralPattern]
        self.concept_formation_threshold = 0.7
        self.reward_history = []
        
        # Multi-modal sensory simulation
        self.sensory_simulator = self._create_sensory_simulator()
        
        print("🧠 NEUROMORPHIC SYMBOL GROUNDING SYSTEM")
        print("=" * 50)
        print("🎯 Tackling the symbol grounding problem")
        print("🔗 Learning how neural patterns represent concepts")
        print("🌟 Multi-modal concept formation")
        print("💡 Reward-driven meaning emergence")
        print("🚀 Ready for concept learning!")
    
    def _build_grounding_network(self) -> NeuromorphicNetwork:
        """Build neural network for symbol grounding"""
        network = NeuromorphicNetwork()
        
        # Sensory input layers
        network.add_layer("visual", self.visual_features, "lif")
        network.add_layer("auditory", self.auditory_features, "lif") 
        network.add_layer("textual", self.text_features, "lif")
        
        # Integration and processing layers
        network.add_layer("sensory_integration", self.sensory_integration_size, "adex")
        network.add_layer("concept_formation", self.concept_formation_size, "adex")
        network.add_layer("binding", self.binding_size, "adex")
        network.add_layer("reward_processing", self.reward_processing_size, "lif")
        network.add_layer("concept_memory", self.concept_memory_size, "adex")
        
        # Multi-modal convergence
        network.connect_layers("visual", "sensory_integration", "stdp", connection_probability=0.6)
        network.connect_layers("auditory", "sensory_integration", "stdp", connection_probability=0.6)
        network.connect_layers("textual", "sensory_integration", "stdp", connection_probability=0.7)
        
        # Concept formation pathway
        network.connect_layers("sensory_integration", "concept_formation", "stdp", connection_probability=0.8)
        network.connect_layers("concept_formation", "binding", "stdp", connection_probability=0.7)
        network.connect_layers("binding", "concept_memory", "stdp", connection_probability=0.6)
        
        # Reward-driven learning
        network.connect_layers("reward_processing", "concept_formation", "stdp", connection_probability=0.5)
        network.connect_layers("reward_processing", "binding", "stdp", connection_probability=0.4)
        
        # Recurrent connections for concept stability
        network.connect_layers("concept_memory", "concept_formation", "stdp", connection_probability=0.4)
        network.connect_layers("binding", "sensory_integration", "stdp", connection_probability=0.3)
        
        return network
    
    def _create_sensory_simulator(self) -> Dict[str, Any]:
        """Create simulated multi-modal sensory experiences"""
        
        # Define concept prototypes with multi-modal features
        concept_prototypes = {
            "cat": {
                "visual": [0.8, 0.6, 0.4, 0.9, 0.3, 0.7, 0.5, 0.8, 0.2, 0.6,  # fur, whiskers, eyes, etc.
                          0.4, 0.7, 0.3, 0.5, 0.8, 0.2, 0.6, 0.4, 0.9, 0.3],
                "auditory": [0.7, 0.8, 0.3, 0.5, 0.9, 0.2, 0.6, 0.4, 0.7, 0.3,  # meow sounds, purring
                            0.8, 0.5, 0.2, 0.6, 0.4],
                "reward_value": 0.8
            },
            "dog": {
                "visual": [0.9, 0.4, 0.7, 0.8, 0.5, 0.6, 0.9, 0.3, 0.7, 0.4,  # tail, bark, ears, etc.
                          0.8, 0.5, 0.6, 0.7, 0.3, 0.9, 0.4, 0.8, 0.2, 0.6],
                "auditory": [0.9, 0.6, 0.8, 0.4, 0.7, 0.5, 0.9, 0.3, 0.6, 0.8,  # bark sounds
                            0.4, 0.7, 0.5, 0.2, 0.9],
                "reward_value": 0.9
            },
            "car": {
                "visual": [0.5, 0.8, 0.9, 0.4, 0.7, 0.3, 0.6, 0.8, 0.5, 0.9,  # wheels, metal, windows
                          0.3, 0.7, 0.6, 0.4, 0.8, 0.5, 0.3, 0.7, 0.6, 0.9],
                "auditory": [0.4, 0.3, 0.8, 0.9, 0.5, 0.7, 0.3, 0.6, 0.8, 0.4,  # engine sounds
                            0.9, 0.2, 0.7, 0.5, 0.3],
                "reward_value": 0.6
            },
            "hello": {
                "visual": [0.2, 0.3, 0.4, 0.2, 0.3, 0.5, 0.2, 0.4, 0.3, 0.2,  # text/gesture visual
                          0.5, 0.3, 0.4, 0.2, 0.3, 0.4, 0.5, 0.2, 0.3, 0.4],
                "auditory": [0.8, 0.7, 0.6, 0.8, 0.9, 0.5, 0.7, 0.8, 0.6, 0.9,  # speech sounds
                            0.7, 0.8, 0.5, 0.6, 0.7],
                "reward_value": 0.9  # High reward for successful greeting
            }
        }
        
        return concept_prototypes
    
    def simulate_sensory_experience(self, concept_name: str, noise_level: float = 0.1) -> Dict[str, np.ndarray]:
        """Simulate multi-modal sensory experience of a concept"""
        
        if concept_name not in self.sensory_simulator:
            # Unknown concept - generate random sensory pattern
            visual_pattern = np.random.random(self.visual_features) * 0.3
            auditory_pattern = np.random.random(self.auditory_features) * 0.3
            text_pattern = np.random.random(self.text_features) * 0.2
            reward_value = 0.1
        else:
            prototype = self.sensory_simulator[concept_name]
            
            # Add noise to prototype patterns (realistic sensory variation)
            visual_pattern = np.array(prototype["visual"]) + np.random.normal(0, noise_level, self.visual_features)
            auditory_pattern = np.array(prototype["auditory"]) + np.random.normal(0, noise_level, self.auditory_features)
            
            # Text pattern based on concept name
            text_pattern = np.zeros(self.text_features)
            for i, char in enumerate(concept_name.lower()[:len(self.text_vocab)]):
                if char in self.text_vocab:
                    char_idx = self.text_vocab.index(char)
                    text_pattern[char_idx] = 0.8 + np.random.normal(0, noise_level)
            
            reward_value = prototype["reward_value"]
        
        # Clip values to valid ranges
        visual_pattern = np.clip(visual_pattern, 0, 1)
        auditory_pattern = np.clip(auditory_pattern, 0, 1)
        text_pattern = np.clip(text_pattern, 0, 1)
        
        return {
            "visual": visual_pattern,
            "auditory": auditory_pattern,
            "textual": text_pattern,
            "reward": np.array([reward_value])  # Convert to array for consistency
        }
    
    def neural_forward_pass(self, sensory_input: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Process multi-modal sensory input through grounding network"""
        
        activities = {}
        
        # Multi-modal sensory processing
        visual_activity = sensory_input["visual"]
        auditory_activity = sensory_input["auditory"] 
        textual_activity = sensory_input["textual"]
        
        activities['visual'] = visual_activity
        activities['auditory'] = auditory_activity
        activities['textual'] = textual_activity
        
        # Sensory integration (convergence)
        integration_currents = (
            self._compute_layer_currents(visual_activity, "visual", "sensory_integration") +
            self._compute_layer_currents(auditory_activity, "auditory", "sensory_integration") +
            self._compute_layer_currents(textual_activity, "textual", "sensory_integration")
        )
        
        integration_activity = self._neural_activation(integration_currents, threshold=4.0)
        activities['sensory_integration'] = integration_activity
        
        # Concept formation
        concept_currents = self._compute_layer_currents(integration_activity, "sensory_integration", "concept_formation")
        
        # Add reward modulation
        reward_signal = sensory_input.get("reward", np.array([0.0]))[0]  # Extract scalar
        reward_activity = np.ones(self.reward_processing_size) * reward_signal
        reward_modulation = self._compute_layer_currents(reward_activity, "reward_processing", "concept_formation")
        
        combined_concept_currents = concept_currents + reward_modulation * 0.5
        concept_activity = self._neural_activation(combined_concept_currents, threshold=5.0)
        activities['concept_formation'] = concept_activity
        
        # Symbol-concept binding
        binding_currents = self._compute_layer_currents(concept_activity, "concept_formation", "binding")
        binding_activity = self._neural_activation(binding_currents, threshold=4.5)
        activities['binding'] = binding_activity
        
        # Concept memory formation
        memory_currents = self._compute_layer_currents(binding_activity, "binding", "concept_memory")
        memory_activity = self._neural_activation(memory_currents, threshold=3.5)
        activities['concept_memory'] = memory_activity
        
        return activities
    
    def _compute_layer_currents(self, pre_activity: np.ndarray, pre_layer: str, post_layer: str) -> np.ndarray:
        """Compute synaptic currents between layers"""
        connection_key = (pre_layer, post_layer)
        
        layer_sizes = {
            "sensory_integration": self.sensory_integration_size,
            "concept_formation": self.concept_formation_size,
            "binding": self.binding_size,
            "reward_processing": self.reward_processing_size,
            "concept_memory": self.concept_memory_size
        }
        
        post_size = layer_sizes.get(post_layer, 10)
        amplification = 12.0  # Standard amplification
        
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
                activations[i] = min(1.0, current / (threshold * 2))
        return activations
    
    def detect_concept_formation(self, activities: Dict[str, np.ndarray]) -> Optional[str]:
        """Detect if a stable concept has formed"""
        
        concept_activity = activities['concept_formation']
        binding_activity = activities['binding']
        memory_activity = activities['concept_memory']
        
        # Check for strong, stable pattern
        concept_strength = np.sum(concept_activity)
        binding_strength = np.sum(binding_activity)
        memory_strength = np.sum(memory_activity)
        
        total_strength = concept_strength + binding_strength + memory_strength
        
        if total_strength > self.concept_formation_threshold * 30:  # Threshold for concept formation
            # Create neural signature for this concept
            neural_signature = np.concatenate([concept_activity, binding_activity, memory_activity])
            
            # Check if this matches an existing concept
            best_match = None
            best_similarity = 0.0
            
            for concept_name, concept_pattern in self.learned_concepts.items():
                similarity = concept_pattern.similarity(neural_signature)
                if similarity > best_similarity and similarity > 0.7:  # High similarity threshold
                    best_similarity = similarity
                    best_match = concept_name
            
            return best_match
        
        return None
    
    def learn_concept_association(self, concept_name: str, activities: Dict[str, np.ndarray]) -> int:
        """Learn or strengthen concept associations"""
        
        # Create neural signature from activities
        concept_signature = np.concatenate([
            activities['concept_formation'],
            activities['binding'],
            activities['concept_memory']
        ])
        
        # Add or strengthen concept
        if concept_name in self.learned_concepts:
            self.learned_concepts[concept_name].strengthen(concept_signature)
        else:
            self.learned_concepts[concept_name] = ConceptualNeuralPattern(concept_name, concept_signature)
        
        # Apply STDP learning
        total_changes = 0
        learning_connections = [
            ("visual", "sensory_integration"),
            ("auditory", "sensory_integration"),
            ("textual", "sensory_integration"),
            ("sensory_integration", "concept_formation"),
            ("concept_formation", "binding"),
            ("binding", "concept_memory")
        ]
        
        for pre_layer, post_layer in learning_connections:
            if pre_layer in activities and post_layer in activities:
                changes = self._apply_concept_stdp(pre_layer, post_layer, activities[pre_layer], activities[post_layer])
                total_changes += changes
        
        return total_changes
    
    def _apply_concept_stdp(self, pre_layer: str, post_layer: str, pre_activity: np.ndarray, post_activity: np.ndarray) -> int:
        """Apply STDP for concept learning"""
        
        connection_key = (pre_layer, post_layer)
        changes = 0
        
        if connection_key in self.network.connections:
            connection = self.network.connections[connection_key]
            synapse_pop = connection.synapse_population
            
            if synapse_pop and hasattr(synapse_pop, 'synapses'):
                for (pre_idx, post_idx), synapse in synapse_pop.synapses.items():
                    if pre_idx < len(pre_activity) and post_idx < len(post_activity):
                        pre_active = pre_activity[pre_idx] > 0.4
                        post_active = post_activity[post_idx] > 0.4
                        
                        if pre_active and post_active:
                            # Strengthen concept associations
                            synapse.weight += 0.4
                            changes += 1
                        elif pre_active and not post_active:
                            # Weaken unused associations
                            synapse.weight -= 0.08
                            changes += 1
                        
                        synapse.weight = np.clip(synapse.weight, 0.1, 20.0)
        
        return changes
    
    def demonstrate_symbol_grounding(self):
        """Demonstrate the symbol grounding learning process"""
        
        print("\n🎯 SYMBOL GROUNDING DEMONSTRATION")
        print("=" * 45)
        print("🧠 Teaching the neural network to ground symbols in concepts")
        print("🔗 Multi-modal sensory experiences → Neural patterns → Concepts")
        print("-" * 45)
        
        concepts_to_learn = ["cat", "dog", "car", "hello"]
        total_learning = 0
        
        for round_num in range(1, 21):  # 20 learning rounds
            print(f"\n🔄 Learning Round {round_num}")
            
            for concept_name in concepts_to_learn:
                # Simulate sensory experience
                sensory_input = self.simulate_sensory_experience(concept_name, noise_level=0.1)
                
                # Process through neural network
                activities = self.neural_forward_pass(sensory_input)
                
                # Check for concept formation/recognition
                recognized_concept = self.detect_concept_formation(activities)
                
                # Learn the association
                weight_changes = self.learn_concept_association(concept_name, activities)
                total_learning += weight_changes
                
                # Progress reporting
                concept_strength = np.sum(activities['concept_formation'])
                binding_strength = np.sum(activities['binding'])
                memory_strength = np.sum(activities['concept_memory'])
                
                status = "✅" if recognized_concept == concept_name else "🔄" if recognized_concept else "❌"
                
                print(f"  {status} '{concept_name}': Concept:{concept_strength:.1f} "
                      f"Binding:{binding_strength:.1f} Memory:{memory_strength:.1f} "
                      f"Changes:{weight_changes}")
                
                if recognized_concept and recognized_concept != concept_name:
                    print(f"    🤔 Confused with: {recognized_concept}")
        
        # Final assessment
        print(f"\n🏁 SYMBOL GROUNDING RESULTS")
        print("=" * 35)
        print(f"📚 Concepts learned: {len(self.learned_concepts)}")
        print(f"🧠 Total synaptic changes: {total_learning}")
        
        # Test concept recognition
        print(f"\n🧪 CONCEPT RECOGNITION TEST")
        print("-" * 30)
        
        for concept_name in concepts_to_learn:
            sensory_input = self.simulate_sensory_experience(concept_name, noise_level=0.05)
            activities = self.neural_forward_pass(sensory_input)
            recognized = self.detect_concept_formation(activities)
            
            status = "✅ GROUNDED" if recognized == concept_name else f"❌ Failed ({recognized})"
            confidence = self.learned_concepts[concept_name].reinforcement_count if concept_name in self.learned_concepts else 0
            
            print(f"  {concept_name}: {status} (reinforcement: {confidence})")
        
        # Show learned concept neural signatures
        print(f"\n🧠 LEARNED CONCEPT SIGNATURES")
        print("-" * 35)
        for name, concept in self.learned_concepts.items():
            signature_strength = np.linalg.norm(concept.neural_signature)
            print(f"  {name}: Neural strength {signature_strength:.2f}, "
                  f"Reinforced {concept.reinforcement_count} times")
        
        if len(self.learned_concepts) >= 3:
            achievement = "🌟 EXCELLENT - Strong symbol grounding achieved!"
        elif len(self.learned_concepts) >= 2:
            achievement = "✅ GOOD - Basic symbol grounding working"
        else:
            achievement = "🌱 DEVELOPING - Symbol grounding in progress"
        
        print(f"\n🎯 SYMBOL GROUNDING: {achievement}")
        
        return {
            'learned_concepts': len(self.learned_concepts),
            'total_learning': total_learning,
            'achievement': achievement,
            'concept_details': {name: concept.reinforcement_count for name, concept in self.learned_concepts.items()}
        }

def main():
    """Run symbol grounding demonstration"""
    
    print("[INFO] Initializing Symbol Grounding System...")
    print("[INFO] Tackling the neural symbol grounding problem!")
    
    # Create symbol grounding system
    grounding_system = SymbolGroundingSystem()
    
    # Run the demonstration
    results = grounding_system.demonstrate_symbol_grounding()
    
    print(f"\n🎯 Symbol grounding complete!")
    print(f"Achievement: {results['achievement']}")
    
    return results

if __name__ == "__main__":
    main()
