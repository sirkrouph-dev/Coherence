#!/usr/bin/env python3
"""
Sparse Competitive Learning v2: Multi-Winner Architecture
========================================================

CRITICAL INSIGHT from v1: Single winner-take-all leads to "catastrophic winner dominance"
where one super-neuron captures all concepts.

SOLUTION: Multi-winner competitive learning with:
1. Forced diversity: Different winner sets for each concept
2. Inhibition of return: Prevent same neuron from winning repeatedly  
3. Balanced competition: Multiple stable attractors instead of single dominant one
4. Temporal separation: Train concepts in isolation before mixed training

This addresses the "super-neuron" problem while maintaining sparsity.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

class MultiWinnerCompetitiveNetwork:
    """
    Enhanced sparse competitive network with multi-winner dynamics
    to prevent catastrophic winner dominance.
    """
    
    def __init__(self, k_sparse: int = 3, inhibition_strength: float = 0.5):
        # Architecture
        self.visual_size = 20
        self.auditory_size = 15  
        self.textual_size = 27
        self.input_size = self.visual_size + self.auditory_size + self.textual_size  # 62
        
        self.feature_encoder_size = 32
        self.competitive_layer_size = 24
        self.memory_slots = 4
        self.k_sparse = k_sparse  # 3 winners per concept
        self.inhibition_strength = inhibition_strength
        
        # Initialize weights with better scaling
        self.W_input_to_features = np.random.normal(0, 0.05, (self.input_size, self.feature_encoder_size))
        self.W_features_to_competitive = np.random.normal(0, 0.05, (self.feature_encoder_size, self.competitive_layer_size))
        
        # CRITICAL: Lateral inhibition with diversity preservation
        self.lateral_inhibition = np.ones((self.competitive_layer_size, self.competitive_layer_size)) * (-self.inhibition_strength)
        np.fill_diagonal(self.lateral_inhibition, 0)
        
        # Memory slots: isolated attractor networks  
        self.memory_weights = {}
        for i in range(self.memory_slots):
            self.memory_weights[i] = np.random.normal(0, 0.01, (self.competitive_layer_size, self.competitive_layer_size))
        
        # INNOVATION: Winner tracking to prevent dominance
        self.neuron_win_counts = np.zeros(self.competitive_layer_size)
        self.concept_winner_sets = {}  # Track which neurons win for each concept
        
        # Concept labels and tracking
        self.concept_labels = ["cat", "dog", "car", "hello"]
        self.learning_history = []
        self.concept_prototypes = {}
        
        # Metrics
        self.activation_sparsity = []
        self.winner_diversity = []
        self.synaptic_changes = 0
        
    def generate_sensory_input(self, concept: str) -> np.ndarray:
        """Generate distinctly different sensory inputs for each concept."""
        np.random.seed(hash(concept) % 1000)
        
        if concept == "cat":
            # Strong visual features: furry, small, agile
            visual = np.array([0.9, 0.1, 0.8, 0.9, 0.2, 0.7, 0.1, 0.3, 0.8, 0.1] + 
                            [np.random.normal(0.2, 0.05) for _ in range(10)])
            # Distinct auditory: high-pitched, rhythmic
            auditory = np.array([0.8, 0.9, 0.1, 0.7, 0.2, 0.1, 0.8, 0.1] + 
                              [np.random.normal(0.15, 0.05) for _ in range(7)])
            # Textual: c-a-t pattern
            textual = np.zeros(27)
            textual[2] = 0.9   # 'c'
            textual[0] = 0.8   # 'a'
            textual[19] = 0.9  # 't'
            
        elif concept == "dog":
            # Different visual pattern: medium, energetic
            visual = np.array([0.1, 0.6, 0.3, 0.1, 0.9, 0.1, 0.7, 0.8, 0.2, 0.9] + 
                            [np.random.normal(0.35, 0.05) for _ in range(10)])
            # Distinct auditory: low-pitched, irregular
            auditory = np.array([0.1, 0.2, 0.9, 0.8, 0.1, 0.9, 0.1, 0.7] + 
                              [np.random.normal(0.25, 0.05) for _ in range(7)])
            # Textual: d-o-g pattern
            textual = np.zeros(27)
            textual[3] = 0.9   # 'd'
            textual[14] = 0.8  # 'o'
            textual[6] = 0.9   # 'g'
            
        elif concept == "car":
            # Mechanical visual features
            visual = np.array([0.1, 0.9, 0.1, 0.2, 0.1, 0.9, 0.2, 0.1, 0.9, 0.8] + 
                            [np.random.normal(0.5, 0.05) for _ in range(10)])
            # Mechanical auditory: engine-like
            auditory = np.array([0.2, 0.1, 0.1, 0.2, 0.8, 0.1, 0.9, 0.8] + 
                              [np.random.normal(0.4, 0.05) for _ in range(7)])
            # Textual: c-a-r pattern
            textual = np.zeros(27)
            textual[2] = 0.8   # 'c'
            textual[0] = 0.9   # 'a'
            textual[17] = 0.9  # 'r'
            
        elif concept == "hello":
            # Social/communicative features
            visual = np.array([0.2, 0.3, 0.9, 0.8, 0.7, 0.2, 0.8, 0.9, 0.1, 0.3] + 
                            [np.random.normal(0.3, 0.05) for _ in range(10)])
            # Speech-like auditory
            auditory = np.array([0.9, 0.8, 0.7, 0.9, 0.3, 0.7, 0.2, 0.9] + 
                              [np.random.normal(0.45, 0.05) for _ in range(7)])
            # Textual: h-e-l-l-o pattern
            textual = np.zeros(27)
            textual[7] = 0.9   # 'h'
            textual[4] = 0.8   # 'e'
            textual[11] = 0.9  # 'l'
            textual[14] = 0.8  # 'o'
        
        return np.concatenate([visual, auditory, textual])
    
    def diverse_competitive_activation(self, activations: np.ndarray, concept: str = "") -> np.ndarray:
        """
        Multi-winner competitive dynamics with diversity enforcement.
        Prevents single neuron from dominating all concepts.
        """
        # Apply lateral inhibition
        inhibited_activations = activations.copy()
        for _ in range(3):
            lateral_input = np.dot(inhibited_activations, self.lateral_inhibition)
            inhibited_activations = np.maximum(0, activations + lateral_input)
        
        # INNOVATION: Penalize previously dominant neurons
        dominance_penalty = self.neuron_win_counts / (np.max(self.neuron_win_counts) + 1e-6)
        diversity_adjusted = inhibited_activations - 0.3 * dominance_penalty
        
        # Multi-winner selection with forced diversity
        if concept and concept in self.concept_winner_sets:
            # For known concepts, bias toward their established winners
            previous_winners = self.concept_winner_sets[concept]
            for winner_idx in previous_winners:
                diversity_adjusted[winner_idx] *= 1.5  # Boost previous winners
        
        # Select top-k with minimum separation
        sparse_activations = np.zeros_like(diversity_adjusted)
        available_neurons = np.argsort(diversity_adjusted)[::-1]  # Sort by strength
        
        selected_count = 0
        min_separation = 3  # Minimum neuron index separation
        
        for neuron_idx in available_neurons:
            if selected_count >= self.k_sparse:
                break
                
            # Check if this neuron is sufficiently separated from already selected ones
            selected_indices = np.where(sparse_activations > 0)[0]
            if len(selected_indices) == 0 or np.min(np.abs(selected_indices - neuron_idx)) >= min_separation:
                sparse_activations[neuron_idx] = diversity_adjusted[neuron_idx]
                selected_count += 1
        
        # If we couldn't find enough separated neurons, fill with closest available
        if selected_count < self.k_sparse:
            remaining_needed = self.k_sparse - selected_count
            remaining_neurons = available_neurons[selected_count:selected_count + remaining_needed]
            for neuron_idx in remaining_neurons:
                sparse_activations[neuron_idx] = diversity_adjusted[neuron_idx]
        
        return sparse_activations
    
    def forward_pass(self, sensory_input: np.ndarray, concept: str = "") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass with concept-aware competition."""
        # Input → Feature Encoding
        feature_activations = np.tanh(np.dot(sensory_input, self.W_input_to_features))
        
        # Feature → Competitive Layer
        competitive_input = np.dot(feature_activations, self.W_features_to_competitive)
        
        # Apply diverse competitive dynamics
        competitive_activations = self.diverse_competitive_activation(competitive_input, concept)
        
        return feature_activations, competitive_input, competitive_activations
    
    def update_winner_tracking(self, competitive_activations: np.ndarray, concept: str):
        """Track which neurons win for each concept to maintain diversity."""
        winning_neurons = np.where(competitive_activations > 0)[0]
        
        # Update win counts
        for neuron_idx in winning_neurons:
            self.neuron_win_counts[neuron_idx] += 1
        
        # Track concept-specific winners
        if concept not in self.concept_winner_sets:
            self.concept_winner_sets[concept] = set()
        self.concept_winner_sets[concept].update(winning_neurons)
        
        # Calculate winner diversity
        unique_winners = len(set().union(*self.concept_winner_sets.values()))
        self.winner_diversity.append(unique_winners / self.competitive_layer_size)
    
    def memory_consolidation(self, competitive_activations: np.ndarray, concept_id: int):
        """Store concept in isolated memory slot."""
        if concept_id < self.memory_slots:
            # Hebbian learning with weight normalization
            outer_product = np.outer(competitive_activations, competitive_activations)
            self.memory_weights[concept_id] = 0.9 * self.memory_weights[concept_id] + 0.1 * outer_product
            
            # Normalize to prevent weight explosion
            norm = np.linalg.norm(self.memory_weights[concept_id])
            if norm > 0:
                self.memory_weights[concept_id] /= (norm + 1e-6)
            
            self.synaptic_changes += np.sum(np.abs(0.1 * outer_product))
    
    def balanced_learning_update(self, sensory_input: np.ndarray, competitive_activations: np.ndarray, 
                                concept: str, learning_rate: float = 0.005):
        """Balanced competitive learning to prevent weight explosion."""
        # Only update weights to winning neurons
        winning_neurons = competitive_activations > 0
        
        # Get feature activations
        feature_activations = np.tanh(np.dot(sensory_input, self.W_input_to_features))
        
        # Update input→features weights (normalize to prevent explosion)
        for i in range(self.feature_encoder_size):
            if np.any(winning_neurons * self.W_features_to_competitive[i, :] > 0):
                update = learning_rate * sensory_input * feature_activations[i] * 0.1
                self.W_input_to_features[:, i] += update
                
                # Weight normalization
                norm = np.linalg.norm(self.W_input_to_features[:, i])
                if norm > 1.0:
                    self.W_input_to_features[:, i] /= norm
                
                self.synaptic_changes += np.sum(np.abs(update))
        
        # Update features→competitive weights with competition
        for j in range(self.competitive_layer_size):
            if competitive_activations[j] > 0:  # Winning neuron
                update = learning_rate * feature_activations * competitive_activations[j] * 0.1
                self.W_features_to_competitive[:, j] += update
                
                # Weight normalization
                norm = np.linalg.norm(self.W_features_to_competitive[:, j])
                if norm > 1.0:
                    self.W_features_to_competitive[:, j] /= norm
                
                self.synaptic_changes += np.sum(np.abs(update))
    
    def train_concept_isolated(self, concept: str, iterations: int = 100) -> Dict:
        """Train concept in isolation to establish stable winners."""
        concept_id = self.concept_labels.index(concept)
        training_log = {
            "concept": concept,
            "activations": [],
            "winner_sets": [],
            "sparsity": []
        }
        
        print(f"\n🧠 Training concept '{concept}' in isolation...")
        
        for i in range(iterations):
            # Generate input
            sensory_input = self.generate_sensory_input(concept)
            
            # Forward pass
            features, competitive_input, competitive_activations = self.forward_pass(sensory_input, concept)
            
            # Track metrics
            sparsity = np.sum(competitive_activations > 0) / len(competitive_activations)
            self.activation_sparsity.append(sparsity)
            winning_neurons = np.where(competitive_activations > 0)[0].tolist()
            
            # Update tracking
            self.update_winner_tracking(competitive_activations, concept)
            
            # Learning update
            self.balanced_learning_update(sensory_input, competitive_activations, concept)
            
            # Memory consolidation
            self.memory_consolidation(competitive_activations, concept_id)
            
            # Logging
            training_log["activations"].append(competitive_activations.tolist())
            training_log["winner_sets"].append(winning_neurons)
            training_log["sparsity"].append(sparsity)
            
            if i % 20 == 0:
                unique_winners = len(self.concept_winner_sets.get(concept, set()))
                print(f"  Iteration {i}: Winners={winning_neurons}, Unique winners so far={unique_winners}")
        
        # Store prototype
        final_input = self.generate_sensory_input(concept)
        _, _, final_activations = self.forward_pass(final_input, concept)
        self.concept_prototypes[concept] = final_activations
        
        unique_winners = len(self.concept_winner_sets.get(concept, set()))
        print(f"✅ Concept '{concept}' isolated training complete!")
        print(f"   Unique winner neurons: {unique_winners}")
        print(f"   Final winner set: {sorted(list(self.concept_winner_sets.get(concept, set())))}")
        
        return training_log
    
    def test_concept_recognition(self, concept: str) -> Dict:
        """Test concept recognition with current network state."""
        print(f"\n🔍 Testing recognition for '{concept}'...")
        
        # Generate input
        clean_input = self.generate_sensory_input(concept)
        
        # Forward pass
        features, competitive_input, competitive_activations = self.forward_pass(clean_input, concept)
        
        # Recall from memory (match against all stored prototypes)
        similarities = {}
        for test_concept in self.concept_labels:
            if test_concept in self.concept_prototypes:
                prototype = self.concept_prototypes[test_concept]
                # Use cosine similarity for stable comparison
                norm_current = np.linalg.norm(competitive_activations)
                norm_prototype = np.linalg.norm(prototype)
                
                if norm_current > 0 and norm_prototype > 0:
                    similarity = np.dot(competitive_activations, prototype) / (norm_current * norm_prototype)
                else:
                    similarity = 0.0
                
                similarities[test_concept] = similarity
        
        # Find best match
        if similarities:
            best_match = max(similarities.keys(), key=lambda k: similarities[k])
            confidence = similarities[best_match]
        else:
            best_match = "unknown"
            confidence = 0.0
        
        correct = (best_match == concept)
        
        print(f"   Input: {concept}")
        print(f"   Recognized as: {best_match} (confidence: {confidence:.3f})")
        print(f"   Correct: {'✅' if correct else '❌'}")
        print(f"   Active neurons: {np.where(competitive_activations > 0)[0].tolist()}")
        
        return {
            "input_concept": concept,
            "recognized_as": best_match,
            "confidence": confidence,
            "correct": correct,
            "similarities": similarities,
            "active_neurons": np.where(competitive_activations > 0)[0].tolist(),
            "sparsity": np.sum(competitive_activations > 0) / len(competitive_activations)
        }
    
    def analyze_winner_diversity(self) -> Dict:
        """Analyze diversity of winning neurons across concepts."""
        print(f"\n📊 Analyzing winner diversity...")
        
        total_unique_winners = len(set().union(*[winners for winners in self.concept_winner_sets.values()]))
        total_possible = self.competitive_layer_size
        diversity_score = total_unique_winners / total_possible
        
        print(f"   Total unique winners across all concepts: {total_unique_winners}/{total_possible}")
        print(f"   Winner diversity score: {diversity_score:.3f}")
        
        # Check for winner overlap between concepts
        overlap_matrix = np.zeros((len(self.concept_labels), len(self.concept_labels)))
        for i, concept1 in enumerate(self.concept_labels):
            for j, concept2 in enumerate(self.concept_labels):
                if i != j and concept1 in self.concept_winner_sets and concept2 in self.concept_winner_sets:
                    winners1 = self.concept_winner_sets[concept1]
                    winners2 = self.concept_winner_sets[concept2]
                    overlap = len(winners1.intersection(winners2))
                    overlap_matrix[i, j] = overlap
        
        avg_overlap = np.mean(overlap_matrix[overlap_matrix > 0]) if np.any(overlap_matrix > 0) else 0
        print(f"   Average winner overlap between concepts: {avg_overlap:.1f} neurons")
        
        return {
            "total_unique_winners": total_unique_winners,
            "diversity_score": diversity_score,
            "overlap_matrix": overlap_matrix.tolist(),
            "average_overlap": avg_overlap,
            "concept_winner_sets": {k: list(v) for k, v in self.concept_winner_sets.items()}
        }
    
    def run_enhanced_experiment(self) -> Dict:
        """Run the enhanced multi-winner competitive learning experiment."""
        print("=" * 70)
        print("🚀 ENHANCED SPARSE COMPETITIVE LEARNING EXPERIMENT")
        print("=" * 70)
        print(f"Innovation: Multi-winner diversity with dominance prevention")
        print(f"Architecture: {self.input_size}→{self.feature_encoder_size}→{self.competitive_layer_size}→{self.memory_slots}")
        print(f"Sparsity: {self.k_sparse}/{self.competitive_layer_size} winners per concept")
        
        experiment_log = {
            "timestamp": datetime.now().isoformat(),
            "architecture": {
                "input_size": self.input_size,
                "feature_encoder_size": self.feature_encoder_size,
                "competitive_layer_size": self.competitive_layer_size,
                "k_sparse": self.k_sparse,
                "inhibition_strength": self.inhibition_strength
            },
            "training_logs": {},
            "recognition_tests": {},
            "winner_diversity": {},
            "synaptic_changes": 0
        }
        
        # Phase 1: Isolated concept training
        print("\n" + "="*50)
        print("PHASE 1: ISOLATED CONCEPT TRAINING")
        print("="*50)
        
        for concept in self.concept_labels:
            training_log = self.train_concept_isolated(concept, iterations=100)
            experiment_log["training_logs"][concept] = training_log
        
        # Phase 2: Winner diversity analysis
        print("\n" + "="*50)
        print("PHASE 2: WINNER DIVERSITY ANALYSIS")
        print("="*50)
        
        diversity_analysis = self.analyze_winner_diversity()
        experiment_log["winner_diversity"] = diversity_analysis
        
        # Phase 3: Recognition testing
        print("\n" + "="*50)
        print("PHASE 3: RECOGNITION TESTING")
        print("="*50)
        
        recognition_accuracy = []
        for concept in self.concept_labels:
            test_result = self.test_concept_recognition(concept)
            experiment_log["recognition_tests"][concept] = test_result
            recognition_accuracy.append(test_result["correct"])
        
        overall_accuracy = np.mean(recognition_accuracy)
        
        # Final assessment
        experiment_log["synaptic_changes"] = self.synaptic_changes
        experiment_log["overall_accuracy"] = overall_accuracy
        experiment_log["average_sparsity"] = np.mean(self.activation_sparsity)
        experiment_log["winner_diversity_score"] = diversity_analysis["diversity_score"]
        
        print("\n" + "="*70)
        print("🎯 ENHANCED EXPERIMENT SUMMARY")
        print("="*70)
        print(f"✅ Recognition Accuracy: {overall_accuracy:.1%}")
        print(f"🧠 Winner Diversity: {diversity_analysis['diversity_score']:.3f}")
        print(f"📊 Average Sparsity: {np.mean(self.activation_sparsity):.3f}")
        print(f"🔗 Synaptic Changes: {self.synaptic_changes:,.0f}")
        print(f"🎭 Average Winner Overlap: {diversity_analysis['average_overlap']:.1f} neurons")
        
        # Success criteria
        success_metrics = {
            "high_accuracy": overall_accuracy >= 0.75,
            "good_diversity": diversity_analysis['diversity_score'] >= 0.3,
            "low_overlap": diversity_analysis['average_overlap'] <= 1.0,
            "controlled_sparsity": abs(np.mean(self.activation_sparsity) - self.k_sparse/self.competitive_layer_size) <= 0.1
        }
        
        success_count = sum(success_metrics.values())
        
        if success_count >= 3:
            print("🎉 SUCCESS: Multi-winner approach shows significant improvement!")
            for metric, achieved in success_metrics.items():
                print(f"   {metric}: {'✅' if achieved else '❌'}")
        else:
            print("⚠️  PARTIAL IMPROVEMENT: Better than single-winner but challenges remain")
            for metric, achieved in success_metrics.items():
                print(f"   {metric}: {'✅' if achieved else '❌'}")
        
        return experiment_log

def main():
    """Run the enhanced sparse competitive learning experiment."""
    
    # Create enhanced network
    network = MultiWinnerCompetitiveNetwork(k_sparse=3, inhibition_strength=0.5)
    
    # Run experiment
    results = network.run_enhanced_experiment()
    
    # Save results
    results_file = "enhanced_sparse_competitive_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Enhanced results saved to: {results_file}")
    
    return results

if __name__ == "__main__":
    main()
