#!/usr/bin/env python3
"""
Balanced Competitive Learning: Final Solution
=============================================

DIAGNOSIS from previous experiments:
1. v1: Catastrophic winner dominance (single super-neuron)
2. v2: Over-inhibition (neurons silenced completely)

ROOT CAUSE: The binding problem requires a delicate balance between:
- Competition (prevent concept collapse)
- Cooperation (maintain meaningful activations)
- Stability (consistent winner sets)

FINAL SOLUTION: Balanced competitive learning with:
1. Soft competition (gradual winner selection, not hard cutoff)
2. Activity homeostasis (maintain minimum activation levels)
3. Progressive learning (gradual competition increase)
4. Cooperative clusters (small groups of neurons per concept)

This should address the binding problem.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

class BalancedCompetitiveNetwork:
    """
    Implementation: Balanced competitive learning that addresses binding problem
    through soft competition and activity homeostasis.
    """
    
    def __init__(self, k_sparse: int = 4, competition_strength: float = 0.3):
        # Architecture
        self.visual_size = 20
        self.auditory_size = 15  
        self.textual_size = 27
        self.input_size = self.visual_size + self.auditory_size + self.textual_size  # 62
        
        self.feature_encoder_size = 32
        self.competitive_layer_size = 24
        self.memory_slots = 4
        self.k_sparse = k_sparse  # 4 cooperative neurons per concept
        self.competition_strength = competition_strength  # Softer competition
        
        # Initialize weights with careful scaling
        self.W_input_to_features = np.random.normal(0, 0.1, (self.input_size, self.feature_encoder_size))
        self.W_features_to_competitive = np.random.normal(0, 0.1, (self.feature_encoder_size, self.competitive_layer_size))
        
        # INNOVATION: Soft lateral inhibition with homeostasis
        self.lateral_weights = np.random.normal(0, 0.05, (self.competitive_layer_size, self.competitive_layer_size))
        np.fill_diagonal(self.lateral_weights, 0)
        
        # Activity homeostasis: maintain minimum activation
        self.baseline_activity = np.ones(self.competitive_layer_size) * 0.1
        self.activity_history = np.zeros(self.competitive_layer_size)
        
        # Memory: concept-specific attractor patterns
        self.concept_attractors = {}
        
        # Concept labels and tracking
        self.concept_labels = ["cat", "dog", "car", "hello"]
        self.concept_prototypes = {}
        self.training_phase = 0  # Progressive competition
        
        # Metrics
        self.activation_sparsity = []
        self.concept_stability = []
        self.synaptic_changes = 0
        
    def generate_sensory_input(self, concept: str) -> np.ndarray:
        """Generate maximally distinct sensory patterns."""
        np.random.seed(hash(concept) % 1000)
        
        if concept == "cat":
            # Visual: specific feature pattern
            visual = np.array([0.9, 0.1, 0.8, 0.1, 0.7, 0.1, 0.9, 0.2, 0.8, 0.1,
                              0.1, 0.9, 0.2, 0.8, 0.1, 0.7, 0.1, 0.9, 0.2, 0.8])
            # Auditory: distinct pattern
            auditory = np.array([0.8, 0.9, 0.1, 0.7, 0.2, 0.1, 0.8, 0.1, 0.9, 0.2,
                               0.7, 0.1, 0.8, 0.9, 0.1])
            # Textual: c-a-t
            textual = np.zeros(27)
            textual[2] = 0.9   # 'c'
            textual[0] = 0.9   # 'a'
            textual[19] = 0.9  # 't'
            
        elif concept == "dog":
            # Completely different pattern
            visual = np.array([0.1, 0.8, 0.2, 0.9, 0.1, 0.7, 0.2, 0.8, 0.1, 0.9,
                              0.9, 0.1, 0.8, 0.2, 0.9, 0.1, 0.7, 0.2, 0.8, 0.1])
            auditory = np.array([0.2, 0.1, 0.9, 0.8, 0.1, 0.7, 0.2, 0.9, 0.1, 0.8,
                               0.1, 0.9, 0.2, 0.1, 0.8])
            textual = np.zeros(27)
            textual[3] = 0.9   # 'd'
            textual[14] = 0.9  # 'o'
            textual[6] = 0.9   # 'g'
            
        elif concept == "car":
            # Another distinct pattern
            visual = np.array([0.2, 0.9, 0.1, 0.8, 0.9, 0.1, 0.2, 0.7, 0.9, 0.2,
                              0.8, 0.2, 0.9, 0.1, 0.8, 0.9, 0.1, 0.2, 0.7, 0.9])
            auditory = np.array([0.1, 0.7, 0.8, 0.2, 0.9, 0.8, 0.1, 0.7, 0.2, 0.9,
                               0.9, 0.8, 0.1, 0.7, 0.2])
            textual = np.zeros(27)
            textual[2] = 0.9   # 'c'
            textual[0] = 0.9   # 'a'
            textual[17] = 0.9  # 'r'
            
        elif concept == "hello":
            # Final distinct pattern
            visual = np.array([0.7, 0.2, 0.9, 0.1, 0.8, 0.7, 0.1, 0.9, 0.2, 0.8,
                              0.2, 0.8, 0.1, 0.9, 0.7, 0.2, 0.8, 0.1, 0.9, 0.7])
            auditory = np.array([0.9, 0.8, 0.7, 0.9, 0.2, 0.8, 0.7, 0.1, 0.9, 0.8,
                               0.2, 0.7, 0.9, 0.8, 0.1])
            textual = np.zeros(27)
            textual[7] = 0.9   # 'h'
            textual[4] = 0.9   # 'e'
            textual[11] = 0.9  # 'l'
            textual[14] = 0.9  # 'o'
        
        return np.concatenate([visual, auditory, textual])
    
    def soft_competitive_activation(self, activations: np.ndarray, training_step: int = 0) -> np.ndarray:
        """
        Soft competitive dynamics with activity homeostasis.
        Progressive competition: starts cooperative, becomes competitive.
        """
        # Progressive competition strength
        progress = min(training_step / 200.0, 1.0)  # Ramp up over 200 steps
        current_competition = self.competition_strength * progress
        
        # Apply homeostatic baseline
        boosted_activations = activations + self.baseline_activity
        
        # Soft lateral interactions (not hard inhibition)
        lateral_input = np.tanh(np.dot(boosted_activations, self.lateral_weights))
        competitive_activations = boosted_activations - current_competition * lateral_input
        
        # Ensure non-negative activations
        competitive_activations = np.maximum(competitive_activations, 0.01)
        
        # Soft k-winners: use sigmoid to create gradual competition
        if np.max(competitive_activations) > 0:
            # Normalize to 0-1 range
            normalized = competitive_activations / np.max(competitive_activations)
            
            # Apply soft k-winner selection
            threshold = np.sort(normalized)[-self.k_sparse]
            soft_mask = 1.0 / (1.0 + np.exp(-10 * (normalized - threshold)))
            
            final_activations = competitive_activations * soft_mask
        else:
            final_activations = competitive_activations
        
        # Update activity history for homeostasis
        self.activity_history = 0.9 * self.activity_history + 0.1 * final_activations
        
        return final_activations
    
    def forward_pass(self, sensory_input: np.ndarray, training_step: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass with progressive competition."""
        # Input → Feature Encoding
        feature_activations = np.tanh(np.dot(sensory_input, self.W_input_to_features))
        
        # Feature → Competitive Layer
        competitive_input = np.dot(feature_activations, self.W_features_to_competitive)
        
        # Apply soft competitive dynamics
        competitive_activations = self.soft_competitive_activation(competitive_input, training_step)
        
        return feature_activations, competitive_activations
    
    def create_concept_attractor(self, competitive_activations: np.ndarray, concept: str):
        """Create stable attractor pattern for concept."""
        if concept not in self.concept_attractors:
            self.concept_attractors[concept] = np.zeros(self.competitive_layer_size)
        
        # Exponential moving average for stability
        alpha = 0.1
        self.concept_attractors[concept] = (1 - alpha) * self.concept_attractors[concept] + alpha * competitive_activations
    
    def balanced_learning_update(self, sensory_input: np.ndarray, competitive_activations: np.ndarray, 
                                concept: str, learning_rate: float = 0.01):
        """Balanced learning with weight stabilization."""
        # Get feature activations
        feature_activations = np.tanh(np.dot(sensory_input, self.W_input_to_features))
        
        # Update input→features weights
        for i in range(self.feature_encoder_size):
            # Simple Hebbian learning
            mean_competitive = np.mean(competitive_activations)
            update = learning_rate * sensory_input * feature_activations[i] * mean_competitive * 0.01
            self.W_input_to_features[:, i] += update
            self.synaptic_changes += np.sum(np.abs(update))
        
        # Update features→competitive weights
        outer_product = np.outer(feature_activations, competitive_activations)
        self.W_features_to_competitive += learning_rate * outer_product * 0.01
        self.synaptic_changes += np.sum(np.abs(learning_rate * outer_product * 0.01))
        
        # Update lateral weights for concept clustering
        lateral_update = np.outer(competitive_activations, competitive_activations)
        self.lateral_weights += learning_rate * lateral_update * 0.001
        self.synaptic_changes += np.sum(np.abs(learning_rate * lateral_update * 0.001))
    
    def train_concept_progressive(self, concept: str, iterations: int = 200) -> Dict:
        """Train concept with progressive competition."""
        training_log = {
            "concept": concept,
            "activations": [],
            "sparsity": [],
            "stability": []
        }
        
        print(f"\n🧠 Training concept '{concept}' with progressive competition...")
        
        previous_winners = set()
        stability_scores = []
        
        for i in range(iterations):
            # Generate input
            sensory_input = self.generate_sensory_input(concept)
            
            # Forward pass with training step
            features, competitive_activations = self.forward_pass(sensory_input, i)
            
            # Track metrics
            sparsity = np.sum(competitive_activations > 0.1) / len(competitive_activations)
            self.activation_sparsity.append(sparsity)
            
            # Calculate stability (consistency of winners)
            current_winners = set(np.where(competitive_activations > 0.2)[0])
            if previous_winners:
                stability = len(current_winners.intersection(previous_winners)) / max(len(current_winners), len(previous_winners), 1)
            else:
                stability = 0.0
            stability_scores.append(stability)
            previous_winners = current_winners
            
            # Learning update
            self.balanced_learning_update(sensory_input, competitive_activations, concept)
            
            # Create attractor
            self.create_concept_attractor(competitive_activations, concept)
            
            # Logging
            training_log["activations"].append(competitive_activations.tolist())
            training_log["sparsity"].append(sparsity)
            training_log["stability"].append(stability)
            
            if i % 40 == 0:
                current_competition = self.competition_strength * min(i / 200.0, 1.0)
                active_neurons = list(np.where(competitive_activations > 0.2)[0])
                print(f"  Step {i}: Competition={current_competition:.3f}, Active={active_neurons}, "
                      f"Stability={stability:.3f}")
        
        # Store final prototype
        final_input = self.generate_sensory_input(concept)
        _, final_activations = self.forward_pass(final_input, iterations)
        self.concept_prototypes[concept] = final_activations
        
        avg_stability = np.mean(stability_scores[-50:]) if stability_scores else 0.0  # Last 50 iterations
        
        print(f"✅ Concept '{concept}' training complete!")
        print(f"   Final stability: {avg_stability:.3f}")
        print(f"   Active neurons: {list(np.where(final_activations > 0.2)[0])}")
        print(f"   Activation strength: {np.max(final_activations):.3f}")
        
        return training_log
    
    def test_concept_recognition(self, concept: str) -> Dict:
        """Test recognition using attractor matching."""
        print(f"\n🔍 Testing recognition for '{concept}'...")
        
        # Generate test input
        test_input = self.generate_sensory_input(concept)
        
        # Get current activations
        _, current_activations = self.forward_pass(test_input, 1000)  # Full competition
        
        # Compare with all concept attractors
        similarities = {}
        for test_concept in self.concept_labels:
            if test_concept in self.concept_attractors:
                attractor = self.concept_attractors[test_concept]
                
                # Cosine similarity
                norm_current = np.linalg.norm(current_activations)
                norm_attractor = np.linalg.norm(attractor)
                
                if norm_current > 0 and norm_attractor > 0:
                    similarity = np.dot(current_activations, attractor) / (norm_current * norm_attractor)
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
        active_neurons = list(np.where(current_activations > 0.2)[0])
        
        print(f"   Input: {concept}")
        print(f"   Recognized as: {best_match} (confidence: {confidence:.3f})")
        print(f"   Correct: {'✅' if correct else '❌'}")
        print(f"   Active neurons: {active_neurons}")
        print(f"   All similarities: {similarities}")
        
        return {
            "input_concept": concept,
            "recognized_as": best_match,
            "confidence": confidence,
            "correct": correct,
            "similarities": similarities,
            "active_neurons": active_neurons,
            "activation_pattern": current_activations.tolist()
        }
    
    def analyze_concept_separation(self) -> Dict:
        """Analyze how well concepts are separated."""
        print(f"\n📊 Analyzing concept separation...")
        
        if not self.concept_attractors:
            return {"error": "No concept attractors available"}
        
        # Calculate pairwise similarities between attractors
        concepts = list(self.concept_attractors.keys())
        similarity_matrix = np.zeros((len(concepts), len(concepts)))
        
        for i, concept1 in enumerate(concepts):
            for j, concept2 in enumerate(concepts):
                if i != j:
                    attractor1 = self.concept_attractors[concept1]
                    attractor2 = self.concept_attractors[concept2]
                    
                    norm1 = np.linalg.norm(attractor1)
                    norm2 = np.linalg.norm(attractor2)
                    
                    if norm1 > 0 and norm2 > 0:
                        similarity = np.dot(attractor1, attractor2) / (norm1 * norm2)
                    else:
                        similarity = 0.0
                    
                    similarity_matrix[i, j] = similarity
        
        avg_inter_similarity = np.mean(similarity_matrix[similarity_matrix > 0])
        
        print(f"   Average inter-concept similarity: {avg_inter_similarity:.3f}")
        print(f"   Separation quality: {1 - avg_inter_similarity:.3f} (higher = better)")
        
        # Analyze neural usage
        all_attractors = np.array([self.concept_attractors[c] for c in concepts])
        neuron_usage = np.sum(all_attractors > 0.1, axis=0)
        unique_neurons = np.sum(neuron_usage == 1)  # Neurons used by only one concept
        shared_neurons = np.sum(neuron_usage > 1)   # Neurons used by multiple concepts
        
        print(f"   Unique neurons (concept-specific): {unique_neurons}")
        print(f"   Shared neurons (multi-concept): {shared_neurons}")
        print(f"   Specialization ratio: {unique_neurons / (unique_neurons + shared_neurons + 1e-6):.3f}")
        
        return {
            "similarity_matrix": similarity_matrix.tolist(),
            "average_inter_similarity": avg_inter_similarity,
            "separation_quality": 1 - avg_inter_similarity,
            "unique_neurons": int(unique_neurons),
            "shared_neurons": int(shared_neurons),
            "specialization_ratio": unique_neurons / (unique_neurons + shared_neurons + 1e-6),
            "concepts": concepts
        }
    
    def run_balanced_experiment(self) -> Dict:
        """Run the complete balanced competitive learning experiment."""
        print("=" * 80)
        print("🚀 BALANCED COMPETITIVE LEARNING: FINAL BINDING PROBLEM SOLUTION")
        print("=" * 80)
        print(f"Innovation: Soft competition + Activity homeostasis + Progressive learning")
        print(f"Architecture: {self.input_size}→{self.feature_encoder_size}→{self.competitive_layer_size}")
        print(f"Target sparsity: {self.k_sparse}/{self.competitive_layer_size} cooperative clusters")
        
        experiment_log = {
            "timestamp": datetime.now().isoformat(),
            "experiment": "balanced_competitive_learning",
            "architecture": {
                "input_size": self.input_size,
                "feature_encoder_size": self.feature_encoder_size,
                "competitive_layer_size": self.competitive_layer_size,
                "k_sparse": self.k_sparse,
                "competition_strength": self.competition_strength
            },
            "training_logs": {},
            "recognition_tests": {},
            "separation_analysis": {},
            "final_metrics": {}
        }
        
        # Phase 1: Progressive concept training
        print("\n" + "="*60)
        print("PHASE 1: PROGRESSIVE CONCEPT TRAINING")
        print("="*60)
        
        for concept in self.concept_labels:
            training_log = self.train_concept_progressive(concept, iterations=200)
            experiment_log["training_logs"][concept] = training_log
        
        # Phase 2: Concept separation analysis
        print("\n" + "="*60)
        print("PHASE 2: CONCEPT SEPARATION ANALYSIS")
        print("="*60)
        
        separation_analysis = self.analyze_concept_separation()
        experiment_log["separation_analysis"] = separation_analysis
        
        # Phase 3: Recognition testing
        print("\n" + "="*60)
        print("PHASE 3: RECOGNITION TESTING")
        print("="*60)
        
        recognition_results = []
        for concept in self.concept_labels:
            test_result = self.test_concept_recognition(concept)
            experiment_log["recognition_tests"][concept] = test_result
            recognition_results.append(test_result["correct"])
        
        overall_accuracy = np.mean(recognition_results)
        
        # Final assessment
        final_metrics = {
            "overall_accuracy": overall_accuracy,
            "average_sparsity": np.mean(self.activation_sparsity),
            "concept_stability": np.mean([np.mean(log["stability"][-50:]) for log in experiment_log["training_logs"].values()]),
            "synaptic_changes": self.synaptic_changes,
            "separation_quality": separation_analysis.get("separation_quality", 0),
            "specialization_ratio": separation_analysis.get("specialization_ratio", 0)
        }
        
        experiment_log["final_metrics"] = final_metrics
        
        print("\n" + "="*80)
        print("🎯 FINAL EXPERIMENT RESULTS")
        print("="*80)
        print(f"✅ Recognition Accuracy: {overall_accuracy:.1%}")
        print(f"🧠 Concept Separation: {separation_analysis.get('separation_quality', 0):.3f}")
        print(f"🎭 Neuron Specialization: {separation_analysis.get('specialization_ratio', 0):.3f}")
        print(f"📊 Average Sparsity: {final_metrics['average_sparsity']:.3f}")
        print(f"🔄 Concept Stability: {final_metrics['concept_stability']:.3f}")
        print(f"🔗 Synaptic Changes: {final_metrics['synaptic_changes']:,.0f}")
        
        # Success evaluation
        success_criteria = {
            "high_accuracy": overall_accuracy >= 0.8,
            "good_separation": separation_analysis.get("separation_quality", 0) >= 0.5,
            "stable_concepts": final_metrics['concept_stability'] >= 0.6,
            "proper_sparsity": 0.1 <= final_metrics['average_sparsity'] <= 0.4
        }
        
        success_count = sum(success_criteria.values())
        
        print(f"\n🏆 SUCCESS CRITERIA ({success_count}/4):")
        for criterion, achieved in success_criteria.items():
            print(f"   {criterion}: {'✅' if achieved else '❌'}")
        
        if success_count >= 3:
            print("\n🎉 SUCCESS: Binding problem appears to be SOLVED!")
            print("   Balanced competitive learning successfully maintains distinct,")
            print("   stable concept representations without catastrophic interference.")
        elif success_count >= 2:
            print("\n🔄 SIGNIFICANT PROGRESS: Major improvement achieved!")
            print("   This approach shows promise for addressing the binding problem.")
        else:
            print("\n⚠️  CHALLENGES REMAIN: Further architectural innovations needed.")
        
        return experiment_log

def main():
    """Run the final balanced competitive learning experiment."""
    
    # Create balanced network
    network = BalancedCompetitiveNetwork(k_sparse=4, competition_strength=0.3)
    
    # Run experiment
    results = network.run_balanced_experiment()
    
    # Save results
    results_file = "balanced_competitive_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Final results saved to: {results_file}")
    
    return results

if __name__ == "__main__":
    main()
