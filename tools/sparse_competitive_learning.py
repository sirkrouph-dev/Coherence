#!/usr/bin/env python3
"""
Sparse Competitive Concept Formation with Attention
==================================================

Implements the suggested architecture to overcome the binding problem:
- Sparse competitive learning (k-sparse: only 3/24 neurons active)
- Lateral inhibition to prevent concept collapse
- Attention gating modulated by context
- Isolated memory slots for stable concept storage
- Winner-take-all output with symbolic binding

This addresses the core roadblock: catastrophic interference where all concepts
collapse to "cat" due to lack of competitive dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

class SparseCompetitiveNetwork:
    """
    Neuromorphic network implementing sparse competitive learning
    to solve the binding problem through winner-take-all dynamics.
    """
    
    def __init__(self, k_sparse: int = 3, inhibition_strength: float = 0.8):
        # Architecture: Inputs → Feature Encoders → Competitive Layer → Memory Slots
        self.visual_size = 20
        self.auditory_size = 15  
        self.textual_size = 27
        self.input_size = self.visual_size + self.auditory_size + self.textual_size  # 62
        
        self.feature_encoder_size = 32
        self.competitive_layer_size = 24
        self.memory_slots = 4  # One per concept
        self.k_sparse = k_sparse  # Only 3 neurons active at once
        self.inhibition_strength = inhibition_strength
        
        # Initialize weights
        self.W_input_to_features = np.random.normal(0, 0.1, (self.input_size, self.feature_encoder_size))
        self.W_features_to_competitive = np.random.normal(0, 0.1, (self.feature_encoder_size, self.competitive_layer_size))
        
        # CRITICAL: Lateral inhibition matrix for competitive dynamics
        self.lateral_inhibition = np.ones((self.competitive_layer_size, self.competitive_layer_size)) * (-self.inhibition_strength)
        np.fill_diagonal(self.lateral_inhibition, 0)  # No self-inhibition
        
        # Memory slots: isolated attractor networks
        self.memory_weights = {}
        for i in range(self.memory_slots):
            self.memory_weights[i] = np.random.normal(0, 0.05, (self.competitive_layer_size, self.competitive_layer_size))
        
        # Attention gate weights (modulated by context)
        self.attention_weights = np.ones(self.feature_encoder_size) * 0.5
        
        # Concept labels and learning state
        self.concept_labels = ["cat", "dog", "car", "hello"]
        self.learning_history = []
        self.concept_prototypes = {}
        
        # Tracking metrics
        self.activation_sparsity = []
        self.concept_separability = []
        self.synaptic_changes = 0
        
    def generate_sensory_input(self, concept: str) -> np.ndarray:
        """Generate multi-modal sensory input for a concept."""
        np.random.seed(hash(concept) % 1000)  # Reproducible but distinct
        
        if concept == "cat":
            # Visual: furry, small, whiskers
            visual = np.array([0.9, 0.1, 0.8, 0.9, 0.2] + [np.random.normal(0.3, 0.1) for _ in range(15)])
            # Auditory: meow sounds, purring
            auditory = np.array([0.8, 0.9, 0.1, 0.7] + [np.random.normal(0.2, 0.1) for _ in range(11)])
            # Textual: letters c-a-t
            textual = np.zeros(27)
            textual[2] = 0.9  # 'c'
            textual[0] = 0.9  # 'a' 
            textual[19] = 0.9  # 't'
            
        elif concept == "dog":
            # Visual: furry, medium, tail wagging
            visual = np.array([0.8, 0.6, 0.3, 0.7, 0.9] + [np.random.normal(0.4, 0.1) for _ in range(15)])
            # Auditory: barking, panting
            auditory = np.array([0.2, 0.1, 0.9, 0.8] + [np.random.normal(0.3, 0.1) for _ in range(11)])
            # Textual: letters d-o-g
            textual = np.zeros(27)
            textual[3] = 0.9  # 'd'
            textual[14] = 0.9  # 'o'
            textual[6] = 0.9  # 'g'
            
        elif concept == "car":
            # Visual: metallic, large, wheels
            visual = np.array([0.1, 0.9, 0.1, 0.2, 0.1] + [np.random.normal(0.6, 0.1) for _ in range(15)])
            # Auditory: engine, honking
            auditory = np.array([0.1, 0.2, 0.1, 0.2] + [np.random.normal(0.7, 0.1) for _ in range(11)])
            # Textual: letters c-a-r
            textual = np.zeros(27)
            textual[2] = 0.9  # 'c'
            textual[0] = 0.9  # 'a'
            textual[17] = 0.9  # 'r'
            
        elif concept == "hello":
            # Visual: hand wave, smile
            visual = np.array([0.2, 0.3, 0.9, 0.8, 0.7] + [np.random.normal(0.4, 0.1) for _ in range(15)])
            # Auditory: voice, greeting tone
            auditory = np.array([0.9, 0.8, 0.7, 0.9] + [np.random.normal(0.5, 0.1) for _ in range(11)])
            # Textual: letters h-e-l-l-o
            textual = np.zeros(27)
            textual[7] = 0.9   # 'h'
            textual[4] = 0.9   # 'e'
            textual[11] = 0.9  # 'l'
            textual[11] = 0.9  # 'l'
            textual[14] = 0.9  # 'o'
        
        else:
            # Unknown concept
            visual = np.random.normal(0.1, 0.05, self.visual_size)
            auditory = np.random.normal(0.1, 0.05, self.auditory_size)
            textual = np.random.normal(0.1, 0.05, self.textual_size)
        
        # Concatenate all modalities
        return np.concatenate([visual, auditory, textual])
    
    def apply_attention_gate(self, features: np.ndarray, context: str = "") -> np.ndarray:
        """Apply attention gating to selectively process features."""
        # Context-dependent attention modulation
        if "visual" in context.lower():
            attention_boost = np.concatenate([np.ones(10) * 1.5, np.ones(22) * 0.7])
        elif "auditory" in context.lower():
            attention_boost = np.concatenate([np.ones(10) * 0.7, np.ones(10) * 1.5, np.ones(12) * 0.7])
        elif "textual" in context.lower():
            attention_boost = np.concatenate([np.ones(20) * 0.7, np.ones(12) * 1.5])
        else:
            attention_boost = np.ones(len(features))
        
        # Apply attention weights
        gated_features = features * self.attention_weights[:len(features)] * attention_boost[:len(features)]
        return gated_features
    
    def competitive_activation(self, activations: np.ndarray) -> np.ndarray:
        """
        Apply k-sparse competitive dynamics with lateral inhibition.
        Only top-k neurons remain active, others are suppressed.
        """
        # Apply lateral inhibition
        inhibited_activations = activations.copy()
        for _ in range(3):  # Iterative inhibition
            lateral_input = np.dot(inhibited_activations, self.lateral_inhibition)
            inhibited_activations = np.maximum(0, activations + lateral_input)
        
        # Winner-take-all: only keep top-k activations
        top_k_indices = np.argsort(inhibited_activations)[-self.k_sparse:]
        sparse_activations = np.zeros_like(inhibited_activations)
        sparse_activations[top_k_indices] = inhibited_activations[top_k_indices]
        
        return sparse_activations
    
    def forward_pass(self, sensory_input: np.ndarray, context: str = "") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass through the sparse competitive network."""
        # Input → Feature Encoding
        feature_activations = np.tanh(np.dot(sensory_input, self.W_input_to_features))
        
        # Apply attention gate
        gated_features = self.apply_attention_gate(feature_activations, context)
        
        # Feature → Competitive Layer
        competitive_input = np.dot(gated_features, self.W_features_to_competitive)
        
        # Apply competitive dynamics (k-sparse)
        competitive_activations = self.competitive_activation(competitive_input)
        
        return feature_activations, gated_features, competitive_activations
    
    def memory_consolidation(self, competitive_activations: np.ndarray, concept_id: int):
        """Store concept in isolated memory slot with attractor dynamics."""
        if concept_id < self.memory_slots:
            # Hebbian learning in memory slot
            outer_product = np.outer(competitive_activations, competitive_activations)
            self.memory_weights[concept_id] = 0.9 * self.memory_weights[concept_id] + 0.1 * outer_product
            self.synaptic_changes += np.sum(np.abs(0.1 * outer_product))
    
    def recall_from_memory(self, competitive_activations: np.ndarray) -> Dict[str, float]:
        """Recall concept from memory slots based on similarity."""
        similarities = {}
        
        for concept_id in range(self.memory_slots):
            # Compute energy of attractor network
            memory_output = np.dot(competitive_activations, self.memory_weights[concept_id])
            energy = np.dot(competitive_activations, memory_output)
            similarities[self.concept_labels[concept_id]] = energy
        
        return similarities
    
    def competitive_learning_update(self, sensory_input: np.ndarray, competitive_activations: np.ndarray, 
                                  target_concept: str, learning_rate: float = 0.01):
        """Update weights using competitive learning rule."""
        # Only update weights to winning neurons (sparse activations)
        winning_neurons = competitive_activations > 0
        
        # Update input→features weights (only for winning path)
        feature_activations = np.tanh(np.dot(sensory_input, self.W_input_to_features))
        for i in range(self.feature_encoder_size):
            if np.any(winning_neurons * self.W_features_to_competitive[i, :] > 0):
                # Strengthen connection to winning neurons
                self.W_input_to_features[:, i] += learning_rate * sensory_input * feature_activations[i]
                self.synaptic_changes += np.sum(np.abs(learning_rate * sensory_input * feature_activations[i]))
        
        # Update features→competitive weights (winner-take-all)
        for j in range(self.competitive_layer_size):
            if competitive_activations[j] > 0:  # Winning neuron
                # Strengthen features that led to this winner
                gated_features = self.apply_attention_gate(feature_activations)
                self.W_features_to_competitive[:, j] += learning_rate * gated_features * competitive_activations[j]
                self.synaptic_changes += np.sum(np.abs(learning_rate * gated_features * competitive_activations[j]))
                
                # Weaken competing connections (lateral inhibition learning)
                for k in range(self.competitive_layer_size):
                    if k != j and competitive_activations[k] == 0:  # Losing neuron
                        self.lateral_inhibition[j, k] -= learning_rate * 0.1
                        self.synaptic_changes += learning_rate * 0.1
    
    def train_concept(self, concept: str, iterations: int = 50) -> Dict:
        """Train network on a specific concept with sparse competitive learning."""
        concept_id = self.concept_labels.index(concept)
        training_log = {
            "concept": concept,
            "activations": [],
            "sparsity": [],
            "winning_neurons": []
        }
        
        print(f"\n🧠 Training concept '{concept}' with sparse competitive learning...")
        
        for i in range(iterations):
            # Generate sensory input
            sensory_input = self.generate_sensory_input(concept)
            
            # Forward pass
            features, gated_features, competitive_activations = self.forward_pass(sensory_input)
            
            # Track sparsity (should be exactly k_sparse neurons active)
            sparsity = np.sum(competitive_activations > 0) / len(competitive_activations)
            self.activation_sparsity.append(sparsity)
            
            # Competitive learning update
            self.competitive_learning_update(sensory_input, competitive_activations, concept)
            
            # Memory consolidation in isolated slot
            self.memory_consolidation(competitive_activations, concept_id)
            
            # Track winning neurons
            winning_neurons = np.where(competitive_activations > 0)[0].tolist()
            
            training_log["activations"].append(competitive_activations.tolist())
            training_log["sparsity"].append(sparsity)
            training_log["winning_neurons"].append(winning_neurons)
            
            if i % 10 == 0:
                print(f"  Iteration {i}: Sparsity={sparsity:.3f}, Winners={winning_neurons}, "
                      f"Max activation={np.max(competitive_activations):.3f}")
        
        # Store concept prototype
        final_input = self.generate_sensory_input(concept)
        _, _, final_activations = self.forward_pass(final_input)
        self.concept_prototypes[concept] = final_activations
        
        print(f"✅ Concept '{concept}' training complete!")
        print(f"   Final sparsity: {np.sum(final_activations > 0)}/{len(final_activations)} neurons active")
        print(f"   Prototype strength: {np.max(final_activations):.3f}")
        
        return training_log
    
    def test_concept_recognition(self, concept: str, noise_level: float = 0.1) -> Dict:
        """Test concept recognition with noisy input."""
        print(f"\n🔍 Testing recognition for '{concept}'...")
        
        # Generate noisy input
        clean_input = self.generate_sensory_input(concept)
        noise = np.random.normal(0, noise_level, len(clean_input))
        noisy_input = clean_input + noise
        
        # Forward pass
        features, gated_features, competitive_activations = self.forward_pass(noisy_input)
        
        # Recall from memory
        similarities = self.recall_from_memory(competitive_activations)
        
        # Find best match
        best_match = max(similarities.keys(), key=lambda k: similarities[k])
        confidence = similarities[best_match]
        
        # Check if recognition is correct
        correct = (best_match == concept)
        
        print(f"   Input concept: {concept}")
        print(f"   Recognized as: {best_match} (confidence: {confidence:.3f})")
        print(f"   Correct: {'✅' if correct else '❌'}")
        print(f"   All similarities: {similarities}")
        
        return {
            "input_concept": concept,
            "recognized_as": best_match,
            "confidence": confidence,
            "correct": correct,
            "similarities": similarities,
            "activations": competitive_activations.tolist(),
            "sparsity": np.sum(competitive_activations > 0) / len(competitive_activations)
        }
    
    def analyze_concept_separability(self) -> Dict:
        """Analyze how well concepts are separated in the network."""
        print(f"\n📊 Analyzing concept separability...")
        
        separability_matrix = np.zeros((len(self.concept_labels), len(self.concept_labels)))
        
        for i, concept1 in enumerate(self.concept_labels):
            for j, concept2 in enumerate(self.concept_labels):
                if i != j:
                    # Get prototype activations
                    proto1 = self.concept_prototypes.get(concept1, np.zeros(self.competitive_layer_size))
                    proto2 = self.concept_prototypes.get(concept2, np.zeros(self.competitive_layer_size))
                    
                    # Calculate cosine similarity
                    norm1 = np.linalg.norm(proto1)
                    norm2 = np.linalg.norm(proto2)
                    
                    if norm1 > 0 and norm2 > 0:
                        similarity = np.dot(proto1, proto2) / (norm1 * norm2)
                    else:
                        similarity = 0
                    
                    separability_matrix[i, j] = similarity
        
        # Calculate average separability (lower is better)
        avg_similarity = np.mean(separability_matrix[separability_matrix > 0])
        
        print(f"   Average inter-concept similarity: {avg_similarity:.3f}")
        print(f"   Separability (lower=better): {avg_similarity:.3f}")
        
        return {
            "separability_matrix": separability_matrix.tolist(),
            "average_similarity": avg_similarity,
            "concept_labels": self.concept_labels
        }
    
    def run_full_experiment(self) -> Dict:
        """Run complete sparse competitive learning experiment."""
        print("=" * 60)
        print("🚀 SPARSE COMPETITIVE CONCEPT FORMATION EXPERIMENT")
        print("=" * 60)
        print(f"Architecture: {self.input_size}→{self.feature_encoder_size}→{self.competitive_layer_size}→{self.memory_slots}")
        print(f"Sparsity: {self.k_sparse}/{self.competitive_layer_size} neurons active")
        print(f"Lateral inhibition: {self.inhibition_strength}")
        
        experiment_log = {
            "timestamp": datetime.now().isoformat(),
            "architecture": {
                "input_size": self.input_size,
                "feature_encoder_size": self.feature_encoder_size,
                "competitive_layer_size": self.competitive_layer_size,
                "memory_slots": self.memory_slots,
                "k_sparse": self.k_sparse,
                "inhibition_strength": self.inhibition_strength
            },
            "training_logs": {},
            "recognition_tests": {},
            "separability_analysis": {},
            "synaptic_changes": 0
        }
        
        # Phase 1: Train all concepts
        print("\n" + "="*40)
        print("PHASE 1: CONCEPT TRAINING")
        print("="*40)
        
        for concept in self.concept_labels:
            training_log = self.train_concept(concept, iterations=50)
            experiment_log["training_logs"][concept] = training_log
        
        # Phase 2: Test recognition
        print("\n" + "="*40)
        print("PHASE 2: RECOGNITION TESTING")
        print("="*40)
        
        recognition_accuracy = []
        for concept in self.concept_labels:
            test_result = self.test_concept_recognition(concept, noise_level=0.1)
            experiment_log["recognition_tests"][concept] = test_result
            recognition_accuracy.append(test_result["correct"])
        
        overall_accuracy = np.mean(recognition_accuracy)
        print(f"\n📈 Overall Recognition Accuracy: {overall_accuracy:.1%}")
        
        # Phase 3: Separability analysis
        print("\n" + "="*40)
        print("PHASE 3: SEPARABILITY ANALYSIS")
        print("="*40)
        
        separability = self.analyze_concept_separability()
        experiment_log["separability_analysis"] = separability
        
        # Final metrics
        experiment_log["synaptic_changes"] = self.synaptic_changes
        experiment_log["overall_accuracy"] = overall_accuracy
        experiment_log["average_sparsity"] = np.mean(self.activation_sparsity)
        
        print("\n" + "="*60)
        print("🎯 EXPERIMENT SUMMARY")
        print("="*60)
        print(f"✅ Recognition Accuracy: {overall_accuracy:.1%}")
        print(f"🧠 Average Sparsity: {np.mean(self.activation_sparsity):.3f}")
        print(f"🔗 Synaptic Changes: {self.synaptic_changes:,.0f}")
        print(f"📊 Concept Separability: {separability['average_similarity']:.3f}")
        
        # Critical test: Check if binding problem is solved
        if overall_accuracy >= 0.75 and separability['average_similarity'] < 0.5:
            print("🎉 SUCCESS: Binding problem appears to be SOLVED!")
            print("   - High recognition accuracy")
            print("   - Low inter-concept similarity")
            print("   - Stable sparse representations")
        else:
            print("⚠️  PARTIAL SUCCESS: Improvements seen but binding problem persists")
            if overall_accuracy < 0.75:
                print(f"   - Recognition accuracy still low: {overall_accuracy:.1%}")
            if separability['average_similarity'] >= 0.5:
                print(f"   - Concepts still too similar: {separability['average_similarity']:.3f}")
        
        return experiment_log

def main():
    """Run the sparse competitive learning experiment."""
    
    # Create network with suggested parameters
    network = SparseCompetitiveNetwork(k_sparse=3, inhibition_strength=0.8)
    
    # Run full experiment
    results = network.run_full_experiment()
    
    # Save results
    results_file = "sparse_competitive_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Results saved to: {results_file}")
    
    # Create visualization
    create_visualization(network, results)

def create_visualization(network: SparseCompetitiveNetwork, results: Dict):
    """Create visualization of sparse competitive learning results."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Sparse Competitive Learning: Solving the Binding Problem', fontsize=16, fontweight='bold')
    
    # 1. Concept prototype activations
    ax1 = axes[0, 0]
    for i, concept in enumerate(network.concept_labels):
        if concept in network.concept_prototypes:
            activations = network.concept_prototypes[concept]
            ax1.bar(range(len(activations)), activations, alpha=0.7, label=concept)
    ax1.set_title('Concept Prototype Activations\n(Sparse Representations)', fontweight='bold')
    ax1.set_xlabel('Competitive Layer Neurons')
    ax1.set_ylabel('Activation Strength')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Sparsity over training
    ax2 = axes[0, 1]
    ax2.plot(network.activation_sparsity, 'b-', linewidth=2)
    ax2.axhline(y=network.k_sparse/network.competitive_layer_size, color='r', linestyle='--', 
                label=f'Target Sparsity ({network.k_sparse}/{network.competitive_layer_size})')
    ax2.set_title('Activation Sparsity During Training\n(Winner-Take-All Dynamics)', fontweight='bold')
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Fraction of Active Neurons')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Recognition accuracy
    ax3 = axes[0, 2]
    concepts = list(results["recognition_tests"].keys())
    accuracies = [1.0 if results["recognition_tests"][c]["correct"] else 0.0 for c in concepts]
    colors = ['green' if acc == 1.0 else 'red' for acc in accuracies]
    bars = ax3.bar(concepts, accuracies, color=colors, alpha=0.7)
    ax3.set_title('Concept Recognition Accuracy\n(Binding Problem Test)', fontweight='bold')
    ax3.set_ylabel('Recognition Accuracy')
    ax3.set_ylim(0, 1.1)
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{acc:.0%}', ha='center', va='bottom', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Separability matrix
    ax4 = axes[1, 0]
    sep_matrix = np.array(results["separability_analysis"]["separability_matrix"])
    im = ax4.imshow(sep_matrix, cmap='RdYlBu_r', vmin=0, vmax=1)
    ax4.set_title('Inter-Concept Similarity Matrix\n(Lower = Better Separation)', fontweight='bold')
    ax4.set_xticks(range(len(concepts)))
    ax4.set_yticks(range(len(concepts)))
    ax4.set_xticklabels(concepts)
    ax4.set_yticklabels(concepts)
    plt.colorbar(im, ax=ax4, label='Similarity')
    
    # Add text annotations
    for i in range(len(concepts)):
        for j in range(len(concepts)):
            if i != j:
                text = ax4.text(j, i, f'{sep_matrix[i, j]:.2f}',
                              ha="center", va="center", color="black", fontweight='bold')
    
    # 5. Concept confidence scores
    ax5 = axes[1, 1]
    for concept in concepts:
        confidences = list(results["recognition_tests"][concept]["similarities"].values())
        ax5.bar(range(len(confidences)), confidences, alpha=0.7, label=f'Input: {concept}')
    ax5.set_title('Recognition Confidence Scores\n(Winner-Take-All Output)', fontweight='bold')
    ax5.set_xlabel('Recognized As')
    ax5.set_ylabel('Confidence Score')
    ax5.set_xticks(range(len(concepts)))
    ax5.set_xticklabels(concepts)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Learning progress summary
    ax6 = axes[1, 2]
    metrics = ['Recognition\nAccuracy', 'Concept\nSeparability', 'Sparsity\nControl']
    values = [
        results["overall_accuracy"],
        1 - results["separability_analysis"]["average_similarity"],  # Invert for better visualization
        1 - abs(results["average_sparsity"] - network.k_sparse/network.competitive_layer_size)  # Closeness to target
    ]
    colors = ['green' if v >= 0.75 else 'orange' if v >= 0.5 else 'red' for v in values]
    bars = ax6.bar(metrics, values, color=colors, alpha=0.7)
    ax6.set_title('Overall Performance Metrics\n(Success Indicators)', fontweight='bold')
    ax6.set_ylabel('Performance Score')
    ax6.set_ylim(0, 1.1)
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sparse_competitive_learning_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Visualization saved as: sparse_competitive_learning_results.png")

if __name__ == "__main__":
    main()
