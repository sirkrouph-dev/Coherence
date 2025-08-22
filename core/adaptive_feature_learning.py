#!/usr/bin/env python3
"""
Adaptive Feature Learning System
===============================

Task 6B.5: Adaptive feature learning system that implements experience-dependent 
feature development, unsupervised learning for feature extraction, and adaptive 
receptive field formation in sensory hierarchies.

Key features:
- Experience-dependent feature development
- Unsupervised learning algorithms (Hebbian, competitive learning)
- Adaptive receptive field formation and reorganization
- Feature map plasticity and self-organization
- Integration with existing sensory processing hierarchies
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import warnings

try:
    from .hierarchical_sensory_processing import (
        SensoryHierarchy, SensoryLayer, LayerConfig,
        ProcessingLevel
    )
    HIERARCHICAL_IMPORTS_AVAILABLE = True
except ImportError:
    HIERARCHICAL_IMPORTS_AVAILABLE = False
    # Minimal fallback classes
    from enum import Enum
    
    class ProcessingLevel(Enum):
        PRIMARY = "primary"
        SECONDARY = "secondary"
        ASSOCIATION = "association"
        INTEGRATION = "integration"
        
    @dataclass
    class LayerConfig:
        name: str
        level: ProcessingLevel
        size: int
        spatial_layout: Optional[Tuple[int, int]] = None
        
    class SensoryLayer:
        def __init__(self, config: LayerConfig, hierarchy_id: str = ""):
            self.config = config
            self.attention_modulation = 1.0


class LearningAlgorithm(Enum):
    """Types of adaptive learning algorithms."""
    HEBBIAN = "hebbian"
    COMPETITIVE = "competitive"
    SPARSE_CODING = "sparse_coding"
    PCA = "pca"
    SELF_ORGANIZING = "self_organizing"


class PlasticityType(Enum):
    """Types of plasticity mechanisms."""
    STRUCTURAL = "structural"     # Changes in connectivity
    FUNCTIONAL = "functional"     # Changes in weights
    HOMEOSTATIC = "homeostatic"   # Activity regulation
    DEVELOPMENTAL = "developmental" # Experience-dependent development


@dataclass
class AdaptiveLearningConfig:
    """Configuration for adaptive learning mechanisms."""
    algorithm: LearningAlgorithm = LearningAlgorithm.HEBBIAN
    learning_rate: float = 0.01
    adaptation_window: int = 1000    # Number of samples for adaptation
    plasticity_threshold: float = 0.1
    competition_strength: float = 0.5
    homeostatic_rate: float = 0.001
    sparsity_target: float = 0.05    # Target sparsity for sparse coding
    feature_decay_rate: float = 0.0001


@dataclass
class ReceptiveFieldConfig:
    """Configuration for adaptive receptive fields."""
    initial_size: int = 5
    max_size: int = 15
    min_size: int = 3
    adaptation_rate: float = 0.005
    overlap_threshold: float = 0.3
    pruning_threshold: float = 0.01
    growth_threshold: float = 0.8


class HebbianLearning:
    """
    Implements Hebbian learning rule: "Neurons that fire together, wire together"
    """
    
    def __init__(self, config: AdaptiveLearningConfig):
        """Initialize Hebbian learning."""
        self.config = config
        
        # Learning state
        self.weight_matrix = None
        self.activity_history = []
        self.correlation_matrix = None
        
        print(f"HebbianLearning initialized with learning rate: {config.learning_rate}")
        
    def initialize_weights(self, input_size: int, output_size: int):
        """Initialize weight matrix."""
        self.weight_matrix = np.random.normal(0, 0.1, (output_size, input_size))
        self.correlation_matrix = np.zeros((input_size, input_size))
        
    def update_weights(self, input_activity: np.ndarray, output_activity: np.ndarray):
        """Update weights using Hebbian rule."""
        if self.weight_matrix is None:
            self.initialize_weights(len(input_activity), len(output_activity))
            
        # Hebbian update: Δw = η * pre * post
        for i in range(len(output_activity)):
            for j in range(len(input_activity)):
                delta_w = (self.config.learning_rate * 
                          input_activity[j] * output_activity[i])
                self.weight_matrix[i, j] += delta_w
                
        # Weight normalization to prevent unbounded growth
        for i in range(len(output_activity)):
            norm = np.linalg.norm(self.weight_matrix[i, :])
            if norm > 0:
                self.weight_matrix[i, :] /= norm
                
        # Update correlation matrix
        self._update_correlations(input_activity)
        
    def _update_correlations(self, input_activity: np.ndarray):
        """Update input correlation matrix."""
        if self.correlation_matrix is None:
            return
            
        # Exponential moving average of correlations
        alpha = 0.01
        outer_product = np.outer(input_activity, input_activity)
        self.correlation_matrix = ((1 - alpha) * self.correlation_matrix + 
                                  alpha * outer_product)
                                  
    def compute_output(self, input_activity: np.ndarray) -> np.ndarray:
        """Compute output using current weights."""
        if self.weight_matrix is None:
            return np.zeros(1)
            
        output = np.dot(self.weight_matrix, input_activity)
        
        # Apply activation function (sigmoid)
        return 1.0 / (1.0 + np.exp(-output))
        
    def get_feature_selectivity(self) -> np.ndarray:
        """Get selectivity of learned features."""
        if self.weight_matrix is None:
            return np.array([])
            
        # Compute selectivity as entropy of weight distribution
        selectivities = []
        for i in range(self.weight_matrix.shape[0]):
            weights = np.abs(self.weight_matrix[i, :])
            if np.sum(weights) > 0:
                normalized_weights = weights / np.sum(weights)
                entropy = -np.sum(normalized_weights * np.log(normalized_weights + 1e-8))
                selectivity = 1.0 - entropy / np.log(len(weights))  # Normalized selectivity
                selectivities.append(selectivity)
            else:
                selectivities.append(0.0)
                
        return np.array(selectivities)


class CompetitiveLearning:
    """
    Implements competitive learning for feature extraction and clustering.
    """
    
    def __init__(self, config: AdaptiveLearningConfig):
        """Initialize competitive learning."""
        self.config = config
        
        # Learning state
        self.weight_matrix = None
        self.neuron_activities = []
        self.winner_history = []
        
        print(f"CompetitiveLearning initialized with competition strength: {config.competition_strength}")
        
    def initialize_weights(self, input_size: int, output_size: int):
        """Initialize weight matrix with random weights."""
        self.weight_matrix = np.random.uniform(0, 1, (output_size, input_size))
        
        # Normalize weights
        for i in range(output_size):
            self.weight_matrix[i, :] /= np.linalg.norm(self.weight_matrix[i, :])
            
    def update_weights(self, input_activity: np.ndarray):
        """Update weights using competitive learning rule."""
        if self.weight_matrix is None:
            self.initialize_weights(len(input_activity), 10)  # Default 10 output neurons
            
        # Compute distances to all neurons
        distances = []
        for i in range(self.weight_matrix.shape[0]):
            distance = np.linalg.norm(input_activity - self.weight_matrix[i, :])
            distances.append(distance)
            
        # Find winner (minimum distance)
        winner_idx = np.argmin(distances)
        self.winner_history.append(winner_idx)
        
        # Update winner's weights towards input
        self.weight_matrix[winner_idx, :] += (
            self.config.learning_rate * 
            (input_activity - self.weight_matrix[winner_idx, :])
        )
        
        # Normalize winner's weights
        norm = np.linalg.norm(self.weight_matrix[winner_idx, :])
        if norm > 0:
            self.weight_matrix[winner_idx, :] /= norm
            
        # Optional: lateral inhibition (reduce weights of nearby neurons)
        inhibition_strength = self.config.competition_strength * 0.1
        for i in range(self.weight_matrix.shape[0]):
            if i != winner_idx:
                self.weight_matrix[i, :] -= (
                    inhibition_strength * self.config.learning_rate *
                    (input_activity - self.weight_matrix[i, :]) * 
                    np.exp(-abs(i - winner_idx))  # Distance-based inhibition
                )
                
    def compute_output(self, input_activity: np.ndarray) -> np.ndarray:
        """Compute competitive output (winner-take-all)."""
        if self.weight_matrix is None:
            return np.zeros(1)
            
        # Compute distances
        distances = []
        for i in range(self.weight_matrix.shape[0]):
            distance = np.linalg.norm(input_activity - self.weight_matrix[i, :])
            distances.append(distance)
            
        # Winner-take-all output
        winner_idx = np.argmin(distances)
        output = np.zeros(self.weight_matrix.shape[0])
        output[winner_idx] = 1.0
        
        return output
        
    def get_cluster_centers(self) -> np.ndarray:
        """Get learned cluster centers (weight vectors)."""
        return self.weight_matrix.copy() if self.weight_matrix is not None else np.array([])


class AdaptiveReceptiveField:
    """
    Implements adaptive receptive fields that can grow, shrink, and reorganize
    based on experience and activity patterns.
    """
    
    def __init__(self, config: ReceptiveFieldConfig, initial_position: Tuple[int, int]):
        """Initialize adaptive receptive field."""
        self.config = config
        self.center_position = initial_position
        self.current_size = config.initial_size
        
        # Receptive field state
        self.activity_map = None
        self.connection_strengths = {}
        self.adaptation_history = []
        
        print(f"AdaptiveReceptiveField initialized at position {initial_position}, size {config.initial_size}")
        
    def update_receptive_field(self, input_map: np.ndarray, activity_pattern: np.ndarray):
        """Update receptive field based on input statistics and activity."""
        
        if self.activity_map is None:
            self.activity_map = np.zeros_like(input_map)
            
        # Update activity map with exponential moving average
        alpha = 0.05
        self.activity_map = (1 - alpha) * self.activity_map + alpha * activity_pattern
        
        # Analyze activity statistics within current receptive field
        rf_bounds = self._get_receptive_field_bounds(input_map.shape)
        rf_activity = self.activity_map[rf_bounds[0]:rf_bounds[1], rf_bounds[2]:rf_bounds[3]]
        
        # Compute activity metrics
        mean_activity = np.mean(rf_activity)
        activity_variance = np.var(rf_activity)
        edge_activity = self._compute_edge_activity(rf_activity)
        
        # Adaptation decisions
        self._adapt_receptive_field_size(mean_activity, edge_activity)
        self._adapt_receptive_field_position(input_map.shape)
        
        # Store adaptation event
        self.adaptation_history.append({
            'size': self.current_size,
            'position': self.center_position,
            'mean_activity': mean_activity,
            'edge_activity': edge_activity
        })
        
    def _get_receptive_field_bounds(self, input_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Get current receptive field bounds."""
        half_size = self.current_size // 2
        
        top = max(0, self.center_position[0] - half_size)
        bottom = min(input_shape[0], self.center_position[0] + half_size + 1)
        left = max(0, self.center_position[1] - half_size)
        right = min(input_shape[1], self.center_position[1] + half_size + 1)
        
        return (top, bottom, left, right)
        
    def _compute_edge_activity(self, rf_activity: np.ndarray) -> float:
        """Compute activity at receptive field edges."""
        if rf_activity.size == 0:
            return 0.0
            
        # Activity at border pixels
        if rf_activity.shape[0] > 2 and rf_activity.shape[1] > 2:
            edge_pixels = np.concatenate([
                rf_activity[0, :],      # top edge
                rf_activity[-1, :],     # bottom edge
                rf_activity[:, 0],      # left edge
                rf_activity[:, -1]      # right edge
            ])
            return np.mean(edge_pixels)
        else:
            return np.mean(rf_activity)
            
    def _adapt_receptive_field_size(self, mean_activity: float, edge_activity: float):
        """Adapt receptive field size based on activity patterns."""
        
        # Grow if edge activity is high (information at borders)
        if edge_activity > self.config.growth_threshold * mean_activity:
            if self.current_size < self.config.max_size:
                self.current_size += 1
                
        # Shrink if overall activity is low or edge activity is very low
        elif (mean_activity < self.config.pruning_threshold or 
              edge_activity < self.config.pruning_threshold * mean_activity):
            if self.current_size > self.config.min_size:
                self.current_size -= 1
                
    def _adapt_receptive_field_position(self, input_shape: Tuple[int, int]):
        """Adapt receptive field position based on activity gradients."""
        
        if self.activity_map is None:
            return
            
        # Compute activity gradient around current position
        y, x = self.center_position
        window_size = 3
        
        # Sample activity in neighboring positions
        positions = [
            (y-1, x), (y+1, x),    # vertical neighbors
            (y, x-1), (y, x+1),    # horizontal neighbors
        ]
        
        best_position = self.center_position
        best_activity = self.activity_map[y, x] if 0 <= y < input_shape[0] and 0 <= x < input_shape[1] else 0
        
        for new_y, new_x in positions:
            if (0 <= new_y < input_shape[0] and 0 <= new_x < input_shape[1]):
                activity = self.activity_map[new_y, new_x]
                if activity > best_activity * (1 + self.config.adaptation_rate):
                    best_position = (new_y, new_x)
                    best_activity = activity
                    
        self.center_position = best_position
        
    def get_receptive_field_mask(self, input_shape: Tuple[int, int]) -> np.ndarray:
        """Get binary mask for current receptive field."""
        mask = np.zeros(input_shape, dtype=bool)
        bounds = self._get_receptive_field_bounds(input_shape)
        mask[bounds[0]:bounds[1], bounds[2]:bounds[3]] = True
        return mask
        
    def compute_response(self, input_data: np.ndarray) -> float:
        """Compute response to input data."""
        if len(input_data.shape) != 2:
            return 0.0
            
        rf_mask = self.get_receptive_field_mask(input_data.shape)
        rf_input = input_data[rf_mask]
        
        # Simple response: mean activity in receptive field
        return np.mean(rf_input) if len(rf_input) > 0 else 0.0


class AdaptiveFeatureLearningLayer(SensoryLayer):
    """
    Sensory layer with adaptive feature learning capabilities.
    Combines multiple learning algorithms and adaptive receptive fields.
    """
    
    def __init__(self, config: LayerConfig, 
                 learning_config: AdaptiveLearningConfig,
                 rf_config: ReceptiveFieldConfig,
                 hierarchy_id: str = ""):
        """Initialize adaptive feature learning layer."""
        super().__init__(config, hierarchy_id)
        
        self.learning_config = learning_config
        self.rf_config = rf_config
        
        # Learning components
        self.hebbian_learner = None
        self.competitive_learner = None
        self.receptive_fields = []
        
        # Feature learning state
        self.learned_features = {}
        self.feature_responses = {}
        self.adaptation_stats = {}
        
        # Initialize learning algorithms
        self._initialize_learning_algorithms()
        
        # Initialize adaptive receptive fields
        self._initialize_receptive_fields()
        
        print(f"AdaptiveFeatureLearningLayer initialized: {config.size} neurons, {learning_config.algorithm.value} learning")
        
    def _initialize_learning_algorithms(self):
        """Initialize learning algorithms based on configuration."""
        
        if (self.learning_config.algorithm == LearningAlgorithm.HEBBIAN or 
            self.learning_config.algorithm == LearningAlgorithm.SPARSE_CODING):
            self.hebbian_learner = HebbianLearning(self.learning_config)
            
        if (self.learning_config.algorithm == LearningAlgorithm.COMPETITIVE or
            self.learning_config.algorithm == LearningAlgorithm.SELF_ORGANIZING):
            self.competitive_learner = CompetitiveLearning(self.learning_config)
            
    def _initialize_receptive_fields(self):
        """Initialize adaptive receptive fields."""
        
        if self.config.spatial_layout:
            height, width = self.config.spatial_layout
            
            # Create grid of receptive fields
            for i in range(height):
                for j in range(width):
                    if len(self.receptive_fields) < self.config.size:
                        rf_position = (i, j)
                        rf = AdaptiveReceptiveField(self.rf_config, rf_position)
                        self.receptive_fields.append(rf)
        else:
            # Create single receptive field for non-spatial layers
            rf_position = (0, 0)
            rf = AdaptiveReceptiveField(self.rf_config, rf_position)
            self.receptive_fields.append(rf)
            
    def process_input(self, input_data: np.ndarray) -> np.ndarray:
        """Process input with adaptive feature learning."""
        
        # Ensure input is 2D for spatial processing
        if len(input_data.shape) == 1:
            side_length = int(np.sqrt(len(input_data)))
            if side_length * side_length == len(input_data):
                input_data = input_data.reshape(side_length, side_length)
            else:
                # Pad to square
                target_size = int(np.ceil(np.sqrt(len(input_data))))
                padded = np.zeros(target_size * target_size)
                padded[:len(input_data)] = input_data
                input_data = padded.reshape(target_size, target_size)
                
        responses = np.zeros(self.config.size)
        
        # Process through adaptive receptive fields
        for i, rf in enumerate(self.receptive_fields[:self.config.size]):
            # Compute response
            response = rf.compute_response(input_data)
            responses[i] = response
            
            # Update receptive field based on activity
            activity_pattern = np.abs(input_data) > np.mean(input_data)
            rf.update_receptive_field(input_data, activity_pattern.astype(float))
            
        # Apply learning algorithms
        if len(responses) > 0:
            self._apply_learning(input_data.flatten()[:100], responses[:100])  # Limit size for efficiency
            
        # Apply attention modulation
        responses *= self.attention_modulation
        
        return responses
        
    def _apply_learning(self, input_vector: np.ndarray, output_vector: np.ndarray):
        """Apply learning algorithms to update features."""
        
        if self.hebbian_learner is not None:
            self.hebbian_learner.update_weights(input_vector, output_vector)
            
            # Store learned features
            if self.hebbian_learner.weight_matrix is not None:
                self.learned_features['hebbian_weights'] = self.hebbian_learner.weight_matrix.copy()
                
        if self.competitive_learner is not None:
            self.competitive_learner.update_weights(input_vector)
            
            # Store cluster centers
            if self.competitive_learner.weight_matrix is not None:
                self.learned_features['cluster_centers'] = self.competitive_learner.get_cluster_centers()
                
    def get_learned_features(self) -> Dict[str, np.ndarray]:
        """Get currently learned features."""
        features = self.learned_features.copy()
        
        # Add receptive field information
        rf_info = []
        for rf in self.receptive_fields:
            rf_info.append({
                'center': rf.center_position,
                'size': rf.current_size,
                'adaptation_count': len(rf.adaptation_history)
            })
        features['receptive_fields'] = rf_info
        
        return features
        
    def get_feature_selectivity(self) -> Dict[str, float]:
        """Get selectivity metrics for learned features."""
        selectivity = {}
        
        if self.hebbian_learner is not None:
            hebbian_selectivity = self.hebbian_learner.get_feature_selectivity()
            if len(hebbian_selectivity) > 0:
                selectivity['hebbian_mean'] = np.mean(hebbian_selectivity)
                selectivity['hebbian_std'] = np.std(hebbian_selectivity)
                
        # Receptive field adaptation metrics
        if self.receptive_fields:
            sizes = [rf.current_size for rf in self.receptive_fields]
            selectivity['rf_size_mean'] = np.mean(sizes)
            selectivity['rf_size_std'] = np.std(sizes)
            
            adaptations = [len(rf.adaptation_history) for rf in self.receptive_fields]
            selectivity['adaptation_count'] = np.mean(adaptations)
            
        return selectivity


def demo_adaptive_feature_learning():
    """Demonstrate the adaptive feature learning system."""
    
    print("=== Adaptive Feature Learning System Demo ===")
    
    # Create adaptive feature learning layer
    print("\n1. Creating Adaptive Feature Learning Layer")
    
    layer_config = LayerConfig(
        name="AdaptiveLayer",
        level=ProcessingLevel.PRIMARY,
        size=16,
        spatial_layout=(4, 4)
    )
    
    learning_config = AdaptiveLearningConfig(
        algorithm=LearningAlgorithm.HEBBIAN,
        learning_rate=0.02,
        adaptation_window=100
    )
    
    rf_config = ReceptiveFieldConfig(
        initial_size=3,
        max_size=7,
        adaptation_rate=0.01
    )
    
    adaptive_layer = AdaptiveFeatureLearningLayer(
        layer_config, learning_config, rf_config
    )
    
    print(f"Created adaptive layer with {len(adaptive_layer.receptive_fields)} receptive fields")
    
    # Test adaptive learning with structured inputs
    print("\n2. Testing Adaptive Learning with Structured Inputs")
    
    # Create sequence of input patterns
    input_patterns = []
    
    # Horizontal stripes
    for i in range(5):
        pattern = np.zeros((8, 8))
        pattern[2:6, :] = 1.0
        pattern += np.random.normal(0, 0.1, (8, 8))
        input_patterns.append(pattern)
        
    # Vertical stripes  
    for i in range(5):
        pattern = np.zeros((8, 8))
        pattern[:, 2:6] = 1.0
        pattern += np.random.normal(0, 0.1, (8, 8))
        input_patterns.append(pattern)
        
    # Diagonal patterns
    for i in range(5):
        pattern = np.zeros((8, 8))
        for j in range(8):
            if 0 <= j-2 < 8:
                pattern[j, j-2:j+1] = 1.0
        pattern += np.random.normal(0, 0.1, (8, 8))
        input_patterns.append(pattern)
        
    print(f"Created {len(input_patterns)} training patterns")
    
    # Train the layer
    responses_history = []
    
    for epoch in range(3):
        for i, pattern in enumerate(input_patterns):
            response = adaptive_layer.process_input(pattern)
            responses_history.append(np.mean(response))
            
    print(f"Training completed: {len(responses_history)} iterations")
    print(f"Final average response: {np.mean(responses_history[-10:]):.3f}")
    
    # Test learned features
    print("\n3. Analyzing Learned Features")
    
    learned_features = adaptive_layer.get_learned_features()
    
    print("Learned features:")
    for feature_type, feature_data in learned_features.items():
        if isinstance(feature_data, np.ndarray):
            print(f"  {feature_type}: shape {feature_data.shape}")
        elif isinstance(feature_data, list):
            print(f"  {feature_type}: {len(feature_data)} items")
            
    # Test feature selectivity
    selectivity = adaptive_layer.get_feature_selectivity()
    
    print("Feature selectivity metrics:")
    for metric, value in selectivity.items():
        print(f"  {metric}: {value:.4f}")
        
    # Test receptive field adaptation
    print("\n4. Testing Receptive Field Adaptation")
    
    # Test with focused input
    focused_pattern = np.zeros((8, 8))
    focused_pattern[3:5, 3:5] = 2.0  # Strong localized activity
    
    initial_sizes = [rf.current_size for rf in adaptive_layer.receptive_fields[:4]]
    
    # Apply focused input multiple times
    for _ in range(20):
        adaptive_layer.process_input(focused_pattern)
        
    final_sizes = [rf.current_size for rf in adaptive_layer.receptive_fields[:4]]
    
    print(f"Receptive field size changes (first 4 RFs):")
    for i in range(4):
        print(f"  RF {i}: {initial_sizes[i]} → {final_sizes[i]}")
        
    # Test competitive learning
    print("\n5. Testing Competitive Learning")
    
    comp_config = AdaptiveLearningConfig(
        algorithm=LearningAlgorithm.COMPETITIVE,
        learning_rate=0.05,
        competition_strength=0.8
    )
    
    comp_layer = AdaptiveFeatureLearningLayer(
        layer_config, comp_config, rf_config
    )
    
    # Train with clustered data
    cluster_patterns = []
    
    # Cluster 1: top-left activity
    for _ in range(10):
        pattern = np.zeros((8, 8))
        pattern[1:4, 1:4] = np.random.rand(3, 3)
        cluster_patterns.append(pattern)
        
    # Cluster 2: bottom-right activity
    for _ in range(10):
        pattern = np.zeros((8, 8))
        pattern[4:7, 4:7] = np.random.rand(3, 3)
        cluster_patterns.append(pattern)
        
    for pattern in cluster_patterns:
        comp_layer.process_input(pattern)
        
    comp_features = comp_layer.get_learned_features()
    
    if 'cluster_centers' in comp_features:
        centers = comp_features['cluster_centers']
        print(f"Learned {centers.shape[0]} cluster centers")
    else:
        print("Competitive learning in progress...")
        
    print("\n✅ Adaptive Feature Learning Demo Complete!")
    
    return adaptive_layer, comp_layer


if __name__ == "__main__":
    # Run demonstration
    adaptive_layer, comp_layer = demo_adaptive_feature_learning()
    
    print("\n=== Task 6B.5 Implementation Summary ===")
    print("✅ Adaptive Feature Learning System - COMPLETED")
    print("\nKey Features Implemented:")
    print("  • HebbianLearning with correlation-based weight updates")
    print("  • CompetitiveLearning with winner-take-all dynamics")  
    print("  • AdaptiveReceptiveField with size and position adaptation")
    print("  • Experience-dependent feature development")
    print("  • Unsupervised learning for feature extraction")
    print("  • Integration with sensory processing hierarchies")
    
    print("\nNext Steps:")
    print("  → Task 7: Working Memory System Implementation")
    print("  → Task 8: Attention Mechanism Implementation")