"""
Enhanced Symbol Detection Module
Implements sophisticated cell assembly detection and temporal binding
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time
from collections import deque


@dataclass
class EnhancedCellAssembly:
    """Enhanced cell assembly with richer dynamics"""
    
    neurons: List[int]
    activation_pattern: np.ndarray
    stability_score: float
    emergence_history: List[float]
    temporal_signature: np.ndarray  # Temporal activation pattern
    binding_strength: float
    coherence_score: float
    concept_grounding: Dict[str, float]
    
    # New enhanced features
    phase_coherence: float = 0.0
    synchrony_index: float = 0.0
    plasticity_trace: Optional[np.ndarray] = None
    prediction_error: float = 0.0


class TemporalBinding:
    """Advanced temporal binding mechanism"""
    
    def __init__(self, window_size: int = 50, decay_rate: float = 0.95):
        self.window_size = window_size
        self.decay_rate = decay_rate
        self.temporal_buffer = deque(maxlen=window_size)
        self.binding_matrix = np.zeros((window_size, window_size))
        
    def add_activity_pattern(self, pattern: np.ndarray, timestamp: float):
        """Add new activity pattern to temporal buffer"""
        self.temporal_buffer.append({
            'pattern': pattern,
            'timestamp': timestamp,
            'activity_level': np.mean(pattern)
        })
        
        # Update binding matrix
        self._update_binding_matrix()
    
    def _update_binding_matrix(self):
        """Update temporal binding matrix"""
        buffer_len = len(self.temporal_buffer)
        if buffer_len < 2:
            return
            
        # Calculate cross-correlations between patterns
        for i in range(buffer_len):
            for j in range(i + 1, buffer_len):
                pattern_i = self.temporal_buffer[i]['pattern']
                pattern_j = self.temporal_buffer[j]['pattern']
                
                # Calculate normalized cross-correlation
                if len(pattern_i) > 0 and len(pattern_j) > 0:
                    min_len = min(len(pattern_i), len(pattern_j))
                    corr = np.corrcoef(pattern_i[:min_len], pattern_j[:min_len])[0, 1]
                    if not np.isnan(corr):
                        self.binding_matrix[i, j] = corr
                        self.binding_matrix[j, i] = corr
    
    def detect_temporal_patterns(self) -> List[Dict]:
        """Detect repeating temporal patterns"""
        patterns = []
        
        if len(self.temporal_buffer) < 3:
            return patterns
        
        # Look for recurring patterns in the binding matrix
        threshold = 0.7  # Correlation threshold for pattern detection
        
        for i in range(len(self.temporal_buffer) - 2):
            for j in range(i + 2, len(self.temporal_buffer)):
                if self.binding_matrix[i, j] > threshold:
                    # Found potential recurring pattern
                    pattern_info = {
                        'start_idx': i,
                        'end_idx': j,
                        'correlation': self.binding_matrix[i, j],
                        'duration': self.temporal_buffer[j]['timestamp'] - self.temporal_buffer[i]['timestamp'],
                        'strength': (self.temporal_buffer[i]['activity_level'] + 
                                   self.temporal_buffer[j]['activity_level']) / 2
                    }
                    patterns.append(pattern_info)
        
        return patterns


class EnhancedAssemblyDetector:
    """Enhanced cell assembly detection with temporal dynamics"""
    
    def __init__(self, layer_size: int, detection_threshold: float = 0.6):
        self.layer_size = layer_size
        self.detection_threshold = detection_threshold
        self.temporal_binding = TemporalBinding()
        self.assembly_history = deque(maxlen=100)
        self.stability_tracker = {}
        
    def detect_enhanced_assemblies(
        self,
        activity_matrix: np.ndarray,
        temporal_context: Optional[np.ndarray] = None
    ) -> List[EnhancedCellAssembly]:
        """Detect enhanced cell assemblies with temporal binding"""
        
        current_time = time.time()
        assemblies = []
        
        if activity_matrix.size == 0:
            return assemblies
        
        # Add current activity to temporal buffer
        mean_activity = np.mean(activity_matrix, axis=0) if activity_matrix.ndim > 1 else activity_matrix
        self.temporal_binding.add_activity_pattern(mean_activity, current_time)
        
        # Detect spatial clusters (cell assemblies)
        spatial_assemblies = self._detect_spatial_clusters(activity_matrix)
        
        # Enhance with temporal information
        for assembly in spatial_assemblies:
            enhanced_assembly = self._enhance_with_temporal_info(assembly, current_time)
            if enhanced_assembly.stability_score > self.detection_threshold:
                assemblies.append(enhanced_assembly)
        
        # Update assembly history
        self.assembly_history.append({
            'timestamp': current_time,
            'assemblies': assemblies,
            'activity_level': np.mean(activity_matrix)
        })
        
        return assemblies
    
    def _detect_spatial_clusters(self, activity_matrix: np.ndarray) -> List[Dict]:
        """Detect spatial clusters in activity matrix"""
        clusters = []
        
        if activity_matrix.ndim == 1:
            activity_vector = activity_matrix
        else:
            activity_vector = np.mean(activity_matrix, axis=0)
        
        # Simple threshold-based clustering
        active_neurons = np.where(activity_vector > np.mean(activity_vector) + np.std(activity_vector))[0]
        
        if len(active_neurons) > 0:
            cluster = {
                'neurons': active_neurons.tolist(),
                'activation_pattern': activity_vector[active_neurons],
                'centroid': np.mean(active_neurons),
                'spread': np.std(active_neurons) if len(active_neurons) > 1 else 0
            }
            clusters.append(cluster)
        
        return clusters
    
    def _enhance_with_temporal_info(self, spatial_cluster: Dict, timestamp: float) -> EnhancedCellAssembly:
        """Enhance spatial cluster with temporal information"""
        
        neurons = spatial_cluster['neurons']
        activation_pattern = spatial_cluster['activation_pattern']
        
        # Calculate temporal signature
        temporal_patterns = self.temporal_binding.detect_temporal_patterns()
        temporal_signature = self._calculate_temporal_signature(temporal_patterns)
        
        # Calculate stability score
        stability_score = self._calculate_enhanced_stability(neurons, timestamp)
        
        # Calculate binding strength
        binding_strength = self._calculate_binding_strength(activation_pattern)
        
        # Calculate coherence
        coherence_score = self._calculate_coherence(activation_pattern, temporal_signature)
        
        # Calculate phase coherence
        phase_coherence = self._calculate_phase_coherence(activation_pattern)
        
        # Calculate synchrony index
        synchrony_index = self._calculate_synchrony_index(neurons, timestamp)
        
        return EnhancedCellAssembly(
            neurons=neurons,
            activation_pattern=activation_pattern,
            stability_score=stability_score,
            emergence_history=[stability_score],
            temporal_signature=temporal_signature,
            binding_strength=binding_strength,
            coherence_score=coherence_score,
            concept_grounding={},
            phase_coherence=phase_coherence,
            synchrony_index=synchrony_index,
            plasticity_trace=np.zeros_like(activation_pattern),
            prediction_error=0.0
        )
    
    def _calculate_temporal_signature(self, temporal_patterns: List[Dict]) -> np.ndarray:
        """Calculate temporal signature from patterns"""
        if not temporal_patterns:
            return np.array([0.0])
        
        # Extract key temporal features
        features = []
        for pattern in temporal_patterns:
            features.extend([
                pattern['correlation'],
                pattern['duration'],
                pattern['strength']
            ])
        
        return np.array(features[:10])  # Limit to 10 features
    
    def _calculate_enhanced_stability(self, neurons: List[int], timestamp: float) -> float:
        """Calculate enhanced stability score"""
        neuron_key = tuple(sorted(neurons))
        
        if neuron_key not in self.stability_tracker:
            self.stability_tracker[neuron_key] = {
                'first_seen': timestamp,
                'last_seen': timestamp,
                'occurrence_count': 1,
                'stability_history': [1.0]
            }
            return 1.0
        
        tracker = self.stability_tracker[neuron_key]
        tracker['last_seen'] = timestamp
        tracker['occurrence_count'] += 1
        
        # Calculate stability based on recurrence and temporal consistency
        time_span = timestamp - tracker['first_seen']
        recurrence_rate = tracker['occurrence_count'] / max(time_span, 1.0)
        
        stability = min(recurrence_rate * 10, 1.0)  # Normalize to [0, 1]
        tracker['stability_history'].append(stability)
        
        return stability
    
    def _calculate_binding_strength(self, activation_pattern: np.ndarray) -> float:
        """Calculate binding strength of the pattern"""
        if len(activation_pattern) == 0:
            return 0.0
        
        # Measure how well the pattern is bound together
        mean_activation = np.mean(activation_pattern)
        variance = np.var(activation_pattern)
        
        # Strong binding = high mean activation with low variance
        if variance == 0:
            return float(mean_activation)
        
        return float(mean_activation / (1 + variance))
    
    def _calculate_coherence(self, activation_pattern: np.ndarray, temporal_signature: np.ndarray) -> float:
        """Calculate coherence between spatial and temporal patterns"""
        if len(activation_pattern) == 0 or len(temporal_signature) == 0:
            return 0.0
        
        # Simple coherence measure
        spatial_entropy = -np.sum(activation_pattern * np.log(activation_pattern + 1e-10))
        temporal_entropy = -np.sum(temporal_signature * np.log(np.abs(temporal_signature) + 1e-10))
        
        # Coherence is inverse of total entropy
        total_entropy = spatial_entropy + temporal_entropy
        return 1.0 / (1.0 + total_entropy)
    
    def _calculate_phase_coherence(self, activation_pattern: np.ndarray) -> float:
        """Calculate phase coherence of activation pattern"""
        if len(activation_pattern) < 2:
            return 0.0
        
        # Calculate phase relationships in the pattern
        fft = np.fft.fft(activation_pattern)
        phases = np.angle(fft)
        
        # Measure phase coherence as consistency of phase differences
        phase_diffs = np.diff(phases)
        phase_coherence = 1.0 - float(np.std(phase_diffs)) / np.pi
        
        return max(0.0, phase_coherence)
    
    def _calculate_synchrony_index(self, neurons: List[int], timestamp: float) -> float:
        """Calculate synchrony index for the neuron group"""
        if len(neurons) < 2:
            return 0.0
        
        # Simple synchrony measure based on neuron proximity
        neuron_distances = []
        for i in range(len(neurons)):
            for j in range(i + 1, len(neurons)):
                distance = abs(neurons[i] - neurons[j])
                neuron_distances.append(distance)
        
        if not neuron_distances:
            return 0.0
        
        # High synchrony = neurons are close together
        mean_distance = np.mean(neuron_distances)
        max_possible_distance = max(neurons) - min(neurons)
        
        if max_possible_distance == 0:
            return 1.0
        
        synchrony = 1.0 - (float(mean_distance) / max_possible_distance)
        return max(0.0, synchrony)
    
    def get_assembly_statistics(self) -> Dict:
        """Get comprehensive assembly statistics"""
        if not self.assembly_history:
            return {}
        
        recent_assemblies = [entry['assemblies'] for entry in self.assembly_history]
        total_assemblies = sum(len(assemblies) for assemblies in recent_assemblies)
        
        # Calculate average properties
        all_assemblies = [assembly for assemblies in recent_assemblies for assembly in assemblies]
        
        if not all_assemblies:
            return {
                'total_detected': 0,
                'average_stability': 0.0,
                'average_coherence': 0.0,
                'average_synchrony': 0.0
            }
        
        return {
            'total_detected': total_assemblies,
            'average_stability': np.mean([a.stability_score for a in all_assemblies]),
            'average_coherence': np.mean([a.coherence_score for a in all_assemblies]),
            'average_binding_strength': np.mean([a.binding_strength for a in all_assemblies]),
            'average_phase_coherence': np.mean([a.phase_coherence for a in all_assemblies]),
            'average_synchrony': np.mean([a.synchrony_index for a in all_assemblies]),
            'unique_assemblies': len(self.stability_tracker)
        }
