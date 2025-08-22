#!/usr/bin/env python3
"""
Auditory Processing Hierarchy Implementation
==========================================

Task 6B.3: Auditory processing pipeline that implements biologically-inspired
auditory cortex processing: FrequencyAnalysisLayer → PatternDetectionLayer → SoundRecognitionLayer

Key features:
- Tonotopic frequency organization (A1-like)
- Spectral analysis and temporal pattern detection
- Sound recognition and classification
- Integration with hierarchical sensory processing framework
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import warnings

try:
    from .hierarchical_sensory_processing import (
        SensoryHierarchy, SensoryLayer, LayerConfig, 
        ConnectionConfig, HierarchyConfig,
        SensoryModalityType, ProcessingLevel
    )
    HIERARCHICAL_IMPORTS_AVAILABLE = True
except ImportError:
    HIERARCHICAL_IMPORTS_AVAILABLE = False
    # Fallback implementations
    from dataclasses import dataclass
    from enum import Enum
    
    class SensoryModalityType(Enum):
        AUDITORY = "auditory"
        
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
        neuron_type: str = "adex"
        spatial_layout: Optional[Tuple[int, int]] = None
        
    @dataclass
    class ConnectionConfig:
        source_layer: str
        target_layer: str
        connection_type: str = "feedforward"
        connection_probability: float = 0.3
        weight_scale: float = 1.0
        plasticity_rule: Optional[str] = "stdp"
        
    @dataclass
    class HierarchyConfig:
        modality: SensoryModalityType
        layers: List['LayerConfig']
        connections: List[ConnectionConfig]
        input_dimensions: Tuple[int, ...] = (64, 32)
        enable_feedback: bool = True
        enable_lateral_inhibition: bool = True
        
    class SensoryLayer:
        def __init__(self, config: LayerConfig, hierarchy_id: str = ""):
            self.config = config
            self.hierarchy_id = hierarchy_id
            self.attention_modulation = 1.0
            
    class SensoryHierarchy:
        def __init__(self, config: HierarchyConfig, network: Optional[Any] = None):
            self.config = config
            self.layers: Dict[str, SensoryLayer] = {}
            self.layer_order: List[str] = []
            
        def apply_attention_modulation(self, layer_name: str, attention_weight: float):
            pass
            
        def get_hierarchy_info(self) -> Dict[str, Any]:
            return {'hierarchy_id': '', 'modality': '', 'total_layers': 0}
            
        def process_sensory_input(self, input_data: np.ndarray) -> Dict[str, np.ndarray]:
            return {}


class AudioFeatureType(Enum):
    """Types of auditory features detected by layers."""
    FREQUENCY = "frequency"
    TEMPORAL_PATTERN = "temporal_pattern"
    SPECTRAL_PATTERN = "spectral_pattern"
    SOUND_OBJECT = "sound_object"


@dataclass
class AudioLayerConfig(LayerConfig):
    """Extended layer configuration for auditory processing layers."""
    feature_type: AudioFeatureType = AudioFeatureType.FREQUENCY
    frequency_range: Tuple[float, float] = (20.0, 20000.0)  # Hz
    n_frequency_bands: int = 64
    temporal_window: float = 0.025  # 25ms window
    overlap_factor: float = 0.5
    response_threshold: float = 0.1


class FrequencyAnalysisLayer(SensoryLayer):
    """
    Primary auditory layer for frequency analysis with tonotopic organization.
    Corresponds to primary auditory cortex (A1).
    """
    
    def __init__(self, config: AudioLayerConfig, hierarchy_id: str = ""):
        """Initialize frequency analysis layer."""
        super().__init__(config, hierarchy_id)
        
        self.frequency_responses = {}
        self.tonotopic_map = {}
        
        # Create tonotopic frequency mapping
        self._create_tonotopic_organization()
        
        print(f"FrequencyAnalysisLayer initialized with {config.n_frequency_bands} frequency bands")
        
    def _create_tonotopic_organization(self):
        """Create tonotopic frequency organization (logarithmic mapping)."""
        freq_min, freq_max = getattr(self.config, 'frequency_range', (20.0, 20000.0))
        n_bands = getattr(self.config, 'n_frequency_bands', 64)
        
        # Logarithmic frequency spacing (like cochlea)
        frequencies = np.logspace(np.log10(freq_min), np.log10(freq_max), n_bands)
        
        # Assign frequency preferences to neurons
        neurons_per_band = max(1, self.config.size // n_bands)
        
        neuron_id = 0
        for band_idx, center_freq in enumerate(frequencies):
            for _ in range(neurons_per_band):
                if neuron_id < self.config.size:
                    self.tonotopic_map[neuron_id] = {
                        'center_frequency': center_freq,
                        'band_index': band_idx,
                        'bandwidth': center_freq * 0.2  # 20% bandwidth
                    }
                    neuron_id += 1
                    
    def process_input(self, input_data: np.ndarray) -> np.ndarray:
        """Process input spectrogram for frequency analysis."""
        responses = np.zeros(self.config.size)
        
        # Assume input is spectrogram: (n_frequency_bins, n_time_frames)
        if len(input_data.shape) == 1:
            # Convert 1D to spectrogram-like
            n_freq_bins = int(np.sqrt(len(input_data)))
            input_data = input_data[:n_freq_bins*n_freq_bins].reshape(n_freq_bins, -1)
        elif len(input_data.shape) != 2:
            input_data = input_data.reshape(-1, 1)
            
        n_freq_bins, n_time_frames = input_data.shape
        
        # Process each neuron's frequency preference
        for neuron_id, freq_info in self.tonotopic_map.items():
            band_idx = freq_info['band_index']
            
            # Map to input frequency bin
            input_freq_bin = min(band_idx, n_freq_bins - 1)
            
            # Extract temporal response for this frequency band
            temporal_response = input_data[input_freq_bin, :]
            
            # Compute response (temporal integration)
            response = np.mean(temporal_response) * self.attention_modulation
            
            # Apply threshold
            threshold = getattr(self.config, 'response_threshold', 0.1)
            responses[neuron_id] = response if response > threshold else 0.0
            
        return responses


class TemporalPatternLayer(SensoryLayer):
    """
    Secondary auditory layer for temporal pattern detection.
    Detects temporal features like onsets, offsets, modulations.
    """
    
    def __init__(self, config: AudioLayerConfig, hierarchy_id: str = ""):
        """Initialize temporal pattern layer."""
        super().__init__(config, hierarchy_id)
        
        self.pattern_templates = {}
        self.temporal_responses = {}
        
        # Create temporal pattern templates
        self._create_temporal_templates()
        
        print(f"TemporalPatternLayer initialized with {len(self.pattern_templates)} pattern templates")
        
    def _create_temporal_templates(self):
        """Create templates for temporal pattern detection."""
        window_length = 10  # Temporal window in samples
        
        self.pattern_templates = {
            'onset': np.array([0.1, 0.2, 0.8, 1.0, 0.6, 0.3, 0.1, 0.05, 0.02, 0.01]),
            'offset': np.array([1.0, 0.8, 0.5, 0.3, 0.1, 0.05, 0.02, 0.01, 0.0, 0.0]),
            'sustained': np.array([0.8, 0.9, 1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.8]),
            'modulated': np.array([0.5, 0.8, 0.3, 0.9, 0.4, 0.8, 0.3, 0.9, 0.4, 0.7])
        }
        
    def process_input(self, input_data: np.ndarray) -> np.ndarray:
        """Process input for temporal pattern detection."""
        responses = np.zeros(self.config.size)
        
        # Assume input represents frequency analysis outputs over time
        if len(input_data.shape) == 1:
            temporal_data = input_data.reshape(-1, 1)
        else:
            # Take mean across frequency bands to get temporal envelope
            temporal_data = np.mean(input_data.reshape(-1, max(1, len(input_data)//16)), axis=1)
            
        # Pad to ensure minimum length
        if len(temporal_data) < 10:
            temporal_data = np.pad(temporal_data, (0, 10-len(temporal_data)), 'constant')
            
        response_idx = 0
        
        # For each pattern template
        for pattern_name, template in self.pattern_templates.items():
            if response_idx >= self.config.size:
                break
                
            # Slide template over temporal data
            template_len = len(template)
            max_correlation = 0.0
            
            for start_idx in range(max(1, len(temporal_data) - template_len + 1)):
                data_segment = temporal_data[start_idx:start_idx + template_len]
                
                # Normalize both
                if np.std(data_segment) > 0 and np.std(template) > 0:
                    norm_data = (data_segment - np.mean(data_segment)) / np.std(data_segment)
                    norm_template = (template - np.mean(template)) / np.std(template)
                    
                    # Compute correlation
                    correlation = np.corrcoef(norm_data, norm_template)[0, 1]
                    if not np.isnan(correlation):
                        max_correlation = max(max_correlation, correlation)
                        
            # Store pattern response
            pattern_response = max(0.0, max_correlation) * self.attention_modulation
            
            # Assign to multiple neurons for this pattern
            neurons_per_pattern = max(1, self.config.size // len(self.pattern_templates))
            for i in range(min(neurons_per_pattern, self.config.size - response_idx)):
                responses[response_idx] = pattern_response
                response_idx += 1
                
        return responses


class SoundRecognitionLayer(SensoryLayer):
    """
    Highest-level auditory layer for sound recognition and classification.
    Integrates frequency and temporal information for sound object recognition.
    """
    
    def __init__(self, config: AudioLayerConfig, hierarchy_id: str = ""):
        """Initialize sound recognition layer."""
        super().__init__(config, hierarchy_id)
        
        self.sound_categories = {}
        self.recognition_responses = {}
        
        # Create sound category prototypes
        self._create_sound_categories()
        
        print(f"SoundRecognitionLayer initialized with {len(self.sound_categories)} sound categories")
        
    def _create_sound_categories(self):
        """Create prototypes for different sound categories."""
        self.sound_categories = {
            'speech': {
                'frequency_bands': [200, 1000, 3000, 8000],  # Formant-like structure
                'temporal_patterns': ['onset', 'modulated'],
                'weights': [0.4, 0.6]
            },
            'music': {
                'frequency_bands': [100, 500, 2000, 5000],   # Harmonic structure
                'temporal_patterns': ['sustained', 'modulated'],
                'weights': [0.5, 0.5]
            },
            'noise': {
                'frequency_bands': [50, 200, 1000, 10000],   # Broadband
                'temporal_patterns': ['sustained'],
                'weights': [1.0]
            },
            'tone': {
                'frequency_bands': [440, 880, 1760],          # Pure tones
                'temporal_patterns': ['onset', 'sustained'],
                'weights': [0.3, 0.7]
            }
        }
        
    def process_input(self, input_data: np.ndarray) -> np.ndarray:
        """Process input for sound recognition."""
        responses = np.zeros(self.config.size)
        
        # Assume input combines frequency and temporal pattern information
        # First half: frequency responses, second half: temporal pattern responses
        input_len = len(input_data)
        freq_responses = input_data[:input_len//2] if input_len > 1 else input_data
        temp_responses = input_data[input_len//2:] if input_len > 1 else input_data
        
        response_idx = 0
        
        # For each sound category
        for sound_name, category in self.sound_categories.items():
            if response_idx >= self.config.size:
                break
                
            # Calculate recognition score
            total_score = 0.0
            total_weight = 0.0
            
            # Frequency component
            if len(freq_responses) > 0:
                freq_bands = category['frequency_bands']
                freq_score = 0.0
                
                # Simple mapping: take responses corresponding to key frequency bands
                for band_idx in range(min(len(freq_bands), len(freq_responses))):
                    freq_score += freq_responses[band_idx]
                    
                if len(freq_bands) > 0:
                    freq_score /= len(freq_bands)
                    
                total_score += freq_score * 0.6  # Weight frequency component
                total_weight += 0.6
                
            # Temporal pattern component  
            if len(temp_responses) > 0:
                patterns = category['temporal_patterns']
                pattern_weights = category['weights']
                
                pattern_score = 0.0
                for pattern_idx in range(min(len(patterns), len(pattern_weights), len(temp_responses))):
                    pattern_response = temp_responses[pattern_idx] if pattern_idx < len(temp_responses) else 0.0
                    weight = pattern_weights[pattern_idx] if pattern_idx < len(pattern_weights) else 1.0
                    pattern_score += pattern_response * weight
                    
                if len(pattern_weights) > 0:
                    pattern_score /= sum(pattern_weights[:len(patterns)])
                    
                total_score += pattern_score * 0.4  # Weight temporal component
                total_weight += 0.4
                
            # Normalize score
            if total_weight > 0:
                recognition_score = (total_score / total_weight) * self.attention_modulation
            else:
                recognition_score = 0.0
                
            # Apply threshold
            threshold = getattr(self.config, 'response_threshold', 0.2)
            final_score = recognition_score if recognition_score > threshold else 0.0
            
            responses[response_idx] = final_score
            response_idx += 1
            
        return responses


def create_auditory_processing_hierarchy() -> HierarchyConfig:
    """Create a complete auditory processing hierarchy configuration."""
    
    # Layer configurations for auditory hierarchy
    layers = [
        AudioLayerConfig(
            name="FrequencyAnalysis",
            level=ProcessingLevel.PRIMARY,
            size=512,
            feature_type=AudioFeatureType.FREQUENCY,
            n_frequency_bands=64,
            frequency_range=(20.0, 20000.0),
            response_threshold=0.1,
            spatial_layout=(16, 32)
        ),
        AudioLayerConfig(
            name="TemporalPattern",
            level=ProcessingLevel.SECONDARY,
            size=128,
            feature_type=AudioFeatureType.TEMPORAL_PATTERN,
            temporal_window=0.025,
            response_threshold=0.15,
            spatial_layout=(8, 16)
        ),
        AudioLayerConfig(
            name="SoundRecognition",
            level=ProcessingLevel.INTEGRATION,
            size=32,
            feature_type=AudioFeatureType.SOUND_OBJECT,
            response_threshold=0.3,
            spatial_layout=(4, 8)
        )
    ]
    
    # Connection configurations
    connections = [
        ConnectionConfig("FrequencyAnalysis", "TemporalPattern", "feedforward", 0.4, 1.0, "stdp"),
        ConnectionConfig("TemporalPattern", "SoundRecognition", "feedforward", 0.3, 1.0, "stdp")
    ]
    
    return HierarchyConfig(
        modality=SensoryModalityType.AUDITORY,
        layers=layers,
        connections=connections,
        input_dimensions=(64, 32),  # Spectrogram dimensions
        enable_feedback=True,
        enable_lateral_inhibition=True
    )


class AuditoryProcessingHierarchy(SensoryHierarchy):
    """Complete auditory processing hierarchy with specialized auditory layers."""
    
    def __init__(self, config: HierarchyConfig, network: Optional[Any] = None):
        """Initialize auditory processing hierarchy."""
        if HIERARCHICAL_IMPORTS_AVAILABLE:
            super().__init__(config, network)
            self._create_specialized_auditory_layers()
        else:
            # Minimal initialization for standalone mode
            self.config = config
            self.layers = {}
            self._create_specialized_auditory_layers()
            
    def _create_specialized_auditory_layers(self):
        """Replace generic layers with specialized auditory processing layers."""
        specialized_layers = {}
        
        for layer_config in self.config.layers:
            layer_name = layer_config.name
            
            if hasattr(layer_config, 'feature_type'):
                if layer_config.feature_type == AudioFeatureType.FREQUENCY:
                    specialized_layers[layer_name] = FrequencyAnalysisLayer(layer_config, getattr(self, 'hierarchy_id', ''))
                elif layer_config.feature_type == AudioFeatureType.TEMPORAL_PATTERN:
                    specialized_layers[layer_name] = TemporalPatternLayer(layer_config, getattr(self, 'hierarchy_id', ''))
                elif layer_config.feature_type == AudioFeatureType.SOUND_OBJECT:
                    specialized_layers[layer_name] = SoundRecognitionLayer(layer_config, getattr(self, 'hierarchy_id', ''))
                    
        self.layers = specialized_layers
        
        print("Auditory hierarchy updated with specialized layers:")
        for name, layer in self.layers.items():
            print(f"  {name}: {type(layer).__name__}")


def demo_auditory_processing_hierarchy():
    """Demonstrate the auditory processing hierarchy."""
    
    print("=== Auditory Processing Hierarchy Demo ===")
    
    # Create auditory hierarchy
    print("\n1. Creating Auditory Processing Hierarchy")
    config = create_auditory_processing_hierarchy()
    hierarchy = AuditoryProcessingHierarchy(config)
    
    # Display hierarchy info
    if hasattr(hierarchy, 'get_hierarchy_info'):
        info = hierarchy.get_hierarchy_info()
        print(f"Hierarchy Summary:")
        print(f"  Total layers: {len(hierarchy.layers)}")
    else:
        print(f"Hierarchy Summary:")
        print(f"  Total layers: {len(hierarchy.layers)}")
    
    # Test auditory processing
    print("\n2. Testing Auditory Input Processing")
    
    # Create test spectrogram (frequency x time)
    n_freq_bins, n_time_frames = 64, 32
    auditory_input = np.zeros((n_freq_bins, n_time_frames))
    
    # Add some frequency components (simulate speech formants)
    auditory_input[10:15, :] = 0.8  # Low frequency component (200 Hz region)
    auditory_input[25:30, 5:25] = 1.0  # Mid frequency burst (1000 Hz region)
    auditory_input[45:50, 10:20] = 0.6  # High frequency component (3000 Hz region)
    
    print(f"Created test spectrogram: {n_freq_bins} x {n_time_frames}")
    
    # Process through hierarchy layers individually
    activations = {}
    current_input = auditory_input
    
    for layer_name in ['FrequencyAnalysis', 'TemporalPattern', 'SoundRecognition']:
        if layer_name in hierarchy.layers:
            layer_output = hierarchy.layers[layer_name].process_input(current_input.flatten())
            activations[layer_name] = layer_output
            current_input = layer_output.reshape(-1, 1)  # Reshape for next layer
            
            mean_activity = np.mean(layer_output)
            max_activity = np.max(layer_output)
            active_neurons = np.sum(layer_output > 0)
            print(f"  {layer_name}: mean={mean_activity:.3f}, max={max_activity:.3f}, active={active_neurons}")
    
    print("\n✅ Auditory Processing Hierarchy Demo Complete!")
    
    return hierarchy


if __name__ == "__main__":
    # Run demonstration
    hierarchy = demo_auditory_processing_hierarchy()
    
    print("\n=== Task 6B.3 Implementation Summary ===")
    print("✅ Auditory Processing Hierarchy Implementation - COMPLETED")
    print("\nKey Features Implemented:")
    print("  • FrequencyAnalysisLayer with tonotopic organization")
    print("  • TemporalPatternLayer with onset/offset detection")
    print("  • SoundRecognitionLayer with speech/music classification")
    print("  • Spectral analysis and temporal pattern detection")
    print("  • Integration with hierarchical sensory processing framework")
    
    print("\nNext Steps:")
    print("  → Task 6B.4: Multi-Modal Integration System")
    print("  → Task 6B.5: Adaptive Feature Learning")