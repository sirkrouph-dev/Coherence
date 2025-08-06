"""
Enhanced sensory encoding system for neuromorphic computing.
Provides detailed input representations, real-time processing, and multi-modal fusion.
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from scipy import signal

from core.enhanced_logging import enhanced_logger


class ModalityType(Enum):
    """Sensory modality types."""

    VISUAL = "visual"
    AUDITORY = "auditory"
    TACTILE = "tactile"
    OLFACTORY = "olfactory"
    GUSTATORY = "gustatory"


@dataclass
class SensoryFeature:
    """Data structure for sensory features."""

    modality: ModalityType
    feature_type: str
    feature_vector: np.ndarray
    confidence: float
    timestamp: float
    metadata: Dict[str, Any]


@dataclass
class EncodedSpikes:
    """Data structure for encoded spike patterns."""

    modality: ModalityType
    spike_times: List[float]
    spike_neurons: List[int]
    encoding_quality: float
    temporal_resolution: float
    metadata: Dict[str, Any]


class VisualEncoder:
    """Enhanced visual encoding with feature extraction."""

    def __init__(
        self, resolution: Tuple[int, int] = (32, 32), feature_types: List[str] = None
    ):
        """
        Initialize visual encoder.

        Args:
            resolution: Image resolution (width, height)
            feature_types: Types of features to extract
        """
        self.resolution = resolution
        self.feature_types = feature_types or ["edges", "corners", "texture", "motion"]
        self.feature_extractors = self._initialize_feature_extractors()

    def _initialize_feature_extractors(self) -> Dict[str, Any]:
        """Initialize feature extraction methods."""
        extractors = {}

        # Edge detection
        extractors["edges"] = {
            "method": "canny",
            "params": {"low_threshold": 50, "high_threshold": 150},
        }

        # Corner detection
        extractors["corners"] = {
            "method": "harris",
            "params": {"block_size": 2, "ksize": 3, "k": 0.04},
        }

        # Texture analysis
        extractors["texture"] = {
            "method": "gabor",
            "params": {
                "frequencies": [0.1, 0.3, 0.5],
                "orientations": [0, 45, 90, 135],
            },
        }

        # Motion detection
        extractors["motion"] = {
            "method": "optical_flow",
            "params": {"win_size": (15, 15)},
        }

        return extractors

    def encode_image(self, image: np.ndarray) -> Dict[str, Any]:
        """Encode image to spike patterns."""
        start_time = time.time()

        # Preprocess image
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Resize to target resolution
        image = cv2.resize(image, self.resolution)

        # Extract features
        features = {}
        for feature_type in self.feature_types:
            features[feature_type] = self._extract_feature(image, feature_type)

        # Convert features to spike patterns
        spike_patterns = {}
        for feature_type, feature_data in features.items():
            spike_patterns[feature_type] = self._features_to_spikes(feature_data)

        encoding_time = time.time() - start_time

        # Log encoding details
        enhanced_logger.log_sensory_encoding(
            "visual",
            image,
            np.sum([len(spikes) for spikes in spike_patterns.values()]),
            encoding_time,
        )

        return {
            "features": features,
            "spike_patterns": spike_patterns,
            "encoding_time": encoding_time,
            "image_shape": image.shape,
            "feature_types": self.feature_types,
        }

    def _extract_feature(self, image: np.ndarray, feature_type: str) -> np.ndarray:
        """Extract specific feature from image."""
        if feature_type == "edges":
            # Ensure image is uint8 for Canny
            if image.dtype != np.uint8:
                image_uint8 = (
                    (image * 255).astype(np.uint8)
                    if image.max() <= 1
                    else image.astype(np.uint8)
                )
            else:
                image_uint8 = image
            return cv2.Canny(
                image_uint8,
                self.feature_extractors["edges"]["params"]["low_threshold"],
                self.feature_extractors["edges"]["params"]["high_threshold"],
            )

        elif feature_type == "corners":
            # Ensure image is proper type for cornerHarris
            if image.dtype != np.uint8 and image.dtype != np.float32:
                if image.max() <= 1:
                    image_processed = image.astype(np.float32)
                else:
                    image_processed = image.astype(np.uint8)
            else:
                image_processed = image
            corners = cv2.cornerHarris(
                image_processed,
                self.feature_extractors["corners"]["params"]["block_size"],
                self.feature_extractors["corners"]["params"]["ksize"],
                self.feature_extractors["corners"]["params"]["k"],
            )
            return np.clip(corners, 0, 255)

        elif feature_type == "texture":
            # Gabor filter bank for texture
            texture_response = np.zeros_like(image, dtype=float)
            for freq in self.feature_extractors["texture"]["params"]["frequencies"]:
                for angle in self.feature_extractors["texture"]["params"][
                    "orientations"
                ]:
                    kernel = cv2.getGaborKernel(
                        (21, 21),
                        8,
                        np.radians(angle),
                        2 * np.pi * freq,
                        0.5,
                        0,
                        ktype=cv2.CV_32F,
                    )
                    texture_response += cv2.filter2D(image, cv2.CV_8UC3, kernel)
            return texture_response

        elif feature_type == "motion":
            # Simplified motion detection (would need temporal data in real
            # implementation)
            return np.random.random(image.shape) * 255  # Placeholder

        else:
            return image

    def _features_to_spikes(self, feature_data: np.ndarray) -> List[Tuple[int, float]]:
        """Convert feature data to spike times."""
        spikes = []
        feature_flat = feature_data.flatten()

        # Rate encoding based on feature intensity
        for i, intensity in enumerate(feature_flat):
            if intensity > 50:  # Threshold for spiking
                # Convert intensity to spike rate
                spike_rate = intensity / 255.0
                spike_time = 1.0 / (spike_rate + 1e-6)  # Avoid division by zero
                spikes.append((i, spike_time))

        return spikes


class AuditoryEncoder:
    """Enhanced auditory encoding with frequency analysis."""

    def __init__(
        self,
        sample_rate: int = 44100,
        frequency_bands: List[Tuple[float, float]] = None,
    ):
        """
        Initialize auditory encoder.

        Args:
            sample_rate: Audio sample rate
            frequency_bands: Frequency bands for analysis
        """
        self.sample_rate = sample_rate
        self.frequency_bands = frequency_bands or [
            (20, 200),  # Low frequency
            (200, 2000),  # Mid frequency
            (2000, 20000),  # High frequency
        ]

    def encode_audio(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Encode audio to spike patterns."""
        start_time = time.time()

        # Ensure audio is mono
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        # Extract frequency features
        features = {}
        for i, (low_freq, high_freq) in enumerate(self.frequency_bands):
            band_name = f"band_{i}_{low_freq}_{high_freq}"
            features[band_name] = self._extract_frequency_band(
                audio_data, low_freq, high_freq
            )

        # Extract temporal features
        features["temporal"] = self._extract_temporal_features(audio_data)

        # Convert to spike patterns
        spike_patterns = {}
        for feature_name, feature_data in features.items():
            spike_patterns[feature_name] = self._audio_features_to_spikes(feature_data)

        encoding_time = time.time() - start_time

        # Log encoding details
        enhanced_logger.log_sensory_encoding(
            "auditory",
            audio_data,
            np.sum([len(spikes) for spikes in spike_patterns.values()]),
            encoding_time,
        )

        return {
            "features": features,
            "spike_patterns": spike_patterns,
            "encoding_time": encoding_time,
            "audio_length": len(audio_data),
            "frequency_bands": self.frequency_bands,
        }

    def _extract_frequency_band(
        self, audio_data: np.ndarray, low_freq: float, high_freq: float
    ) -> np.ndarray:
        """Extract frequency band from audio."""
        # Design bandpass filter
        nyquist = self.sample_rate / 2
        low_norm = low_freq / nyquist
        high_norm = high_freq / nyquist

        # Butterworth bandpass filter
        b, a = signal.butter(4, [low_norm, high_norm], btype="band")
        filtered_audio = signal.filtfilt(b, a, audio_data)

        # Compute power spectrum
        freqs, psd = signal.welch(filtered_audio, self.sample_rate, nperseg=1024)

        return psd

    def _extract_temporal_features(self, audio_data: np.ndarray) -> np.ndarray:
        """Extract temporal features from audio."""
        # Envelope detection
        envelope = np.abs(signal.hilbert(audio_data))

        # Onset detection
        onset_strength = np.diff(envelope)
        onset_strength = np.concatenate([[0], onset_strength])

        return onset_strength

    def _audio_features_to_spikes(
        self, feature_data: np.ndarray
    ) -> List[Tuple[int, float]]:
        """Convert audio features to spike times."""
        spikes = []

        # Rate encoding based on feature intensity
        for i, intensity in enumerate(feature_data):
            if intensity > np.mean(feature_data) + np.std(feature_data):
                # Convert intensity to spike rate
                spike_rate = intensity / np.max(feature_data)
                spike_time = 1.0 / (spike_rate + 1e-6)
                spikes.append((i, spike_time))

        return spikes


class TactileEncoder:
    """Enhanced tactile encoding with pressure and vibration analysis."""

    def __init__(
        self,
        sensor_array_size: Tuple[int, int] = (8, 8),
        pressure_range: Tuple[float, float] = (0, 100),
    ):
        """
        Initialize tactile encoder.

        Args:
            sensor_array_size: Size of tactile sensor array
            pressure_range: Range of pressure values
        """
        self.sensor_array_size = sensor_array_size
        self.pressure_range = pressure_range
        self.total_sensors = sensor_array_size[0] * sensor_array_size[1]

    def encode_tactile(self, tactile_data: np.ndarray) -> Dict[str, Any]:
        """Encode tactile data to spike patterns."""
        start_time = time.time()

        # Ensure correct shape
        if tactile_data.shape != self.sensor_array_size:
            tactile_data = cv2.resize(tactile_data, self.sensor_array_size)

        # Extract tactile features
        features = {
            "pressure": self._extract_pressure_features(tactile_data),
            "vibration": self._extract_vibration_features(tactile_data),
            "texture": self._extract_texture_features(tactile_data),
            "contact_area": self._extract_contact_features(tactile_data),
        }

        # Convert to spike patterns
        spike_patterns = {}
        for feature_name, feature_data in features.items():
            spike_patterns[feature_name] = self._tactile_features_to_spikes(
                feature_data
            )

        encoding_time = time.time() - start_time

        # Log encoding details
        enhanced_logger.log_sensory_encoding(
            "tactile",
            tactile_data,
            np.sum([len(spikes) for spikes in spike_patterns.values()]),
            encoding_time,
        )

        return {
            "features": features,
            "spike_patterns": spike_patterns,
            "encoding_time": encoding_time,
            "sensor_array_size": self.sensor_array_size,
            "pressure_range": self.pressure_range,
        }

    def _extract_pressure_features(self, tactile_data: np.ndarray) -> np.ndarray:
        """Extract pressure distribution features."""
        # Pressure gradient
        grad_x = np.gradient(tactile_data, axis=1)
        grad_y = np.gradient(tactile_data, axis=0)
        pressure_gradient = np.sqrt(grad_x**2 + grad_y**2)

        return pressure_gradient

    def _extract_vibration_features(self, tactile_data: np.ndarray) -> np.ndarray:
        """Extract vibration patterns."""
        # High-frequency components (simulated)
        vibration = np.random.normal(0, 0.1, tactile_data.shape)
        vibration *= tactile_data  # Only vibrate where there's contact

        return vibration

    def _extract_texture_features(self, tactile_data: np.ndarray) -> np.ndarray:
        """Extract texture features."""
        # Local texture analysis using Gabor filters
        texture_response = np.zeros_like(tactile_data, dtype=float)

        for angle in [0, 45, 90, 135]:
            kernel = cv2.getGaborKernel(
                (5, 5), 2, np.radians(angle), 2 * np.pi * 0.3, 0.5, 0, ktype=cv2.CV_32F
            )
            # Convert tactile_data to float32 for filtering
            tactile_float = tactile_data.astype(np.float32)
            texture_response += cv2.filter2D(tactile_float, -1, kernel)

        return texture_response

    def _extract_contact_features(self, tactile_data: np.ndarray) -> np.ndarray:
        """Extract contact area features."""
        # Contact area detection
        contact_threshold = np.mean(tactile_data) + np.std(tactile_data)
        contact_mask = tactile_data > contact_threshold

        # Contact area features
        contact_area = np.sum(contact_mask)
        contact_centroid = np.mean(np.where(contact_mask), axis=1)

        return np.array([contact_area, contact_centroid[0], contact_centroid[1]])

    def _tactile_features_to_spikes(
        self, feature_data: np.ndarray
    ) -> List[Tuple[int, float]]:
        """Convert tactile features to spike times."""
        spikes = []
        feature_flat = feature_data.flatten()

        # Rate encoding based on tactile intensity
        for i, intensity in enumerate(feature_flat):
            if intensity > np.mean(feature_flat):
                # Convert intensity to spike rate
                spike_rate = intensity / np.max(feature_flat)
                spike_time = 1.0 / (spike_rate + 1e-6)
                spikes.append((i, spike_time))

        return spikes


class MultiModalFusion:
    """Multi-modal sensory fusion system."""

    def __init__(self):
        """Initialize multi-modal fusion system."""
        self.fusion_weights = {"visual": 0.4, "auditory": 0.3, "tactile": 0.3}
        self.temporal_window = 100  # ms
        self.fusion_history = []

    def fuse_modalities(
        self, encoded_inputs: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Fuse multiple sensory modalities."""
        start_time = time.time()

        # Extract spike patterns from each modality
        fused_spikes = []
        modality_weights = []

        for modality, encoding_data in encoded_inputs.items():
            if modality in self.fusion_weights:
                weight = self.fusion_weights[modality]
                spike_patterns = encoding_data["spike_patterns"]

                # Combine all spike patterns for this modality
                modality_spikes = []
                for feature_name, spikes in spike_patterns.items():
                    for neuron_id, spike_time in spikes:
                        modality_spikes.append((neuron_id, spike_time, weight))

                fused_spikes.extend(modality_spikes)
                modality_weights.extend([weight] * len(modality_spikes))

        # Sort spikes by time
        fused_spikes.sort(key=lambda x: x[1])

        # Apply temporal fusion
        temporal_fused = self._apply_temporal_fusion(fused_spikes)

        # Calculate fusion quality
        fusion_quality = self._calculate_fusion_quality(encoded_inputs)

        fusion_time = time.time() - start_time

        return {
            "fused_spikes": temporal_fused,
            "fusion_quality": fusion_quality,
            "modality_weights": self.fusion_weights,
            "fusion_time": fusion_time,
            "temporal_window": self.temporal_window,
        }

    def _apply_temporal_fusion(
        self, spikes: List[Tuple[int, float, float]]
    ) -> List[Tuple[int, float]]:
        """Apply temporal fusion to spike patterns."""
        fused = []
        current_window = []

        for neuron_id, spike_time, weight in spikes:
            # Add to current temporal window
            current_window.append((neuron_id, spike_time, weight))

            # Process window if it exceeds temporal window
            if spike_time > self.temporal_window:
                # Fuse spikes in current window
                window_fused = self._fuse_window(current_window)
                fused.extend(window_fused)

                # Remove processed spikes from window
                current_window = [
                    (n, t, w)
                    for n, t, w in current_window
                    if t > spike_time - self.temporal_window
                ]

        # Process remaining spikes
        if current_window:
            window_fused = self._fuse_window(current_window)
            fused.extend(window_fused)

        return fused

    def _fuse_window(
        self, window_spikes: List[Tuple[int, float, float]]
    ) -> List[Tuple[int, float]]:
        """Fuse spikes within a temporal window."""
        fused = []

        # Group spikes by neuron
        neuron_spikes = {}
        for neuron_id, spike_time, weight in window_spikes:
            if neuron_id not in neuron_spikes:
                neuron_spikes[neuron_id] = []
            neuron_spikes[neuron_id].append((spike_time, weight))

        # Fuse each neuron's spikes
        for neuron_id, spikes in neuron_spikes.items():
            if len(spikes) > 1:
                # Weighted average of spike times
                total_weight = sum(weight for _, weight in spikes)
                weighted_time = (
                    sum(time * weight for time, weight in spikes) / total_weight
                )
                fused.append((neuron_id, weighted_time))
            else:
                # Single spike
                fused.append((neuron_id, spikes[0][0]))

        return fused

    def _calculate_fusion_quality(
        self, encoded_inputs: Dict[str, Dict[str, Any]]
    ) -> float:
        """Calculate quality of multi-modal fusion."""
        # Simple quality metric based on modality availability and consistency
        available_modalities = len(encoded_inputs)
        total_modalities = len(self.fusion_weights)

        # Quality based on modality coverage
        coverage_quality = available_modalities / total_modalities

        # Quality based on temporal consistency (simplified)
        temporal_quality = 0.8  # Placeholder

        return (coverage_quality + temporal_quality) / 2


class EnhancedSensoryEncoder:
    """Main sensory encoding system."""

    def __init__(self):
        """Initialize enhanced sensory encoder."""
        self.visual_encoder = VisualEncoder()
        self.auditory_encoder = AuditoryEncoder()
        self.tactile_encoder = TactileEncoder()
        self.fusion_system = MultiModalFusion()

    def encode_sensory_inputs(self, inputs: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Encode all sensory inputs."""
        start_time = time.time()

        encoded_inputs = {}

        # Encode each modality
        if "visual" in inputs:
            encoded_inputs["visual"] = self.visual_encoder.encode_image(
                inputs["visual"]
            )

        if "auditory" in inputs:
            encoded_inputs["auditory"] = self.auditory_encoder.encode_audio(
                inputs["auditory"]
            )

        if "tactile" in inputs:
            encoded_inputs["tactile"] = self.tactile_encoder.encode_tactile(
                inputs["tactile"]
            )

        # Fuse modalities
        fused_result = self.fusion_system.fuse_modalities(encoded_inputs)

        total_time = time.time() - start_time

        return {
            "encoded_inputs": encoded_inputs,
            "fused_result": fused_result,
            "total_encoding_time": total_time,
            "modalities_encoded": list(encoded_inputs.keys()),
        }

    def get_encoding_statistics(self) -> Dict[str, Any]:
        """Get statistics about encoding performance."""
        return {
            "visual_features": len(self.visual_encoder.feature_types),
            "auditory_bands": len(self.auditory_encoder.frequency_bands),
            "tactile_sensors": self.tactile_encoder.total_sensors,
            "fusion_weights": self.fusion_system.fusion_weights,
            "temporal_window": self.fusion_system.temporal_window,
        }


# Global encoder instance
enhanced_encoder = EnhancedSensoryEncoder()
