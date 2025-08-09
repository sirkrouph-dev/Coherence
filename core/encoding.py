"""
Basic sensory encoding system for neuromorphic computing.
Provides fundamental input encoding methods for various sensory modalities.
"""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


class RateEncoder:
    """Basic rate encoding for neural inputs."""

    def __init__(self, max_rate: float = 100.0):
        """
        Initialize rate encoder.

        Args:
            max_rate: Maximum firing rate in Hz
        """
        self.max_rate = max_rate

    def encode(self, value: float, duration: float = 100.0, dt: float = 1.0) -> List[Tuple[int, float]]:
        """
        Convert input value to spike times using rate encoding.

        Args:
            value: Input value to encode (0-1 range)
            duration: Duration of encoding window in ms
            dt: Time step in ms

        Returns:
            List of (neuron_id, spike_time) tuples
        """
        # Clamp value to [0, 1]
        value = np.clip(value, 0.0, 1.0)
        
        # Calculate spike rate
        spike_rate = value * self.max_rate  # Hz
        
        # Generate spike times
        spikes = []
        if spike_rate > 0:
            # Inter-spike interval in ms
            isi = 1000.0 / spike_rate
            
            # Generate spikes within duration
            t = 0.0
            neuron_id = 0
            while t < duration:
                # Add some jitter for biological realism
                jitter = np.random.uniform(-0.1, 0.1) * isi
                spike_time = t + jitter
                
                if 0 <= spike_time < duration:
                    spikes.append((neuron_id, spike_time))
                
                t += isi
        
        return spikes
    
    def encode_array(self, input_values: np.ndarray, duration: float = 100.0, dt: float = 1.0) -> List[Tuple[int, float]]:
        """
        Convert array of input values to spike times.

        Args:
            input_values: Array of input values to encode
            duration: Duration of encoding window in ms  
            dt: Time step in ms

        Returns:
            List of (neuron_id, spike_time) tuples for all neurons
        """
        all_spikes = []
        
        for neuron_id, value in enumerate(input_values.flatten()):
            if value > 0:
                # Clamp value to [0, 1]
                value = np.clip(value, 0.0, 1.0)
                spike_rate = value * self.max_rate
                
                if spike_rate > 0:
                    isi = 1000.0 / spike_rate
                    t = np.random.uniform(0, isi)  # Random phase
                    
                    while t < duration:
                        if 0 <= t < duration:
                            all_spikes.append((neuron_id, t))
                        t += isi
        
        # Sort by spike time
        all_spikes.sort(key=lambda x: x[1])
        return all_spikes


class RetinalEncoder:
    """Retinal encoding for visual inputs."""

    def __init__(self, resolution: Tuple[int, int] = (32, 32)):
        """
        Initialize retinal encoder.

        Args:
            resolution: Image resolution
        """
        self.resolution = resolution

    def encode(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Encode image using retinal processing.

        Args:
            image: Input image

        Returns:
            Encoded representation
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Resize to target resolution
        image = cv2.resize(image, self.resolution)

        # ON-center/OFF-surround and OFF-center/ON-surround
        on_center = self._compute_center_surround(image, True)
        off_center = self._compute_center_surround(image, False)

        return {"on_center": on_center, "off_center": off_center, "original": image}

    def _compute_center_surround(
        self, image: np.ndarray, on_center: bool
    ) -> np.ndarray:
        """Compute center-surround receptive fields."""
        # Simple Gaussian difference for center-surround
        center = cv2.GaussianBlur(image, (3, 3), 1)
        surround = cv2.GaussianBlur(image, (9, 9), 3)

        if on_center:
            response = center - surround
        else:
            response = surround - center

        return np.clip(response, 0, 255)


class CochlearEncoder:
    """Cochlear encoding for auditory inputs."""

    def __init__(self, num_channels: int = 32, sample_rate: int = 44100):
        """
        Initialize cochlear encoder.

        Args:
            num_channels: Number of frequency channels
            sample_rate: Audio sample rate
        """
        self.num_channels = num_channels
        self.sample_rate = sample_rate

    def encode(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Encode audio using cochlear model.

        Args:
            audio: Audio signal

        Returns:
            Encoded representation
        """
        # Simple frequency decomposition (simplified cochlear model)
        frequencies = np.fft.fft(audio)
        freq_bins = np.array_split(frequencies, self.num_channels)

        channel_responses = []
        for bin_data in freq_bins:
            power = np.abs(bin_data).mean()
            channel_responses.append(power)

        return {
            "channel_responses": np.array(channel_responses),
            "sample_rate": self.sample_rate,
        }


class SomatosensoryEncoder:
    """Somatosensory encoding for tactile inputs."""

    def __init__(self, resolution: Tuple[int, int] = (8, 8)):
        """
        Initialize somatosensory encoder.

        Args:
            resolution: Tactile sensor resolution
        """
        self.resolution = resolution

    def encode(self, pressure_map: np.ndarray) -> Dict[str, Any]:
        """
        Encode tactile/pressure inputs.

        Args:
            pressure_map: Pressure sensor data

        Returns:
            Encoded representation
        """
        # Resize to target resolution
        if pressure_map.shape != self.resolution:
            pressure_map = cv2.resize(pressure_map, self.resolution)

        # Detect edges (rapid pressure changes)
        edges = cv2.Sobel(pressure_map, cv2.CV_64F, 1, 1)

        # Sustained pressure response
        sustained = pressure_map.copy()

        return {
            "sustained": sustained,
            "transient": edges,
            "resolution": self.resolution,
        }


class MultiModalEncoder:
    """Multi-modal encoder for combining different sensory inputs."""

    def __init__(self):
        """Initialize multi-modal encoder."""
        self.visual_encoder = RetinalEncoder()
        self.auditory_encoder = CochlearEncoder()
        self.tactile_encoder = SomatosensoryEncoder()
        self.rate_encoder = RateEncoder()

    def encode(self, sensory_inputs: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Encode multiple sensory modalities.

        Args:
            sensory_inputs: Dictionary of sensory inputs by modality

        Returns:
            Encoded representations for all modalities
        """
        encoded_outputs = {}

        for modality, data in sensory_inputs.items():
            if modality == "visual":
                encoded_outputs[modality] = self.visual_encoder.encode(data)
            elif modality == "auditory":
                encoded_outputs[modality] = self.auditory_encoder.encode(data)
            elif modality == "tactile":
                encoded_outputs[modality] = self.tactile_encoder.encode(data)
            else:
                # Default to rate encoding
                encoded_outputs[modality] = self.rate_encoder.encode(data)

        return encoded_outputs

    def fuse_modalities(self, encoded_outputs: Dict[str, Any]) -> np.ndarray:
        """
        Fuse multiple modalities into a single representation.

        Args:
            encoded_outputs: Encoded outputs from different modalities

        Returns:
            Fused multi-modal representation
        """
        fused = []

        for modality, data in encoded_outputs.items():
            if isinstance(data, dict):
                # Extract the main features from each modality
                if "on_center" in data:  # Visual
                    fused.append(data["on_center"].flatten())
                elif "channel_responses" in data:  # Auditory
                    fused.append(data["channel_responses"])
                elif "sustained" in data:  # Tactile
                    fused.append(data["sustained"].flatten())
            else:
                fused.append(data.flatten())

        if fused:
            return np.concatenate(fused)
        else:
            return np.array([])


class TemporalEncoder:
    """Temporal encoding for time-series data."""

    def __init__(self, time_window: float = 100.0):
        """
        Initialize temporal encoder.

        Args:
            time_window: Time window for encoding in ms
        """
        self.time_window = time_window

    def encode(
        self, time_series: np.ndarray, timestamps: Optional[np.ndarray] = None
    ) -> List[Tuple[int, float]]:
        """
        Encode time-series data to spike times.

        Args:
            time_series: Time-series data
            timestamps: Optional timestamps for data points

        Returns:
            List of (neuron_id, spike_time) tuples
        """
        spikes = []

        if timestamps is None:
            # Generate uniform timestamps
            timestamps = np.linspace(0, self.time_window, len(time_series))

        for i, (value, time) in enumerate(zip(time_series, timestamps)):
            if value > 0.5:  # Simple threshold
                spikes.append((i, time))

        return spikes


class PopulationEncoder:
    """Population encoding using multiple neurons per value."""

    def __init__(
        self, num_neurons: int = 10, value_range: Tuple[float, float] = (0, 1)
    ):
        """
        Initialize population encoder.

        Args:
            num_neurons: Number of neurons per encoded value
            value_range: Range of input values
        """
        self.num_neurons = num_neurons
        self.value_range = value_range

        # Create preferred values for each neuron
        self.preferred_values = np.linspace(value_range[0], value_range[1], num_neurons)
        self.tuning_width = (value_range[1] - value_range[0]) / (num_neurons * 2)

    def encode(self, value: float) -> np.ndarray:
        """
        Encode a value using population coding.

        Args:
            value: Value to encode

        Returns:
            Population response vector
        """
        # Gaussian tuning curves
        responses = np.exp(
            -0.5 * ((value - self.preferred_values) / self.tuning_width) ** 2
        )
        return responses / (responses.sum() + 1e-6)  # Normalize
