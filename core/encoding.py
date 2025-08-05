"""
Sensory encoding systems for the neuromorphic programming system.
Implements biologically plausible spike encoding for different sensory modalities.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from scipy import signal
from scipy.ndimage import gaussian_filter


class SensoryEncoder:
    """Base class for sensory encoders."""
    
    def __init__(self, output_size: int):
        """
        Initialize sensory encoder.
        
        Args:
            output_size: Number of output neurons
        """
        self.output_size = output_size
        self.current_time = 0.0
        
    def encode(self, input_data: Any, time_window: float = 100.0) -> List[Tuple[int, float]]:
        """
        Encode input data into spike trains.
        
        Args:
            input_data: Input data to encode
            time_window: Time window for encoding (ms)
            
        Returns:
            List of (neuron_id, spike_time) tuples
        """
        raise NotImplementedError
        
    def reset(self):
        """Reset encoder to initial state."""
        self.current_time = 0.0


class RetinalEncoder(SensoryEncoder):
    """
    Retinal encoder for visual input.
    
    Implements difference-of-Gaussians (DoG) receptive fields.
    """
    
    def __init__(self, resolution: Tuple[int, int] = (32, 32), 
                 output_size: Optional[int] = None):
        """
        Initialize retinal encoder.
        
        Args:
            resolution: Image resolution (width, height)
            output_size: Number of output neurons (default: 2 * width * height)
        """
        if output_size is None:
            output_size = 2 * resolution[0] * resolution[1]  # On + Off cells
        super().__init__(output_size)
        
        self.resolution = resolution
        self.width, self.height = resolution
        
        # Create receptive fields
        self.receptive_fields = self._create_receptive_fields()
        
    def _create_receptive_fields(self) -> List[Dict[str, Any]]:
        """Create DoG receptive fields for each output neuron."""
        fields = []
        neuron_id = 0
        
        for y in range(self.height):
            for x in range(self.width):
                # On-center cell
                fields.append({
                    'neuron_id': neuron_id,
                    'x': x, 'y': y,
                    'type': 'on',
                    'center_sigma': 1.0,
                    'surround_sigma': 2.0,
                    'center_weight': 1.0,
                    'surround_weight': -0.5
                })
                neuron_id += 1
                
                # Off-center cell
                fields.append({
                    'neuron_id': neuron_id,
                    'x': x, 'y': y,
                    'type': 'off',
                    'center_sigma': 1.0,
                    'surround_sigma': 2.0,
                    'center_weight': -1.0,
                    'surround_weight': 0.5
                })
                neuron_id += 1
                
        return fields
        
    def _compute_dog_response(self, image: np.ndarray, field: Dict[str, Any]) -> float:
        """Compute DoG response for a receptive field."""
        x, y = field['x'], field['y']
        
        # Create center and surround filters
        center_filter = self._create_gaussian_filter(
            field['center_sigma'], field['center_weight']
        )
        surround_filter = self._create_gaussian_filter(
            field['surround_sigma'], field['surround_weight']
        )
        
        # Apply filters
        center_response = self._apply_filter(image, center_filter, x, y)
        surround_response = self._apply_filter(image, surround_filter, x, y)
        
        return center_response + surround_response
        
    def _create_gaussian_filter(self, sigma: float, weight: float) -> np.ndarray:
        """Create Gaussian filter."""
        size = int(4 * sigma) + 1
        x, y = np.meshgrid(np.arange(size), np.arange(size))
        center = size // 2
        
        filter_kernel = weight * np.exp(-((x - center)**2 + (y - center)**2) / (2 * sigma**2))
        return filter_kernel
        
    def _apply_filter(self, image: np.ndarray, filter_kernel: np.ndarray, 
                     x: int, y: int) -> float:
        """Apply filter at specific position."""
        h, w = filter_kernel.shape
        h_half, w_half = h // 2, w // 2
        
        # Extract region around (x, y)
        y_start = max(0, y - h_half)
        y_end = min(image.shape[0], y + h_half + 1)
        x_start = max(0, x - w_half)
        x_end = min(image.shape[1], x + w_half + 1)
        
        image_region = image[y_start:y_end, x_start:x_end]
        filter_region = filter_kernel[
            max(0, h_half - y):min(h, h_half + (image.shape[0] - y)),
            max(0, w_half - x):min(w, w_half + (image.shape[1] - x))
        ]
        
        return np.sum(image_region * filter_region)
        
    def encode(self, image: np.ndarray, time_window: float = 100.0) -> List[Tuple[int, float]]:
        """
        Encode image into spike trains.
        
        Args:
            image: Input image (grayscale, values 0-1)
            time_window: Time window for encoding (ms)
            
        Returns:
            List of (neuron_id, spike_time) tuples
        """
        spikes = []
        
        # Normalize image
        if image.max() > 1.0:
            image = image / 255.0
            
        # Compute responses for all receptive fields
        responses = []
        for field in self.receptive_fields:
            response = self._compute_dog_response(image, field)
            responses.append((field['neuron_id'], response))
            
        # Convert responses to spike times
        max_response = max(abs(r) for _, r in responses) if responses else 1.0
        
        for neuron_id, response in responses:
            if abs(response) > 0.1 * max_response:  # Threshold
                # Convert response to spike time (stronger response = earlier spike)
                spike_time = time_window * (1.0 - abs(response) / max_response)
                spikes.append((neuron_id, spike_time))
                
        return spikes


class CochlearEncoder(SensoryEncoder):
    """
    Cochlear encoder for auditory input.
    
    Implements tonotopic mapping and cochlear filtering.
    """
    
    def __init__(self, frequency_bands: int = 64, sample_rate: int = 44100,
                 output_size: Optional[int] = None):
        """
        Initialize cochlear encoder.
        
        Args:
            frequency_bands: Number of frequency bands
            sample_rate: Audio sample rate (Hz)
            output_size: Number of output neurons (default: frequency_bands)
        """
        if output_size is None:
            output_size = frequency_bands
        super().__init__(output_size)
        
        self.frequency_bands = frequency_bands
        self.sample_rate = sample_rate
        
        # Create tonotopic map (logarithmic frequency spacing)
        self.frequencies = self._create_tonotopic_map()
        
        # Create cochlear filters
        self.filters = self._create_cochlear_filters()
        
    def _create_tonotopic_map(self) -> np.ndarray:
        """Create logarithmic frequency mapping."""
        # Human hearing range: 20 Hz to 20 kHz
        min_freq = 20.0
        max_freq = 20000.0
        
        # Logarithmic spacing
        log_min = np.log10(min_freq)
        log_max = np.log10(max_freq)
        log_frequencies = np.linspace(log_min, log_max, self.frequency_bands)
        
        return 10.0 ** log_frequencies
        
    def _create_cochlear_filters(self) -> List[Dict[str, Any]]:
        """Create cochlear filter bank."""
        filters = []
        
        for i, freq in enumerate(self.frequencies):
            # Gammatone filter parameters
            filter_params = {
                'neuron_id': i,
                'frequency': freq,
                'bandwidth': freq / 10.0,  # 1/10 octave bandwidth
                'order': 4  # Filter order
            }
            filters.append(filter_params)
            
        return filters
        
    def _apply_gammatone_filter(self, audio: np.ndarray, filter_params: Dict[str, Any]) -> float:
        """Apply gammatone filter to audio signal."""
        freq = filter_params['frequency']
        bandwidth = filter_params['bandwidth']
        order = filter_params['order']
        
        # Simplified gammatone filter implementation
        # In practice, you would use a proper gammatone filter library
        
        # For now, use a simple bandpass filter approximation
        nyquist = self.sample_rate / 2.0
        low_freq = freq - bandwidth / 2.0
        high_freq = freq + bandwidth / 2.0
        
        # Normalize frequencies
        low_freq_norm = low_freq / nyquist
        high_freq_norm = high_freq / nyquist
        
        # Create bandpass filter
        b, a = signal.butter(4, [low_freq_norm, high_freq_norm], btype='band')
        filtered_audio = signal.filtfilt(b, a, audio)
        
        # Compute response magnitude
        response = np.sqrt(np.mean(filtered_audio**2))
        
        return response
        
    def encode(self, audio: np.ndarray, time_window: float = 100.0) -> List[Tuple[int, float]]:
        """
        Encode audio into spike trains.
        
        Args:
            audio: Input audio signal
            time_window: Time window for encoding (ms)
            
        Returns:
            List of (neuron_id, spike_time) tuples
        """
        spikes = []
        
        # Normalize audio
        if audio.max() > 1.0:
            audio = audio / np.max(np.abs(audio))
            
        # Compute responses for all frequency bands
        responses = []
        for filter_params in self.filters:
            response = self._apply_gammatone_filter(audio, filter_params)
            responses.append((filter_params['neuron_id'], response))
            
        # Convert responses to spike times
        max_response = max(r for _, r in responses) if responses else 1.0
        
        for neuron_id, response in responses:
            if response > 0.1 * max_response:  # Threshold
                # Convert response to spike time (stronger response = earlier spike)
                spike_time = time_window * (1.0 - response / max_response)
                spikes.append((neuron_id, spike_time))
                
        return spikes


class SomatosensoryEncoder(SensoryEncoder):
    """
    Somatosensory encoder for tactile input.
    
    Implements mechanoreceptor-like encoding.
    """
    
    def __init__(self, sensor_grid: Tuple[int, int] = (16, 16),
                 output_size: Optional[int] = None):
        """
        Initialize somatosensory encoder.
        
        Args:
            sensor_grid: Grid size for tactile sensors
            output_size: Number of output neurons (default: 2 * width * height)
        """
        if output_size is None:
            output_size = 2 * sensor_grid[0] * sensor_grid[1]  # Fast + Slow adapting
        super().__init__(output_size)
        
        self.sensor_grid = sensor_grid
        self.width, self.height = sensor_grid
        
        # Create tactile receptive fields
        self.receptive_fields = self._create_tactile_fields()
        
    def _create_tactile_fields(self) -> List[Dict[str, Any]]:
        """Create tactile receptive fields."""
        fields = []
        neuron_id = 0
        
        for y in range(self.height):
            for x in range(self.width):
                # Fast adapting (FA) mechanoreceptor
                fields.append({
                    'neuron_id': neuron_id,
                    'x': x, 'y': y,
                    'type': 'FA',
                    'sensitivity': 1.0,
                    'adaptation_rate': 0.8
                })
                neuron_id += 1
                
                # Slow adapting (SA) mechanoreceptor
                fields.append({
                    'neuron_id': neuron_id,
                    'x': x, 'y': y,
                    'type': 'SA',
                    'sensitivity': 0.5,
                    'adaptation_rate': 0.2
                })
                neuron_id += 1
                
        return fields
        
    def _compute_tactile_response(self, pressure_map: np.ndarray, field: Dict[str, Any]) -> float:
        """Compute tactile response for a receptive field."""
        x, y = field['x'], field['y']
        
        # Get pressure at sensor location
        if 0 <= y < pressure_map.shape[0] and 0 <= x < pressure_map.shape[1]:
            pressure = pressure_map[y, x]
        else:
            pressure = 0.0
            
        # Apply sensitivity and adaptation
        sensitivity = field['sensitivity']
        adaptation_rate = field['adaptation_rate']
        
        # Simple pressure-to-response mapping
        response = sensitivity * pressure * (1.0 - adaptation_rate)
        
        return response
        
    def encode(self, pressure_map: np.ndarray, time_window: float = 100.0) -> List[Tuple[int, float]]:
        """
        Encode tactile pressure map into spike trains.
        
        Args:
            pressure_map: Pressure map (values 0-1)
            time_window: Time window for encoding (ms)
            
        Returns:
            List of (neuron_id, spike_time) tuples
        """
        spikes = []
        
        # Normalize pressure map
        if pressure_map.max() > 1.0:
            pressure_map = pressure_map / pressure_map.max()
            
        # Compute responses for all tactile fields
        responses = []
        for field in self.receptive_fields:
            response = self._compute_tactile_response(pressure_map, field)
            responses.append((field['neuron_id'], response))
            
        # Convert responses to spike times
        max_response = max(r for _, r in responses) if responses else 1.0
        
        for neuron_id, response in responses:
            if response > 0.1 * max_response:  # Threshold
                # Convert response to spike time (stronger response = earlier spike)
                spike_time = time_window * (1.0 - response / max_response)
                spikes.append((neuron_id, spike_time))
                
        return spikes


class MultiModalEncoder:
    """Combines multiple sensory encoders."""
    
    def __init__(self, encoders: Dict[str, SensoryEncoder]):
        """
        Initialize multimodal encoder.
        
        Args:
            encoders: Dictionary of sensory encoders
        """
        self.encoders = encoders
        self.total_output_size = sum(encoder.output_size for encoder in encoders.values())
        
    def encode(self, inputs: Dict[str, Any], time_window: float = 100.0) -> List[Tuple[int, float]]:
        """
        Encode multiple sensory inputs.
        
        Args:
            inputs: Dictionary of sensory inputs
            time_window: Time window for encoding (ms)
            
        Returns:
            List of (neuron_id, spike_time) tuples
        """
        all_spikes = []
        neuron_offset = 0
        
        for modality, encoder in self.encoders.items():
            if modality in inputs:
                spikes = encoder.encode(inputs[modality], time_window)
                
                # Adjust neuron IDs to avoid conflicts
                adjusted_spikes = [(neuron_id + neuron_offset, spike_time) 
                                 for neuron_id, spike_time in spikes]
                all_spikes.extend(adjusted_spikes)
                
                neuron_offset += encoder.output_size
                
        return all_spikes
        
    def reset(self):
        """Reset all encoders."""
        for encoder in self.encoders.values():
            encoder.reset()


class RateEncoder(SensoryEncoder):
    """
    Rate-based encoder for continuous signals.
    
    Converts continuous values to spike rates.
    """
    
    def __init__(self, input_size: int, output_size: int, max_rate: float = 100.0):
        """
        Initialize rate encoder.
        
        Args:
            input_size: Number of input dimensions
            output_size: Number of output neurons
            max_rate: Maximum firing rate (Hz)
        """
        super().__init__(output_size)
        self.input_size = input_size
        self.max_rate = max_rate
        
        # Create encoding weights
        self.encoding_weights = np.random.randn(output_size, input_size)
        self.encoding_weights /= np.linalg.norm(self.encoding_weights, axis=1, keepdims=True)
        
    def encode(self, input_vector: np.ndarray, time_window: float = 100.0) -> List[Tuple[int, float]]:
        """
        Encode continuous input vector into spike trains.
        
        Args:
            input_vector: Input vector (normalized to [0, 1])
            time_window: Time window for encoding (ms)
            
        Returns:
            List of (neuron_id, spike_time) tuples
        """
        spikes = []
        
        # Normalize input
        if input_vector.max() > 1.0:
            input_vector = input_vector / input_vector.max()
            
        # Compute encoding responses
        responses = self.encoding_weights @ input_vector
        
        # Convert to spike times
        for neuron_id, response in enumerate(responses):
            if response > 0:
                # Convert rate to spike time
                rate = response * self.max_rate
                if rate > 0:
                    # Poisson spike generation
                    spike_times = np.random.exponential(1000.0 / rate, 
                                                      size=int(time_window * rate / 1000.0))
                    spike_times = np.cumsum(spike_times)
                    spike_times = spike_times[spike_times < time_window]
                    
                    for spike_time in spike_times:
                        spikes.append((neuron_id, spike_time))
                        
        return spikes 