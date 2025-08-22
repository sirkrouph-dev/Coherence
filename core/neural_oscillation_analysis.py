#!/usr/bin/env python3
\"\"\"
Neural Oscillation Analysis System
=================================

Task 6: Implements spectral analysis and oscillation detection for neuromorphic networks.
Detects gamma (30-100 Hz) and theta (4-8 Hz) rhythms, analyzes coherence between layers,
and modulates plasticity based on oscillatory phase.
\"\"\"

import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import scipy.signal as signal
from scipy.fft import fft, fftfreq
import warnings

try:
    from scipy import signal as scipy_signal
    from scipy.stats import pearsonr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn(\"SciPy not available - using simplified oscillation analysis\")


class OscillationType(Enum):
    \"\"\"Types of neural oscillations.\"\"\"
    GAMMA = \"gamma\"  # 30-100 Hz
    BETA = \"beta\"  # 13-30 Hz
    ALPHA = \"alpha\"  # 8-13 Hz
    THETA = \"theta\"  # 4-8 Hz
    DELTA = \"delta\"  # 0.5-4 Hz


@dataclass
class OscillationBand:
    \"\"\"Definition of an oscillation frequency band.\"\"\"
    name: str
    freq_min: float
    freq_max: float
    typical_freq: float
    
    def contains_frequency(self, freq: float) -> bool:
        \"\"\"Check if frequency falls within this band.\"\"\"
        return self.freq_min <= freq <= self.freq_max


@dataclass 
class OscillationDetectionConfig:
    \"\"\"Configuration for oscillation detection.\"\"\"
    
    # Sampling parameters
    sampling_rate: float = 1000.0  # Hz
    window_size: float = 1.0  # seconds
    overlap: float = 0.5  # 50% overlap
    
    # Detection thresholds
    power_threshold: float = 2.0  # Minimum relative power
    peak_prominence: float = 0.1  # Minimum peak prominence
    min_duration: float = 0.1  # Minimum oscillation duration (s)
    
    # Frequency bands
    gamma_band: OscillationBand = field(default_factory=lambda: OscillationBand(\"gamma\", 30, 100, 40))
    beta_band: OscillationBand = field(default_factory=lambda: OscillationBand(\"beta\", 13, 30, 20))
    alpha_band: OscillationBand = field(default_factory=lambda: OscillationBand(\"alpha\", 8, 13, 10))
    theta_band: OscillationBand = field(default_factory=lambda: OscillationBand(\"theta\", 4, 8, 6))
    delta_band: OscillationBand = field(default_factory=lambda: OscillationBand(\"delta\", 0.5, 4, 2))
    
    # Coherence analysis
    coherence_window: float = 0.5  # seconds
    coherence_threshold: float = 0.3  # Minimum coherence
    
    def get_band(self, oscillation_type: OscillationType) -> OscillationBand:
        \"\"\"Get frequency band for oscillation type.\"\"\"
        band_map = {
            OscillationType.GAMMA: self.gamma_band,
            OscillationType.BETA: self.beta_band,
            OscillationType.ALPHA: self.alpha_band,
            OscillationType.THETA: self.theta_band,
            OscillationType.DELTA: self.delta_band
        }
        return band_map[oscillation_type]


@dataclass
class OscillationResult:
    \"\"\"Result of oscillation detection.\"\"\"
    oscillation_type: OscillationType
    frequency: float
    power: float
    phase: float
    duration: float
    start_time: float
    end_time: float
    confidence: float
    

@dataclass
class CoherenceResult:
    \"\"\"Result of coherence analysis between signals.\"\"\"
    coherence_value: float
    frequency: float
    phase_difference: float
    significance: float


class OscillationAnalyzer:
    \"\"\"Main class for analyzing neural oscillations in spike data.\"\"\"
    
    def __init__(self, config: Optional[OscillationDetectionConfig] = None):
        \"\"\"Initialize oscillation analyzer.\"\"\"
        self.config = config or OscillationDetectionConfig()
        self.detected_oscillations = []
        self.power_spectra = {}
        self.coherence_matrices = {}
        
        print(\"Neural Oscillation Analyzer initialized\")
        print(f\"  Sampling rate: {self.config.sampling_rate} Hz\")
        print(f\"  Window size: {self.config.window_size} s\")
        print(f\"  Gamma band: {self.config.gamma_band.freq_min}-{self.config.gamma_band.freq_max} Hz\")
        print(f\"  Theta band: {self.config.theta_band.freq_min}-{self.config.theta_band.freq_max} Hz\")
        
    def spike_times_to_signal(self, spike_times: List[float], duration: float, 
                             bin_size: float = 0.001) -> Tuple[np.ndarray, np.ndarray]:
        \"\"\"Convert spike times to continuous signal for spectral analysis.\"\"\"
        
        # Create time bins
        time_bins = np.arange(0, duration, bin_size)
        
        # Create spike histogram
        spike_counts, _ = np.histogram(spike_times, bins=time_bins)
        
        # Smooth with Gaussian to create continuous signal
        if SCIPY_AVAILABLE:
            # Use Gaussian smoothing
            sigma = 5  # bins
            smoothed_signal = scipy_signal.gaussian(len(spike_counts), sigma)
            signal_array = np.convolve(spike_counts, smoothed_signal, mode='same')
        else:
            # Simple moving average
            window = 5
            signal_array = np.convolve(spike_counts, np.ones(window)/window, mode='same')
            
        return time_bins[:-1], signal_array
        
    def compute_power_spectrum(self, signal_data: np.ndarray, 
                              time_array: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        \"\"\"Compute power spectral density of signal.\"\"\"
        
        if time_array is not None:
            dt = np.mean(np.diff(time_array))
            fs = 1.0 / dt
        else:
            fs = self.config.sampling_rate
            
        if SCIPY_AVAILABLE:
            # Use Welch's method for robust PSD estimation
            nperseg = min(len(signal_data), int(fs * self.config.window_size))
            freqs, psd = scipy_signal.welch(
                signal_data, 
                fs=fs, 
                nperseg=nperseg,
                overlap=int(nperseg * self.config.overlap)
            )
        else:
            # Simple FFT-based PSD
            fft_result = fft(signal_data)
            freqs = fftfreq(len(signal_data), 1/fs)
            psd = np.abs(fft_result)**2
            
            # Take positive frequencies only
            positive_freq_idx = freqs >= 0
            freqs = freqs[positive_freq_idx]
            psd = psd[positive_freq_idx]
            
        return freqs, psd
        
    def detect_gamma_oscillations(self, signal_data: np.ndarray, 
                                 time_array: Optional[np.ndarray] = None) -> List[OscillationResult]:
        \"\"\"Detect gamma oscillations (30-100 Hz) in signal.\"\"\"
        
        freqs, psd = self.compute_power_spectrum(signal_data, time_array)
        gamma_band = self.config.get_band(OscillationType.GAMMA)
        
        # Find frequencies within gamma band
        gamma_mask = (freqs >= gamma_band.freq_min) & (freqs <= gamma_band.freq_max)
        gamma_freqs = freqs[gamma_mask]
        gamma_power = psd[gamma_mask]
        
        if len(gamma_power) == 0:
            return []
            
        # Find peaks in gamma band
        if SCIPY_AVAILABLE:
            peaks, properties = scipy_signal.find_peaks(
                gamma_power, 
                prominence=self.config.peak_prominence * np.max(gamma_power)
            )
        else:
            # Simple peak detection
            peaks = []
            for i in range(1, len(gamma_power)-1):
                if (gamma_power[i] > gamma_power[i-1] and 
                    gamma_power[i] > gamma_power[i+1] and
                    gamma_power[i] > self.config.peak_prominence * np.max(gamma_power)):
                    peaks.append(i)
            peaks = np.array(peaks)
            
        # Create oscillation results
        gamma_oscillations = []
        for peak_idx in peaks:
            freq = gamma_freqs[peak_idx]
            power = gamma_power[peak_idx]
            
            # Estimate phase (simplified)
            phase = np.angle(fft(signal_data))[int(freq * len(signal_data) / self.config.sampling_rate)]
            
            # Calculate confidence based on peak prominence
            baseline_power = np.median(gamma_power)
            confidence = min(1.0, (power - baseline_power) / baseline_power)
            
            oscillation = OscillationResult(
                oscillation_type=OscillationType.GAMMA,
                frequency=freq,
                power=power,
                phase=phase,
                duration=self.config.window_size,  # Simplified
                start_time=0.0,
                end_time=self.config.window_size,
                confidence=confidence
            )
            
            gamma_oscillations.append(oscillation)
            
        return gamma_oscillations
        
    def detect_theta_oscillations(self, signal_data: np.ndarray,
                                 time_array: Optional[np.ndarray] = None) -> List[OscillationResult]:
        \"\"\"Detect theta oscillations (4-8 Hz) in signal.\"\"\"
        
        freqs, psd = self.compute_power_spectrum(signal_data, time_array)
        theta_band = self.config.get_band(OscillationType.THETA)
        
        # Find frequencies within theta band
        theta_mask = (freqs >= theta_band.freq_min) & (freqs <= theta_band.freq_max)
        theta_freqs = freqs[theta_mask]
        theta_power = psd[theta_mask]
        
        if len(theta_power) == 0:
            return []
            
        # Find dominant theta frequency
        peak_idx = np.argmax(theta_power)
        peak_power = theta_power[peak_idx]
        baseline_power = np.median(psd)
        
        # Check if theta power is significant
        if peak_power > baseline_power * self.config.power_threshold:
            freq = theta_freqs[peak_idx]
            
            # Estimate phase
            phase = np.angle(fft(signal_data))[int(freq * len(signal_data) / self.config.sampling_rate)]
            
            confidence = min(1.0, (peak_power - baseline_power) / baseline_power)
            
            oscillation = OscillationResult(
                oscillation_type=OscillationType.THETA,
                frequency=freq,
                power=peak_power,
                phase=phase,
                duration=self.config.window_size,
                start_time=0.0,
                end_time=self.config.window_size,
                confidence=confidence
            )
            
            return [oscillation]
            
        return []
        
    def analyze_spike_data(self, spike_times: List[float], duration: float) -> List[OscillationResult]:
        \"\"\"Analyze spike data for all oscillation types.\"\"\"
        
        # Convert spike times to signal
        time_array, signal_data = self.spike_times_to_signal(spike_times, duration)
        
        # Detect different oscillation types
        all_oscillations = []
        
        # Gamma oscillations
        gamma_oscillations = self.detect_gamma_oscillations(signal_data, time_array)
        all_oscillations.extend(gamma_oscillations)
        
        # Theta oscillations
        theta_oscillations = self.detect_theta_oscillations(signal_data, time_array)
        all_oscillations.extend(theta_oscillations)
        
        # Store results
        self.detected_oscillations.extend(all_oscillations)
        
        # Store power spectrum
        freqs, psd = self.compute_power_spectrum(signal_data, time_array)
        self.power_spectra[len(self.power_spectra)] = {'freqs': freqs, 'psd': psd}
        
        return all_oscillations
        
    def compute_coherence(self, signal1: np.ndarray, signal2: np.ndarray,
                         time_array: Optional[np.ndarray] = None) -> Dict[str, CoherenceResult]:
        \"\"\"Compute coherence between two signals across frequency bands.\"\"\"
        
        if time_array is not None:
            dt = np.mean(np.diff(time_array))
            fs = 1.0 / dt
        else:
            fs = self.config.sampling_rate
            
        coherence_results = {}
        
        if SCIPY_AVAILABLE:
            # Compute coherence using scipy
            freqs, coherence = scipy_signal.coherence(
                signal1, signal2, fs=fs,
                nperseg=int(fs * self.config.coherence_window)
            )
            
            # Analyze coherence in each frequency band
            for osc_type in OscillationType:
                band = self.config.get_band(osc_type)
                band_mask = (freqs >= band.freq_min) & (freqs <= band.freq_max)
                
                if np.any(band_mask):
                    band_coherence = coherence[band_mask]
                    band_freqs = freqs[band_mask]
                    
                    # Find peak coherence in band
                    peak_idx = np.argmax(band_coherence)
                    peak_coherence = band_coherence[peak_idx]
                    peak_freq = band_freqs[peak_idx]
                    
                    # Compute phase difference (simplified)
                    cross_spectrum = np.fft.fft(signal1) * np.conj(np.fft.fft(signal2))
                    phase_diff = np.angle(cross_spectrum[int(peak_freq * len(signal1) / fs)])
                    
                    # Significance test (simplified)
                    significance = 1.0 if peak_coherence > self.config.coherence_threshold else 0.0
                    
                    coherence_results[osc_type.value] = CoherenceResult(
                        coherence_value=peak_coherence,
                        frequency=peak_freq,
                        phase_difference=phase_diff,
                        significance=significance
                    )
        else:
            # Simplified coherence calculation
            correlation, _ = pearsonr(signal1, signal2) if len(signal1) == len(signal2) else (0.0, 1.0)
            
            for osc_type in OscillationType:
                coherence_results[osc_type.value] = CoherenceResult(
                    coherence_value=abs(correlation),
                    frequency=self.config.get_band(osc_type).typical_freq,
                    phase_difference=0.0,
                    significance=1.0 if abs(correlation) > self.config.coherence_threshold else 0.0
                )
                
        return coherence_results
        
    def analyze_layer_coherence(self, layer_spike_data: Dict[str, List[float]], 
                               duration: float) -> Dict[str, Dict[str, CoherenceResult]]:
        \"\"\"Analyze coherence between different network layers.\"\"\"
        
        # Convert spike data to signals for each layer
        layer_signals = {}
        for layer_name, spike_times in layer_spike_data.items():
            time_array, signal_data = self.spike_times_to_signal(spike_times, duration)
            layer_signals[layer_name] = signal_data
            
        # Compute pairwise coherence between layers
        coherence_matrix = {}
        layer_names = list(layer_signals.keys())
        
        for i, layer1 in enumerate(layer_names):
            coherence_matrix[layer1] = {}
            for j, layer2 in enumerate(layer_names):
                if i != j:
                    coherence_results = self.compute_coherence(
                        layer_signals[layer1], 
                        layer_signals[layer2]
                    )
                    coherence_matrix[layer1][layer2] = coherence_results
                else:
                    # Self-coherence (always 1.0)
                    coherence_matrix[layer1][layer2] = {
                        osc_type.value: CoherenceResult(1.0, 
                            self.config.get_band(osc_type).typical_freq, 0.0, 1.0)
                        for osc_type in OscillationType
                    }
                    
        # Store results
        self.coherence_matrices[len(self.coherence_matrices)] = coherence_matrix
        
        return coherence_matrix
        
    def get_oscillation_summary(self) -> Dict[str, Any]:
        \"\"\"Get summary of detected oscillations.\"\"\"
        
        summary = {
            'total_oscillations': len(self.detected_oscillations),
            'by_type': {},
            'frequency_ranges': {},
            'mean_confidence': 0.0
        }
        
        if self.detected_oscillations:
            # Count by type
            for osc_type in OscillationType:
                type_oscillations = [osc for osc in self.detected_oscillations 
                                   if osc.oscillation_type == osc_type]
                summary['by_type'][osc_type.value] = {
                    'count': len(type_oscillations),
                    'mean_frequency': np.mean([osc.frequency for osc in type_oscillations]) if type_oscillations else 0.0,
                    'mean_power': np.mean([osc.power for osc in type_oscillations]) if type_oscillations else 0.0
                }
                
            # Overall statistics
            all_confidences = [osc.confidence for osc in self.detected_oscillations]
            summary['mean_confidence'] = np.mean(all_confidences)
            
            # Frequency ranges
            all_frequencies = [osc.frequency for osc in self.detected_oscillations]
            summary['frequency_ranges'] = {
                'min': np.min(all_frequencies),
                'max': np.max(all_frequencies),
                'mean': np.mean(all_frequencies)
            }
            
        return summary
        
    def plot_power_spectrum(self, spectrum_id: int = 0, show_bands: bool = True) -> None:
        \"\"\"Plot power spectrum with oscillation bands marked.\"\"\"
        
        if spectrum_id not in self.power_spectra:
            print(f\"No power spectrum with ID {spectrum_id}\")
            return
            
        spectrum_data = self.power_spectra[spectrum_id]
        freqs = spectrum_data['freqs']
        psd = spectrum_data['psd']
        
        plt.figure(figsize=(12, 6))
        plt.semilogy(freqs, psd)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectral Density')
        plt.title('Neural Oscillation Power Spectrum')
        
        if show_bands:
            # Mark frequency bands
            band_colors = {
                'gamma': 'red',
                'beta': 'orange', 
                'alpha': 'green',
                'theta': 'blue',
                'delta': 'purple'
            }
            
            for osc_type in OscillationType:
                band = self.config.get_band(osc_type)
                color = band_colors.get(osc_type.value, 'gray')
                plt.axvspan(band.freq_min, band.freq_max, alpha=0.2, color=color, label=f'{osc_type.value} band')
                
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
    def reset_analysis(self):
        \"\"\"Reset all analysis results.\"\"\"
        self.detected_oscillations = []
        self.power_spectra = {}
        self.coherence_matrices = {}


class OscillationModulatedPlasticity:
    \"\"\"Plasticity rule modulated by oscillatory phase.\"\"\"
    
    def __init__(self, base_learning_rate: float = 0.01, 
                 gamma_modulation_strength: float = 0.5,
                 theta_modulation_strength: float = 0.3):
        \"\"\"Initialize oscillation-modulated plasticity.\"\"\"
        self.base_learning_rate = base_learning_rate
        self.gamma_modulation_strength = gamma_modulation_strength
        self.theta_modulation_strength = theta_modulation_strength
        
        self.current_gamma_phase = 0.0
        self.current_theta_phase = 0.0
        
        print(\"Oscillation-Modulated Plasticity initialized\")
        print(f\"  Base learning rate: {base_learning_rate}\")
        print(f\"  Gamma modulation: {gamma_modulation_strength}\")
        print(f\"  Theta modulation: {theta_modulation_strength}\")
        
    def update_oscillation_phases(self, gamma_phase: float, theta_phase: float):
        \"\"\"Update current oscillation phases.\"\"\"
        self.current_gamma_phase = gamma_phase
        self.current_theta_phase = theta_phase
        
    def compute_modulated_learning_rate(self, pre_spike: bool, post_spike: bool) -> float:
        \"\"\"Compute learning rate modulated by oscillation phase.\"\"\"
        
        # Base STDP learning rate
        learning_rate = self.base_learning_rate
        
        # Gamma phase modulation (enhances fast learning)
        if pre_spike and post_spike:
            # Gamma phase modulation: enhance LTP during certain phases
            gamma_factor = 1.0 + self.gamma_modulation_strength * np.cos(self.current_gamma_phase)
            learning_rate *= gamma_factor
            
        # Theta phase modulation (gates learning windows)
        # Theta creates temporal windows for enhanced plasticity
        theta_gate = 0.5 + 0.5 * np.cos(self.current_theta_phase)  # 0 to 1
        theta_factor = theta_gate * self.theta_modulation_strength + (1 - self.theta_modulation_strength)
        learning_rate *= theta_factor
        
        return learning_rate
        
    def compute_weight_change(self, pre_spike: bool, post_spike: bool, 
                            current_weight: float, dt: float = 0.001) -> float:
        \"\"\"Compute weight change with oscillation modulation.\"\"\"
        
        # Get modulated learning rate
        learning_rate = self.compute_modulated_learning_rate(pre_spike, post_spike)
        
        # Standard STDP rule with modulated learning rate
        if pre_spike and post_spike:
            # LTP
            delta_w = learning_rate * (1.0 - current_weight / 10.0)  # Weight-dependent
        elif pre_spike and not post_spike:
            # Weak LTD
            delta_w = -learning_rate * 0.1 * current_weight / 10.0
        else:
            delta_w = 0.0
            
        return delta_w


def create_demo_oscillation_analysis():
    \"\"\"Demonstrate oscillation analysis capabilities.\"\"\"
    
    print(\"=== Neural Oscillation Analysis Demo ===\")
    
    # Create analyzer
    config = OscillationDetectionConfig(
        sampling_rate=1000.0,
        window_size=2.0,
        power_threshold=1.5
    )
    analyzer = OscillationAnalyzer(config)
    
    # Generate synthetic spike data with embedded oscillations
    duration = 2.0  # 2 seconds
    
    # Create gamma-modulated spikes (40 Hz)
    gamma_freq = 40.0
    theta_freq = 6.0
    
    spike_times = []
    t = 0.0
    dt = 0.001
    
    while t < duration:
        # Gamma modulation
        gamma_modulation = 0.5 + 0.5 * np.sin(2 * np.pi * gamma_freq * t)
        
        # Theta modulation  
        theta_modulation = 0.3 + 0.7 * np.sin(2 * np.pi * theta_freq * t)
        
        # Combined spike probability
        spike_prob = 0.01 * gamma_modulation * theta_modulation
        
        if np.random.random() < spike_prob:
            spike_times.append(t)
            
        t += dt
        
    print(f\"\nGenerated {len(spike_times)} spikes over {duration} seconds\")
    
    # Analyze oscillations
    oscillations = analyzer.analyze_spike_data(spike_times, duration)
    
    print(f\"\nDetected {len(oscillations)} oscillations:\")
    for i, osc in enumerate(oscillations):
        print(f\"  {i+1}. {osc.oscillation_type.value}: {osc.frequency:.1f} Hz, \"
              f\"power: {osc.power:.3f}, confidence: {osc.confidence:.3f}\")
              
    # Get summary
    summary = analyzer.get_oscillation_summary()
    print(f\"\nOscillation Summary:\")
    print(f\"  Total oscillations: {summary['total_oscillations']}\")
    print(f\"  Mean confidence: {summary['mean_confidence']:.3f}\")
    
    for osc_type, stats in summary['by_type'].items():
        if stats['count'] > 0:
            print(f\"  {osc_type}: {stats['count']} detected, \"
                  f\"mean freq: {stats['mean_frequency']:.1f} Hz\")
    
    # Test oscillation-modulated plasticity
    print(\"\n=== Oscillation-Modulated Plasticity Demo ===\")
    
    plasticity = OscillationModulatedPlasticity(
        base_learning_rate=0.01,
        gamma_modulation_strength=0.5,
        theta_modulation_strength=0.3
    )
    
    # Simulate plasticity at different oscillation phases
    phases = np.linspace(0, 2*np.pi, 8)
    weight = 1.0
    
    print(\"\nWeight changes at different oscillation phases:\")
    for i, phase in enumerate(phases):
        plasticity.update_oscillation_phases(gamma_phase=phase, theta_phase=phase/5)
        
        delta_w = plasticity.compute_weight_change(
            pre_spike=True, post_spike=True, current_weight=weight
        )
        
        weight += delta_w
        print(f\"  Phase {phase:.2f}: Δw = {delta_w:+.4f}, weight = {weight:.4f}\")
        
    return analyzer, plasticity


if __name__ == \"__main__\":
    # Run demonstration
    analyzer, plasticity = create_demo_oscillation_analysis()
    
    print(\"\n✅ Neural Oscillation Analysis System demonstration completed!\")
    print(\"\nKey features implemented:\")
    print(\"  • Gamma oscillation detection (30-100 Hz)\")
    print(\"  • Theta oscillation detection (4-8 Hz)\")
    print(\"  • Power spectral density analysis\")
    print(\"  • Inter-layer coherence analysis\")
    print(\"  • Oscillation-modulated plasticity\")
    print(\"  • Phase-dependent learning rate modulation\")
