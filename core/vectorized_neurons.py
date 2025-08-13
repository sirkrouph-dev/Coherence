"""
Vectorized neuron implementations optimized for large-scale networks.
Uses Structure of Arrays (SoA) for maximum NumPy performance.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod

from .logging_utils import neuromorphic_logger


class VectorizedNeuronModel(ABC):
    """
    Base class for vectorized neuron models using Structure of Arrays.
    Optimized for large-scale simulations with NumPy operations.
    """
    
    def __init__(self, population_size: int, **kwargs):
        """Initialize vectorized neuron population."""
        self.population_size = population_size
        self.current_time = 0.0
        
        # Core state arrays - all neurons in single arrays
        self.membrane_potential = np.full(population_size, -65.0, dtype=np.float32)
        self.spike_flags = np.zeros(population_size, dtype=bool)
        self.refractory_time = np.zeros(population_size, dtype=np.float32)
        
        # Spike tracking - more memory efficient
        self.last_spike_time = np.full(population_size, -np.inf, dtype=np.float32)
        self.spike_count = np.zeros(population_size, dtype=np.uint32)
        
        # Model-specific initialization
        self._initialize_parameters(**kwargs)
        
    @abstractmethod
    def _initialize_parameters(self, **kwargs):
        """Initialize model-specific parameters."""
        pass
    
    @abstractmethod
    def _compute_membrane_dynamics(self, dt: float, I_syn: np.ndarray) -> np.ndarray:
        """Compute membrane potential derivatives."""
        pass
    
    def step(self, dt: float, I_syn: np.ndarray) -> np.ndarray:
        """
        Advance all neurons by one time step - FULLY VECTORIZED.
        
        Args:
            dt: Time step in milliseconds
            I_syn: Synaptic currents for each neuron [population_size]
            
        Returns:
            Boolean array indicating which neurons spiked [population_size]
        """
        if len(I_syn) != self.population_size:
            raise ValueError(f"Expected {self.population_size} currents, got {len(I_syn)}")
            
        # Reset spike flags
        self.spike_flags.fill(False)
        
        # Handle refractory period - VECTORIZED
        active_mask = self.refractory_time <= 0
        self.refractory_time -= dt
        self.refractory_time = np.maximum(0, self.refractory_time)
        
        # Compute dynamics only for non-refractory neurons
        if np.any(active_mask):
            # Get membrane dynamics - VECTORIZED
            dv_dt = self._compute_membrane_dynamics(dt, I_syn)
            
            # Update membrane potential - VECTORIZED
            self.membrane_potential[active_mask] += dv_dt[active_mask] * dt
        
        # Spike detection - VECTORIZED
        self._detect_spikes(dt)
        
        self.current_time += dt
        return self.spike_flags.copy()
    
    def _detect_spikes(self, dt: float):
        """Detect spikes and handle reset - VECTORIZED."""
        # Subclass implements specific spike conditions
        pass
    
    def reset(self):
        """Reset all neurons to initial state."""
        self.membrane_potential.fill(-65.0)
        self.spike_flags.fill(False)
        self.refractory_time.fill(0.0)
        self.last_spike_time.fill(-np.inf)
        self.spike_count.fill(0)
        self.current_time = 0.0
    
    def get_spike_times(self) -> List[List[float]]:
        """Get spike times - optimized version returns approximate."""
        # For large populations, exact spike times are memory intensive
        # Return last spike time per neuron instead
        return [[self.last_spike_time[i]] if self.last_spike_time[i] > -np.inf else [] 
                for i in range(self.population_size)]
    
    def get_firing_rates(self, time_window: float = 1000.0) -> np.ndarray:
        """Get firing rates over recent time window."""
        if self.current_time < time_window:
            window = self.current_time
        else:
            window = time_window
            
        if window <= 0:
            return np.zeros(self.population_size)
            
        # Approximate firing rate from recent activity
        recent_activity = (self.current_time - self.last_spike_time) < window
        return (self.spike_count * 1000.0 / max(self.current_time, 1.0)).astype(np.float32)


class VectorizedAdExNeuron(VectorizedNeuronModel):
    """
    Vectorized Adaptive Exponential Integrate-and-Fire neuron.
    Optimized for large populations with NumPy operations.
    """
    
    def _initialize_parameters(self, **kwargs):
        """Initialize AdEx-specific parameters."""
        # Extract parameters with defaults
        self.tau_m = kwargs.get('tau_m', 20.0)
        self.v_rest = kwargs.get('v_rest', -65.0)
        self.v_thresh = kwargs.get('v_thresh', -55.0)
        self.delta_t = kwargs.get('delta_t', 2.0)
        self.tau_w = kwargs.get('tau_w', 144.0)
        self.a = kwargs.get('a', 4.0)
        self.b = kwargs.get('b', 0.0805)
        self.v_reset = kwargs.get('v_reset', -65.0)
        self.refractory_period = kwargs.get('refractory_period', 2.0)
        
        # Adaptation current for each neuron
        self.adaptation_current = np.zeros(self.population_size, dtype=np.float32)
        
        # Add parameter heterogeneity for realistic dynamics
        self._add_parameter_heterogeneity()
    
    def _add_parameter_heterogeneity(self):
        """Add realistic parameter heterogeneity across population."""
        # Small variations in key parameters
        np.random.seed(42)  # Reproducible heterogeneity
        
        # Threshold variation (±2mV)
        thresh_var = np.random.normal(0, 1.0, self.population_size)
        self.v_thresh_array = np.full(self.population_size, self.v_thresh) + thresh_var
        
        # Membrane time constant variation (±10%)
        tau_var = np.random.normal(1.0, 0.1, self.population_size)
        self.tau_m_array = np.full(self.population_size, self.tau_m) * tau_var
        
        # Adaptation coupling variation (±20%)
        a_var = np.random.normal(1.0, 0.2, self.population_size)
        self.a_array = np.full(self.population_size, self.a) * a_var
    
    def _compute_membrane_dynamics(self, dt: float, I_syn: np.ndarray) -> np.ndarray:
        """Compute AdEx membrane dynamics - FULLY VECTORIZED."""
        # Input resistance scaling (100 MOhm typical)
        R_input = 100.0
        
        # AdEx dynamics - VECTORIZED
        exp_term = self.delta_t * np.exp(
            (self.membrane_potential - self.v_thresh_array) / self.delta_t
        )
        
        dv_dt = (
            -(self.membrane_potential - self.v_rest) 
            + exp_term
            - self.adaptation_current
            + I_syn * R_input
        ) / self.tau_m_array
        
        # Update adaptation current - VECTORIZED
        dw_dt = (self.a_array * (self.membrane_potential - self.v_rest) - self.adaptation_current) / self.tau_w
        self.adaptation_current += dw_dt * dt
        
        return dv_dt
    
    def _detect_spikes(self, dt: float):
        """Detect AdEx spikes and reset - VECTORIZED."""
        # Spike condition - VECTORIZED
        spike_mask = self.membrane_potential >= self.v_thresh_array
        
        if np.any(spike_mask):
            # Mark spikes
            self.spike_flags[spike_mask] = True
            
            # Update spike tracking
            self.last_spike_time[spike_mask] = self.current_time
            self.spike_count[spike_mask] += 1
            
            # Reset membrane potential
            self.membrane_potential[spike_mask] = self.v_reset
            
            # Add adaptation current increment
            self.adaptation_current[spike_mask] += self.b
            
            # Set refractory period
            self.refractory_time[spike_mask] = self.refractory_period


class VectorizedLIFNeuron(VectorizedNeuronModel):
    """
    Vectorized Leaky Integrate-and-Fire neuron.
    Simplified model for maximum computational efficiency.
    """
    
    def _initialize_parameters(self, **kwargs):
        """Initialize LIF-specific parameters."""
        self.tau_m = kwargs.get('tau_m', 15.0)  # Faster for better spiking
        self.v_rest = kwargs.get('v_rest', -65.0)
        self.v_thresh = kwargs.get('v_thresh', -60.0)  # Lower threshold
        self.v_reset = kwargs.get('v_reset', -65.0)
        self.refractory_period = kwargs.get('refractory_period', 2.0)
        
        # Add heterogeneity
        np.random.seed(42)
        thresh_var = np.random.normal(0, 0.5, self.population_size)
        self.v_thresh_array = np.full(self.population_size, self.v_thresh) + thresh_var
        
        tau_var = np.random.normal(1.0, 0.05, self.population_size)
        self.tau_m_array = np.full(self.population_size, self.tau_m) * tau_var
    
    def _compute_membrane_dynamics(self, dt: float, I_syn: np.ndarray) -> np.ndarray:
        """Compute LIF membrane dynamics - FULLY VECTORIZED."""
        R_input = 100.0  # Input resistance
        
        dv_dt = (
            -(self.membrane_potential - self.v_rest) + I_syn * R_input
        ) / self.tau_m_array
        
        return dv_dt
    
    def _detect_spikes(self, dt: float):
        """Detect LIF spikes and reset - VECTORIZED."""
        spike_mask = self.membrane_potential >= self.v_thresh_array
        
        if np.any(spike_mask):
            self.spike_flags[spike_mask] = True
            self.last_spike_time[spike_mask] = self.current_time
            self.spike_count[spike_mask] += 1
            self.membrane_potential[spike_mask] = self.v_reset
            self.refractory_time[spike_mask] = self.refractory_period


class VectorizedNeuronPopulation:
    """
    High-performance neuron population using vectorized models.
    Replaces the old object-oriented NeuronPopulation for large networks.
    """
    
    def __init__(self, size: int, neuron_type: str = "adex", **kwargs):
        """
        Initialize vectorized neuron population.
        
        Args:
            size: Number of neurons
            neuron_type: Type of neuron model
            **kwargs: Parameters for neuron model
        """
        self.size = size
        self.neuron_type = neuron_type
        
        # Create vectorized neuron model
        if neuron_type.lower() == "adex":
            self.model = VectorizedAdExNeuron(size, **kwargs)
        elif neuron_type.lower() == "lif":
            self.model = VectorizedLIFNeuron(size, **kwargs)
        else:
            raise ValueError(f"Unsupported vectorized neuron type: {neuron_type}")
        
        # Performance tracking
        self.step_count = 0
        self.total_spikes = 0
    
    def step(self, dt: float, I_syn: np.ndarray) -> np.ndarray:
        """
        Advance population by one time step - VECTORIZED.
        
        Args:
            dt: Time step in milliseconds
            I_syn: Synaptic currents [size]
            
        Returns:
            Boolean array indicating spikes [size]
        """
        spikes = self.model.step(dt, I_syn)
        
        # Update performance tracking
        self.step_count += 1
        self.total_spikes += np.sum(spikes)
        
        return spikes
    
    def get_membrane_potentials(self) -> np.ndarray:
        """Get membrane potentials for all neurons."""
        return self.model.membrane_potential.copy()
    
    def get_spike_states(self) -> np.ndarray:
        """Get current spike states."""
        return self.model.spike_flags.copy()
    
    def get_firing_rates(self, time_window: float = 1000.0) -> np.ndarray:
        """Get firing rates over time window."""
        return self.model.get_firing_rates(time_window)
    
    def reset(self):
        """Reset population to initial state."""
        self.model.reset()
        self.step_count = 0
        self.total_spikes = 0
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if self.step_count > 0:
            avg_spikes_per_step = self.total_spikes / self.step_count
            avg_firing_rate = avg_spikes_per_step / self.size
        else:
            avg_spikes_per_step = 0
            avg_firing_rate = 0
            
        return {
            "population_size": self.size,
            "neuron_type": self.neuron_type,
            "total_steps": self.step_count,
            "total_spikes": self.total_spikes,
            "avg_spikes_per_step": avg_spikes_per_step,
            "avg_firing_rate_hz": avg_firing_rate * 1000.0,  # Convert to Hz
            "current_time": self.model.current_time
        }


# Factory function for easy creation
def create_vectorized_population(size: int, neuron_type: str = "adex", **kwargs) -> VectorizedNeuronPopulation:
    """
    Create a vectorized neuron population.
    
    Args:
        size: Population size
        neuron_type: Type of neurons ("adex", "lif")
        **kwargs: Neuron parameters
        
    Returns:
        VectorizedNeuronPopulation instance
    """
    # Add optimized parameters for spiking
    if neuron_type.lower() == "lif":
        kwargs.setdefault("tau_m", 10.0)  # Faster integration
        kwargs.setdefault("v_thresh", -60.0)  # Lower threshold
    elif neuron_type.lower() == "adex":
        kwargs.setdefault("tau_m", 15.0)
        kwargs.setdefault("v_thresh", -60.0)
        kwargs.setdefault("a", 2.0)  # Reduced adaptation
        
    return VectorizedNeuronPopulation(size, neuron_type, **kwargs)
