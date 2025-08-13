"""
High-performance neuromorphic network optimized for large-scale simulations.
Uses vectorized operations and sparse matrices for maximum efficiency.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import time
import psutil

from .vectorized_neurons import VectorizedNeuronPopulation, create_vectorized_population
from .vectorized_synapses import VectorizedSynapseManager
from .logging_utils import neuromorphic_logger


class VectorizedNetworkLayer:
    """
    High-performance network layer using vectorized neurons.
    Optimized for large populations.
    """
    
    def __init__(self, name: str, size: int, neuron_type: str = "adex", **kwargs):
        """
        Initialize vectorized network layer.
        
        Args:
            name: Layer name
            size: Number of neurons
            neuron_type: Type of neurons
            **kwargs: Neuron parameters
        """
        self.name = name
        self.size = size
        self.neuron_type = neuron_type
        
        # Create vectorized neuron population
        self.population = create_vectorized_population(size, neuron_type, **kwargs)
        
        # Timing and performance tracking
        self.current_time = 0.0
        self.step_times = []
        
    def step(self, dt: float, I_syn: np.ndarray) -> np.ndarray:
        """
        Advance layer by one time step - VECTORIZED.
        
        Args:
            dt: Time step
            I_syn: Synaptic currents [size]
            
        Returns:
            Boolean array of spikes [size]
        """
        start_time = time.perf_counter()
        
        # Step the vectorized population
        spikes = self.population.step(dt, I_syn)
        
        self.current_time += dt
        
        # Track timing
        step_time = time.perf_counter() - start_time
        self.step_times.append(step_time)
        
        # Keep only recent timing data
        if len(self.step_times) > 1000:
            self.step_times.pop(0)
        
        return spikes
    
    def get_membrane_potentials(self) -> np.ndarray:
        """Get membrane potentials for all neurons."""
        return self.population.get_membrane_potentials()
    
    def get_spike_states(self) -> np.ndarray:
        """Get current spike states."""
        return self.population.get_spike_states()
    
    def get_firing_rates(self, time_window: float = 1000.0) -> np.ndarray:
        """Get firing rates over time window."""
        return self.population.get_firing_rates(time_window)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get layer performance statistics."""
        stats = self.population.get_performance_stats()
        
        if self.step_times:
            avg_step_time = float(np.mean(self.step_times))
            stats.update({
                "avg_step_time_ms": avg_step_time * 1000,
                "step_time_std_ms": float(np.std(self.step_times)) * 1000,
                "max_step_time_ms": float(np.max(self.step_times)) * 1000,
                "steps_per_second": 1.0 / max(avg_step_time, 1e-6)
            })
        
        return stats
    
    def reset(self):
        """Reset layer to initial state."""
        self.population.reset()
        self.current_time = 0.0
        self.step_times.clear()


class HighPerformanceNeuromorphicNetwork:
    """
    High-performance neuromorphic network for large-scale simulations.
    Uses vectorized operations throughout for maximum efficiency.
    """
    
    # Resource limits for large networks
    MAX_NEURONS = 10_000_000  # 10 million neurons
    MAX_SYNAPSES = 1_000_000_000  # 1 billion synapses
    MAX_LAYERS = 10000
    
    def __init__(self, enable_monitoring: bool = True):
        """
        Initialize high-performance network.
        
        Args:
            enable_monitoring: Whether to enable performance monitoring
        """
        self.layers: Dict[str, VectorizedNetworkLayer] = {}
        self.synapse_manager = VectorizedSynapseManager()
        self.current_time = 0.0
        self.enable_monitoring = enable_monitoring
        
        # Performance tracking
        self.simulation_history = []
        self.step_times = []
        self.memory_usage = []
        self.total_neurons = 0
        
        # Pre-allocated arrays for efficiency
        self.layer_spikes: Dict[str, np.ndarray] = {}
        self.layer_currents: Dict[str, np.ndarray] = {}
        
        neuromorphic_logger.log_info("Initialized high-performance neuromorphic network")
    
    def add_layer(self, name: str, size: int, neuron_type: str = "adex", **kwargs):
        """
        Add a layer to the network.
        
        Args:
            name: Layer name
            size: Number of neurons
            neuron_type: Type of neurons
            **kwargs: Neuron parameters
        """
        # Validation
        if not isinstance(name, str) or not name:
            raise ValueError("Layer name must be a non-empty string")
        if name in self.layers:
            raise ValueError(f"Layer '{name}' already exists")
        if not isinstance(size, int) or size <= 0:
            raise ValueError(f"Layer size must be positive integer, got {size}")
        if size > self.MAX_NEURONS:
            raise ValueError(f"Layer size {size} exceeds maximum {self.MAX_NEURONS}")
        if len(self.layers) >= self.MAX_LAYERS:
            raise ValueError(f"Maximum layers ({self.MAX_LAYERS}) exceeded")
        if self.total_neurons + size > self.MAX_NEURONS:
            raise ValueError(f"Total neurons would exceed limit {self.MAX_NEURONS}")
        
        # Create layer
        layer = VectorizedNetworkLayer(name, size, neuron_type, **kwargs)
        self.layers[name] = layer
        self.total_neurons += size
        
        # Register with synapse manager
        self.synapse_manager.add_layer(name, size)
        
        # Pre-allocate arrays
        self.layer_spikes[name] = np.zeros(size, dtype=bool)
        self.layer_currents[name] = np.zeros(size, dtype=np.float32)
        
        neuromorphic_logger.log_info(
            f"Added layer '{name}': {size} {neuron_type} neurons "
            f"(total: {self.total_neurons})"
        )
    
    def connect_layers(
        self,
        pre_layer: str,
        post_layer: str,
        synapse_type: str = "stdp",
        connection_probability: float = 0.1,
        **kwargs
    ):
        """
        Connect two layers with synapses.
        
        Args:
            pre_layer: Presynaptic layer name
            post_layer: Postsynaptic layer name
            synapse_type: Type of synapses
            connection_probability: Connection probability
            **kwargs: Synapse parameters
        """
        if pre_layer not in self.layers:
            raise ValueError(f"Pre-layer '{pre_layer}' not found")
        if post_layer not in self.layers:
            raise ValueError(f"Post-layer '{post_layer}' not found")
        if not 0.0 <= connection_probability <= 1.0:
            raise ValueError(f"Connection probability must be 0-1, got {connection_probability}")
        
        # Estimate synapse count
        pre_size = self.layers[pre_layer].size
        post_size = self.layers[post_layer].size
        estimated_synapses = int(pre_size * post_size * connection_probability)
        
        current_synapses = self.synapse_manager.get_total_synapses()
        if current_synapses + estimated_synapses > self.MAX_SYNAPSES:
            raise ValueError(f"Would exceed synapse limit {self.MAX_SYNAPSES}")
        
        # Create connection
        self.synapse_manager.connect_layers(
            pre_layer, post_layer, synapse_type, connection_probability, **kwargs
        )
        
        neuromorphic_logger.log_info(
            f"Connected {pre_layer} -> {post_layer}: ~{estimated_synapses} {synapse_type} synapses"
        )
    
    def step(self, dt: float):
        """
        Advance network by one time step - FULLY VECTORIZED.
        
        Args:
            dt: Time step in milliseconds
        """
        step_start_time = time.perf_counter()
        
        # Step 1: Compute synaptic currents for all layers - VECTORIZED
        current_layer_spikes = {name: layer.get_spike_states() 
                               for name, layer in self.layers.items()}
        
        layer_currents = self.synapse_manager.compute_layer_currents(
            current_layer_spikes, self.current_time
        )
        
        # Step 2: Step all neuron layers - VECTORIZED
        layer_spikes = {}
        for name, layer in self.layers.items():
            currents = layer_currents.get(name, np.zeros(layer.size, dtype=np.float32))
            spikes = layer.step(dt, currents)
            layer_spikes[name] = spikes
            
            # Update pre-allocated arrays
            self.layer_spikes[name] = spikes
            self.layer_currents[name] = currents
        
        # Step 3: Update synaptic weights - VECTORIZED
        self.synapse_manager.update_all_weights(layer_spikes, self.current_time)
        
        # Step 4: Step synapse populations
        self.synapse_manager.step_all(dt)
        
        self.current_time += dt
        
        # Performance monitoring
        if self.enable_monitoring:
            step_time = time.perf_counter() - step_start_time
            self.step_times.append(step_time)
            
            # Monitor memory usage periodically
            if len(self.step_times) % 100 == 0:
                memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
                self.memory_usage.append(memory_mb)
            
            # Keep recent history only
            if len(self.step_times) > 1000:
                self.step_times.pop(0)
            if len(self.memory_usage) > 100:
                self.memory_usage.pop(0)
    
    def run_simulation(self, duration: float, dt: float = 1.0) -> Dict[str, Any]:
        """
        Run high-performance simulation.
        
        Args:
            duration: Simulation duration in milliseconds
            dt: Time step in milliseconds
            
        Returns:
            Simulation results with performance metrics
        """
        if duration <= 0:
            raise ValueError(f"Duration must be positive, got {duration}")
        if dt <= 0:
            raise ValueError(f"Time step must be positive, got {dt}")
        if dt > duration:
            raise ValueError(f"Time step {dt} cannot exceed duration {duration}")
        
        num_steps = int(duration / dt)
        
        neuromorphic_logger.log_info(
            f"Starting high-performance simulation: {duration}ms, "
            f"{num_steps} steps, {self.total_neurons} neurons"
        )
        
        simulation_start = time.perf_counter()
        self.reset()
        
        # Main simulation loop - HIGHLY OPTIMIZED
        for step in range(num_steps):
            self.step(dt)
            
            # Minimal history recording for large networks
            if step % 100 == 0:
                total_spikes = sum(np.sum(spikes) for spikes in self.layer_spikes.values())
                self.simulation_history.append({
                    "time": self.current_time,
                    "total_spikes": total_spikes,
                    "step": step
                })
        
        simulation_time = time.perf_counter() - simulation_start
        
        # Collect results
        results = self._collect_results(simulation_time, duration, dt)
        
        neuromorphic_logger.log_info(
            f"Simulation completed: {simulation_time:.3f}s wall time, "
            f"{results['total_spikes']} total spikes"
        )
        
        return results
    
    def _collect_results(self, simulation_time: float, duration: float, dt: float) -> Dict[str, Any]:
        """Collect simulation results with performance metrics."""
        # Count total spikes
        total_spikes = 0
        layer_spike_counts = {}
        layer_firing_rates = {}
        
        for name, layer in self.layers.items():
            stats = layer.get_performance_stats()
            layer_spike_counts[name] = stats["total_spikes"]
            total_spikes += stats["total_spikes"]
            
            # Calculate average firing rate
            if duration > 0:
                avg_rate = (stats["total_spikes"] / layer.size) * (1000.0 / duration)
                layer_firing_rates[name] = avg_rate
            else:
                layer_firing_rates[name] = 0.0
        
        # Performance metrics
        performance_metrics = self._get_performance_metrics(simulation_time, dt)
        
        # Network statistics
        network_stats = {
            "total_neurons": self.total_neurons,
            "total_layers": len(self.layers),
            "total_synapses": self.synapse_manager.get_total_synapses(),
            "synapse_stats": self.synapse_manager.get_all_statistics()
        }
        
        return {
            "duration": duration,
            "dt": dt,
            "simulation_time": simulation_time,
            "final_time": self.current_time,
            "total_spikes": total_spikes,
            "layer_spike_counts": layer_spike_counts,
            "layer_firing_rates": layer_firing_rates,
            "performance_metrics": performance_metrics,
            "network_statistics": network_stats,
            "simulation_history": self.simulation_history[-10:]  # Last 10 entries only
        }
    
    def _get_performance_metrics(self, simulation_time: float, dt: float) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        metrics = {
            "wall_clock_time": simulation_time,
            "simulation_speed": (self.current_time / max(simulation_time, 1e-6)),
            "steps_per_second": len(self.step_times) / max(simulation_time, 1e-6)
        }
        
        if self.step_times:
            step_times_ms = np.array(self.step_times) * 1000
            metrics.update({
                "avg_step_time_ms": float(np.mean(step_times_ms)),
                "std_step_time_ms": float(np.std(step_times_ms)),
                "min_step_time_ms": float(np.min(step_times_ms)),
                "max_step_time_ms": float(np.max(step_times_ms)),
                "median_step_time_ms": float(np.median(step_times_ms))
            })
        
        if self.memory_usage:
            metrics.update({
                "peak_memory_mb": float(np.max(self.memory_usage)),
                "avg_memory_mb": float(np.mean(self.memory_usage)),
                "current_memory_mb": float(self.memory_usage[-1]) if self.memory_usage else 0
            })
        
        # Throughput metrics
        if simulation_time > 0 and self.total_neurons > 0:
            neurons_per_second = self.total_neurons * len(self.step_times) / simulation_time
            metrics["neurons_per_second"] = neurons_per_second
            metrics["neuron_steps_per_second"] = neurons_per_second
        
        return metrics
    
    def reset(self):
        """Reset network to initial state."""
        for layer in self.layers.values():
            layer.reset()
        
        self.synapse_manager.reset_all()
        self.current_time = 0.0
        self.simulation_history.clear()
        self.step_times.clear()
        
        # Reset pre-allocated arrays
        for name in self.layer_spikes:
            self.layer_spikes[name].fill(False)
            self.layer_currents[name].fill(0.0)
    
    def get_network_info(self) -> Dict[str, Any]:
        """Get comprehensive network information."""
        layer_info = {}
        for name, layer in self.layers.items():
            layer_info[name] = {
                "size": layer.size,
                "neuron_type": layer.neuron_type,
                "performance": layer.get_performance_stats()
            }
        
        return {
            "layers": layer_info,
            "total_neurons": self.total_neurons,
            "total_layers": len(self.layers),
            "synapse_statistics": self.synapse_manager.get_all_statistics(),
            "current_time": self.current_time,
            "memory_monitoring": self.enable_monitoring
        }
    
    def inject_input(self, layer_name: str, neuron_indices: np.ndarray, current: float):
        """
        Inject external input current into specific neurons.
        
        Args:
            layer_name: Target layer name
            neuron_indices: Indices of neurons to stimulate
            current: Input current amplitude
        """
        if layer_name not in self.layers:
            raise ValueError(f"Layer '{layer_name}' not found")
        
        layer = self.layers[layer_name]
        if np.any(neuron_indices >= layer.size):
            raise ValueError("Neuron indices exceed layer size")
        
        # Add current to pre-allocated current array
        self.layer_currents[layer_name][neuron_indices] += current
    
    def get_firing_rates(self, time_window: float = 1000.0) -> Dict[str, np.ndarray]:
        """Get firing rates for all layers."""
        return {name: layer.get_firing_rates(time_window) 
                for name, layer in self.layers.items()}
    
    def get_membrane_potentials(self) -> Dict[str, np.ndarray]:
        """Get membrane potentials for all layers."""
        return {name: layer.get_membrane_potentials() 
                for name, layer in self.layers.items()}


# Factory function for easy network creation
def create_high_performance_network(enable_monitoring: bool = True) -> HighPerformanceNeuromorphicNetwork:
    """
    Create a high-performance neuromorphic network.
    
    Args:
        enable_monitoring: Enable performance monitoring
        
    Returns:
        HighPerformanceNeuromorphicNetwork instance
    """
    return HighPerformanceNeuromorphicNetwork(enable_monitoring)
