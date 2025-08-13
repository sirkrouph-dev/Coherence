"""
Vectorized synapse implementations optimized for large-scale networks.
Uses sparse matrices and vectorized operations for maximum performance.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from scipy.sparse import csr_matrix, dok_matrix
import time

from .logging_utils import neuromorphic_logger


class VectorizedSynapsePopulation:
    """
    High-performance synapse population using sparse matrices.
    Replaces object-oriented synapse collections for large networks.
    """
    
    def __init__(
        self,
        pre_size: int,
        post_size: int,
        synapse_type: str = "stdp",
        connection_probability: float = 0.1,
        **kwargs
    ):
        """
        Initialize vectorized synapse population.
        
        Args:
            pre_size: Number of presynaptic neurons
            post_size: Number of postsynaptic neurons
            synapse_type: Type of synapses
            connection_probability: Probability of connection
            **kwargs: Synapse parameters
        """
        self.pre_size = pre_size
        self.post_size = post_size
        self.synapse_type = synapse_type
        self.connection_probability = connection_probability
        
        # Initialize sparse connectivity and weights
        self._create_connectivity(**kwargs)
        self._initialize_plasticity(**kwargs)
        
        # Performance tracking
        self.step_count = 0
        self.weight_updates = 0
        
    def _create_connectivity(self, **kwargs):
        """Create sparse connectivity matrix."""
        # Use seed for reproducible connectivity
        np.random.seed(42)
        
        # Create random connectivity
        n_connections = int(self.pre_size * self.post_size * self.connection_probability)
        
        # Generate random connections
        pre_indices = np.random.randint(0, self.pre_size, n_connections)
        post_indices = np.random.randint(0, self.post_size, n_connections)
        
        # Remove duplicates
        connections = set(zip(pre_indices, post_indices))
        connections = list(connections)
        
        if len(connections) == 0:
            # Ensure at least some connections
            connections = [(0, 0)]
        
        # Unpack connections
        pre_idx, post_idx = zip(*connections)
        self.pre_indices = np.array(pre_idx, dtype=np.int32)
        self.post_indices = np.array(post_idx, dtype=np.int32)
        self.n_connections = len(connections)
        
        # Initialize weights
        initial_weight = kwargs.get('weight', 2.0)
        weight_std = kwargs.get('weight_std', 0.2)
        
        # Add weight heterogeneity
        weights = np.random.normal(initial_weight, weight_std, self.n_connections)
        weights = np.maximum(0.1, weights)  # Ensure positive weights
        
        # Create sparse weight matrix
        self.weight_matrix = csr_matrix(
            (weights, (self.post_indices, self.pre_indices)),
            shape=(self.post_size, self.pre_size),
            dtype=np.float32
        )
        
        # Track individual synapse weights for plasticity
        self.synapse_weights = weights.astype(np.float32)
        
        neuromorphic_logger.log_info(
            f"Created vectorized synapse population: {self.n_connections} connections "
            f"({self.connection_probability:.1%} density)"
        )
    
    def _initialize_plasticity(self, **kwargs):
        """Initialize plasticity-related arrays."""
        if self.synapse_type.lower() == "stdp":
            # STDP parameters
            self.tau_plus = kwargs.get('tau_plus', 20.0)
            self.tau_minus = kwargs.get('tau_minus', 20.0)
            self.A_plus = kwargs.get('A_plus', 0.02)  # Stronger for better learning
            self.A_minus = kwargs.get('A_minus', 0.02)
            self.w_min = kwargs.get('w_min', 0.0)
            self.w_max = kwargs.get('w_max', 10.0)
            
            # Synaptic parameters
            self.tau_syn = kwargs.get('tau_syn', 5.0)
            self.E_rev = kwargs.get('E_rev', 0.0)
            
            # Traces for each synapse
            self.pre_traces = np.zeros(self.n_connections, dtype=np.float32)
            self.post_traces = np.zeros(self.n_connections, dtype=np.float32)
            self.last_pre_spike = np.full(self.pre_size, -np.inf, dtype=np.float32)
            self.last_post_spike = np.full(self.post_size, -np.inf, dtype=np.float32)
            
            # Synaptic currents
            self.synaptic_conductances = np.zeros(self.n_connections, dtype=np.float32)
        
        elif self.synapse_type.lower() == "static":
            # Static synapses - no plasticity
            self.tau_syn = kwargs.get('tau_syn', 5.0)
            self.E_rev = kwargs.get('E_rev', 0.0)  # Add E_rev for static synapses
            self.synaptic_conductances = np.zeros(self.n_connections, dtype=np.float32)
        
        else:
            raise ValueError(f"Unsupported synapse type: {self.synapse_type}")
    
    def compute_synaptic_currents(
        self, 
        pre_spikes: np.ndarray, 
        current_time: float
    ) -> np.ndarray:
        """
        Compute synaptic currents - FULLY VECTORIZED.
        
        Args:
            pre_spikes: Boolean array of presynaptic spikes [pre_size]
            current_time: Current simulation time
            
        Returns:
            Synaptic currents for postsynaptic neurons [post_size]
        """
        if len(pre_spikes) != self.pre_size:
            raise ValueError(f"Expected {self.pre_size} pre_spikes, got {len(pre_spikes)}")
        
        # Update synaptic conductances - VECTORIZED
        dt = 1.0  # Assume 1ms timestep
        
        # Decay conductances
        self.synaptic_conductances *= np.exp(-dt / self.tau_syn)
        
        # Add new spikes - VECTORIZED
        spiking_synapses = pre_spikes[self.pre_indices]
        self.synaptic_conductances[spiking_synapses] += self.synapse_weights[spiking_synapses]
        
        # Compute currents using sparse matrix multiplication
        # Create temporary conductance matrix
        conductance_matrix = csr_matrix(
            (self.synaptic_conductances, (self.post_indices, self.pre_indices)),
            shape=(self.post_size, self.pre_size),
            dtype=np.float32
        )
        
        # Sum conductances for each postsynaptic neuron
        post_conductances = np.array(conductance_matrix.sum(axis=1)).flatten()
        
        # Convert to currents (simplified: assume V_post = -65mV)
        V_post = -65.0
        currents = post_conductances * (self.E_rev - V_post)
        
        return currents.astype(np.float32)
    
    def update_weights(
        self,
        pre_spikes: np.ndarray,
        post_spikes: np.ndarray,
        current_time: float
    ):
        """
        Update synaptic weights using plasticity rules - VECTORIZED.
        
        Args:
            pre_spikes: Presynaptic spikes [pre_size]
            post_spikes: Postsynaptic spikes [post_size]
            current_time: Current simulation time
        """
        if self.synapse_type.lower() != "stdp":
            return  # No plasticity for static synapses
        
        dt = 1.0  # Assume 1ms timestep
        
        # Update spike times
        self.last_pre_spike[pre_spikes] = current_time
        self.last_post_spike[post_spikes] = current_time
        
        # Decay traces - VECTORIZED
        self.pre_traces *= np.exp(-dt / self.tau_plus)
        self.post_traces *= np.exp(-dt / self.tau_minus)
        
        # Update traces for spiking neurons
        pre_trace_updates = pre_spikes[self.pre_indices]
        post_trace_updates = post_spikes[self.post_indices]
        
        self.pre_traces[pre_trace_updates] += 1.0
        self.post_traces[post_trace_updates] += 1.0
        
        # Compute weight changes - VECTORIZED
        delta_w = np.zeros(self.n_connections, dtype=np.float32)
        
        # LTD: pre-before-post
        if np.any(pre_spikes):
            ltd_synapses = pre_trace_updates
            if np.any(ltd_synapses):
                # Get corresponding post traces
                post_traces_at_synapses = self.post_traces[ltd_synapses]
                delta_w[ltd_synapses] -= self.A_minus * post_traces_at_synapses
        
        # LTP: post-before-pre  
        if np.any(post_spikes):
            ltp_synapses = post_trace_updates
            if np.any(ltp_synapses):
                # Get corresponding pre traces
                pre_traces_at_synapses = self.pre_traces[ltp_synapses]
                delta_w[ltp_synapses] += self.A_plus * pre_traces_at_synapses
        
        # Apply weight changes with bounds - VECTORIZED
        if np.any(delta_w != 0):
            self.synapse_weights += delta_w
            self.synapse_weights = np.clip(self.synapse_weights, self.w_min, self.w_max)
            
            # Update sparse matrix
            self.weight_matrix.data = self.synapse_weights
            
            self.weight_updates += np.sum(delta_w != 0)
    
    def step(self, dt: float):
        """Advance synapses by one time step."""
        self.step_count += 1
        # Additional per-step updates can go here
    
    def get_weight_matrix(self) -> csr_matrix:
        """Get the sparse weight matrix."""
        return self.weight_matrix.copy()
    
    def get_dense_weight_matrix(self) -> np.ndarray:
        """Get dense weight matrix (use carefully for large networks)."""
        return self.weight_matrix.toarray()
    
    def scale_weights(self, factor: float):
        """Scale all weights by a factor."""
        self.synapse_weights *= factor
        self.synapse_weights = np.clip(self.synapse_weights, self.w_min, self.w_max)
        self.weight_matrix.data = self.synapse_weights
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get synapse population statistics."""
        return {
            "n_connections": self.n_connections,
            "density": self.n_connections / (self.pre_size * self.post_size),
            "mean_weight": float(np.mean(self.synapse_weights)),
            "std_weight": float(np.std(self.synapse_weights)),
            "min_weight": float(np.min(self.synapse_weights)),
            "max_weight": float(np.max(self.synapse_weights)),
            "total_weight": float(np.sum(self.synapse_weights)),
            "step_count": self.step_count,
            "weight_updates": self.weight_updates,
            "synapse_type": self.synapse_type
        }
    
    def reset(self):
        """Reset synapse population to initial state."""
        # Reset conductances and traces
        if hasattr(self, 'synaptic_conductances'):
            self.synaptic_conductances.fill(0.0)
        if hasattr(self, 'pre_traces'):
            self.pre_traces.fill(0.0)
            self.post_traces.fill(0.0)
            self.last_pre_spike.fill(-np.inf)
            self.last_post_spike.fill(-np.inf)
        
        self.step_count = 0
        self.weight_updates = 0


class VectorizedSynapseManager:
    """
    Manager for multiple vectorized synapse populations.
    Handles connections between different neuron populations.
    """
    
    def __init__(self):
        """Initialize synapse manager."""
        self.synapse_populations: Dict[Tuple[str, str], VectorizedSynapsePopulation] = {}
        self.layer_sizes: Dict[str, int] = {}
    
    def add_layer(self, name: str, size: int):
        """Register a layer for synapse management."""
        self.layer_sizes[name] = size
    
    def connect_layers(
        self,
        pre_layer: str,
        post_layer: str,
        synapse_type: str = "stdp",
        connection_probability: float = 0.1,
        **kwargs
    ):
        """
        Connect two layers with vectorized synapses.
        
        Args:
            pre_layer: Name of presynaptic layer
            post_layer: Name of postsynaptic layer
            synapse_type: Type of synapses
            connection_probability: Connection probability
            **kwargs: Synapse parameters
        """
        if pre_layer not in self.layer_sizes:
            raise ValueError(f"Pre-layer '{pre_layer}' not registered")
        if post_layer not in self.layer_sizes:
            raise ValueError(f"Post-layer '{post_layer}' not registered")
        
        pre_size = self.layer_sizes[pre_layer]
        post_size = self.layer_sizes[post_layer]
        
        # Create vectorized synapse population
        synapse_pop = VectorizedSynapsePopulation(
            pre_size, post_size, synapse_type, connection_probability, **kwargs
        )
        
        connection_key = (pre_layer, post_layer)
        self.synapse_populations[connection_key] = synapse_pop
        
        neuromorphic_logger.log_info(
            f"Connected {pre_layer} ({pre_size}) -> {post_layer} ({post_size}) "
            f"with {synapse_pop.n_connections} {synapse_type} synapses"
        )
    
    def compute_layer_currents(
        self,
        layer_spikes: Dict[str, np.ndarray],
        current_time: float
    ) -> Dict[str, np.ndarray]:
        """
        Compute synaptic currents for all layers.
        
        Args:
            layer_spikes: Spike arrays for each layer
            current_time: Current simulation time
            
        Returns:
            Synaptic currents for each postsynaptic layer
        """
        layer_currents = {name: np.zeros(size, dtype=np.float32) 
                         for name, size in self.layer_sizes.items()}
        
        for (pre_layer, post_layer), synapse_pop in self.synapse_populations.items():
            if pre_layer in layer_spikes:
                pre_spikes = layer_spikes[pre_layer]
                currents = synapse_pop.compute_synaptic_currents(pre_spikes, current_time)
                layer_currents[post_layer] += currents
        
        return layer_currents
    
    def update_all_weights(
        self,
        layer_spikes: Dict[str, np.ndarray],
        current_time: float
    ):
        """Update weights for all synapse populations."""
        for (pre_layer, post_layer), synapse_pop in self.synapse_populations.items():
            if pre_layer in layer_spikes and post_layer in layer_spikes:
                synapse_pop.update_weights(
                    layer_spikes[pre_layer],
                    layer_spikes[post_layer],
                    current_time
                )
    
    def step_all(self, dt: float):
        """Step all synapse populations."""
        for synapse_pop in self.synapse_populations.values():
            synapse_pop.step(dt)
    
    def get_total_synapses(self) -> int:
        """Get total number of synapses."""
        return sum(pop.n_connections for pop in self.synapse_populations.values())
    
    def get_all_statistics(self) -> Dict[str, Any]:
        """Get statistics for all synapse populations."""
        stats = {
            "total_connections": self.get_total_synapses(),
            "n_populations": len(self.synapse_populations),
            "populations": {}
        }
        
        for (pre, post), pop in self.synapse_populations.items():
            connection_name = f"{pre}->{post}"
            stats["populations"][connection_name] = pop.get_statistics()
        
        return stats
    
    def reset_all(self):
        """Reset all synapse populations."""
        for synapse_pop in self.synapse_populations.values():
            synapse_pop.reset()
