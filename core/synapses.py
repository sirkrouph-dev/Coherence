"""
Core synapse models for the neuromorphic programming system.
Implements biologically plausible synaptic plasticity mechanisms.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum


class SynapseType(Enum):
    """Types of synapses."""
    EXCITATORY = "excitatory"
    INHIBITORY = "inhibitory"
    MODULATORY = "modulatory"


class SynapseModel:
    """Base class for all synapse models."""
    
    def __init__(self, synapse_id: int, pre_neuron_id: int, post_neuron_id: int,
                 weight: float = 1.0, synapse_type: SynapseType = SynapseType.EXCITATORY):
        """
        Initialize synapse.
        
        Args:
            synapse_id: Unique identifier for the synapse
            pre_neuron_id: ID of presynaptic neuron
            post_neuron_id: ID of postsynaptic neuron
            weight: Initial synaptic weight
            synapse_type: Type of synapse (excitatory/inhibitory/modulatory)
        """
        self.synapse_id = synapse_id
        self.pre_neuron_id = pre_neuron_id
        self.post_neuron_id = post_neuron_id
        self.weight = weight
        self.synapse_type = synapse_type
        self.weight_history = [weight]
        
    def compute_current(self, pre_spike_time: float, current_time: float) -> float:
        """
        Compute synaptic current.
        
        Args:
            pre_spike_time: Time of presynaptic spike
            current_time: Current simulation time
            
        Returns:
            Synaptic current
        """
        raise NotImplementedError
        
    def update_weight(self, delta_w: float):
        """Update synaptic weight."""
        self.weight += delta_w
        self.weight_history.append(self.weight)
        
    def reset(self):
        """Reset synapse to initial state."""
        self.weight = self.weight_history[0]
        self.weight_history = [self.weight]


class STDP_Synapse(SynapseModel):
    """
    Spike-Timing-Dependent Plasticity synapse.
    
    Implements STDP learning rule based on spike timing.
    """
    
    def __init__(self, synapse_id: int, pre_neuron_id: int, post_neuron_id: int,
                 weight: float = 1.0, synapse_type: SynapseType = SynapseType.EXCITATORY,
                 tau_stdp: float = 20.0, A_plus: float = 0.01, A_minus: float = 0.01,
                 tau_syn: float = 5.0, E_rev: float = 0.0):
        """
        Initialize STDP synapse.
        
        Args:
            synapse_id: Unique identifier for the synapse
            pre_neuron_id: ID of presynaptic neuron
            post_neuron_id: ID of postsynaptic neuron
            weight: Initial synaptic weight
            synapse_type: Type of synapse
            tau_stdp: STDP time constant (ms)
            A_plus: LTP amplitude
            A_minus: LTD amplitude
            tau_syn: Synaptic time constant (ms)
            E_rev: Reversal potential (mV)
        """
        super().__init__(synapse_id, pre_neuron_id, post_neuron_id, weight, synapse_type)
        
        # STDP parameters
        self.tau_stdp = tau_stdp
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau_syn = tau_syn
        self.E_rev = E_rev
        
        # State variables
        self.last_pre_spike = -np.inf
        self.last_post_spike = -np.inf
        self.synaptic_current = 0.0
        self.current_time = 0.0
        
    def pre_spike(self, spike_time: float):
        """
        Handle presynaptic spike.
        
        Args:
            spike_time: Time of presynaptic spike
        """
        self.current_time = spike_time
        
        # STDP: Pre-before-post strengthens synapse (LTP)
        if spike_time - self.last_post_spike < self.tau_stdp:
            delta_t = spike_time - self.last_post_spike
            delta_w = self.A_plus * np.exp(-delta_t / self.tau_stdp)
            self.update_weight(delta_w)
            
        self.last_pre_spike = spike_time
        
    def post_spike(self, spike_time: float):
        """
        Handle postsynaptic spike.
        
        Args:
            spike_time: Time of postsynaptic spike
        """
        self.current_time = spike_time
        
        # STDP: Post-before-pre weakens synapse (LTD)
        if spike_time - self.last_pre_spike < self.tau_stdp:
            delta_t = spike_time - self.last_pre_spike
            delta_w = -self.A_minus * np.exp(-delta_t / self.tau_stdp)
            self.update_weight(delta_w)
            
        self.last_post_spike = spike_time
        
    def compute_current(self, pre_spike_time: float, current_time: float) -> float:
        """Compute synaptic current using exponential decay."""
        if current_time < pre_spike_time:
            return 0.0
            
        # Exponential decay of synaptic current
        dt = current_time - pre_spike_time
        current = self.weight * np.exp(-dt / self.tau_syn)
        
        # Apply reversal potential
        if self.synapse_type == SynapseType.EXCITATORY:
            return current
        elif self.synapse_type == SynapseType.INHIBITORY:
            return -current
        else:
            return current
            
    def step(self, dt: float):
        """Advance synapse state by one time step."""
        self.current_time += dt
        self.synaptic_current *= np.exp(-dt / self.tau_syn)
        
    def reset(self):
        """Reset synapse to initial state."""
        super().reset()
        self.last_pre_spike = -np.inf
        self.last_post_spike = -np.inf
        self.synaptic_current = 0.0
        self.current_time = 0.0


class ShortTermPlasticitySynapse(SynapseModel):
    """
    Synapse with short-term plasticity (STP).
    
    Implements depression and facilitation mechanisms.
    """
    
    def __init__(self, synapse_id: int, pre_neuron_id: int, post_neuron_id: int,
                 weight: float = 1.0, synapse_type: SynapseType = SynapseType.EXCITATORY,
                 tau_dep: float = 100.0, tau_fac: float = 500.0,
                 U: float = 0.5, tau_syn: float = 5.0):
        """
        Initialize STP synapse.
        
        Args:
            synapse_id: Unique identifier for the synapse
            pre_neuron_id: ID of presynaptic neuron
            post_neuron_id: ID of postsynaptic neuron
            weight: Initial synaptic weight
            synapse_type: Type of synapse
            tau_dep: Depression time constant (ms)
            tau_fac: Facilitation time constant (ms)
            U: Utilization parameter
            tau_syn: Synaptic time constant (ms)
        """
        super().__init__(synapse_id, pre_neuron_id, post_neuron_id, weight, synapse_type)
        
        # STP parameters
        self.tau_dep = tau_dep
        self.tau_fac = tau_fac
        self.U = U
        self.tau_syn = tau_syn
        
        # State variables
        self.x = 1.0  # Available resources
        self.u = 0.0  # Utilization
        self.last_spike_time = -np.inf
        self.current_time = 0.0
        
    def pre_spike(self, spike_time: float):
        """Handle presynaptic spike with STP."""
        self.current_time = spike_time
        
        # Update STP variables
        dt = spike_time - self.last_spike_time
        if dt > 0:
            # Recovery of available resources
            self.x = 1.0 - (1.0 - self.x) * np.exp(-dt / self.tau_dep)
            # Decay of utilization
            self.u = self.u * np.exp(-dt / self.tau_fac)
            
        # Release of neurotransmitter
        self.u += self.U * (1.0 - self.u)
        self.x -= self.u * self.x
        
        self.last_spike_time = spike_time
        
    def compute_current(self, pre_spike_time: float, current_time: float) -> float:
        """Compute synaptic current with STP."""
        if current_time < pre_spike_time:
            return 0.0
            
        # Effective weight with STP
        effective_weight = self.weight * self.x * self.u
        
        # Exponential decay
        dt = current_time - pre_spike_time
        current = effective_weight * np.exp(-dt / self.tau_syn)
        
        if self.synapse_type == SynapseType.EXCITATORY:
            return current
        elif self.synapse_type == SynapseType.INHIBITORY:
            return -current
        else:
            return current
            
    def step(self, dt: float):
        """Advance synapse state by one time step."""
        self.current_time += dt
        
    def reset(self):
        """Reset synapse to initial state."""
        super().reset()
        self.x = 1.0
        self.u = 0.0
        self.last_spike_time = -np.inf
        self.current_time = 0.0


class NeuromodulatorySynapse(SynapseModel):
    """
    Synapse with neuromodulatory learning.
    
    Implements reward-modulated plasticity.
    """
    
    def __init__(self, synapse_id: int, pre_neuron_id: int, post_neuron_id: int,
                 weight: float = 1.0, synapse_type: SynapseType = SynapseType.EXCITATORY,
                 tau_syn: float = 5.0, learning_rate: float = 0.01,
                 neuromodulator_level: float = 0.0):
        """
        Initialize neuromodulatory synapse.
        
        Args:
            synapse_id: Unique identifier for the synapse
            pre_neuron_id: ID of presynaptic neuron
            post_neuron_id: ID of postsynaptic neuron
            weight: Initial synaptic weight
            synapse_type: Type of synapse
            tau_syn: Synaptic time constant (ms)
            learning_rate: Base learning rate
            neuromodulator_level: Current neuromodulator level
        """
        super().__init__(synapse_id, pre_neuron_id, post_neuron_id, weight, synapse_type)
        
        # Neuromodulatory parameters
        self.tau_syn = tau_syn
        self.learning_rate = learning_rate
        self.neuromodulator_level = neuromodulator_level
        
        # State variables
        self.last_pre_spike = -np.inf
        self.last_post_spike = -np.inf
        self.current_time = 0.0
        
    def update_neuromodulator(self, level: float):
        """Update neuromodulator level."""
        self.neuromodulator_level = np.clip(level, 0.0, 1.0)
        
    def pre_spike(self, spike_time: float):
        """Handle presynaptic spike with neuromodulation."""
        self.current_time = spike_time
        self.last_pre_spike = spike_time
        
    def post_spike(self, spike_time: float):
        """Handle postsynaptic spike with neuromodulation."""
        self.current_time = spike_time
        self.last_post_spike = spike_time
        
        # Neuromodulatory weight update
        if self.last_pre_spike > 0:
            # Reward-modulated plasticity
            delta_w = self.learning_rate * self.neuromodulator_level
            self.update_weight(delta_w)
        
    def compute_current(self, pre_spike_time: float, current_time: float) -> float:
        """Compute synaptic current."""
        if current_time < pre_spike_time:
            return 0.0
            
        # Exponential decay
        dt = current_time - pre_spike_time
        current = self.weight * np.exp(-dt / self.tau_syn)
        
        if self.synapse_type == SynapseType.EXCITATORY:
            return current
        elif self.synapse_type == SynapseType.INHIBITORY:
            return -current
        else:
            return current
            
    def step(self, dt: float):
        """Advance synapse state by one time step."""
        self.current_time += dt
        
    def reset(self):
        """Reset synapse to initial state."""
        super().reset()
        self.last_pre_spike = -np.inf
        self.last_post_spike = -np.inf
        self.current_time = 0.0


class SynapseFactory:
    """Factory for creating different types of synapses."""
    
    @staticmethod
    def create_synapse(synapse_type: str, synapse_id: int, pre_neuron_id: int, 
                      post_neuron_id: int, **kwargs) -> SynapseModel:
        """
        Create a synapse of the specified type.
        
        Args:
            synapse_type: Type of synapse to create ('stdp', 'stp', 'neuromodulatory')
            synapse_id: Unique identifier for the synapse
            pre_neuron_id: ID of presynaptic neuron
            post_neuron_id: ID of postsynaptic neuron
            **kwargs: Additional parameters for the synapse model
            
        Returns:
            Synapse instance
        """
        if synapse_type.lower() == "stdp":
            return STDP_Synapse(synapse_id, pre_neuron_id, post_neuron_id, **kwargs)
        elif synapse_type.lower() == "stp":
            return ShortTermPlasticitySynapse(synapse_id, pre_neuron_id, post_neuron_id, **kwargs)
        elif synapse_type.lower() == "neuromodulatory":
            return NeuromodulatorySynapse(synapse_id, pre_neuron_id, post_neuron_id, **kwargs)
        else:
            raise ValueError(f"Unknown synapse type: {synapse_type}")


class SynapsePopulation:
    """Collection of synapses between neuron populations."""
    
    def __init__(self, pre_population_size: int, post_population_size: int,
                 synapse_type: str = "stdp", connection_probability: float = 0.1,
                 **kwargs):
        """
        Initialize synapse population.
        
        Args:
            pre_population_size: Size of presynaptic population
            post_population_size: Size of postsynaptic population
            synapse_type: Type of synapses to create
            connection_probability: Probability of connection between neurons
            **kwargs: Parameters for synapse models
        """
        self.pre_population_size = pre_population_size
        self.post_population_size = post_population_size
        self.synapse_type = synapse_type
        self.connection_probability = connection_probability
        
        # Create synapses
        self.synapses = {}
        synapse_id = 0
        
        for pre_id in range(pre_population_size):
            for post_id in range(post_population_size):
                if np.random.random() < connection_probability:
                    synapse = SynapseFactory.create_synapse(
                        synapse_type, synapse_id, pre_id, post_id, **kwargs
                    )
                    self.synapses[(pre_id, post_id)] = synapse
                    synapse_id += 1
                    
    def get_synaptic_currents(self, pre_spikes: List[bool], current_time: float) -> List[float]:
        """
        Compute synaptic currents for all postsynaptic neurons.
        
        Args:
            pre_spikes: List of presynaptic spike indicators
            current_time: Current simulation time
            
        Returns:
            List of synaptic currents for postsynaptic neurons
        """
        currents = [0.0] * self.post_population_size
        
        for (pre_id, post_id), synapse in self.synapses.items():
            if pre_spikes[pre_id]:
                current = synapse.compute_current(current_time, current_time)
                currents[post_id] += current
                
        return currents
        
    def update_weights(self, pre_spikes: List[bool], post_spikes: List[bool], 
                      current_time: float):
        """
        Update synaptic weights based on spike timing.
        
        Args:
            pre_spikes: List of presynaptic spike indicators
            post_spikes: List of postsynaptic spike indicators
            current_time: Current simulation time
        """
        for (pre_id, post_id), synapse in self.synapses.items():
            if pre_spikes[pre_id]:
                synapse.pre_spike(current_time)
            if post_spikes[post_id]:
                synapse.post_spike(current_time)
                
    def step(self, dt: float):
        """Advance all synapses by one time step."""
        for synapse in self.synapses.values():
            synapse.step(dt)
            
    def reset(self):
        """Reset all synapses to initial state."""
        for synapse in self.synapses.values():
            synapse.reset()
            
    def get_weight_matrix(self) -> np.ndarray:
        """Get weight matrix between populations."""
        weight_matrix = np.zeros((self.pre_population_size, self.post_population_size))
        
        for (pre_id, post_id), synapse in self.synapses.items():
            weight_matrix[pre_id, post_id] = synapse.weight
            
        return weight_matrix
        
    def get_weight_history(self) -> Dict[Tuple[int, int], List[float]]:
        """Get weight history for all synapses."""
        history = {}
        for (pre_id, post_id), synapse in self.synapses.items():
            history[(pre_id, post_id)] = synapse.weight_history.copy()
        return history 