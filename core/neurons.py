"""
Core neuron models for the neuromorphic programming system.
Implements biologically plausible neuron models with temporal dynamics.
"""

import heapq
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.logging_utils import neuromorphic_logger


class NeuronModel:
    """Base class for all neuron models."""

    def __init__(self, neuron_id: int):
        self.neuron_id = neuron_id
        self.spike_times: List[float] = []
        self.membrane_potential: float = -65.0  # mV
        self.is_spiking: bool = False
    
    @property
    def v(self) -> float:
        """Membrane potential alias for compatibility."""
        return self.membrane_potential
    
    @v.setter
    def v(self, value: float):
        """Set membrane potential."""
        self.membrane_potential = value

    def step(self, dt: float, I_syn: float) -> bool:
        """Advance neuron state by one time step.

        Args:
            dt: Time step in milliseconds
            I_syn: Synaptic current in nA

        Returns:
            True if neuron spiked, False otherwise
        """
        raise NotImplementedError

    def reset(self):
        """Reset neuron to initial state."""
        self.spike_times.clear()
        self.is_spiking = False

    def get_spike_times(self) -> List[float]:
        """Get list of spike times for this neuron."""
        return self.spike_times.copy()


class AdaptiveExponentialIntegrateAndFire(NeuronModel):
    """
    Adaptive Exponential Integrate-and-Fire neuron model.

    Combines biological realism with computational efficiency.
    Based on Brette & Gerstner (2005).
    """

    def __init__(
        self,
        neuron_id: int,
        tau_m: float = 20.0,
        v_rest: float = -65.0,
        v_thresh: float = -55.0,
        delta_t: float = 2.0,
        tau_w: float = 144.0,
        a: float = 4.0,
        b: float = 0.0805,
        v_reset: float = -65.0,
        refractory_period: float = 2.0,
    ):
        """
        Initialize AdEx neuron.

        Args:
            neuron_id: Unique identifier for the neuron
            tau_m: Membrane time constant (ms)
            v_rest: Resting potential (mV)
            v_thresh: Threshold potential (mV)
            delta_t: Slope factor (mV)
            tau_w: Adaptation time constant (ms)
            a: Subthreshold adaptation (nS)
            b: Spike-triggered adaptation (nA)
            v_reset: Reset potential after spike (mV)
            refractory_period: Refractory period (ms)
        """
        super().__init__(neuron_id)

        # Model parameters
        self.tau_m = tau_m
        self.v_rest = v_rest
        self.v_thresh = v_thresh
        self.delta_t = delta_t
        self.tau_w = tau_w
        self.a = a
        self.b = b
        self.v_reset = v_reset
        self.refractory_period = refractory_period

        # State variables
        self.membrane_potential = v_rest
        self.adaptation_current = 0.0
        self.refractory_time = 0.0
        self.current_time = 0.0
    
    @property
    def w(self) -> float:
        """Adaptation current alias for compatibility."""
        return self.adaptation_current
    
    @w.setter
    def w(self, value: float):
        """Set adaptation current."""
        self.adaptation_current = value

    def step(self, dt: float, I_syn: float) -> bool:
        """Advance AdEx neuron by one time step."""
        self.current_time += dt
        # Clear spike flag at the start of the step
        self.is_spiking = False

        # Handle refractory period
        if self.refractory_time > 0:
            self.refractory_time -= dt
            return False

        # Update membrane potential
        dv_dt = (
            -(self.membrane_potential - self.v_rest)
            + self.delta_t
            * np.exp((self.membrane_potential - self.v_thresh) / self.delta_t)
            - self.adaptation_current
            + I_syn
        ) / self.tau_m
        self.membrane_potential += dv_dt * dt

        # Update adaptation current
        dw_dt = (
            self.a * (self.membrane_potential - self.v_rest) - self.adaptation_current
        ) / self.tau_w
        self.adaptation_current += dw_dt * dt

        # Check for spike
        if self.membrane_potential >= self.v_thresh:
            self._spike()
            # Log spike activity (only for first few neurons to avoid spam)
            if self.neuron_id < 5:  # Only log spikes for first 5 neurons
                neuromorphic_logger.log_neuron_activity(
                    neuron_id=self.neuron_id,
                    layer_name="adex_neuron",
                    membrane_potential=self.membrane_potential,
                    spiked=True,
                    adaptation_current=self.adaptation_current,
                )
            return True

        # Log regular neuron activity (only occasionally and for first few neurons)
        if (
            self.neuron_id < 3 and self.current_time % 10.0 < 0.1
        ):  # Log every 10ms for first 3 neurons
            neuromorphic_logger.log_neuron_activity(
                neuron_id=self.neuron_id,
                layer_name="adex_neuron",
                membrane_potential=self.membrane_potential,
                spiked=False,
                adaptation_current=self.adaptation_current,
            )

        return False

    def _spike(self):
        """Handle spike generation."""
        self.spike_times.append(self.current_time)
        self.membrane_potential = self.v_reset
        self.adaptation_current += self.b
        self.refractory_time = self.refractory_period
        self.is_spiking = True

    def reset(self):
        """Reset neuron to initial state."""
        super().reset()
        self.membrane_potential = self.v_rest
        self.adaptation_current = 0.0
        self.refractory_time = 0.0
        self.current_time = 0.0


class HodgkinHuxleyNeuron(NeuronModel):
    """
    Hodgkin-Huxley neuron model.

    Full biological model with sodium and potassium channels.
    Based on Hodgkin & Huxley (1952).
    """

    def __init__(
        self,
        neuron_id: int,
        C_m: float = 1.0,  # Membrane capacitance (μF/cm²)
        g_Na: float = 120.0,  # Sodium conductance (mS/cm²)
        g_K: float = 36.0,  # Potassium conductance (mS/cm²)
        g_L: float = 0.3,  # Leak conductance (mS/cm²)
        E_Na: float = 55.0,  # Sodium reversal potential (mV)
        E_K: float = -77.0,  # Potassium reversal potential (mV)
        E_L: float = -54.4,
    ):  # Leak reversal potential (mV)
        """
        Initialize Hodgkin-Huxley neuron.

        Args:
            neuron_id: Unique identifier for the neuron
            C_m: Membrane capacitance
            g_Na: Sodium conductance
            g_K: Potassium conductance
            g_L: Leak conductance
            E_Na: Sodium reversal potential
            E_K: Potassium reversal potential
            E_L: Leak reversal potential
        """
        super().__init__(neuron_id)

        # Model parameters
        self.C_m = C_m
        self.g_Na = g_Na
        self.g_K = g_K
        self.g_L = g_L
        self.E_Na = E_Na
        self.E_K = E_K
        self.E_L = E_L

        # State variables
        self.membrane_potential = -65.0
        self.m = 0.0  # Sodium activation
        self.h = 1.0  # Sodium inactivation
        self.n = 0.0  # Potassium activation
        self.current_time = 0.0

    def step(self, dt: float, I_syn: float) -> bool:
        """Advance Hodgkin-Huxley neuron by one time step."""
        self.current_time += dt

        # Calculate channel conductances
        g_Na_current = self.g_Na * (self.m**3) * self.h
        g_K_current = self.g_K * (self.n**4)
        g_L_current = self.g_L

        # Calculate currents
        I_Na = g_Na_current * (self.membrane_potential - self.E_Na)
        I_K = g_K_current * (self.membrane_potential - self.E_K)
        I_L = g_L_current * (self.membrane_potential - self.E_L)

        # Total membrane current
        I_total = I_Na + I_K + I_L + I_syn

        # Update membrane potential
        dv_dt = -I_total / self.C_m
        self.membrane_potential += dv_dt * dt

        # Update gating variables
        self._update_gating_variables(dt)

        # Check for spike (simplified threshold)
        if self.membrane_potential > 0:
            self._spike()
            return True

        return False

    def _update_gating_variables(self, dt: float):
        """Update Hodgkin-Huxley gating variables."""
        v = self.membrane_potential

        # Alpha and beta functions
        alpha_m = 0.1 * (v + 40) / (1 - np.exp(-(v + 40) / 10))
        beta_m = 4 * np.exp(-(v + 65) / 18)

        alpha_h = 0.07 * np.exp(-(v + 65) / 20)
        beta_h = 1 / (1 + np.exp(-(v + 35) / 10))

        alpha_n = 0.01 * (v + 55) / (1 - np.exp(-(v + 55) / 10))
        beta_n = 0.125 * np.exp(-(v + 65) / 80)

        # Update gating variables
        dm_dt = alpha_m * (1 - self.m) - beta_m * self.m
        dh_dt = alpha_h * (1 - self.h) - beta_h * self.h
        dn_dt = alpha_n * (1 - self.n) - beta_n * self.n

        self.m += dm_dt * dt
        self.h += dh_dt * dt
        self.n += dn_dt * dt

    def _spike(self):
        """Handle spike generation."""
        self.spike_times.append(self.current_time)
        self.is_spiking = True

    def reset(self):
        """Reset neuron to initial state."""
        super().reset()
        self.membrane_potential = -65.0
        self.m = 0.0
        self.h = 1.0
        self.n = 0.0
        self.current_time = 0.0


class LeakyIntegrateAndFire(NeuronModel):
    """
    Leaky Integrate-and-Fire neuron model.

    Simplified model for computational efficiency.
    """

    def __init__(
        self,
        neuron_id: int,
        tau_m: float = 20.0,
        v_rest: float = -65.0,
        v_thresh: float = -55.0,
        v_reset: float = -65.0,
        refractory_period: float = 2.0,
    ):
        """
        Initialize LIF neuron.

        Args:
            neuron_id: Unique identifier for the neuron
            tau_m: Membrane time constant (ms)
            v_rest: Resting potential (mV)
            v_thresh: Threshold potential (mV)
            v_reset: Reset potential (mV)
            refractory_period: Refractory period (ms)
        """
        super().__init__(neuron_id)

        # Model parameters
        self.tau_m = tau_m
        self.v_rest = v_rest
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.refractory_period = refractory_period

        # State variables
        self.membrane_potential = v_rest
        self.refractory_time = 0.0
        self.current_time = 0.0

    def step(self, dt: float, I_syn: float) -> bool:
        """Advance LIF neuron by one time step."""
        self.current_time += dt
        # Clear spike flag at the start of the step
        self.is_spiking = False

        # Handle refractory period
        if self.refractory_time > 0:
            self.refractory_time -= dt
            return False

        # Update membrane potential
        dv_dt = (-(self.membrane_potential - self.v_rest) + I_syn) / self.tau_m
        self.membrane_potential += dv_dt * dt

        # Check for spike
        if self.membrane_potential >= self.v_thresh:
            self._spike()
            return True

        return False

    def _spike(self):
        """Handle spike generation."""
        self.spike_times.append(self.current_time)
        self.membrane_potential = self.v_reset
        self.refractory_time = self.refractory_period
        self.is_spiking = True

    def reset(self):
        """Reset neuron to initial state."""
        super().reset()
        self.membrane_potential = self.v_rest
        self.refractory_time = 0.0
        self.current_time = 0.0


class NeuronFactory:
    """Factory for creating different types of neurons."""

    @staticmethod
    def create_neuron(neuron_type: str, neuron_id: int, **kwargs) -> NeuronModel:
        """
        Create a neuron of the specified type.

        Args:
            neuron_type: Type of neuron to create ('adex', 'hh', 'lif')
            neuron_id: Unique identifier for the neuron
            **kwargs: Additional parameters for the neuron model

        Returns:
            Neuron instance
        """
        if neuron_type.lower() == "adex":
            return AdaptiveExponentialIntegrateAndFire(neuron_id, **kwargs)
        elif neuron_type.lower() in ["hh", "hodgkin_huxley"]:
            return HodgkinHuxleyNeuron(neuron_id, **kwargs)
        elif neuron_type.lower() == "lif":
            return LeakyIntegrateAndFire(neuron_id, **kwargs)
        else:
            raise ValueError(f"Unknown neuron type: {neuron_type}")


class NeuronPopulation:
    """Collection of neurons of the same type."""

    def __init__(self, size: int, neuron_type: str = "adex", **kwargs):
        """
        Initialize neuron population.

        Args:
            size: Number of neurons in the population
            neuron_type: Type of neurons to create
            **kwargs: Parameters for neuron models
        """
        self.size = size
        self.neuron_type = neuron_type
        self.neurons = []

        # Create neurons
        for i in range(size):
            per_neuron_kwargs = dict(kwargs)
            # Introduce slight parameter heterogeneity to break symmetry in populations
            if neuron_type.lower() == "lif" and "v_thresh" not in per_neuron_kwargs:
                # Restore modest deterministic heterogeneity without lowering thresholds
                jitter = (-0.5 + (i % 5) * 0.25)  # values: -0.5, -0.25, 0.0, 0.25, 0.5
                per_neuron_kwargs["v_thresh"] = -55.0 + jitter
            if neuron_type.lower() == "lif" and "tau_m" not in per_neuron_kwargs:
                # Slightly faster membrane to support spiking in short windows (temporal integration)
                tau_base = 12.0
                tau_jitter = 1.0 + 0.1 * (((i * 13) % 3) - 1)  # ~±10%
                per_neuron_kwargs["tau_m"] = max(1.0, tau_base * tau_jitter)
            neuron = NeuronFactory.create_neuron(neuron_type, i, **per_neuron_kwargs)
            self.neurons.append(neuron)

    def step(self, dt: float, I_syn: List[float]) -> List[bool]:
        """
        Advance all neurons by one time step.

        Args:
            dt: Time step in milliseconds
            I_syn: List of synaptic currents for each neuron

        Returns:
            List of boolean values indicating which neurons spiked
        """
        if len(I_syn) != self.size:
            raise ValueError(f"Expected {self.size} synaptic currents, got {len(I_syn)}")
        
        spikes = []
        for neuron, I in zip(self.neurons, I_syn):
            spiked = neuron.step(dt, I)
            spikes.append(spiked)
        return spikes

    def reset(self):
        """Reset all neurons to initial state."""
        for neuron in self.neurons:
            neuron.reset()

    def get_spike_times(self) -> List[List[float]]:
        """Get spike times for all neurons."""
        return [neuron.get_spike_times() for neuron in self.neurons]

    def get_membrane_potentials(self) -> List[float]:
        """Get current membrane potentials for all neurons."""
        return [neuron.membrane_potential for neuron in self.neurons]
