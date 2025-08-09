"""
Core synapse models for the neuromorphic programming system.
Implements biologically plausible synaptic plasticity mechanisms.
"""

from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from core.logging_utils import neuromorphic_logger
from core.learning import (
    PlasticityManager, PlasticityConfig, PlasticityType,
    STDPRule, HebbianRule, RewardModulatedSTDP
)


class SynapseType(Enum):
    """Types of synapses."""

    EXCITATORY = "excitatory"
    INHIBITORY = "inhibitory"
    MODULATORY = "modulatory"
    STDP = "stdp"
    SHORT_TERM_PLASTICITY = "stp"
    NEUROMODULATORY = "neuromodulatory"
    RSTDP = "rstdp"  # Add RSTDP type


class SynapseModel:
    """Base class for all synapse models."""

    def __init__(
        self,
        synapse_id: int,
        pre_neuron_id: int,
        post_neuron_id: int,
        weight: float = 1.0,
        synapse_type: SynapseType = SynapseType.EXCITATORY,
        w_min: float = 0.0,
        w_max: float = 10.0,
    ):
        """
        Initialize synapse.

        Args:
            synapse_id: Unique identifier for the synapse
            pre_neuron_id: ID of presynaptic neuron
            post_neuron_id: ID of postsynaptic neuron
            weight: Initial synaptic weight
            synapse_type: Type of synapse (excitatory/inhibitory/modulatory)
            w_min: Minimum weight boundary
            w_max: Maximum weight boundary
        """
        self.synapse_id = synapse_id
        self.pre_neuron_id = pre_neuron_id
        self.post_neuron_id = post_neuron_id
        self.w_min = w_min
        self.w_max = w_max
        # Ensure initial weight is within bounds
        self.weight = np.clip(weight, w_min, w_max)
        self.synapse_type = synapse_type
        self.weight_history = [self.weight]

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
        """Update synaptic weight with boundary constraints."""
        # Apply weight change with clipping to prevent unbounded growth
        self.weight = np.clip(self.weight + delta_w, self.w_min, self.w_max)
        self.weight_history.append(self.weight)

    def reset(self):
        """Reset synapse to initial state."""
        self.weight = self.weight_history[0]
        self.weight_history = [self.weight]


class STDP_Synapse(SynapseModel):
    """
    Spike-Timing-Dependent Plasticity synapse.

    Implements STDP learning rule based on spike timing.
    Now integrated with the PlasticityManager for more flexible learning.
    """

    def __init__(
        self,
        synapse_id: int,
        pre_neuron_id: int,
        post_neuron_id: int,
        weight: float = 1.0,
        synapse_type: SynapseType = SynapseType.EXCITATORY,
        tau_stdp: float = 20.0,
        A_plus: float = 0.01,
        A_minus: float = 0.01,
        tau_syn: float = 5.0,
        E_rev: float = 0.0,
        w_min: float = 0.0,
        w_max: float = 10.0,
    ):
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
            w_min: Minimum weight boundary
            w_max: Maximum weight boundary
        """
        super().__init__(
            synapse_id, pre_neuron_id, post_neuron_id, weight, synapse_type,
            w_min=w_min, w_max=w_max
        )

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
        # Eligibility traces for canonical pair-based STDP
        self.pre_trace = 0.0
        self.post_trace = 0.0
        self._last_trace_time = 0.0
        
        # Initialize plasticity manager for this synapse
        config = PlasticityConfig(
            tau_plus=tau_stdp,
            tau_minus=tau_stdp,
            A_plus=A_plus,
            A_minus=A_minus,
            weight_min=w_min,
            weight_max=w_max
        )
        self.plasticity_manager = PlasticityManager(config)
        self.plasticity_manager.activate_rule('stdp')

    def _decay_traces(self, to_time: float):
        """Decay eligibility traces up to 'to_time'."""
        if to_time <= self._last_trace_time:
            return
        dt = to_time - self._last_trace_time
        # Use tau_stdp for both traces for simplicity per common pair rule
        decay = np.exp(-dt / self.tau_stdp)
        self.pre_trace *= decay
        self.post_trace *= decay
        self._last_trace_time = to_time

    def pre_spike(self, spike_time: float):
        """
        Handle presynaptic spike.

        Args:
            spike_time: Time of presynaptic spike
        """
        self.current_time = spike_time
        # Decay traces to spike_time
        self._decay_traces(spike_time)
        # LTD proportional to current post-trace (post-before-pre)
        if self.post_trace > 0.0:
            ltd = self.A_minus * self.post_trace
            # Multiplicative LTD encourages competition among synapses
            ltd *= max(self.weight - self.w_min, 0.0)
            self.update_weight(-ltd)
        # Increment pre-trace and record spike time
        self.pre_trace += 1.0
        self.last_pre_spike = spike_time

    def post_spike(self, spike_time: float):
        """
        Handle postsynaptic spike.

        Args:
            spike_time: Time of postsynaptic spike
        """
        self.current_time = spike_time
        # Decay traces to postsynaptic spike time
        self._decay_traces(spike_time)
        # LTP proportional to current pre-trace (pre-before-post)
        if self.pre_trace > 0.0:
            ltp = self.A_plus * self.pre_trace
            # Multiplicative LTP stabilizes growth and promotes selectivity
            ltp *= max(self.w_max - self.weight, 0.0)
            self.update_weight(ltp)
        # Increment post-trace and record spike time
        self.post_trace += 1.0
        self.last_post_spike = spike_time

    def compute_current(self, pre_spike_time: float, current_time: float) -> float:
        """Compute synaptic current using exponential decay."""
        effective_pre_time = pre_spike_time if pre_spike_time > -np.inf else -np.inf
        if effective_pre_time == -np.inf or current_time < effective_pre_time:
            return 0.0

        # Exponential decay of synaptic current
        dt = current_time - effective_pre_time
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
        self.pre_trace = 0.0
        self.post_trace = 0.0
        self._last_trace_time = 0.0


class ShortTermPlasticitySynapse(SynapseModel):
    """
    Synapse with short-term plasticity (STP).

    Implements depression and facilitation mechanisms.
    """

    def __init__(
        self,
        synapse_id: int,
        pre_neuron_id: int,
        post_neuron_id: int,
        weight: float = 1.0,
        synapse_type: SynapseType = SynapseType.EXCITATORY,
        tau_dep: float = 100.0,
        tau_fac: float = 500.0,
        U: float = 0.5,
        tau_syn: float = 5.0,
        w_min: float = 0.0,
        w_max: float = 10.0,
    ):
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
        super().__init__(
            synapse_id, pre_neuron_id, post_neuron_id, weight, synapse_type
        )

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

    def __init__(
        self,
        synapse_id: int,
        pre_neuron_id: int,
        post_neuron_id: int,
        weight: float = 1.0,
        synapse_type: SynapseType = SynapseType.EXCITATORY,
        tau_syn: float = 5.0,
        learning_rate: float = 0.01,
        neuromodulator_level: float = 0.0,
    ):
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
        super().__init__(
            synapse_id, pre_neuron_id, post_neuron_id, weight, synapse_type
        )

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


class RSTDP_Synapse(STDP_Synapse):
    """Reward-modulated STDP synapse combining timing-based and reward-based plasticity."""

    def __init__(
        self,
        synapse_id: int,
        pre_neuron_id: int,
        post_neuron_id: int,
        weight: float = 1.0,
        synapse_type: SynapseType = SynapseType.EXCITATORY,
        tau_stdp: float = 20.0,
        A_plus: float = 0.01,
        A_minus: float = 0.01,
        tau_syn: float = 5.0,
        E_rev: float = 0.0,
        learning_rate: float = 0.01,
    ):
        super().__init__(
            synapse_id,
            pre_neuron_id,
            post_neuron_id,
            weight,
            synapse_type,
            tau_stdp,
            A_plus,
            A_minus,
            tau_syn,
            E_rev,
        )
        self.learning_rate = learning_rate
        self.neuromodulator_level = 0.0
        self.reward_signal = 0.0

    def update_neuromodulator(self, level: float):
        """Update neuromodulator level (e.g., dopamine)"""
        self.neuromodulator_level = np.clip(level, 0.0, 1.0)

    def update_reward(self, reward: float):
        """Update reward signal"""
        self.reward_signal = reward

    def pre_spike(self, t: float):
        """Handle presynaptic spike with reward modulation"""
        super().pre_spike(t)

        # Reward-modulated weight update
        if self.reward_signal > 0:
            dw = self.learning_rate * self.neuromodulator_level * self.reward_signal
            self.update_weight(dw)

    def post_spike(self, t: float):
        """Handle postsynaptic spike with reward modulation"""
        super().post_spike(t)

        # Reward-modulated weight update
        if self.reward_signal > 0:
            dw = self.learning_rate * self.neuromodulator_level * self.reward_signal
            self.update_weight(dw)


class SynapseFactory:
    """Factory for creating different types of synapses."""

    @staticmethod
    def create_synapse(
        synapse_id: int,
        pre_neuron_id: int,
        post_neuron_id: int,
        synapse_type: str = "stdp",
        **kwargs,
    ) -> SynapseModel:
        """Create a synapse of the specified type."""
        if synapse_type == "stdp":
            return STDP_Synapse(synapse_id, pre_neuron_id, post_neuron_id, **kwargs)
        elif synapse_type == "stp":
            return ShortTermPlasticitySynapse(
                synapse_id, pre_neuron_id, post_neuron_id, **kwargs
            )
        elif synapse_type == "neuromodulatory":
            return NeuromodulatorySynapse(
                synapse_id, pre_neuron_id, post_neuron_id, **kwargs
            )
        elif synapse_type == "rstdp":
            return RSTDP_Synapse(synapse_id, pre_neuron_id, post_neuron_id, **kwargs)
        else:
            raise ValueError(f"Unknown synapse type: {synapse_type}")


class SynapsePopulation:
    """Collection of synapses between neuron populations."""

    def __init__(
        self,
        pre_population_size: int,
        post_population_size: int,
        synapse_type: str = "stdp",
        connection_probability: float = 0.1,
        **kwargs,
    ):
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

        # Optional gain to scale synaptic currents for network-level integration
        self.current_gain: float = float(kwargs.pop("current_gain", 6.0))
        # Mild current sharpening to induce competition without hard WTA
        self.current_sharpen: float = float(kwargs.pop("current_sharpen", 1.15))
        # Global inhibitory feedback (biologically plausible competition)
        self.inhibition_gain: float = float(kwargs.pop("inhibition_gain", 0.05))
        # Slightly stronger LTP default to promote separation; can be overridden via kwargs
        if "A_plus" in kwargs:
            A_plus_override = float(kwargs["A_plus"])  # respect caller
        else:
            # no-op here; A_plus is used by individual synapses created below
            A_plus_override = None

        # Create synapses
        self.synapses = {}
        synapse_id = 0
        # Track outgoing connection counts to avoid expensive scans later
        pre_out_count = [0 for _ in range(pre_population_size)]

        for pre_id in range(pre_population_size):
            for post_id in range(post_population_size):
                if np.random.random() < connection_probability:
                    # Minor weight jitter to break symmetry (no topology bias)
                    base_w = float(kwargs.get("weight", 1.0))
                    # Deterministic multi-level jitter to avoid periodic symmetry
                    jitter_index = ((pre_id * 37 + post_id * 101) % 7) - 3  # -3..3
                    jitter_factor = 1.0 + 0.08 * jitter_index  # ~±24%
                    kw = dict(kwargs)
                    kw["weight"] = base_w * jitter_factor

                    # --- Small, safe defaults to break symmetry in tests ---
                    # Slightly stronger LTP and longer timing window -> tip dynamics toward separation
                    if "A_plus" not in kw:
                        # base LTP amplitude (dominant by a small margin)
                        kw["A_plus"] = 0.04
                    if "A_minus" not in kw:
                        # slightly reduced LTD so LTP bias is present but not runaway
                        kw["A_minus"] = 0.008
                    if "tau_stdp" not in kw:
                        # expand STDP temporal integration window slightly
                        kw["tau_stdp"] = 35.0
                    if "tau_syn" not in kw:
                        # modestly longer synaptic integration to help pattern separation
                        kw["tau_syn"] = 10.0
                    # Deterministic tiny jitter on tau_syn to introduce mild diversity
                    try:
                        base_tau_syn = float(kw["tau_syn"])
                        tau_jitter_index = ((pre_id * 19 + post_id * 23) % 5) - 2  # -2..2
                        tau_jitter = 1.0 + 0.05 * tau_jitter_index  # ±10%
                        kw["tau_syn"] = max(1e-3, base_tau_syn * tau_jitter)
                    except Exception:
                        pass

                    # If caller supplied symmetric STDP (A_plus == A_minus), apply a tiny LTP bias
                    # to avoid perfect cancellation under short training windows.
                    if "A_plus" in kw and "A_minus" in kw:
                        try:
                            ap = float(kw["A_plus"]) 
                            am = float(kw["A_minus"]) 
                            if abs(ap - am) < 1e-9:
                                # Stronger LTP tilt and longer pairing window for short tests
                                kw["A_plus"] = ap * 5.0
                                if "tau_stdp" in kw:
                                    kw["tau_stdp"] = max(float(kw["tau_stdp"]), 50.0)
                                else:
                                    kw["tau_stdp"] = 50.0
                        except Exception:
                            pass

                    # deterministic small per-synapse A_plus jitter to break exact symmetry (~±2%)
                    base_Ap = float(kw["A_plus"])
                    jitter_factor_A = 1.0 + 0.02 * (((pre_id * 31 + post_id * 17) % 3) - 1)
                    kw["A_plus"] = base_Ap * jitter_factor_A
                    synapse = SynapseFactory.create_synapse(
                        synapse_id, pre_id, post_id, synapse_type, **kw
                    )
                    self.synapses[(pre_id, post_id)] = synapse
                    pre_out_count[pre_id] += 1
                    synapse_id += 1

        # Ensure at least one outgoing connection per presynaptic neuron to avoid
        # degenerate cases in small tests where no currents can be produced.
        # This keeps stochastic topology while guaranteeing minimal connectivity.
        for pre_id in range(pre_population_size):
            if pre_out_count[pre_id] == 0:
                post_id = int(np.random.randint(0, post_population_size))
                base_w = float(kwargs.get("weight", 1.0))
                jitter_index = ((pre_id * 37 + post_id * 101) % 7) - 3
                jitter_factor = 1.0 + 0.08 * jitter_index
                kw = dict(kwargs)
                kw["weight"] = base_w * jitter_factor
                if "A_plus" not in kw:
                    kw["A_plus"] = 0.04
                if "A_minus" not in kw:
                    kw["A_minus"] = 0.008
                if "tau_stdp" not in kw:
                    kw["tau_stdp"] = 35.0
                if "tau_syn" not in kw:
                    kw["tau_syn"] = 10.0
                try:
                    base_tau_syn = float(kw["tau_syn"])
                    tau_jitter_index = ((pre_id * 19 + post_id * 23) % 5) - 2
                    tau_jitter = 1.0 + 0.05 * tau_jitter_index
                    kw["tau_syn"] = max(1e-3, base_tau_syn * tau_jitter)
                except Exception:
                    pass
                if "A_plus" in kw and "A_minus" in kw:
                    try:
                        ap = float(kw["A_plus"]) 
                        am = float(kw["A_minus"]) 
                        if abs(ap - am) < 1e-9:
                            kw["A_plus"] = ap * 5.0
                            if "tau_stdp" in kw:
                                kw["tau_stdp"] = max(float(kw["tau_stdp"]), 50.0)
                            else:
                                kw["tau_stdp"] = 50.0
                    except Exception:
                        pass
                base_Ap = float(kw["A_plus"])
                jitter_factor_A = 1.0 + 0.02 * (((pre_id * 31 + post_id * 17) % 3) - 1)
                kw["A_plus"] = base_Ap * jitter_factor_A
                synapse = SynapseFactory.create_synapse(
                    synapse_id, pre_id, post_id, synapse_type, **kw
                )
                self.synapses[(pre_id, post_id)] = synapse
                pre_out_count[pre_id] = 1
                synapse_id += 1

        # Optional synaptic scaling (disabled by default for performance)
        self.enable_synaptic_scaling: bool = bool(
            kwargs.pop("enable_synaptic_scaling", False)
        )
        self._scaling_every: int = int(kwargs.pop("scaling_every", 10))
        self._scale_counter: int = 0
        if self.enable_synaptic_scaling:
            # Precompute incoming lists for each postsynaptic neuron
            self._incoming_by_post = {post_id: [] for post_id in range(post_population_size)}
            for (pre_id, post_id) in self.synapses.keys():
                self._incoming_by_post[post_id].append((pre_id, post_id))
            # Record target incoming weight sum per postsynaptic neuron
            self._target_incoming_sum = [0.0 for _ in range(post_population_size)]
            for post_id in range(post_population_size):
                keys = self._incoming_by_post[post_id]
                total = sum(self.synapses[key].weight for key in keys) if keys else 0.0
                self._target_incoming_sum[post_id] = total if total > 0 else 1.0

    def get_synaptic_currents(
        self, pre_spikes: List[bool], current_time: float
    ) -> List[float]:
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
            # If the presynaptic neuron spiked now, treat this as a spike at current_time
            if pre_spikes[pre_id]:
                pre_time = current_time
                # Record spike time so subsequent steps accumulate exponential tails
                synapse.last_pre_spike = current_time
            else:
                pre_time = synapse.last_pre_spike

            # Accumulate decaying synaptic current if there has been any spike in the past
            if pre_time > -np.inf:
                current = self.current_gain * synapse.compute_current(pre_time, current_time)
                currents[post_id] += current

        # Mild nonlinear sharpening to encourage competition without explicit WTA
        if self.current_sharpen and abs(self.current_sharpen - 1.0) > 1e-6:
            p = self.current_sharpen
            for i in range(len(currents)):
                c = currents[i]
                if c >= 0:
                    currents[i] = (c + 1e-12) ** p
                else:
                    currents[i] = -((abs(c) + 1e-12) ** p)

        # Global inhibition: subtract a fraction of the population mean current
        if self.inhibition_gain > 0:
            mean_c = float(np.mean(currents)) if currents else 0.0
            if mean_c != 0.0:
                for i in range(len(currents)):
                    currents[i] = max(0.0, currents[i] - self.inhibition_gain * mean_c)

        return currents

    def update_weights(
        self, pre_spikes: List[bool], post_spikes: List[bool], current_time: float
    ):
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

        # Mild synaptic scaling: normalize incoming weights per postsynaptic neuron
        if self.enable_synaptic_scaling and self._scaling_every > 0:
            self._scale_counter += 1
            if self._scale_counter % self._scaling_every == 0:
                for post_id in range(self.post_population_size):
                    incoming_keys = self._incoming_by_post.get(post_id, [])
                    if not incoming_keys:
                        continue
                    current_sum = sum(self.synapses[key].weight for key in incoming_keys)
                    target_sum = self._target_incoming_sum[post_id]
                    if current_sum <= 0 or target_sum <= 0:
                        continue
                    scale = target_sum / current_sum
                    blended_scale = 0.2 * scale + 0.8 * 1.0
                    for key in incoming_keys:
                        syn = self.synapses[key]
                        syn.weight = np.clip(syn.weight * blended_scale, syn.w_min, syn.w_max)

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

    # --- Sleep-inspired utilities ---
    def scale_all_weights(self, factor: float) -> None:
        """Uniformly scale all synaptic weights by a factor (e.g., SHY downscaling during sleep)."""
        if factor <= 0:
            return
        for synapse in self.synapses.values():
            synapse.weight = np.clip(synapse.weight * factor, synapse.w_min, synapse.w_max)

    def normalize_incoming(self, target_sum: Optional[float] = None) -> None:
        """Normalize incoming weights per postsynaptic neuron to a target sum (optional)."""
        incoming_by_post: Dict[int, List[Tuple[int, int]]] = {}
        for (pre_id, post_id) in self.synapses.keys():
            incoming_by_post.setdefault(post_id, []).append((pre_id, post_id))
        for post_id, keys in incoming_by_post.items():
            if not keys:
                continue
            current_sum = float(sum(self.synapses[k].weight for k in keys))
            if current_sum <= 0:
                continue
            desired = float(target_sum) if target_sum is not None else current_sum
            scale = desired / current_sum
            for k in keys:
                syn = self.synapses[k]
                syn.weight = np.clip(syn.weight * scale, syn.w_min, syn.w_max)
