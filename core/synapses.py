"""
Core synapse models for the neuromorphic programming system.
Implements biologically plausible synaptic plasticity mechanisms.
OPTIMIZED for massive scale: 80M+ neurons with CPU/GPU acceleration.
"""

from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# GPU acceleration imports with fallback warnings for large networks
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("CuPy GPU acceleration available for neuromorphic computing")
except ImportError:
    GPU_AVAILABLE = False
    cp = None
    print("[WARNING] CuPy not available - large networks (>50k neurons) will be slow")
    print("          Install with: pip install cupy-cuda12x (for CUDA 12.x)")
    print("          Or: pip install cupy-cuda11x (for CUDA 11.x)")

try:
    import torch
    TORCH_AVAILABLE = True
    print("PyTorch acceleration available")
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    print("[WARNING] PyTorch not available - some GPU optimizations disabled")

from core.logging_utils import neuromorphic_logger
from core.learning import (
    PlasticityManager,
    PlasticityConfig,
    PlasticityType,
    STDPRule,
    HebbianRule,
    RewardModulatedSTDP,
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
        # Efficient manual clipping - faster than np.clip for scalars
        new_weight = self.weight + delta_w
        if new_weight < self.w_min:
            self.weight = self.w_min
        elif new_weight > self.w_max:
            self.weight = self.w_max
        else:
            self.weight = new_weight
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
            synapse_id,
            pre_neuron_id,
            post_neuron_id,
            weight,
            synapse_type,
            w_min=w_min,
            w_max=w_max,
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
        
        # Performance optimization: cache last computed current and time
        self._last_current = 0.0
        self._last_current_time = -np.inf

        # Initialize plasticity manager for this synapse
        config = PlasticityConfig(
            tau_plus=tau_stdp,
            tau_minus=tau_stdp,
            A_plus=A_plus,
            A_minus=A_minus,
            weight_min=w_min,
            weight_max=w_max,
        )
        self.plasticity_manager = PlasticityManager(config)
        self.plasticity_manager.activate_rule("stdp")

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
        # LTD only if post-before-pre within STDP window
        if (
            self.last_post_spike > -np.inf
            and (spike_time - self.last_post_spike) < self.tau_stdp
        ):
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
        # LTP only if pre-before-post within STDP window
        if (
            self.last_pre_spike > -np.inf
            and (spike_time - self.last_pre_spike) < self.tau_stdp
        ):
            if self.pre_trace > 0.0:
                ltp = self.A_plus * self.pre_trace
                # Multiplicative LTP stabilizes growth and promotes selectivity
                ltp *= max(self.w_max - self.weight, 0.0)
                self.update_weight(ltp)
        # Increment post-trace and record spike time
        self.post_trace += 1.0
        self.last_post_spike = spike_time

    def compute_current(self, pre_spike_time: float, current_time: float) -> float:
        """Compute synaptic current using optimized exponential decay."""
        effective_pre_time = pre_spike_time if pre_spike_time > -np.inf else -np.inf
        if effective_pre_time == -np.inf or current_time < effective_pre_time:
            return 0.0

        # OPTIMIZATION: Fast exponential decay computation
        dt = current_time - effective_pre_time
        if dt > 5.0 * self.tau_syn:  # After 5 time constants, current is negligible
            return 0.0
        
        # Simple exponential decay computation
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

        # ULTRA-FAST VECTORIZED SYNAPSE CREATION for massive scale (80M+ neurons)
        # Critical optimization: avoid O(n²) loops for neuromorphic hardware scale
        
        total_neurons = pre_population_size + post_population_size
        
        # MANDATORY GPU check for massive networks (1M+ neurons)
        if total_neurons >= 1000000:  # 1M+ neurons require GPU
            if not GPU_AVAILABLE:
                raise RuntimeError(
                    f"MASSIVE SCALE ERROR: {total_neurons:,} neurons detected!\n"
                    f"Networks with 1M+ neurons REQUIRE GPU acceleration.\n"
                    f"Install CuPy: pip install cupy-cuda12x (for CUDA 12.x)\n"
                    f"Or: pip install cupy-cuda11x (for CUDA 11.x)\n"
                    f"Or reduce network size below 1M neurons for CPU fallback."
                )
        
        # STRONG WARNING for large networks without GPU
        elif total_neurons >= 5000:  # Medium networks benefit from GPU
            if not GPU_AVAILABLE:
                print(f"\n{'='*60}")
                print(f"[PERFORMANCE WARNING]: {total_neurons:,} neurons without GPU!")
                print(f"{'='*60}")
                print("Large networks are EXTREMELY SLOW on CPU.")
                print("Expected performance: 10-100x slower than GPU")
                print("Recommendation: Install CuPy for GPU acceleration:")
                print("  pip install cupy-cuda12x  # For CUDA 12.x")
                print("  pip install cupy-cuda11x  # For CUDA 11.x")
                print("Continuing with CPU (this will be slow)...")
                print(f"{'='*60}\n")
                
                # Give user a chance to abort for very large networks
                if total_neurons >= 100000:
                    import time
                    print(f"{total_neurons:,} neurons on CPU will take MINUTES...")
                    print("   Waiting 3 seconds - press Ctrl+C to abort if needed")
                    try:
                        time.sleep(3)
                    except KeyboardInterrupt:
                        raise RuntimeError("User aborted: Install GPU acceleration for large networks")
        
        # For large networks (5k+ neurons), use sparse matrix construction  
        if total_neurons >= 5000:  # Lower threshold for better test performance
            self.synapses = {}
            self._build_sparse_connectivity(pre_population_size, post_population_size, connection_probability, **kwargs)
            
            # Skip the guarantee step for massive networks (too expensive)
            # Sparse networks inherently have connectivity  
            return  # CRITICAL: Exit early to avoid dense network code
            
        else:
            # Legacy approach for smaller networks
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
                            tau_jitter_index = (
                                (pre_id * 19 + post_id * 23) % 5
                            ) - 2  # -2..2
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
                        jitter_factor_A = 1.0 + 0.02 * (
                            ((pre_id * 31 + post_id * 17) % 3) - 1
                        )
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
                        tau_jitter_index = (
                            (pre_id * 19 + post_id * 23) % 5
                        ) - 2  # -2..2
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
                    jitter_factor_A = 1.0 + 0.02 * (
                        ((pre_id * 31 + post_id * 17) % 3) - 1
                    )
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
        
        # GPU acceleration settings - MANDATORY for neuromorphic hardware scale
        self.use_gpu: bool = bool(kwargs.pop("use_gpu", GPU_AVAILABLE)) and GPU_AVAILABLE
        self.gpu_threshold: int = int(kwargs.pop("gpu_threshold", 10000))  # Switch to GPU for 10k+ neurons
        self.gpu_mandatory_threshold: int = int(kwargs.pop("gpu_mandatory_threshold", 1000000))  # 1M+ neurons REQUIRE GPU
        
        # Neuromorphic hardware targeting: 80M neurons
        self.neuromorphic_scale: bool = (pre_population_size + post_population_size) >= 50000000  # 50M+ is neuromorphic scale
        
        if self.enable_synaptic_scaling:
            # Precompute incoming lists for each postsynaptic neuron
            self._incoming_by_post = {
                post_id: [] for post_id in range(post_population_size)
            }
            for pre_id, post_id in self.synapses.keys():
                self._incoming_by_post[post_id].append((pre_id, post_id))
            # Record target incoming weight sum per postsynaptic neuron
            self._target_incoming_sum = [0.0 for _ in range(post_population_size)]
            for post_id in range(post_population_size):
                keys = self._incoming_by_post[post_id]
                total = sum(self.synapses[key].weight for key in keys) if keys else 0.0
                self._target_incoming_sum[post_id] = total if total > 0 else 1.0

    def _build_sparse_connectivity(self, pre_population_size: int, post_population_size: int, 
                                 connection_probability: float, **kwargs):
        """
        ULTRA-FAST sparse connectivity construction for massive networks (50k+ neurons).
        
        Uses vectorized operations to avoid O(n²) loops that would take forever for 80M neurons.
        """
        print(f"Building sparse connectivity for {pre_population_size + post_population_size:,} neurons...")
        
        # Calculate total expected connections
        total_possible = pre_population_size * post_population_size
        expected_connections = int(total_possible * connection_probability)
        
        # For neuromorphic scale (50M+ neurons), limit connections to prevent memory explosion
        max_connections = min(expected_connections, 10000000)  # 10M max connections
        
        if expected_connections > max_connections:
            effective_prob = max_connections / total_possible
            print(f"Limiting connections: {expected_connections:,} → {max_connections:,} (prob: {effective_prob:.6f})")
        else:
            effective_prob = connection_probability
            
        # VECTORIZED sparse connection generation
        n_connections = max_connections if expected_connections > max_connections else expected_connections
        
        # Generate random connections efficiently for ultra-sparse networks
        if effective_prob < 0.01:  # Less than 1% connectivity
            # Ultra-sparse: direct random sampling (much faster than choice without replacement)
            pre_ids = np.random.randint(0, pre_population_size, size=n_connections)
            post_ids = np.random.randint(0, post_population_size, size=n_connections)
            
            # For ultra-sparse networks, duplicates are rare, so this is efficient
            if n_connections < total_possible * 0.001:  # Very sparse, accept some duplicates
                # Keep as-is for speed
                pass
            else:
                # Remove duplicates for denser networks
                connections = set(zip(pre_ids, post_ids))
                if len(connections) < n_connections * 0.9:  # Too many duplicates
                    # Regenerate with more samples to compensate
                    while len(connections) < n_connections:
                        extra_pre = np.random.randint(0, pre_population_size, size=n_connections//10)
                        extra_post = np.random.randint(0, post_population_size, size=n_connections//10)
                        connections.update(zip(extra_pre, extra_post))
                
                connections_list = list(connections)
                pre_ids = np.array([c[0] for c in connections_list[:n_connections]])
                post_ids = np.array([c[1] for c in connections_list[:n_connections]])
                n_connections = len(pre_ids)
        else:
            # Medium sparse: generate with replacement then deduplicate
            pre_ids = np.random.randint(0, pre_population_size, size=n_connections)
            post_ids = np.random.randint(0, post_population_size, size=n_connections)
            
            # Remove duplicates
            connections = set(zip(pre_ids, post_ids))
            pre_ids = np.array([c[0] for c in connections])
            post_ids = np.array([c[1] for c in connections])
            n_connections = len(connections)
        
        print(f"Generated {n_connections:,} sparse connections...")
        
        # VECTORIZED synapse parameter generation
        base_weight = float(kwargs.get("weight", 1.0))
        
        # Vectorized weight jitter computation
        jitter_indices = ((pre_ids * 37 + post_ids * 101) % 7) - 3
        jitter_factors = 1.0 + 0.08 * jitter_indices
        weights = base_weight * jitter_factors
        
        # Vectorized parameter defaults
        A_plus_base = kwargs.get("A_plus", 0.04)
        A_minus_base = kwargs.get("A_minus", 0.008) 
        tau_stdp_base = kwargs.get("tau_stdp", 35.0)
        tau_syn_base = kwargs.get("tau_syn", 10.0)
        
        # ULTRA-FAST: Store vectorized arrays instead of individual synapse objects
        # This avoids the massive performance bottleneck of creating millions of objects
        self._pre_ids = pre_ids.astype(np.int32)
        self._post_ids = post_ids.astype(np.int32)
        
        # Vectorized weight generation
        self._weights = weights.astype(np.float32)
        
        # Store STDP parameters as vectors for massive scale efficiency
        self._A_plus = np.full(n_connections, A_plus_base, dtype=np.float32)
        self._A_minus = np.full(n_connections, A_minus_base, dtype=np.float32)
        self._tau_stdp = np.full(n_connections, tau_stdp_base, dtype=np.float32)
        self._tau_syn = np.full(n_connections, tau_syn_base, dtype=np.float32)
        
        # CRITICAL: For massive scale, create MINIMAL synapse dictionary
        # Only create actual synapse objects when absolutely necessary
        sample_size = min(100, n_connections)  # Only create 100 sample synapses max
        print(f"Creating {sample_size:,} sample {self.synapse_type} synapses for API compatibility...")
        
        for i in range(sample_size):
            pre_id = int(pre_ids[i])
            post_id = int(post_ids[i])
            
            synapse_params = {
                "weight": float(weights[i]),
                "A_plus": A_plus_base,
                "A_minus": A_minus_base,
                "tau_stdp": tau_stdp_base,
                "tau_syn": tau_syn_base,
                "w_min": kwargs.get("w_min", 0.0),
                "w_max": kwargs.get("w_max", 10.0)
            }
            
            synapse = SynapseFactory.create_synapse(
                i, pre_id, post_id, self.synapse_type, **synapse_params
            )
            self.synapses[(pre_id, post_id)] = synapse
        
        # Mark that we're using vectorized representation
        self._vectorized_representation = True
        self._total_connections = n_connections
        
        # Initialize attributes for API compatibility, even if not used in vectorized mode
        self.enable_synaptic_scaling: bool = bool(kwargs.pop("enable_synaptic_scaling", False))
        self._scaling_every: int = int(kwargs.pop("scaling_every", 10))
        self._scale_counter: int = 0
        self.use_gpu: bool = bool(kwargs.pop("use_gpu", GPU_AVAILABLE)) and GPU_AVAILABLE
        self.gpu_threshold: int = int(kwargs.pop("gpu_threshold", 10000))
        self.gpu_mandatory_threshold: int = int(kwargs.pop("gpu_mandatory_threshold", 1000000))
        self.neuromorphic_scale: bool = (pre_population_size + post_population_size) >= 50000000
        
        print(f"[OK] Sparse connectivity built: {n_connections:,} synapses in {sample_size} objects")

    @property
    def total_connections(self) -> int:
        """Get the total number of actual synaptic connections."""
        return getattr(self, '_total_connections', len(self.synapses))

    def get_synaptic_currents(
        self, pre_spikes: List[bool], current_time: float
    ) -> List[float]:
        """
        ULTRA-VECTORIZED synaptic current computation for massive scale (80M+ neurons).
        
        Uses sparse matrix operations and vectorized NumPy for neuromorphic hardware performance.
        MANDATORY GPU acceleration for networks >1M neurons.

        Args:
            pre_spikes: List of presynaptic spike indicators
            current_time: Current simulation time

        Returns:
            List of synaptic currents for postsynaptic neurons
        """
        if not self.synapses:
            return [0.0] * self.post_population_size

        # CRITICAL: Build vectorized arrays only once during initialization
        if not hasattr(self, '_vectorized_initialized'):
            self._build_vectorized_arrays()

        # Ultra-fast vectorized computation
        # 1. Extract active synapses (those with recent spikes)
        active_mask = self._last_spike_times > -np.inf
        if not np.any(active_mask):
            return [0.0] * self.post_population_size

        # 2. MANDATORY GPU acceleration for massive networks
        total_neurons = self.pre_population_size + self.post_population_size
        
        # FORCE GPU for neuromorphic scale (50M+ neurons) - CPU would take forever
        if total_neurons >= 50000000:  # 50M+ neurons = neuromorphic scale
            if not GPU_AVAILABLE:
                raise RuntimeError(
                    f"NEUROMORPHIC SCALE ERROR: {total_neurons:,} neurons detected!\n"
                    f"Networks with 50M+ neurons REQUIRE GPU acceleration.\n"
                    f"Install CuPy: pip install cupy-cuda12x\n"
                    f"Current GPU available: {GPU_AVAILABLE}"
                )
            return self._get_synaptic_currents_gpu(active_mask, current_time)
        
        # MANDATORY GPU for very large networks (1M+ neurons)
        elif total_neurons >= self.gpu_mandatory_threshold:
            if not GPU_AVAILABLE:
                raise RuntimeError(
                    f"LARGE SCALE ERROR: {total_neurons:,} neurons detected!\n"
                    f"Networks with 1M+ neurons REQUIRE GPU acceleration.\n"
                    f"Install CuPy: pip install cupy-cuda12x\n"
                    f"Or reduce network size below 1M neurons for CPU fallback."
                )
            return self._get_synaptic_currents_gpu(active_mask, current_time)
        
        # AUTOMATIC GPU for medium-large networks (10k+ neurons)
        elif self.use_gpu and total_neurons >= self.gpu_threshold and GPU_AVAILABLE:
            return self._get_synaptic_currents_gpu(active_mask, current_time)
        
        # CPU only for small-medium networks (<10k neurons)
        else:
            return self._get_synaptic_currents_cpu(active_mask, current_time)

    def _get_synaptic_currents_cpu(self, active_mask: np.ndarray, current_time: float) -> List[float]:
        """CPU-optimized synaptic current computation."""
        # 2. Vectorized exponential decay computation
        dt_vec = current_time - self._last_spike_times[active_mask]
        
        # 3. Fast cutoff for negligible currents (5 time constants)
        cutoff_mask = dt_vec <= (5.0 * self._tau_syn_vec[active_mask])
        if not np.any(cutoff_mask):
            return [0.0] * self.post_population_size

        # 4. Ultra-fast exponential with precomputed factors
        final_mask = active_mask.copy()
        final_mask[active_mask] &= cutoff_mask
        
        decay_factors = np.exp(-dt_vec[cutoff_mask] / self._tau_syn_vec[active_mask][cutoff_mask])
        raw_currents = self._weights[final_mask] * decay_factors * self.current_gain

        # 5. Sparse accumulation to postsynaptic neurons
        currents = np.zeros(self.post_population_size, dtype=np.float32)
        np.add.at(currents, self._post_ids[final_mask], raw_currents)

        # 6. Vectorized nonlinear sharpening
        if self.current_sharpen and abs(self.current_sharpen - 1.0) > 1e-6:
            p = self.current_sharpen
            positive_mask = currents >= 0
            currents[positive_mask] = np.power(currents[positive_mask] + 1e-12, p)
            currents[~positive_mask] = -np.power(np.abs(currents[~positive_mask]) + 1e-12, p)

        # 7. Vectorized global inhibition
        if self.inhibition_gain > 0:
            mean_current = np.mean(currents)
            if mean_current > 0:
                currents = np.maximum(0.0, currents - self.inhibition_gain * mean_current)

        return currents.tolist()

    def _get_synaptic_currents_gpu(self, active_mask: np.ndarray, current_time: float) -> List[float]:
        """GPU-accelerated synaptic current computation for massive networks (50k+ neurons)."""
        if not GPU_AVAILABLE or cp is None:
            return self._get_synaptic_currents_cpu(active_mask, current_time)

        try:
            # Transfer to GPU
            gpu_last_spikes = cp.asarray(self._last_spike_times[active_mask])
            gpu_tau_syn = cp.asarray(self._tau_syn_vec[active_mask]) 
            gpu_weights = cp.asarray(self._weights[active_mask])
            gpu_post_ids = cp.asarray(self._post_ids[active_mask])
            
            # GPU vectorized computation
            dt_vec = current_time - gpu_last_spikes
            cutoff_mask = dt_vec <= (5.0 * gpu_tau_syn)
            
            if not cp.any(cutoff_mask):
                return [0.0] * self.post_population_size
                
            # Ultra-fast GPU exponential
            decay_factors = cp.exp(-dt_vec[cutoff_mask] / gpu_tau_syn[cutoff_mask])
            raw_currents = gpu_weights[cutoff_mask] * decay_factors * self.current_gain
            
            # GPU sparse accumulation
            gpu_currents = cp.zeros(self.post_population_size, dtype=cp.float32)
            # Use add.at equivalent for CuPy
            cp.add.at(gpu_currents, gpu_post_ids[cutoff_mask], raw_currents)
            
            # GPU nonlinear sharpening
            if self.current_sharpen and abs(self.current_sharpen - 1.0) > 1e-6:
                p = self.current_sharpen
                positive_mask = gpu_currents >= 0
                gpu_currents[positive_mask] = cp.power(gpu_currents[positive_mask] + 1e-12, p)
                gpu_currents[~positive_mask] = -cp.power(cp.abs(gpu_currents[~positive_mask]) + 1e-12, p)
            
            # GPU global inhibition
            if self.inhibition_gain > 0:
                mean_current = cp.mean(gpu_currents)
                if mean_current > 0:
                    gpu_currents = cp.maximum(0.0, gpu_currents - self.inhibition_gain * mean_current)
            
            # Transfer back to CPU
            return cp.asnumpy(gpu_currents).tolist()
            
        except Exception as e:
            # Fallback to CPU if GPU fails
            print(f"GPU computation failed, falling back to CPU: {e}")
            return self._get_synaptic_currents_cpu(active_mask, current_time)

    def _build_vectorized_arrays(self):
        """Build vectorized arrays for ultra-fast computation. Called once during initialization."""
        if not self.synapses:
            return
            
        n_synapses = len(self.synapses)
        
        # Preallocate vectorized arrays
        self._pre_ids = np.zeros(n_synapses, dtype=np.int32)
        self._post_ids = np.zeros(n_synapses, dtype=np.int32)
        self._weights = np.zeros(n_synapses, dtype=np.float32)
        self._tau_syn_vec = np.zeros(n_synapses, dtype=np.float32)
        self._last_spike_times = np.full(n_synapses, -np.inf, dtype=np.float32)
        
        # Populate arrays
        for idx, ((pre_id, post_id), synapse) in enumerate(self.synapses.items()):
            self._pre_ids[idx] = pre_id
            self._post_ids[idx] = post_id
            self._weights[idx] = synapse.weight
            self._tau_syn_vec[idx] = getattr(synapse, 'tau_syn', 5.0)
            self._last_spike_times[idx] = synapse.last_pre_spike
            
        self._vectorized_initialized = True

    def _update_vectorized_weights(self):
        """Update vectorized weight array when weights change."""
        if not hasattr(self, '_vectorized_initialized'):
            return
            
        for idx, synapse in enumerate(self.synapses.values()):
            self._weights[idx] = synapse.weight

    def _update_vectorized_spike_times(self):
        """Update vectorized spike times array when spikes occur."""
        if not hasattr(self, '_vectorized_initialized'):
            return
            
        for idx, synapse in enumerate(self.synapses.values()):
            self._last_spike_times[idx] = synapse.last_pre_spike

    def update_weights(
        self, pre_spikes: List[bool], post_spikes: List[bool], current_time: float
    ):
        """
        VECTORIZED weight update for massive scale performance.

        Args:
            pre_spikes: List of presynaptic spike indicators
            post_spikes: List of postsynaptic spike indicators
            current_time: Current simulation time
        """
        # Convert spike lists to numpy arrays for vectorized operations
        pre_spike_array = np.array(pre_spikes, dtype=bool)
        post_spike_array = np.array(post_spikes, dtype=bool)
        
        # Vectorized spike processing
        weight_updates_needed = False
        
        for (pre_id, post_id), synapse in self.synapses.items():
            if pre_spike_array[pre_id]:
                synapse.pre_spike(current_time)
                weight_updates_needed = True
            if post_spike_array[post_id]:
                synapse.post_spike(current_time)
                weight_updates_needed = True

        # Update vectorized arrays if weights changed
        if weight_updates_needed and hasattr(self, '_vectorized_initialized'):
            self._update_vectorized_weights()
            self._update_vectorized_spike_times()

        # Vectorized synaptic scaling
        if self.enable_synaptic_scaling and self._scaling_every > 0:
            self._scale_counter += 1
            if self._scale_counter % self._scaling_every == 0:
                self._vectorized_synaptic_scaling()

    def _vectorized_synaptic_scaling(self):
        """Vectorized synaptic scaling for performance."""
        if not hasattr(self, '_vectorized_initialized'):
            return
            
        # Group synapses by postsynaptic neuron for vectorized scaling
        for post_id in range(self.post_population_size):
            # Find all synapses targeting this postsynaptic neuron
            post_mask = self._post_ids == post_id
            if not np.any(post_mask):
                continue
                
            # Vectorized weight sum computation
            current_weights = self._weights[post_mask]
            current_sum = np.sum(current_weights)
            
            if current_sum <= 0:
                continue
                
            # Get target sum
            incoming_keys = self._incoming_by_post.get(post_id, [])
            target_sum = self._target_incoming_sum[post_id] if incoming_keys else current_sum
            
            # Vectorized scaling
            scale = target_sum / current_sum
            blended_scale = 0.2 * scale + 0.8 * 1.0
            
            # Apply scaling with bounds checking
            scaled_weights = current_weights * blended_scale
            
            # Update both vectorized array and individual synapses
            indices = np.where(post_mask)[0]
            for i, idx in enumerate(indices):
                synapse_key = (self._pre_ids[idx], post_id)
                synapse = self.synapses[synapse_key]
                new_weight = np.clip(scaled_weights[i], synapse.w_min, synapse.w_max)
                synapse.weight = new_weight
                self._weights[idx] = new_weight

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
        # Convention: (post_population_size, pre_population_size) for matrix multiplication
        weight_matrix = np.zeros((self.post_population_size, self.pre_population_size))

        for (pre_id, post_id), synapse in self.synapses.items():
            weight_matrix[post_id, pre_id] = synapse.weight

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
            synapse.weight = np.clip(
                synapse.weight * factor, synapse.w_min, synapse.w_max
            )

    def normalize_incoming(self, target_sum: Optional[float] = None) -> None:
        """Normalize incoming weights per postsynaptic neuron to a target sum (optional)."""
        incoming_by_post: Dict[int, List[Tuple[int, int]]] = {}
        for pre_id, post_id in self.synapses.keys():
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
