"""
Learning and plasticity mechanisms for the neuromorphic programming system.
Implements STDP, Hebbian, reward-modulated, and custom plasticity rules.
"""

import json
import yaml
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np

from core.logging_utils import neuromorphic_logger


class PlasticityType(Enum):
    """Types of plasticity mechanisms."""

    STDP = "stdp"  # Spike-Timing-Dependent Plasticity
    HEBBIAN = "hebbian"  # Classical Hebbian learning
    BCM = "bcm"  # Bienenstock-Cooper-Munro
    RSTDP = "rstdp"  # Reward-modulated STDP
    TRIPLET_STDP = "triplet_stdp"  # Triplet-based STDP
    HETEROSYNAPTIC = "heterosynaptic"  # Heterosynaptic plasticity
    HOMEOSTATIC = "homeostatic"  # Homeostatic plasticity
    CUSTOM = "custom"  # User-defined plasticity


@dataclass
class PlasticityConfig:
    """Configuration for plasticity mechanisms."""

    # Common parameters
    learning_rate: float = 0.01
    weight_min: float = 0.0
    weight_max: float = 10.0
    enabled: bool = True

    # STDP parameters
    tau_plus: float = 20.0  # LTP time constant (ms)
    tau_minus: float = 20.0  # LTD time constant (ms)
    A_plus: float = 0.01  # LTP amplitude
    A_minus: float = 0.01  # LTD amplitude

    # Hebbian parameters
    hebbian_threshold: float = 0.5
    hebbian_decay: float = 0.99

    # BCM parameters
    bcm_threshold: float = 0.5
    bcm_time_constant: float = 1000.0

    # Reward modulation parameters
    reward_decay: float = 0.9
    reward_sensitivity: float = 1.0
    dopamine_time_constant: float = 200.0
    
    # Enhanced dopamine system parameters
    baseline_dopamine: float = 0.2
    max_dopamine: float = 2.0
    min_dopamine: float = 0.0
    dopamine_decay_rate: float = 0.95
    prediction_learning_rate: float = 0.1
    discount_factor: float = 0.9
    trace_decay: float = 0.8
    eligibility_decay: float = 0.9
    eligibility_strength: float = 1.0

    # Triplet STDP parameters
    tau_x: float = 5.0
    tau_y: float = 10.0
    A2_plus: float = 0.01
    A3_plus: float = 0.001
    A2_minus: float = 0.01
    A3_minus: float = 0.001

    # Homeostatic parameters
    target_rate: float = 5.0  # Target firing rate (Hz)
    homeostatic_time_constant: float = 10000.0
    target_total_strength: float = 100.0  # Target total synaptic strength
    synaptic_scaling_rate: float = 0.001  # Rate of synaptic scaling
    excitability_scaling_rate: float = 0.0001  # Rate of excitability adjustment
    activity_threshold: float = 0.5  # Threshold for activity-dependent adjustments

    # Metaplasticity parameters
    metaplasticity_threshold: float = 0.5  # Initial plasticity threshold
    threshold_adaptation_rate: float = 0.001  # Rate of threshold adaptation
    learning_rate_modulation: float = 1.0  # Global learning rate modulation
    metaplasticity_window: int = 1000  # Window for activity history
    
    # Synaptic competition parameters
    competition_strength: float = 0.1
    normalization_target: float = 10.0
    normalization_rate: float = 0.01
    saturation_steepness: float = 2.0
    wta_threshold: float = 0.8
    wta_strength: float = 0.5
    soft_bound_width: float = 0.1
    
    # Multi-plasticity combination parameters
    use_stdp: bool = True
    use_homeostatic: bool = True
    use_metaplasticity: bool = False
    use_competition: bool = True
    use_reward_modulation: bool = False
    stdp_weight: float = 1.0
    homeostatic_weight: float = 0.1
    metaplasticity_weight: float = 0.5
    competition_weight: float = 0.3
    reward_weight: float = 1.0
    
    # Custom rule parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, filepath: Union[str, Path]) -> "PlasticityConfig":
        """Load configuration from YAML file."""
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_json(cls, filepath: Union[str, Path]) -> "PlasticityConfig":
        """Load configuration from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls(**data)

    def to_yaml(self, filepath: Union[str, Path]):
        """Save configuration to YAML file."""
        data = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        with open(filepath, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

    def to_json(self, filepath: Union[str, Path]):
        """Save configuration to JSON file."""
        data = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)


class PlasticityRule(ABC):
    """Abstract base class for plasticity rules."""

    def __init__(self, config: PlasticityConfig):
        """
        Initialize plasticity rule.

        Args:
            config: Plasticity configuration
        """
        self.config = config
        self.weight_history = []

    @abstractmethod
    def compute_weight_change(
        self, pre_activity: float, post_activity: float, current_weight: float, **kwargs
    ) -> float:
        """
        Compute weight change based on neural activity.

        Args:
            pre_activity: Presynaptic activity
            post_activity: Postsynaptic activity
            current_weight: Current synaptic weight
            **kwargs: Additional parameters specific to the rule

        Returns:
            Weight change (delta_w)
        """
        pass

    def apply_weight_bounds(self, weight: float) -> float:
        """Apply weight boundaries."""
        return np.clip(weight, self.config.weight_min, self.config.weight_max)

    def update_weight(
        self, current_weight: float, pre_activity: float, post_activity: float, **kwargs
    ) -> float:
        """
        Update synaptic weight.

        Args:
            current_weight: Current synaptic weight
            pre_activity: Presynaptic activity
            post_activity: Postsynaptic activity
            **kwargs: Additional parameters

        Returns:
            Updated weight
        """
        if not self.config.enabled:
            return current_weight

        delta_w = self.compute_weight_change(
            pre_activity, post_activity, current_weight, **kwargs
        )
        new_weight = self.apply_weight_bounds(current_weight + delta_w)
        self.weight_history.append(new_weight)
        return new_weight


class STDPRule(PlasticityRule):
    """Spike-Timing-Dependent Plasticity rule."""

    def __init__(self, config: PlasticityConfig):
        super().__init__(config)
        self.pre_trace = 0.0
        self.post_trace = 0.0

    def compute_weight_change(
        self,
        pre_activity: float,
        post_activity: float,
        current_weight: float,
        dt: float = 1.0,
        pre_spike: bool = False,
        post_spike: bool = False,
        **kwargs,
    ) -> float:
        """
        Compute STDP weight change.

        Args:
            pre_activity: Presynaptic trace
            post_activity: Postsynaptic trace
            current_weight: Current weight
            dt: Time step
            pre_spike: Whether presynaptic neuron spiked
            post_spike: Whether postsynaptic neuron spiked

        Returns:
            Weight change
        """
        delta_w = 0.0

        # Update traces
        self.pre_trace *= np.exp(-dt / self.config.tau_plus)
        self.post_trace *= np.exp(-dt / self.config.tau_minus)

        # LTP: Pre-spike followed by post-spike
        if pre_spike:
            delta_w -= self.config.A_minus * self.post_trace
            self.pre_trace += 1.0

        # LTD: Post-spike followed by pre-spike
        if post_spike:
            delta_w += self.config.A_plus * self.pre_trace
            self.post_trace += 1.0

        # Apply weight-dependent scaling
        if current_weight > 0:
            if delta_w > 0:  # LTP
                delta_w *= (
                    self.config.weight_max - current_weight
                ) / self.config.weight_max
            else:  # LTD
                delta_w *= current_weight / self.config.weight_max

        return delta_w * self.config.learning_rate


class HebbianRule(PlasticityRule):
    """Classical Hebbian learning rule: 'Cells that fire together, wire together'."""

    def compute_weight_change(
        self, pre_activity: float, post_activity: float, current_weight: float, **kwargs
    ) -> float:
        """
        Compute Hebbian weight change.

        Args:
            pre_activity: Presynaptic activity (0-1)
            post_activity: Postsynaptic activity (0-1)
            current_weight: Current weight

        Returns:
            Weight change
        """
        # Basic Hebbian rule with decay
        correlation = pre_activity * post_activity

        # Apply threshold
        if correlation > self.config.hebbian_threshold:
            # Potentiation
            delta_w = self.config.learning_rate * correlation
            # Add weight-dependent scaling to prevent unbounded growth
            delta_w *= (
                self.config.weight_max - current_weight
            ) / self.config.weight_max
        else:
            # Decay term for stability
            delta_w = (
                -self.config.learning_rate
                * (1 - self.config.hebbian_decay)
                * current_weight
            )

        return delta_w


class BCMRule(PlasticityRule):
    """Bienenstock-Cooper-Munro (BCM) learning rule with sliding threshold."""

    def __init__(self, config: PlasticityConfig):
        super().__init__(config)
        self.sliding_threshold = config.bcm_threshold
        self.activity_history = []

    def compute_weight_change(
        self, pre_activity: float, post_activity: float, current_weight: float, **kwargs
    ) -> float:
        """
        Compute BCM weight change.

        Args:
            pre_activity: Presynaptic activity
            post_activity: Postsynaptic activity
            current_weight: Current weight

        Returns:
            Weight change
        """
        # BCM learning function
        phi = post_activity * (post_activity - self.sliding_threshold)
        delta_w = self.config.learning_rate * phi * pre_activity

        # Update sliding threshold based on activity history
        self.activity_history.append(post_activity)
        if len(self.activity_history) > 100:
            self.activity_history.pop(0)

        # Sliding threshold adapts to maintain stability
        mean_activity = (
            np.mean(self.activity_history) if self.activity_history else post_activity
        )
        tau = self.config.bcm_time_constant
        self.sliding_threshold += (mean_activity**2 - self.sliding_threshold) / tau

        return delta_w


class RewardModulatedSTDP(STDPRule):
    """Reward-modulated STDP for reinforcement learning."""

    def __init__(self, config: PlasticityConfig):
        super().__init__(config)
        self.eligibility_trace = 0.0
        self.reward_signal = 0.0
        self.dopamine_trace = 0.0

    def set_reward(self, reward: float):
        """Set reward/punishment signal."""
        self.reward_signal = reward
        self.dopamine_trace = reward  # Simplified dopamine model

    def compute_weight_change(
        self,
        pre_activity: float,
        post_activity: float,
        current_weight: float,
        dt: float = 1.0,
        pre_spike: bool = False,
        post_spike: bool = False,
        **kwargs,
    ) -> float:
        """
        Compute reward-modulated STDP weight change.

        Args:
            pre_activity: Presynaptic activity
            post_activity: Postsynaptic activity
            current_weight: Current weight
            dt: Time step
            pre_spike: Whether presynaptic neuron spiked
            post_spike: Whether postsynaptic neuron spiked

        Returns:
            Weight change
        """
        # Compute standard STDP
        stdp_change = super().compute_weight_change(
            pre_activity,
            post_activity,
            current_weight,
            dt,
            pre_spike,
            post_spike,
            **kwargs,
        )

        # Update eligibility trace
        self.eligibility_trace *= self.config.reward_decay
        self.eligibility_trace += stdp_change

        # Update dopamine trace
        self.dopamine_trace *= np.exp(-dt / self.config.dopamine_time_constant)

        # Modulate by reward/dopamine
        modulated_change = (
            self.eligibility_trace
            * self.dopamine_trace
            * self.config.reward_sensitivity
        )

        return modulated_change


class TripletSTDP(PlasticityRule):
    """Triplet-based STDP rule for more accurate modeling of experimental data."""

    def __init__(self, config: PlasticityConfig):
        super().__init__(config)
        self.r1 = 0.0  # Fast presynaptic trace
        self.r2 = 0.0  # Slow presynaptic trace
        self.o1 = 0.0  # Fast postsynaptic trace
        self.o2 = 0.0  # Slow postsynaptic trace

    def compute_weight_change(
        self,
        pre_activity: float,
        post_activity: float,
        current_weight: float,
        dt: float = 1.0,
        pre_spike: bool = False,
        post_spike: bool = False,
        **kwargs,
    ) -> float:
        """
        Compute triplet STDP weight change.

        Args:
            pre_activity: Presynaptic activity
            post_activity: Postsynaptic activity
            current_weight: Current weight
            dt: Time step
            pre_spike: Whether presynaptic neuron spiked
            post_spike: Whether postsynaptic neuron spiked

        Returns:
            Weight change
        """
        delta_w = 0.0

        # Decay traces
        self.r1 *= np.exp(-dt / self.config.tau_x)
        self.r2 *= np.exp(-dt / self.config.tau_plus)
        self.o1 *= np.exp(-dt / self.config.tau_y)
        self.o2 *= np.exp(-dt / self.config.tau_minus)

        if pre_spike:
            # LTD
            delta_w -= self.o1 * (self.config.A2_minus + self.config.A3_minus * self.r2)
            # Update traces
            self.r1 = 1.0
            self.r2 = 1.0

        if post_spike:
            # LTP
            delta_w += self.r1 * (self.config.A2_plus + self.config.A3_plus * self.o2)
            # Update traces
            self.o1 = 1.0
            self.o2 = 1.0

        return delta_w * self.config.learning_rate


class HomeostaticPlasticity(PlasticityRule):
    """
    Enhanced homeostatic plasticity to maintain stable firing rates and prevent runaway dynamics.
    
    Implements multiple homeostatic mechanisms:
    - Synaptic scaling to maintain total synaptic strength
    - Intrinsic excitability regulation
    - Activity-dependent threshold adjustment
    """

    def __init__(self, config: PlasticityConfig):
        super().__init__(config)
        self.firing_rate_estimate = 0.0
        self.time_window = []
        
        # Enhanced homeostatic mechanisms
        self.total_synaptic_strength = 0.0
        self.target_total_strength = getattr(config, 'target_total_strength', 100.0)
        self.intrinsic_excitability = 1.0
        self.activity_threshold = getattr(config, 'activity_threshold', 0.5)
        
        # Scaling factors for different mechanisms
        self.synaptic_scaling_rate = getattr(config, 'synaptic_scaling_rate', 0.001)
        self.excitability_scaling_rate = getattr(config, 'excitability_scaling_rate', 0.0001)
        
        # History tracking for better homeostasis
        self.activity_history = []
        self.weight_history_homeostatic = []
        
    def update_total_synaptic_strength(self, all_weights: np.ndarray):
        """
        Update estimate of total synaptic strength for scaling.
        
        Args:
            all_weights: Array of all synaptic weights for this neuron
        """
        self.total_synaptic_strength = np.sum(all_weights)
        
    def compute_synaptic_scaling(self, current_weight: float, all_weights: np.ndarray) -> float:
        """
        Compute synaptic scaling to maintain total synaptic strength.
        
        Args:
            current_weight: Current weight of this synapse
            all_weights: All weights for the postsynaptic neuron
            
        Returns:
            Scaling factor for this synapse
        """
        if len(all_weights) == 0 or np.sum(all_weights) == 0:
            return 0.0
            
        current_total = np.sum(all_weights)
        if current_total == 0:
            return 0.0
            
        # Multiplicative scaling to maintain total strength
        scaling_factor = self.target_total_strength / current_total
        
        # Gradual adjustment to prevent instability
        adjustment = (scaling_factor - 1.0) * self.synaptic_scaling_rate
        
        return current_weight * adjustment
        
    def compute_intrinsic_excitability_change(self, current_rate: float) -> float:
        """
        Compute change in intrinsic excitability based on activity.
        
        Args:
            current_rate: Current firing rate
            
        Returns:
            Change in intrinsic excitability
        """
        rate_error = self.config.target_rate - current_rate
        
        # Intrinsic excitability adjusts to maintain target rate
        excitability_change = rate_error * self.excitability_scaling_rate
        
        # Update intrinsic excitability with bounds
        self.intrinsic_excitability = np.clip(
            self.intrinsic_excitability + excitability_change,
            0.1, 3.0  # Reasonable bounds for excitability
        )
        
        return excitability_change

    def compute_weight_change(
        self,
        pre_activity: float,
        post_activity: float,
        current_weight: float,
        post_spike: bool = False,
        dt: float = 1.0,
        all_weights: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:
        """
        Compute enhanced homeostatic weight change.

        Args:
            pre_activity: Presynaptic activity
            post_activity: Postsynaptic activity
            current_weight: Current weight
            post_spike: Whether postsynaptic neuron spiked
            dt: Time step
            all_weights: All synaptic weights for this postsynaptic neuron

        Returns:
            Weight change
        """
        # Update firing rate estimate
        if post_spike:
            self.time_window.append(1.0)
        else:
            self.time_window.append(0.0)

        # Keep sliding window
        window_size = int(1000.0 / dt)  # 1 second window
        if len(self.time_window) > window_size:
            self.time_window.pop(0)

        # Calculate current firing rate (Hz)
        if self.time_window:
            # Convert to Hz: spikes per second
            # dt is in ms, so convert to seconds
            window_duration_sec = len(self.time_window) * dt / 1000.0
            total_spikes = sum(self.time_window)
            current_rate = total_spikes / window_duration_sec if window_duration_sec > 0 else 0.0
        else:
            current_rate = 0.0
        
        # Track activity history
        self.activity_history.append(current_rate)
        if len(self.activity_history) > 1000:  # Keep last 1000 samples
            self.activity_history.pop(0)

        # Basic homeostatic adjustment (original mechanism)
        rate_error = self.config.target_rate - current_rate
        
        # Scale adjustment by rate error magnitude and make it more responsive
        if rate_error > 0:  # Too low activity, increase weights
            basic_adjustment = (
                self.config.learning_rate
                * rate_error
                * (self.config.weight_max - current_weight) / self.config.weight_max
                / self.config.homeostatic_time_constant
            )
        else:  # Too high activity, decrease weights
            basic_adjustment = (
                self.config.learning_rate
                * rate_error
                * current_weight / self.config.weight_max
                / self.config.homeostatic_time_constant
            )
        
        # Enhanced mechanisms
        delta_w = basic_adjustment
        
        # 1. Synaptic scaling component
        if all_weights is not None and len(all_weights) > 0:
            scaling_adjustment = self.compute_synaptic_scaling(current_weight, all_weights)
            delta_w += scaling_adjustment
            
        # 2. Activity-dependent threshold adjustment
        if len(self.activity_history) > 10:
            mean_activity = np.mean(self.activity_history[-100:])  # Recent activity
            if mean_activity > self.config.target_rate * 1.5:  # Too active
                # Reduce weights more aggressively
                delta_w -= self.config.learning_rate * 0.1 * current_weight
            elif mean_activity < self.config.target_rate * 0.5:  # Too quiet
                # Increase weights more aggressively
                delta_w += self.config.learning_rate * 0.1 * current_weight
                
        # 3. Prevent runaway excitation/depression
        if current_rate > self.config.target_rate * 3.0:  # Runaway excitation
            # Strong depression
            delta_w -= self.config.learning_rate * 0.5 * current_weight
        elif current_rate < 0.1:  # Near silence
            # Gentle potentiation
            delta_w += self.config.learning_rate * 0.2 * (self.config.weight_max - current_weight)
            
        # 4. Update intrinsic excitability (for external use)
        self.compute_intrinsic_excitability_change(current_rate)
        
        # Track weight changes for analysis
        self.weight_history_homeostatic.append(delta_w)
        if len(self.weight_history_homeostatic) > 1000:
            self.weight_history_homeostatic.pop(0)

        return delta_w
        
    def get_homeostatic_state(self) -> Dict[str, float]:
        """
        Get current homeostatic state for monitoring.
        
        Returns:
            Dictionary with homeostatic state variables
        """
        return {
            'firing_rate_estimate': self.firing_rate_estimate,
            'total_synaptic_strength': self.total_synaptic_strength,
            'intrinsic_excitability': self.intrinsic_excitability,
            'mean_recent_activity': np.mean(self.activity_history[-100:]) if len(self.activity_history) >= 100 else 0.0,
            'activity_variance': np.var(self.activity_history[-100:]) if len(self.activity_history) >= 100 else 0.0,
            'target_rate': self.config.target_rate
        }
        
    def reset_homeostatic_state(self):
        """Reset homeostatic state variables."""
        self.firing_rate_estimate = 0.0
        self.time_window = []
        self.total_synaptic_strength = 0.0
        self.intrinsic_excitability = 1.0
        self.activity_history = []
        self.weight_history_homeostatic = []


class MetaplasticityRule(PlasticityRule):
    """
    Metaplasticity rule implementing 'plasticity of plasticity'.
    
    The plasticity threshold and learning rates adapt based on recent activity history,
    implementing the Bienenstock-Cooper-Munro (BCM) sliding threshold concept
    and activity-dependent learning rate modulation.
    """
    
    def __init__(self, config: PlasticityConfig):
        super().__init__(config)
        
        # Metaplasticity parameters
        self.plasticity_threshold = getattr(config, 'metaplasticity_threshold', 0.5)
        self.threshold_adaptation_rate = getattr(config, 'threshold_adaptation_rate', 0.001)
        self.learning_rate_modulation = getattr(config, 'learning_rate_modulation', 1.0)
        
        # Activity history for threshold adaptation
        self.activity_history = []
        self.plasticity_history = []
        self.recent_activity_window = getattr(config, 'metaplasticity_window', 1000)
        
        # Dynamic learning rates
        self.current_ltp_rate = config.A_plus
        self.current_ltd_rate = config.A_minus
        
        # Sliding threshold for LTP/LTD induction
        self.sliding_threshold = self.plasticity_threshold
        
    def update_sliding_threshold(self, recent_activity: float):
        """
        Update the sliding threshold based on recent activity.
        
        Args:
            recent_activity: Recent postsynaptic activity level
        """
        # BCM-like sliding threshold
        # Threshold increases with high activity, decreases with low activity
        target_threshold = recent_activity ** 2  # Quadratic relationship as in BCM
        
        # Gradual adaptation toward target
        threshold_error = target_threshold - self.sliding_threshold
        self.sliding_threshold += threshold_error * self.threshold_adaptation_rate
        
        # Keep threshold in reasonable bounds
        self.sliding_threshold = np.clip(self.sliding_threshold, 0.1, 2.0)
        
    def modulate_learning_rates(self, recent_activity: float, activity_variance: float):
        """
        Modulate learning rates based on activity history.
        
        Args:
            recent_activity: Mean recent activity
            activity_variance: Variance in recent activity
        """
        # High variance (unstable activity) -> reduce learning rates for stability
        variance_factor = 1.0 / (1.0 + activity_variance * 2.0)
        
        # Activity level modulation
        if recent_activity > self.sliding_threshold:
            # High activity -> favor LTD to prevent runaway potentiation
            ltp_modulation = 0.5 * variance_factor
            ltd_modulation = 1.5 * variance_factor
        else:
            # Low activity -> favor LTP to maintain connectivity
            ltp_modulation = 1.5 * variance_factor
            ltd_modulation = 0.5 * variance_factor
            
        # Update current learning rates
        self.current_ltp_rate = self.config.A_plus * ltp_modulation * self.learning_rate_modulation
        self.current_ltd_rate = self.config.A_minus * ltd_modulation * self.learning_rate_modulation
        
    def compute_weight_change(
        self,
        pre_activity: float,
        post_activity: float,
        current_weight: float,
        dt: float = 1.0,
        pre_spike: bool = False,
        post_spike: bool = False,
        **kwargs
    ) -> float:
        """
        Compute metaplastic weight change.
        
        Args:
            pre_activity: Presynaptic activity
            post_activity: Postsynaptic activity  
            current_weight: Current weight
            dt: Time step
            pre_spike: Whether presynaptic neuron spiked
            post_spike: Whether postsynaptic neuron spiked
            
        Returns:
            Weight change modulated by metaplasticity
        """
        # Track activity history
        self.activity_history.append(post_activity)
        if len(self.activity_history) > self.recent_activity_window:
            self.activity_history.pop(0)
            
        # Calculate recent activity statistics
        if len(self.activity_history) >= 10:
            recent_activity = np.mean(self.activity_history[-100:]) if len(self.activity_history) >= 100 else np.mean(self.activity_history)
            activity_variance = np.var(self.activity_history[-100:]) if len(self.activity_history) >= 100 else np.var(self.activity_history)
            
            # Update metaplastic parameters
            self.update_sliding_threshold(recent_activity)
            self.modulate_learning_rates(recent_activity, activity_variance)
        else:
            recent_activity = post_activity
            activity_variance = 0.0
            
        # Compute plasticity based on sliding threshold
        delta_w = 0.0
        
        if pre_spike and post_spike:
            # Coincident activity - direction depends on postsynaptic activity level
            if post_activity > self.sliding_threshold:
                # Above threshold -> LTP
                delta_w = self.current_ltp_rate * (self.config.weight_max - current_weight) / self.config.weight_max
            else:
                # Below threshold -> LTD  
                delta_w = -self.current_ltd_rate * current_weight / self.config.weight_max
                
        elif pre_spike and not post_spike:
            # Pre-spike without post-spike -> weak LTD
            delta_w = -self.current_ltd_rate * 0.1 * current_weight / self.config.weight_max
            
        elif post_spike and not pre_spike:
            # Post-spike without pre-spike -> very weak LTD (heterosynaptic)
            delta_w = -self.current_ltd_rate * 0.05 * current_weight / self.config.weight_max
            
        # Track plasticity history
        self.plasticity_history.append(abs(delta_w))
        if len(self.plasticity_history) > self.recent_activity_window:
            self.plasticity_history.pop(0)
            
        return delta_w * self.config.learning_rate
        
    def get_metaplastic_state(self) -> Dict[str, float]:
        """
        Get current metaplastic state for monitoring.
        
        Returns:
            Dictionary with metaplastic state variables
        """
        recent_activity = np.mean(self.activity_history[-100:]) if len(self.activity_history) >= 100 else 0.0
        activity_variance = np.var(self.activity_history[-100:]) if len(self.activity_history) >= 100 else 0.0
        recent_plasticity = np.mean(self.plasticity_history[-100:]) if len(self.plasticity_history) >= 100 else 0.0
        
        return {
            'sliding_threshold': self.sliding_threshold,
            'current_ltp_rate': self.current_ltp_rate,
            'current_ltd_rate': self.current_ltd_rate,
            'recent_activity': recent_activity,
            'activity_variance': activity_variance,
            'recent_plasticity': recent_plasticity,
            'plasticity_threshold': self.plasticity_threshold
        }
        
    def reset_metaplastic_state(self):
        """Reset metaplastic state variables."""
        self.activity_history = []
        self.plasticity_history = []
        self.sliding_threshold = self.plasticity_threshold
        self.current_ltp_rate = self.config.A_plus
        self.current_ltd_rate = self.config.A_minus


class DopamineNeuromodulationSystem:
    """
    Sophisticated dopamine neuromodulation system implementing reward prediction error.
    
    This system models dopamine neurons that:
    - Respond to unexpected rewards (positive prediction error)
    - Show depression for expected rewards that don't occur (negative prediction error)
    - Learn to predict rewards through temporal difference learning
    - Modulate synaptic plasticity based on dopamine levels
    """
    
    def __init__(self, config: PlasticityConfig):
        """
        Initialize dopamine neuromodulation system.
        
        Args:
            config: Plasticity configuration containing dopamine parameters
        """
        self.config = config
        
        # Dopamine system parameters
        self.baseline_dopamine = getattr(config, 'baseline_dopamine', 0.2)
        self.max_dopamine = getattr(config, 'max_dopamine', 2.0)
        self.min_dopamine = getattr(config, 'min_dopamine', 0.0)
        self.dopamine_decay_rate = getattr(config, 'dopamine_decay_rate', 0.95)
        
        # Reward prediction parameters
        self.learning_rate_prediction = getattr(config, 'prediction_learning_rate', 0.1)
        self.discount_factor = getattr(config, 'discount_factor', 0.9)
        self.trace_decay = getattr(config, 'trace_decay', 0.8)
        
        # State variables
        self.current_dopamine = self.baseline_dopamine
        self.reward_prediction = 0.0
        self.eligibility_trace = 0.0
        self.recent_rewards = []
        self.recent_predictions = []
        self.prediction_errors = []
        
        # Context tracking for temporal difference learning
        self.previous_state_value = 0.0
        self.current_state_value = 0.0
        
    def compute_reward_prediction_error(self, reward: float, state_value: float = 0.0) -> float:
        """
        Compute reward prediction error using temporal difference learning.
        
        Args:
            reward: Actual reward received
            state_value: Current state value estimate
            
        Returns:
            Reward prediction error (delta)
        """
        # TD error: δ = r + γV(s') - V(s)
        # For simplicity, we use reward prediction instead of full state values
        predicted_reward = self.reward_prediction
        prediction_error = reward - predicted_reward
        
        # Update reward prediction using TD learning
        self.reward_prediction += self.learning_rate_prediction * prediction_error
        
        # Keep prediction in reasonable bounds
        self.reward_prediction = np.clip(self.reward_prediction, -2.0, 2.0)
        
        # Track history
        self.recent_rewards.append(reward)
        self.recent_predictions.append(predicted_reward)
        self.prediction_errors.append(prediction_error)
        
        # Keep history bounded
        if len(self.recent_rewards) > 1000:
            self.recent_rewards.pop(0)
            self.recent_predictions.pop(0)
            self.prediction_errors.pop(0)
            
        return prediction_error
        
    def update_dopamine_level(self, reward: float, context_strength: float = 1.0) -> float:
        """
        Update dopamine level based on reward prediction error.
        
        Args:
            reward: Reward signal received
            context_strength: Strength of contextual cues (0-1)
            
        Returns:
            Updated dopamine level
        """
        # Compute prediction error
        prediction_error = self.compute_reward_prediction_error(reward, context_strength)
        
        # Dopamine response based on prediction error
        if prediction_error > 0:
            # Positive prediction error -> dopamine burst
            dopamine_change = prediction_error * 0.5
        elif prediction_error < 0:
            # Negative prediction error -> dopamine dip
            dopamine_change = prediction_error * 0.3
        else:
            # No prediction error -> return to baseline
            dopamine_change = 0.0
            
        # Update dopamine level
        self.current_dopamine += dopamine_change
        
        # Apply bounds
        self.current_dopamine = np.clip(
            self.current_dopamine, 
            self.min_dopamine, 
            self.max_dopamine
        )
        
        return self.current_dopamine
        
    def decay_dopamine(self, dt: float = 1.0):
        """
        Apply exponential decay to dopamine level toward baseline.
        
        Args:
            dt: Time step for decay
        """
        # Exponential decay toward baseline
        decay_factor = self.dopamine_decay_rate ** dt
        self.current_dopamine = (
            self.current_dopamine * decay_factor + 
            self.baseline_dopamine * (1 - decay_factor)
        )
        
        # Update eligibility trace
        self.eligibility_trace *= self.trace_decay
        
    def get_plasticity_modulation(self) -> float:
        """
        Get current plasticity modulation factor based on dopamine level.
        
        Returns:
            Modulation factor for synaptic plasticity (0-2, where 1 is baseline)
        """
        # Normalize dopamine level to modulation factor
        # Baseline dopamine -> 1.0 modulation
        # High dopamine -> enhanced plasticity
        # Low dopamine -> reduced plasticity
        modulation = self.current_dopamine / self.baseline_dopamine
        return np.clip(modulation, 0.1, 3.0)
        
    def set_eligibility_trace(self, value: float):
        """Set eligibility trace for reward-modulated plasticity."""
        self.eligibility_trace = value
        
    def get_dopamine_state(self) -> Dict[str, float]:
        """
        Get current dopamine system state for monitoring.
        
        Returns:
            Dictionary with dopamine system state variables
        """
        recent_error = np.mean(self.prediction_errors[-10:]) if len(self.prediction_errors) >= 10 else 0.0
        prediction_accuracy = 1.0 - np.mean(np.abs(self.prediction_errors[-100:])) if len(self.prediction_errors) >= 100 else 0.0
        
        return {
            'current_dopamine': self.current_dopamine,
            'baseline_dopamine': self.baseline_dopamine,
            'reward_prediction': self.reward_prediction,
            'recent_prediction_error': recent_error,
            'prediction_accuracy': max(0.0, prediction_accuracy),
            'eligibility_trace': self.eligibility_trace,
            'plasticity_modulation': self.get_plasticity_modulation()
        }
        
    def reset_dopamine_system(self):
        """Reset dopamine system to initial state."""
        self.current_dopamine = self.baseline_dopamine
        self.reward_prediction = 0.0
        self.eligibility_trace = 0.0
        self.recent_rewards = []
        self.recent_predictions = []
        self.prediction_errors = []
        self.previous_state_value = 0.0
        self.current_state_value = 0.0


class EnhancedRewardModulatedSTDP(STDPRule):
    """
    Enhanced reward-modulated STDP with sophisticated dopamine system.
    
    Integrates with DopamineNeuromodulationSystem for realistic reward learning.
    """
    
    def __init__(self, config: PlasticityConfig):
        super().__init__(config)
        
        # Initialize dopamine system
        self.dopamine_system = DopamineNeuromodulationSystem(config)
        
        # Enhanced eligibility trace parameters
        self.eligibility_decay = getattr(config, 'eligibility_decay', 0.9)
        self.eligibility_strength = getattr(config, 'eligibility_strength', 1.0)
        
        # Separate eligibility traces for LTP and LTD
        self.ltp_eligibility = 0.0
        self.ltd_eligibility = 0.0
        
    def compute_weight_change(
        self,
        pre_activity: float,
        post_activity: float,
        current_weight: float,
        dt: float = 1.0,
        pre_spike: bool = False,
        post_spike: bool = False,
        reward: Optional[float] = None,
        **kwargs,
    ) -> float:
        """
        Compute enhanced reward-modulated STDP weight change.
        
        Args:
            pre_activity: Presynaptic activity
            post_activity: Postsynaptic activity
            current_weight: Current weight
            dt: Time step
            pre_spike: Whether presynaptic neuron spiked
            post_spike: Whether postsynaptic neuron spiked
            reward: Reward signal (if provided)
            
        Returns:
            Weight change modulated by dopamine
        """
        # Compute standard STDP eligibility
        stdp_change = super().compute_weight_change(
            pre_activity, post_activity, current_weight, dt, pre_spike, post_spike, **kwargs
        )
        
        # Update eligibility traces
        self.ltp_eligibility *= np.exp(-dt / self.config.tau_plus)
        self.ltd_eligibility *= np.exp(-dt / self.config.tau_minus)
        
        # Add to eligibility based on STDP
        if stdp_change > 0:  # LTP
            self.ltp_eligibility += stdp_change * self.eligibility_strength
        elif stdp_change < 0:  # LTD
            self.ltd_eligibility += abs(stdp_change) * self.eligibility_strength
            
        # Update dopamine system if reward provided
        if reward is not None:
            self.dopamine_system.update_dopamine_level(reward)
        else:
            # Natural decay
            self.dopamine_system.decay_dopamine(dt)
            
        # Get plasticity modulation from dopamine
        modulation = self.dopamine_system.get_plasticity_modulation()
        
        # Apply dopamine modulation to eligibility traces
        if modulation > 1.0:  # High dopamine enhances LTP
            ltp_change = self.ltp_eligibility * (modulation - 1.0)
            ltd_change = -self.ltd_eligibility * 0.5  # Slight LTD reduction
        elif modulation < 1.0:  # Low dopamine enhances LTD
            ltp_change = self.ltp_eligibility * 0.5  # Reduced LTP
            ltd_change = -self.ltd_eligibility * (1.0 - modulation)
        else:  # Baseline dopamine
            ltp_change = self.ltp_eligibility * 0.1
            ltd_change = -self.ltd_eligibility * 0.1
            
        total_change = ltp_change + ltd_change
        
        # Apply weight-dependent scaling
        if total_change > 0:  # LTP
            total_change *= (self.config.weight_max - current_weight) / self.config.weight_max
        else:  # LTD
            total_change *= current_weight / self.config.weight_max
            
        return total_change * self.config.learning_rate
        
    def set_reward(self, reward: float, context_strength: float = 1.0):
        """
        Set reward signal for dopamine system.
        
        Args:
            reward: Reward value
            context_strength: Contextual strength for prediction
        """
        self.dopamine_system.update_dopamine_level(reward, context_strength)
        
    def get_dopamine_state(self) -> Dict[str, float]:
        """Get current dopamine system state."""
        state = self.dopamine_system.get_dopamine_state()
        state.update({
            'ltp_eligibility': self.ltp_eligibility,
            'ltd_eligibility': self.ltd_eligibility
        })
        return state
        
    def reset_reward_system(self):
        """Reset reward modulation system."""
        self.dopamine_system.reset_dopamine_system()
        self.ltp_eligibility = 0.0
        self.ltd_eligibility = 0.0


class SynapticCompetitionRule(PlasticityRule):
    """
    Synaptic competition and saturation rule implementing realistic weight dynamics.
    
    This rule implements:
    - Competition between synapses on the same postsynaptic neuron
    - Realistic upper and lower bounds with soft saturation
    - Weight normalization to prevent unbounded growth
    - Winner-take-all dynamics when appropriate
    """
    
    def __init__(self, config: PlasticityConfig):
        super().__init__(config)
        
        # Competition parameters
        self.competition_strength = getattr(config, 'competition_strength', 0.1)
        self.normalization_target = getattr(config, 'normalization_target', 10.0)
        self.normalization_rate = getattr(config, 'normalization_rate', 0.01)
        self.saturation_steepness = getattr(config, 'saturation_steepness', 2.0)
        
        # Winner-take-all parameters
        self.wta_threshold = getattr(config, 'wta_threshold', 0.8)
        self.wta_strength = getattr(config, 'wta_strength', 0.5)
        
        # Soft bounds parameters
        self.soft_bound_width = getattr(config, 'soft_bound_width', 0.1)
        
        # State tracking
        self.recent_activities = []
        self.competition_history = []
        
    def apply_soft_bounds(self, weight: float, target_change: float) -> float:
        """
        Apply soft saturation bounds that become stronger near limits.
        
        Args:
            weight: Current weight
            target_change: Desired weight change
            
        Returns:
            Bounded weight change
        """
        # Distance from bounds (0 to 1)
        upper_distance = (self.config.weight_max - weight) / self.config.weight_max
        lower_distance = (weight - self.config.weight_min) / self.config.weight_max
        
        # Soft saturation function: sigmoid-like
        if target_change > 0:  # Potentiation
            # Reduce potentiation as we approach upper bound
            saturation_factor = 1.0 / (1.0 + np.exp(-self.saturation_steepness * (upper_distance - 0.5)))
            bounded_change = target_change * saturation_factor
        else:  # Depression
            # Reduce depression as we approach lower bound
            saturation_factor = 1.0 / (1.0 + np.exp(-self.saturation_steepness * (lower_distance - 0.5)))
            bounded_change = target_change * saturation_factor
            
        return bounded_change
        
    def compute_competition_effect(
        self, 
        current_weight: float, 
        all_weights: np.ndarray, 
        activity_strength: float
    ) -> float:
        """
        Compute competitive weight change based on other synapses.
        
        Args:
            current_weight: Weight of this synapse
            all_weights: All weights on the same postsynaptic neuron
            activity_strength: Strength of current activity
            
        Returns:
            Competition-based weight change
        """
        if len(all_weights) <= 1:
            return 0.0
            
        # Normalize weights for comparison
        total_weight = np.sum(all_weights)
        if total_weight == 0:
            return 0.0
            
        weight_fraction = current_weight / total_weight
        
        # Competition based on relative strength
        # Strong synapses get stronger, weak synapses get weaker
        mean_fraction = 1.0 / len(all_weights)  # Equal share
        
        if weight_fraction > mean_fraction:
            # Above average -> competitive advantage
            competition_change = self.competition_strength * activity_strength * (weight_fraction - mean_fraction)
        else:
            # Below average -> competitive disadvantage
            competition_change = -self.competition_strength * activity_strength * (mean_fraction - weight_fraction)
            
        return competition_change
        
    def compute_normalization_effect(self, current_weight: float, all_weights: np.ndarray) -> float:
        """
        Compute weight normalization to maintain total synaptic strength.
        
        Args:
            current_weight: Weight of this synapse
            all_weights: All weights on the same postsynaptic neuron
            
        Returns:
            Normalization-based weight change
        """
        if len(all_weights) == 0:
            return 0.0
            
        total_weight = np.sum(all_weights)
        if total_weight == 0:
            return 0.0
            
        # Target normalization
        normalization_factor = self.normalization_target / total_weight
        
        # Gradual adjustment toward normalized weight
        target_weight = current_weight * normalization_factor
        normalization_change = (target_weight - current_weight) * self.normalization_rate
        
        return normalization_change
        
    def compute_winner_take_all_effect(
        self, 
        current_weight: float, 
        all_weights: np.ndarray, 
        activity_strength: float
    ) -> float:
        """
        Compute winner-take-all dynamics for strong competition.
        
        Args:
            current_weight: Weight of this synapse
            all_weights: All weights on the same postsynaptic neuron
            activity_strength: Strength of current activity
            
        Returns:
            Winner-take-all weight change
        """
        if len(all_weights) <= 1 or activity_strength < self.wta_threshold:
            return 0.0
            
        # Find the strongest synapse
        max_weight = np.max(all_weights)
        if max_weight == 0:
            return 0.0
            
        # Winner-take-all only for very strong activity
        if current_weight == max_weight:
            # This is the winner -> strengthen
            wta_change = self.wta_strength * activity_strength * (self.config.weight_max - current_weight) / self.config.weight_max
        else:
            # This is a loser -> weaken
            wta_change = -self.wta_strength * activity_strength * current_weight / self.config.weight_max
            
        return wta_change

    def compute_weight_change(
        self,
        pre_activity: float,
        post_activity: float,
        current_weight: float,
        all_weights: Optional[np.ndarray] = None,
        base_change: float = 0.0,
        **kwargs,
    ) -> float:
        """
        Compute competitive weight change with saturation.
        
        Args:
            pre_activity: Presynaptic activity
            post_activity: Postsynaptic activity
            current_weight: Current weight
            all_weights: All weights on the same postsynaptic neuron
            base_change: Base weight change from other plasticity rules
            
        Returns:
            Total weight change including competition and saturation
        """
        # Track activity
        activity_strength = pre_activity * post_activity
        self.recent_activities.append(activity_strength)
        if len(self.recent_activities) > 100:
            self.recent_activities.pop(0)
            
        # Start with base change (from other plasticity rules)
        total_change = base_change
        
        # Add competitive effects if we have information about other synapses
        if all_weights is not None and len(all_weights) > 1:
            # Competition effect
            competition_change = self.compute_competition_effect(
                current_weight, all_weights, activity_strength
            )
            total_change += competition_change
            
            # Normalization effect
            normalization_change = self.compute_normalization_effect(
                current_weight, all_weights
            )
            total_change += normalization_change
            
            # Winner-take-all effect for strong activity
            wta_change = self.compute_winner_take_all_effect(
                current_weight, all_weights, activity_strength
            )
            total_change += wta_change
            
            # Track competition history
            self.competition_history.append({
                'competition': competition_change,
                'normalization': normalization_change,
                'wta': wta_change,
                'activity': activity_strength
            })
            if len(self.competition_history) > 1000:
                self.competition_history.pop(0)
        
        # Apply soft bounds
        bounded_change = self.apply_soft_bounds(current_weight, total_change)
        
        return bounded_change * self.config.learning_rate
        
    def get_competition_state(self) -> Dict[str, float]:
        """
        Get current competition state for monitoring.
        
        Returns:
            Dictionary with competition state variables
        """
        recent_activity = np.mean(self.recent_activities[-10:]) if len(self.recent_activities) >= 1 else 0.0
        
        if len(self.competition_history) >= 10:
            recent_competition = np.mean([h['competition'] for h in self.competition_history[-10:]])
            recent_normalization = np.mean([h['normalization'] for h in self.competition_history[-10:]])
            recent_wta = np.mean([h['wta'] for h in self.competition_history[-10:]])
        else:
            recent_competition = 0.0
            recent_normalization = 0.0
            recent_wta = 0.0
            
        return {
            'recent_activity': recent_activity,
            'recent_competition': recent_competition,
            'recent_normalization': recent_normalization,
            'recent_wta': recent_wta,
            'competition_strength': self.competition_strength,
            'normalization_target': self.normalization_target,
            'wta_threshold': self.wta_threshold
        }
        
    def reset_competition_state(self):
        """Reset competition state variables."""
        self.recent_activities = []
        self.competition_history = []


class MultiPlasticityRule(PlasticityRule):
    """
    Multi-plasticity rule that combines multiple plasticity mechanisms.
    
    This rule can combine:
    - STDP
    - Homeostatic plasticity
    - Metaplasticity
    - Reward modulation
    - Synaptic competition
    """
    
    def __init__(self, config: PlasticityConfig):
        super().__init__(config)
        
        # Initialize component rules
        self.stdp_rule = STDPRule(config) if getattr(config, 'use_stdp', True) else None
        self.homeostatic_rule = HomeostaticPlasticity(config) if getattr(config, 'use_homeostatic', True) else None
        self.metaplasticity_rule = MetaplasticityRule(config) if getattr(config, 'use_metaplasticity', False) else None
        self.competition_rule = SynapticCompetitionRule(config) if getattr(config, 'use_competition', True) else None
        self.reward_rule = EnhancedRewardModulatedSTDP(config) if getattr(config, 'use_reward_modulation', False) else None
        
        # Combination weights
        self.stdp_weight = getattr(config, 'stdp_weight', 1.0)
        self.homeostatic_weight = getattr(config, 'homeostatic_weight', 0.1)
        self.metaplasticity_weight = getattr(config, 'metaplasticity_weight', 0.5)
        self.competition_weight = getattr(config, 'competition_weight', 0.3)
        self.reward_weight = getattr(config, 'reward_weight', 1.0)
        
    def compute_weight_change(
        self,
        pre_activity: float,
        post_activity: float,
        current_weight: float,
        all_weights: Optional[np.ndarray] = None,
        reward: Optional[float] = None,
        **kwargs,
    ) -> float:
        """
        Compute combined weight change from multiple plasticity mechanisms.
        
        Args:
            pre_activity: Presynaptic activity
            post_activity: Postsynaptic activity
            current_weight: Current weight
            all_weights: All weights on the same postsynaptic neuron
            reward: Reward signal (if available)
            
        Returns:
            Combined weight change
        """
        total_change = 0.0
        
        # STDP component
        if self.stdp_rule is not None:
            stdp_change = self.stdp_rule.compute_weight_change(
                pre_activity, post_activity, current_weight, **kwargs
            )
            total_change += stdp_change * self.stdp_weight
        else:
            stdp_change = 0.0
            
        # Homeostatic component
        if self.homeostatic_rule is not None:
            homeostatic_change = self.homeostatic_rule.compute_weight_change(
                pre_activity, post_activity, current_weight, 
                all_weights=all_weights, **kwargs
            )
            total_change += homeostatic_change * self.homeostatic_weight
            
        # Metaplasticity component
        if self.metaplasticity_rule is not None:
            metaplasticity_change = self.metaplasticity_rule.compute_weight_change(
                pre_activity, post_activity, current_weight, **kwargs
            )
            total_change += metaplasticity_change * self.metaplasticity_weight
            
        # Reward modulation component
        if self.reward_rule is not None and reward is not None:
            reward_change = self.reward_rule.compute_weight_change(
                pre_activity, post_activity, current_weight, reward=reward, **kwargs
            )
            total_change += reward_change * self.reward_weight
            
        # Competition component (applied to the combined change)
        if self.competition_rule is not None:
            competition_change = self.competition_rule.compute_weight_change(
                pre_activity, post_activity, current_weight,
                all_weights=all_weights, base_change=total_change, **kwargs
            )
            # Competition rule already includes the base change
            total_change = competition_change * self.competition_weight + total_change * (1 - self.competition_weight)
            
        return total_change
        
    def get_multi_plasticity_state(self) -> Dict[str, Any]:
        """Get state from all component plasticity rules."""
        state = {}
        
        if self.homeostatic_rule is not None:
            state['homeostatic'] = self.homeostatic_rule.get_homeostatic_state()
            
        if self.metaplasticity_rule is not None:
            state['metaplasticity'] = self.metaplasticity_rule.get_metaplastic_state()
            
        if self.competition_rule is not None:
            state['competition'] = self.competition_rule.get_competition_state()
            
        if self.reward_rule is not None:
            state['reward'] = self.reward_rule.get_dopamine_state()
            
        return state


class CustomPlasticityRule(PlasticityRule):
    """Custom user-defined plasticity rule."""

    def __init__(
        self, config: PlasticityConfig, update_function: Optional[Callable] = None
    ):
        """
        Initialize custom plasticity rule.

        Args:
            config: Plasticity configuration
            update_function: Custom weight update function
        """
        super().__init__(config)
        self.update_function = update_function
        self.state = {}  # User-defined state variables

    def set_update_function(self, func: Callable):
        """
        Set custom update function.

        The function should have signature:
        func(pre_activity, post_activity, current_weight, state, config) -> delta_w
        """
        self.update_function = func

    def compute_weight_change(
        self, pre_activity: float, post_activity: float, current_weight: float, **kwargs
    ) -> float:
        """
        Compute custom weight change.

        Args:
            pre_activity: Presynaptic activity
            post_activity: Postsynaptic activity
            current_weight: Current weight

        Returns:
            Weight change
        """
        if self.update_function is None:
            return 0.0

        try:
            delta_w = self.update_function(
                pre_activity,
                post_activity,
                current_weight,
                self.state,
                self.config,
                **kwargs,
            )
            return delta_w
        except Exception as e:
            neuromorphic_logger.log_event(
                "custom_plasticity_error",
                error=str(e),
                pre_activity=pre_activity,
                post_activity=post_activity,
            )
            return 0.0


class PlasticityManager:
    """Manager for multiple plasticity mechanisms."""

    def __init__(self, config: Optional[PlasticityConfig] = None):
        """
        Initialize plasticity manager.

        Args:
            config: Default plasticity configuration
        """
        self.config = config or PlasticityConfig()
        self.rules: Dict[str, PlasticityRule] = {}
        self.active_rules: List[str] = []

        # Initialize default rules
        self._initialize_default_rules()

    def _initialize_default_rules(self):
        """Initialize default plasticity rules."""
        self.rules["stdp"] = STDPRule(self.config)
        self.rules["hebbian"] = HebbianRule(self.config)
        self.rules["bcm"] = BCMRule(self.config)
        self.rules["rstdp"] = RewardModulatedSTDP(self.config)
        self.rules["enhanced_rstdp"] = EnhancedRewardModulatedSTDP(self.config)
        self.rules["triplet_stdp"] = TripletSTDP(self.config)
        self.rules["homeostatic"] = HomeostaticPlasticity(self.config)
        self.rules["metaplasticity"] = MetaplasticityRule(self.config)
        self.rules["competition"] = SynapticCompetitionRule(self.config)
        self.rules["multi_plasticity"] = MultiPlasticityRule(self.config)

    def add_custom_rule(self, name: str, rule: Union[PlasticityRule, Callable]):
        """
        Add a custom plasticity rule.

        Args:
            name: Name for the custom rule
            rule: PlasticityRule instance or update function
        """
        if isinstance(rule, PlasticityRule):
            self.rules[name] = rule
        elif callable(rule):
            self.rules[name] = CustomPlasticityRule(self.config, rule)
        else:
            raise ValueError("Rule must be a PlasticityRule instance or callable")

    def activate_rule(self, name: str):
        """Activate a plasticity rule."""
        if name not in self.rules:
            raise ValueError(f"Unknown plasticity rule: {name}")
        if name not in self.active_rules:
            self.active_rules.append(name)

    def deactivate_rule(self, name: str):
        """Deactivate a plasticity rule."""
        if name in self.active_rules:
            self.active_rules.remove(name)

    def update_weights(
        self,
        weights: np.ndarray,
        pre_activity: np.ndarray,
        post_activity: np.ndarray,
        rule_name: Optional[str] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Update weight matrix using active plasticity rules.

        Args:
            weights: Current weight matrix
            pre_activity: Presynaptic activity
            post_activity: Postsynaptic activity
            rule_name: Specific rule to use (if None, use all active rules)
            **kwargs: Additional parameters for rules

        Returns:
            Updated weight matrix
        """
        if rule_name:
            rules_to_apply = [rule_name] if rule_name in self.rules else []
        else:
            rules_to_apply = self.active_rules

        if not rules_to_apply:
            return weights

        updated_weights = weights.copy()

        for rule_name in rules_to_apply:
            rule = self.rules[rule_name]

            # Apply rule to each synapse
            for i in range(weights.shape[0]):  # Pre
                for j in range(weights.shape[1]):  # Post
                    if weights[i, j] > 0:  # Only update existing connections
                        # Extract specific spike information for this synapse
                        synapse_kwargs = kwargs.copy()
                        if "pre_spike" in kwargs:
                            synapse_kwargs["pre_spike"] = bool(kwargs["pre_spike"][i])
                        if "post_spike" in kwargs:
                            synapse_kwargs["post_spike"] = bool(kwargs["post_spike"][j])

                        updated_weights[i, j] = rule.update_weight(
                            weights[i, j],
                            pre_activity[i],
                            post_activity[j],
                            **synapse_kwargs,
                        )

        return updated_weights

    def set_reward(self, reward: float, context_strength: float = 1.0):
        """Set reward signal for reward-modulated rules."""
        if "rstdp" in self.rules and isinstance(
            self.rules["rstdp"], RewardModulatedSTDP
        ):
            self.rules["rstdp"].set_reward(reward)
            
        if "enhanced_rstdp" in self.rules and isinstance(
            self.rules["enhanced_rstdp"], EnhancedRewardModulatedSTDP
        ):
            self.rules["enhanced_rstdp"].set_reward(reward, context_strength)
            
    def get_dopamine_state(self) -> Optional[Dict[str, float]]:
        """Get dopamine system state from enhanced reward-modulated rules."""
        if "enhanced_rstdp" in self.rules and isinstance(
            self.rules["enhanced_rstdp"], EnhancedRewardModulatedSTDP
        ):
            return self.rules["enhanced_rstdp"].get_dopamine_state()
        return None
        
    def reset_dopamine_system(self):
        """Reset dopamine system in reward-modulated rules."""
        if "enhanced_rstdp" in self.rules and isinstance(
            self.rules["enhanced_rstdp"], EnhancedRewardModulatedSTDP
        ):
            self.rules["enhanced_rstdp"].reset_reward_system()

    def load_config(self, filepath: Union[str, Path], format: str = "yaml"):
        """
        Load configuration from file.

        Args:
            filepath: Path to configuration file
            format: File format ('yaml' or 'json')
        """
        if format == "yaml":
            self.config = PlasticityConfig.from_yaml(filepath)
        elif format == "json":
            self.config = PlasticityConfig.from_json(filepath)
        else:
            raise ValueError(f"Unknown format: {format}")

        # Reinitialize rules with new config
        self._initialize_default_rules()

    def save_config(self, filepath: Union[str, Path], format: str = "yaml"):
        """
        Save configuration to file.

        Args:
            filepath: Path to save configuration
            format: File format ('yaml' or 'json')
        """
        if format == "yaml":
            self.config.to_yaml(filepath)
        elif format == "json":
            self.config.to_json(filepath)
        else:
            raise ValueError(f"Unknown format: {format}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about plasticity rules."""
        stats = {
            "active_rules": self.active_rules,
            "available_rules": list(self.rules.keys()),
            "config": {
                k: v for k, v in self.config.__dict__.items() if not k.startswith("_")
            },
            "weight_histories": {},
        }

        for name, rule in self.rules.items():
            if hasattr(rule, "weight_history") and rule.weight_history:
                stats["weight_histories"][name] = {
                    "mean": np.mean(rule.weight_history),
                    "std": np.std(rule.weight_history),
                    "min": np.min(rule.weight_history),
                    "max": np.max(rule.weight_history),
                    "length": len(rule.weight_history),
                }

        return stats


# Example custom plasticity rule
def example_custom_rule(
    pre_activity, post_activity, current_weight, state, config, **kwargs
):
    """
    Example custom plasticity rule.

    This implements a simple voltage-dependent plasticity rule.
    """
    # Get voltage if provided
    post_voltage = kwargs.get("post_voltage", -65.0)

    # Initialize state if needed
    if "voltage_history" not in state:
        state["voltage_history"] = []

    state["voltage_history"].append(post_voltage)

    # Voltage-dependent plasticity
    if post_voltage > -50.0:  # Depolarized
        delta_w = config.learning_rate * pre_activity * 0.1
    else:  # Hyperpolarized
        delta_w = -config.learning_rate * pre_activity * 0.05

    return delta_w
