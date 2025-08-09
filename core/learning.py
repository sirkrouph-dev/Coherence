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
    
    # Custom rule parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, filepath: Union[str, Path]) -> 'PlasticityConfig':
        """Load configuration from YAML file."""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    @classmethod
    def from_json(cls, filepath: Union[str, Path]) -> 'PlasticityConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def to_yaml(self, filepath: Union[str, Path]):
        """Save configuration to YAML file."""
        data = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    
    def to_json(self, filepath: Union[str, Path]):
        """Save configuration to JSON file."""
        data = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        with open(filepath, 'w') as f:
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
        self,
        pre_activity: float,
        post_activity: float,
        current_weight: float,
        **kwargs
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
        self,
        current_weight: float,
        pre_activity: float,
        post_activity: float,
        **kwargs
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
        **kwargs
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
                delta_w *= (self.config.weight_max - current_weight) / self.config.weight_max
            else:  # LTD
                delta_w *= current_weight / self.config.weight_max
                
        return delta_w * self.config.learning_rate


class HebbianRule(PlasticityRule):
    """Classical Hebbian learning rule: 'Cells that fire together, wire together'."""
    
    def compute_weight_change(
        self,
        pre_activity: float,
        post_activity: float,
        current_weight: float,
        **kwargs
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
            delta_w *= (self.config.weight_max - current_weight) / self.config.weight_max
        else:
            # Decay term for stability
            delta_w = -self.config.learning_rate * (1 - self.config.hebbian_decay) * current_weight
            
        return delta_w


class BCMRule(PlasticityRule):
    """Bienenstock-Cooper-Munro (BCM) learning rule with sliding threshold."""
    
    def __init__(self, config: PlasticityConfig):
        super().__init__(config)
        self.sliding_threshold = config.bcm_threshold
        self.activity_history = []
        
    def compute_weight_change(
        self,
        pre_activity: float,
        post_activity: float,
        current_weight: float,
        **kwargs
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
        mean_activity = np.mean(self.activity_history) if self.activity_history else post_activity
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
        **kwargs
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
            pre_activity, post_activity, current_weight,
            dt, pre_spike, post_spike, **kwargs
        )
        
        # Update eligibility trace
        self.eligibility_trace *= self.config.reward_decay
        self.eligibility_trace += stdp_change
        
        # Update dopamine trace
        self.dopamine_trace *= np.exp(-dt / self.config.dopamine_time_constant)
        
        # Modulate by reward/dopamine
        modulated_change = self.eligibility_trace * self.dopamine_trace * self.config.reward_sensitivity
        
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
        **kwargs
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
    """Homeostatic plasticity to maintain stable firing rates."""
    
    def __init__(self, config: PlasticityConfig):
        super().__init__(config)
        self.firing_rate_estimate = 0.0
        self.time_window = []
        
    def compute_weight_change(
        self,
        pre_activity: float,
        post_activity: float,
        current_weight: float,
        post_spike: bool = False,
        dt: float = 1.0,
        **kwargs
    ) -> float:
        """
        Compute homeostatic weight change.
        
        Args:
            pre_activity: Presynaptic activity
            post_activity: Postsynaptic activity
            current_weight: Current weight
            post_spike: Whether postsynaptic neuron spiked
            dt: Time step
            
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
            
        # Calculate current firing rate
        current_rate = sum(self.time_window) / (len(self.time_window) * dt / 1000.0) if self.time_window else 0.0
        
        # Homeostatic adjustment
        rate_error = self.config.target_rate - current_rate
        delta_w = (self.config.learning_rate * rate_error * current_weight / 
                  self.config.homeostatic_time_constant)
        
        return delta_w


class CustomPlasticityRule(PlasticityRule):
    """Custom user-defined plasticity rule."""
    
    def __init__(self, config: PlasticityConfig, update_function: Optional[Callable] = None):
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
        self,
        pre_activity: float,
        post_activity: float,
        current_weight: float,
        **kwargs
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
                pre_activity, post_activity, current_weight,
                self.state, self.config, **kwargs
            )
            return delta_w
        except Exception as e:
            neuromorphic_logger.log_event(
                "custom_plasticity_error",
                error=str(e),
                pre_activity=pre_activity,
                post_activity=post_activity
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
        self.rules['stdp'] = STDPRule(self.config)
        self.rules['hebbian'] = HebbianRule(self.config)
        self.rules['bcm'] = BCMRule(self.config)
        self.rules['rstdp'] = RewardModulatedSTDP(self.config)
        self.rules['triplet_stdp'] = TripletSTDP(self.config)
        self.rules['homeostatic'] = HomeostaticPlasticity(self.config)
        
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
        **kwargs
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
                        if 'pre_spike' in kwargs:
                            synapse_kwargs['pre_spike'] = bool(kwargs['pre_spike'][i])
                        if 'post_spike' in kwargs:
                            synapse_kwargs['post_spike'] = bool(kwargs['post_spike'][j])
                        
                        updated_weights[i, j] = rule.update_weight(
                            weights[i, j],
                            pre_activity[i],
                            post_activity[j],
                            **synapse_kwargs
                        )
                        
        return updated_weights
        
    def set_reward(self, reward: float):
        """Set reward signal for reward-modulated rules."""
        if 'rstdp' in self.rules and isinstance(self.rules['rstdp'], RewardModulatedSTDP):
            self.rules['rstdp'].set_reward(reward)
            
    def load_config(self, filepath: Union[str, Path], format: str = 'yaml'):
        """
        Load configuration from file.
        
        Args:
            filepath: Path to configuration file
            format: File format ('yaml' or 'json')
        """
        if format == 'yaml':
            self.config = PlasticityConfig.from_yaml(filepath)
        elif format == 'json':
            self.config = PlasticityConfig.from_json(filepath)
        else:
            raise ValueError(f"Unknown format: {format}")
            
        # Reinitialize rules with new config
        self._initialize_default_rules()
        
    def save_config(self, filepath: Union[str, Path], format: str = 'yaml'):
        """
        Save configuration to file.
        
        Args:
            filepath: Path to save configuration
            format: File format ('yaml' or 'json')
        """
        if format == 'yaml':
            self.config.to_yaml(filepath)
        elif format == 'json':
            self.config.to_json(filepath)
        else:
            raise ValueError(f"Unknown format: {format}")
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about plasticity rules."""
        stats = {
            'active_rules': self.active_rules,
            'available_rules': list(self.rules.keys()),
            'config': {k: v for k, v in self.config.__dict__.items() if not k.startswith('_')},
            'weight_histories': {}
        }
        
        for name, rule in self.rules.items():
            if hasattr(rule, 'weight_history') and rule.weight_history:
                stats['weight_histories'][name] = {
                    'mean': np.mean(rule.weight_history),
                    'std': np.std(rule.weight_history),
                    'min': np.min(rule.weight_history),
                    'max': np.max(rule.weight_history),
                    'length': len(rule.weight_history)
                }
                
        return stats


# Example custom plasticity rule
def example_custom_rule(pre_activity, post_activity, current_weight, state, config, **kwargs):
    """
    Example custom plasticity rule.
    
    This implements a simple voltage-dependent plasticity rule.
    """
    # Get voltage if provided
    post_voltage = kwargs.get('post_voltage', -65.0)
    
    # Initialize state if needed
    if 'voltage_history' not in state:
        state['voltage_history'] = []
        
    state['voltage_history'].append(post_voltage)
    
    # Voltage-dependent plasticity
    if post_voltage > -50.0:  # Depolarized
        delta_w = config.learning_rate * pre_activity * 0.1
    else:  # Hyperpolarized
        delta_w = -config.learning_rate * pre_activity * 0.05
        
    return delta_w
