"""
Neuromodulatory systems for the neuromorphic programming system.
Implements reward-based learning, homeostatic regulation, and behavioral state control.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class NeuromodulatorType(Enum):
    """Types of neuromodulators."""

    DOPAMINE = "dopamine"
    SEROTONIN = "serotonin"
    ACETYLCHOLINE = "acetylcholine"
    NOREPINEPHRINE = "norepinephrine"


class NeuromodulatorySystem:
    """Base class for neuromodulatory systems."""

    def __init__(self, modulator_type: NeuromodulatorType):
        """
        Initialize neuromodulatory system.

        Args:
            modulator_type: Type of neuromodulator
        """
        self.modulator_type = modulator_type
        self.current_level = 0.0
        self.target_level = 0.0
        self.decay_rate = 0.95
        self.learning_rate = 0.01

    def update(self, input_signal: float, dt: float):
        """
        Update neuromodulator level.

        Args:
            input_signal: Input signal for modulation
            dt: Time step
        """
        # Update level based on input signal
        self.current_level += (
            self.learning_rate * (input_signal - self.current_level) * dt
        )

        # Apply decay
        self.current_level *= self.decay_rate

        # Clamp to [0, 1]
        self.current_level = np.clip(self.current_level, 0.0, 1.0)

    def get_level(self) -> float:
        """Get current neuromodulator level."""
        return self.current_level

    def reset(self):
        """Reset neuromodulator to initial state."""
        self.current_level = 0.0


class DopaminergicSystem(NeuromodulatorySystem):
    """
    Dopaminergic system for reward prediction error.

    Implements temporal difference learning and reinforcement learning.
    """

    def __init__(self, learning_rate: float = 0.01, discount_factor: float = 0.9):
        """
        Initialize dopaminergic system.

        Args:
            learning_rate: Learning rate for dopamine updates
            discount_factor: Temporal discount factor
        """
        super().__init__(NeuromodulatorType.DOPAMINE)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.reward_prediction = 0.0
        self.reward_history = []

    def compute_reward_prediction_error(
        self, reward: float, expected_reward: float
    ) -> float:
        """
        Compute reward prediction error.

        Args:
            reward: Actual reward received
            expected_reward: Expected reward

        Returns:
            Reward prediction error
        """
        prediction_error = reward - expected_reward
        return prediction_error

    def update(self, reward: float, expected_reward: float, dt: float):
        """
        Update dopamine level based on reward prediction error.

        Args:
            reward: Actual reward received
            expected_reward: Expected reward
            dt: Time step
        """
        # Compute reward prediction error
        prediction_error = self.compute_reward_prediction_error(reward, expected_reward)

        # Update dopamine level (positive prediction error increases dopamine)
        dopamine_signal = np.tanh(prediction_error)
        super().update(dopamine_signal, dt)

        # Update reward prediction
        self.reward_prediction = expected_reward + self.learning_rate * prediction_error

        # Store reward history
        self.reward_history.append(
            {
                "reward": reward,
                "expected_reward": expected_reward,
                "prediction_error": prediction_error,
                "dopamine_level": self.current_level,
            }
        )

    def get_learning_rate_modulation(self) -> float:
        """Get learning rate modulation factor."""
        # Dopamine modulates learning rate
        return 1.0 + 2.0 * self.current_level


class SerotonergicSystem(NeuromodulatorySystem):
    """
    Serotonergic system for mood and behavioral state regulation.

    Implements mood-dependent learning and behavioral flexibility.
    """

    def __init__(self, mood_decay_rate: float = 0.99):
        """
        Initialize serotonergic system.

        Args:
            mood_decay_rate: Decay rate for mood state
        """
        super().__init__(NeuromodulatorType.SEROTONIN)
        self.mood_decay_rate = mood_decay_rate
        self.mood_state = 0.5  # Neutral mood
        self.stress_level = 0.0

    def update_mood(self, positive_events: float, negative_events: float, dt: float):
        """
        Update mood state based on events.

        Args:
            positive_events: Number/strength of positive events
            negative_events: Number/strength of negative events
            dt: Time step
        """
        # Update mood based on event balance
        mood_change = (positive_events - negative_events) * dt
        self.mood_state += mood_change

        # Apply mood decay
        self.mood_state *= self.mood_decay_rate

        # Clamp mood to [0, 1]
        self.mood_state = np.clip(self.mood_state, 0.0, 1.0)

        # Update serotonin level based on mood
        serotonin_signal = self.mood_state - 0.5  # Center around neutral
        super().update(serotonin_signal, dt)

    def get_behavioral_flexibility(self) -> float:
        """Get behavioral flexibility factor."""
        # Serotonin increases behavioral flexibility
        return 0.5 + self.current_level


class CholinergicSystem(NeuromodulatorySystem):
    """
    Cholinergic system for attention and learning rate modulation.

    Implements attention mechanisms and learning rate control.
    """

    def __init__(self, attention_threshold: float = 0.1):
        """
        Initialize cholinergic system.

        Args:
            attention_threshold: Threshold for attention activation
        """
        super().__init__(NeuromodulatorType.ACETYLCHOLINE)
        self.attention_threshold = attention_threshold
        self.attention_state = 0.0
        self.novelty_detector = 0.0

    def update_attention(
        self, sensory_input: np.ndarray, expected_input: np.ndarray, dt: float
    ):
        """
        Update attention based on sensory input.

        Args:
            sensory_input: Current sensory input
            expected_input: Expected sensory input
            dt: Time step
        """
        # Compute novelty (difference from expected)
        novelty = np.mean(np.abs(sensory_input - expected_input))

        # Update novelty detector
        self.novelty_detector = 0.9 * self.novelty_detector + 0.1 * novelty

        # Update attention state
        if novelty > self.attention_threshold:
            self.attention_state = min(1.0, self.attention_state + dt)
        else:
            self.attention_state = max(0.0, self.attention_state - dt)

        # Update acetylcholine level
        acetylcholine_signal = self.attention_state
        super().update(acetylcholine_signal, dt)

    def get_attention_level(self) -> float:
        """Get current attention level."""
        return self.attention_state

    def get_learning_rate_modulation(self) -> float:
        """Get learning rate modulation factor."""
        # Acetylcholine modulates learning rate based on attention
        return 1.0 + self.current_level


class NoradrenergicSystem(NeuromodulatorySystem):
    """
    Noradrenergic system for arousal and vigilance.

    Implements arousal-dependent processing and vigilance control.
    """

    def __init__(self, arousal_decay_rate: float = 0.98):
        """
        Initialize noradrenergic system.

        Args:
            arousal_decay_rate: Decay rate for arousal state
        """
        super().__init__(NeuromodulatorType.NOREPINEPHRINE)
        self.arousal_decay_rate = arousal_decay_rate
        self.arousal_state = 0.5  # Moderate arousal
        self.threat_level = 0.0

    def update_arousal(self, threat_signals: float, task_difficulty: float, dt: float):
        """
        Update arousal based on environmental factors.

        Args:
            threat_signals: Level of threat signals
            task_difficulty: Difficulty of current task
            dt: Time step
        """
        # Update threat level
        self.threat_level = 0.9 * self.threat_level + 0.1 * threat_signals

        # Compute arousal signal
        arousal_signal = 0.5 * self.threat_level + 0.3 * task_difficulty

        # Update arousal state
        self.arousal_state += (arousal_signal - self.arousal_state) * dt
        self.arousal_state *= self.arousal_decay_rate

        # Clamp arousal to [0, 1]
        self.arousal_state = np.clip(self.arousal_state, 0.0, 1.0)

        # Update norepinephrine level
        norepinephrine_signal = self.arousal_state
        super().update(norepinephrine_signal, dt)

    def get_vigilance_level(self) -> float:
        """Get current vigilance level."""
        return self.arousal_state

    def get_processing_gain(self) -> float:
        """Get processing gain factor."""
        # Norepinephrine increases processing gain
        return 1.0 + 0.5 * self.current_level


class NeuromodulatoryController:
    """Central controller for all neuromodulatory systems."""

    def __init__(self):
        """Initialize neuromodulatory controller."""
        self.systems = {
            NeuromodulatorType.DOPAMINE: DopaminergicSystem(),
            NeuromodulatorType.SEROTONIN: SerotonergicSystem(),
            NeuromodulatorType.ACETYLCHOLINE: CholinergicSystem(),
            NeuromodulatorType.NOREPINEPHRINE: NoradrenergicSystem(),
        }
        self.global_learning_rate = 0.01

    def update(
        self,
        sensory_input: np.ndarray,
        reward: float,
        expected_reward: float,
        positive_events: float = 0.0,
        negative_events: float = 0.0,
        threat_signals: float = 0.0,
        task_difficulty: float = 0.5,
        dt: float = 0.1,
    ):
        """
        Update all neuromodulatory systems.

        Args:
            sensory_input: Current sensory input
            reward: Current reward
            expected_reward: Expected reward
            positive_events: Positive events count/strength
            negative_events: Negative events count/strength
            threat_signals: Threat signal level
            task_difficulty: Current task difficulty
            dt: Time step
        """
        # Update dopaminergic system
        self.systems[NeuromodulatorType.DOPAMINE].update(reward, expected_reward, dt)

        # Update serotonergic system
        self.systems[NeuromodulatorType.SEROTONIN].update_mood(
            positive_events, negative_events, dt
        )

        # Update cholinergic system (simplified - using reward as novelty)
        novelty = abs(reward - expected_reward)
        self.systems[NeuromodulatorType.ACETYLCHOLINE].update_attention(
            np.array([novelty]), np.array([0.0]), dt
        )

        # Update noradrenergic system
        self.systems[NeuromodulatorType.NOREPINEPHRINE].update_arousal(
            threat_signals, task_difficulty, dt
        )

    def get_learning_rate_modulation(self) -> float:
        """Get overall learning rate modulation."""
        # Combine effects of dopamine and acetylcholine
        dopamine_mod = self.systems[
            NeuromodulatorType.DOPAMINE
        ].get_learning_rate_modulation()
        acetylcholine_mod = self.systems[
            NeuromodulatorType.ACETYLCHOLINE
        ].get_learning_rate_modulation()

        return (dopamine_mod + acetylcholine_mod) / 2.0

    def get_behavioral_flexibility(self) -> float:
        """Get behavioral flexibility factor."""
        return self.systems[NeuromodulatorType.SEROTONIN].get_behavioral_flexibility()

    def get_attention_level(self) -> float:
        """Get attention level."""
        return self.systems[NeuromodulatorType.ACETYLCHOLINE].get_attention_level()

    def get_vigilance_level(self) -> float:
        """Get vigilance level."""
        return self.systems[NeuromodulatorType.NOREPINEPHRINE].get_vigilance_level()

    def get_processing_gain(self) -> float:
        """Get processing gain factor."""
        return self.systems[NeuromodulatorType.NOREPINEPHRINE].get_processing_gain()

    def get_modulator_levels(self) -> Dict[NeuromodulatorType, float]:
        """Get all neuromodulator levels."""
        return {
            mod_type: system.get_level() for mod_type, system in self.systems.items()
        }

    def reset(self):
        """Reset all neuromodulatory systems."""
        for system in self.systems.values():
            system.reset()


class HomeostaticRegulator:
    """Homeostatic regulation system for network stability."""

    def __init__(self, target_firing_rate: float = 10.0, adaptation_rate: float = 0.01):
        """
        Initialize homeostatic regulator.

        Args:
            target_firing_rate: Target firing rate (Hz)
            adaptation_rate: Rate of homeostatic adaptation
        """
        self.target_firing_rate = target_firing_rate
        self.adaptation_rate = adaptation_rate
        self.current_firing_rates = {}
        self.scaling_factors = {}

    def update_firing_rates(
        self, layer_name: str, spike_times: List[List[float]], time_window: float
    ):
        """
        Update firing rates for a layer.

        Args:
            layer_name: Name of the layer
            spike_times: Spike times for each neuron
            time_window: Time window for rate calculation (ms)
        """
        rates = []
        for neuron_spikes in spike_times:
            # Count spikes in time window
            spike_count = len([t for t in neuron_spikes if t <= time_window])
            rate = (spike_count / time_window) * 1000.0  # Convert to Hz
            rates.append(rate)

        self.current_firing_rates[layer_name] = rates

    def compute_scaling_factors(self) -> Dict[str, float]:
        """
        Compute homeostatic scaling factors.

        Returns:
            Dictionary of scaling factors for each layer
        """
        scaling_factors = {}

        for layer_name, rates in self.current_firing_rates.items():
            if rates:
                mean_rate = np.mean(rates)
                if mean_rate > 0:
                    # Compute scaling factor to approach target rate
                    scaling_factor = self.target_firing_rate / mean_rate
                    scaling_factor = np.clip(scaling_factor, 0.1, 10.0)  # Limit scaling

                    # Smooth adaptation
                    if layer_name in self.scaling_factors:
                        old_factor = self.scaling_factors[layer_name]
                        scaling_factor = (
                            1 - self.adaptation_rate
                        ) * old_factor + self.adaptation_rate * scaling_factor

                    scaling_factors[layer_name] = scaling_factor
                else:
                    scaling_factors[layer_name] = 1.0
            else:
                scaling_factors[layer_name] = 1.0

        self.scaling_factors = scaling_factors
        return scaling_factors

    def apply_homeostasis(self, network, scaling_factors: Dict[str, float]):
        """
        Apply homeostatic scaling to network weights.

        Args:
            network: Neuromorphic network
            scaling_factors: Scaling factors for each layer
        """
        for layer_name, scaling_factor in scaling_factors.items():
            if layer_name in network.layers:
                # Apply scaling to synaptic weights
                for (pre_name, post_name), connection in network.connections.items():
                    if post_name == layer_name and connection.synapse_population:
                        for synapse in connection.synapse_population.synapses.values():
                            synapse.weight *= scaling_factor

    def reset(self):
        """Reset homeostatic regulator."""
        self.current_firing_rates.clear()
        self.scaling_factors.clear()


class RewardSystem:
    """Reward system for reinforcement learning."""

    def __init__(self, reward_decay: float = 0.95):
        """
        Initialize reward system.

        Args:
            reward_decay: Decay rate for reward signals
        """
        self.reward_decay = reward_decay
        self.current_reward = 0.0
        self.expected_reward = 0.0
        self.reward_history = []

    def compute_reward(self, action: Any, outcome: Any, target: Any) -> float:
        """
        Compute reward based on action and outcome.

        Args:
            action: Action taken
            outcome: Outcome achieved
            target: Target outcome

        Returns:
            Reward value
        """
        # Simple reward computation - can be customized
        if isinstance(outcome, (int, float)) and isinstance(target, (int, float)):
            # Numeric outcome
            error = abs(outcome - target)
            reward = 1.0 / (1.0 + error)
        elif isinstance(outcome, np.ndarray) and isinstance(target, np.ndarray):
            # Array outcome
            error = np.mean(np.abs(outcome - target))
            reward = 1.0 / (1.0 + error)
        else:
            # Binary outcome
            reward = 1.0 if outcome == target else 0.0

        return reward

    def update(self, reward: float, dt: float):
        """
        Update reward system.

        Args:
            reward: Current reward
            dt: Time step
        """
        self.current_reward = reward
        self.expected_reward = self.expected_reward * self.reward_decay + reward * (
            1 - self.reward_decay
        )

        self.reward_history.append(
            {
                "reward": reward,
                "expected_reward": self.expected_reward,
                "prediction_error": reward - self.expected_reward,
            }
        )

    def get_reward_prediction_error(self) -> float:
        """Get current reward prediction error."""
        return self.current_reward - self.expected_reward

    def reset(self):
        """Reset reward system."""
        self.current_reward = 0.0
        self.expected_reward = 0.0
        self.reward_history.clear()


class AdaptiveLearningController(NeuromodulatoryController):
    """Extended neuromodulatory controller with adaptive learning rules."""

    def __init__(self):
        super().__init__()
        self.learning_rates = {}

    def update_learning_rates(self, network_info: Dict[str, Any]):
        """Dynamically adjust learning rates based on network state"""
        dopamine = self.systems[NeuromodulatorType.DOPAMINE].current_level
        ach = self.systems[NeuromodulatorType.ACETYLCHOLINE].current_level

        # Base learning rate with modulation
        base_rate = 0.01
        lr = base_rate * (1.0 + 2.0 * dopamine) * (1.0 + ach)

        # Apply to all connections
        for conn_name, conn_info in network_info["connections"].items():
            self.learning_rates[conn_name] = lr

        return lr

    def apply_learning(self, network):
        """Apply adaptive learning to network synapses"""
        for connection in network.connections.values():
            if connection.synapse_population:
                for synapse in connection.synapse_population.synapses.values():
                    if hasattr(synapse, "update_neuromodulator"):
                        synapse.update_neuromodulator(
                            self.systems[NeuromodulatorType.DOPAMINE].current_level
                        )
