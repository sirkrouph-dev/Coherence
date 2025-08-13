"""
Unified neuromorphic programming system with integrated components.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.encoding import MultiModalEncoder
from core.network import EventDrivenSimulator, NetworkBuilder, NeuromorphicNetwork
from core.neuromodulation import (
    AdaptiveLearningController,
    HomeostaticRegulator,
    NeuromodulatorType,
    NeuromodulatoryController,
    RewardSystem,
)
from core.neurons import NeuronFactory
from core.synapses import SynapseFactory


class NeuromorphicSystem:
    """Complete neuromorphic programming system with integrated components."""

    def __init__(self, backend="time_driven"):
        self.network = NeuromorphicNetwork()
        self.encoder = MultiModalEncoder()
        self.modulatory_controller = AdaptiveLearningController()
        self.homeostatic_regulator = HomeostaticRegulator()
        self.reward_system = RewardSystem()
        self.simulator = EventDrivenSimulator()
        self.backend = backend  # 'time_driven' or 'event_driven'

        # Tracking
        self.current_time = 0.0
        self.simulation_history = []

    # Note: MultiModalEncoder doesn't support dynamic encoder addition
    # This functionality would need to be implemented in the encoder class

    def build_network(self, network_config: Dict[str, Any]):
        """Build network from configuration"""
        builder = NetworkBuilder()

        # Create layers
        for layer in network_config["layers"]:
            if layer["type"] == "sensory":
                builder.add_sensory_layer(
                    layer["name"], layer["size"], layer.get("encoding_type", "rate")
                )
            elif layer["type"] == "processing":
                builder.add_processing_layer(
                    layer["name"], layer["size"], layer.get("neuron_type", "adex")
                )
            elif layer["type"] == "motor":
                builder.add_motor_layer(layer["name"], layer["size"])

        # Create connections
        for connection in network_config["connections"]:
            # Extract parameters
            params = connection.get("params", {})
            connection_probability = connection.get("probability", 0.1)

            # Remove probability from params if it exists
            if "probability" in params:
                del params["probability"]

            builder.connect_layers(
                connection["pre_layer"],
                connection["post_layer"],
                connection_type=connection.get("connection_type", "random"),
                synapse_type=connection.get("synapse_type", "stdp"),
                connection_probability=connection_probability,
                **params,
            )

        self.network = builder.build()
        self.simulator.set_network(self.network)

    def encode_input(
        self, inputs: Dict[str, Any], time_window: float = 100.0
    ) -> Dict[str, Any]:
        """Encode multimodal sensory inputs"""
        # MultiModalEncoder.encode only takes inputs, not time_window
        return self.encoder.encode(inputs)

    def run_simulation(
        self, duration: float, dt: float = 0.1, inputs: Optional[Dict[str, Any]] = None
    ):
        """Run simulation with optional sensory input"""
        # Reset system
        self.network.reset()
        self.current_time = 0.0
        self.simulation_history = []
        
        #
        #
        # For now, use time-driven simulation for compatibility
        # TODO: Implement proper event-driven simulation with sensory inputs
        results = self.network.run_simulation(duration, dt)

        return results

    def update_learning(
        self,
        reward: float,
        expected_reward: float,
        positive_events: float = 0.0,
        negative_events: float = 0.0,
        threat_signals: float = 0.0,
        task_difficulty: float = 0.5,
        dt: float = 0.1,
    ):
        """Update neuromodulatory systems based on experience"""
        # Update reward system
        self.reward_system.update(reward, dt)

        # Update neuromodulators
        self.modulatory_controller.update(
            sensory_input=np.array([reward]),  # Simplified representation
            reward=reward,
            expected_reward=expected_reward,
            positive_events=positive_events,
            negative_events=negative_events,
            threat_signals=threat_signals,
            task_difficulty=task_difficulty,
            dt=dt,
        )

        # Apply homeostatic regulation
        scaling_factors = self.homeostatic_regulator.compute_scaling_factors()
        self.homeostatic_regulator.apply_homeostasis(self.network, scaling_factors)

    def get_learning_rate(self) -> float:
        """Get current learning rate with neuromodulation"""
        base_rate = 0.01
        modulation = self.modulatory_controller.get_learning_rate_modulation()
        return base_rate * modulation

    def get_network_state(self) -> Dict[str, Any]:
        """Get comprehensive network state"""
        state = {
            "time": self.current_time,
            "modulators": self.modulatory_controller.get_modulator_levels(),
            "reward_prediction_error": self.reward_system.get_reward_prediction_error(),
            "dopamine_learning_rate": self.modulatory_controller.systems[
                NeuromodulatorType.DOPAMINE
            ].get_learning_rate_modulation(),
            "attention_level": self.modulatory_controller.get_attention_level(),
            "vigilance_level": self.modulatory_controller.get_vigilance_level(),
            "firing_rates": self.homeostatic_regulator.current_firing_rates,
        }
        return state
