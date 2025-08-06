"""
High-level API for the neuromorphic programming system.
Provides easy-to-use interfaces for building and simulating neuromorphic networks.
"""

import time
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from core.encoding import (CochlearEncoder, MultiModalEncoder, RateEncoder,
                           RetinalEncoder, SomatosensoryEncoder)
from core.logging_utils import (TrainingTracker, neuromorphic_logger,
                                trace_function)
from core.network import (EventDrivenSimulator, NetworkBuilder,
                          NeuromorphicNetwork)
from core.neuromodulation import (HomeostaticRegulator,
                                  NeuromodulatoryController, RewardSystem)
from core.neurons import NeuronFactory
from core.synapses import SynapseFactory, SynapseType


class NeuromorphicAPI:
    """High-level API for neuromorphic programming."""

    def __init__(self):
        """Initialize neuromorphic API."""
        neuromorphic_logger.log_system_event("API_INIT", {"status": "initializing"})

        self.network = None
        self.simulator = EventDrivenSimulator()
        self.neuromodulatory_controller = NeuromodulatoryController()
        self.homeostatic_regulator = HomeostaticRegulator()
        self.reward_system = RewardSystem()
        self.visualization_tools = NeuromorphicVisualizer()
        self.training_tracker = TrainingTracker()

        neuromorphic_logger.log_system_event("API_INIT", {"status": "complete"})

    def create_network(self) -> "NeuromorphicAPI":
        """Create a new neuromorphic network."""
        self.network = NeuromorphicNetwork()
        self.simulator.set_network(self.network)
        return self

    def add_sensory_layer(
        self, name: str, size: int, encoding_type: str = "rate"
    ) -> "NeuromorphicAPI":
        """
        Add a sensory input layer.

        Args:
            name: Layer name
            size: Number of neurons
            encoding_type: Type of encoding ("rate", "retinal", "cochlear", "somatosensory")

        Returns:
            Self for method chaining
        """
        if self.network is None:
            self.create_network()

        if encoding_type == "retinal":
            neuron_type = "lif"  # LIF for sensory encoding
        elif encoding_type in ["cochlear", "somatosensory"]:
            neuron_type = "lif"
        else:
            neuron_type = "lif"

        self.network.add_layer(name, size, neuron_type)
        return self

    def add_processing_layer(
        self, name: str, size: int, neuron_type: str = "adex"
    ) -> "NeuromorphicAPI":
        """
        Add a processing layer.

        Args:
            name: Layer name
            size: Number of neurons
            neuron_type: Type of neurons ("adex", "hh", "lif")

        Returns:
            Self for method chaining
        """
        if self.network is None:
            self.create_network()

        self.network.add_layer(name, size, neuron_type)
        return self

    def add_motor_layer(self, name: str, size: int) -> "NeuromorphicAPI":
        """
        Add a motor output layer.

        Args:
            name: Layer name
            size: Number of neurons

        Returns:
            Self for method chaining
        """
        if self.network is None:
            self.create_network()

        self.network.add_layer(name, size, neuron_type="lif")
        return self

    def connect_layers(
        self,
        pre_layer: str,
        post_layer: str,
        connection_type: str = "random",
        synapse_type: str = "stdp",
        connection_probability: float = 0.1,
        **kwargs,
    ) -> "NeuromorphicAPI":
        """
        Connect layers with specified pattern and plasticity rules.

        Args:
            pre_layer: Name of presynaptic layer
            post_layer: Name of postsynaptic layer
            connection_type: Type of connection ("random", "feedforward", "lateral", "feedback")
            synapse_type: Type of synapses ("stdp", "stp", "neuromodulatory")
            connection_probability: Probability of connection between neurons
            **kwargs: Additional parameters for synapse models

        Returns:
            Self for method chaining
        """
        if self.network is None:
            raise ValueError("No network created. Call create_network() first.")

        # Set connection probability based on type
        if connection_type == "feedforward":
            connection_probability = 0.3
        elif connection_type == "lateral":
            connection_probability = 0.1
        elif connection_type == "feedback":
            connection_probability = 0.2

        self.network.connect_layers(
            pre_layer, post_layer, synapse_type, connection_probability, **kwargs
        )
        return self

    def add_learning_rule(
        self, pre_layer: str, post_layer: str, learning_type: str = "stdp"
    ) -> "NeuromorphicAPI":
        """
        Add learning rule to connection.

        Args:
            pre_layer: Name of presynaptic layer
            post_layer: Name of postsynaptic layer
            learning_type: Type of learning rule ("stdp", "neuromodulatory")

        Returns:
            Self for method chaining
        """
        # Learning rules are already configured in synapse types
        # This method is for future extensibility
        return self

    def run_simulation(
        self,
        duration: float,
        dt: float = 0.1,
        external_inputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run network simulation.

        Args:
            duration: Simulation duration in milliseconds
            dt: Time step in milliseconds
            external_inputs: External inputs for sensory layers

        Returns:
            Simulation results
        """
        if self.network is None:
            raise ValueError("No network created. Call create_network() first.")

        # Add external inputs if provided
        if external_inputs:
            self._add_external_inputs(external_inputs)

        # Run simulation
        results = self.network.run_simulation(duration, dt)

        # Update neuromodulatory systems
        self._update_neuromodulatory_systems(results)

        return results

    def _add_external_inputs(self, external_inputs: Dict[str, Any]):
        """Add external inputs to the network."""
        for layer_name, input_data in external_inputs.items():
            if layer_name in self.network.layers:
                # Convert input data to spike events
                if isinstance(input_data, (list, tuple)):
                    for neuron_id, spike_time in input_data:
                        if neuron_id < self.network.layers[layer_name].size:
                            self.simulator.add_spike_event(
                                neuron_id, layer_name, spike_time
                            )

    def _update_neuromodulatory_systems(self, simulation_results: Dict[str, Any]):
        """Update neuromodulatory systems based on simulation results."""
        # Extract relevant information from simulation results
        # This is a simplified update - in practice, you'd extract more detailed
        # information

        # Update homeostatic regulator
        for layer_name, spike_times in simulation_results.get(
            "layer_spike_times", {}
        ).items():
            self.homeostatic_regulator.update_firing_rates(
                layer_name, spike_times, simulation_results.get("duration", 100.0)
            )

    @trace_function
    def train_sensorimotor_system(
        self,
        training_data: List[Dict[str, Any]],
        epochs: int = 100,
        trial_duration: float = 100.0,
    ) -> Dict[str, Any]:
        """
        Train a sensorimotor system on provided data.

        Args:
            training_data: List of training trials
            epochs: Number of training epochs
            trial_duration: Duration of each trial (ms)

        Returns:
            Training results
        """
        neuromorphic_logger.log_system_event(
            "TRAINING_START",
            {
                "epochs": epochs,
                "num_trials": len(training_data),
                "trial_duration": trial_duration,
            },
        )

        training_history = []
        start_time = time.time()

        for epoch in range(epochs):
            neuromorphic_logger.logger.info(f"Starting epoch {epoch}/{epochs}")
            epoch_rewards = []
            epoch_start_time = time.time()

            for trial_idx, trial in enumerate(training_data):
                neuromorphic_logger.logger.debug(
                    f"Processing trial {trial_idx + 1}/{len(training_data)} in epoch {epoch}"
                )

                # Present sensory input
                sensory_inputs = self._encode_sensory_inputs(trial)
                neuromorphic_logger.logger.debug(
                    f"Encoded sensory inputs: {list(sensory_inputs.keys())}"
                )

                # Run simulation
                simulation_start = time.time()
                results = self.run_simulation(
                    trial_duration, external_inputs=sensory_inputs
                )
                simulation_time = time.time() - simulation_start

                neuromorphic_logger.log_performance_metrics(
                    {
                        "simulation_time": simulation_time,
                        "trial": trial_idx,
                        "epoch": epoch,
                    }
                )

                # Compute reward
                motor_output = self._extract_motor_output(results)
                reward = self.reward_system.compute_reward(
                    motor_output, trial.get("outcome"), trial.get("target")
                )
                epoch_rewards.append(reward)

                # Log training step
                neuromodulator_levels = (
                    self.neuromodulatory_controller.get_modulator_levels()
                )
                network_info = self.network.get_network_info()

                neuromorphic_logger.log_training_step(
                    epoch=epoch,
                    trial=trial_idx,
                    reward=reward,
                    neuromodulator_levels=neuromodulator_levels,
                    network_info=network_info,
                )

                # Track trial details
                self.training_tracker.log_training_trial(
                    epoch=epoch,
                    trial=trial_idx,
                    reward=reward,
                    sensory_inputs=sensory_inputs,
                    motor_output=motor_output,
                )

                # Update neuromodulatory systems
                self.neuromodulatory_controller.update(
                    sensory_inputs, reward, self.reward_system.expected_reward
                )

                # Update reward system
                self.reward_system.update(reward, 0.1)

            # Apply homeostatic regulation
            neuromorphic_logger.logger.debug("Applying homeostatic regulation")
            scaling_factors = self.homeostatic_regulator.compute_scaling_factors()
            self.homeostatic_regulator.apply_homeostasis(self.network, scaling_factors)

            # Log epoch summary
            avg_reward = np.mean(epoch_rewards)
            epoch_time = time.time() - epoch_start_time

            neuromodulator_levels = (
                self.neuromodulatory_controller.get_modulator_levels()
            )
            self.training_tracker.log_training_epoch(
                epoch, epoch_rewards, neuromodulator_levels
            )

            training_history.append(
                {
                    "epoch": epoch,
                    "average_reward": avg_reward,
                    "epoch_time": epoch_time,
                    "neuromodulator_levels": neuromodulator_levels,
                }
            )

            neuromorphic_logger.logger.info(
                f"Epoch {epoch} completed: Avg reward = {avg_reward:.4f}, Time = {epoch_time:.2f}s"
            )

        total_training_time = time.time() - start_time
        training_summary = self.training_tracker.get_training_summary()

        neuromorphic_logger.log_system_event(
            "TRAINING_COMPLETE",
            {"total_time": total_training_time, "training_summary": training_summary},
        )

        return {
            "training_history": training_history,
            "final_network_state": self.network.get_network_info(),
            "training_summary": training_summary,
        }

    def _encode_sensory_inputs(
        self, trial: Dict[str, Any]
    ) -> Dict[str, List[Tuple[int, float]]]:
        """Encode sensory inputs from trial data."""
        sensory_inputs = {}

        for modality, data in trial.items():
            if modality in ["visual", "auditory", "tactile"]:
                # Create appropriate encoder
                if modality == "visual":
                    encoder = RetinalEncoder()
                elif modality == "auditory":
                    encoder = CochlearEncoder()
                elif modality == "tactile":
                    encoder = SomatosensoryEncoder()

                # Encode data
                spikes = encoder.encode(data, time_window=100.0)
                sensory_inputs[modality] = spikes

        return sensory_inputs

    def _extract_motor_output(self, simulation_results: Dict[str, Any]) -> np.ndarray:
        """Extract motor output from simulation results."""
        # Find motor layer spike times
        motor_spikes = None
        for layer_name, spike_times in simulation_results.get(
            "layer_spike_times", {}
        ).items():
            if "motor" in layer_name.lower():
                motor_spikes = spike_times
                break

        if motor_spikes is None:
            return np.zeros(10)  # Default motor output

        # Convert spike times to motor output
        motor_output = np.zeros(len(motor_spikes))
        for i, spikes in enumerate(motor_spikes):
            motor_output[i] = len(spikes)  # Simple rate-based output

        return motor_output

    def get_network_info(self) -> Dict[str, Any]:
        """Get information about the current network."""
        if self.network is None:
            return {}
        return self.network.get_network_info()

    def reset(self):
        """Reset the entire system."""
        if self.network:
            self.network.reset()
        self.simulator.reset()
        self.neuromodulatory_controller.reset()
        self.homeostatic_regulator.reset()
        self.reward_system.reset()

    def visualize_network(self, results: Dict[str, Any]):
        """Visualize network activity and results."""
        self.visualization_tools.plot_network_activity(results)

    def save_network(self, filename: str):
        """Save network configuration to file."""
        # Implementation for saving network state
        pass

    def load_network(self, filename: str):
        """Load network configuration from file."""
        # Implementation for loading network state
        pass


class NeuromorphicVisualizer:
    """Visualization tools for neuromorphic networks."""

    def __init__(self):
        """Initialize visualizer."""
        self.figures = {}

    def plot_spike_raster(
        self,
        spike_data: Dict[str, List[List[float]]],
        title: str = "Spike Raster",
        figsize: Tuple[int, int] = (12, 8),
    ):
        """Plot spike raster from simulation data."""
        plt.figure(figsize=figsize)

        neuron_offset = 0
        for layer_name, spike_times in spike_data.items():
            for neuron_id, spikes in enumerate(spike_times):
                if spikes:  # Only plot neurons that spiked
                    plt.plot(
                        spikes,
                        [neuron_id + neuron_offset] * len(spikes),
                        "k.",
                        markersize=1,
                        alpha=0.7,
                    )
            neuron_offset += len(spike_times)

        plt.xlabel("Time (ms)")
        plt.ylabel("Neuron ID")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_weight_evolution(
        self,
        weight_history: Dict[Tuple[int, int], List[float]],
        title: str = "Synaptic Weight Evolution",
        figsize: Tuple[int, int] = (12, 6),
    ):
        """Plot evolution of synaptic weights over time."""
        plt.figure(figsize=figsize)

        for synapse_id, weights in weight_history.items():
            plt.plot(weights, label=f"Synapse {synapse_id}", alpha=0.7)

        plt.xlabel("Time Step")
        plt.ylabel("Weight")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_network_activity(
        self,
        results: Dict[str, Any],
        title: str = "Network Activity",
        figsize: Tuple[int, int] = (10, 8),
    ):
        """Plot network activity heatmap."""
        # Extract spike data
        spike_times = results.get("layer_spike_times", {})

        if not spike_times:
            print("No spike data available for visualization")
            return

        # Create activity matrix
        max_time = results.get("duration", 100.0)
        time_bins = 100
        bin_size = max_time / time_bins

        activity_matrix = []
        neuron_labels = []

        for layer_name, layer_spikes in spike_times.items():
            for neuron_id, spikes in enumerate(layer_spikes):
                # Create activity histogram
                activity = np.zeros(time_bins)
                for spike_time in spikes:
                    bin_idx = min(int(spike_time / bin_size), time_bins - 1)
                    activity[bin_idx] += 1

                activity_matrix.append(activity)
                neuron_labels.append(f"{layer_name}_{neuron_id}")

        activity_matrix = np.array(activity_matrix)

        # Plot heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(
            activity_matrix,
            cmap="hot",
            aspect="auto",
            xticklabels=np.linspace(0, max_time, 10),
            yticklabels=neuron_labels[:: max(1, len(neuron_labels) // 20)],
        )
        plt.xlabel("Time (ms)")
        plt.ylabel("Neuron")
        plt.title(title)
        plt.show()

    def plot_neuromodulator_levels(
        self,
        neuromodulator_data: List[Dict[str, Any]],
        title: str = "Neuromodulator Levels",
        figsize: Tuple[int, int] = (12, 6),
    ):
        """Plot neuromodulator levels over time."""
        if not neuromodulator_data:
            return

        epochs = [d["epoch"] for d in neuromodulator_data]
        modulator_levels = neuromodulator_data[0]["neuromodulator_levels"]
        modulators = list(modulator_levels.keys())

        plt.figure(figsize=figsize)

        for modulator in modulators:
            levels = [
                d["neuromodulator_levels"][modulator] for d in neuromodulator_data
            ]
            plt.plot(epochs, levels, label=modulator.value, linewidth=2)

        plt.xlabel("Epoch")
        plt.ylabel("Level")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_learning_curves(
        self,
        training_history: List[Dict[str, Any]],
        title: str = "Learning Curves",
        figsize: Tuple[int, int] = (12, 6),
    ):
        """Plot learning curves from training history."""
        if not training_history:
            return

        epochs = [d["epoch"] for d in training_history]
        rewards = [d["average_reward"] for d in training_history]

        plt.figure(figsize=figsize)
        plt.plot(epochs, rewards, "b-", linewidth=2, label="Average Reward")
        plt.xlabel("Epoch")
        plt.ylabel("Average Reward")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


class SensorimotorSystem:
    """Complete sensorimotor control system."""

    def __init__(self):
        """Initialize sensorimotor system."""
        self.api = NeuromorphicAPI()
        self.setup_network()

    def setup_network(self):
        """Setup the sensorimotor network architecture."""
        # Create network using builder pattern
        self.api.create_network()

        # Sensory layers
        self.api.add_sensory_layer("visual_input", 1024, "retinal")
        self.api.add_sensory_layer("auditory_input", 256, "cochlear")
        self.api.add_sensory_layer("tactile_input", 512, "somatosensory")

        # Processing layers
        self.api.add_processing_layer("sensory_integration", 512, "adex")
        self.api.add_processing_layer("pattern_recognition", 256, "adex")
        self.api.add_processing_layer("decision_making", 128, "adex")

        # Motor layers
        self.api.add_motor_layer("motor_planning", 64)
        self.api.add_motor_layer("motor_output", 32)

        # Connect layers
        # Sensory integration
        self.api.connect_layers("visual_input", "sensory_integration", "feedforward")
        self.api.connect_layers("auditory_input", "sensory_integration", "feedforward")
        self.api.connect_layers("tactile_input", "sensory_integration", "feedforward")

        # Processing hierarchy
        self.api.connect_layers(
            "sensory_integration", "pattern_recognition", "feedforward"
        )
        self.api.connect_layers("pattern_recognition", "decision_making", "feedforward")

        # Motor control
        self.api.connect_layers("decision_making", "motor_planning", "feedforward")
        self.api.connect_layers("motor_planning", "motor_output", "feedforward")

        # Feedback loops
        self.api.connect_layers("motor_output", "sensory_integration", "feedback")

    def train(
        self, training_data: List[Dict[str, Any]], epochs: int = 100
    ) -> Dict[str, Any]:
        """Train the sensorimotor system."""
        return self.api.train_sensorimotor_system(training_data, epochs)

    def run_trial(
        self, sensory_inputs: Dict[str, Any], duration: float = 100.0
    ) -> Dict[str, Any]:
        """Run a single trial."""
        return self.api.run_simulation(duration, external_inputs=sensory_inputs)

    def get_network_info(self) -> Dict[str, Any]:
        """Get network information."""
        return self.api.get_network_info()
