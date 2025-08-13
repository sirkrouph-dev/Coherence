"""
High-level API for the neuromorphic programming system.
Provides easy-to-use interfaces for building and simulating neuromorphic networks.
"""

import time
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from core.encoding import (
    CochlearEncoder,
    MultiModalEncoder,
    RateEncoder,
    RetinalEncoder,
    SomatosensoryEncoder,
)
from core.logging_utils import TrainingTracker, neuromorphic_logger, trace_function
from core.network import EventDrivenSimulator, NetworkBuilder, NeuromorphicNetwork
from core.neuromodulation import (
    HomeostaticRegulator,
    NeuromodulatoryController,
    RewardSystem,
)
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

        Creates synaptic connections between two layers with configurable
        connection patterns and learning rules. Connection probability is
        automatically adjusted based on connection type.

        Args:
            pre_layer: Name of presynaptic layer.
            post_layer: Name of postsynaptic layer.
            connection_type: Type of connection pattern. Options:
                - "random": Random connections with specified probability
                - "feedforward": Dense forward connections (0.3 probability)
                - "lateral": Sparse lateral connections (0.1 probability)
                - "feedback": Feedback connections (0.2 probability)
            synapse_type: Type of synaptic plasticity. Options:
                - "stdp": Spike-Timing Dependent Plasticity
                - "stp": Short-Term Plasticity
                - "neuromodulatory": Neuromodulator-dependent plasticity
            connection_probability: Base probability of connection between neurons.
                May be overridden by connection_type.
            **kwargs: Additional parameters for synapse models, such as:
                - weight_init: Initial synaptic weight
                - tau_pre: Presynaptic trace time constant
                - tau_post: Postsynaptic trace time constant
                - learning_rate: Plasticity learning rate

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If no network has been created.

        Example:
            >>> api = NeuromorphicAPI()
            >>> api.create_network()
            >>> api.add_sensory_layer("visual", 100)
            >>> api.add_processing_layer("v1", 50)
            >>> api.connect_layers("visual", "v1", "feedforward", "stdp")
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

    def visualize_network(self, results: Dict[str, Any]) -> None:
        """Visualize network activity and simulation results.

        Creates visualization plots showing network activity patterns,
        spike rasters, and other relevant metrics from simulation results.

        Args:
            results: Dictionary containing simulation results with keys:
                - 'layer_spike_times': Spike times for each layer
                - 'duration': Simulation duration in milliseconds
                - 'weight_history': Optional synaptic weight evolution
                - 'neuromodulator_levels': Optional neuromodulator data

        Returns:
            None

        Example:
            >>> api = NeuromorphicAPI()
            >>> # ... build and configure network ...
            >>> results = api.run_simulation(duration=1000)
            >>> api.visualize_network(results)
        """
        self.visualization_tools.plot_network_activity(results)

    def save_network(self, filename: str) -> None:
        """Save network configuration to file.

        Serializes the current network state, including layer configurations,
        connections, and parameters to a file for later restoration.

        Args:
            filename: Path to the file where network will be saved.
                Should use .json or .pkl extension.

        Returns:
            None

        Raises:
            ValueError: If no network exists to save.
            IOError: If unable to write to the specified file.

        Example:
            >>> api = NeuromorphicAPI()
            >>> # ... build and configure network ...
            >>> api.save_network("my_network.json")

        Note:
            This method is currently a placeholder for future implementation.
        """
        # Implementation for saving network state
        pass

    def load_network(self, filename: str) -> None:
        """Load network configuration from file.

        Restores a previously saved network state, including all layers,
        connections, and parameters from a serialized file.

        Args:
            filename: Path to the file containing saved network.
                Should match the format used by save_network().

        Returns:
            None

        Raises:
            FileNotFoundError: If the specified file doesn't exist.
            ValueError: If the file contains invalid network data.
            IOError: If unable to read from the specified file.

        Example:
            >>> api = NeuromorphicAPI()
            >>> api.load_network("my_network.json")
            >>> # Network is now restored and ready for simulation

        Note:
            This method is currently a placeholder for future implementation.
        """
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
    ) -> None:
        """Plot spike raster from simulation data.

        Creates a raster plot showing spike times for all neurons across layers.
        Each dot represents a single spike event.

        Args:
            spike_data: Dictionary mapping layer names to lists of spike times.
                Each inner list contains spike times for one neuron.
            title: Title for the plot (default: "Spike Raster").
            figsize: Figure size as (width, height) in inches (default: (12, 8)).

        Returns:
            None

        Example:
            >>> visualizer = NeuromorphicVisualizer()
            >>> spike_data = {"layer1": [[10.5, 20.3], [15.2, 30.1]]}
            >>> visualizer.plot_spike_raster(spike_data)
        """
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
    ) -> None:
        """Plot evolution of synaptic weights over time.

        Visualizes how synaptic weights change during simulation or training,
        showing plasticity effects from learning rules like STDP.

        Args:
            weight_history: Dictionary mapping synapse IDs (pre, post) to lists
                of weight values over time.
            title: Title for the plot (default: "Synaptic Weight Evolution").
            figsize: Figure size as (width, height) in inches (default: (12, 6)).

        Returns:
            None

        Example:
            >>> visualizer = NeuromorphicVisualizer()
            >>> history = {(0, 1): [0.5, 0.52, 0.54], (1, 2): [0.3, 0.28, 0.25]}
            >>> visualizer.plot_weight_evolution(history)
        """
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
    ) -> None:
        """Plot network activity heatmap.

        Creates a heatmap visualization showing firing rate patterns across
        all neurons over time, useful for identifying network states and
        activity propagation.

        Args:
            results: Simulation results dictionary containing:
                - 'layer_spike_times': Spike times for each layer
                - 'duration': Total simulation duration in ms
            title: Title for the plot (default: "Network Activity").
            figsize: Figure size as (width, height) in inches (default: (10, 8)).

        Returns:
            None

        Example:
            >>> visualizer = NeuromorphicVisualizer()
            >>> results = {"layer_spike_times": {...}, "duration": 1000}
            >>> visualizer.plot_network_activity(results)
        """
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
    ) -> None:
        """Plot neuromodulator levels over time.

        Visualizes the dynamics of neuromodulatory systems (dopamine, serotonin,
        acetylcholine, norepinephrine) during training or simulation.

        Args:
            neuromodulator_data: List of dictionaries containing:
                - 'epoch': Training epoch number
                - 'neuromodulator_levels': Dict mapping modulator names to levels
            title: Title for the plot (default: "Neuromodulator Levels").
            figsize: Figure size as (width, height) in inches (default: (12, 6)).

        Returns:
            None

        Example:
            >>> visualizer = NeuromorphicVisualizer()
            >>> data = [{"epoch": 0, "neuromodulator_levels": {"dopamine": 0.5}}]
            >>> visualizer.plot_neuromodulator_levels(data)
        """
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
    ) -> None:
        """Plot learning curves from training history.

        Visualizes training progress over epochs, showing how average rewards
        evolve during the learning process.

        Args:
            training_history: List of dictionaries containing:
                - 'epoch': Training epoch number
                - 'average_reward': Mean reward for that epoch
            title: Title for the plot (default: "Learning Curves").
            figsize: Figure size as (width, height) in inches (default: (12, 6)).

        Returns:
            None

        Example:
            >>> visualizer = NeuromorphicVisualizer()
            >>> history = [{"epoch": 0, "average_reward": 0.1},
            ...            {"epoch": 1, "average_reward": 0.15}]
            >>> visualizer.plot_learning_curves(history)
        """
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
        """Train the sensorimotor system.

        Executes supervised training of the sensorimotor network using
        provided sensory-motor data pairs with reward-based learning.

        Args:
            training_data: List of training trials, each containing:
                - Sensory inputs (visual, auditory, tactile data)
                - Target motor outputs
                - Expected outcomes for reward computation
            epochs: Number of training epochs to run (default: 100).

        Returns:
            Dictionary containing:
            - 'training_history': List of epoch statistics
            - 'final_network_state': Network configuration after training
            - 'training_summary': Overall training metrics

        Example:
            >>> system = SensorimotorSystem()
            >>> data = [{"visual": image_data, "target": motor_command}]
            >>> results = system.train(data, epochs=50)
            >>> print(f"Final reward: {results['training_history'][-1]['average_reward']}")
        """
        return self.api.train_sensorimotor_system(training_data, epochs)

    def run_trial(
        self, sensory_inputs: Dict[str, Any], duration: float = 100.0
    ) -> Dict[str, Any]:
        """Run a single trial with specified sensory inputs.

        Executes one simulation trial, processing sensory inputs through
        the network to generate motor outputs.

        Args:
            sensory_inputs: Dictionary mapping sensory modalities to input data:
                - 'visual': Visual input data (images, patterns)
                - 'auditory': Audio input data (sounds, frequencies)
                - 'tactile': Touch/pressure sensor data
            duration: Trial duration in milliseconds (default: 100.0).

        Returns:
            Dictionary containing:
            - 'layer_spike_times': Spike activity for all layers
            - 'motor_output': Generated motor commands
            - 'duration': Actual simulation duration

        Example:
            >>> system = SensorimotorSystem()
            >>> inputs = {"visual": retinal_pattern, "auditory": sound_wave}
            >>> results = system.run_trial(inputs, duration=200)
            >>> motor_output = results['motor_output']
        """
        return self.api.run_simulation(duration, external_inputs=sensory_inputs)

    def get_network_info(self) -> Dict[str, Any]:
        """Get network information and statistics.

        Retrieves comprehensive information about the current network state,
        including layer configurations, connection statistics, and parameters.

        Returns:
            Dictionary containing:
            - 'layers': Information about each layer (size, neuron types)
            - 'connections': Connection statistics between layers
            - 'total_neurons': Total number of neurons in the network
            - 'total_synapses': Total number of synaptic connections

        Example:
            >>> system = SensorimotorSystem()
            >>> info = system.get_network_info()
            >>> print(f"Network has {info['total_neurons']} neurons")
        """
        return self.api.get_network_info()
