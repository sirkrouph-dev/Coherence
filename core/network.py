"""
Network architecture and simulation engine for the neuromorphic programming system.
Implements hierarchical network structures and event-driven simulation.
"""

import heapq
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .neurons import NeuronPopulation
from .synapses import SynapsePopulation, SynapseType


class NetworkLayer:
    """A layer in the neuromorphic network."""

    def __init__(self, name: str, size: int, neuron_type: str = "adex", **kwargs):
        """
        Initialize network layer.

        Args:
            name: Layer name
            size: Number of neurons in the layer
            neuron_type: Type of neurons to create
            **kwargs: Parameters for neuron models
        """
        self.name = name
        self.size = size
        self.neuron_type = neuron_type
        self.neuron_population = NeuronPopulation(size, neuron_type, **kwargs)
        self.spike_times = [[] for _ in range(size)]
        self.current_time = 0.0

    def step(self, dt: float, I_syn: List[float]) -> List[bool]:
        """Advance layer by one time step."""
        spikes = self.neuron_population.step(dt, I_syn)

        # Record spike times
        for i, spiked in enumerate(spikes):
            if spiked:
                self.spike_times[i].append(self.current_time)

        self.current_time += dt
        return spikes

    def reset(self):
        """Reset layer to initial state."""
        self.neuron_population.reset()
        self.spike_times = [[] for _ in range(self.size)]
        self.current_time = 0.0

    def get_spike_times(self) -> List[List[float]]:
        """Get spike times for all neurons in the layer."""
        return self.spike_times.copy()

    def get_membrane_potentials(self) -> List[float]:
        """Get current membrane potentials for all neurons."""
        return self.neuron_population.get_membrane_potentials()


class NetworkConnection:
    """Connection between two network layers."""

    def __init__(
        self,
        pre_layer: str,
        post_layer: str,
        synapse_type: str = "stdp",
        connection_probability: float = 0.1,
        **kwargs,
    ):
        """
        Initialize network connection.

        Args:
            pre_layer: Name of presynaptic layer
            post_layer: Name of postsynaptic layer
            synapse_type: Type of synapses to create
            connection_probability: Probability of connection between neurons
            **kwargs: Parameters for synapse models
        """
        self.pre_layer = pre_layer
        self.post_layer = post_layer
        self.synapse_type = synapse_type
        self.connection_probability = connection_probability
        self.synapse_population = None
        self.kwargs = kwargs

    def initialize(self, pre_size: int, post_size: int):
        """Initialize synapse population."""
        self.synapse_population = SynapsePopulation(
            pre_size,
            post_size,
            self.synapse_type,
            self.connection_probability,
            **self.kwargs,
        )

    def get_synaptic_currents(
        self, pre_spikes: List[bool], current_time: float
    ) -> List[float]:
        """Compute synaptic currents."""
        if self.synapse_population is None:
            return [0.0] * len(pre_spikes)
        return self.synapse_population.get_synaptic_currents(pre_spikes, current_time)

    def update_weights(
        self, pre_spikes: List[bool], post_spikes: List[bool], current_time: float
    ):
        """Update synaptic weights."""
        if self.synapse_population is not None:
            self.synapse_population.update_weights(
                pre_spikes, post_spikes, current_time
            )

    def step(self, dt: float):
        """Advance connection by one time step."""
        if self.synapse_population is not None:
            self.synapse_population.step(dt)

    def reset(self):
        """Reset connection to initial state."""
        if self.synapse_population is not None:
            self.synapse_population.reset()

    def get_weight_matrix(self) -> Optional[np.ndarray]:
        """Get weight matrix."""
        if self.synapse_population is not None:
            return self.synapse_population.get_weight_matrix()
        return None


class NeuromorphicNetwork:
    """Complete neuromorphic network with layers and connections."""

    def __init__(self):
        """Initialize neuromorphic network."""
        self.layers: Dict[str, NetworkLayer] = {}
        self.connections: Dict[Tuple[str, str], NetworkConnection] = {}
        self.current_time = 0.0
        self.simulation_history = []

    def add_layer(self, name: str, size: int, neuron_type: str = "adex", **kwargs):
        """
        Add a layer to the network.

        Args:
            name: Layer name
            size: Number of neurons in the layer
            neuron_type: Type of neurons to create
            **kwargs: Parameters for neuron models
        """
        layer = NetworkLayer(name, size, neuron_type, **kwargs)
        self.layers[name] = layer

    def connect_layers(
        self,
        pre_layer: str,
        post_layer: str,
        synapse_type: str = "stdp",
        connection_probability: float = 0.1,
        **kwargs,
    ):
        """
        Connect two layers.

        Args:
            pre_layer: Name of presynaptic layer
            post_layer: Name of postsynaptic layer
            synapse_type: Type of synapses to create
            connection_probability: Probability of connection between neurons
            **kwargs: Parameters for synapse models
        """
        if pre_layer not in self.layers:
            raise ValueError(f"Presynaptic layer '{pre_layer}' not found")
        if post_layer not in self.layers:
            raise ValueError(f"Postsynaptic layer '{post_layer}' not found")

        connection = NetworkConnection(
            pre_layer, post_layer, synapse_type, connection_probability, **kwargs
        )
        self.connections[(pre_layer, post_layer)] = connection

        # Initialize synapse population
        pre_size = self.layers[pre_layer].size
        post_size = self.layers[post_layer].size
        connection.initialize(pre_size, post_size)

    def step(self, dt: float):
        """Advance network by one time step."""
        # Collect spike information from all layers
        layer_spikes = {}
        layer_currents = defaultdict(lambda: [0.0] * 1000)  # Default currents

        # Step all layers
        for layer_name, layer in self.layers.items():
            # Get synaptic currents for this layer
            currents = layer_currents[layer_name]
            if len(currents) != layer.size:
                currents = [0.0] * layer.size

            # Step layer
            spikes = layer.step(dt, currents)
            layer_spikes[layer_name] = spikes

        # Update synaptic currents based on connections
        for (pre_name, post_name), connection in self.connections.items():
            pre_spikes = layer_spikes[pre_name]
            post_spikes = layer_spikes[post_name]

            # Compute synaptic currents
            currents = connection.get_synaptic_currents(pre_spikes, self.current_time)

            # Add to postsynaptic layer currents
            for i, current in enumerate(currents):
                layer_currents[post_name][i] += current

            # Update synaptic weights
            connection.update_weights(pre_spikes, post_spikes, self.current_time)

        # Step all connections
        for connection in self.connections.values():
            connection.step(dt)

        self.current_time += dt

        # Record simulation state
        self.simulation_history.append(
            {
                "time": self.current_time,
                "layer_spikes": layer_spikes.copy(),
                "layer_potentials": {
                    name: layer.get_membrane_potentials()
                    for name, layer in self.layers.items()
                },
            }
        )

    def run_simulation(self, duration: float, dt: float = 0.1) -> Dict[str, Any]:
        """
        Run network simulation.

        Args:
            duration: Simulation duration in milliseconds
            dt: Time step in milliseconds

        Returns:
            Simulation results
        """
        self.reset()

        num_steps = int(duration / dt)
        for _ in range(num_steps):
            self.step(dt)

        return {
            "duration": duration,
            "dt": dt,
            "final_time": self.current_time,
            "layer_spike_times": {
                name: layer.get_spike_times() for name, layer in self.layers.items()
            },
            "weight_matrices": {
                conn_name: conn.get_weight_matrix()
                for conn_name, conn in self.connections.items()
            },
            "simulation_history": self.simulation_history,
        }

    def reset(self):
        """Reset network to initial state."""
        for layer in self.layers.values():
            layer.reset()
        for connection in self.connections.values():
            connection.reset()
        self.current_time = 0.0
        self.simulation_history.clear()

    def get_network_info(self) -> Dict[str, Any]:
        """Get information about the network structure."""
        info = {
            "layers": {},
            "connections": {},
            "total_neurons": 0,
            "total_synapses": 0,
        }

        for name, layer in self.layers.items():
            info["layers"][name] = {
                "size": layer.size,
                "neuron_type": layer.neuron_type,
            }
            info["total_neurons"] += layer.size

        for (pre_name, post_name), connection in self.connections.items():
            info["connections"][f"{pre_name}->{post_name}"] = {
                "synapse_type": connection.synapse_type,
                "connection_probability": connection.connection_probability,
            }
            if connection.synapse_population is not None:
                info["total_synapses"] += len(connection.synapse_population.synapses)

        return info


class EventDrivenSimulator:
    """Event-driven simulation engine for spiking networks."""

    def __init__(self):
        """Initialize event-driven simulator."""
        self.event_queue = []
        self.current_time = 0.0
        self.network = None
        self.spike_events = []

    def set_network(self, network: NeuromorphicNetwork):
        """Set the network to simulate."""
        self.network = network

    def add_spike_event(self, neuron_id: int, layer_name: str, spike_time: float):
        """Add a spike event to the queue."""
        heapq.heappush(self.event_queue, (spike_time, "spike", neuron_id, layer_name))

    def add_external_input(
        self, layer_name: str, neuron_id: int, input_time: float, input_strength: float
    ):
        """Add external input event."""
        heapq.heappush(
            self.event_queue,
            (input_time, "input", neuron_id, layer_name, input_strength),
        )

    def run_simulation(self, duration: float) -> Dict[str, Any]:
        """
        Run event-driven simulation.

        Args:
            duration: Simulation duration in milliseconds

        Returns:
            Simulation results
        """
        if self.network is None:
            raise ValueError("No network set for simulation")

        self.network.reset()
        self.current_time = 0.0
        self.spike_events.clear()

        # Process events until duration is reached
        while self.event_queue and self.current_time < duration:
            event_time, event_type, *event_data = heapq.heappop(self.event_queue)
            self.current_time = event_time

            if event_type == "spike":
                neuron_id, layer_name = event_data
                self._process_spike(neuron_id, layer_name, event_time)
            elif event_type == "input":
                neuron_id, layer_name, input_strength = event_data
                self._process_external_input(
                    neuron_id, layer_name, input_strength, event_time
                )

        return {
            "duration": duration,
            "final_time": self.current_time,
            "spike_events": self.spike_events,
            "network_results": self.network.get_network_info(),
        }

    def _process_spike(self, neuron_id: int, layer_name: str, spike_time: float):
        """Process a spike event."""
        if layer_name not in self.network.layers:
            return

        layer = self.network.layers[layer_name]

        # Record spike
        self.spike_events.append(
            {
                "time": spike_time,
                "neuron_id": neuron_id,
                "layer_name": layer_name,
                "event_type": "spike",
            }
        )

        # Propagate spike to connected layers
        for (pre_name, post_name), connection in self.network.connections.items():
            if pre_name == layer_name:
                # This layer is presynaptic
                post_layer = self.network.layers[post_name]

                # Create spike events for postsynaptic neurons
                for post_neuron_id in range(post_layer.size):
                    if (
                        neuron_id,
                        post_neuron_id,
                    ) in connection.synapse_population.synapses:
                        # Add postsynaptic spike event with delay
                        delay = 1.0  # ms synaptic delay
                        post_spike_time = spike_time + delay
                        heapq.heappush(
                            self.event_queue,
                            (post_spike_time, "spike", post_neuron_id, post_name),
                        )

    def _process_external_input(
        self, neuron_id: int, layer_name: str, input_strength: float, input_time: float
    ):
        """Process external input event."""
        if layer_name not in self.network.layers:
            return

        layer = self.network.layers[layer_name]

        # Apply external input to neuron
        if neuron_id < layer.size:
            # Add external current to neuron
            # This would need to be implemented in the neuron model
            pass

    def reset(self):
        """Reset simulator to initial state."""
        self.event_queue.clear()
        self.current_time = 0.0
        self.spike_events.clear()
        if self.network is not None:
            self.network.reset()


class NetworkBuilder:
    """Helper class for building neuromorphic networks."""

    def __init__(self):
        """Initialize network builder."""
        self.network = NeuromorphicNetwork()

    def add_sensory_layer(self, name: str, size: int, encoding_type: str = "rate"):
        """Add a sensory input layer."""
        self.network.add_layer(name, size, neuron_type="lif")
        return self

    def add_processing_layer(self, name: str, size: int, neuron_type: str = "adex"):
        """Add a processing layer."""
        self.network.add_layer(name, size, neuron_type=neuron_type)
        return self

    def add_motor_layer(self, name: str, size: int):
        """Add a motor output layer."""
        self.network.add_layer(name, size, neuron_type="lif")
        return self

    def connect_layers(
        self, pre_layer: str, post_layer: str, connection_type: str = "random", **kwargs
    ):
        """Connect layers with specified pattern."""
        # Extract connection_probability from kwargs to avoid conflicts
        connection_probability = kwargs.pop("connection_probability", 0.1)

        if connection_type == "random":
            self.network.connect_layers(
                pre_layer,
                post_layer,
                connection_probability=connection_probability,
                **kwargs,
            )
        elif connection_type == "feedforward":
            self.network.connect_layers(
                pre_layer,
                post_layer,
                connection_probability=connection_probability,
                **kwargs,
            )
        elif connection_type == "lateral":
            # Lateral connections within the same layer
            self.network.connect_layers(
                pre_layer,
                post_layer,
                connection_probability=connection_probability,
                **kwargs,
            )
        elif connection_type == "feedback":
            self.network.connect_layers(
                pre_layer,
                post_layer,
                connection_probability=connection_probability,
                **kwargs,
            )
        return self

    def build(self) -> NeuromorphicNetwork:
        """Build and return the network."""
        return self.network
