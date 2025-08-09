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
    
    # Resource limits to prevent memory exhaustion
    MAX_NEURONS = 1_000_000
    MAX_SYNAPSES = 100_000_000
    MAX_LAYERS = 1000
    MAX_SIMULATION_STEPS = 1_000_000
    MAX_INPUT_STRENGTH = 1000.0

    def __init__(self):
        """Initialize neuromorphic network."""
        self.layers: Dict[str, NetworkLayer] = {}
        self.connections: Dict[Tuple[str, str], NetworkConnection] = {}
        self.current_time = 0.0
        self.simulation_history = []
        self.total_neurons = 0
        self.total_synapses = 0

    def add_layer(self, name: str, size: int, neuron_type: str = "adex", **kwargs):
        """
        Add a layer to the network with validation.

        Args:
            name: Layer name
            size: Number of neurons in the layer
            neuron_type: Type of neurons to create
            **kwargs: Parameters for neuron models
        """
        # Input validation
        if not isinstance(name, str) or not name:
            raise ValueError("Layer name must be a non-empty string")
        
        if not isinstance(size, int) or size <= 0:
            raise ValueError(f"Layer size must be a positive integer, got {size}")
        
        if size > self.MAX_NEURONS:
            raise ValueError(f"Layer size {size} exceeds maximum allowed neurons {self.MAX_NEURONS}")
        
        if len(self.layers) >= self.MAX_LAYERS:
            raise ValueError(f"Maximum number of layers ({self.MAX_LAYERS}) exceeded")
        
        if self.total_neurons + size > self.MAX_NEURONS:
            raise ValueError(f"Adding layer would exceed total neuron limit of {self.MAX_NEURONS}")
        
        # Validate neuron type
        valid_types = ["adex", "lif", "hodgkin_huxley", "izhikevich"]
        if neuron_type not in valid_types:
            raise ValueError(f"Invalid neuron type '{neuron_type}'. Must be one of {valid_types}")
        
        layer = NetworkLayer(name, size, neuron_type, **kwargs)
        self.layers[name] = layer
        self.total_neurons += size

    def connect_layers(
        self,
        pre_layer: str,
        post_layer: str,
        synapse_type: str = "stdp",
        connection_probability: float = 0.1,
        **kwargs,
    ):
        """
        Connect two layers with validation.

        Args:
            pre_layer: Name of presynaptic layer
            post_layer: Name of postsynaptic layer
            synapse_type: Type of synapses to create
            connection_probability: Probability of connection between neurons
            **kwargs: Parameters for synapse models
        """
        # Validate layer names
        if pre_layer not in self.layers:
            raise ValueError(f"Presynaptic layer '{pre_layer}' not found")
        if post_layer not in self.layers:
            raise ValueError(f"Postsynaptic layer '{post_layer}' not found")
        
        # Validate connection probability
        if not 0.0 <= connection_probability <= 1.0:
            raise ValueError(f"Connection probability must be between 0 and 1, got {connection_probability}")
        
        # Validate synapse type
        valid_synapse_types = ["stdp", "stp", "neuromodulatory", "rstdp"]
        if synapse_type not in valid_synapse_types:
            raise ValueError(f"Invalid synapse type '{synapse_type}'. Must be one of {valid_synapse_types}")
        
        # Check if connection would exceed synapse limit
        pre_size = self.layers[pre_layer].size
        post_size = self.layers[post_layer].size
        expected_synapses = int(pre_size * post_size * connection_probability)
        
        if self.total_synapses + expected_synapses > self.MAX_SYNAPSES:
            raise ValueError(f"Adding connection would exceed total synapse limit of {self.MAX_SYNAPSES}")

        connection = NetworkConnection(
            pre_layer, post_layer, synapse_type, connection_probability, **kwargs
        )
        self.connections[(pre_layer, post_layer)] = connection

        # Initialize synapse population
        connection.initialize(pre_size, post_size)
        self.total_synapses += expected_synapses

    def step(self, dt: float):
        """Advance network by one time step."""
        # Initialize layer currents properly for each layer
        layer_currents = {}
        for layer_name, layer in self.layers.items():
            layer_currents[layer_name] = [0.0] * layer.size
        
        # First, compute synaptic currents from previous timestep spikes
        for (pre_name, post_name), connection in self.connections.items():
            if connection.synapse_population is not None:
                # Get previous spikes (we'll use a simple approach here)
                pre_layer = self.layers[pre_name]
                pre_spikes = [False] * pre_layer.size
                # Check which neurons are above threshold (simplified spike detection)
                for i, neuron in enumerate(pre_layer.neuron_population.neurons):
                    if hasattr(neuron, 'is_spiking') and neuron.is_spiking:
                        pre_spikes[i] = True
                
                # Compute synaptic currents
                currents = connection.get_synaptic_currents(pre_spikes, self.current_time)
                
                # Add to postsynaptic layer currents
                for i, current in enumerate(currents):
                    if i < len(layer_currents[post_name]):
                        layer_currents[post_name][i] += current
        
        # Step all layers with their synaptic currents
        layer_spikes = {}
        for layer_name, layer in self.layers.items():
            currents = layer_currents[layer_name]
            spikes = layer.step(dt, currents)
            layer_spikes[layer_name] = spikes
        
        # Update synaptic weights based on current spikes
        for (pre_name, post_name), connection in self.connections.items():
            if pre_name in layer_spikes and post_name in layer_spikes:
                pre_spikes = layer_spikes[pre_name]
                post_spikes = layer_spikes[post_name]
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
        Run network simulation with validation.

        Args:
            duration: Simulation duration in milliseconds
            dt: Time step in milliseconds

        Returns:
            Simulation results
        """
        # Validate inputs
        if not isinstance(duration, (int, float)) or duration <= 0:
            raise ValueError(f"Duration must be positive, got {duration}")
        
        if not isinstance(dt, (int, float)) or dt <= 0:
            raise ValueError(f"Time step must be positive, got {dt}")
        
        if dt > duration:
            raise ValueError(f"Time step {dt} cannot be larger than duration {duration}")
        
        num_steps = int(duration / dt)
        if num_steps > self.MAX_SIMULATION_STEPS:
            raise ValueError(f"Simulation would require {num_steps} steps, exceeding limit of {self.MAX_SIMULATION_STEPS}")
        
        self.reset()

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

    # --- Sleep-inspired phase (optional, opt-in) ---
    def run_sleep_phase(
        self,
        duration: float = 0.0,
        dt: float = 0.1,
        *,
        downscale_factor: Optional[float] = None,
        normalize_incoming: bool = False,
        replay: Optional[Dict[str, np.ndarray]] = None,
        noise_std: Optional[float] = None,
    ) -> None:
        """
        Execute an optional sleep/rest phase that can (a) replay activity and (b) apply
        biologically inspired consolidation operations (global downscaling, normalization).

        Args:
            duration: Sleep duration in ms. If > 0 and replay provided, runs replay-driven spikes.
            dt: Time step in ms for sleep stepping.
            downscale_factor: If provided (e.g., 0.98), uniformly downscale all weights
                per connection after the phase (SHY-like consolidation).
            normalize_incoming: If True, normalize incoming weights per postsynaptic neuron
                after the phase to stabilize representations.
            replay: Optional map from layer name -> external current vector (length = layer size)
                to apply during sleep at each step. This simulates offline replay.

        Notes:
            - This method is opt-in and is never invoked automatically.
            - It preserves normal learning dynamics. STDP remains active during replay.
            - If duration <= 0 or no replay is provided, only consolidation ops are applied.
            - If noise_std > 0, additive Gaussian current noise (mean 0, std noise_std)
              is applied to each layer during sleep steps (biologically plausible background).
        """
        # Replay phase (optional)
        if duration > 0 and replay:
            num_steps = int(max(1, duration / dt))
            for _ in range(num_steps):
                # Prepare zero currents per layer
                layer_currents: Dict[str, List[float]] = {
                    name: [0.0] * layer.size for name, layer in self.layers.items()
                }

                # Add synaptic currents based on previous spikes
                for (pre_name, post_name), connection in self.connections.items():
                    if connection.synapse_population is None:
                        continue
                    pre_layer = self.layers[pre_name]
                    pre_spikes = [False] * pre_layer.size
                    for i, neuron in enumerate(pre_layer.neuron_population.neurons):
                        if hasattr(neuron, "is_spiking") and neuron.is_spiking:
                            pre_spikes[i] = True
                    currents = connection.get_synaptic_currents(pre_spikes, self.current_time)
                    target = layer_currents[post_name]
                    for i, c in enumerate(currents):
                        if i < len(target):
                            target[i] += c

                # Add replay external currents
                for layer_name, ext_current in replay.items():
                    if layer_name in self.layers:
                        # Ensure correct length
                        layer = self.layers[layer_name]
                        if len(ext_current) == layer.size:
                            lc = layer_currents[layer_name]
                            for i in range(layer.size):
                                lc[i] += float(ext_current[i])

                # Additive Gaussian noise to all layers (optional)
                if noise_std is not None and noise_std > 0.0:
                    for lname, layer in self.layers.items():
                        lc = layer_currents[lname]
                        if lc:
                            lc_arr = np.asarray(lc, dtype=float)
                            lc_arr += np.random.normal(0.0, float(noise_std), size=layer.size)
                            # write back
                            for i in range(layer.size):
                                lc[i] = float(lc_arr[i])

                # Step all layers
                layer_spikes: Dict[str, List[bool]] = {}
                for layer_name, layer in self.layers.items():
                    spikes = layer.step(dt, layer_currents[layer_name])
                    layer_spikes[layer_name] = spikes

                # Update weights via active plasticity (STDP) during replay
                for (pre_name, post_name), connection in self.connections.items():
                    if pre_name in layer_spikes and post_name in layer_spikes:
                        connection.update_weights(
                            layer_spikes[pre_name], layer_spikes[post_name], self.current_time
                        )

                # Advance time
                for connection in self.connections.values():
                    connection.step(dt)
                self.current_time += dt

        # Post-sleep consolidation ops (optional)
        if downscale_factor is not None or normalize_incoming:
            for connection in self.connections.values():
                sp = connection.synapse_population
                if sp is None:
                    continue
                if downscale_factor is not None and downscale_factor > 0:
                    sp.scale_all_weights(downscale_factor)
                if normalize_incoming:
                    sp.normalize_incoming()


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
        """Add external input event with validation."""
        # Import security manager for validation
        from .security_manager import SecurityManager
        
        # Validate inputs
        if self.network and layer_name not in self.network.layers:
            raise ValueError(f"Invalid layer: {layer_name}")
        
        if self.network and neuron_id >= self.network.layers[layer_name].size:
            raise ValueError(f"Invalid neuron ID {neuron_id} for layer {layer_name}")
        
        # Validate input strength to prevent excessive values
        input_strength = SecurityManager.validate_network_input(
            input_strength, 
            min_val=-NeuromorphicNetwork.MAX_INPUT_STRENGTH,
            max_val=NeuromorphicNetwork.MAX_INPUT_STRENGTH,
            dtype=float
        )
        
        # Validate input time
        if input_time < 0:
            raise ValueError(f"Input time cannot be negative: {input_time}")
        
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
