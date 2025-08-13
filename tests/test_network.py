"""
Comprehensive tests for network functionality.
Tests network creation, simulation, and management.
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.network import (
    NeuromorphicNetwork,
    NetworkLayer,
    EventDrivenSimulator,
    NetworkConnection
)
from core.neurons import NeuronPopulation
from core.synapses import SynapsePopulation


class TestNetworkLayer(unittest.TestCase):
    """Test network layer functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.layer_size = 10
        self.neuron_type = "lif"
        
    def test_layer_creation(self):
        """Test network layer creation."""
        layer = NetworkLayer(
            name="test_layer",
            size=self.layer_size,
            neuron_type=self.neuron_type
        )
        
        self.assertIsNotNone(layer)
        self.assertEqual(layer.name, "test_layer")
        self.assertEqual(layer.size, self.layer_size)
        self.assertIsInstance(layer.neuron_population, NeuronPopulation)
        
    def test_layer_neuron_types(self):
        """Test different neuron types in layers."""
        # Test LIF neurons
        lif_layer = NetworkLayer("lif_layer", 5, "lif")
        self.assertEqual(lif_layer.size, 5)
        self.assertIsInstance(lif_layer.neuron_population, NeuronPopulation)
        
        # Test AdEx neurons
        adex_layer = NetworkLayer("adex_layer", 3, "adex")
        self.assertEqual(adex_layer.size, 3)
        self.assertIsInstance(adex_layer.neuron_population, NeuronPopulation)
        
    def test_layer_step(self):
        """Test layer simulation step."""
        layer = NetworkLayer("test_layer", 5, "lif")
        
        # Create synaptic currents
        I_syn = [1.0, 0.5, 2.0, 0.0, 1.5]
        
        # Step the layer
        spikes = layer.step(dt=0.1, I_syn=I_syn)
        
        # Should return spike array
        self.assertEqual(len(spikes), 5)
        self.assertTrue(all(isinstance(spike, bool) for spike in spikes))
        
    def test_layer_reset(self):
        """Test layer reset functionality."""
        layer = NetworkLayer("test_layer", 3, "lif")
        
        # Step with some input
        layer.step(dt=0.1, I_syn=[1.0, 1.0, 1.0])
        
        # Reset
        layer.reset()
        
        # Should be reset state
        self.assertEqual(layer.current_time, 0.0)
        
    def test_layer_spike_times(self):
        """Test spike time recording."""
        layer = NetworkLayer("test_layer", 2, "lif")
        
        # Step multiple times with high current
        for _ in range(10):
            layer.step(dt=0.1, I_syn=[5.0, 5.0])
            
        # Get spike times
        spike_times = layer.get_spike_times()
        self.assertEqual(len(spike_times), 2)
        self.assertIsInstance(spike_times, list)


class TestNeuromorphicNetwork(unittest.TestCase):
    """Test main network functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.network = NeuromorphicNetwork()
        
    def test_network_creation(self):
        """Test network creation."""
        self.assertIsNotNone(self.network)
        self.assertTrue(hasattr(self.network, 'layers'))
        self.assertTrue(hasattr(self.network, 'connections'))
        
    def test_layer_addition(self):
        """Test adding layers to network."""
        # Add input layer
        self.network.add_layer("input", 10, "lif")
        
        # Check layer was added
        self.assertIn("input", self.network.layers)
        self.assertEqual(self.network.layers["input"].size, 10)
        
        # Add hidden layer
        self.network.add_layer("hidden", 5, "adex")
        
        # Check second layer
        self.assertIn("hidden", self.network.layers)
        self.assertEqual(self.network.layers["hidden"].size, 5)
        
    def test_layer_connections(self):
        """Test connecting layers."""
        # Add layers
        self.network.add_layer("input", 8, "lif")
        self.network.add_layer("output", 4, "lif")
        
        # Connect layers
        self.network.connect_layers(
            "input", "output", 
            synapse_type="stdp", 
            connection_probability=0.5
        )
        
        # Check connection exists
        connection_key = ("input", "output")
        self.assertIn(connection_key, self.network.connections)
        
        # Check connection object
        connection = self.network.connections[connection_key]
        self.assertIsInstance(connection, NetworkConnection)
        
    def test_network_simulation(self):
        """Test basic network simulation."""
        # Create simple network
        self.network.add_layer("input", 3, "lif")
        self.network.add_layer("output", 2, "lif")
        self.network.connect_layers("input", "output", "stdp", 0.8)
        
        # Run simulation
        results = self.network.run_simulation(duration=10.0, dt=0.1)
        
        # Check results
        self.assertIsInstance(results, dict)
        self.assertIn("layer_spike_times", results)
        self.assertIn("input", results["layer_spike_times"])
        self.assertIn("output", results["layer_spike_times"])
        
    def test_network_info(self):
        """Test network information retrieval."""
        # Create network
        self.network.add_layer("layer1", 6, "lif")
        self.network.add_layer("layer2", 4, "adex")
        self.network.connect_layers("layer1", "layer2", "stdp", 0.5)
        
        # Get network info
        info = self.network.get_network_info()
        
        # Should have information
        self.assertIn("total_neurons", info)
        self.assertIn("total_synapses", info)
        self.assertIn("layers", info)
        self.assertIn("connections", info)
        
        # Check values
        self.assertEqual(info["total_neurons"], 10)
        
    def test_network_reset(self):
        """Test network reset functionality."""
        # Create and simulate network
        self.network.add_layer("test", 5, "lif")
        self.network.run_simulation(duration=5.0, dt=0.1)
        
        # Reset network
        self.network.reset()
        
        # Should be reset
        self.assertEqual(self.network.current_time, 0.0)


class TestNetworkConnection(unittest.TestCase):
    """Test network connection functionality."""
    
    def test_connection_creation(self):
        """Test connection creation."""
        connection = NetworkConnection(
            pre_layer="input",
            post_layer="output",
            synapse_type="stdp",
            connection_probability=0.3
        )
        
        self.assertIsNotNone(connection)
        self.assertEqual(connection.pre_layer, "input")
        self.assertEqual(connection.post_layer, "output")
        self.assertEqual(connection.synapse_type, "stdp")
        self.assertEqual(connection.connection_probability, 0.3)
        
    def test_connection_initialization(self):
        """Test connection initialization with sizes."""
        connection = NetworkConnection("pre", "post", "stdp", 0.5)
        
        # Initialize with sizes
        connection.initialize(pre_size=10, post_size=5)
        
        # Should have synapse population
        self.assertIsNotNone(connection.synapse_population)
        self.assertIsInstance(connection.synapse_population, SynapsePopulation)
        
    def test_connection_weight_matrix(self):
        """Test weight matrix retrieval."""
        connection = NetworkConnection("pre", "post", "stdp", 0.5)
        connection.initialize(pre_size=3, post_size=2)
        
        # Get weight matrix
        weights = connection.get_weight_matrix()
        
        # Should have valid matrix or None
        if weights is not None:
            self.assertIsInstance(weights, np.ndarray)
            self.assertEqual(weights.shape, (2, 3))  # post x pre


class TestEventDrivenSimulator(unittest.TestCase):
    """Test event-driven simulation engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.network = NeuromorphicNetwork()
        self.network.add_layer("test", 3, "lif")
        self.simulator = EventDrivenSimulator()
        
    def test_simulator_creation(self):
        """Test simulator creation."""
        self.assertIsNotNone(self.simulator)
        self.assertTrue(hasattr(self.simulator, 'event_queue'))
        
    def test_network_setting(self):
        """Test setting network in simulator."""
        self.simulator.set_network(self.network)
        self.assertEqual(self.simulator.network, self.network)
        
    def test_external_input_addition(self):
        """Test adding external input events."""
        self.simulator.set_network(self.network)
        
        # Add external input
        self.simulator.add_external_input(
            layer_name="test",
            neuron_id=0,
            input_time=5.0,
            input_strength=2.0
        )
        
        # Should have event in queue
        self.assertGreater(len(self.simulator.event_queue), 0)
        
    def test_spike_event_addition(self):
        """Test adding spike events."""
        self.simulator.set_network(self.network)
        
        # Add spike event
        self.simulator.add_spike_event(
            neuron_id=1,
            layer_name="test",
            spike_time=10.0
        )
        
        # Should have event in queue
        self.assertGreater(len(self.simulator.event_queue), 0)
        
    def test_event_driven_simulation(self):
        """Test running event-driven simulation."""
        self.simulator.set_network(self.network)
        
        # Add some events
        self.simulator.add_external_input("test", 0, 1.0, 3.0)
        self.simulator.add_spike_event(1, "test", 5.0)
        
        # Run simulation
        results = self.simulator.run_simulation(duration=20.0)
        
        # Should have results
        self.assertIsInstance(results, dict)
        self.assertIn("duration", results)
        self.assertIn("spike_events", results)


class TestNetworkValidation(unittest.TestCase):
    """Test network validation and error handling."""
    
    def test_invalid_layer_size(self):
        """Test invalid layer size handling."""
        network = NeuromorphicNetwork()
        
        # Try negative size
        with self.assertRaises(ValueError):
            network.add_layer("invalid", -5, "lif")
            
        # Try zero size
        with self.assertRaises(ValueError):
            network.add_layer("invalid", 0, "lif")
            
    def test_invalid_neuron_type(self):
        """Test invalid neuron type handling."""
        network = NeuromorphicNetwork()
        
        # Try invalid neuron type
        with self.assertRaises(ValueError):
            network.add_layer("invalid", 5, "nonexistent")
            
    def test_invalid_connections(self):
        """Test invalid connection attempts."""
        network = NeuromorphicNetwork()
        network.add_layer("layer1", 5, "lif")
        
        # Try to connect non-existent layer
        with self.assertRaises(ValueError):
            network.connect_layers("layer1", "nonexistent", "stdp", 0.5)
            
    def test_invalid_connection_probability(self):
        """Test invalid connection probability."""
        network = NeuromorphicNetwork()
        network.add_layer("layer1", 5, "lif")
        network.add_layer("layer2", 3, "lif")
        
        # Try negative probability
        with self.assertRaises(ValueError):
            network.connect_layers("layer1", "layer2", "stdp", -0.1)
            
        # Try probability > 1
        with self.assertRaises(ValueError):
            network.connect_layers("layer1", "layer2", "stdp", 1.5)
            
    def test_invalid_simulation_parameters(self):
        """Test invalid simulation parameters."""
        network = NeuromorphicNetwork()
        network.add_layer("test", 2, "lif")
        
        # Try negative duration
        with self.assertRaises(ValueError):
            network.run_simulation(duration=-1.0)
            
        # Try negative time step
        with self.assertRaises(ValueError):
            network.run_simulation(duration=10.0, dt=-0.1)
            
        # Try time step larger than duration
        with self.assertRaises(ValueError):
            network.run_simulation(duration=1.0, dt=2.0)


class TestNetworkIntegration(unittest.TestCase):
    """Test integrated network functionality."""
    
    def test_simple_feedforward_network(self):
        """Test simple feedforward network."""
        network = NeuromorphicNetwork()
        
        # Create feedforward architecture
        network.add_layer("input", 5, "lif")
        network.add_layer("hidden", 3, "adex")
        network.add_layer("output", 2, "lif")
        
        # Connect layers
        network.connect_layers("input", "hidden", "stdp", 0.4)
        network.connect_layers("hidden", "output", "stdp", 0.6)
        
        # Run simulation
        results = network.run_simulation(duration=20.0, dt=0.1)
        
        # Should have activity in all layers
        self.assertIn("input", results["layer_spike_times"])
        self.assertIn("hidden", results["layer_spike_times"])
        self.assertIn("output", results["layer_spike_times"])
        
    def test_recurrent_network(self):
        """Test network with recurrent connections."""
        network = NeuromorphicNetwork()
        
        # Create recurrent network
        network.add_layer("excitatory", 8, "lif")
        network.add_layer("inhibitory", 2, "adex")
        
        # Connect with recurrence
        network.connect_layers("excitatory", "inhibitory", "stdp", 0.3)
        network.connect_layers("inhibitory", "excitatory", "stdp", 0.2)
        network.connect_layers("excitatory", "excitatory", "stdp", 0.1)
        
        # Run simulation
        results = network.run_simulation(duration=30.0, dt=0.1)
        
        # Should complete without errors
        self.assertIsInstance(results, dict)
        
    def test_multiple_layer_types(self):
        """Test network with multiple neuron types."""
        network = NeuromorphicNetwork()
        
        # Create diverse network
        network.add_layer("sensory", 6, "lif")
        network.add_layer("cortical", 4, "adex")
        
        # Connect different types
        network.connect_layers("sensory", "cortical", "stdp", 0.5)
        
        # Run simulation
        results = network.run_simulation(duration=15.0, dt=0.1)
        
        # Should handle different neuron types
        self.assertIn("sensory", results["layer_spike_times"])
        self.assertIn("cortical", results["layer_spike_times"])


class TestNetworkEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def test_empty_network_simulation(self):
        """Test simulation with empty network."""
        network = NeuromorphicNetwork()
        
        # Should handle empty simulation
        results = network.run_simulation(duration=1.0, dt=0.1)
        self.assertIsInstance(results, dict)
        
    def test_single_neuron_network(self):
        """Test network with single neuron."""
        network = NeuromorphicNetwork()
        network.add_layer("single", 1, "lif")
        
        # Should handle single neuron
        results = network.run_simulation(duration=5.0, dt=0.1)
        self.assertIsInstance(results, dict)
        
    def test_very_small_time_step(self):
        """Test very small time steps."""
        network = NeuromorphicNetwork()
        network.add_layer("test", 2, "lif")
        
        # Very small time step
        results = network.run_simulation(duration=0.5, dt=0.001)
        self.assertIsInstance(results, dict)
        
    def test_network_with_no_connections(self):
        """Test network with layers but no connections."""
        network = NeuromorphicNetwork()
        network.add_layer("isolated1", 3, "lif")
        network.add_layer("isolated2", 2, "adex")
        
        # No connections between layers
        results = network.run_simulation(duration=10.0, dt=0.1)
        
        # Should still work
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results["layer_spike_times"]), 2)


if __name__ == '__main__':
    unittest.main()
