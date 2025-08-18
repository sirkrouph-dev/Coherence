#!/usr/bin/env python3
"""
Test suite for modular network architecture mechanisms.

This module tests the modular network system to ensure it:
- Creates network modules with dense intra-module connectivity
- Implements sparse inter-module connections
- Supports hierarchical module organization
- Shows small-world network properties (high clustering, short paths)
"""

import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the core modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.brain_topology import (
    ModularNetworkArchitecture,
    BrainInspiredNetworkBuilder,
    SpatialPosition,
    NetworkModule
)


class TestModularNetworkArchitecture(unittest.TestCase):
    """Test modular network architecture system."""
    
    def setUp(self):
        """Set up test parameters."""
        self.modular_arch = ModularNetworkArchitecture()
        
    def test_modular_layout_creation(self):
        """Test creation of modular spatial layout."""
        total_neurons = 200
        num_modules = 4
        
        positions, modules = self.modular_arch.create_modular_layout(
            total_neurons, num_modules
        )
        
        # Should create correct number of positions and modules
        self.assertEqual(len(positions), total_neurons)
        self.assertEqual(len(modules), num_modules)
        
        # All neurons should be assigned to modules
        total_module_neurons = sum(len(module.neuron_ids) for module in modules.values())
        self.assertEqual(total_module_neurons, total_neurons)
        
        # Check module properties
        for module_name, module in modules.items():
            self.assertIsInstance(module, NetworkModule)
            self.assertGreater(len(module.neuron_ids), 0)
            self.assertEqual(module.excitatory_fraction, 0.8)
            
            # Check that module neurons are positioned near module center
            distances = []
            for neuron_id in module.neuron_ids:
                pos = positions[neuron_id]
                distance = pos.distance_to(module.center_position)
                distances.append(distance)
                
            # Most neurons should be within reasonable distance of center
            mean_distance = np.mean(distances)
            self.assertLess(mean_distance, module.radius * 2)
            
    def test_modular_connectivity_computation(self):
        """Test computation of modular connectivity patterns."""
        # Create simple test modules
        positions = {
            0: SpatialPosition(0, 0, 0),
            1: SpatialPosition(5, 0, 0),
            2: SpatialPosition(100, 0, 0),  # Far from others
            3: SpatialPosition(105, 0, 0)
        }
        
        modules = {
            'module_0': NetworkModule('module_0', SpatialPosition(2.5, 0, 0), 20, [0, 1]),
            'module_1': NetworkModule('module_1', SpatialPosition(102.5, 0, 0), 20, [2, 3])
        }
        
        connections = self.modular_arch.compute_modular_connectivity(modules, positions)
        
        # Should have some connections
        self.assertGreater(len(connections), 0)
        
        # Analyze connection patterns
        intra_module_connections = 0
        inter_module_connections = 0
        
        for (pre_id, post_id), weight in connections.items():
            # Determine if intra or inter-module
            pre_module = 'module_0' if pre_id in [0, 1] else 'module_1'
            post_module = 'module_0' if post_id in [0, 1] else 'module_1'
            
            if pre_module == post_module:
                intra_module_connections += 1
            else:
                inter_module_connections += 1
                
            # Check weight is positive
            self.assertGreater(weight, 0)
            
        # Should have more intra-module than inter-module connections
        # (due to higher probability and closer distances)
        if intra_module_connections + inter_module_connections > 0:
            intra_fraction = intra_module_connections / (intra_module_connections + inter_module_connections)
            self.assertGreater(intra_fraction, 0.3, "Should have significant intra-module connectivity")
            
    def test_small_world_rewiring(self):
        """Test small-world rewiring functionality."""
        # Create initial regular connections
        initial_connections = {
            (0, 1): 1.0,
            (1, 2): 1.0,
            (2, 3): 1.0,
            (3, 0): 1.0,
            (0, 2): 0.5,
            (1, 3): 0.5
        }
        
        rewired_connections = self.modular_arch.add_small_world_rewiring(
            initial_connections, rewiring_prob=0.5
        )
        
        # Should have similar number of connections
        self.assertAlmostEqual(len(rewired_connections), len(initial_connections), delta=2)
        
        # Some connections should be different
        common_connections = set(initial_connections.keys()) & set(rewired_connections.keys())
        self.assertLess(len(common_connections), len(initial_connections), 
                       "Some connections should be rewired")
        
    def test_network_properties_analysis(self):
        """Test analysis of network properties."""
        # Create test network
        connections = {
            (0, 1): 1.0, (1, 0): 1.0,  # Bidirectional for clustering
            (1, 2): 1.0, (2, 1): 1.0,
            (2, 3): 1.0, (3, 2): 1.0,
            (3, 0): 1.0, (0, 3): 1.0,
            (0, 2): 1.0, (2, 0): 1.0,  # Triangle: 0-1-2
            (4, 5): 1.0, (5, 4): 1.0,  # Separate module
            (0, 4): 0.5  # Inter-module connection
        }
        
        modules = {
            'module_0': NetworkModule('module_0', SpatialPosition(0, 0, 0), 10, [0, 1, 2, 3]),
            'module_1': NetworkModule('module_1', SpatialPosition(50, 0, 0), 10, [4, 5])
        }
        
        analysis = self.modular_arch.analyze_network_properties(connections, modules)
        
        # Check that analysis contains expected metrics
        expected_metrics = [
            'clustering_coefficient', 'average_path_length', 'modularity',
            'small_world_index', 'total_connections', 'intra_module_connections',
            'inter_module_connections'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, analysis)
            self.assertIsInstance(analysis[metric], (int, float))
            
        # Check reasonable values
        self.assertGreaterEqual(analysis['clustering_coefficient'], 0.0)
        self.assertLessEqual(analysis['clustering_coefficient'], 1.0)
        self.assertGreaterEqual(analysis['modularity'], 0.0)
        self.assertLessEqual(analysis['modularity'], 1.0)
        
        # Should have high modularity (most connections intra-module)
        self.assertGreater(analysis['modularity'], 0.5, "Should have high modularity")
        
    def test_parameter_configuration(self):
        """Test configuration of modular architecture parameters."""
        # Test default parameters
        self.assertGreater(self.modular_arch.intra_module_probability, 
                          self.modular_arch.inter_module_probability)
        self.assertGreater(self.modular_arch.clustering_target, 0.3)
        self.assertLess(self.modular_arch.path_length_target, 10.0)
        
        # Test parameter modification
        self.modular_arch.intra_module_probability = 0.5
        self.modular_arch.inter_module_probability = 0.01
        
        self.assertEqual(self.modular_arch.intra_module_probability, 0.5)
        self.assertEqual(self.modular_arch.inter_module_probability, 0.01)


class TestBrainInspiredNetworkBuilderModular(unittest.TestCase):
    """Test modular network creation in brain-inspired network builder."""
    
    def setUp(self):
        """Set up test parameters."""
        self.builder = BrainInspiredNetworkBuilder()
        
    def test_modular_cortical_network_creation(self):
        """Test creation of modular cortical network."""
        total_neurons = 120
        num_modules = 3
        
        network = self.builder.create_modular_cortical_network(
            total_neurons=total_neurons,
            num_modules=num_modules,
            enable_small_world=True
        )
        
        # Should create network with modular structure
        self.assertIsNotNone(network)
        
        # Should have layers for each module
        layer_names = list(network.layers.keys())
        module_layers = [name for name in layer_names if 'module_' in name]
        self.assertGreater(len(module_layers), 0, "Should have module-specific layers")
        
        # Should have both excitatory and inhibitory layers
        excitatory_layers = [name for name in layer_names if 'excitatory' in name]
        inhibitory_layers = [name for name in layer_names if 'inhibitory' in name]
        
        self.assertGreater(len(excitatory_layers), 0, "Should have excitatory layers")
        self.assertGreater(len(inhibitory_layers), 0, "Should have inhibitory layers")
        
        # Total neurons should match
        total_layer_neurons = sum(layer.size for layer in network.layers.values())
        self.assertEqual(total_layer_neurons, total_neurons)
        
    def test_modular_network_connections(self):
        """Test that modular networks have appropriate connections."""
        network = self.builder.create_modular_cortical_network(
            total_neurons=80,
            num_modules=2,
            enable_small_world=False  # Disable for clearer testing
        )
        
        # Should have connections
        self.assertGreater(len(network.connections), 0, "Should have connections")
        
        # Check connection types
        connection_names = [f"{pre}->{post}" for (pre, post) in network.connections.keys()]
        
        # Should have intra-module connections
        intra_module_connections = [name for name in connection_names 
                                   if name.split('->')[0].split('_')[0] == name.split('->')[1].split('_')[0]]
        self.assertGreater(len(intra_module_connections), 0, "Should have intra-module connections")
        
        # Should have inter-module connections
        inter_module_connections = [name for name in connection_names 
                                   if name.split('->')[0].split('_')[0] != name.split('->')[1].split('_')[0]]
        self.assertGreater(len(inter_module_connections), 0, "Should have inter-module connections")
        
    def test_small_world_properties(self):
        """Test that networks with small-world rewiring have appropriate properties."""
        # Create network with small-world properties
        network = self.builder.create_modular_cortical_network(
            total_neurons=100,
            num_modules=4,
            enable_small_world=True
        )
        
        # Analyze network properties
        analysis = self.builder.analyze_network_modularity()
        
        # Should have analysis results
        self.assertNotIn('error', analysis)
        self.assertIn('clustering_coefficient', analysis)
        self.assertIn('average_path_length', analysis)
        self.assertIn('small_world_index', analysis)
        
        # Should have reasonable small-world properties
        clustering = analysis['clustering_coefficient']
        path_length = analysis['average_path_length']
        
        self.assertGreater(clustering, 0.1, "Should have reasonable clustering")
        self.assertGreater(path_length, 1.0, "Should have reasonable path length")
        
        if path_length > 0:
            small_world_index = clustering / path_length
            self.assertGreater(small_world_index, 0.05, "Should have small-world properties")
            
    def test_modularity_analysis(self):
        """Test modularity analysis functionality."""
        network = self.builder.create_modular_cortical_network(
            total_neurons=60,
            num_modules=3
        )
        
        analysis = self.builder.analyze_network_modularity()
        
        # Should have complete analysis
        expected_fields = [
            'clustering_coefficient', 'average_path_length', 'modularity',
            'small_world_index', 'num_modules', 'neurons_per_module', 'module_names'
        ]
        
        for field in expected_fields:
            self.assertIn(field, analysis, f"Analysis should contain {field}")
            
        # Check module information
        self.assertEqual(analysis['num_modules'], 3)
        self.assertEqual(len(analysis['neurons_per_module']), 3)
        self.assertEqual(len(analysis['module_names']), 3)
        
        # Should have high modularity
        self.assertGreater(analysis['modularity'], 0.4, "Modular network should have high modularity")
        
    def test_different_module_counts(self):
        """Test networks with different numbers of modules."""
        module_counts = [2, 3, 5, 8]
        
        for num_modules in module_counts:
            with self.subTest(num_modules=num_modules):
                network = self.builder.create_modular_cortical_network(
                    total_neurons=num_modules * 20,  # 20 neurons per module
                    num_modules=num_modules
                )
                
                # Should create network successfully
                self.assertIsNotNone(network)
                
                # Should have appropriate number of layers
                layer_names = list(network.layers.keys())
                module_layers = set(name.split('_')[0] + '_' + name.split('_')[1] 
                                  for name in layer_names if 'module_' in name)
                
                # Should have layers for each module (allowing for E/I split)
                self.assertGreaterEqual(len(module_layers), num_modules)
                
    def test_modular_ei_balance(self):
        """Test that modular networks maintain E/I balance."""
        network = self.builder.create_modular_cortical_network(
            total_neurons=100,
            num_modules=4,
            excitatory_fraction=0.75  # Custom E/I ratio
        )
        
        # Get network statistics
        stats = self.builder.get_network_statistics()
        
        # Should maintain E/I balance
        self.assertIn('ei_balance', stats)
        ei_balance = stats['ei_balance']
        
        # Should be reasonably close to target
        actual_e_fraction = ei_balance['actual_excitatory_fraction']
        self.assertAlmostEqual(actual_e_fraction, 0.75, places=1)
        
    def test_modular_network_simulation_compatibility(self):
        """Test that modular networks can be simulated."""
        network = self.builder.create_modular_cortical_network(
            total_neurons=40,
            num_modules=2
        )
        
        # Should be able to run simulation
        try:
            results = network.run_simulation(duration=5.0, dt=0.1)
            
            # Should return valid results
            self.assertIn('duration', results)
            self.assertIn('layer_spike_times', results)
            self.assertEqual(results['duration'], 5.0)
            
        except Exception as e:
            self.fail(f"Failed to simulate modular network: {e}")


class TestModularNetworkEdgeCases(unittest.TestCase):
    """Test edge cases for modular network architecture."""
    
    def setUp(self):
        """Set up test parameters."""
        self.modular_arch = ModularNetworkArchitecture()
        
    def test_single_module_network(self):
        """Test behavior with single module."""
        positions, modules = self.modular_arch.create_modular_layout(
            total_neurons=50, num_modules=1
        )
        
        # Should create single module
        self.assertEqual(len(modules), 1)
        self.assertEqual(len(positions), 50)
        
        # All neurons should be in the single module
        module = list(modules.values())[0]
        self.assertEqual(len(module.neuron_ids), 50)
        
    def test_more_modules_than_neurons(self):
        """Test behavior when requesting more modules than neurons."""
        positions, modules = self.modular_arch.create_modular_layout(
            total_neurons=3, num_modules=5
        )
        
        # Should create modules but some may be empty
        self.assertEqual(len(modules), 5)
        self.assertEqual(len(positions), 3)
        
        # Total neurons in modules should equal input
        total_module_neurons = sum(len(module.neuron_ids) for module in modules.values())
        self.assertEqual(total_module_neurons, 3)
        
    def test_empty_connections(self):
        """Test network analysis with no connections."""
        empty_connections = {}
        modules = {
            'module_0': NetworkModule('module_0', SpatialPosition(0, 0, 0), 10, [0, 1])
        }
        
        analysis = self.modular_arch.analyze_network_properties(empty_connections, modules)
        
        # Should handle gracefully
        self.assertEqual(analysis['total_connections'], 0)
        self.assertEqual(analysis['clustering_coefficient'], 0.0)
        
    def test_very_small_networks(self):
        """Test with very small networks."""
        builder = BrainInspiredNetworkBuilder()
        
        network = builder.create_modular_cortical_network(
            total_neurons=10,
            num_modules=2
        )
        
        # Should create network successfully
        self.assertIsNotNone(network)
        
        # Should have some layers
        self.assertGreater(len(network.layers), 0)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)