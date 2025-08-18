#!/usr/bin/env python3
"""
Test suite for distance-dependent connectivity mechanisms.

This module tests the distance-dependent connectivity system to ensure it:
- Creates spatial layouts for positioning neurons in 2D/3D space
- Implements exponential decay connection probability with distance
- Supports configurable spatial scales for different connection types
- Shows realistic distance dependence in connectivity patterns
"""

import unittest
import numpy as np
import sys
import os
import math

# Add the parent directory to the path so we can import the core modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.brain_topology import (
    SpatialPosition,
    SpatialNetworkLayout,
    DistanceDependentConnectivity,
    BrainInspiredNetworkBuilder,
    NeuronType,
    NetworkModule
)


class TestSpatialPosition(unittest.TestCase):
    """Test spatial position calculations."""
    
    def test_distance_calculation(self):
        """Test distance calculation between positions."""
        pos1 = SpatialPosition(0.0, 0.0, 0.0)
        pos2 = SpatialPosition(3.0, 4.0, 0.0)
        
        # Should calculate Euclidean distance correctly
        distance = pos1.distance_to(pos2)
        expected_distance = 5.0  # 3-4-5 triangle
        self.assertAlmostEqual(distance, expected_distance, places=6)
        
        # Test 3D distance
        pos3 = SpatialPosition(1.0, 1.0, 1.0)
        pos4 = SpatialPosition(4.0, 5.0, 5.0)
        
        distance_3d = pos3.distance_to(pos4)
        expected_3d = math.sqrt(3**2 + 4**2 + 4**2)  # sqrt(9 + 16 + 16) = sqrt(41)
        self.assertAlmostEqual(distance_3d, expected_3d, places=6)
        
    def test_distance_symmetry(self):
        """Test that distance calculation is symmetric."""
        pos1 = SpatialPosition(1.0, 2.0, 3.0)
        pos2 = SpatialPosition(4.0, 6.0, 8.0)
        
        dist1to2 = pos1.distance_to(pos2)
        dist2to1 = pos2.distance_to(pos1)
        
        self.assertAlmostEqual(dist1to2, dist2to1, places=10)


class TestSpatialNetworkLayout(unittest.TestCase):
    """Test spatial network layout system."""
    
    def setUp(self):
        """Set up test parameters."""
        self.layout = SpatialNetworkLayout(dimensions=2, bounds=(100.0, 100.0, 10.0))
        
    def test_grid_layout_2d(self):
        """Test 2D grid layout creation."""
        num_neurons = 16
        spacing = 5.0
        
        positions = self.layout.create_grid_layout(num_neurons, spacing)
        
        # Should create correct number of positions
        self.assertEqual(len(positions), num_neurons)
        
        # Check that positions are on a grid
        for i, pos in positions.items():
            self.assertEqual(pos.z, 0.0)  # 2D layout
            self.assertTrue(pos.x >= 0.0)
            self.assertTrue(pos.y >= 0.0)
            
        # Check spacing between adjacent neurons
        pos0 = positions[0]  # (0, 0, 0)
        pos1 = positions[1]  # (5, 0, 0)
        distance = pos0.distance_to(pos1)
        self.assertAlmostEqual(distance, spacing, places=6)
        
    def test_grid_layout_3d(self):
        """Test 3D grid layout creation."""
        layout_3d = SpatialNetworkLayout(dimensions=3, bounds=(50.0, 50.0, 50.0))
        num_neurons = 27  # 3x3x3 cube
        spacing = 2.0
        
        positions = layout_3d.create_grid_layout(num_neurons, spacing)
        
        # Should create correct number of positions
        self.assertEqual(len(positions), num_neurons)
        
        # Check that positions span 3D space
        z_values = [pos.z for pos in positions.values()]
        self.assertGreater(max(z_values), 0.0)  # Should have non-zero z values
        
    def test_random_layout(self):
        """Test random layout creation."""
        num_neurons = 100
        
        positions = self.layout.create_random_layout(num_neurons)
        
        # Should create correct number of positions
        self.assertEqual(len(positions), num_neurons)
        
        # All positions should be within bounds
        for pos in positions.values():
            self.assertTrue(0 <= pos.x <= self.layout.bounds[0])
            self.assertTrue(0 <= pos.y <= self.layout.bounds[1])
            self.assertTrue(0 <= pos.z <= self.layout.bounds[2])
            
        # Positions should be reasonably distributed (not all the same)
        x_values = [pos.x for pos in positions.values()]
        y_values = [pos.y for pos in positions.values()]
        
        self.assertGreater(np.std(x_values), 5.0)  # Should have some spread
        self.assertGreater(np.std(y_values), 5.0)
        
    def test_clustered_layout(self):
        """Test clustered layout creation."""
        num_neurons = 200
        num_clusters = 4
        cluster_radius = 20.0
        
        positions = self.layout.create_clustered_layout(
            num_neurons, num_clusters, cluster_radius
        )
        
        # Should create correct number of positions
        self.assertEqual(len(positions), num_neurons)
        
        # Should create modules
        self.assertEqual(len(self.layout.modules), num_clusters)
        
        # Check that modules contain correct number of neurons
        total_module_neurons = sum(len(module.neuron_ids) for module in self.layout.modules.values())
        self.assertEqual(total_module_neurons, num_neurons)
        
        # Check that neurons are clustered around module centers
        for module in self.layout.modules.values():
            distances_to_center = []
            for neuron_id in module.neuron_ids:
                pos = positions[neuron_id]
                distance = pos.distance_to(module.center_position)
                distances_to_center.append(distance)
                
            # Most neurons should be within reasonable distance of center
            mean_distance = np.mean(distances_to_center)
            self.assertLess(mean_distance, cluster_radius)
            
    def test_cortical_column_layout(self):
        """Test cortical column layout creation."""
        num_neurons = 300
        num_layers = 6
        column_radius = 15.0
        layer_thickness = 5.0
        
        positions = self.layout.create_cortical_column_layout(
            num_neurons, num_layers, column_radius, layer_thickness
        )
        
        # Should create correct number of positions
        self.assertEqual(len(positions), num_neurons)
        
        # Check layered structure
        z_values = [pos.z for pos in positions.values()]
        unique_layers = len(set(int(z / layer_thickness) for z in z_values))
        self.assertLessEqual(unique_layers, num_layers)
        
        # Check that neurons are within column radius
        center_x = self.layout.bounds[0] / 2
        center_y = self.layout.bounds[1] / 2
        
        for pos in positions.values():
            distance_from_center = math.sqrt((pos.x - center_x)**2 + (pos.y - center_y)**2)
            self.assertLessEqual(distance_from_center, column_radius)
            
    def test_distance_matrix_calculation(self):
        """Test distance matrix calculation."""
        # Create simple layout
        positions = self.layout.create_grid_layout(4, spacing=10.0)
        neuron_ids = list(positions.keys())
        
        distance_matrix = self.layout.get_distance_matrix(neuron_ids)
        
        # Should be square matrix
        self.assertEqual(distance_matrix.shape, (4, 4))
        
        # Diagonal should be zero
        for i in range(4):
            self.assertEqual(distance_matrix[i, i], 0.0)
            
        # Should be symmetric
        for i in range(4):
            for j in range(4):
                self.assertAlmostEqual(distance_matrix[i, j], distance_matrix[j, i], places=10)
                
        # Check specific distances
        # Neurons 0 and 1 should be 10.0 apart (adjacent in grid)
        self.assertAlmostEqual(distance_matrix[0, 1], 10.0, places=6)


class TestDistanceDependentConnectivity(unittest.TestCase):
    """Test distance-dependent connectivity builder."""
    
    def setUp(self):
        """Set up test parameters."""
        self.layout = SpatialNetworkLayout(dimensions=2, bounds=(100.0, 100.0, 10.0))
        self.layout.create_grid_layout(16, spacing=10.0)  # 4x4 grid
        self.connectivity = DistanceDependentConnectivity(self.layout)
        
    def test_connection_probability_calculation(self):
        """Test connection probability based on distance."""
        # Test close neurons (should have high probability)
        prob_close = self.connectivity.compute_connection_probability(0, 1, 'E_to_E')
        
        # Test distant neurons (should have lower probability)
        prob_distant = self.connectivity.compute_connection_probability(0, 15, 'E_to_E')
        
        # Close neurons should have higher connection probability
        self.assertGreater(prob_close, prob_distant)
        
        # Both should be valid probabilities
        self.assertTrue(0.0 <= prob_close <= 1.0)
        self.assertTrue(0.0 <= prob_distant <= 1.0)
        
    def test_connection_probability_decay(self):
        """Test exponential decay of connection probability with distance."""
        # Test multiple distances
        probabilities = []
        distances = []
        
        for target_id in range(1, 8):  # Test different distances from neuron 0
            prob = self.connectivity.compute_connection_probability(0, target_id, 'E_to_E')
            distance = self.layout.neuron_positions[0].distance_to(
                self.layout.neuron_positions[target_id]
            )
            
            probabilities.append(prob)
            distances.append(distance)
            
        # Probability should generally decrease with distance
        # (allowing for some noise due to discrete grid positions)
        correlation = np.corrcoef(distances, probabilities)[0, 1]
        self.assertLess(correlation, -0.3)  # Negative correlation
        
    def test_connection_type_differences(self):
        """Test that different connection types have different probabilities."""
        neuron_id_1, neuron_id_2 = 0, 1
        
        prob_e_to_e = self.connectivity.compute_connection_probability(neuron_id_1, neuron_id_2, 'E_to_E')
        prob_e_to_i = self.connectivity.compute_connection_probability(neuron_id_1, neuron_id_2, 'E_to_I')
        prob_i_to_e = self.connectivity.compute_connection_probability(neuron_id_1, neuron_id_2, 'I_to_E')
        prob_i_to_i = self.connectivity.compute_connection_probability(neuron_id_1, neuron_id_2, 'I_to_I')
        
        # All should be valid probabilities
        for prob in [prob_e_to_e, prob_e_to_i, prob_i_to_e, prob_i_to_i]:
            self.assertTrue(0.0 <= prob <= 1.0)
            
        # Should have different values (based on default parameters)
        probabilities = [prob_e_to_e, prob_e_to_i, prob_i_to_e, prob_i_to_i]
        unique_probs = len(set(probabilities))
        self.assertGreater(unique_probs, 1)  # Should have some variation
        
    def test_maximum_distance_cutoff(self):
        """Test that connections beyond maximum distance have zero probability."""
        # Set a small maximum distance
        self.connectivity.set_connection_parameters('E_to_E', max_distance=5.0)
        
        # Test connection beyond maximum distance
        prob = self.connectivity.compute_connection_probability(0, 15, 'E_to_E')  # Far apart
        
        # Should be zero
        self.assertEqual(prob, 0.0)
        
    def test_connection_matrix_generation(self):
        """Test generation of connection and weight matrices."""
        pre_ids = [0, 1, 2, 3]
        post_ids = [4, 5, 6, 7]
        pre_types = [NeuronType.EXCITATORY_PYRAMIDAL] * 4
        post_types = [NeuronType.EXCITATORY_PYRAMIDAL] * 4
        
        conn_matrix, weight_matrix = self.connectivity.generate_connection_matrix(
            pre_ids, post_ids, pre_types, post_types
        )
        
        # Should have correct shape
        self.assertEqual(conn_matrix.shape, (4, 4))
        self.assertEqual(weight_matrix.shape, (4, 4))
        
        # Connection matrix should be boolean
        self.assertTrue(conn_matrix.dtype == bool)
        
        # Weight matrix should have weights only where connections exist
        connected_weights = weight_matrix[conn_matrix]
        unconnected_weights = weight_matrix[~conn_matrix]
        
        if len(connected_weights) > 0:
            self.assertTrue(np.all(connected_weights > 0))  # Excitatory weights should be positive
        if len(unconnected_weights) > 0:
            self.assertTrue(np.all(unconnected_weights == 0))  # No weights where no connections
            
    def test_excitatory_inhibitory_weight_signs(self):
        """Test that excitatory and inhibitory connections have correct weight signs."""
        pre_ids = [0, 1]
        post_ids = [2, 3]
        
        # Test excitatory to excitatory
        exc_types = [NeuronType.EXCITATORY_PYRAMIDAL] * 2
        conn_matrix, weight_matrix = self.connectivity.generate_connection_matrix(
            pre_ids, post_ids, exc_types, exc_types
        )
        
        connected_weights = weight_matrix[conn_matrix]
        if len(connected_weights) > 0:
            self.assertTrue(np.all(connected_weights > 0))  # Should be positive
            
        # Test inhibitory to excitatory
        inh_types = [NeuronType.INHIBITORY_BASKET] * 2
        conn_matrix, weight_matrix = self.connectivity.generate_connection_matrix(
            pre_ids, post_ids, inh_types, exc_types
        )
        
        connected_weights = weight_matrix[conn_matrix]
        if len(connected_weights) > 0:
            self.assertTrue(np.all(connected_weights < 0))  # Should be negative
            
    def test_parameter_customization(self):
        """Test customization of connection parameters."""
        # Set custom parameters
        custom_params = {
            'base_probability': 0.5,
            'spatial_scale': 100.0,
            'max_distance': 500.0
        }
        
        self.connectivity.set_connection_parameters('E_to_E', **custom_params)
        
        # Check that parameters were updated
        updated_params = self.connectivity.connection_params['E_to_E']
        for key, value in custom_params.items():
            self.assertEqual(updated_params[key], value)
            
        # Test that connection probability reflects new parameters
        prob = self.connectivity.compute_connection_probability(0, 15, 'E_to_E')
        
        # With higher base probability and spatial scale, should get higher probability
        self.assertGreater(prob, 0.1)  # Should be reasonably high


class TestBrainInspiredNetworkBuilder(unittest.TestCase):
    """Test brain-inspired network builder."""
    
    def setUp(self):
        """Set up test parameters."""
        self.builder = BrainInspiredNetworkBuilder()
        
    def test_cortical_network_creation(self):
        """Test creation of cortical-like network."""
        total_neurons = 100
        excitatory_fraction = 0.8
        
        network = self.builder.create_cortical_network(
            total_neurons=total_neurons,
            excitatory_fraction=excitatory_fraction,
            layout_type="clustered",
            num_clusters=2
        )
        
        # Should create network with correct structure
        self.assertIsNotNone(network)
        self.assertIn("excitatory", network.layers)
        self.assertIn("inhibitory", network.layers)
        
        # Check layer sizes
        exc_size = network.layers["excitatory"].size
        inh_size = network.layers["inhibitory"].size
        
        self.assertEqual(exc_size + inh_size, total_neurons)
        self.assertAlmostEqual(exc_size / total_neurons, excitatory_fraction, places=1)
        
        # Should have connections between layers
        self.assertGreater(len(network.connections), 0)
        
    def test_different_layout_types(self):
        """Test different spatial layout types."""
        layout_types = ["grid", "random", "clustered", "cortical"]
        
        for layout_type in layout_types:
            with self.subTest(layout_type=layout_type):
                try:
                    network = self.builder.create_cortical_network(
                        total_neurons=50,
                        layout_type=layout_type
                    )
                    
                    # Should successfully create network
                    self.assertIsNotNone(network)
                    self.assertGreater(len(network.layers), 0)
                    
                except Exception as e:
                    self.fail(f"Failed to create network with {layout_type} layout: {e}")
                    
    def test_network_statistics(self):
        """Test network statistics generation."""
        network = self.builder.create_cortical_network(
            total_neurons=80,
            layout_type="clustered",
            num_clusters=3
        )
        
        stats = self.builder.get_network_statistics()
        
        # Should contain expected statistics
        expected_keys = ['total_neurons', 'total_connections', 'spatial_bounds', 'modules']
        for key in expected_keys:
            self.assertIn(key, stats)
            
        # Check values
        self.assertEqual(stats['total_neurons'], 80)
        self.assertEqual(stats['modules'], 3)
        self.assertGreater(stats['total_connections'], 0)
        
    def test_excitatory_inhibitory_balance(self):
        """Test that E/I balance is maintained."""
        total_neurons = 200
        excitatory_fraction = 0.75  # Custom fraction
        
        network = self.builder.create_cortical_network(
            total_neurons=total_neurons,
            excitatory_fraction=excitatory_fraction
        )
        
        exc_size = network.layers["excitatory"].size
        inh_size = network.layers["inhibitory"].size
        
        actual_exc_fraction = exc_size / (exc_size + inh_size)
        
        # Should be close to requested fraction
        self.assertAlmostEqual(actual_exc_fraction, excitatory_fraction, places=1)
        
    def test_spatial_bounds_respect(self):
        """Test that spatial bounds are respected."""
        bounds = (50.0, 60.0, 20.0)
        
        network = self.builder.create_cortical_network(
            total_neurons=100,
            spatial_bounds=bounds,
            layout_type="random"
        )
        
        # Check that spatial layout uses correct bounds
        self.assertEqual(self.builder.spatial_layout.bounds, bounds)
        
        # Check that all positions are within bounds
        for pos in self.builder.spatial_layout.neuron_positions.values():
            self.assertTrue(0 <= pos.x <= bounds[0])
            self.assertTrue(0 <= pos.y <= bounds[1])
            self.assertTrue(0 <= pos.z <= bounds[2])


class TestNetworkModuleIntegration(unittest.TestCase):
    """Test integration with existing network module."""
    
    def test_network_simulation_compatibility(self):
        """Test that brain-inspired networks can be simulated."""
        builder = BrainInspiredNetworkBuilder()
        network = builder.create_cortical_network(
            total_neurons=50,
            layout_type="grid"
        )
        
        # Should be able to run simulation
        try:
            results = network.run_simulation(duration=10.0, dt=0.1)
            
            # Should return valid results
            self.assertIn('duration', results)
            self.assertIn('layer_spike_times', results)
            self.assertEqual(results['duration'], 10.0)
            
        except Exception as e:
            self.fail(f"Failed to simulate brain-inspired network: {e}")
            
    def test_network_reset_functionality(self):
        """Test that brain-inspired networks can be reset."""
        builder = BrainInspiredNetworkBuilder()
        network = builder.create_cortical_network(total_neurons=30)
        
        # Run brief simulation
        network.run_simulation(duration=5.0, dt=0.1)
        
        # Reset network
        network.reset()
        
        # Should be back to initial state
        self.assertEqual(network.current_time, 0.0)
        self.assertEqual(len(network.simulation_history), 0)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)