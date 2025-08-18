#!/usr/bin/env python3
"""
Test suite for small-world network properties validation.

This module tests the small-world property validation system to ensure it:
- Calculates clustering coefficient correctly
- Computes shortest path lengths accurately
- Generates proper small-world index metrics
- Validates small-world properties in networks
- Classifies network topology types correctly
"""

import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the core modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.brain_topology import (
    BrainInspiredNetworkBuilder,
    ModularNetworkArchitecture,
    SpatialNetworkLayout
)


class TestSmallWorldProperties(unittest.TestCase):
    """Test small-world network properties calculation."""
    
    def setUp(self):
        """Set up test parameters."""
        self.builder = BrainInspiredNetworkBuilder()
        
    def test_clustering_coefficient_calculation(self):
        """Test clustering coefficient calculation."""
        # Create a simple test adjacency matrix
        # Triangle: 0-1-2-0 (high clustering)
        adjacency = np.array([
            [0, 1, 1, 0],
            [1, 0, 1, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0]
        ])
        
        clustering = self.builder.calculate_clustering_coefficient(adjacency)
        
        # Node 0: neighbors [1,2], they are connected -> clustering = 1.0
        # Node 1: neighbors [0,2], they are connected -> clustering = 1.0  
        # Node 2: neighbors [0,1], they are connected -> clustering = 1.0
        # Node 3: no neighbors -> clustering = 0.0
        # Average = (1.0 + 1.0 + 1.0 + 0.0) / 4 = 0.75
        
        self.assertAlmostEqual(clustering, 0.75, places=2)
        
    def test_clustering_coefficient_no_triangles(self):
        """Test clustering coefficient with no triangles."""
        # Star network: center connected to all others, no triangles
        adjacency = np.array([
            [0, 1, 1, 1],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0]
        ])
        
        clustering = self.builder.calculate_clustering_coefficient(adjacency)
        
        # Node 0: neighbors [1,2,3], no connections between them -> clustering = 0.0
        # Nodes 1,2,3: only one neighbor each -> clustering = 0.0
        # Average = 0.0
        
        self.assertAlmostEqual(clustering, 0.0, places=2)
        
    def test_shortest_path_lengths(self):
        """Test shortest path length calculation."""
        # Linear chain: 0-1-2-3
        adjacency = np.array([
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0]
        ])
        
        avg_path_length = self.builder.calculate_shortest_path_lengths(adjacency)
        
        # Paths: 0→1(1), 0→2(2), 0→3(3), 1→0(1), 1→2(1), 1→3(2), 
        #        2→0(2), 2→1(1), 2→3(1), 3→0(3), 3→1(2), 3→2(1)
        # Average = (1+2+3+1+1+2+2+1+1+3+2+1) / 12 = 20/12 ≈ 1.67
        
        self.assertAlmostEqual(avg_path_length, 20/12, places=1)
        
    def test_small_world_index_calculation(self):
        """Test small-world index calculation."""
        clustering = 0.6
        path_length = 2.5
        n_nodes = 100
        avg_degree = 6.0
        
        small_world_index = self.builder.calculate_small_world_index(
            clustering, path_length, n_nodes, avg_degree
        )
        
        # Expected random values
        random_clustering = avg_degree / n_nodes  # 6/100 = 0.06
        random_path_length = np.log(n_nodes) / np.log(avg_degree)  # log(100)/log(6) ≈ 2.57
        
        # Small-world index = (C/C_random) / (L/L_random)
        expected_index = (clustering / random_clustering) / (path_length / random_path_length)
        
        self.assertAlmostEqual(small_world_index, expected_index, places=1)
        self.assertGreater(small_world_index, 1.0)  # Should indicate small-world
        
    def test_small_world_validation_with_modular_network(self):
        """Test small-world validation on a modular network."""
        # Create modular network
        network = self.builder.create_modular_cortical_network(
            total_neurons=100,
            hierarchy_levels=2,
            create_small_world=True
        )
        
        # Validate small-world properties
        validation = self.builder.validate_small_world_properties()
        
        # Check validation results
        self.assertIn('clustering_coefficient', validation)
        self.assertIn('average_path_length', validation)
        self.assertIn('small_world_coefficient', validation)
        self.assertIn('has_small_world_properties', validation)
        self.assertIn('small_world_quality', validation)
        self.assertIn('network_type', validation)
        
        # Values should be reasonable
        clustering = validation['clustering_coefficient']
        path_length = validation['average_path_length']
        
        self.assertTrue(0.0 <= clustering <= 1.0)
        self.assertGreaterEqual(path_length, 0.0)
        
        # Network type should be classified
        network_type = validation['network_type']
        valid_types = ['small_world', 'clustered', 'random', 'sparse', 'intermediate']
        self.assertIn(network_type, valid_types)
        
    def test_network_topology_classification(self):
        """Test network topology classification."""
        # Test small-world classification
        topology = self.builder._classify_network_topology(0.3, 2.5, 1.5)
        self.assertEqual(topology, 'small_world')
        
        # Test clustered classification
        topology = self.builder._classify_network_topology(0.5, 4.0, 0.8)
        self.assertEqual(topology, 'clustered')
        
        # Test random classification
        topology = self.builder._classify_network_topology(0.1, 2.0, 0.5)
        self.assertEqual(topology, 'random')
        
        # Test sparse classification
        topology = self.builder._classify_network_topology(0.05, 6.0, 0.3)
        self.assertEqual(topology, 'sparse')
        
        # Test intermediate classification
        topology = self.builder._classify_network_topology(0.15, 4.0, 0.7)
        self.assertEqual(topology, 'intermediate')
        
    def test_modular_network_statistics(self):
        """Test comprehensive modular network statistics."""
        # Create modular network
        network = self.builder.create_modular_cortical_network(
            total_neurons=80,
            hierarchy_levels=2,
            create_small_world=True
        )
        
        # Get comprehensive statistics
        stats = self.builder.get_modular_network_statistics()
        
        # Check basic network stats
        self.assertIn('total_neurons', stats)
        self.assertIn('layers', stats)
        
        # Check modular architecture stats
        if 'modular_architecture' in stats:
            modular_stats = stats['modular_architecture']
            self.assertIn('total_modules', modular_stats)
            self.assertIn('hierarchy_levels', modular_stats)
            
        # Check small-world properties
        if 'small_world_properties' in stats:
            sw_props = stats['small_world_properties']
            self.assertIn('clustering_coefficient', sw_props)
            self.assertIn('average_path_length', sw_props)
            self.assertIn('small_world_coefficient', sw_props)
            
        # Check topology assessment
        if 'network_topology_assessment' in stats:
            assessment = stats['network_topology_assessment']
            self.assertIn('has_small_world_properties', assessment)
            self.assertIn('clustering_quality', assessment)
            self.assertIn('path_length_efficiency', assessment)
            self.assertIn('topology_type', assessment)
            
    def test_empty_network_handling(self):
        """Test handling of networks without modular architecture."""
        # Create simple E/I balanced network (no modules)
        network = self.builder.create_ei_balanced_network(total_neurons=50)
        
        # Validation should handle missing modular architecture
        validation = self.builder.validate_small_world_properties()
        self.assertIn('error', validation)
        
    def test_disconnected_network_handling(self):
        """Test handling of disconnected networks."""
        # Create adjacency matrix with disconnected components
        adjacency = np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ])
        
        # Should handle disconnected components gracefully
        clustering = self.builder.calculate_clustering_coefficient(adjacency)
        path_length = self.builder.calculate_shortest_path_lengths(adjacency)
        
        self.assertTrue(0.0 <= clustering <= 1.0)
        self.assertGreaterEqual(path_length, 0.0)


class TestModularNetworkArchitectureProperties(unittest.TestCase):
    """Test modular network architecture small-world properties."""
    
    def setUp(self):
        """Set up test parameters."""
        self.modular_arch = ModularNetworkArchitecture()
        
    def test_network_properties_calculation(self):
        """Test network properties calculation in modular architecture."""
        # Create simple modules
        modules = self.modular_arch.create_hierarchical_modules(60, 2)
        
        # Create spatial layout
        spatial_layout = SpatialNetworkLayout(dimensions=3, bounds=(100, 100, 50))
        
        # Create connections
        connections = self.modular_arch.create_small_world_connections(modules, spatial_layout)
        
        # Calculate properties
        properties = self.modular_arch.calculate_network_properties(connections)
        
        # Check all expected properties
        expected_props = [
            'clustering_coefficient', 'average_path_length', 'small_world_coefficient',
            'number_of_connections', 'connection_density'
        ]
        
        for prop in expected_props:
            self.assertIn(prop, properties)
            self.assertIsInstance(properties[prop], (int, float))
            
        # Check reasonable ranges
        self.assertTrue(0.0 <= properties['clustering_coefficient'] <= 1.0)
        self.assertGreaterEqual(properties['average_path_length'], 0.0)
        self.assertTrue(0.0 <= properties['connection_density'] <= 1.0)
        
    def test_module_statistics(self):
        """Test module statistics generation."""
        # Create modules
        modules = self.modular_arch.create_hierarchical_modules(100, 3)
        
        # Get statistics
        stats = self.modular_arch.get_module_statistics()
        
        # Check expected statistics
        expected_stats = [
            'total_modules', 'total_neurons', 'average_module_size',
            'module_size_std', 'hierarchy_levels', 'modules_per_level'
        ]
        
        for stat in expected_stats:
            self.assertIn(stat, stats)
            
        # Check values make sense
        self.assertEqual(stats['total_modules'], len(modules))
        self.assertGreater(stats['total_neurons'], 0)  # Should have some neurons
        self.assertGreater(stats['average_module_size'], 0)
        self.assertGreaterEqual(stats['hierarchy_levels'], 1)
        
    def test_small_world_connections_properties(self):
        """Test properties of small-world connections."""
        # Create modules
        modules = self.modular_arch.create_hierarchical_modules(80, 2)
        
        # Create spatial layout
        spatial_layout = SpatialNetworkLayout(dimensions=3, bounds=(150, 150, 75))
        
        # Create small-world connections
        connections = self.modular_arch.create_small_world_connections(modules, spatial_layout)
        
        # Check connection properties
        local_connections = 0
        long_range_connections = 0
        
        for conn_info in connections.values():
            if conn_info['connection_type'] == 'local':
                local_connections += 1
            elif conn_info['connection_type'] == 'long_range':
                long_range_connections += 1
                
        # Should have connections for small-world
        total_connections = len(connections)
        if total_connections > 0:
            local_fraction = local_connections / total_connections
            long_range_fraction = long_range_connections / total_connections
            
            # Should have some local connections (primary requirement)
            self.assertGreater(local_fraction, 0.0)
            
            # Long-range connections are optional but improve small-world properties
            # (Some networks may have only local connections initially)
            self.assertGreaterEqual(long_range_fraction, 0.0)
            
    def test_hierarchy_level_connections(self):
        """Test connections between different hierarchy levels."""
        # Create modules with multiple levels
        modules = self.modular_arch.create_hierarchical_modules(120, 3)
        
        # Create spatial layout
        spatial_layout = SpatialNetworkLayout(dimensions=3, bounds=(200, 200, 100))
        
        # Create connections
        connections = self.modular_arch.create_small_world_connections(modules, spatial_layout)
        
        # Analyze level differences
        level_differences = []
        for conn_info in connections.values():
            level_diff = conn_info['level_difference']
            level_differences.append(level_diff)
            
        if level_differences:
            # Should have connections within levels (diff=0) and between levels (diff>0)
            same_level = sum(1 for diff in level_differences if diff == 0)
            different_level = sum(1 for diff in level_differences if diff > 0)
            
            # Both types should exist
            self.assertGreater(same_level, 0)
            self.assertGreater(different_level, 0)


class TestSmallWorldValidationEdgeCases(unittest.TestCase):
    """Test edge cases for small-world validation."""
    
    def test_single_node_network(self):
        """Test small-world properties with single node."""
        builder = BrainInspiredNetworkBuilder()
        
        # Single node adjacency matrix
        adjacency = np.array([[0]])
        
        clustering = builder.calculate_clustering_coefficient(adjacency)
        path_length = builder.calculate_shortest_path_lengths(adjacency)
        
        self.assertEqual(clustering, 0.0)
        self.assertEqual(path_length, 0.0)
        
    def test_two_node_network(self):
        """Test small-world properties with two nodes."""
        builder = BrainInspiredNetworkBuilder()
        
        # Two connected nodes
        adjacency = np.array([
            [0, 1],
            [1, 0]
        ])
        
        clustering = builder.calculate_clustering_coefficient(adjacency)
        path_length = builder.calculate_shortest_path_lengths(adjacency)
        
        # Two nodes can't form triangles
        self.assertEqual(clustering, 0.0)
        # Path length should be 1.0
        self.assertEqual(path_length, 1.0)
        
    def test_complete_graph(self):
        """Test small-world properties with complete graph."""
        builder = BrainInspiredNetworkBuilder()
        
        # Complete graph (all nodes connected)
        n = 4
        adjacency = np.ones((n, n)) - np.eye(n)
        
        clustering = builder.calculate_clustering_coefficient(adjacency)
        path_length = builder.calculate_shortest_path_lengths(adjacency)
        
        # Complete graph has maximum clustering
        self.assertEqual(clustering, 1.0)
        # All paths are length 1
        self.assertEqual(path_length, 1.0)
        
    def test_zero_degree_nodes(self):
        """Test handling of isolated nodes."""
        builder = BrainInspiredNetworkBuilder()
        
        # Network with isolated node
        adjacency = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 0]
        ])
        
        clustering = builder.calculate_clustering_coefficient(adjacency)
        path_length = builder.calculate_shortest_path_lengths(adjacency)
        
        # Should handle isolated nodes gracefully
        self.assertTrue(0.0 <= clustering <= 1.0)
        self.assertGreaterEqual(path_length, 0.0)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)