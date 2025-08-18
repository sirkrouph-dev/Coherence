#!/usr/bin/env python3
"""
Test suite for excitatory/inhibitory balance mechanisms.

This module tests the E/I balance system to ensure it:
- Implements 80% excitatory, 20% inhibitory neuron populations
- Sets appropriate connection probabilities for E→E, E→I, I→E, I→I
- Creates different inhibitory neuron types (basket, chandelier cells)
- Maintains stable activity with proper E/I balance
"""

import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the core modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.brain_topology import (
    ExcitatoryInhibitoryBalance,
    BrainInspiredNetworkBuilder,
    NeuronType
)


class TestExcitatoryInhibitoryBalance(unittest.TestCase):
    """Test excitatory/inhibitory balance system."""
    
    def setUp(self):
        """Set up test parameters."""
        self.ei_balance = ExcitatoryInhibitoryBalance()
        
    def test_default_ei_ratios(self):
        """Test default E/I ratios are correct."""
        self.assertAlmostEqual(self.ei_balance.excitatory_fraction, 0.8, places=2)
        self.assertAlmostEqual(self.ei_balance.inhibitory_fraction, 0.2, places=2)
        self.assertAlmostEqual(
            self.ei_balance.excitatory_fraction + self.ei_balance.inhibitory_fraction, 
            1.0, places=10
        )
        
    def test_population_size_calculation(self):
        """Test calculation of population sizes."""
        total_neurons = 1000
        sizes = self.ei_balance.calculate_population_sizes(total_neurons)
        
        # Check total adds up
        calculated_total = sizes['excitatory'] + sizes['inhibitory_total']
        self.assertEqual(calculated_total, total_neurons)
        
        # Check E/I ratio
        actual_e_fraction = sizes['excitatory'] / total_neurons
        self.assertAlmostEqual(actual_e_fraction, 0.8, places=1)
        
        # Check inhibitory subtypes add up
        inhibitory_subtypes_total = sizes['basket'] + sizes['chandelier'] + sizes['martinotti']
        self.assertEqual(inhibitory_subtypes_total, sizes['inhibitory_total'])
        
        # Check inhibitory type distributions
        basket_fraction = sizes['basket'] / sizes['inhibitory_total']
        chandelier_fraction = sizes['chandelier'] / sizes['inhibitory_total']
        martinotti_fraction = sizes['martinotti'] / sizes['inhibitory_total']
        
        self.assertAlmostEqual(basket_fraction, 0.6, places=1)
        self.assertAlmostEqual(chandelier_fraction, 0.25, places=1)
        self.assertAlmostEqual(martinotti_fraction, 0.15, places=1)
        
    def test_connection_parameters(self):
        """Test connection parameters for different E/I combinations."""
        # Test E→E connections
        ee_params = self.ei_balance.get_connection_parameters('excitatory', 'excitatory')
        self.assertEqual(ee_params['connection_type'], 'E_to_E')
        self.assertGreater(ee_params['connection_probability'], 0.0)
        self.assertGreater(ee_params['synaptic_strength'], 0.0)  # Excitatory
        
        # Test E→I connections
        ei_params = self.ei_balance.get_connection_parameters('excitatory', 'inhibitory')
        self.assertEqual(ei_params['connection_type'], 'E_to_I')
        self.assertGreater(ei_params['connection_probability'], ee_params['connection_probability'])  # Should be higher
        self.assertGreater(ei_params['synaptic_strength'], 0.0)  # Excitatory
        
        # Test I→E connections
        ie_params = self.ei_balance.get_connection_parameters('inhibitory', 'excitatory')
        self.assertEqual(ie_params['connection_type'], 'I_to_E')
        self.assertGreater(ie_params['connection_probability'], 0.0)
        self.assertLess(ie_params['synaptic_strength'], 0.0)  # Inhibitory (negative)
        
        # Test I→I connections
        ii_params = self.ei_balance.get_connection_parameters('inhibitory', 'inhibitory')
        self.assertEqual(ii_params['connection_type'], 'I_to_I')
        self.assertGreater(ii_params['connection_probability'], 0.0)
        self.assertLess(ii_params['synaptic_strength'], 0.0)  # Inhibitory (negative)
        
    def test_realistic_connection_probabilities(self):
        """Test that connection probabilities are in realistic ranges."""
        # Based on cortical data, E→I should be higher than E→E
        ee_prob = self.ei_balance.connection_probabilities['E_to_E']
        ei_prob = self.ei_balance.connection_probabilities['E_to_I']
        ie_prob = self.ei_balance.connection_probabilities['I_to_E']
        ii_prob = self.ei_balance.connection_probabilities['I_to_I']
        
        # Check realistic ranges (based on cortical literature)
        self.assertTrue(0.01 <= ee_prob <= 0.1, f"E→E probability {ee_prob} not in realistic range")
        self.assertTrue(0.1 <= ei_prob <= 0.3, f"E→I probability {ei_prob} not in realistic range")
        self.assertTrue(0.1 <= ie_prob <= 0.4, f"I→E probability {ie_prob} not in realistic range")
        self.assertTrue(0.05 <= ii_prob <= 0.2, f"I→I probability {ii_prob} not in realistic range")
        
        # Check relative ordering
        self.assertGreater(ei_prob, ee_prob, "E→I should be more probable than E→E")
        self.assertGreater(ie_prob, ee_prob, "I→E should be more probable than E→E")
        
    def test_synaptic_strength_ratios(self):
        """Test that synaptic strengths have realistic ratios."""
        ee_strength = self.ei_balance.synaptic_strengths['E_to_E']
        ei_strength = self.ei_balance.synaptic_strengths['E_to_I']
        ie_strength = self.ei_balance.synaptic_strengths['I_to_E']
        ii_strength = self.ei_balance.synaptic_strengths['I_to_I']
        
        # Excitatory strengths should be positive
        self.assertGreater(ee_strength, 0.0)
        self.assertGreater(ei_strength, 0.0)
        
        # Inhibitory strengths should be negative
        self.assertLess(ie_strength, 0.0)
        self.assertLess(ii_strength, 0.0)
        
        # Check relative magnitudes
        self.assertGreater(abs(ie_strength), abs(ee_strength), "Inhibition should be stronger than excitation")
        
    def test_inhibitory_type_distributions(self):
        """Test inhibitory neuron type distributions."""
        types = self.ei_balance.inhibitory_types
        
        # Should sum to 1.0
        total_fraction = sum(types.values())
        self.assertAlmostEqual(total_fraction, 1.0, places=10)
        
        # Check individual fractions are reasonable
        self.assertGreater(types['basket'], 0.5, "Basket cells should be majority")
        self.assertGreater(types['chandelier'], 0.1, "Chandelier cells should be present")
        self.assertGreater(types['martinotti'], 0.1, "Martinotti cells should be present")
        
    def test_balance_validation(self):
        """Test E/I balance validation."""
        # Create mock network stats
        mock_stats = {
            'total_neurons': 1000,
            'layers': {
                'excitatory': {'size': 800},
                'inhibitory': {'size': 200}
            }
        }
        
        validation = self.ei_balance.validate_ei_balance(mock_stats)
        
        # Should detect proper balance
        self.assertTrue(validation['is_balanced'])
        self.assertAlmostEqual(validation['actual_excitatory_fraction'], 0.8, places=2)
        self.assertAlmostEqual(validation['actual_inhibitory_fraction'], 0.2, places=2)
        self.assertLess(validation['excitatory_balance_error'], 0.05)
        self.assertLess(validation['inhibitory_balance_error'], 0.05)
        self.assertGreater(validation['balance_quality'], 0.9)
        
    def test_imbalanced_network_detection(self):
        """Test detection of imbalanced networks."""
        # Create imbalanced network stats
        imbalanced_stats = {
            'total_neurons': 1000,
            'layers': {
                'excitatory': {'size': 900},  # Too many excitatory
                'inhibitory': {'size': 100}   # Too few inhibitory
            }
        }
        
        validation = self.ei_balance.validate_ei_balance(imbalanced_stats)
        
        # Should detect imbalance
        self.assertFalse(validation['is_balanced'])
        self.assertGreater(validation['excitatory_balance_error'], 0.05)
        self.assertGreater(validation['inhibitory_balance_error'], 0.05)
        self.assertLess(validation['balance_quality'], 0.9)


class TestBrainInspiredNetworkBuilderEIBalance(unittest.TestCase):
    """Test E/I balance in brain-inspired network builder."""
    
    def setUp(self):
        """Set up test parameters."""
        self.builder = BrainInspiredNetworkBuilder()
        
    def test_default_ei_balance_network_creation(self):
        """Test network creation with default E/I balance."""
        total_neurons = 200
        
        network = self.builder.create_cortical_network(
            total_neurons=total_neurons,
            layout_type="random"
        )
        
        # Check that network was created
        self.assertIsNotNone(network)
        
        # Get statistics and validate E/I balance
        stats = self.builder.get_network_statistics()
        
        self.assertEqual(stats['total_neurons'], total_neurons)
        self.assertIn('ei_balance', stats)
        
        ei_balance = stats['ei_balance']
        self.assertTrue(ei_balance.get('is_balanced', False), 
                       f"Network not balanced: {ei_balance}")
        
    def test_custom_ei_fraction(self):
        """Test network creation with custom E/I fraction."""
        total_neurons = 100
        custom_e_fraction = 0.75
        
        network = self.builder.create_cortical_network(
            total_neurons=total_neurons,
            excitatory_fraction=custom_e_fraction,
            layout_type="grid"
        )
        
        stats = self.builder.get_network_statistics()
        ei_balance = stats['ei_balance']
        
        # Should be close to custom fraction
        actual_e_fraction = ei_balance['actual_excitatory_fraction']
        self.assertAlmostEqual(actual_e_fraction, custom_e_fraction, places=1)
        
    def test_detailed_inhibitory_types(self):
        """Test network creation with detailed inhibitory types."""
        total_neurons = 150
        
        network = self.builder.create_cortical_network(
            total_neurons=total_neurons,
            use_detailed_inhibitory_types=True,
            layout_type="clustered",
            num_clusters=2
        )
        
        # Should have separate inhibitory layers
        layer_names = list(network.layers.keys())
        self.assertIn('excitatory', layer_names)
        
        # Check for inhibitory subtypes
        inhibitory_layers = [name for name in layer_names if 'cells' in name]
        self.assertGreater(len(inhibitory_layers), 0, "Should have inhibitory cell layers")
        
        # Possible inhibitory layer names
        possible_inh_layers = ['basket_cells', 'chandelier_cells', 'martinotti_cells']
        for inh_layer in inhibitory_layers:
            self.assertIn(inh_layer, possible_inh_layers)
            
    def test_simple_inhibitory_layer(self):
        """Test network creation with simple inhibitory layer."""
        total_neurons = 80
        
        network = self.builder.create_cortical_network(
            total_neurons=total_neurons,
            use_detailed_inhibitory_types=False,
            layout_type="random"
        )
        
        # Should have simple E/I structure
        layer_names = list(network.layers.keys())
        self.assertIn('excitatory', layer_names)
        self.assertIn('inhibitory', layer_names)
        self.assertEqual(len(layer_names), 2)  # Only two layers
        
    def test_ei_connections_created(self):
        """Test that E/I connections are properly created."""
        total_neurons = 60
        
        network = self.builder.create_cortical_network(
            total_neurons=total_neurons,
            use_detailed_inhibitory_types=False
        )
        
        # Check connections exist
        connections = network.connections
        self.assertGreater(len(connections), 0, "Should have connections")
        
        # Check for E/I connection types
        connection_names = [f"{pre}->{post}" for (pre, post) in connections.keys()]
        
        # Should have at least E→E and some inhibitory connections
        has_ee = any('excitatory->excitatory' in name for name in connection_names)
        has_inhibitory = any('inhibitory' in name for name in connection_names)
        
        self.assertTrue(has_ee, f"Should have E→E connections. Found: {connection_names}")
        self.assertTrue(has_inhibitory, f"Should have inhibitory connections. Found: {connection_names}")
        
    def test_connection_probabilities_applied(self):
        """Test that realistic connection probabilities are applied."""
        total_neurons = 100
        
        network = self.builder.create_cortical_network(
            total_neurons=total_neurons,
            use_detailed_inhibitory_types=False
        )
        
        stats = self.builder.get_network_statistics()
        connection_details = stats.get('connection_details', {})
        
        # Check that connections have realistic probabilities
        for conn_name, details in connection_details.items():
            prob = details.get('connection_probability', 0)
            self.assertTrue(0.01 <= prob <= 0.5, 
                           f"Connection {conn_name} has unrealistic probability {prob}")
            
    def test_network_validation(self):
        """Test network validation functionality."""
        total_neurons = 120
        
        network = self.builder.create_cortical_network(
            total_neurons=total_neurons
        )
        
        validation = self.builder.validate_network_balance()
        
        # Should have validation results
        self.assertIn('ei_balance', validation)
        self.assertIn('overall_quality', validation)
        self.assertIn('recommendations', validation)
        self.assertIn('warnings', validation)
        
        # Quality should be reasonable for a properly constructed network
        quality = validation['overall_quality']
        self.assertIn(quality, ['excellent', 'good', 'acceptable', 'poor'])
        
        # Should have some recommendations or minimal warnings
        recommendations = validation['recommendations']
        warnings = validation['warnings']
        self.assertTrue(len(recommendations) > 0 or len(warnings) == 0)
        
    def test_different_network_sizes(self):
        """Test E/I balance with different network sizes."""
        sizes = [50, 100, 200, 500]
        
        for size in sizes:
            with self.subTest(size=size):
                network = self.builder.create_cortical_network(
                    total_neurons=size,
                    layout_type="random"
                )
                
                stats = self.builder.get_network_statistics()
                ei_balance = stats['ei_balance']
                
                # Should maintain balance regardless of size
                self.assertTrue(ei_balance.get('is_balanced', False),
                               f"Size {size} network not balanced: {ei_balance}")
                
                # Balance quality should be reasonable
                quality = ei_balance.get('balance_quality', 0)
                self.assertGreater(quality, 0.7, 
                                 f"Size {size} network has poor balance quality: {quality}")
                                 
    def test_ei_balance_with_different_layouts(self):
        """Test E/I balance with different spatial layouts."""
        layouts = ["grid", "random", "clustered"]
        
        for layout in layouts:
            with self.subTest(layout=layout):
                try:
                    network = self.builder.create_cortical_network(
                        total_neurons=80,
                        layout_type=layout
                    )
                    
                    stats = self.builder.get_network_statistics()
                    ei_balance = stats['ei_balance']
                    
                    # E/I balance should be maintained regardless of layout
                    self.assertTrue(ei_balance.get('is_balanced', False),
                                   f"Layout {layout} network not balanced")
                                   
                except Exception as e:
                    self.fail(f"Failed to create balanced network with {layout} layout: {e}")


class TestEIBalanceEdgeCases(unittest.TestCase):
    """Test edge cases for E/I balance system."""
    
    def setUp(self):
        """Set up test parameters."""
        self.ei_balance = ExcitatoryInhibitoryBalance()
        
    def test_very_small_networks(self):
        """Test E/I balance with very small networks."""
        small_sizes = [10, 20, 30]
        
        for size in small_sizes:
            with self.subTest(size=size):
                population_sizes = self.ei_balance.calculate_population_sizes(size)
                
                # Should still maintain approximate ratios
                total = population_sizes['excitatory'] + population_sizes['inhibitory_total']
                self.assertEqual(total, size)
                
                # At least some of each type (allowing for rounding)
                self.assertGreater(population_sizes['excitatory'], 0)
                self.assertGreater(population_sizes['inhibitory_total'], 0)
                
    def test_extreme_ei_ratios(self):
        """Test with extreme E/I ratios."""
        # Very high excitatory fraction
        self.ei_balance.excitatory_fraction = 0.95
        self.ei_balance.inhibitory_fraction = 0.05
        
        sizes = self.ei_balance.calculate_population_sizes(100)
        
        self.assertAlmostEqual(sizes['excitatory'] / 100, 0.95, places=1)
        self.assertGreater(sizes['inhibitory_total'], 0)  # Should still have some inhibitory
        
    def test_zero_neurons(self):
        """Test behavior with zero neurons."""
        sizes = self.ei_balance.calculate_population_sizes(0)
        
        for key, value in sizes.items():
            self.assertEqual(value, 0)
            
    def test_validation_with_missing_data(self):
        """Test validation with incomplete network statistics."""
        # Missing layers
        incomplete_stats = {
            'total_neurons': 100
        }
        
        validation = self.ei_balance.validate_ei_balance(incomplete_stats)
        
        # Should handle gracefully
        self.assertIsInstance(validation, dict)
        self.assertIn('actual_excitatory_fraction', validation)
        
    def test_validation_with_zero_neurons(self):
        """Test validation with zero neurons."""
        empty_stats = {
            'total_neurons': 0,
            'layers': {}
        }
        
        validation = self.ei_balance.validate_ei_balance(empty_stats)
        
        # Should handle gracefully
        self.assertIn('error', validation)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)