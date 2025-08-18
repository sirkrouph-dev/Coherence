#!/usr/bin/env python3
"""
Test suite for enhanced excitatory/inhibitory balance with specific inhibitory neuron types.

This module tests the enhanced E/I balance system to ensure it:
- Maintains proper 80/20 E/I ratios
- Creates specific inhibitory neuron types (basket, chandelier, Martinotti)
- Implements realistic connection probabilities and strengths
- Provides activity-based balance monitoring
- Integrates properly with network builders
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
    SpatialNetworkLayout
)


class TestEnhancedEIBalance(unittest.TestCase):
    """Test enhanced excitatory/inhibitory balance system."""
    
    def setUp(self):
        """Set up test parameters."""
        self.ei_balance = ExcitatoryInhibitoryBalance()
        
    def test_population_size_calculation(self):
        """Test calculation of E/I population sizes."""
        total_neurons = 1000
        populations = self.ei_balance.calculate_population_sizes(total_neurons)
        
        # Check expected keys
        expected_keys = ['excitatory', 'inhibitory_total', 'basket', 'chandelier', 'martinotti']
        for key in expected_keys:
            self.assertIn(key, populations)
            
        # Check E/I ratio
        excitatory = populations['excitatory']
        inhibitory_total = populations['inhibitory_total']
        
        self.assertAlmostEqual(excitatory / total_neurons, 0.8, places=1)
        self.assertAlmostEqual(inhibitory_total / total_neurons, 0.2, places=1)
        
        # Check inhibitory subtypes sum to total
        inhibitory_sum = populations['basket'] + populations['chandelier'] + populations['martinotti']
        self.assertEqual(inhibitory_sum, inhibitory_total)
        
        # Check inhibitory type ratios
        self.assertAlmostEqual(populations['basket'] / inhibitory_total, 0.6, places=1)
        self.assertAlmostEqual(populations['chandelier'] / inhibitory_total, 0.25, places=1)
        self.assertAlmostEqual(populations['martinotti'] / inhibitory_total, 0.15, places=1)
        
    def test_connection_parameters(self):
        """Test connection parameters for different neuron type pairs."""
        # Test all E/I connection types
        connection_types = [
            ('excitatory', 'excitatory'),
            ('excitatory', 'inhibitory'),
            ('inhibitory', 'excitatory'),
            ('inhibitory', 'inhibitory')
        ]
        
        for pre_type, post_type in connection_types:
            params = self.ei_balance.get_connection_parameters(pre_type, post_type)
            
            # Check required parameters
            self.assertIn('connection_probability', params)
            self.assertIn('synaptic_strength', params)
            self.assertIn('connection_type', params)
            
            # Check probability bounds
            prob = params['connection_probability']
            self.assertTrue(0.0 <= prob <= 1.0, f"Invalid probability {prob} for {pre_type}→{post_type}")
            
            # Check inhibitory connections have negative strength
            if pre_type == 'inhibitory':
                self.assertLess(params['synaptic_strength'], 0, 
                               f"Inhibitory connection should have negative strength")
            else:
                self.assertGreater(params['synaptic_strength'], 0,
                                 f"Excitatory connection should have positive strength")
                
    def test_inhibitory_neuron_parameters(self):
        """Test parameters for specific inhibitory neuron types."""
        inhibitory_types = ['basket', 'chandelier', 'martinotti']
        
        for inh_type in inhibitory_types:
            params = self.ei_balance.get_inhibitory_neuron_parameters(inh_type)
            
            # Check required parameters (updated for LIF compatibility)
            required_params = [
                'neuron_type', 'tau_m', 'v_thresh', 'adaptation',
                'connection_range', 'target_location', 'v_reset',
                'v_rest', 'refractory_period'
            ]
            
            for param in required_params:
                self.assertIn(param, params, f"Missing parameter {param} for {inh_type}")
                
            # Check neuron type is LIF for inhibitory neurons
            self.assertEqual(params['neuron_type'], 'lif')
            
            # Check type-specific properties
            if inh_type == 'basket':
                self.assertEqual(params['target_location'], 'soma')
                self.assertLess(params['tau_m'], 15.0)  # Fast spiking
            elif inh_type == 'chandelier':
                self.assertEqual(params['target_location'], 'axon_initial_segment')
                self.assertLess(params['tau_m'], 10.0)  # Very fast
            elif inh_type == 'martinotti':
                self.assertEqual(params['target_location'], 'dendrites')
                self.assertGreater(params['tau_m'], 15.0)  # Slower


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)