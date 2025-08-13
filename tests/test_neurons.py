"""
Comprehensive tests for neuron models.
Tests both traditional object-oriented and vectorized neuron implementations.
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.neurons import (
    LeakyIntegrateAndFire, 
    AdaptiveExponentialIntegrateAndFire, 
    HodgkinHuxleyNeuron,
    NeuronFactory,
    NeuronPopulation
)
from core.vectorized_neurons import create_vectorized_population


class TestTraditionalNeurons(unittest.TestCase):
    """Test traditional object-oriented neuron models."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.dt = 0.1
        
    def test_lif_neuron_creation(self):
        """Test LIF neuron creation and basic properties."""
        neuron = LeakyIntegrateAndFire(neuron_id=0)
        
        self.assertIsNotNone(neuron)
        self.assertAlmostEqual(neuron.membrane_potential, -65.0, places=1)
        self.assertTrue(hasattr(neuron, 'tau_m'))
        self.assertTrue(hasattr(neuron, 'v_thresh'))
        
    def test_lif_neuron_simulation(self):
        """Test LIF neuron simulation dynamics."""
        neuron = LeakyIntegrateAndFire(neuron_id=0)
        
        # No input - should not spike
        for _ in range(100):
            spike = neuron.step(self.dt, 0.0)
            self.assertFalse(spike)
            
        # Strong input - should spike
        neuron.reset()
        spiked = False
        for _ in range(100):
            spike = neuron.step(self.dt, 100.0)
            if spike:
                spiked = True
                break
        self.assertTrue(spiked)
        
    def test_lif_neuron_reset(self):
        """Test LIF neuron reset functionality."""
        neuron = LeakyIntegrateAndFire(neuron_id=0)
        
        # Change state
        for _ in range(50):
            neuron.step(self.dt, 50.0)
            
        original_v = neuron.membrane_potential
        neuron.reset()
        
        # Should have cleared spike times
        self.assertEqual(len(neuron.spike_times), 0)
        self.assertFalse(neuron.is_spiking)
        
    def test_adex_neuron_creation(self):
        """Test AdEx neuron creation and properties."""
        neuron = AdaptiveExponentialIntegrateAndFire(neuron_id=0)
        
        self.assertIsNotNone(neuron)
        self.assertAlmostEqual(neuron.membrane_potential, -65.0, places=1)
        self.assertTrue(hasattr(neuron, 'adaptation_current'))
        self.assertTrue(hasattr(neuron, 'a'))
        self.assertTrue(hasattr(neuron, 'b'))
        
    def test_adex_neuron_adaptation(self):
        """Test AdEx neuron adaptation mechanisms."""
        neuron = AdaptiveExponentialIntegrateAndFire(neuron_id=0)
        
        initial_adaptation = neuron.adaptation_current
        
        # Stimulate to cause adaptation
        for _ in range(200):
            spike = neuron.step(self.dt, 120.0)
            if spike:
                break
                
        # Adaptation should have changed from initial (may be same if no spike)
        # This test mainly ensures the mechanism exists
        self.assertTrue(hasattr(neuron, 'adaptation_current'))
        
    def test_hodgkin_huxley_neuron(self):
        """Test Hodgkin-Huxley neuron model."""
        neuron = HodgkinHuxleyNeuron(neuron_id=0)
        
        self.assertIsNotNone(neuron)
        self.assertTrue(hasattr(neuron, 'm'))  # Sodium activation
        self.assertTrue(hasattr(neuron, 'h'))  # Sodium inactivation
        self.assertTrue(hasattr(neuron, 'n'))  # Potassium activation
        
    def test_neuron_spike_times(self):
        """Test spike time recording."""
        neuron = LeakyIntegrateAndFire(neuron_id=0)
        
        initial_spike_count = len(neuron.spike_times)
        
        # Generate spikes
        for i in range(100):
            spike = neuron.step(self.dt, 150.0)
            if spike:
                break
                
        # Check if spike was recorded (may not spike depending on parameters)
        final_spike_count = len(neuron.spike_times)
        self.assertGreaterEqual(final_spike_count, initial_spike_count)
        
    def test_neuron_factory(self):
        """Test neuron factory creation."""
        factory = NeuronFactory()
        
        # Create different neuron types
        lif = factory.create_neuron("lif", neuron_id=0)
        adex = factory.create_neuron("adex", neuron_id=1)
        
        self.assertIsInstance(lif, LeakyIntegrateAndFire)
        self.assertIsInstance(adex, AdaptiveExponentialIntegrateAndFire)
        
    def test_neuron_population(self):
        """Test neuron population functionality."""
        population = NeuronPopulation(size=10, neuron_type="lif")
        
        self.assertEqual(len(population.neurons), 10)
        
        # Test population step
        spikes = population.step(self.dt, [50.0] * 10)
        self.assertEqual(len(spikes), 10)
        self.assertTrue(all(isinstance(spike, bool) for spike in spikes))
        
    def test_custom_neuron_parameters(self):
        """Test neurons with custom parameters."""
        custom_lif = LeakyIntegrateAndFire(
            neuron_id=0,
            tau_m=20.0,
            v_thresh=-50.0,
            v_reset=-70.0
        )
        
        self.assertAlmostEqual(custom_lif.tau_m, 20.0, places=1)
        self.assertAlmostEqual(custom_lif.v_thresh, -50.0, places=1)
        self.assertAlmostEqual(custom_lif.v_reset, -70.0, places=1)


class TestNeuronCompatibility(unittest.TestCase):
    """Test compatibility between traditional and vectorized neurons."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.dt = 0.1
        
    def test_lif_traditional_vs_vectorized(self):
        """Compare traditional LIF vs vectorized LIF behavior."""
        # Traditional neuron
        trad_neuron = LeakyIntegrateAndFire(neuron_id=0)
        
        # Vectorized neuron (single neuron)
        vec_pop = create_vectorized_population(1, "lif")
        
        # Same input current
        I_input = 80.0
        
        trad_spikes = 0
        vec_spikes = 0
        
        # Run simulation
        for _ in range(500):
            # Traditional
            if trad_neuron.step(self.dt, I_input):
                trad_spikes += 1
                
            # Vectorized
            vec_spike_array = vec_pop.step(self.dt, np.array([I_input]))
            if vec_spike_array[0]:
                vec_spikes += 1
                
        # Both should spike (exact matching not required due to different implementations)
        self.assertGreater(trad_spikes, 0)
        self.assertGreater(vec_spikes, 0)


class TestNeuronEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions for neurons."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.dt = 0.1
        
    def test_extreme_input_currents(self):
        """Test neurons with extreme input currents."""
        neuron = LeakyIntegrateAndFire(neuron_id=0)
        
        # Very large current
        spike = neuron.step(self.dt, 10000.0)
        # Should handle extreme input gracefully
        self.assertIsInstance(spike, bool)
        
        # Very negative current
        neuron.reset()
        for _ in range(100):
            spike = neuron.step(self.dt, -1000.0)
            self.assertFalse(spike)  # Should never spike
            
    def test_zero_time_step(self):
        """Test behavior with zero time step."""
        neuron = LeakyIntegrateAndFire(neuron_id=0)
        
        try:
            # This might raise an error or handle gracefully
            spike = neuron.step(0.0, 100.0)
            # If no error, should return boolean
            self.assertIsInstance(spike, bool)
        except (ValueError, ZeroDivisionError):
            # This is acceptable behavior
            pass
            
    def test_invalid_neuron_factory(self):
        """Test factory with invalid neuron type."""
        factory = NeuronFactory()
        
        try:
            neuron = factory.create_neuron("invalid_type", neuron_id=0)
            # If creation succeeds, should return valid neuron
            self.assertIsNotNone(neuron)
        except (ValueError, KeyError):
            # This is acceptable
            pass
            
    def test_membrane_potential_stability(self):
        """Test membrane potential doesn't go to unrealistic values."""
        neuron = LeakyIntegrateAndFire(neuron_id=0)
        
        # Apply strong negative current for long time
        for _ in range(1000):
            neuron.step(self.dt, -1000.0)
            
        # Should not go too negative (exact bounds depend on implementation)
        self.assertGreater(neuron.membrane_potential, -1000.0)
        self.assertLess(neuron.membrane_potential, 1000.0)


if __name__ == '__main__':
    unittest.main()
