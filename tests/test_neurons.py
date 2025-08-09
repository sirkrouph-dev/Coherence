"""
Comprehensive unit tests for neuron models.
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.neurons import (
    NeuronModel, AdaptiveExponentialIntegrateAndFire,
    HodgkinHuxleyNeuron, LeakyIntegrateAndFire, NeuronPopulation
)


class TestNeuronModel(unittest.TestCase):
    """Test base neuron model functionality."""
    
    def test_neuron_initialization(self):
        """Test neuron model initialization."""
        neuron = AdaptiveExponentialIntegrateAndFire(neuron_id=1)
        self.assertEqual(neuron.neuron_id, 1)
        self.assertFalse(neuron.is_spiking)
        self.assertEqual(len(neuron.spike_times), 0)
    
    def test_reset_functionality(self):
        """Test neuron reset."""
        neuron = LeakyIntegrateAndFire(neuron_id=1)
        neuron.v = -50.0  # Change membrane potential
        neuron.spike_times = [1.0, 2.0, 3.0]
        neuron.reset()
        
        # Check reset worked
        self.assertEqual(neuron.v, neuron.v_rest)
        self.assertEqual(len(neuron.spike_times), 0)
        self.assertFalse(neuron.is_spiking)


class TestAdaptiveExponentialIntegrateAndFire(unittest.TestCase):
    """Test AdEx neuron model."""
    
    def setUp(self):
        """Set up test neuron."""
        self.neuron = AdaptiveExponentialIntegrateAndFire(
            neuron_id=1,
            tau_m=20.0,
            v_rest=-65.0,
            v_thresh=-55.0,
            v_reset=-65.0,
            delta_t=2.0,
            tau_w=144.0,
            a=4.0,
            b=0.0805
        )
    
    def test_initialization(self):
        """Test AdEx initialization."""
        self.assertEqual(self.neuron.v, self.neuron.v_rest)
        self.assertEqual(self.neuron.w, 0.0)
        self.assertFalse(self.neuron.is_spiking)
    
    def test_subthreshold_dynamics(self):
        """Test subthreshold membrane dynamics."""
        dt = 0.1
        I_syn = 5.0  # Small current
        
        initial_v = self.neuron.v
        self.neuron.step(dt, I_syn)
        
        # Membrane potential should increase but not spike
        self.assertGreater(self.neuron.v, initial_v)
        self.assertLess(self.neuron.v, self.neuron.v_thresh)
        self.assertFalse(self.neuron.is_spiking)
    
    def test_spike_generation(self):
        """Test spike generation."""
        dt = 0.1
        I_syn = 100.0  # Large current to trigger spike
        
        # Step until spike
        for _ in range(100):
            self.neuron.step(dt, I_syn)
            if self.neuron.is_spiking:
                break
        
        # Should have spiked
        self.assertTrue(len(self.neuron.spike_times) > 0)
        # Should be in refractory period
        self.assertGreater(self.neuron.refractory_time, 0)
    
    def test_adaptation_current(self):
        """Test adaptation current dynamics."""
        dt = 0.1
        I_syn = 50.0
        
        initial_w = self.neuron.w
        
        # Generate a spike
        for _ in range(100):
            self.neuron.step(dt, I_syn)
            if self.neuron.is_spiking:
                break
        
        # Adaptation current should have increased after spike
        self.assertGreater(self.neuron.w, initial_w)
    
    def test_refractory_period(self):
        """Test refractory period prevents spiking."""
        self.neuron.refractory_time = 5.0  # Set refractory
        dt = 0.1
        I_syn = 1000.0  # Very large current
        
        initial_v = self.neuron.v
        self.neuron.step(dt, I_syn)
        
        # Should not spike during refractory
        self.assertFalse(self.neuron.is_spiking)
        self.assertEqual(self.neuron.v, initial_v)  # No change during refractory


class TestHodgkinHuxley(unittest.TestCase):
    """Test Hodgkin-Huxley neuron model."""
    
    def setUp(self):
        """Set up test neuron."""
        self.neuron = HodgkinHuxleyNeuron(neuron_id=2)
    
    def test_initialization(self):
        """Test HH initialization."""
        self.assertAlmostEqual(self.neuron.v, -65.0, places=1)
        self.assertFalse(self.neuron.is_spiking)
        # Check gating variables are initialized
        self.assertIsNotNone(self.neuron.m)
        self.assertIsNotNone(self.neuron.h)
        self.assertIsNotNone(self.neuron.n)
    
    def test_gating_variables(self):
        """Test gating variable dynamics."""
        dt = 0.01
        I_syn = 10.0
        
        initial_m = self.neuron.m
        initial_h = self.neuron.h
        initial_n = self.neuron.n
        
        self.neuron.step(dt, I_syn)
        
        # Gating variables should change
        self.assertNotEqual(self.neuron.m, initial_m)
        self.assertNotEqual(self.neuron.h, initial_h)
        self.assertNotEqual(self.neuron.n, initial_n)
        
        # Should be bounded between 0 and 1
        self.assertTrue(0 <= self.neuron.m <= 1)
        self.assertTrue(0 <= self.neuron.h <= 1)
        self.assertTrue(0 <= self.neuron.n <= 1)
    
    def test_action_potential(self):
        """Test action potential generation."""
        dt = 0.01
        I_syn = 40.0  # Strong current
        
        max_v = -65.0
        for _ in range(500):  # 5ms simulation
            self.neuron.step(dt, I_syn)
            max_v = max(max_v, self.neuron.v)
            if self.neuron.v > 0:  # Action potential peak
                break
        
        # Should generate action potential
        self.assertGreater(max_v, 0)  # Should depolarize above 0mV


class TestLeakyIntegrateAndFire(unittest.TestCase):
    """Test LIF neuron model."""
    
    def setUp(self):
        """Set up test neuron."""
        self.neuron = LeakyIntegrateAndFire(
            neuron_id=3,
            tau_m=10.0,
            v_rest=-65.0,
            v_thresh=-55.0,
            v_reset=-70.0
        )
    
    def test_initialization(self):
        """Test LIF initialization."""
        self.assertEqual(self.neuron.v, self.neuron.v_rest)
        self.assertFalse(self.neuron.is_spiking)
    
    def test_leak_dynamics(self):
        """Test membrane leak."""
        # Set membrane potential above rest
        self.neuron.v = -60.0
        dt = 0.1
        I_syn = 0.0  # No input
        
        self.neuron.step(dt, I_syn)
        
        # Should decay toward rest
        self.assertLess(self.neuron.v, -60.0)
        self.assertGreater(self.neuron.v, self.neuron.v_rest)
    
    def test_integration(self):
        """Test current integration."""
        dt = 0.1
        I_syn = 10.0
        
        initial_v = self.neuron.v
        self.neuron.step(dt, I_syn)
        
        # Should integrate current
        self.assertGreater(self.neuron.v, initial_v)
    
    def test_threshold_reset(self):
        """Test threshold crossing and reset."""
        dt = 0.1
        I_syn = 100.0  # Large current
        
        # Step until spike
        spiked = False
        for _ in range(100):
            self.neuron.step(dt, I_syn)
            if self.neuron.is_spiking:
                spiked = True
                break
        
        self.assertTrue(spiked)
        # After spike, should reset
        self.neuron.step(dt, 0)  # One more step
        self.assertAlmostEqual(self.neuron.v, self.neuron.v_reset, places=1)


class TestNeuronPopulation(unittest.TestCase):
    """Test neuron population functionality."""
    
    def setUp(self):
        """Set up test population."""
        self.pop_size = 10
        self.population = NeuronPopulation(
            size=self.pop_size,
            neuron_type="lif"
        )
    
    def test_population_size(self):
        """Test population creation."""
        self.assertEqual(len(self.population.neurons), self.pop_size)
        for i, neuron in enumerate(self.population.neurons):
            self.assertEqual(neuron.neuron_id, i)
    
    def test_population_step(self):
        """Test population step."""
        dt = 0.1
        I_syn = [10.0] * self.pop_size
        
        spikes = self.population.step(dt, I_syn)
        
        self.assertEqual(len(spikes), self.pop_size)
        self.assertTrue(all(isinstance(s, bool) for s in spikes))
    
    def test_population_reset(self):
        """Test population reset."""
        # Generate some spikes
        dt = 0.1
        I_syn = [100.0] * self.pop_size
        
        for _ in range(10):
            self.population.step(dt, I_syn)
        
        # Reset
        self.population.reset()
        
        # Check all neurons reset
        for neuron in self.population.neurons:
            self.assertEqual(len(neuron.spike_times), 0)
            self.assertFalse(neuron.is_spiking)
    
    def test_get_membrane_potentials(self):
        """Test getting membrane potentials."""
        potentials = self.population.get_membrane_potentials()
        
        self.assertEqual(len(potentials), self.pop_size)
        self.assertTrue(all(isinstance(v, float) for v in potentials))
    
    def test_different_neuron_types(self):
        """Test creating populations with different neuron types."""
        # Test AdEx population
        adex_pop = NeuronPopulation(size=5, neuron_type="adex")
        self.assertTrue(all(
            isinstance(n, AdaptiveExponentialIntegrateAndFire)
            for n in adex_pop.neurons
        ))
        
        # Test HH population
        hh_pop = NeuronPopulation(size=5, neuron_type="hodgkin_huxley")
        self.assertTrue(all(
            isinstance(n, HodgkinHuxleyNeuron)
            for n in hh_pop.neurons
        ))
    
    def test_invalid_input_length(self):
        """Test handling of invalid input length."""
        dt = 0.1
        I_syn = [10.0] * (self.pop_size - 1)  # Wrong length
        
        with self.assertRaises(ValueError):
            self.population.step(dt, I_syn)


class TestNumericalStability(unittest.TestCase):
    """Test numerical stability of neuron models."""
    
    def test_extreme_currents(self):
        """Test handling of extreme input currents."""
        neuron = AdaptiveExponentialIntegrateAndFire(neuron_id=1)
        dt = 0.1
        
        # Test very large positive current
        neuron.step(dt, 1e6)
        self.assertFalse(np.isnan(neuron.v))
        self.assertFalse(np.isinf(neuron.v))
        
        # Test very large negative current
        neuron.reset()
        neuron.step(dt, -1e6)
        self.assertFalse(np.isnan(neuron.v))
        self.assertFalse(np.isinf(neuron.v))
    
    def test_long_simulation(self):
        """Test stability over long simulation."""
        neuron = LeakyIntegrateAndFire(neuron_id=1)
        dt = 0.1
        I_syn = 10.0
        
        # Run for 10000 steps
        for _ in range(10000):
            neuron.step(dt, I_syn)
            
            # Check for numerical issues
            self.assertFalse(np.isnan(neuron.v))
            self.assertFalse(np.isinf(neuron.v))
            self.assertTrue(-200 < neuron.v < 100)  # Reasonable bounds


if __name__ == "__main__":
    unittest.main()
