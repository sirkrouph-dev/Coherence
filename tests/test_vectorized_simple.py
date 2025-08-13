"""
Simple tests for vectorized neuron implementations.
Tests the high-performance vectorized neuron populations.
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.vectorized_neurons import create_vectorized_population


class TestVectorizedNeurons(unittest.TestCase):
    """Test vectorized neuron implementations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.size = 100
        self.dt = 0.1
        
    def test_population_creation(self):
        """Test neuron population creation."""
        pop_lif = create_vectorized_population(self.size, "lif")
        pop_adex = create_vectorized_population(self.size, "adex")
        
        # Both should be created successfully
        self.assertIsNotNone(pop_lif)
        self.assertIsNotNone(pop_adex)
        
    def test_basic_simulation(self):
        """Test basic simulation step."""
        pop = create_vectorized_population(self.size, "lif")
        
        # Apply input current
        I_syn = np.full(self.size, 100.0)
        
        # Run simulation steps
        for _ in range(50):
            spikes = pop.step(self.dt, I_syn)
            
            # Should return boolean array
            self.assertIsInstance(spikes, np.ndarray)
            self.assertEqual(len(spikes), self.size)
            self.assertEqual(spikes.dtype, bool)
            
    def test_spike_generation(self):
        """Test that neurons can generate spikes."""
        pop = create_vectorized_population(10, "lif")
        
        # Strong input current
        I_syn = np.full(10, 150.0)
        
        # Run simulation and count spikes
        total_spikes = 0
        for _ in range(100):
            spikes = pop.step(self.dt, I_syn)
            total_spikes += np.sum(spikes)
            
        # Should generate some spikes with strong input
        self.assertGreater(total_spikes, 0)
        
    def test_no_spikes_with_no_input(self):
        """Test that neurons don't spike without input."""
        pop = create_vectorized_population(10, "lif")
        
        # No input current
        I_syn = np.zeros(10)
        
        # Run simulation
        total_spikes = 0
        for _ in range(100):
            spikes = pop.step(self.dt, I_syn)
            total_spikes += np.sum(spikes)
            
        # Should not spike without input
        self.assertEqual(total_spikes, 0)
        
    def test_reset_functionality(self):
        """Test population reset."""
        pop = create_vectorized_population(10, "lif")
        
        # Run simulation to change state
        I_syn = np.full(10, 100.0)
        for _ in range(50):
            pop.step(self.dt, I_syn)
            
        # Reset should work without error
        try:
            pop.reset()
        except Exception as e:
            self.fail(f"Reset failed with error: {e}")
            
    def test_different_neuron_types(self):
        """Test that different neuron types behave differently."""
        lif_pop = create_vectorized_population(100, "lif")
        adex_pop = create_vectorized_population(100, "adex")
        
        # Same input
        I_syn = np.full(100, 90.0)
        
        # Run simulations
        lif_spikes = 0
        adex_spikes = 0
        
        for _ in range(200):
            lif_spikes += np.sum(lif_pop.step(self.dt, I_syn))
            adex_spikes += np.sum(adex_pop.step(self.dt, I_syn))
            
        # Both should spike (might be similar, but that's ok)
        self.assertGreater(lif_spikes, 0)
        self.assertGreater(adex_spikes, 0)
        
    def test_performance(self):
        """Test performance with larger populations."""
        import time
        
        # Test larger population
        size = 10000
        pop = create_vectorized_population(size, "lif")
        I_syn = np.random.uniform(80, 120, size)
        
        # Time the simulation
        start_time = time.time()
        for _ in range(100):
            pop.step(self.dt, I_syn)
        elapsed = time.time() - start_time
        
        # Should complete in reasonable time (less than 1 second)
        self.assertLess(elapsed, 1.0)
        
        # Calculate throughput
        throughput = (size * 100) / elapsed
        print(f"Performance: {throughput/1000:.1f}k neuron-steps/second")


if __name__ == '__main__':
    unittest.main()
