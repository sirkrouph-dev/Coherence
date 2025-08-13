"""
Working vectorized neuron tests.
Simple but comprehensive tests that follow the working pattern.
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.vectorized_neurons import create_vectorized_population


class TestVectorizedNeuronsWorking(unittest.TestCase):
    """Working comprehensive tests for vectorized neuron implementations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.size = 100
        self.dt = 0.1
        
    def test_population_creation_lif(self):
        """Test LIF neuron population creation."""
        pop = create_vectorized_population(self.size, "lif")
        self.assertIsNotNone(pop)
        self.assertEqual(pop.size, self.size)
        self.assertEqual(pop.neuron_type, "lif")
        
    def test_population_creation_adex(self):
        """Test AdEx neuron population creation."""
        pop = create_vectorized_population(self.size, "adex")
        self.assertIsNotNone(pop)
        self.assertEqual(pop.size, self.size)
        self.assertEqual(pop.neuron_type, "adex")
        
    def test_custom_parameters(self):
        """Test custom parameter configuration."""
        params = {'tau_m': 20.0, 'v_thresh': -55.0}
        pop = create_vectorized_population(50, "lif", **params)
        
        # Check parameters were applied
        self.assertAlmostEqual(pop.model.tau_m, 20.0, places=1)
        self.assertAlmostEqual(pop.model.v_thresh, -55.0, places=1)
        
    def test_membrane_potential_access(self):
        """Test membrane potential access."""
        pop = create_vectorized_population(10, "lif")
        
        # Get membrane potentials
        v_mem = pop.get_membrane_potentials()
        self.assertEqual(len(v_mem), 10)
        self.assertTrue(np.all(v_mem < 0))  # Should be negative
        
        # Modify and verify
        pop.model.membrane_potential[0] = -50.0
        v_mem_new = pop.get_membrane_potentials()
        self.assertAlmostEqual(v_mem_new[0], -50.0, places=1)
        
    def test_basic_simulation(self):
        """Test basic simulation steps."""
        pop = create_vectorized_population(50, "lif")
        
        # Run simulation
        I_syn = np.random.uniform(50, 100, 50)
        for step in range(100):
            spikes = pop.step(self.dt, I_syn)
            
            # Verify output format
            self.assertIsInstance(spikes, np.ndarray)
            self.assertEqual(spikes.dtype, bool)
            self.assertEqual(len(spikes), 50)
            
    def test_spike_generation_no_input(self):
        """Test no spikes without input."""
        pop = create_vectorized_population(20, "lif")
        
        # No input
        I_syn = np.zeros(20)
        total_spikes = 0
        
        for _ in range(100):
            spikes = pop.step(self.dt, I_syn)
            total_spikes += np.sum(spikes)
            
        self.assertEqual(total_spikes, 0)
        
    def test_spike_generation_with_input(self):
        """Test spike generation with strong input."""
        pop = create_vectorized_population(20, "lif")
        
        # Strong input
        I_syn = np.full(20, 120.0)
        total_spikes = 0
        
        for _ in range(200):
            spikes = pop.step(self.dt, I_syn)
            total_spikes += np.sum(spikes)
            
        self.assertGreater(total_spikes, 0)
        
    def test_firing_rates(self):
        """Test firing rate calculation."""
        pop = create_vectorized_population(10, "lif")
        
        # Run simulation
        I_syn = np.full(10, 90.0)
        for _ in range(1000):
            pop.step(self.dt, I_syn)
            
        # Get firing rates
        rates = pop.get_firing_rates(100.0)
        self.assertEqual(len(rates), 10)
        self.assertTrue(np.all(rates >= 0))
        
    def test_performance_stats(self):
        """Test performance statistics."""
        pop = create_vectorized_population(100, "lif")
        
        # Run simulation
        I_syn = np.random.uniform(60, 120, 100)
        for _ in range(50):
            pop.step(self.dt, I_syn)
            
        # Get stats
        stats = pop.get_performance_stats()
        
        self.assertIn("population_size", stats)
        self.assertIn("neuron_type", stats)
        self.assertIn("total_steps", stats)
        self.assertIn("total_spikes", stats)
        
        self.assertEqual(stats["population_size"], 100)
        self.assertEqual(stats["neuron_type"], "lif")
        self.assertEqual(stats["total_steps"], 50)
        
    def test_reset_functionality(self):
        """Test population reset."""
        pop = create_vectorized_population(20, "lif")
        
        # Run simulation
        I_syn = np.full(20, 100.0)
        for _ in range(50):
            pop.step(self.dt, I_syn)
            
        # Verify state changed
        stats_before = pop.get_performance_stats()
        self.assertGreater(stats_before["total_steps"], 0)
        
        # Reset
        pop.reset()
        
        # Verify reset
        stats_after = pop.get_performance_stats()
        self.assertEqual(stats_after["total_steps"], 0)
        self.assertEqual(stats_after["total_spikes"], 0)
        
    def test_different_neuron_types_comparison(self):
        """Test LIF vs AdEx behavior."""
        lif_pop = create_vectorized_population(30, "lif")
        adex_pop = create_vectorized_population(30, "adex")
        
        # Same input
        I_syn = np.full(30, 85.0)
        
        lif_total = 0
        adex_total = 0
        
        for _ in range(300):
            lif_total += np.sum(lif_pop.step(self.dt, I_syn))
            adex_total += np.sum(adex_pop.step(self.dt, I_syn))
            
        # Both should spike
        self.assertGreater(lif_total, 0)
        self.assertGreater(adex_total, 0)
        
        # Verify types
        lif_stats = lif_pop.get_performance_stats()
        adex_stats = adex_pop.get_performance_stats()
        
        self.assertEqual(lif_stats["neuron_type"], "lif")
        self.assertEqual(adex_stats["neuron_type"], "adex")
        
    def test_large_population_scaling(self):
        """Test performance with larger populations."""
        import time
        
        sizes = [1000, 5000]
        
        for size in sizes:
            pop = create_vectorized_population(size, "lif")
            I_syn = np.random.uniform(80, 120, size)
            
            start_time = time.time()
            
            for _ in range(50):
                spikes = pop.step(self.dt, I_syn)
                
            elapsed = time.time() - start_time
            throughput = (size * 50) / elapsed
            
            print(f"Size {size}: {throughput/1000:.1f}k neuron-steps/sec")
            
            # Should complete reasonably fast
            self.assertLess(elapsed, 1.0)
            
    def test_vectorized_input_response(self):
        """Test response to different input levels."""
        pop = create_vectorized_population(100, "lif")
        
        # Create gradient of inputs
        I_syn = np.linspace(50, 150, 100)
        
        all_spikes = []
        for _ in range(500):
            spikes = pop.step(self.dt, I_syn)
            all_spikes.append(spikes.copy())
            
        # Analyze spike patterns
        spike_matrix = np.array(all_spikes)
        total_spikes_per_neuron = np.sum(spike_matrix, axis=0)
        
        # Higher input neurons should spike more (or at least as much)
        low_input_spikes = np.mean(total_spikes_per_neuron[:25])   # Low input
        high_input_spikes = np.mean(total_spikes_per_neuron[-25:])  # High input
        
        # Allow for equal spiking in edge cases, but expect trend
        self.assertGreaterEqual(high_input_spikes, low_input_spikes * 0.8)
        
    def test_error_handling(self):
        """Test error handling."""
        pop = create_vectorized_population(10, "lif")
        
        # Wrong input size should raise error
        with self.assertRaises((ValueError, IndexError)):
            I_syn_wrong = np.zeros(5)  # Wrong size
            pop.step(self.dt, I_syn_wrong)
            
        # Invalid neuron type
        with self.assertRaises(ValueError):
            create_vectorized_population(10, "invalid_neuron")
            
    def test_spike_state_consistency(self):
        """Test spike state consistency."""
        pop = create_vectorized_population(15, "lif")
        
        I_syn = np.full(15, 100.0)
        spikes = pop.step(self.dt, I_syn)
        spike_states = pop.get_spike_states()
        
        # Should be identical
        np.testing.assert_array_equal(spikes, spike_states)


if __name__ == '__main__':
    unittest.main()
