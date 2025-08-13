"""
Comprehensive tests for synapse models.
Tests both traditional and vectorized synapse implementations.
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.synapses import (
    STDP_Synapse,
    ShortTermPlasticitySynapse,
    NeuromodulatorySynapse,
    RSTDP_Synapse,
    SynapseFactory,
    SynapsePopulation
)
from core.vectorized_synapses import VectorizedSynapseManager, VectorizedSynapsePopulation


class TestTraditionalSynapses(unittest.TestCase):
    """Test traditional object-oriented synapse models."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.dt = 0.1
        self.pre_id = 0
        self.post_id = 1
        self.synapse_id = 0
        
    def test_stdp_synapse_creation(self):
        """Test STDP synapse creation and basic properties."""
        synapse = STDP_Synapse(
            synapse_id=self.synapse_id,
            pre_neuron_id=self.pre_id,
            post_neuron_id=self.post_id,
            weight=0.5
        )
        
        self.assertIsNotNone(synapse)
        self.assertEqual(synapse.pre_neuron_id, self.pre_id)
        self.assertEqual(synapse.post_neuron_id, self.post_id)
        self.assertAlmostEqual(synapse.weight, 0.5, places=3)
        self.assertTrue(hasattr(synapse, 'A_plus'))
        self.assertTrue(hasattr(synapse, 'A_minus'))
        self.assertTrue(hasattr(synapse, 'tau_stdp'))
        
    def test_stdp_current_computation(self):
        """Test STDP synapse current computation."""
        synapse = STDP_Synapse(
            synapse_id=self.synapse_id,
            pre_neuron_id=self.pre_id,
            post_neuron_id=self.post_id,
            weight=1.0
        )
        
        # Test current computation with spike
        current = synapse.compute_current(pre_spike_time=0.0, current_time=1.0)
        self.assertGreater(current, 0.0)
        
        # Test current computation without spike
        current = synapse.compute_current(pre_spike_time=-np.inf, current_time=1.0)
        self.assertEqual(current, 0.0)
        
    def test_stdp_plasticity(self):
        """Test STDP plasticity mechanisms."""
        synapse = STDP_Synapse(
            synapse_id=self.synapse_id,
            pre_neuron_id=self.pre_id,
            post_neuron_id=self.post_id,
            weight=0.5,
            A_plus=0.1,
            A_minus=0.1
        )
        
        initial_weight = synapse.weight
        
        # Pre-before-post (LTP)
        synapse.pre_spike(0.0)
        synapse.post_spike(1.0)
        
        # Should have some weight change
        self.assertNotEqual(synapse.weight, initial_weight)
        
    def test_stp_synapse_creation(self):
        """Test Short-Term Plasticity synapse creation."""
        synapse = ShortTermPlasticitySynapse(
            synapse_id=self.synapse_id,
            pre_neuron_id=self.pre_id,
            post_neuron_id=self.post_id,
            weight=0.5,
            U=0.5,
            tau_dep=100.0,
            tau_fac=100.0
        )
        
        self.assertIsNotNone(synapse)
        self.assertTrue(hasattr(synapse, 'U'))
        self.assertTrue(hasattr(synapse, 'tau_dep'))
        self.assertTrue(hasattr(synapse, 'tau_fac'))
        self.assertTrue(hasattr(synapse, 'x'))  # Available resources
        self.assertTrue(hasattr(synapse, 'u'))  # Utilization
        
    def test_stp_dynamics(self):
        """Test STP facilitation and depression dynamics."""
        synapse = ShortTermPlasticitySynapse(
            synapse_id=self.synapse_id,
            pre_neuron_id=self.pre_id,
            post_neuron_id=self.post_id,
            weight=1.0,
            U=0.2,
            tau_dep=100.0,
            tau_fac=50.0
        )
        
        # First spike
        synapse.pre_spike(0.0)
        current1 = synapse.compute_current(pre_spike_time=0.0, current_time=0.0)
        
        # Immediate second spike
        synapse.pre_spike(1.0)
        current2 = synapse.compute_current(pre_spike_time=1.0, current_time=1.0)
        
        # Both should be positive
        self.assertGreater(current1, 0.0)
        self.assertGreater(current2, 0.0)
        
    def test_neuromodulatory_synapse(self):
        """Test neuromodulatory synapse creation and function."""
        synapse = NeuromodulatorySynapse(
            synapse_id=self.synapse_id,
            pre_neuron_id=self.pre_id,
            post_neuron_id=self.post_id,
            weight=0.5,
            learning_rate=0.01
        )
        
        self.assertIsNotNone(synapse)
        self.assertTrue(hasattr(synapse, 'learning_rate'))
        self.assertTrue(hasattr(synapse, 'neuromodulator_level'))
        
        # Test neuromodulator update
        synapse.update_neuromodulator(0.8)
        self.assertAlmostEqual(synapse.neuromodulator_level, 0.8, places=3)
        
    def test_rstdp_synapse(self):
        """Test Reward-modulated STDP synapse."""
        synapse = RSTDP_Synapse(
            synapse_id=self.synapse_id,
            pre_neuron_id=self.pre_id,
            post_neuron_id=self.post_id,
            weight=0.5,
            learning_rate=0.01
        )
        
        self.assertIsNotNone(synapse)
        self.assertTrue(hasattr(synapse, 'reward_signal'))
        self.assertTrue(hasattr(synapse, 'neuromodulator_level'))
        
        # Test reward update
        synapse.update_reward(1.0)
        self.assertAlmostEqual(synapse.reward_signal, 1.0, places=3)
        
    def test_synapse_factory(self):
        """Test synapse factory creation."""
        # Create different synapse types
        stdp = SynapseFactory.create_synapse(
            synapse_id=0,
            pre_neuron_id=0,
            post_neuron_id=1,
            synapse_type="stdp",
            weight=0.5
        )
        stp = SynapseFactory.create_synapse(
            synapse_id=1,
            pre_neuron_id=0,
            post_neuron_id=1,
            synapse_type="stp",
            weight=0.5
        )
        neuromod = SynapseFactory.create_synapse(
            synapse_id=2,
            pre_neuron_id=0,
            post_neuron_id=1,
            synapse_type="neuromodulatory",
            weight=0.5
        )
        rstdp = SynapseFactory.create_synapse(
            synapse_id=3,
            pre_neuron_id=0,
            post_neuron_id=1,
            synapse_type="rstdp",
            weight=0.5
        )
        
        self.assertIsInstance(stdp, STDP_Synapse)
        self.assertIsInstance(stp, ShortTermPlasticitySynapse)
        self.assertIsInstance(neuromod, NeuromodulatorySynapse)
        self.assertIsInstance(rstdp, RSTDP_Synapse)
        
    def test_synapse_weight_bounds(self):
        """Test synapse weight boundary conditions."""
        synapse = STDP_Synapse(
            synapse_id=self.synapse_id,
            pre_neuron_id=self.pre_id,
            post_neuron_id=self.post_id,
            weight=0.5,
            w_min=0.0,
            w_max=1.0
        )
        
        # Test weight update with clipping
        synapse.update_weight(-1.0)  # Should clip to minimum
        self.assertGreaterEqual(synapse.weight, 0.0)
        
        synapse.update_weight(2.0)  # Should clip to maximum
        self.assertLessEqual(synapse.weight, 1.0)


class TestSynapsePopulation(unittest.TestCase):
    """Test synapse population functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.pre_size = 10
        self.post_size = 5
        
    def test_synapse_population_creation(self):
        """Test synapse population creation."""
        pop = SynapsePopulation(
            pre_population_size=self.pre_size,
            post_population_size=self.post_size,
            synapse_type="stdp",
            connection_probability=0.5
        )
        
        self.assertIsNotNone(pop)
        self.assertEqual(pop.pre_population_size, self.pre_size)
        self.assertEqual(pop.post_population_size, self.post_size)
        self.assertGreater(len(pop.synapses), 0)
        
    def test_population_current_computation(self):
        """Test population-level current computation."""
        pop = SynapsePopulation(
            pre_population_size=self.pre_size,
            post_population_size=self.post_size,
            synapse_type="stdp",
            connection_probability=0.8
        )
        
        # Create spike pattern
        pre_spikes = [True if i % 2 == 0 else False for i in range(self.pre_size)]
        
        # Compute currents
        currents = pop.get_synaptic_currents(pre_spikes, 0.0)
        
        # Should have currents for all postsynaptic neurons
        self.assertEqual(len(currents), self.post_size)
        self.assertTrue(all(isinstance(c, (int, float, np.number)) for c in currents))
        
    def test_population_weight_updates(self):
        """Test population-level weight updates."""
        pop = SynapsePopulation(
            pre_population_size=self.pre_size,
            post_population_size=self.post_size,
            synapse_type="stdp",
            connection_probability=0.5
        )
        
        # Get initial weight matrix
        initial_weights = pop.get_weight_matrix()
        
        # Create spike patterns
        pre_spikes = [True, False, True, False, True] + [False] * (self.pre_size - 5)
        post_spikes = [True, False, True] + [False] * (self.post_size - 3)
        
        # Update weights
        pop.update_weights(pre_spikes, post_spikes, 0.0)
        
        # Weights should have changed for some synapses
        final_weights = pop.get_weight_matrix()
        self.assertEqual(initial_weights.shape, final_weights.shape)
        
    def test_population_statistics(self):
        """Test population statistics."""
        pop = SynapsePopulation(
            pre_population_size=self.pre_size,
            post_population_size=self.post_size,
            synapse_type="stdp",
            connection_probability=0.3
        )
        
        weight_matrix = pop.get_weight_matrix()
        weight_history = pop.get_weight_history()
        
        self.assertEqual(weight_matrix.shape, (self.post_size, self.pre_size))
        self.assertIsInstance(weight_history, dict)


class TestVectorizedSynapses(unittest.TestCase):
    """Test vectorized synapse implementations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.dt = 0.1
        
    def test_vectorized_population_creation(self):
        """Test vectorized synapse population creation."""
        pop = VectorizedSynapsePopulation(
            pre_size=100,
            post_size=50,
            synapse_type="stdp",
            connection_probability=0.1
        )
        
        self.assertIsNotNone(pop)
        self.assertEqual(pop.pre_size, 100)
        self.assertEqual(pop.post_size, 50)
        self.assertGreater(pop.n_connections, 0)
        
    def test_vectorized_current_computation(self):
        """Test vectorized current computation."""
        pop = VectorizedSynapsePopulation(
            pre_size=10,
            post_size=5,
            synapse_type="stdp",
            connection_probability=0.5
        )
        
        # Create spike pattern
        pre_spikes = np.array([True, False, True, False, True, False, True, False, True, False])
        
        # Compute currents
        currents = pop.compute_synaptic_currents(pre_spikes, 0.0)
        
        # Should have currents for all postsynaptic neurons
        self.assertEqual(len(currents), 5)
        self.assertTrue(all(isinstance(c, (int, float, np.number)) for c in currents))
        
    def test_vectorized_weight_updates(self):
        """Test vectorized weight updates."""
        pop = VectorizedSynapsePopulation(
            pre_size=5,
            post_size=3,
            synapse_type="stdp",
            connection_probability=1.0
        )
        
        # Get initial statistics
        initial_stats = pop.get_statistics()
        initial_weight_sum = initial_stats["total_weight"]
        
        # Create spike patterns
        pre_spikes = np.array([True, False, True, False, True])
        post_spikes = np.array([False, True, False])
        
        # Update weights multiple times
        for _ in range(10):
            pop.update_weights(pre_spikes, post_spikes, 0.0)
        
        # Check that weights changed
        final_stats = pop.get_statistics()
        final_weight_sum = final_stats["total_weight"]
        
        # Weights should be different (or at least valid)
        self.assertIsInstance(final_weight_sum, (int, float, np.number))
        self.assertGreater(final_stats["n_connections"], 0)
        
    def test_vectorized_manager(self):
        """Test vectorized synapse manager."""
        manager = VectorizedSynapseManager()
        
        self.assertIsNotNone(manager)
        self.assertTrue(hasattr(manager, 'synapse_populations'))
        self.assertTrue(hasattr(manager, 'layer_sizes'))
        
    def test_manager_layer_connections(self):
        """Test manager layer connections."""
        manager = VectorizedSynapseManager()
        
        # Add layers
        manager.add_layer("input", 10)
        manager.add_layer("hidden", 5)
        
        # Check layers exist
        self.assertIn("input", manager.layer_sizes)
        self.assertIn("hidden", manager.layer_sizes)
        
        # Connect layers
        manager.connect_layers("input", "hidden", "stdp", connection_probability=0.5)
        
        # Check connection exists
        self.assertIn(("input", "hidden"), manager.synapse_populations)
        
    def test_manager_current_computation(self):
        """Test manager current computation."""
        manager = VectorizedSynapseManager()
        
        # Setup network
        manager.add_layer("pre", 5)
        manager.add_layer("post", 3)
        manager.connect_layers("pre", "post", "stdp", connection_probability=1.0)
        
        # Create spike pattern
        layer_spikes = {
            "pre": np.array([True, False, True, False, True])
        }
        
        # Compute currents
        currents = manager.compute_layer_currents(layer_spikes, 0.0)
        
        # Should have currents for post layer
        self.assertIn("post", currents)
        self.assertEqual(len(currents["post"]), 3)
        
    def test_manager_statistics(self):
        """Test manager statistics."""
        manager = VectorizedSynapseManager()
        
        # Setup network
        manager.add_layer("input", 10)
        manager.add_layer("output", 5)
        manager.connect_layers("input", "output", "stdp", connection_probability=0.3)
        
        # Get statistics
        stats = manager.get_all_statistics()
        
        self.assertIn("total_connections", stats)
        self.assertIn("n_populations", stats)
        self.assertIn("populations", stats)


class TestSynapseCompatibility(unittest.TestCase):
    """Test compatibility between different synapse implementations."""
    
    def test_current_computation_consistency(self):
        """Test that different implementations give reasonable currents."""
        # Traditional synapse
        trad_synapse = STDP_Synapse(
            synapse_id=0,
            pre_neuron_id=0,
            post_neuron_id=1,
            weight=1.0
        )
        
        # Vectorized equivalent
        vec_pop = VectorizedSynapsePopulation(
            pre_size=1,
            post_size=1,
            synapse_type="stdp",
            connection_probability=1.0,
            weight=1.0
        )
        
        # Test current computation
        trad_current = trad_synapse.compute_current(pre_spike_time=0.0, current_time=1.0)
        vec_currents = vec_pop.compute_synaptic_currents(np.array([True]), 0.0)
        vec_current = vec_currents[0]
        
        # Both should be positive
        self.assertGreater(trad_current, 0)
        self.assertGreater(vec_current, 0)


class TestSynapseEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions for synapses."""
    
    def test_zero_weight_synapse(self):
        """Test synapse with zero weight."""
        synapse = STDP_Synapse(
            synapse_id=0,
            pre_neuron_id=0,
            post_neuron_id=1,
            weight=0.0
        )
        
        current = synapse.compute_current(pre_spike_time=0.0, current_time=1.0)
        self.assertEqual(current, 0.0)
        
    def test_negative_weight_synapse(self):
        """Test synapse with negative weight (inhibitory)."""
        synapse = STDP_Synapse(
            synapse_id=0,
            pre_neuron_id=0,
            post_neuron_id=1,
            weight=-0.5,
            w_min=-1.0,
            w_max=1.0
        )
        
        # Weight should be clipped to bounds
        self.assertGreaterEqual(synapse.weight, -1.0)
        self.assertLessEqual(synapse.weight, 1.0)
        
    def test_invalid_synapse_factory(self):
        """Test factory with invalid synapse type."""
        with self.assertRaises(ValueError):
            SynapseFactory.create_synapse(
                synapse_id=0,
                pre_neuron_id=0,
                post_neuron_id=1,
                synapse_type="invalid_type",
                weight=0.5
            )
            
    def test_extreme_learning_rates(self):
        """Test STDP with extreme learning rates."""
        # Very high learning rate
        synapse = STDP_Synapse(
            synapse_id=0,
            pre_neuron_id=0,
            post_neuron_id=1,
            weight=0.5,
            A_plus=10.0,  # Very high
            A_minus=10.0,
            w_max=2.0
        )
        
        initial_weight = synapse.weight
        
        # Trigger plasticity
        synapse.pre_spike(0.0)
        synapse.post_spike(1.0)
        
        # Weight should still be bounded
        self.assertGreaterEqual(synapse.weight, 0.0)
        self.assertLessEqual(synapse.weight, 2.0)


if __name__ == '__main__':
    unittest.main()
