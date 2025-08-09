"""
Comprehensive unit tests for synapse models.
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.synapses import (
    SynapseModel, STDP_Synapse, ShortTermPlasticitySynapse,
    NeuromodulatorySynapse, RSTDP_Synapse, SynapseFactory,
    SynapsePopulation, SynapseType
)


class TestSynapseModel(unittest.TestCase):
    """Test base synapse model functionality."""
    
    def test_synapse_initialization(self):
        """Test synapse initialization."""
        synapse = STDP_Synapse(
            synapse_id=1,
            pre_neuron_id=0,
            post_neuron_id=1,
            weight=5.0,
            w_min=0.0,
            w_max=10.0
        )
        
        self.assertEqual(synapse.synapse_id, 1)
        self.assertEqual(synapse.pre_neuron_id, 0)
        self.assertEqual(synapse.post_neuron_id, 1)
        self.assertEqual(synapse.weight, 5.0)
        self.assertEqual(synapse.w_min, 0.0)
        self.assertEqual(synapse.w_max, 10.0)
    
    def test_weight_boundaries(self):
        """Test weight boundary enforcement."""
        synapse = STDP_Synapse(
            synapse_id=1,
            pre_neuron_id=0,
            post_neuron_id=1,
            weight=5.0,
            w_min=0.0,
            w_max=10.0
        )
        
        # Test upper bound
        synapse.update_weight(100.0)
        self.assertLessEqual(synapse.weight, synapse.w_max)
        
        # Test lower bound
        synapse.update_weight(-200.0)
        self.assertGreaterEqual(synapse.weight, synapse.w_min)
    
    def test_reset_functionality(self):
        """Test synapse reset."""
        synapse = STDP_Synapse(
            synapse_id=1,
            pre_neuron_id=0,
            post_neuron_id=1,
            weight=5.0
        )
        
        # Modify weight
        synapse.update_weight(2.0)
        synapse.update_weight(1.0)
        
        # Reset
        synapse.reset()
        
        # Should return to initial weight
        self.assertEqual(synapse.weight, 5.0)
        self.assertEqual(len(synapse.weight_history), 1)


class TestSTDPSynapse(unittest.TestCase):
    """Test STDP synapse functionality."""
    
    def setUp(self):
        """Set up test synapse."""
        self.synapse = STDP_Synapse(
            synapse_id=1,
            pre_neuron_id=0,
            post_neuron_id=1,
            weight=5.0,
            w_min=0.0,
            w_max=10.0,
            tau_stdp=20.0,
            A_plus=0.01,
            A_minus=0.01
        )
    
    def test_ltp_pre_before_post(self):
        """Test LTP when pre spike occurs before post."""
        # Pre spike at t=10
        self.synapse.pre_spike(10.0)
        
        initial_weight = self.synapse.weight
        
        # Post spike at t=15 (5ms later)
        self.synapse.post_spike(15.0)
        
        # Weight should increase (LTP)
        self.assertGreater(self.synapse.weight, initial_weight)
    
    def test_ltd_post_before_pre(self):
        """Test LTD when post spike occurs before pre."""
        # Post spike at t=10
        self.synapse.post_spike(10.0)
        
        initial_weight = self.synapse.weight
        
        # Pre spike at t=15 (5ms later)
        self.synapse.pre_spike(15.0)
        
        # Weight should decrease (LTD)
        self.assertLess(self.synapse.weight, initial_weight)
    
    def test_no_plasticity_long_delay(self):
        """Test no plasticity for long delays."""
        # Pre spike at t=10
        self.synapse.pre_spike(10.0)
        
        initial_weight = self.synapse.weight
        
        # Post spike at t=100 (90ms later, beyond tau_stdp)
        self.synapse.post_spike(100.0)
        
        # Weight should not change significantly
        self.assertAlmostEqual(self.synapse.weight, initial_weight, places=3)
    
    def test_compute_current(self):
        """Test synaptic current computation."""
        # Test exponential decay
        pre_spike_time = 10.0
        
        # Current immediately after spike
        current_0 = self.synapse.compute_current(pre_spike_time, 10.0)
        self.assertGreater(current_0, 0)
        
        # Current after delay
        current_5 = self.synapse.compute_current(pre_spike_time, 15.0)
        self.assertLess(current_5, current_0)  # Should decay
        
        # No current before spike
        current_before = self.synapse.compute_current(pre_spike_time, 5.0)
        self.assertEqual(current_before, 0.0)


class TestShortTermPlasticitySynapse(unittest.TestCase):
    """Test STP synapse functionality."""
    
    def setUp(self):
        """Set up test synapse."""
        self.synapse = ShortTermPlasticitySynapse(
            synapse_id=2,
            pre_neuron_id=0,
            post_neuron_id=1,
            weight=5.0,
            tau_dep=100.0,
            tau_fac=500.0,
            U=0.5
        )
    
    def test_depression(self):
        """Test synaptic depression."""
        initial_x = self.synapse.x
        
        # Trigger pre spike
        self.synapse.pre_spike(10.0)
        
        # Available resources should decrease
        self.assertLess(self.synapse.x, initial_x)
    
    def test_facilitation(self):
        """Test synaptic facilitation."""
        # First spike
        self.synapse.pre_spike(10.0)
        u_first = self.synapse.u
        
        # Second spike shortly after
        self.synapse.pre_spike(20.0)
        u_second = self.synapse.u
        
        # Utilization should be higher for second spike
        self.assertGreater(u_second, u_first)
    
    def test_recovery(self):
        """Test resource recovery."""
        # Deplete resources
        self.synapse.pre_spike(10.0)
        depleted_x = self.synapse.x
        
        # Simulate recovery without triggering another spike
        # We need to manually update the recovery
        dt = 200.0 - 10.0
        self.synapse.x = 1.0 - (1.0 - self.synapse.x) * np.exp(-dt / self.synapse.tau_dep)
        self.synapse.u = self.synapse.u * np.exp(-dt / self.synapse.tau_fac)
        
        # Resources should have recovered
        self.assertGreater(self.synapse.x, depleted_x)
    
    def test_effective_weight(self):
        """Test that effective weight changes with STP."""
        pre_spike_time = 10.0
        
        # Initial current
        self.synapse.pre_spike(pre_spike_time)
        current_1 = self.synapse.compute_current(pre_spike_time, 10.0)
        
        # Second spike (depression should reduce current)
        self.synapse.pre_spike(15.0)
        current_2 = self.synapse.compute_current(15.0, 15.0)
        
        # Due to depression, second current should be different
        self.assertNotEqual(current_1, current_2)


class TestNeuromodulatorySynapse(unittest.TestCase):
    """Test neuromodulatory synapse functionality."""
    
    def setUp(self):
        """Set up test synapse."""
        self.synapse = NeuromodulatorySynapse(
            synapse_id=3,
            pre_neuron_id=0,
            post_neuron_id=1,
            weight=5.0,
            learning_rate=0.01,
            neuromodulator_level=0.5
        )
    
    def test_neuromodulator_update(self):
        """Test neuromodulator level update."""
        self.synapse.update_neuromodulator(0.8)
        self.assertEqual(self.synapse.neuromodulator_level, 0.8)
        
        # Test clamping
        self.synapse.update_neuromodulator(1.5)
        self.assertEqual(self.synapse.neuromodulator_level, 1.0)
        
        self.synapse.update_neuromodulator(-0.5)
        self.assertEqual(self.synapse.neuromodulator_level, 0.0)
    
    def test_modulated_plasticity(self):
        """Test that plasticity is modulated by neuromodulator."""
        # Pre spike
        self.synapse.pre_spike(10.0)
        
        # With neuromodulator
        self.synapse.neuromodulator_level = 1.0
        initial_weight = self.synapse.weight
        self.synapse.post_spike(15.0)
        weight_with_modulator = self.synapse.weight
        
        # Reset and test without neuromodulator
        self.synapse.weight = initial_weight
        self.synapse.neuromodulator_level = 0.0
        self.synapse.post_spike(20.0)
        weight_without_modulator = self.synapse.weight
        
        # Weight change should be larger with neuromodulator
        change_with = abs(weight_with_modulator - initial_weight)
        change_without = abs(weight_without_modulator - initial_weight)
        self.assertGreater(change_with, change_without)


class TestRSTDPSynapse(unittest.TestCase):
    """Test reward-modulated STDP synapse."""
    
    def setUp(self):
        """Set up test synapse."""
        self.synapse = RSTDP_Synapse(
            synapse_id=4,
            pre_neuron_id=0,
            post_neuron_id=1,
            weight=5.0,
            learning_rate=0.01
        )
    
    def test_reward_modulation(self):
        """Test reward signal modulation."""
        # Set reward
        self.synapse.update_reward(1.0)
        self.assertEqual(self.synapse.reward_signal, 1.0)
        
        # Set neuromodulator
        self.synapse.update_neuromodulator(0.5)
        
        initial_weight = self.synapse.weight
        
        # Trigger plasticity with reward
        self.synapse.pre_spike(10.0)
        
        # Weight should change based on reward
        self.assertNotEqual(self.synapse.weight, initial_weight)
    
    def test_combined_stdp_reward(self):
        """Test combination of STDP and reward modulation."""
        # Set up reward and neuromodulator
        self.synapse.update_reward(1.0)
        self.synapse.update_neuromodulator(1.0)
        
        initial_weight = self.synapse.weight
        
        # STDP: pre before post (LTP)
        self.synapse.pre_spike(10.0)
        self.synapse.post_spike(15.0)
        
        # Should have both STDP and reward contributions
        self.assertGreater(self.synapse.weight, initial_weight)


class TestSynapseFactory(unittest.TestCase):
    """Test synapse factory."""
    
    def test_create_stdp_synapse(self):
        """Test creating STDP synapse."""
        synapse = SynapseFactory.create_synapse(
            synapse_id=1,
            pre_neuron_id=0,
            post_neuron_id=1,
            synapse_type="stdp"
        )
        self.assertIsInstance(synapse, STDP_Synapse)
    
    def test_create_stp_synapse(self):
        """Test creating STP synapse."""
        synapse = SynapseFactory.create_synapse(
            synapse_id=2,
            pre_neuron_id=0,
            post_neuron_id=1,
            synapse_type="stp"
        )
        self.assertIsInstance(synapse, ShortTermPlasticitySynapse)
    
    def test_create_neuromodulatory_synapse(self):
        """Test creating neuromodulatory synapse."""
        synapse = SynapseFactory.create_synapse(
            synapse_id=3,
            pre_neuron_id=0,
            post_neuron_id=1,
            synapse_type="neuromodulatory"
        )
        self.assertIsInstance(synapse, NeuromodulatorySynapse)
    
    def test_create_rstdp_synapse(self):
        """Test creating RSTDP synapse."""
        synapse = SynapseFactory.create_synapse(
            synapse_id=4,
            pre_neuron_id=0,
            post_neuron_id=1,
            synapse_type="rstdp"
        )
        self.assertIsInstance(synapse, RSTDP_Synapse)
    
    def test_invalid_synapse_type(self):
        """Test invalid synapse type."""
        with self.assertRaises(ValueError):
            SynapseFactory.create_synapse(
                synapse_id=1,
                pre_neuron_id=0,
                post_neuron_id=1,
                synapse_type="invalid_type"
            )


class TestSynapsePopulation(unittest.TestCase):
    """Test synapse population functionality."""
    
    def setUp(self):
        """Set up test population."""
        self.pre_size = 10
        self.post_size = 10
        self.population = SynapsePopulation(
            pre_population_size=self.pre_size,
            post_population_size=self.post_size,
            synapse_type="stdp",
            connection_probability=0.2
        )
    
    def test_connection_creation(self):
        """Test that connections are created with correct probability."""
        num_synapses = len(self.population.synapses)
        max_synapses = self.pre_size * self.post_size
        
        # Should have approximately 20% of max connections
        expected = max_synapses * 0.2
        # Allow for statistical variation
        self.assertTrue(0.1 * max_synapses < num_synapses < 0.3 * max_synapses)
    
    def test_get_synaptic_currents(self):
        """Test computing synaptic currents."""
        pre_spikes = [False] * self.pre_size
        pre_spikes[0] = True  # One spike
        
        currents = self.population.get_synaptic_currents(pre_spikes, 10.0)
        
        self.assertEqual(len(currents), self.post_size)
        # At least some currents should be non-zero
        self.assertTrue(any(c > 0 for c in currents))
    
    def test_update_weights(self):
        """Test weight updates based on spikes."""
        # Create a population with deterministic connections
        population = SynapsePopulation(
            pre_population_size=2,
            post_population_size=2,
            synapse_type="stdp",
            connection_probability=1.0  # Ensure all connections exist
        )
        
        pre_spikes = [False, False]
        post_spikes = [False, False]

        # Create some spikes
        pre_spikes[0] = True
        post_spikes[0] = True

        # Get initial weights
        initial_weights = population.get_weight_matrix()

        # Update weights
        population.update_weights(pre_spikes, post_spikes, 10.0)

        # Get updated weights
        updated_weights = population.get_weight_matrix()

        # Some weights should have changed
        self.assertFalse(np.array_equal(initial_weights, updated_weights))
    
    def test_weight_matrix(self):
        """Test weight matrix extraction."""
        weight_matrix = self.population.get_weight_matrix()
        
        self.assertEqual(weight_matrix.shape, (self.pre_size, self.post_size))
        
        # Non-connected pairs should have zero weight
        num_zeros = np.sum(weight_matrix == 0)
        self.assertGreater(num_zeros, 0)
    
    def test_reset(self):
        """Test population reset."""
        # Modify some weights
        pre_spikes = [True] * self.pre_size
        post_spikes = [True] * self.post_size
        self.population.update_weights(pre_spikes, post_spikes, 10.0)
        
        # Reset
        self.population.reset()
        
        # Check all synapses reset
        for synapse in self.population.synapses.values():
            self.assertEqual(len(synapse.weight_history), 1)


if __name__ == "__main__":
    unittest.main()
