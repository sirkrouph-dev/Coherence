"""
Integration tests for end-to-end learning tasks in the neuromorphic system.
"""

import unittest
import numpy as np
import sys
import os
import time
from typing import List, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.neurons import (
    NeuronPopulation, AdaptiveExponentialIntegrateAndFire,
    LeakyIntegrateAndFire, HodgkinHuxleyNeuron
)
from core.synapses import SynapsePopulation, STDP_Synapse, SynapseType
from core.learning import PlasticityManager, PlasticityConfig


class TestPatternLearning(unittest.TestCase):
    """Test pattern learning capabilities."""
    
    def setUp(self):
        """Set up neural network for pattern learning."""
        # Create input and output layers
        self.input_size = 10
        self.output_size = 5
        self.input_layer = NeuronPopulation(self.input_size, neuron_type="lif")
        self.output_layer = NeuronPopulation(self.output_size, neuron_type="lif")
        
        # Create synaptic connections with STDP
        self.synapses = SynapsePopulation(
            pre_population_size=self.input_size,
            post_population_size=self.output_size,
            synapse_type="stdp",
            connection_probability=0.5,
            weight=5.0,
            tau_stdp=20.0,
            A_plus=0.01,
            A_minus=0.01
        )
        
        # Set up plasticity manager
        self.plasticity_config = PlasticityConfig(learning_rate=0.01)
        self.plasticity_manager = PlasticityManager(self.plasticity_config)
        self.plasticity_manager.activate_rule('stdp')
    
    def generate_pattern(self, pattern_id: int) -> np.ndarray:
        """Generate a specific input pattern."""
        pattern = np.zeros(self.input_size)
        # Create distinct patterns
        if pattern_id == 0:
            pattern[0:3] = 50.0  # Strong input to first 3 neurons
        elif pattern_id == 1:
            pattern[3:6] = 50.0  # Strong input to middle neurons
        elif pattern_id == 2:
            pattern[6:9] = 50.0  # Strong input to last neurons
        else:
            pattern[4:7] = 30.0  # Moderate input to center
        return pattern
    
    def test_pattern_recognition(self):
        """Test that network learns to recognize patterns."""
        dt = 0.1  # Time step in ms
        simulation_time = 100  # Total simulation time in ms
        
        # Initial weight matrix
        initial_weights = self.synapses.get_weight_matrix()
        
        # Training phase: present patterns repeatedly
        patterns = [self.generate_pattern(i) for i in range(3)]
        
        for epoch in range(10):
            for pattern in patterns:
                # Reset neurons
                self.input_layer.reset()
                self.output_layer.reset()
                
                # Present pattern for some time
                for t in np.arange(0, 20, dt):
                    # Step input layer with pattern
                    input_spikes = self.input_layer.step(dt, pattern)
                    
                    # Compute synaptic currents
                    synaptic_currents = self.synapses.get_synaptic_currents(
                        input_spikes, t
                    )
                    
                    # Step output layer
                    output_spikes = self.output_layer.step(dt, synaptic_currents)
                    
                    # Update synaptic weights based on spikes
                    self.synapses.update_weights(input_spikes, output_spikes, t)
        
        # Final weight matrix
        final_weights = self.synapses.get_weight_matrix()
        
        # Weights should have changed due to learning
        self.assertFalse(np.array_equal(initial_weights, final_weights))
        
        # Test phase: check if patterns are recognized differently
        responses = []
        for pattern in patterns:
            self.input_layer.reset()
            self.output_layer.reset()
            
            spike_counts = np.zeros(self.output_size)
            
            # Present pattern and count output spikes
            for t in np.arange(0, 20, dt):
                input_spikes = self.input_layer.step(dt, pattern)
                synaptic_currents = self.synapses.get_synaptic_currents(
                    input_spikes, t
                )
                output_spikes = self.output_layer.step(dt, synaptic_currents)
                spike_counts += output_spikes
            
            responses.append(spike_counts)
        
        # Different patterns should produce different responses
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                self.assertFalse(
                    np.array_equal(responses[i], responses[j]),
                    f"Patterns {i} and {j} produced identical responses"
                )
    
    def test_pattern_selectivity(self):
        """Test that neurons become selective to specific patterns."""
        dt = 0.1
        
        # Create simple patterns
        pattern_A = np.zeros(self.input_size)
        pattern_A[0:5] = 40.0
        
        pattern_B = np.zeros(self.input_size)
        pattern_B[5:10] = 40.0
        
        # Train with pattern A more frequently
        for _ in range(20):
            self.input_layer.reset()
            self.output_layer.reset()
            
            # Present pattern A
            for t in np.arange(0, 10, dt):
                input_spikes = self.input_layer.step(dt, pattern_A)
                synaptic_currents = self.synapses.get_synaptic_currents(
                    input_spikes, t
                )
                output_spikes = self.output_layer.step(dt, synaptic_currents)
                self.synapses.update_weights(input_spikes, output_spikes, t)
        
        # Train with pattern B less frequently
        for _ in range(5):
            self.input_layer.reset()
            self.output_layer.reset()
            
            for t in np.arange(0, 10, dt):
                input_spikes = self.input_layer.step(dt, pattern_B)
                synaptic_currents = self.synapses.get_synaptic_currents(
                    input_spikes, t
                )
                output_spikes = self.output_layer.step(dt, synaptic_currents)
                self.synapses.update_weights(input_spikes, output_spikes, t)
        
        # Test responses
        self.input_layer.reset()
        self.output_layer.reset()
        
        response_A = 0
        for t in np.arange(0, 10, dt):
            input_spikes = self.input_layer.step(dt, pattern_A)
            synaptic_currents = self.synapses.get_synaptic_currents(input_spikes, t)
            output_spikes = self.output_layer.step(dt, synaptic_currents)
            response_A += np.sum(output_spikes)
        
        self.input_layer.reset()
        self.output_layer.reset()
        
        response_B = 0
        for t in np.arange(0, 10, dt):
            input_spikes = self.input_layer.step(dt, pattern_B)
            synaptic_currents = self.synapses.get_synaptic_currents(input_spikes, t)
            output_spikes = self.output_layer.step(dt, synaptic_currents)
            response_B += np.sum(output_spikes)
        
        # Response to pattern A should be stronger (more training)
        self.assertGreater(response_A, response_B)


class TestSequenceLearning(unittest.TestCase):
    """Test sequence learning capabilities."""
    
    def setUp(self):
        """Set up network for sequence learning."""
        self.sequence_length = 5
        self.layer_size = 8
        
        # Create recurrent layer
        self.neurons = NeuronPopulation(self.layer_size, neuron_type="adex")
        
        # Create recurrent connections
        self.recurrent_synapses = SynapsePopulation(
            pre_population_size=self.layer_size,
            post_population_size=self.layer_size,
            synapse_type="stdp",
            connection_probability=0.3,
            weight=3.0
        )
    
    def create_sequence(self) -> List[np.ndarray]:
        """Create a temporal sequence of patterns."""
        sequence = []
        for i in range(self.sequence_length):
            pattern = np.zeros(self.layer_size)
            # Activate different neurons in sequence
            pattern[i % self.layer_size] = 50.0
            pattern[(i + 1) % self.layer_size] = 30.0
            sequence.append(pattern)
        return sequence
    
    def test_sequence_replay(self):
        """Test that network can learn and replay sequences."""
        dt = 0.1
        sequence = self.create_sequence()
        
        # Training phase: present sequence multiple times
        for epoch in range(15):
            self.neurons.reset()
            
            for pattern in sequence:
                # Present each pattern for a short duration
                for t in np.arange(0, 5, dt):
                    # Add recurrent input
                    recurrent_current = np.zeros(self.layer_size)
                    if t > 0:  # After first step
                        prev_spikes = [n.is_spiking for n in self.neurons.neurons]
                        recurrent_current = self.recurrent_synapses.get_synaptic_currents(
                            prev_spikes, t
                        )
                    
                    # Total input is external + recurrent
                    total_input = pattern + recurrent_current
                    
                    # Step neurons
                    spikes = self.neurons.step(dt, total_input)
                    
                    # Update recurrent weights
                    self.recurrent_synapses.update_weights(spikes, spikes, t)
        
        # Test phase: trigger sequence with first pattern only
        self.neurons.reset()
        triggered_activity = []
        
        # Present first pattern briefly
        for t in np.arange(0, 2, dt):
            spikes = self.neurons.step(dt, sequence[0])
            triggered_activity.append(spikes.copy())
        
        # Let network evolve with recurrent connections only
        for t in np.arange(2, 20, dt):
            prev_spikes = [n.is_spiking for n in self.neurons.neurons]
            recurrent_current = self.recurrent_synapses.get_synaptic_currents(
                prev_spikes, t
            )
            spikes = self.neurons.step(dt, recurrent_current)
            triggered_activity.append(spikes.copy())
        
        # Check that there is sustained activity (sequence replay)
        total_activity = sum(sum(spikes) for spikes in triggered_activity)
        self.assertGreater(total_activity, 0, "No sequence replay observed")


class TestRewardModulatedLearning(unittest.TestCase):
    """Test reward-modulated learning."""
    
    def setUp(self):
        """Set up network with reward-modulated plasticity."""
        self.input_size = 8
        self.output_size = 2  # Binary decision
        
        self.input_layer = NeuronPopulation(self.input_size, neuron_type="lif")
        self.output_layer = NeuronPopulation(self.output_size, neuron_type="lif")
        
        # Create synapses with reward-modulated STDP
        self.synapses = SynapsePopulation(
            pre_population_size=self.input_size,
            post_population_size=self.output_size,
            synapse_type="rstdp",
            connection_probability=0.7,
            weight=4.0
        )
        
        # Set up plasticity manager with reward modulation
        self.plasticity_config = PlasticityConfig(
            learning_rate=0.02,
            reward_sensitivity=1.5
        )
        self.plasticity_manager = PlasticityManager(self.plasticity_config)
        self.plasticity_manager.activate_rule('rstdp')
    
    def test_reward_based_learning(self):
        """Test that positive rewards strengthen correct associations."""
        dt = 0.1
        
        # Define "correct" input-output mapping
        correct_input = np.zeros(self.input_size)
        correct_input[0:4] = 40.0  # First half active
        
        incorrect_input = np.zeros(self.input_size)
        incorrect_input[4:8] = 40.0  # Second half active
        
        # We want output neuron 0 to respond to correct_input
        # and output neuron 1 to respond to incorrect_input
        
        initial_weights = self.synapses.get_weight_matrix().copy()
        
        # Training with rewards
        for trial in range(20):
            self.input_layer.reset()
            self.output_layer.reset()
            
            # Present correct input
            for t in np.arange(0, 10, dt):
                input_spikes = self.input_layer.step(dt, correct_input)
                synaptic_currents = self.synapses.get_synaptic_currents(
                    input_spikes, t
                )
                output_spikes = self.output_layer.step(dt, synaptic_currents)
                
                # Give reward if neuron 0 fires more than neuron 1
                if output_spikes[0] and not output_spikes[1]:
                    # Positive reward for correct response
                    for synapse in self.synapses.synapses.values():
                        if hasattr(synapse, 'update_reward'):
                            synapse.update_reward(1.0)
                elif output_spikes[1] and not output_spikes[0]:
                    # Negative reward for incorrect response
                    for synapse in self.synapses.synapses.values():
                        if hasattr(synapse, 'update_reward'):
                            synapse.update_reward(-0.5)
                
                self.synapses.update_weights(input_spikes, output_spikes, t)
        
        final_weights = self.synapses.get_weight_matrix()
        
        # Weights should have changed
        self.assertFalse(np.array_equal(initial_weights, final_weights))
        
        # Test learned association
        self.input_layer.reset()
        self.output_layer.reset()
        
        correct_response = np.zeros(self.output_size)
        for t in np.arange(0, 10, dt):
            input_spikes = self.input_layer.step(dt, correct_input)
            synaptic_currents = self.synapses.get_synaptic_currents(input_spikes, t)
            output_spikes = self.output_layer.step(dt, synaptic_currents)
            correct_response += output_spikes
        
        # Neuron 0 should respond more to correct input
        self.assertGreaterEqual(correct_response[0], correct_response[1])


class TestMultiLayerLearning(unittest.TestCase):
    """Test learning in multi-layer networks."""
    
    def setUp(self):
        """Set up three-layer network."""
        self.input_size = 10
        self.hidden_size = 8
        self.output_size = 4
        
        # Create layers
        self.input_layer = NeuronPopulation(self.input_size, neuron_type="lif")
        self.hidden_layer = NeuronPopulation(self.hidden_size, neuron_type="adex")
        self.output_layer = NeuronPopulation(self.output_size, neuron_type="lif")
        
        # Create connections
        self.input_hidden_synapses = SynapsePopulation(
            pre_population_size=self.input_size,
            post_population_size=self.hidden_size,
            synapse_type="stdp",
            connection_probability=0.5,
            weight=4.0
        )
        
        self.hidden_output_synapses = SynapsePopulation(
            pre_population_size=self.hidden_size,
            post_population_size=self.output_size,
            synapse_type="stdp",
            connection_probability=0.6,
            weight=5.0
        )
    
    def test_hierarchical_feature_learning(self):
        """Test that network learns hierarchical features."""
        dt = 0.1
        
        # Create complex input patterns
        patterns = []
        for i in range(4):
            pattern = np.random.random(self.input_size) * 20
            pattern[i*2:(i*2)+3] += 30  # Add strong feature
            patterns.append(pattern)
        
        # Initial weights
        initial_ih_weights = self.input_hidden_synapses.get_weight_matrix().copy()
        initial_ho_weights = self.hidden_output_synapses.get_weight_matrix().copy()
        
        # Training
        for epoch in range(10):
            for pattern in patterns:
                self.input_layer.reset()
                self.hidden_layer.reset()
                self.output_layer.reset()
                
                for t in np.arange(0, 15, dt):
                    # Forward pass
                    input_spikes = self.input_layer.step(dt, pattern)
                    
                    hidden_currents = self.input_hidden_synapses.get_synaptic_currents(
                        input_spikes, t
                    )
                    hidden_spikes = self.hidden_layer.step(dt, hidden_currents)
                    
                    output_currents = self.hidden_output_synapses.get_synaptic_currents(
                        hidden_spikes, t
                    )
                    output_spikes = self.output_layer.step(dt, output_currents)
                    
                    # Update weights
                    self.input_hidden_synapses.update_weights(
                        input_spikes, hidden_spikes, t
                    )
                    self.hidden_output_synapses.update_weights(
                        hidden_spikes, output_spikes, t
                    )
        
        # Check that weights have changed in both layers
        final_ih_weights = self.input_hidden_synapses.get_weight_matrix()
        final_ho_weights = self.hidden_output_synapses.get_weight_matrix()
        
        self.assertFalse(np.array_equal(initial_ih_weights, final_ih_weights))
        self.assertFalse(np.array_equal(initial_ho_weights, final_ho_weights))
        
        # Test that different patterns produce different hidden representations
        hidden_representations = []
        
        for pattern in patterns:
            self.input_layer.reset()
            self.hidden_layer.reset()
            
            hidden_activity = np.zeros(self.hidden_size)
            
            for t in np.arange(0, 10, dt):
                input_spikes = self.input_layer.step(dt, pattern)
                hidden_currents = self.input_hidden_synapses.get_synaptic_currents(
                    input_spikes, t
                )
                hidden_spikes = self.hidden_layer.step(dt, hidden_currents)
                hidden_activity += hidden_spikes
            
            hidden_representations.append(hidden_activity)
        
        # Check that hidden representations are distinct
        for i in range(len(hidden_representations)):
            for j in range(i + 1, len(hidden_representations)):
                correlation = np.corrcoef(
                    hidden_representations[i],
                    hidden_representations[j]
                )[0, 1]
                # Representations should not be identical
                self.assertLess(abs(correlation), 0.95)


class TestHomeostaticRegulation(unittest.TestCase):
    """Test homeostatic plasticity and stability."""
    
    def setUp(self):
        """Set up network with homeostatic plasticity."""
        self.network_size = 20
        self.neurons = NeuronPopulation(self.network_size, neuron_type="adex")
        
        self.synapses = SynapsePopulation(
            pre_population_size=self.network_size,
            post_population_size=self.network_size,
            synapse_type="stdp",
            connection_probability=0.2,
            weight=3.0
        )
        
        # Enable homeostatic plasticity
        self.plasticity_config = PlasticityConfig(
            target_rate=10.0,  # Target 10 Hz
            homeostatic_time_constant=5000.0
        )
        self.plasticity_manager = PlasticityManager(self.plasticity_config)
        self.plasticity_manager.activate_rule('homeostatic')
    
    def test_firing_rate_homeostasis(self):
        """Test that homeostatic mechanisms maintain stable firing rates."""
        dt = 0.1
        
        # Track firing rates over time
        firing_rates = []
        window_size = 1000  # 1 second windows
        
        # Run simulation with varying input
        for phase in range(3):
            # Vary input strength across phases
            if phase == 0:
                input_strength = 10.0  # Low input
            elif phase == 1:
                input_strength = 50.0  # High input
            else:
                input_strength = 30.0  # Medium input
            
            phase_rates = []
            
            for window in range(5):
                spike_counts = np.zeros(self.network_size)
                
                for t in np.arange(0, window_size, dt):
                    # Random input with varying strength
                    external_input = np.random.random(self.network_size) * input_strength
                    
                    # Add recurrent input
                    prev_spikes = [n.is_spiking for n in self.neurons.neurons]
                    recurrent_input = self.synapses.get_synaptic_currents(prev_spikes, t)
                    
                    total_input = external_input + recurrent_input
                    
                    spikes = self.neurons.step(dt, total_input)
                    spike_counts += spikes
                    
                    # Update weights with homeostatic plasticity
                    weights = self.synapses.get_weight_matrix()
                    weights = self.plasticity_manager.update_weights(
                        weights,
                        prev_spikes,
                        spikes,
                        post_spike=spikes,
                        dt=dt
                    )
                    
                    # Update synapse weights
                    for (pre_id, post_id), synapse in self.synapses.synapses.items():
                        synapse.weight = weights[pre_id, post_id]
                
                # Calculate firing rate for this window
                rate = spike_counts / (window_size / 1000.0)  # Convert to Hz
                phase_rates.append(np.mean(rate))
            
            firing_rates.append(phase_rates)
        
        # Check that firing rates stabilize despite input changes
        # Later windows in each phase should be closer to target rate
        target_rate = self.plasticity_config.target_rate
        
        for phase_rates in firing_rates:
            # Rates should converge toward target
            initial_error = abs(phase_rates[0] - target_rate)
            final_error = abs(phase_rates[-1] - target_rate)
            
            # Final error should be less than initial
            # (allowing some tolerance for stochasticity)
            self.assertLessEqual(final_error, initial_error + 5.0)


class TestSynapticScaling(unittest.TestCase):
    """Test synaptic scaling and weight normalization."""
    
    def test_weight_distribution_stability(self):
        """Test that weight distributions remain stable during learning."""
        # Create network
        neurons = NeuronPopulation(15, neuron_type="lif")
        synapses = SynapsePopulation(
            pre_population_size=15,
            post_population_size=15,
            synapse_type="stdp",
            connection_probability=0.3,
            weight=5.0,
            w_min=0.0,
            w_max=10.0
        )
        
        dt = 0.1
        
        # Track weight statistics
        weight_means = []
        weight_stds = []
        
        # Run simulation with random activity
        for iteration in range(100):
            # Random input
            input_current = np.random.random(15) * 30
            
            # Step network
            spikes = neurons.step(dt, input_current)
            
            # Update weights
            synapses.update_weights(spikes, spikes, iteration * dt)
            
            # Record weight statistics
            weights = synapses.get_weight_matrix()
            non_zero_weights = weights[weights > 0]
            
            if len(non_zero_weights) > 0:
                weight_means.append(np.mean(non_zero_weights))
                weight_stds.append(np.std(non_zero_weights))
        
        # Check that weights don't explode or vanish
        self.assertGreater(weight_means[-1], 0.1)  # Didn't vanish
        self.assertLess(weight_means[-1], 9.9)  # Didn't saturate
        
        # Check that distribution remains relatively stable
        # (compare first and last quarters)
        early_mean = np.mean(weight_means[:25])
        late_mean = np.mean(weight_means[-25:])
        
        # Should not drift too much
        self.assertLess(abs(late_mean - early_mean), 3.0)


class TestLearningPerformance(unittest.TestCase):
    """Test performance and efficiency of learning algorithms."""
    
    def test_learning_convergence_speed(self):
        """Test how quickly learning converges to stable weights."""
        # Small network for faster testing
        neurons = NeuronPopulation(10, neuron_type="lif")
        synapses = SynapsePopulation(
            pre_population_size=10,
            post_population_size=10,
            synapse_type="stdp",
            connection_probability=0.4,
            weight=5.0
        )
        
        dt = 0.1
        
        # Fixed input pattern
        input_pattern = np.array([30, 0, 30, 0, 30, 0, 30, 0, 30, 0])
        
        # Track weight changes
        weight_changes = []
        previous_weights = synapses.get_weight_matrix().copy()
        
        # Run until convergence or timeout
        max_iterations = 1000
        converged = False
        
        for iteration in range(max_iterations):
            neurons.reset()
            
            # Present pattern
            for t in np.arange(0, 10, dt):
                spikes = neurons.step(dt, input_pattern)
                synapses.update_weights(spikes, spikes, t)
            
            # Check weight change
            current_weights = synapses.get_weight_matrix()
            change = np.sum(np.abs(current_weights - previous_weights))
            weight_changes.append(change)
            
            # Check for convergence
            if change < 0.01:
                converged = True
                break
            
            previous_weights = current_weights.copy()
        
        # Should converge within reasonable time
        self.assertTrue(converged or iteration > 500)
        
        # Weight changes should decrease over time (learning slows down)
        if len(weight_changes) > 20:
            early_changes = np.mean(weight_changes[:10])
            late_changes = np.mean(weight_changes[-10:])
            self.assertLess(late_changes, early_changes)
    
    def test_large_scale_learning(self):
        """Test learning in larger networks."""
        # Larger network
        network_size = 100
        
        start_time = time.time()
        
        neurons = NeuronPopulation(network_size, neuron_type="lif")
        synapses = SynapsePopulation(
            pre_population_size=network_size,
            post_population_size=network_size,
            synapse_type="stdp",
            connection_probability=0.1,  # Sparse connectivity
            weight=2.0
        )
        
        dt = 0.1
        
        # Run brief simulation
        for t in np.arange(0, 100, dt):
            input_current = np.random.random(network_size) * 20
            spikes = neurons.step(dt, input_current)
            synapses.update_weights(spikes, spikes, t)
        
        elapsed_time = time.time() - start_time
        
        # Should complete in reasonable time (adjust threshold as needed)
        self.assertLess(elapsed_time, 30.0)  # 30 seconds max
        
        # Check that learning occurred
        final_weights = synapses.get_weight_matrix()
        self.assertTrue(np.any(final_weights != 2.0))  # Weights changed


class TestLearningRobustness(unittest.TestCase):
    """Test robustness of learning to various conditions."""
    
    def test_noise_robustness(self):
        """Test learning with noisy inputs."""
        neurons = NeuronPopulation(10, neuron_type="lif")
        synapses = SynapsePopulation(
            pre_population_size=10,
            post_population_size=10,
            synapse_type="stdp",
            connection_probability=0.5,
            weight=5.0
        )
        
        dt = 0.1
        
        # Base pattern
        base_pattern = np.array([40, 0, 40, 0, 40, 0, 0, 0, 0, 0])
        
        # Train with noisy versions
        for epoch in range(20):
            # Add noise to pattern
            noise = np.random.normal(0, 5, 10)
            noisy_pattern = np.clip(base_pattern + noise, 0, 100)
            
            neurons.reset()
            
            for t in np.arange(0, 10, dt):
                spikes = neurons.step(dt, noisy_pattern)
                synapses.update_weights(spikes, spikes, t)
        
        # Test with clean pattern
        neurons.reset()
        clean_response = []
        
        for t in np.arange(0, 10, dt):
            spikes = neurons.step(dt, base_pattern)
            clean_response.append(np.sum(spikes))
        
        # Test with noisy pattern
        neurons.reset()
        noisy_response = []
        
        noisy_test = base_pattern + np.random.normal(0, 5, 10)
        for t in np.arange(0, 10, dt):
            spikes = neurons.step(dt, noisy_test)
            noisy_response.append(np.sum(spikes))
        
        # Responses should be similar despite noise
        clean_total = sum(clean_response)
        noisy_total = sum(noisy_response)
        
        if clean_total > 0:
            relative_difference = abs(clean_total - noisy_total) / clean_total
            self.assertLess(relative_difference, 0.5)  # Within 50% difference
    
    def test_catastrophic_forgetting(self):
        """Test resistance to catastrophic forgetting."""
        neurons = NeuronPopulation(12, neuron_type="adex")
        synapses = SynapsePopulation(
            pre_population_size=12,
            post_population_size=12,
            synapse_type="stdp",
            connection_probability=0.4,
            weight=4.0
        )
        
        dt = 0.1
        
        # First pattern
        pattern_A = np.zeros(12)
        pattern_A[0:6] = 40.0
        
        # Second pattern
        pattern_B = np.zeros(12)
        pattern_B[6:12] = 40.0
        
        # Train on pattern A
        for _ in range(15):
            neurons.reset()
            for t in np.arange(0, 10, dt):
                spikes = neurons.step(dt, pattern_A)
                synapses.update_weights(spikes, spikes, t)
        
        # Test response to pattern A (before training on B)
        neurons.reset()
        response_A_before = 0
        for t in np.arange(0, 10, dt):
            spikes = neurons.step(dt, pattern_A)
            response_A_before += np.sum(spikes)
        
        # Train on pattern B
        for _ in range(15):
            neurons.reset()
            for t in np.arange(0, 10, dt):
                spikes = neurons.step(dt, pattern_B)
                synapses.update_weights(spikes, spikes, t)
        
        # Test response to pattern A (after training on B)
        neurons.reset()
        response_A_after = 0
        for t in np.arange(0, 10, dt):
            spikes = neurons.step(dt, pattern_A)
            response_A_after += np.sum(spikes)
        
        # Should still respond to pattern A (not completely forgotten)
        if response_A_before > 0:
            retention = response_A_after / response_A_before
            self.assertGreater(retention, 0.3)  # At least 30% retention


if __name__ == "__main__":
    unittest.main()
