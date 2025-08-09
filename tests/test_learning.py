"""
Comprehensive unit tests for learning and plasticity mechanisms.
"""

import unittest
import numpy as np
import tempfile
import json
import yaml
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.learning import (
    PlasticityType, PlasticityConfig, PlasticityRule,
    STDPRule, HebbianRule, BCMRule, RewardModulatedSTDP,
    TripletSTDP, HomeostaticPlasticity, CustomPlasticityRule,
    PlasticityManager, example_custom_rule
)


class TestPlasticityConfig(unittest.TestCase):
    """Test PlasticityConfig functionality."""
    
    def test_default_initialization(self):
        """Test default configuration values."""
        config = PlasticityConfig()
        
        self.assertEqual(config.learning_rate, 0.01)
        self.assertEqual(config.weight_min, 0.0)
        self.assertEqual(config.weight_max, 10.0)
        self.assertTrue(config.enabled)
        self.assertEqual(config.tau_plus, 20.0)
        self.assertEqual(config.tau_minus, 20.0)
    
    def test_custom_initialization(self):
        """Test custom configuration values."""
        config = PlasticityConfig(
            learning_rate=0.05,
            weight_min=-5.0,
            weight_max=15.0,
            tau_plus=30.0
        )
        
        self.assertEqual(config.learning_rate, 0.05)
        self.assertEqual(config.weight_min, -5.0)
        self.assertEqual(config.weight_max, 15.0)
        self.assertEqual(config.tau_plus, 30.0)
    
    def test_yaml_serialization(self):
        """Test YAML save and load."""
        config = PlasticityConfig(learning_rate=0.02, tau_plus=25.0)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config.to_yaml(f.name)
            loaded_config = PlasticityConfig.from_yaml(f.name)
        
        os.unlink(f.name)
        
        self.assertEqual(loaded_config.learning_rate, 0.02)
        self.assertEqual(loaded_config.tau_plus, 25.0)
    
    def test_json_serialization(self):
        """Test JSON save and load."""
        config = PlasticityConfig(learning_rate=0.03, weight_max=20.0)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config.to_json(f.name)
            loaded_config = PlasticityConfig.from_json(f.name)
        
        os.unlink(f.name)
        
        self.assertEqual(loaded_config.learning_rate, 0.03)
        self.assertEqual(loaded_config.weight_max, 20.0)


class TestSTDPRule(unittest.TestCase):
    """Test STDP plasticity rule."""
    
    def setUp(self):
        """Set up test configuration and rule."""
        self.config = PlasticityConfig(
            learning_rate=0.01,
            tau_plus=20.0,
            tau_minus=20.0,
            A_plus=0.01,
            A_minus=0.01,
            weight_min=0.0,
            weight_max=10.0
        )
        self.rule = STDPRule(self.config)
    
    def test_initialization(self):
        """Test STDP rule initialization."""
        self.assertEqual(self.rule.pre_trace, 0.0)
        self.assertEqual(self.rule.post_trace, 0.0)
        self.assertEqual(len(self.rule.weight_history), 0)
    
    def test_ltp_weight_change(self):
        """Test LTP (Long-Term Potentiation) weight change."""
        # Pre spike followed by post spike should increase weight
        current_weight = 5.0
        
        # Simulate pre spike
        delta_w = self.rule.compute_weight_change(
            pre_activity=1.0,
            post_activity=0.0,
            current_weight=current_weight,
            dt=1.0,
            pre_spike=True,
            post_spike=False
        )
        
        # Simulate post spike shortly after
        self.rule.pre_trace = 0.9  # Simulate decay
        delta_w = self.rule.compute_weight_change(
            pre_activity=0.0,
            post_activity=1.0,
            current_weight=current_weight,
            dt=1.0,
            pre_spike=False,
            post_spike=True
        )
        
        # Weight should increase (LTP)
        self.assertGreater(delta_w, 0)
    
    def test_ltd_weight_change(self):
        """Test LTD (Long-Term Depression) weight change."""
        current_weight = 5.0
        
        # Simulate post spike
        delta_w = self.rule.compute_weight_change(
            pre_activity=0.0,
            post_activity=1.0,
            current_weight=current_weight,
            dt=1.0,
            pre_spike=False,
            post_spike=True
        )
        
        # Simulate pre spike shortly after
        self.rule.post_trace = 0.9  # Simulate decay
        delta_w = self.rule.compute_weight_change(
            pre_activity=1.0,
            post_activity=0.0,
            current_weight=current_weight,
            dt=1.0,
            pre_spike=True,
            post_spike=False
        )
        
        # Weight should decrease (LTD)
        self.assertLess(delta_w, 0)
    
    def test_trace_decay(self):
        """Test trace decay over time."""
        # Set initial traces
        self.rule.pre_trace = 1.0
        self.rule.post_trace = 1.0
        
        # Compute with no spikes (just decay)
        self.rule.compute_weight_change(
            pre_activity=0.0,
            post_activity=0.0,
            current_weight=5.0,
            dt=10.0,
            pre_spike=False,
            post_spike=False
        )
        
        # Traces should decay
        self.assertLess(self.rule.pre_trace, 1.0)
        self.assertLess(self.rule.post_trace, 1.0)
    
    def test_weight_bounds(self):
        """Test weight boundary enforcement."""
        # Test upper bound
        new_weight = self.rule.update_weight(
            current_weight=9.9,
            pre_activity=1.0,
            post_activity=1.0,
            pre_spike=True,
            post_spike=False
        )
        self.assertLessEqual(new_weight, self.config.weight_max)
        
        # Test lower bound
        new_weight = self.rule.update_weight(
            current_weight=0.1,
            pre_activity=1.0,
            post_activity=1.0,
            pre_spike=False,
            post_spike=True
        )
        self.assertGreaterEqual(new_weight, self.config.weight_min)


class TestHebbianRule(unittest.TestCase):
    """Test Hebbian plasticity rule."""
    
    def setUp(self):
        """Set up test configuration and rule."""
        self.config = PlasticityConfig(
            learning_rate=0.01,
            hebbian_threshold=0.5,
            hebbian_decay=0.99,
            weight_min=0.0,
            weight_max=10.0
        )
        self.rule = HebbianRule(self.config)
    
    def test_correlation_above_threshold(self):
        """Test weight increase when correlation is above threshold."""
        current_weight = 5.0
        
        # High correlation (above threshold)
        delta_w = self.rule.compute_weight_change(
            pre_activity=0.9,
            post_activity=0.9,
            current_weight=current_weight
        )
        
        # Weight should increase
        self.assertGreater(delta_w, 0)
    
    def test_correlation_below_threshold(self):
        """Test weight decay when correlation is below threshold."""
        current_weight = 5.0
        
        # Low correlation (below threshold)
        delta_w = self.rule.compute_weight_change(
            pre_activity=0.3,
            post_activity=0.3,
            current_weight=current_weight
        )
        
        # Weight should decrease (decay)
        self.assertLess(delta_w, 0)
    
    def test_weight_dependent_scaling(self):
        """Test weight-dependent scaling for stability."""
        # Test with weight near maximum
        delta_w_high = self.rule.compute_weight_change(
            pre_activity=0.9,
            post_activity=0.9,
            current_weight=9.0
        )
        
        # Test with weight near minimum
        delta_w_low = self.rule.compute_weight_change(
            pre_activity=0.9,
            post_activity=0.9,
            current_weight=1.0
        )
        
        # Change should be smaller when weight is near maximum
        self.assertLess(abs(delta_w_high), abs(delta_w_low))


class TestBCMRule(unittest.TestCase):
    """Test BCM plasticity rule."""
    
    def setUp(self):
        """Set up test configuration and rule."""
        self.config = PlasticityConfig(
            learning_rate=0.01,
            bcm_threshold=0.5,
            bcm_time_constant=1000.0,
            weight_min=0.0,
            weight_max=10.0
        )
        self.rule = BCMRule(self.config)
    
    def test_sliding_threshold_initialization(self):
        """Test sliding threshold initialization."""
        self.assertEqual(self.rule.sliding_threshold, self.config.bcm_threshold)
        self.assertEqual(len(self.rule.activity_history), 0)
    
    def test_potentiation_above_threshold(self):
        """Test potentiation when activity is above threshold."""
        current_weight = 5.0
        
        # High postsynaptic activity
        delta_w = self.rule.compute_weight_change(
            pre_activity=0.8,
            post_activity=0.8,  # Above threshold
            current_weight=current_weight
        )
        
        # Weight should increase
        self.assertGreater(delta_w, 0)
    
    def test_depression_below_threshold(self):
        """Test depression when activity is below threshold."""
        current_weight = 5.0
        
        # Low postsynaptic activity
        delta_w = self.rule.compute_weight_change(
            pre_activity=0.8,
            post_activity=0.2,  # Below threshold
            current_weight=current_weight
        )
        
        # Weight should decrease
        self.assertLess(delta_w, 0)
    
    def test_sliding_threshold_adaptation(self):
        """Test sliding threshold adaptation."""
        initial_threshold = self.rule.sliding_threshold
        
        # Generate high activity
        for _ in range(10):
            self.rule.compute_weight_change(
                pre_activity=0.5,
                post_activity=0.9,
                current_weight=5.0
            )
        
        # Threshold should have adapted
        self.assertNotEqual(self.rule.sliding_threshold, initial_threshold)
    
    def test_activity_history_management(self):
        """Test activity history management."""
        # Generate many activities
        for i in range(150):
            self.rule.compute_weight_change(
                pre_activity=0.5,
                post_activity=0.5 + i * 0.001,
                current_weight=5.0
            )
        
        # History should be limited to 100 entries
        self.assertLessEqual(len(self.rule.activity_history), 100)


class TestRewardModulatedSTDP(unittest.TestCase):
    """Test reward-modulated STDP rule."""
    
    def setUp(self):
        """Set up test configuration and rule."""
        self.config = PlasticityConfig(
            learning_rate=0.01,
            reward_decay=0.9,
            reward_sensitivity=1.0,
            dopamine_time_constant=200.0
        )
        self.rule = RewardModulatedSTDP(self.config)
    
    def test_reward_signal_setting(self):
        """Test setting reward signal."""
        self.rule.set_reward(0.5)
        self.assertEqual(self.rule.reward_signal, 0.5)
        self.assertEqual(self.rule.dopamine_trace, 0.5)
    
    def test_eligibility_trace_accumulation(self):
        """Test eligibility trace accumulation."""
        # Generate STDP change
        self.rule.pre_trace = 0.5
        delta_w = self.rule.compute_weight_change(
            pre_activity=0.0,
            post_activity=1.0,
            current_weight=5.0,
            dt=1.0,
            pre_spike=False,
            post_spike=True
        )
        
        # Eligibility trace should be updated
        self.assertNotEqual(self.rule.eligibility_trace, 0.0)
    
    def test_reward_modulation(self):
        """Test that reward modulates plasticity."""
        # Set up eligibility trace
        self.rule.eligibility_trace = 0.1
        
        # Without reward
        self.rule.dopamine_trace = 0.0
        delta_w_no_reward = self.rule.compute_weight_change(
            pre_activity=0.5,
            post_activity=0.5,
            current_weight=5.0,
            dt=1.0,
            pre_spike=False,
            post_spike=False
        )
        
        # With reward
        self.rule.dopamine_trace = 1.0
        delta_w_with_reward = self.rule.compute_weight_change(
            pre_activity=0.5,
            post_activity=0.5,
            current_weight=5.0,
            dt=1.0,
            pre_spike=False,
            post_spike=False
        )
        
        # Change should be larger with reward
        self.assertGreater(abs(delta_w_with_reward), abs(delta_w_no_reward))
    
    def test_dopamine_trace_decay(self):
        """Test dopamine trace decay."""
        self.rule.dopamine_trace = 1.0
        
        # Compute with time step
        self.rule.compute_weight_change(
            pre_activity=0.0,
            post_activity=0.0,
            current_weight=5.0,
            dt=10.0,
            pre_spike=False,
            post_spike=False
        )
        
        # Dopamine trace should decay
        self.assertLess(self.rule.dopamine_trace, 1.0)


class TestTripletSTDP(unittest.TestCase):
    """Test triplet STDP rule."""
    
    def setUp(self):
        """Set up test configuration and rule."""
        self.config = PlasticityConfig(
            learning_rate=0.01,
            tau_x=5.0,
            tau_y=10.0,
            tau_plus=20.0,
            tau_minus=20.0,
            A2_plus=0.01,
            A3_plus=0.001,
            A2_minus=0.01,
            A3_minus=0.001
        )
        self.rule = TripletSTDP(self.config)
    
    def test_trace_initialization(self):
        """Test trace initialization."""
        self.assertEqual(self.rule.r1, 0.0)
        self.assertEqual(self.rule.r2, 0.0)
        self.assertEqual(self.rule.o1, 0.0)
        self.assertEqual(self.rule.o2, 0.0)
    
    def test_triplet_ltp(self):
        """Test triplet LTP mechanism."""
        # Set presynaptic trace
        self.rule.r1 = 0.5
        self.rule.o2 = 0.3
        
        # Post spike
        delta_w = self.rule.compute_weight_change(
            pre_activity=0.0,
            post_activity=1.0,
            current_weight=5.0,
            dt=1.0,
            pre_spike=False,
            post_spike=True
        )
        
        # Should have LTP
        self.assertGreater(delta_w, 0)
    
    def test_triplet_ltd(self):
        """Test triplet LTD mechanism."""
        # Set postsynaptic trace
        self.rule.o1 = 0.5
        self.rule.r2 = 0.3
        
        # Pre spike
        delta_w = self.rule.compute_weight_change(
            pre_activity=1.0,
            post_activity=0.0,
            current_weight=5.0,
            dt=1.0,
            pre_spike=True,
            post_spike=False
        )
        
        # Should have LTD
        self.assertLess(delta_w, 0)
    
    def test_trace_updates(self):
        """Test trace updates on spikes."""
        # Pre spike
        self.rule.compute_weight_change(
            pre_activity=1.0,
            post_activity=0.0,
            current_weight=5.0,
            dt=1.0,
            pre_spike=True,
            post_spike=False
        )
        
        self.assertEqual(self.rule.r1, 1.0)
        self.assertEqual(self.rule.r2, 1.0)
        
        # Post spike
        self.rule.compute_weight_change(
            pre_activity=0.0,
            post_activity=1.0,
            current_weight=5.0,
            dt=1.0,
            pre_spike=False,
            post_spike=True
        )
        
        self.assertEqual(self.rule.o1, 1.0)
        self.assertEqual(self.rule.o2, 1.0)


class TestHomeostaticPlasticity(unittest.TestCase):
    """Test homeostatic plasticity rule."""
    
    def setUp(self):
        """Set up test configuration and rule."""
        self.config = PlasticityConfig(
            learning_rate=0.01,
            target_rate=5.0,  # 5 Hz
            homeostatic_time_constant=10000.0
        )
        self.rule = HomeostaticPlasticity(self.config)
    
    def test_initialization(self):
        """Test initialization."""
        self.assertEqual(self.rule.firing_rate_estimate, 0.0)
        self.assertEqual(len(self.rule.time_window), 0)
    
    def test_firing_rate_estimation(self):
        """Test firing rate estimation."""
        dt = 1.0  # 1 ms
        
        # Simulate spikes
        for i in range(100):
            post_spike = (i % 20 == 0)  # Spike every 20 ms
            self.rule.compute_weight_change(
                pre_activity=0.5,
                post_activity=0.5,
                current_weight=5.0,
                post_spike=post_spike,
                dt=dt
            )
        
        # Should have recorded spike history
        self.assertGreater(len(self.rule.time_window), 0)
    
    def test_homeostatic_adjustment(self):
        """Test homeostatic weight adjustment."""
        current_weight = 5.0
        dt = 1.0
        
        # Simulate low firing rate
        for i in range(1000):
            self.rule.compute_weight_change(
                pre_activity=0.5,
                post_activity=0.5,
                current_weight=current_weight,
                post_spike=False,  # No spikes
                dt=dt
            )
        
        # Weight change should be positive (increase to reach target rate)
        delta_w = self.rule.compute_weight_change(
            pre_activity=0.5,
            post_activity=0.5,
            current_weight=current_weight,
            post_spike=False,
            dt=dt
        )
        
        self.assertGreater(delta_w, 0)
    
    def test_window_size_limit(self):
        """Test that time window size is limited."""
        dt = 1.0
        
        # Generate many time points
        for i in range(2000):
            self.rule.compute_weight_change(
                pre_activity=0.5,
                post_activity=0.5,
                current_weight=5.0,
                post_spike=(i % 100 == 0),
                dt=dt
            )
        
        # Window should be limited to approximately 1 second
        self.assertLessEqual(len(self.rule.time_window), 1100)


class TestCustomPlasticityRule(unittest.TestCase):
    """Test custom plasticity rule."""
    
    def setUp(self):
        """Set up test configuration and rule."""
        self.config = PlasticityConfig(learning_rate=0.01)
        self.rule = CustomPlasticityRule(self.config)
    
    def test_no_function_returns_zero(self):
        """Test that no function returns zero change."""
        delta_w = self.rule.compute_weight_change(
            pre_activity=0.5,
            post_activity=0.5,
            current_weight=5.0
        )
        self.assertEqual(delta_w, 0.0)
    
    def test_custom_function_execution(self):
        """Test custom function execution."""
        def custom_rule(pre, post, weight, state, config, **kwargs):
            return config.learning_rate * pre * post
        
        self.rule.set_update_function(custom_rule)
        
        delta_w = self.rule.compute_weight_change(
            pre_activity=0.5,
            post_activity=0.8,
            current_weight=5.0
        )
        
        expected = self.config.learning_rate * 0.5 * 0.8
        self.assertAlmostEqual(delta_w, expected, places=6)
    
    def test_state_persistence(self):
        """Test that state persists between calls."""
        def stateful_rule(pre, post, weight, state, config, **kwargs):
            if 'counter' not in state:
                state['counter'] = 0
            state['counter'] += 1
            return state['counter'] * 0.01
        
        self.rule.set_update_function(stateful_rule)
        
        # First call
        delta_w1 = self.rule.compute_weight_change(0.5, 0.5, 5.0)
        # Second call
        delta_w2 = self.rule.compute_weight_change(0.5, 0.5, 5.0)
        
        self.assertEqual(delta_w1, 0.01)
        self.assertEqual(delta_w2, 0.02)
    
    def test_example_custom_rule(self):
        """Test the example custom rule."""
        self.rule.set_update_function(example_custom_rule)
        
        # Test with depolarized voltage
        delta_w_depolarized = self.rule.compute_weight_change(
            pre_activity=1.0,
            post_activity=0.5,
            current_weight=5.0,
            post_voltage=-45.0
        )
        self.assertGreater(delta_w_depolarized, 0)
        
        # Test with hyperpolarized voltage
        delta_w_hyperpolarized = self.rule.compute_weight_change(
            pre_activity=1.0,
            post_activity=0.5,
            current_weight=5.0,
            post_voltage=-70.0
        )
        self.assertLess(delta_w_hyperpolarized, 0)


class TestPlasticityManager(unittest.TestCase):
    """Test plasticity manager."""
    
    def setUp(self):
        """Set up test manager."""
        self.config = PlasticityConfig(learning_rate=0.01)
        self.manager = PlasticityManager(self.config)
    
    def test_default_rules_initialization(self):
        """Test that default rules are initialized."""
        expected_rules = ['stdp', 'hebbian', 'bcm', 'rstdp', 'triplet_stdp', 'homeostatic']
        for rule_name in expected_rules:
            self.assertIn(rule_name, self.manager.rules)
    
    def test_activate_deactivate_rules(self):
        """Test activating and deactivating rules."""
        # Activate rule
        self.manager.activate_rule('stdp')
        self.assertIn('stdp', self.manager.active_rules)
        
        # Deactivate rule
        self.manager.deactivate_rule('stdp')
        self.assertNotIn('stdp', self.manager.active_rules)
    
    def test_add_custom_rule(self):
        """Test adding custom rules."""
        # Add as function
        def my_rule(pre, post, weight, state, config, **kwargs):
            return 0.001
        
        self.manager.add_custom_rule('my_rule', my_rule)
        self.assertIn('my_rule', self.manager.rules)
        
        # Add as PlasticityRule instance
        custom_rule = CustomPlasticityRule(self.config, my_rule)
        self.manager.add_custom_rule('my_rule2', custom_rule)
        self.assertIn('my_rule2', self.manager.rules)
    
    def test_update_weights_single_rule(self):
        """Test weight update with single rule."""
        weights = np.array([[1.0, 2.0], [3.0, 4.0]])
        pre_activity = np.array([0.5, 0.8])
        post_activity = np.array([0.6, 0.9])
        
        self.manager.activate_rule('hebbian')
        
        updated_weights = self.manager.update_weights(
            weights, pre_activity, post_activity, rule_name='hebbian'
        )
        
        # Weights should have changed
        self.assertFalse(np.array_equal(weights, updated_weights))
    
    def test_update_weights_multiple_rules(self):
        """Test weight update with multiple active rules."""
        weights = np.array([[1.0, 2.0], [3.0, 4.0]])
        pre_activity = np.array([0.5, 0.8])
        post_activity = np.array([0.6, 0.9])
        
        self.manager.activate_rule('hebbian')
        self.manager.activate_rule('homeostatic')
        
        updated_weights = self.manager.update_weights(
            weights, pre_activity, post_activity
        )
        
        # Weights should have changed
        self.assertFalse(np.array_equal(weights, updated_weights))
    
    def test_reward_setting(self):
        """Test setting reward for RSTDP."""
        self.manager.set_reward(0.5)
        
        rstdp_rule = self.manager.rules['rstdp']
        if isinstance(rstdp_rule, RewardModulatedSTDP):
            self.assertEqual(rstdp_rule.reward_signal, 0.5)
    
    def test_config_save_load(self):
        """Test configuration save and load."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            self.manager.save_config(f.name, format='yaml')
            
            # Create new manager and load config
            new_manager = PlasticityManager()
            new_manager.load_config(f.name, format='yaml')
        
        os.unlink(f.name)
        
        self.assertEqual(
            new_manager.config.learning_rate,
            self.manager.config.learning_rate
        )
    
    def test_statistics(self):
        """Test getting statistics."""
        # Activate some rules
        self.manager.activate_rule('stdp')
        self.manager.activate_rule('hebbian')
        
        # Perform some updates to generate history
        weights = np.array([[1.0, 2.0], [3.0, 4.0]])
        pre_activity = np.array([0.5, 0.8])
        post_activity = np.array([0.6, 0.9])
        
        for _ in range(5):
            weights = self.manager.update_weights(
                weights, pre_activity, post_activity
            )
        
        stats = self.manager.get_statistics()
        
        self.assertIn('active_rules', stats)
        self.assertIn('available_rules', stats)
        self.assertIn('config', stats)
        self.assertEqual(len(stats['active_rules']), 2)


class TestPlasticityIntegration(unittest.TestCase):
    """Integration tests for plasticity mechanisms."""
    
    def test_combined_plasticity_effects(self):
        """Test combined effects of multiple plasticity rules."""
        config = PlasticityConfig(learning_rate=0.01)
        manager = PlasticityManager(config)
        
        # Activate multiple rules
        manager.activate_rule('stdp')
        manager.activate_rule('homeostatic')
        
        # Initial weights
        weights = np.random.uniform(1, 5, (10, 10))
        initial_mean = np.mean(weights)
        
        # Simulate activity over time
        for t in range(100):
            pre_activity = np.random.random(10)
            post_activity = np.random.random(10)
            
            # Add spike information for STDP
            pre_spikes = pre_activity > 0.8
            post_spikes = post_activity > 0.8
            
            weights = manager.update_weights(
                weights,
                pre_activity,
                post_activity,
                pre_spike=pre_spikes,
                post_spike=post_spikes,
                dt=1.0
            )
        
        # Weights should have evolved
        final_mean = np.mean(weights)
        self.assertNotAlmostEqual(initial_mean, final_mean, places=2)
        
        # Should still be within bounds
        self.assertTrue(np.all(weights >= config.weight_min))
        self.assertTrue(np.all(weights <= config.weight_max))
    
    def test_plasticity_stability(self):
        """Test that plasticity rules maintain stability."""
        config = PlasticityConfig(
            learning_rate=0.01,
            weight_min=0.0,
            weight_max=10.0
        )
        
        # Test each rule for stability
        rules_to_test = ['stdp', 'hebbian', 'bcm', 'homeostatic']
        
        for rule_name in rules_to_test:
            with self.subTest(rule=rule_name):
                manager = PlasticityManager(config)
                manager.activate_rule(rule_name)
                
                weights = np.ones((5, 5)) * 5.0  # Start at middle
                
                # Run for many iterations
                for _ in range(1000):
                    pre_activity = np.random.random(5)
                    post_activity = np.random.random(5)
                    
                    weights = manager.update_weights(
                        weights,
                        pre_activity,
                        post_activity,
                        dt=1.0
                    )
                    
                    # Check for numerical stability
                    self.assertFalse(np.any(np.isnan(weights)))
                    self.assertFalse(np.any(np.isinf(weights)))
                    
                    # Check bounds
                    self.assertTrue(np.all(weights >= config.weight_min))
                    self.assertTrue(np.all(weights <= config.weight_max))


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def test_zero_learning_rate(self):
        """Test with zero learning rate."""
        config = PlasticityConfig(learning_rate=0.0)
        rule = HebbianRule(config)
        
        delta_w = rule.compute_weight_change(
            pre_activity=1.0,
            post_activity=1.0,
            current_weight=5.0
        )
        
        self.assertEqual(delta_w, 0.0)
    
    def test_disabled_rule(self):
        """Test disabled plasticity rule."""
        config = PlasticityConfig(enabled=False)
        rule = STDPRule(config)
        
        new_weight = rule.update_weight(
            current_weight=5.0,
            pre_activity=1.0,
            post_activity=1.0
        )
        
        self.assertEqual(new_weight, 5.0)  # No change when disabled
    
    def test_extreme_weights(self):
        """Test with extreme weight values."""
        config = PlasticityConfig(weight_min=-1e6, weight_max=1e6)
        rule = HebbianRule(config)
        
        # Test with very large weight
        delta_w = rule.compute_weight_change(
            pre_activity=0.5,
            post_activity=0.5,
            current_weight=1e5
        )
        
        self.assertFalse(np.isnan(delta_w))
        self.assertFalse(np.isinf(delta_w))
    
    def test_invalid_rule_activation(self):
        """Test activating non-existent rule."""
        manager = PlasticityManager()
        
        with self.assertRaises(ValueError):
            manager.activate_rule('non_existent_rule')


if __name__ == "__main__":
    unittest.main()
