"""
Simplified tests for learning and plasticity mechanisms.
Tests only the available plasticity components.
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.learning import (
    PlasticityConfig,
    PlasticityManager,
    PlasticityType,
    STDPRule,
    HebbianRule,
    RewardModulatedSTDP,
    BCMRule,
    TripletSTDP,
    HomeostaticPlasticity,
    CustomPlasticityRule
)


class TestPlasticityConfig(unittest.TestCase):
    """Test plasticity configuration and parameter management."""
    
    def test_config_creation(self):
        """Test plasticity config creation."""
        config = PlasticityConfig(
            tau_plus=20.0,
            tau_minus=20.0,
            A_plus=0.01,
            A_minus=0.01,
            weight_min=0.0,
            weight_max=1.0
        )
        
        self.assertIsNotNone(config)
        self.assertEqual(config.tau_plus, 20.0)
        self.assertEqual(config.tau_minus, 20.0)
        self.assertEqual(config.A_plus, 0.01)
        self.assertEqual(config.A_minus, 0.01)
        
    def test_config_defaults(self):
        """Test default plasticity configuration."""
        config = PlasticityConfig()
        
        # Should have reasonable defaults
        self.assertGreater(config.tau_plus, 0)
        self.assertGreater(config.tau_minus, 0)
        self.assertGreater(config.A_plus, 0)
        self.assertGreater(config.A_minus, 0)
        self.assertGreaterEqual(config.weight_min, 0)
        self.assertGreater(config.weight_max, config.weight_min)


class TestPlasticityRules(unittest.TestCase):
    """Test individual plasticity rules."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = PlasticityConfig(
            tau_plus=20.0,
            tau_minus=20.0,
            A_plus=0.02,
            A_minus=0.02,
            weight_min=0.0,
            weight_max=2.0
        )
        
    def test_stdp_rule_creation(self):
        """Test STDP rule creation."""
        rule = STDPRule(self.config)
        
        self.assertIsNotNone(rule)
        self.assertEqual(rule.config, self.config)
        self.assertTrue(hasattr(rule, 'compute_weight_change'))
        
    def test_stdp_computation(self):
        """Test STDP weight change computation."""
        rule = STDPRule(self.config)
        
        # Test basic computation
        delta_w = rule.compute_weight_change(
            pre_activity=1.0,
            post_activity=1.0,
            current_weight=1.0,
            dt=1.0,
            pre_spike=True,
            post_spike=False
        )
        
        # Should be a valid number
        self.assertIsInstance(delta_w, (int, float, np.number))
        
    def test_hebbian_rule(self):
        """Test Hebbian learning rule."""
        rule = HebbianRule(self.config)
        
        # Correlated activity should strengthen synapse
        delta_w = rule.compute_weight_change(
            pre_activity=1.0,
            post_activity=1.0,
            current_weight=0.5
        )
        
        # Should be positive
        self.assertGreater(delta_w, 0.0)
        
        # Anti-correlated activity should weaken synapse
        delta_w = rule.compute_weight_change(
            pre_activity=1.0,
            post_activity=0.0,
            current_weight=0.5
        )
        
        # Should be negative or zero
        self.assertLessEqual(delta_w, 0.0)
        
    def test_bcm_rule(self):
        """Test BCM learning rule."""
        rule = BCMRule(self.config)
        
        # Test weight change computation
        delta_w = rule.compute_weight_change(
            pre_activity=1.0,
            post_activity=0.8,
            current_weight=0.5
        )
        
        self.assertIsInstance(delta_w, (int, float, np.number))
        
    def test_reward_modulated_stdp(self):
        """Test reward-modulated STDP."""
        rule = RewardModulatedSTDP(self.config)
        
        # Set reward
        rule.set_reward(1.0)
        
        # Test with reward
        delta_w_reward = rule.compute_weight_change(
            pre_activity=1.0,
            post_activity=1.0,
            current_weight=1.0,
            dt=1.0,
            pre_spike=True,
            post_spike=False
        )
        
        # Test without reward
        rule.set_reward(0.0)
        delta_w_no_reward = rule.compute_weight_change(
            pre_activity=1.0,
            post_activity=1.0,
            current_weight=1.0,
            dt=1.0,
            pre_spike=True,
            post_spike=False
        )
        
        # Both should be valid numbers
        self.assertIsInstance(delta_w_reward, (int, float, np.number))
        self.assertIsInstance(delta_w_no_reward, (int, float, np.number))
        
    def test_triplet_stdp(self):
        """Test triplet STDP rule."""
        rule = TripletSTDP(self.config)
        
        delta_w = rule.compute_weight_change(
            pre_activity=1.0,
            post_activity=1.0,
            current_weight=1.0,
            dt=1.0,
            pre_spike=True,
            post_spike=False
        )
        
        self.assertIsInstance(delta_w, (int, float, np.number))
        
    def test_homeostatic_plasticity(self):
        """Test homeostatic plasticity rule."""
        rule = HomeostaticPlasticity(self.config)
        
        delta_w = rule.compute_weight_change(
            pre_activity=1.0,
            post_activity=1.0,
            current_weight=1.0,
            post_spike=True,
            dt=1.0
        )
        
        self.assertIsInstance(delta_w, (int, float, np.number))


class TestPlasticityManager(unittest.TestCase):
    """Test plasticity manager functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = PlasticityConfig()
        self.manager = PlasticityManager(self.config)
        
    def test_manager_creation(self):
        """Test plasticity manager creation."""
        self.assertIsNotNone(self.manager)
        self.assertEqual(self.manager.config, self.config)
        self.assertTrue(hasattr(self.manager, 'active_rules'))
        
    def test_rule_activation(self):
        """Test plasticity rule activation."""
        # Activate STDP
        self.manager.activate_rule("stdp")
        self.assertIn("stdp", self.manager.active_rules)
        
        # Activate Hebbian
        self.manager.activate_rule("hebbian")
        self.assertIn("hebbian", self.manager.active_rules)
        
        # Deactivate rule
        self.manager.deactivate_rule("stdp")
        self.assertNotIn("stdp", self.manager.active_rules)
        
    def test_weight_updates(self):
        """Test weight matrix updates."""
        # Activate STDP
        self.manager.activate_rule("stdp")
        
        # Create test weight matrix
        weights = np.array([[0.5, 0.8], [0.3, 0.9]])
        pre_activity = np.array([1.0, 0.5])
        post_activity = np.array([0.8, 1.0])
        
        # Update weights
        updated_weights = self.manager.update_weights(
            weights=weights,
            pre_activity=pre_activity,
            post_activity=post_activity,
            pre_spike=np.array([True, False]),
            post_spike=np.array([False, True])
        )
        
        # Should return updated weight matrix
        self.assertEqual(updated_weights.shape, weights.shape)
        self.assertTrue(np.all(updated_weights >= 0))
        
    def test_custom_rule(self):
        """Test custom plasticity rule."""
        def custom_update(pre_activity, post_activity, current_weight, state, config, **kwargs):
            return 0.01 * pre_activity * post_activity
        
        # Add custom rule
        self.manager.add_custom_rule("custom", custom_update)
        
        # Check it was added
        self.assertIn("custom", self.manager.rules)
        
        # Activate and test
        self.manager.activate_rule("custom")
        self.assertIn("custom", self.manager.active_rules)
        
    def test_reward_setting(self):
        """Test reward signal setting."""
        # Set reward
        self.manager.set_reward(1.0)
        
        # Should not raise error
        self.assertTrue(True)
        
    def test_statistics(self):
        """Test manager statistics."""
        stats = self.manager.get_statistics()
        
        self.assertIn("active_rules", stats)
        self.assertIn("available_rules", stats)
        self.assertIn("config", stats)


class TestCustomPlasticity(unittest.TestCase):
    """Test custom plasticity mechanisms."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = PlasticityConfig()
        
    def test_custom_rule_creation(self):
        """Test custom rule creation."""
        def update_func(pre_activity, post_activity, current_weight, state, config, **kwargs):
            return 0.01 * (pre_activity - post_activity)
        
        rule = CustomPlasticityRule(self.config, update_func)
        
        self.assertIsNotNone(rule)
        self.assertEqual(rule.update_function, update_func)
        
    def test_custom_rule_computation(self):
        """Test custom rule weight computation."""
        def update_func(pre_activity, post_activity, current_weight, state, config, **kwargs):
            return 0.01 * pre_activity * post_activity
        
        rule = CustomPlasticityRule(self.config, update_func)
        
        delta_w = rule.compute_weight_change(
            pre_activity=1.0,
            post_activity=0.5,
            current_weight=0.8
        )
        
        self.assertAlmostEqual(delta_w, 0.005, places=6)
        
    def test_custom_rule_with_state(self):
        """Test custom rule with state variables."""
        def stateful_update(pre_activity, post_activity, current_weight, state, config, **kwargs):
            if 'counter' not in state:
                state['counter'] = 0
            state['counter'] += 1
            return 0.01 * state['counter']
        
        rule = CustomPlasticityRule(self.config, stateful_update)
        
        # First call
        delta_w1 = rule.compute_weight_change(1.0, 1.0, 1.0)
        # Second call
        delta_w2 = rule.compute_weight_change(1.0, 1.0, 1.0)
        
        # Should be different due to state
        self.assertNotEqual(delta_w1, delta_w2)
        self.assertAlmostEqual(delta_w1, 0.01, places=6)
        self.assertAlmostEqual(delta_w2, 0.02, places=6)


class TestLearningIntegration(unittest.TestCase):
    """Test integration of different learning mechanisms."""
    
    def setUp(self):
        """Set up integrated learning system."""
        self.config = PlasticityConfig()
        self.plasticity_manager = PlasticityManager(self.config)
        
    def test_integrated_learning_cycle(self):
        """Test complete learning cycle with multiple mechanisms."""
        # Initialize
        self.plasticity_manager.activate_rule("stdp")
        self.plasticity_manager.activate_rule("hebbian")
        
        # Create test data
        weights = np.array([[0.5, 0.3], [0.7, 0.4]])
        pre_activity = np.array([1.0, 0.8])
        post_activity = np.array([0.6, 1.0])
        
        # Simulate learning episodes
        for episode in range(5):
            # Apply plasticity
            weights = self.plasticity_manager.update_weights(
                weights=weights,
                pre_activity=pre_activity,
                post_activity=post_activity,
                pre_spike=np.array([True, False]),
                post_spike=np.array([False, True])
            )
        
        # Should have valid weights
        self.assertEqual(weights.shape, (2, 2))
        self.assertTrue(np.all(weights >= 0))
        
    def test_multiple_rule_interaction(self):
        """Test interaction between multiple plasticity rules."""
        # Activate multiple rules
        self.plasticity_manager.activate_rule("stdp")
        self.plasticity_manager.activate_rule("bcm")
        
        weights = np.array([[1.0]])
        pre_activity = np.array([1.0])
        post_activity = np.array([1.0])
        
        # Apply multiple rules
        updated_weights = self.plasticity_manager.update_weights(
            weights=weights,
            pre_activity=pre_activity,
            post_activity=post_activity,
            pre_spike=np.array([True]),
            post_spike=np.array([True])
        )
        
        # Should work without error
        self.assertEqual(updated_weights.shape, (1, 1))


class TestLearningEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions in learning."""
    
    def test_extreme_parameters(self):
        """Test learning with extreme parameters."""
        # Very small time constants
        config = PlasticityConfig(tau_plus=0.1, tau_minus=0.1)
        rule = STDPRule(config)
        
        delta_w = rule.compute_weight_change(1.0, 1.0, 1.0, dt=1.0)
        self.assertIsInstance(delta_w, (int, float, np.number))
        
        # Very large learning rates
        config = PlasticityConfig(A_plus=100.0, A_minus=100.0)
        rule = STDPRule(config)
        
        delta_w = rule.compute_weight_change(1.0, 1.0, 1.0, dt=1.0)
        self.assertIsInstance(delta_w, (int, float, np.number))
        
    def test_zero_activity(self):
        """Test learning with zero activity."""
        config = PlasticityConfig()
        rule = HebbianRule(config)
        
        # Update with zero activity
        delta_w = rule.compute_weight_change(0.0, 0.0, 1.0)
        
        self.assertIsInstance(delta_w, (int, float, np.number))
        
    def test_invalid_rule_activation(self):
        """Test handling of invalid rule activation."""
        manager = PlasticityManager()
        
        # Test with invalid rule name
        with self.assertRaises(ValueError):
            manager.activate_rule("invalid_rule")
            
    def test_disabled_plasticity(self):
        """Test plasticity with disabled config."""
        config = PlasticityConfig(enabled=False)
        rule = STDPRule(config)
        
        # Update weight
        new_weight = rule.update_weight(1.0, 1.0, 1.0)
        
        # Should remain unchanged
        self.assertEqual(new_weight, 1.0)


if __name__ == '__main__':
    unittest.main()
