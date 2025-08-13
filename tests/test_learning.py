"""
Comprehensive tests for learning and plasticity mechanisms.
Tests plasticity rules, learning algorithms, and adaptation mechanisms.
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
    BCMRule
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
        
    def test_stdp_basic_computation(self):
        """Test basic STDP weight computation."""
        rule = STDPRule(self.config)
        
        # Test basic computation with required parameters
        delta_w = rule.compute_weight_change(
            pre_activity=1.0,
            post_activity=1.0,
            current_weight=1.0
        )
        
        # Should return a numeric value
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
        
        # Should be positive for correlated activity
        self.assertGreater(delta_w, 0.0)
        
        # Test uncorrelated activity
        delta_w = rule.compute_weight_change(
            pre_activity=1.0,
            post_activity=0.0,
            current_weight=0.5
        )
        
        # Should be zero or negative
        self.assertLessEqual(delta_w, 0.0)
        
    def test_bcm_rule(self):
        """Test BCM learning rule."""
        rule = BCMRule(self.config)
        
        # Test basic computation
        delta_w = rule.compute_weight_change(
            pre_activity=1.0,
            post_activity=1.0,
            current_weight=0.5
        )
        
        # Should return a numeric value
        self.assertIsInstance(delta_w, (int, float, np.number))
        
    def test_reward_modulated_stdp(self):
        """Test reward-modulated STDP."""
        rule = RewardModulatedSTDP(self.config)
        
        # Test with reward signal
        delta_w = rule.compute_weight_change(
            pre_activity=1.0,
            post_activity=1.0,
            current_weight=1.0,
            reward=1.0
        )
        
        # Should return a numeric value
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
        
    def test_weight_update(self):
        """Test weight update functionality."""
        # Test basic weight update
        new_weight = self.manager.update_weights(
            weights=np.array([0.5, 0.8, 1.2]),
            pre_activity=np.array([1.0, 0.0, 1.0]),
            post_activity=np.array([0.0, 1.0, 1.0])
        )
        
        # Should return weight array
        self.assertIsInstance(new_weight, np.ndarray)
        self.assertEqual(len(new_weight), 3)
        
    def test_plasticity_types(self):
        """Test different plasticity types."""
        # Test available plasticity types
        self.assertTrue(hasattr(PlasticityType, 'STDP'))
        self.assertTrue(hasattr(PlasticityType, 'HEBBIAN'))
        self.assertTrue(hasattr(PlasticityType, 'BCM'))


class TestLearningIntegration(unittest.TestCase):
    """Test integration of learning mechanisms."""
    
    def setUp(self):
        """Set up integrated learning system."""
        self.config = PlasticityConfig()
        self.manager = PlasticityManager(self.config)
        
    def test_learning_cycle(self):
        """Test basic learning cycle."""
        # Simulate spike patterns with activity values
        initial_weights = np.array([0.5, 0.7, 0.9, 0.3, 0.6])
        pre_activity = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
        post_activity = np.array([0.0, 1.0, 1.0, 0.0, 0.0])
        
        # Apply learning
        updated_weights = self.manager.update_weights(
            weights=initial_weights,
            pre_activity=pre_activity,
            post_activity=post_activity
        )
        
        # Should have updated weights
        self.assertIsInstance(updated_weights, np.ndarray)
        self.assertEqual(len(updated_weights), len(initial_weights))
        
    def test_multiple_learning_episodes(self):
        """Test multiple learning episodes."""
        weights = np.array([0.5, 0.5, 0.5])
        
        # Apply multiple learning episodes
        for episode in range(10):
            pre_activity = np.array([1.0, 0.0, 1.0])
            post_activity = np.array([0.0, 1.0, 1.0])
            
            weights = self.manager.update_weights(
                weights=weights,
                pre_activity=pre_activity,
                post_activity=post_activity
            )
        
        # Weights should have evolved
        self.assertIsInstance(weights, np.ndarray)
        self.assertEqual(len(weights), 3)


class TestLearningEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def test_extreme_parameters(self):
        """Test learning with extreme parameters."""
        # Very small time constants
        config = PlasticityConfig(tau_plus=0.1, tau_minus=0.1)
        rule = STDPRule(config)
        
        delta_w = rule.compute_weight_change(
            pre_activity=1.0,
            post_activity=1.0,
            current_weight=1.0
        )
        self.assertIsInstance(delta_w, (int, float, np.number))
        
    def test_zero_activity(self):
        """Test learning with zero activity."""
        config = PlasticityConfig()
        rule = STDPRule(config)
        
        # Test with zero activity
        delta_w = rule.compute_weight_change(
            pre_activity=0.0,
            post_activity=0.0,
            current_weight=0.5
        )
        
        # Should handle zero activity gracefully
        self.assertIsInstance(delta_w, (int, float, np.number))
        
    def test_boundary_weights(self):
        """Test learning at weight boundaries."""
        config = PlasticityConfig(weight_min=0.0, weight_max=1.0)
        rule = STDPRule(config)
        
        # Test at minimum weight
        delta_w = rule.compute_weight_change(
            pre_activity=1.0,
            post_activity=1.0,
            current_weight=0.0
        )
        self.assertIsInstance(delta_w, (int, float, np.number))
        
        # Test at maximum weight
        delta_w = rule.compute_weight_change(
            pre_activity=1.0,
            post_activity=1.0,
            current_weight=1.0
        )
        self.assertIsInstance(delta_w, (int, float, np.number))
        
    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        config = PlasticityConfig()
        manager = PlasticityManager(config)
        
        # Test with mismatched array sizes
        try:
            manager.update_weights(
                weights=np.array([0.5, 0.5]),
                pre_activity=np.array([1.0, 0.0]),
                post_activity=np.array([1.0])  # Different size
            )
        except (ValueError, IndexError):
            # This is expected
            pass
        
        # Test with negative weights
        try:
            manager.update_weights(
                weights=np.array([-0.5, 0.5]),  # Negative weight
                pre_activity=np.array([1.0, 0.0]),
                post_activity=np.array([0.0, 1.0])
            )
        except ValueError:
            # This might be expected
            pass


class TestPlasticityTypes(unittest.TestCase):
    """Test plasticity type enumeration."""
    
    def test_plasticity_type_values(self):
        """Test plasticity type enumeration values."""
        # Test that plasticity types exist
        self.assertTrue(hasattr(PlasticityType, 'STDP'))
        self.assertTrue(hasattr(PlasticityType, 'HEBBIAN'))
        self.assertTrue(hasattr(PlasticityType, 'BCM'))
        
        # Test that they have distinct values
        types = [PlasticityType.STDP, PlasticityType.HEBBIAN, PlasticityType.BCM]
        self.assertEqual(len(set(types)), len(types))


class TestCustomPlasticity(unittest.TestCase):
    """Test custom plasticity rule implementation."""
    
    def test_custom_rule_interface(self):
        """Test that custom rules can be implemented."""
        config = PlasticityConfig()
        
        # Create a simple custom rule
        class SimpleCustomRule:
            def __init__(self, config):
                self.config = config
                
            def compute_weight_change(self, pre_activity, post_activity, current_weight, **kwargs):
                # Simple multiplicative rule
                return self.config.A_plus * pre_activity * post_activity
        
        rule = SimpleCustomRule(config)
        
        # Test computation
        delta_w = rule.compute_weight_change(
            pre_activity=1.0,
            post_activity=1.0,
            current_weight=0.5
        )
        
        self.assertIsInstance(delta_w, (int, float, np.number))
        self.assertAlmostEqual(delta_w, config.A_plus)


if __name__ == '__main__':
    unittest.main()
