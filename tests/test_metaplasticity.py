#!/usr/bin/env python3
"""
Test suite for metaplasticity mechanisms.

This module tests the metaplasticity system to ensure it:
- Adapts plasticity thresholds based on activity history
- Modulates learning rates based on recent activity patterns
- Implements sliding threshold for LTP/LTD induction
- Stabilizes learning in dynamic environments
"""

import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the core modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.learning import (
    MetaplasticityRule,
    PlasticityConfig,
    PlasticityManager
)


class TestMetaplasticityRule(unittest.TestCase):
    """Test metaplasticity rule mechanisms."""
    
    def setUp(self):
        """Set up test parameters."""
        self.config = PlasticityConfig(
            A_plus=0.01,
            A_minus=0.01,
            learning_rate=0.1,
            metaplasticity_threshold=0.5,
            threshold_adaptation_rate=0.01,
            learning_rate_modulation=1.0,
            metaplasticity_window=100,
            weight_min=0.0,
            weight_max=10.0
        )
        self.metaplasticity = MetaplasticityRule(self.config)
        
    def test_sliding_threshold_adaptation(self):
        """Test that sliding threshold adapts to activity levels."""
        initial_threshold = self.metaplasticity.sliding_threshold
        
        # Test with high activity
        high_activity = 1.0
        self.metaplasticity.update_sliding_threshold(high_activity)
        
        # Threshold should increase with high activity
        self.assertGreater(self.metaplasticity.sliding_threshold, initial_threshold,
                          "Threshold should increase with high activity")
        
        # Reset and test with low activity
        self.metaplasticity.sliding_threshold = initial_threshold
        low_activity = 0.1
        self.metaplasticity.update_sliding_threshold(low_activity)
        
        # Threshold should decrease with low activity
        self.assertLess(self.metaplasticity.sliding_threshold, initial_threshold,
                       "Threshold should decrease with low activity")
        
    def test_learning_rate_modulation(self):
        """Test learning rate modulation based on activity."""
        initial_ltp_rate = self.metaplasticity.current_ltp_rate
        initial_ltd_rate = self.metaplasticity.current_ltd_rate
        
        # Test with high activity and low variance (stable high activity)
        high_activity = 1.0
        low_variance = 0.1
        self.metaplasticity.sliding_threshold = 0.5  # Set threshold below activity
        self.metaplasticity.modulate_learning_rates(high_activity, low_variance)
        
        # Should favor LTD over LTP for high activity
        self.assertLess(self.metaplasticity.current_ltp_rate, self.metaplasticity.current_ltd_rate,
                       "High activity should favor LTD over LTP")
        
        # Reset and test with low activity
        self.metaplasticity.current_ltp_rate = initial_ltp_rate
        self.metaplasticity.current_ltd_rate = initial_ltd_rate
        low_activity = 0.2
        self.metaplasticity.sliding_threshold = 0.5  # Set threshold above activity
        self.metaplasticity.modulate_learning_rates(low_activity, low_variance)
        
        # Should favor LTP over LTD for low activity
        self.assertGreater(self.metaplasticity.current_ltp_rate, self.metaplasticity.current_ltd_rate,
                          "Low activity should favor LTP over LTD")
        
    def test_variance_based_modulation(self):
        """Test that high variance reduces learning rates."""
        # Test with high variance (unstable activity)
        activity = 0.5
        high_variance = 1.0
        self.metaplasticity.modulate_learning_rates(activity, high_variance)
        high_var_ltp = self.metaplasticity.current_ltp_rate
        high_var_ltd = self.metaplasticity.current_ltd_rate
        
        # Reset and test with low variance
        self.metaplasticity.current_ltp_rate = self.config.A_plus
        self.metaplasticity.current_ltd_rate = self.config.A_minus
        low_variance = 0.1
        self.metaplasticity.modulate_learning_rates(activity, low_variance)
        low_var_ltp = self.metaplasticity.current_ltp_rate
        low_var_ltd = self.metaplasticity.current_ltd_rate
        
        # High variance should reduce learning rates
        self.assertLess(high_var_ltp, low_var_ltp, "High variance should reduce LTP rate")
        self.assertLess(high_var_ltd, low_var_ltd, "High variance should reduce LTD rate")
        
    def test_activity_dependent_plasticity(self):
        """Test that plasticity direction depends on activity relative to threshold."""
        current_weight = 5.0
        
        # Simulate high activity (above threshold)
        for _ in range(50):
            delta_w = self.metaplasticity.compute_weight_change(
                pre_activity=1.0,
                post_activity=0.8,  # High activity
                current_weight=current_weight,
                pre_spike=True,
                post_spike=True
            )
            current_weight += delta_w
            
        # Should show net potentiation for high activity
        self.assertGreater(current_weight, 5.0, "High activity should lead to potentiation")
        
        # Reset for low activity test
        self.metaplasticity = MetaplasticityRule(self.config)
        current_weight = 5.0
        
        # Simulate low activity (below threshold)
        for _ in range(50):
            delta_w = self.metaplasticity.compute_weight_change(
                pre_activity=1.0,
                post_activity=0.2,  # Low activity
                current_weight=current_weight,
                pre_spike=True,
                post_spike=True
            )
            current_weight += delta_w
            
        # Should show net depression for low activity
        self.assertLess(current_weight, 5.0, "Low activity should lead to depression")
        
    def test_metaplastic_state_monitoring(self):
        """Test metaplastic state monitoring and reporting."""
        # Run some activity to populate state
        for i in range(200):
            activity_level = 0.5 + 0.3 * np.sin(i * 0.1)  # Oscillating activity
            self.metaplasticity.compute_weight_change(
                pre_activity=1.0,
                post_activity=activity_level,
                current_weight=5.0,
                pre_spike=(i % 10 == 0),
                post_spike=(i % 8 == 0)
            )
            
        # Get metaplastic state
        state = self.metaplasticity.get_metaplastic_state()
        
        # Check all expected fields are present
        expected_fields = [
            'sliding_threshold', 'current_ltp_rate', 'current_ltd_rate',
            'recent_activity', 'activity_variance', 'recent_plasticity',
            'plasticity_threshold'
        ]
        
        for field in expected_fields:
            self.assertIn(field, state, f"State should contain {field}")
            self.assertIsInstance(state[field], (int, float), f"{field} should be numeric")
            
        # Check reasonable values
        self.assertGreater(state['sliding_threshold'], 0.0)
        self.assertGreater(state['current_ltp_rate'], 0.0)
        self.assertGreater(state['current_ltd_rate'], 0.0)
        
    def test_metaplastic_reset(self):
        """Test metaplastic state reset functionality."""
        # Populate state with some activity
        for i in range(100):
            self.metaplasticity.compute_weight_change(
                pre_activity=1.0,
                post_activity=0.7,
                current_weight=5.0,
                pre_spike=(i % 5 == 0),
                post_spike=(i % 7 == 0)
            )
            
        # Verify state has been populated
        self.assertGreater(len(self.metaplasticity.activity_history), 0)
        self.assertGreater(len(self.metaplasticity.plasticity_history), 0)
        
        # Reset state
        self.metaplasticity.reset_metaplastic_state()
        
        # Verify reset
        self.assertEqual(len(self.metaplasticity.activity_history), 0)
        self.assertEqual(len(self.metaplasticity.plasticity_history), 0)
        self.assertEqual(self.metaplasticity.sliding_threshold, self.config.metaplasticity_threshold)
        self.assertEqual(self.metaplasticity.current_ltp_rate, self.config.A_plus)
        self.assertEqual(self.metaplasticity.current_ltd_rate, self.config.A_minus)
        
    def test_threshold_bounds(self):
        """Test that sliding threshold stays within reasonable bounds."""
        # Test with extremely high activity
        for _ in range(100):
            self.metaplasticity.update_sliding_threshold(10.0)  # Very high activity
            
        # Should be bounded above
        self.assertLessEqual(self.metaplasticity.sliding_threshold, 2.0,
                           "Threshold should be bounded above")
        
        # Reset and test with extremely low activity
        self.metaplasticity.sliding_threshold = self.config.metaplasticity_threshold
        for _ in range(100):
            self.metaplasticity.update_sliding_threshold(0.0)  # Very low activity
            
        # Should be bounded below
        self.assertGreaterEqual(self.metaplasticity.sliding_threshold, 0.1,
                              "Threshold should be bounded below")
        
    def test_heterosynaptic_plasticity(self):
        """Test heterosynaptic plasticity (plasticity without pre-post pairing)."""
        current_weight = 5.0
        
        # Test post-spike without pre-spike (heterosynaptic LTD)
        delta_w = self.metaplasticity.compute_weight_change(
            pre_activity=0.0,
            post_activity=0.8,
            current_weight=current_weight,
            pre_spike=False,
            post_spike=True
        )
        
        # Should produce small LTD
        self.assertLess(delta_w, 0.0, "Post-spike without pre-spike should produce LTD")
        self.assertGreater(delta_w, -0.01, "Heterosynaptic LTD should be weak")
        
    def test_integration_with_plasticity_manager(self):
        """Test integration with PlasticityManager."""
        manager = PlasticityManager(self.config)
        
        # Activate metaplasticity
        manager.activate_rule("metaplasticity")
        
        # Create test weight matrix and activity
        weights = np.array([[1.0, 2.0], [3.0, 4.0]])
        pre_activity = np.array([0.5, 0.8])
        post_activity = np.array([0.3, 0.9])
        
        # Update weights
        updated_weights = manager.update_weights(
            weights, pre_activity, post_activity,
            pre_spike=np.array([True, False]),
            post_spike=np.array([False, True]),
            dt=1.0
        )
        
        # Should return updated weights
        self.assertEqual(updated_weights.shape, weights.shape)
        self.assertIsInstance(updated_weights, np.ndarray)
        
    def test_stability_in_dynamic_environment(self):
        """Test that metaplasticity stabilizes learning in dynamic environments."""
        weights = []
        current_weight = 5.0
        
        # Simulate dynamic environment with changing activity patterns
        for epoch in range(5):
            # Each epoch has different activity statistics
            if epoch % 2 == 0:
                # High activity epoch
                activity_pattern = lambda i: 0.7 + 0.2 * np.sin(i * 0.1)
            else:
                # Low activity epoch
                activity_pattern = lambda i: 0.3 + 0.1 * np.sin(i * 0.1)
                
            epoch_weights = []
            for step in range(100):
                activity = activity_pattern(step)
                delta_w = self.metaplasticity.compute_weight_change(
                    pre_activity=1.0,
                    post_activity=activity,
                    current_weight=current_weight,
                    pre_spike=True,
                    post_spike=(step % 5 == 0)
                )
                current_weight = np.clip(current_weight + delta_w, 0.0, 10.0)
                epoch_weights.append(current_weight)
                
            weights.extend(epoch_weights)
            
        # Check that weights remain stable (don't diverge)
        final_weights = weights[-100:]
        weight_variance = np.var(final_weights)
        
        # Should show reasonable stability
        self.assertLess(weight_variance, 1.0, "Weights should remain reasonably stable")
        
        # Check that weights stay within bounds
        self.assertGreaterEqual(min(weights), 0.0, "Weights should not go negative")
        self.assertLessEqual(max(weights), 10.0, "Weights should not exceed maximum")


class TestMetaplasticityEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions for metaplasticity."""
    
    def setUp(self):
        """Set up test parameters."""
        self.config = PlasticityConfig(metaplasticity_threshold=0.5)
        self.metaplasticity = MetaplasticityRule(self.config)
        
    def test_zero_activity(self):
        """Test behavior with zero activity."""
        delta_w = self.metaplasticity.compute_weight_change(
            pre_activity=0.0,
            post_activity=0.0,
            current_weight=5.0,
            pre_spike=False,
            post_spike=False
        )
        
        # Should handle zero activity gracefully
        self.assertEqual(delta_w, 0.0, "Zero activity should produce no weight change")
        
    def test_extreme_activity_values(self):
        """Test behavior with extreme activity values."""
        # Test with very high activity
        delta_w = self.metaplasticity.compute_weight_change(
            pre_activity=100.0,
            post_activity=100.0,
            current_weight=5.0,
            pre_spike=True,
            post_spike=True
        )
        
        # Should remain stable
        self.assertFalse(np.isnan(delta_w))
        self.assertFalse(np.isinf(delta_w))
        
    def test_empty_history(self):
        """Test behavior with empty activity history."""
        # Should work even with no history
        delta_w = self.metaplasticity.compute_weight_change(
            pre_activity=1.0,
            post_activity=0.5,
            current_weight=5.0,
            pre_spike=True,
            post_spike=True
        )
        
        self.assertIsInstance(delta_w, (int, float))
        self.assertFalse(np.isnan(delta_w))
        
    def test_constant_activity(self):
        """Test behavior with constant activity over long periods."""
        current_weight = 5.0
        constant_activity = 0.6
        
        # Run with constant activity for long period
        for _ in range(500):
            delta_w = self.metaplasticity.compute_weight_change(
                pre_activity=1.0,
                post_activity=constant_activity,
                current_weight=current_weight,
                pre_spike=True,
                post_spike=True
            )
            current_weight += delta_w
            
        # Should converge to stable state
        state = self.metaplasticity.get_metaplastic_state()
        self.assertAlmostEqual(state['recent_activity'], constant_activity, places=1)
        self.assertLess(state['activity_variance'], 0.1, "Variance should be low for constant activity")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)