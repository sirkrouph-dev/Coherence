#!/usr/bin/env python3
"""
Test suite for dopamine neuromodulation system.

This module tests the dopamine neuromodulation system to ensure it:
- Responds to reward/punishment with appropriate dopamine changes
- Implements reward prediction error calculation
- Modulates synaptic plasticity based on dopamine levels
- Learns to predict rewards through temporal difference learning
"""

import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the core modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.learning import (
    DopamineNeuromodulationSystem,
    EnhancedRewardModulatedSTDP,
    PlasticityConfig,
    PlasticityManager
)


class TestDopamineNeuromodulationSystem(unittest.TestCase):
    """Test dopamine neuromodulation system."""
    
    def setUp(self):
        """Set up test parameters."""
        self.config = PlasticityConfig(
            baseline_dopamine=0.5,
            max_dopamine=2.0,
            min_dopamine=0.0,
            dopamine_decay_rate=0.9,
            prediction_learning_rate=0.1,
            discount_factor=0.9,
            trace_decay=0.8
        )
        self.dopamine_system = DopamineNeuromodulationSystem(self.config)
        
    def test_reward_prediction_error_calculation(self):
        """Test reward prediction error computation."""
        # Initially, no prediction should exist
        initial_prediction = self.dopamine_system.reward_prediction
        self.assertEqual(initial_prediction, 0.0)
        
        # Give unexpected reward
        reward = 1.0
        prediction_error = self.dopamine_system.compute_reward_prediction_error(reward)
        
        # Should have positive prediction error (unexpected reward)
        self.assertGreater(prediction_error, 0.0, "Unexpected reward should give positive prediction error")
        
        # Prediction should increase
        self.assertGreater(self.dopamine_system.reward_prediction, initial_prediction)
        
        # Give same reward again - prediction error should be smaller
        second_error = self.dopamine_system.compute_reward_prediction_error(reward)
        self.assertLess(second_error, prediction_error, "Expected reward should give smaller prediction error")
        
    def test_dopamine_response_to_prediction_error(self):
        """Test dopamine level changes based on prediction error."""
        initial_dopamine = self.dopamine_system.current_dopamine
        
        # Positive prediction error should increase dopamine
        reward = 1.0
        new_dopamine = self.dopamine_system.update_dopamine_level(reward)
        self.assertGreater(new_dopamine, initial_dopamine, "Positive prediction error should increase dopamine")
        
        # Reset and test negative prediction error
        self.dopamine_system.current_dopamine = self.config.baseline_dopamine
        self.dopamine_system.reward_prediction = 1.0  # Expect reward
        
        # Give no reward when expected
        no_reward = 0.0
        new_dopamine = self.dopamine_system.update_dopamine_level(no_reward)
        self.assertLess(new_dopamine, self.config.baseline_dopamine, 
                       "Negative prediction error should decrease dopamine")
        
    def test_dopamine_bounds(self):
        """Test that dopamine levels stay within bounds."""
        # Test upper bound
        for _ in range(10):
            self.dopamine_system.update_dopamine_level(10.0)  # Large reward
            
        self.assertLessEqual(self.dopamine_system.current_dopamine, self.config.max_dopamine,
                           "Dopamine should not exceed maximum")
        
        # Reset and test lower bound
        self.dopamine_system.current_dopamine = self.config.baseline_dopamine
        self.dopamine_system.reward_prediction = 5.0  # High expectation
        
        for _ in range(10):
            self.dopamine_system.update_dopamine_level(-5.0)  # Large punishment
            
        self.assertGreaterEqual(self.dopamine_system.current_dopamine, self.config.min_dopamine,
                              "Dopamine should not go below minimum")
        
    def test_dopamine_decay(self):
        """Test dopamine decay toward baseline."""
        # Set dopamine above baseline
        self.dopamine_system.current_dopamine = 1.5
        initial_dopamine = self.dopamine_system.current_dopamine
        
        # Apply decay
        for _ in range(10):
            self.dopamine_system.decay_dopamine(dt=1.0)
            
        # Should decay toward baseline
        self.assertLess(self.dopamine_system.current_dopamine, initial_dopamine)
        self.assertGreater(self.dopamine_system.current_dopamine, self.config.baseline_dopamine * 0.8)
        
    def test_plasticity_modulation(self):
        """Test plasticity modulation based on dopamine level."""
        # Baseline dopamine should give ~1.0 modulation
        self.dopamine_system.current_dopamine = self.config.baseline_dopamine
        modulation = self.dopamine_system.get_plasticity_modulation()
        self.assertAlmostEqual(modulation, 1.0, places=1)
        
        # High dopamine should enhance plasticity
        self.dopamine_system.current_dopamine = self.config.max_dopamine
        high_modulation = self.dopamine_system.get_plasticity_modulation()
        self.assertGreater(high_modulation, 1.0, "High dopamine should enhance plasticity")
        
        # Low dopamine should reduce plasticity
        self.dopamine_system.current_dopamine = self.config.min_dopamine + 0.1
        low_modulation = self.dopamine_system.get_plasticity_modulation()
        self.assertLess(low_modulation, 1.0, "Low dopamine should reduce plasticity")
        
    def test_reward_prediction_learning(self):
        """Test that system learns to predict rewards."""
        # Simulate consistent reward pattern
        rewards = [1.0, 1.0, 1.0, 0.0, 0.0]  # Reward for first 3, then none
        prediction_errors = []
        
        # Train on pattern multiple times
        for epoch in range(10):
            for reward in rewards:
                error = self.dopamine_system.compute_reward_prediction_error(reward)
                prediction_errors.append(abs(error))
                
        # Prediction errors should decrease over time (learning)
        early_errors = np.mean(prediction_errors[:10])
        late_errors = np.mean(prediction_errors[-10:])
        self.assertLess(late_errors, early_errors, "Prediction errors should decrease with learning")
        
    def test_dopamine_state_monitoring(self):
        """Test dopamine state monitoring and reporting."""
        # Run some activity to populate state
        rewards = [1.0, 0.5, 0.0, -0.5, 1.0]
        for reward in rewards:
            self.dopamine_system.update_dopamine_level(reward)
            
        # Get dopamine state
        state = self.dopamine_system.get_dopamine_state()
        
        # Check all expected fields are present
        expected_fields = [
            'current_dopamine', 'baseline_dopamine', 'reward_prediction',
            'recent_prediction_error', 'prediction_accuracy', 'eligibility_trace',
            'plasticity_modulation'
        ]
        
        for field in expected_fields:
            self.assertIn(field, state, f"State should contain {field}")
            self.assertIsInstance(state[field], (int, float), f"{field} should be numeric")
            
        # Check reasonable values
        self.assertGreaterEqual(state['current_dopamine'], self.config.min_dopamine)
        self.assertLessEqual(state['current_dopamine'], self.config.max_dopamine)
        self.assertGreaterEqual(state['prediction_accuracy'], 0.0)
        self.assertLessEqual(state['prediction_accuracy'], 1.0)
        
    def test_dopamine_system_reset(self):
        """Test dopamine system reset functionality."""
        # Populate state with some activity
        for reward in [1.0, 0.5, -0.5, 1.0]:
            self.dopamine_system.update_dopamine_level(reward)
            
        # Verify state has been populated
        self.assertNotEqual(self.dopamine_system.current_dopamine, self.config.baseline_dopamine)
        self.assertGreater(len(self.dopamine_system.recent_rewards), 0)
        
        # Reset system
        self.dopamine_system.reset_dopamine_system()
        
        # Verify reset
        self.assertEqual(self.dopamine_system.current_dopamine, self.config.baseline_dopamine)
        self.assertEqual(self.dopamine_system.reward_prediction, 0.0)
        self.assertEqual(len(self.dopamine_system.recent_rewards), 0)
        self.assertEqual(len(self.dopamine_system.recent_predictions), 0)


class TestEnhancedRewardModulatedSTDP(unittest.TestCase):
    """Test enhanced reward-modulated STDP with dopamine system."""
    
    def setUp(self):
        """Set up test parameters."""
        self.config = PlasticityConfig(
            A_plus=0.01,
            A_minus=0.01,
            learning_rate=0.1,
            baseline_dopamine=0.5,
            max_dopamine=2.0,
            eligibility_strength=1.0,
            weight_min=0.0,
            weight_max=10.0
        )
        self.enhanced_rstdp = EnhancedRewardModulatedSTDP(self.config)
        
    def test_eligibility_trace_accumulation(self):
        """Test that eligibility traces accumulate with STDP events."""
        current_weight = 5.0
        
        # Simulate LTP-inducing pattern (pre before post)
        delta_w = self.enhanced_rstdp.compute_weight_change(
            pre_activity=1.0,
            post_activity=1.0,
            current_weight=current_weight,
            pre_spike=True,
            post_spike=True
        )
        
        # Should accumulate LTP eligibility
        self.assertGreater(self.enhanced_rstdp.ltp_eligibility, 0.0,
                          "LTP pattern should accumulate LTP eligibility")
        
    def test_reward_modulated_plasticity(self):
        """Test that rewards modulate plasticity through dopamine."""
        current_weight = 5.0
        
        # Build up eligibility trace
        for _ in range(5):
            self.enhanced_rstdp.compute_weight_change(
                pre_activity=1.0,
                post_activity=1.0,
                current_weight=current_weight,
                pre_spike=True,
                post_spike=True
            )
            
        # Apply reward
        reward_change = self.enhanced_rstdp.compute_weight_change(
            pre_activity=1.0,
            post_activity=1.0,
            current_weight=current_weight,
            reward=1.0  # Positive reward
        )
        
        # Should produce positive weight change due to reward
        self.assertGreater(reward_change, 0.0, "Reward should enhance LTP")
        
        # Test with punishment
        punishment_change = self.enhanced_rstdp.compute_weight_change(
            pre_activity=1.0,
            post_activity=1.0,
            current_weight=current_weight,
            reward=-1.0  # Punishment
        )
        
        # Should produce different (likely negative) weight change
        self.assertNotEqual(punishment_change, reward_change,
                           "Punishment should produce different plasticity than reward")
        
    def test_dopamine_level_effects(self):
        """Test effects of different dopamine levels on plasticity."""
        current_weight = 5.0
        
        # Set high dopamine
        self.enhanced_rstdp.dopamine_system.current_dopamine = 1.5
        self.enhanced_rstdp.ltp_eligibility = 0.1  # Some eligibility
        
        high_da_change = self.enhanced_rstdp.compute_weight_change(
            pre_activity=1.0,
            post_activity=1.0,
            current_weight=current_weight
        )
        
        # Reset and set low dopamine
        self.enhanced_rstdp.dopamine_system.current_dopamine = 0.1
        self.enhanced_rstdp.ltp_eligibility = 0.1  # Same eligibility
        
        low_da_change = self.enhanced_rstdp.compute_weight_change(
            pre_activity=1.0,
            post_activity=1.0,
            current_weight=current_weight
        )
        
        # High dopamine should produce more positive changes
        self.assertGreater(high_da_change, low_da_change,
                          "High dopamine should enhance plasticity more than low dopamine")
        
    def test_reward_prediction_learning(self):
        """Test that the system learns to predict rewards."""
        # Simulate consistent reward pattern
        current_weight = 5.0
        
        # Pattern: activity followed by reward
        initial_prediction = self.enhanced_rstdp.dopamine_system.reward_prediction
        
        for _ in range(20):
            # Activity
            self.enhanced_rstdp.compute_weight_change(
                pre_activity=1.0,
                post_activity=1.0,
                current_weight=current_weight,
                pre_spike=True,
                post_spike=True
            )
            
            # Reward
            self.enhanced_rstdp.compute_weight_change(
                pre_activity=1.0,
                post_activity=1.0,
                current_weight=current_weight,
                reward=1.0
            )
            
        # Prediction should have increased
        final_prediction = self.enhanced_rstdp.dopamine_system.reward_prediction
        self.assertGreater(final_prediction, initial_prediction,
                          "System should learn to predict rewards")
        
    def test_integration_with_plasticity_manager(self):
        """Test integration with PlasticityManager."""
        manager = PlasticityManager(self.config)
        
        # Activate enhanced reward-modulated STDP
        manager.activate_rule("enhanced_rstdp")
        
        # Set reward
        manager.set_reward(1.0, context_strength=0.8)
        
        # Get dopamine state
        dopamine_state = manager.get_dopamine_state()
        self.assertIsNotNone(dopamine_state, "Should return dopamine state")
        self.assertIn('current_dopamine', dopamine_state)
        
        # Create test weight matrix and activity
        weights = np.array([[1.0, 2.0], [3.0, 4.0]])
        pre_activity = np.array([0.5, 0.8])
        post_activity = np.array([0.3, 0.9])
        
        # Update weights
        updated_weights = manager.update_weights(
            weights, pre_activity, post_activity,
            pre_spike=np.array([True, False]),
            post_spike=np.array([False, True]),
            reward=1.0
        )
        
        # Should return updated weights
        self.assertEqual(updated_weights.shape, weights.shape)
        self.assertIsInstance(updated_weights, np.ndarray)
        
    def test_eligibility_trace_decay(self):
        """Test that eligibility traces decay over time."""
        current_weight = 5.0
        
        # Build up eligibility
        self.enhanced_rstdp.compute_weight_change(
            pre_activity=1.0,
            post_activity=1.0,
            current_weight=current_weight,
            pre_spike=True,
            post_spike=True
        )
        
        initial_ltp_eligibility = self.enhanced_rstdp.ltp_eligibility
        
        # Let time pass without activity
        for _ in range(10):
            self.enhanced_rstdp.compute_weight_change(
                pre_activity=0.0,
                post_activity=0.0,
                current_weight=current_weight,
                dt=1.0
            )
            
        # Eligibility should have decayed
        self.assertLess(self.enhanced_rstdp.ltp_eligibility, initial_ltp_eligibility,
                       "Eligibility traces should decay over time")


class TestDopamineSystemEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions for dopamine system."""
    
    def setUp(self):
        """Set up test parameters."""
        self.config = PlasticityConfig(baseline_dopamine=0.5)
        self.dopamine_system = DopamineNeuromodulationSystem(self.config)
        
    def test_extreme_rewards(self):
        """Test behavior with extreme reward values."""
        # Very large positive reward
        large_reward = 100.0
        self.dopamine_system.update_dopamine_level(large_reward)
        
        # Should remain bounded
        self.assertLessEqual(self.dopamine_system.current_dopamine, self.config.max_dopamine)
        
        # Very large negative reward
        large_punishment = -100.0
        self.dopamine_system.update_dopamine_level(large_punishment)
        
        # Should remain bounded
        self.assertGreaterEqual(self.dopamine_system.current_dopamine, self.config.min_dopamine)
        
    def test_zero_rewards(self):
        """Test behavior with zero rewards."""
        for _ in range(100):
            prediction_error = self.dopamine_system.compute_reward_prediction_error(0.0)
            
        # Should handle zero rewards gracefully
        self.assertFalse(np.isnan(self.dopamine_system.reward_prediction))
        self.assertFalse(np.isinf(self.dopamine_system.reward_prediction))
        
    def test_rapid_reward_changes(self):
        """Test behavior with rapidly changing rewards."""
        rewards = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0] * 20
        
        for reward in rewards:
            self.dopamine_system.update_dopamine_level(reward)
            
        # System should remain stable
        self.assertFalse(np.isnan(self.dopamine_system.current_dopamine))
        self.assertFalse(np.isinf(self.dopamine_system.current_dopamine))
        self.assertGreaterEqual(self.dopamine_system.current_dopamine, self.config.min_dopamine)
        self.assertLessEqual(self.dopamine_system.current_dopamine, self.config.max_dopamine)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)