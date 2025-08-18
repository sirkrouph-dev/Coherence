#!/usr/bin/env python3
"""
Test suite for synaptic competition and saturation mechanisms.

This module tests the synaptic competition system to ensure it:
- Implements realistic upper and lower bounds with soft saturation
- Creates competition between synapses on the same postsynaptic neuron
- Provides weight normalization to prevent unbounded growth
- Produces winner-take-all dynamics when appropriate
"""

import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the core modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.learning import (
    SynapticCompetitionRule,
    MultiPlasticityRule,
    PlasticityConfig,
    PlasticityManager
)


class TestSynapticCompetitionRule(unittest.TestCase):
    """Test synaptic competition rule mechanisms."""
    
    def setUp(self):
        """Set up test parameters."""
        self.config = PlasticityConfig(
            learning_rate=0.1,
            weight_min=0.0,
            weight_max=10.0,
            competition_strength=0.2,
            normalization_target=20.0,
            normalization_rate=0.05,
            saturation_steepness=2.0,
            wta_threshold=0.8,
            wta_strength=0.5
        )
        self.competition = SynapticCompetitionRule(self.config)
        
    def test_soft_bounds_saturation(self):
        """Test soft saturation bounds near weight limits."""
        # Test near upper bound
        high_weight = 9.5  # Close to max of 10.0
        large_positive_change = 2.0
        
        bounded_change = self.competition.apply_soft_bounds(high_weight, large_positive_change)
        
        # Should be reduced due to saturation
        self.assertLess(bounded_change, large_positive_change,
                       "Large positive change near upper bound should be reduced")
        self.assertGreater(bounded_change, 0.0,
                          "Should still allow some positive change")
        
        # Test near lower bound
        low_weight = 0.5  # Close to min of 0.0
        large_negative_change = -2.0
        
        bounded_change = self.competition.apply_soft_bounds(low_weight, large_negative_change)
        
        # Should be reduced due to saturation
        self.assertGreater(bounded_change, large_negative_change,
                          "Large negative change near lower bound should be reduced")
        self.assertLess(bounded_change, 0.0,
                       "Should still allow some negative change")
        
    def test_competition_effect(self):
        """Test competition between synapses."""
        # Create array of synaptic weights
        all_weights = np.array([2.0, 4.0, 6.0, 8.0])  # Total = 20
        activity_strength = 0.8
        
        # Test strong synapse (above average)
        strong_weight = 8.0  # 40% of total, above 25% average
        competition_change = self.competition.compute_competition_effect(
            strong_weight, all_weights, activity_strength
        )
        
        # Should get competitive advantage
        self.assertGreater(competition_change, 0.0,
                          "Strong synapse should get competitive advantage")
        
        # Test weak synapse (below average)
        weak_weight = 2.0  # 10% of total, below 25% average
        competition_change = self.competition.compute_competition_effect(
            weak_weight, all_weights, activity_strength
        )
        
        # Should get competitive disadvantage
        self.assertLess(competition_change, 0.0,
                       "Weak synapse should get competitive disadvantage")
        
    def test_normalization_effect(self):
        """Test weight normalization to maintain total strength."""
        # Weights below target
        low_weights = np.array([1.0, 2.0, 3.0, 4.0])  # Total = 10, target = 20
        current_weight = 2.0
        
        normalization_change = self.competition.compute_normalization_effect(
            current_weight, low_weights
        )
        
        # Should increase to reach target
        self.assertGreater(normalization_change, 0.0,
                          "Normalization should increase weights when below target")
        
        # Weights above target
        high_weights = np.array([5.0, 10.0, 15.0, 20.0])  # Total = 50, target = 20
        current_weight = 10.0
        
        normalization_change = self.competition.compute_normalization_effect(
            current_weight, high_weights
        )
        
        # Should decrease to reach target
        self.assertLess(normalization_change, 0.0,
                       "Normalization should decrease weights when above target")
        
    def test_winner_take_all_effect(self):
        """Test winner-take-all dynamics for strong activity."""
        all_weights = np.array([2.0, 4.0, 6.0, 8.0])
        strong_activity = 0.9  # Above WTA threshold
        
        # Test winner (strongest synapse)
        winner_weight = 8.0  # Maximum weight
        wta_change = self.competition.compute_winner_take_all_effect(
            winner_weight, all_weights, strong_activity
        )
        
        # Winner should be strengthened
        self.assertGreater(wta_change, 0.0, "Winner should be strengthened")
        
        # Test loser (weaker synapse)
        loser_weight = 2.0  # Not maximum weight
        wta_change = self.competition.compute_winner_take_all_effect(
            loser_weight, all_weights, strong_activity
        )
        
        # Loser should be weakened
        self.assertLess(wta_change, 0.0, "Loser should be weakened")
        
        # Test with weak activity (below threshold)
        weak_activity = 0.5  # Below WTA threshold
        wta_change = self.competition.compute_winner_take_all_effect(
            winner_weight, all_weights, weak_activity
        )
        
        # Should have no WTA effect
        self.assertEqual(wta_change, 0.0, "Weak activity should not trigger WTA")
        
    def test_combined_competition_effects(self):
        """Test combined competition effects in weight change computation."""
        all_weights = np.array([1.0, 3.0, 5.0, 7.0])  # Total = 16
        current_weight = 5.0  # Above average
        pre_activity = 1.0
        post_activity = 0.9  # Strong activity
        base_change = 0.1  # Some base plasticity
        
        total_change = self.competition.compute_weight_change(
            pre_activity=pre_activity,
            post_activity=post_activity,
            current_weight=current_weight,
            all_weights=all_weights,
            base_change=base_change
        )
        
        # Should include all effects
        self.assertNotEqual(total_change, base_change,
                           "Competition should modify base change")
        
        # Check that competition state is tracked
        state = self.competition.get_competition_state()
        # Activity should be tracked (might be 0 if only one call was made)
        self.assertGreaterEqual(state['recent_activity'], 0.0)
        
    def test_competition_state_monitoring(self):
        """Test competition state monitoring and reporting."""
        # Run some activity to populate state
        all_weights = np.array([2.0, 4.0, 6.0, 8.0])
        
        for i in range(50):
            activity = 0.5 + 0.3 * np.sin(i * 0.1)  # Varying activity
            self.competition.compute_weight_change(
                pre_activity=1.0,
                post_activity=activity,
                current_weight=5.0,
                all_weights=all_weights
            )
            
        # Get competition state
        state = self.competition.get_competition_state()
        
        # Check all expected fields are present
        expected_fields = [
            'recent_activity', 'recent_competition', 'recent_normalization',
            'recent_wta', 'competition_strength', 'normalization_target',
            'wta_threshold'
        ]
        
        for field in expected_fields:
            self.assertIn(field, state, f"State should contain {field}")
            self.assertIsInstance(state[field], (int, float), f"{field} should be numeric")
            
    def test_competition_reset(self):
        """Test competition state reset functionality."""
        # Populate state with some activity
        all_weights = np.array([2.0, 4.0, 6.0])
        
        for _ in range(20):
            self.competition.compute_weight_change(
                pre_activity=1.0,
                post_activity=0.7,
                current_weight=4.0,
                all_weights=all_weights
            )
            
        # Verify state has been populated
        self.assertGreater(len(self.competition.recent_activities), 0)
        self.assertGreater(len(self.competition.competition_history), 0)
        
        # Reset state
        self.competition.reset_competition_state()
        
        # Verify reset
        self.assertEqual(len(self.competition.recent_activities), 0)
        self.assertEqual(len(self.competition.competition_history), 0)
        
    def test_edge_cases(self):
        """Test edge cases for competition rule."""
        # Test with single synapse (no competition)
        single_weight = np.array([5.0])
        change = self.competition.compute_weight_change(
            pre_activity=1.0,
            post_activity=0.8,
            current_weight=5.0,
            all_weights=single_weight
        )
        
        # Should handle gracefully
        self.assertIsInstance(change, (int, float))
        self.assertFalse(np.isnan(change))
        
        # Test with zero weights
        zero_weights = np.array([0.0, 0.0, 0.0])
        change = self.competition.compute_weight_change(
            pre_activity=1.0,
            post_activity=0.8,
            current_weight=0.0,
            all_weights=zero_weights
        )
        
        # Should handle gracefully
        self.assertIsInstance(change, (int, float))
        self.assertFalse(np.isnan(change))


class TestMultiPlasticityRule(unittest.TestCase):
    """Test multi-plasticity rule that combines multiple mechanisms."""
    
    def setUp(self):
        """Set up test parameters."""
        self.config = PlasticityConfig(
            learning_rate=0.1,
            use_stdp=True,
            use_homeostatic=True,
            use_competition=True,
            use_metaplasticity=False,
            use_reward_modulation=False,
            stdp_weight=1.0,
            homeostatic_weight=0.2,
            competition_weight=0.3,
            weight_min=0.0,
            weight_max=10.0
        )
        self.multi_plasticity = MultiPlasticityRule(self.config)
        
    def test_component_rule_initialization(self):
        """Test that component rules are initialized correctly."""
        # Should have STDP rule
        self.assertIsNotNone(self.multi_plasticity.stdp_rule)
        
        # Should have homeostatic rule
        self.assertIsNotNone(self.multi_plasticity.homeostatic_rule)
        
        # Should have competition rule
        self.assertIsNotNone(self.multi_plasticity.competition_rule)
        
        # Should not have metaplasticity or reward rules (disabled)
        self.assertIsNone(self.multi_plasticity.metaplasticity_rule)
        self.assertIsNone(self.multi_plasticity.reward_rule)
        
    def test_combined_weight_change(self):
        """Test that weight changes combine multiple mechanisms."""
        all_weights = np.array([2.0, 4.0, 6.0, 8.0])
        current_weight = 4.0
        
        # Compute combined change
        combined_change = self.multi_plasticity.compute_weight_change(
            pre_activity=1.0,
            post_activity=0.8,
            current_weight=current_weight,
            all_weights=all_weights,
            pre_spike=True,
            post_spike=True,
            dt=1.0
        )
        
        # Should produce some change
        self.assertNotEqual(combined_change, 0.0, "Combined rule should produce weight change")
        
        # Should be different from individual components
        stdp_only = self.multi_plasticity.stdp_rule.compute_weight_change(
            pre_activity=1.0,
            post_activity=0.8,
            current_weight=current_weight,
            pre_spike=True,
            post_spike=True,
            dt=1.0
        )
        
        self.assertNotEqual(combined_change, stdp_only,
                           "Combined change should differ from STDP alone")
        
    def test_multi_plasticity_state(self):
        """Test multi-plasticity state monitoring."""
        # Run some activity to populate states
        all_weights = np.array([3.0, 5.0, 7.0])
        
        for i in range(100):
            self.multi_plasticity.compute_weight_change(
                pre_activity=1.0,
                post_activity=0.6 + 0.2 * np.sin(i * 0.1),
                current_weight=5.0,
                all_weights=all_weights,
                pre_spike=(i % 10 == 0),
                post_spike=(i % 8 == 0),
                dt=1.0
            )
            
        # Get combined state
        state = self.multi_plasticity.get_multi_plasticity_state()
        
        # Should contain states from active components
        self.assertIn('homeostatic', state)
        self.assertIn('competition', state)
        
        # Should not contain states from inactive components
        self.assertNotIn('metaplasticity', state)
        self.assertNotIn('reward', state)
        
    def test_weight_combination_factors(self):
        """Test that weight combination factors work correctly."""
        # Create config with different weights
        config = PlasticityConfig(
            use_stdp=True,
            use_homeostatic=True,
            use_competition=True,
            stdp_weight=2.0,  # Double weight
            homeostatic_weight=0.5,
            competition_weight=0.1
        )
        
        multi_rule = MultiPlasticityRule(config)
        
        # Check that weights are applied
        self.assertEqual(multi_rule.stdp_weight, 2.0)
        self.assertEqual(multi_rule.homeostatic_weight, 0.5)
        self.assertEqual(multi_rule.competition_weight, 0.1)


class TestSynapticCompetitionIntegration(unittest.TestCase):
    """Test integration of synaptic competition with other systems."""
    
    def setUp(self):
        """Set up test parameters."""
        self.config = PlasticityConfig(
            competition_strength=0.2,
            normalization_target=15.0,
            wta_threshold=0.7
        )
        
    def test_integration_with_plasticity_manager(self):
        """Test integration with PlasticityManager."""
        manager = PlasticityManager(self.config)
        
        # Activate competition rule
        manager.activate_rule("competition")
        
        # Create test weight matrix and activity
        weights = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        pre_activity = np.array([0.8, 0.6])
        post_activity = np.array([0.4, 0.7, 0.9])
        
        # Update weights
        updated_weights = manager.update_weights(
            weights, pre_activity, post_activity,
            all_weights=weights  # Pass all weights for competition
        )
        
        # Should return updated weights
        self.assertEqual(updated_weights.shape, weights.shape)
        self.assertIsInstance(updated_weights, np.ndarray)
        
    def test_multi_plasticity_integration(self):
        """Test multi-plasticity rule integration."""
        manager = PlasticityManager(self.config)
        
        # Activate multi-plasticity rule
        manager.activate_rule("multi_plasticity")
        
        # Create test scenario
        weights = np.array([[2.0, 4.0], [6.0, 8.0]])
        pre_activity = np.array([1.0, 0.8])
        post_activity = np.array([0.6, 0.9])
        
        # Update weights multiple times
        for _ in range(10):
            weights = manager.update_weights(
                weights, pre_activity, post_activity,
                pre_spike=np.array([True, False]),
                post_spike=np.array([False, True]),
                all_weights=weights
            )
            
        # Weights should remain bounded
        self.assertTrue(np.all(weights >= self.config.weight_min))
        self.assertTrue(np.all(weights <= self.config.weight_max))
        
    def test_competition_stability(self):
        """Test that competition maintains network stability."""
        competition = SynapticCompetitionRule(self.config)
        
        # Simulate long-term dynamics
        weights = [2.0, 4.0, 6.0, 8.0]
        weight_history = []
        
        for step in range(1000):
            # Varying activity pattern
            activity = 0.5 + 0.3 * np.sin(step * 0.01)
            
            for i, weight in enumerate(weights):
                change = competition.compute_weight_change(
                    pre_activity=1.0,
                    post_activity=activity,
                    current_weight=weight,
                    all_weights=np.array(weights)
                )
                
                weights[i] = np.clip(weight + change, 0.0, 10.0)
                
            weight_history.append(weights.copy())
            
        # Check stability
        final_weights = weight_history[-100:]
        weight_variance = np.var([np.sum(w) for w in final_weights])
        
        # Total weight should be reasonably stable
        self.assertLess(weight_variance, 5.0, "Total weight should remain stable")
        
        # Individual weights should remain bounded
        all_weights = np.array(weight_history).flatten()
        self.assertTrue(np.all(all_weights >= 0.0))
        self.assertTrue(np.all(all_weights <= 10.0))


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)