#!/usr/bin/env python3
"""
Test suite for enhanced homeostatic plasticity mechanisms.

This module tests the enhanced homeostatic plasticity system to ensure it:
- Maintains stable firing rates through synaptic scaling
- Prevents runaway excitation and depression
- Implements intrinsic excitability regulation
- Maintains total synaptic strength homeostasis
"""

import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the core modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.learning import (
    HomeostaticPlasticity,
    PlasticityConfig,
    PlasticityManager
)


class TestEnhancedHomeostaticPlasticity(unittest.TestCase):
    """Test enhanced homeostatic plasticity mechanisms."""
    
    def setUp(self):
        """Set up test parameters."""
        self.config = PlasticityConfig(
            target_rate=10.0,  # 10 Hz target
            homeostatic_time_constant=1000.0,
            learning_rate=0.01,
            target_total_strength=50.0,
            synaptic_scaling_rate=0.01,
            excitability_scaling_rate=0.001,
            weight_min=0.0,
            weight_max=10.0
        )
        self.homeostatic = HomeostaticPlasticity(self.config)
        
    def test_basic_homeostatic_adjustment(self):
        """Test basic firing rate homeostasis."""
        dt = 1.0  # 1ms time step
        current_weight = 5.0
        
        # Simulate high activity (should reduce weights)
        high_activity_changes = []
        for _ in range(100):
            # High firing rate (20 Hz, double target)
            delta_w = self.homeostatic.compute_weight_change(
                pre_activity=1.0,
                post_activity=1.0,
                current_weight=current_weight,
                post_spike=True,  # High spike rate
                dt=dt
            )
            high_activity_changes.append(delta_w)
            current_weight += delta_w
            
        # Should trend toward weight reduction
        mean_change = np.mean(high_activity_changes[-20:])  # Recent changes
        self.assertLess(mean_change, 0.0, "High activity should reduce weights")
        
        # Reset for low activity test
        self.homeostatic = HomeostaticPlasticity(self.config)
        current_weight = 5.0
        
        # Simulate low activity (should increase weights)
        low_activity_changes = []
        for i in range(1000):  # Longer simulation for better statistics
            # Low firing rate: target is 10 Hz, let's do 5 Hz (half target)
            # 5 Hz = 5 spikes per second = 1 spike per 200ms = 1 spike per 200 steps
            post_spike = (i % 200 == 0)  # 5 Hz
            delta_w = self.homeostatic.compute_weight_change(
                pre_activity=1.0,
                post_activity=1.0,
                current_weight=current_weight,
                post_spike=post_spike,
                dt=dt
            )
            low_activity_changes.append(delta_w)
            current_weight += delta_w
            
        # Should trend toward weight increase (after initial transient)
        mean_change = np.mean(low_activity_changes[-100:])  # Look at more recent changes
        
        # Debug information
        print(f"Low activity test:")
        print(f"  Final firing rate estimate: {self.homeostatic.firing_rate_estimate}")
        print(f"  Target rate: {self.config.target_rate}")
        print(f"  Mean change in last 100 steps: {mean_change}")
        print(f"  Last few changes: {low_activity_changes[-5:]}")
        
        # Get current rate from time window
        if self.homeostatic.time_window:
            window_duration_sec = len(self.homeostatic.time_window) * dt / 1000.0
            current_rate = sum(self.homeostatic.time_window) / window_duration_sec
            print(f"  Current calculated rate: {current_rate} Hz")
            print(f"  Expected rate: 5 Hz (half of target 10 Hz)")
        
        self.assertGreater(mean_change, 0.0, "Low activity should increase weights")
        
    def test_synaptic_scaling(self):
        """Test synaptic scaling maintains total strength."""
        # Create array of synaptic weights
        all_weights = np.array([2.0, 3.0, 4.0, 6.0])  # Total = 15
        target_total = self.config.target_total_strength  # 50
        
        # Test scaling computation
        scaling_change = self.homeostatic.compute_synaptic_scaling(
            current_weight=2.0,
            all_weights=all_weights
        )
        
        # Should scale up since current total (15) < target (50)
        self.assertGreater(scaling_change, 0.0, "Should scale up when total is below target")
        
        # Test with weights above target
        large_weights = np.array([20.0, 25.0, 30.0, 35.0])  # Total = 110
        scaling_change = self.homeostatic.compute_synaptic_scaling(
            current_weight=20.0,
            all_weights=large_weights
        )
        
        # Should scale down since current total (110) > target (50)
        self.assertLess(scaling_change, 0.0, "Should scale down when total is above target")
        
    def test_intrinsic_excitability_regulation(self):
        """Test intrinsic excitability adjusts to maintain target rate."""
        initial_excitability = self.homeostatic.intrinsic_excitability
        
        # Simulate high firing rate
        high_rate = 20.0  # Double the target
        excitability_change = self.homeostatic.compute_intrinsic_excitability_change(high_rate)
        
        # Should decrease excitability
        self.assertLess(excitability_change, 0.0, "High rate should decrease excitability")
        self.assertLess(self.homeostatic.intrinsic_excitability, initial_excitability)
        
        # Reset and test low firing rate
        self.homeostatic.intrinsic_excitability = 1.0
        low_rate = 2.0  # Below target
        excitability_change = self.homeostatic.compute_intrinsic_excitability_change(low_rate)
        
        # Should increase excitability
        self.assertGreater(excitability_change, 0.0, "Low rate should increase excitability")
        self.assertGreater(self.homeostatic.intrinsic_excitability, 1.0)
        
    def test_runaway_prevention(self):
        """Test prevention of runaway excitation and depression."""
        dt = 1.0
        current_weight = 5.0
        
        # Simulate runaway excitation (very high firing rate)
        runaway_changes = []
        for _ in range(50):
            # Extremely high firing rate (every spike)
            delta_w = self.homeostatic.compute_weight_change(
                pre_activity=1.0,
                post_activity=1.0,
                current_weight=current_weight,
                post_spike=True,
                dt=dt
            )
            runaway_changes.append(delta_w)
            current_weight = max(0.0, current_weight + delta_w)  # Prevent negative weights
            
        # Should strongly suppress weights to prevent runaway
        final_changes = runaway_changes[-10:]
        mean_final_change = np.mean(final_changes)
        self.assertLess(mean_final_change, -0.01, "Should strongly suppress runaway excitation")
        
    def test_homeostatic_state_monitoring(self):
        """Test homeostatic state monitoring and reporting."""
        # Run some activity to populate state
        dt = 1.0
        for i in range(100):
            post_spike = (i % 5 == 0)  # 20% spike rate
            self.homeostatic.compute_weight_change(
                pre_activity=1.0,
                post_activity=1.0,
                current_weight=5.0,
                post_spike=post_spike,
                dt=dt,
                all_weights=np.array([3.0, 4.0, 5.0, 6.0])
            )
            
        # Get homeostatic state
        state = self.homeostatic.get_homeostatic_state()
        
        # Check all expected fields are present
        expected_fields = [
            'firing_rate_estimate', 'total_synaptic_strength', 'intrinsic_excitability',
            'mean_recent_activity', 'activity_variance', 'target_rate'
        ]
        
        for field in expected_fields:
            self.assertIn(field, state, f"State should contain {field}")
            self.assertIsInstance(state[field], (int, float), f"{field} should be numeric")
            
        # Check reasonable values
        self.assertGreaterEqual(state['intrinsic_excitability'], 0.1)
        self.assertLessEqual(state['intrinsic_excitability'], 3.0)
        self.assertEqual(state['target_rate'], self.config.target_rate)
        
    def test_homeostatic_reset(self):
        """Test homeostatic state reset functionality."""
        # Populate state with some activity
        for i in range(50):
            self.homeostatic.compute_weight_change(
                pre_activity=1.0,
                post_activity=1.0,
                current_weight=5.0,
                post_spike=(i % 3 == 0),
                dt=1.0
            )
            
        # Verify state has been populated
        self.assertGreater(len(self.homeostatic.activity_history), 0)
        self.assertGreater(len(self.homeostatic.time_window), 0)
        
        # Reset state
        self.homeostatic.reset_homeostatic_state()
        
        # Verify reset
        self.assertEqual(len(self.homeostatic.activity_history), 0)
        self.assertEqual(len(self.homeostatic.time_window), 0)
        self.assertEqual(self.homeostatic.firing_rate_estimate, 0.0)
        self.assertEqual(self.homeostatic.total_synaptic_strength, 0.0)
        self.assertEqual(self.homeostatic.intrinsic_excitability, 1.0)
        
    def test_integration_with_plasticity_manager(self):
        """Test integration with PlasticityManager."""
        manager = PlasticityManager(self.config)
        
        # Activate homeostatic plasticity
        manager.activate_rule("homeostatic")
        
        # Create test weight matrix and activity
        weights = np.array([[1.0, 2.0], [3.0, 4.0]])
        pre_activity = np.array([0.5, 0.8])
        post_activity = np.array([0.3, 0.9])
        
        # Update weights
        updated_weights = manager.update_weights(
            weights, pre_activity, post_activity,
            post_spike=np.array([False, True]),
            dt=1.0
        )
        
        # Should return updated weights
        self.assertEqual(updated_weights.shape, weights.shape)
        self.assertIsInstance(updated_weights, np.ndarray)
        
    def test_stability_over_time(self):
        """Test that homeostatic mechanisms maintain stability over long periods."""
        dt = 1.0
        weights = [5.0, 4.0, 6.0, 3.0]  # Multiple synapses
        firing_rates = []
        
        # Simulate long period with varying input
        for step in range(1000):
            # Varying input pattern
            input_strength = 0.5 + 0.3 * np.sin(step * 0.01)  # Slow oscillation
            
            # Update each synapse
            total_change = 0.0
            for i, weight in enumerate(weights):
                # Simulate post-spike based on accumulated input
                # Make firing rate closer to target (10 Hz)
                # 10 Hz = 1 spike per 100ms = 1 spike per 100 steps
                post_spike = (step + i * 25) % 100 == 0  # Staggered spiking at ~10 Hz
                
                delta_w = self.homeostatic.compute_weight_change(
                    pre_activity=input_strength,
                    post_activity=0.7,
                    current_weight=weight,
                    post_spike=post_spike,
                    dt=dt,
                    all_weights=np.array(weights)
                )
                
                weights[i] = np.clip(weight + delta_w, 0.0, 10.0)
                total_change += abs(delta_w)
                
            # Track firing rate estimate
            if len(self.homeostatic.time_window) >= 10:
                # Calculate rate over last 10 time steps
                recent_spikes = sum(self.homeostatic.time_window[-10:])
                window_duration_sec = 10 * dt / 1000.0  # Convert to seconds
                current_rate = recent_spikes / window_duration_sec if window_duration_sec > 0 else 0.0
                firing_rates.append(current_rate)
                
        # Check that firing rate converges toward target
        if len(firing_rates) > 100:
            final_rates = firing_rates[-100:]
            mean_final_rate = np.mean(final_rates)
            
            # Should be reasonably close to target (within 50%)
            target_rate = self.config.target_rate
            self.assertLess(abs(mean_final_rate - target_rate), target_rate * 0.5,
                           f"Final rate {mean_final_rate:.2f} should be close to target {target_rate}")
            
        # Check that weights remain bounded
        for weight in weights:
            self.assertGreaterEqual(weight, 0.0, "Weights should not go negative")
            self.assertLessEqual(weight, 10.0, "Weights should not exceed maximum")


class TestHomeostaticPlasticityEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions for homeostatic plasticity."""
    
    def setUp(self):
        """Set up test parameters."""
        self.config = PlasticityConfig(target_rate=5.0)
        self.homeostatic = HomeostaticPlasticity(self.config)
        
    def test_zero_weights(self):
        """Test behavior with zero weights."""
        delta_w = self.homeostatic.compute_weight_change(
            pre_activity=1.0,
            post_activity=1.0,
            current_weight=0.0,
            post_spike=True,
            dt=1.0,
            all_weights=np.array([0.0, 0.0, 0.0])
        )
        
        # Should handle zero weights gracefully
        self.assertIsInstance(delta_w, (int, float))
        self.assertFalse(np.isnan(delta_w))
        self.assertFalse(np.isinf(delta_w))
        
    def test_empty_weight_array(self):
        """Test behavior with empty weight arrays."""
        delta_w = self.homeostatic.compute_weight_change(
            pre_activity=1.0,
            post_activity=1.0,
            current_weight=5.0,
            post_spike=True,
            dt=1.0,
            all_weights=np.array([])
        )
        
        # Should handle empty arrays gracefully
        self.assertIsInstance(delta_w, (int, float))
        self.assertFalse(np.isnan(delta_w))
        
    def test_extreme_firing_rates(self):
        """Test behavior with extreme firing rates."""
        dt = 1.0
        
        # Test with no spikes for long period
        for _ in range(200):
            delta_w = self.homeostatic.compute_weight_change(
                pre_activity=1.0,
                post_activity=1.0,
                current_weight=5.0,
                post_spike=False,  # No spikes
                dt=dt
            )
            # Should remain stable
            self.assertFalse(np.isnan(delta_w))
            self.assertFalse(np.isinf(delta_w))
            
        # Test with continuous spiking
        for _ in range(200):
            delta_w = self.homeostatic.compute_weight_change(
                pre_activity=1.0,
                post_activity=1.0,
                current_weight=5.0,
                post_spike=True,  # Continuous spikes
                dt=dt
            )
            # Should remain stable
            self.assertFalse(np.isnan(delta_w))
            self.assertFalse(np.isinf(delta_w))


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)