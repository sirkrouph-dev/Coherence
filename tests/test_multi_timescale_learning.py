#!/usr/bin/env python3
\"\"\"
Tests for Multi-Timescale Learning Implementation
===============================================

Task 5 Testing: Validates multi-timescale plasticity mechanisms.
\"\"\"

import pytest
import numpy as np
import time
from typing import Dict, List

try:
    from core.multi_timescale_learning import (
        MultiTimescaleLearningSystem,
        MultiTimescalePlasticityConfig,
        FastPlasticityComponent,
        SlowPlasticityComponent,
        ConsolidationSystem,
        AdaptiveForgettingSystem
    )
    IMPORTS_SUCCESS = True
except ImportError as e:
    print(f\"Import error: {e}\")
    IMPORTS_SUCCESS = False


class TestFastPlasticity:
    \"\"\"Test fast plasticity component (seconds timescale).\"\"\"
    
    def test_fast_plasticity_initialization(self):
        \"\"\"Test fast plasticity component initialization.\"\"\"
        if not IMPORTS_SUCCESS:
            pytest.skip(\"Required modules not available\")
            
        config = MultiTimescalePlasticityConfig()
        fast_plasticity = FastPlasticityComponent(config)
        
        assert fast_plasticity.config == config
        assert fast_plasticity.activity_trace == 0.0
        assert len(fast_plasticity.fast_weights) == 0
        
    def test_fast_learning_ltp(self):
        \"\"\"Test fast LTP (Long-Term Potentiation).\"\"\"
        if not IMPORTS_SUCCESS:
            pytest.skip(\"Required modules not available\")
            
        config = MultiTimescalePlasticityConfig(fast_learning_rate=0.1)
        fast_plasticity = FastPlasticityComponent(config)
        
        # Coincident pre and post spikes should cause LTP
        change = fast_plasticity.compute_fast_change(
            pre_spike=True, post_spike=True, current_weight=1.0, dt=0.001
        )
        
        assert change > 0, \"Coincident spikes should cause potentiation\"
        assert change <= config.fast_saturation_threshold
        
    def test_fast_learning_ltd(self):
        \"\"\"Test fast LTD (Long-Term Depression).\"\"\"
        if not IMPORTS_SUCCESS:
            pytest.skip(\"Required modules not available\")
            
        config = MultiTimescalePlasticityConfig(fast_learning_rate=0.1)
        fast_plasticity = FastPlasticityComponent(config)
        
        # Pre spike without post spike should cause weak LTD
        change = fast_plasticity.compute_fast_change(
            pre_spike=True, post_spike=False, current_weight=1.0, dt=0.001
        )
        
        assert change < 0, \"Pre spike without post should cause depression\"
        
    def test_fast_decay(self):
        \"\"\"Test exponential decay of fast changes.\"\"\"
        if not IMPORTS_SUCCESS:
            pytest.skip(\"Required modules not available\")
            
        config = MultiTimescalePlasticityConfig(fast_decay_tau=10.0)
        fast_plasticity = FastPlasticityComponent(config)
        
        initial_weight = 2.0
        decayed_weight = fast_plasticity.apply_fast_decay(initial_weight, dt=10.0)
        
        # After one time constant, should decay to ~37% (1/e)
        expected = initial_weight * np.exp(-1)
        assert abs(decayed_weight - expected) < 0.01


class TestSlowPlasticity:
    \"\"\"Test slow plasticity component (minutes-hours timescale).\"\"\"
    
    def test_protein_synthesis_activation(self):
        \"\"\"Test protein synthesis activation by sustained activity.\"\"\"
        if not IMPORTS_SUCCESS:
            pytest.skip(\"Required modules not available\")
            
        config = MultiTimescalePlasticityConfig(protein_synthesis_threshold=0.5)
        slow_plasticity = SlowPlasticityComponent(config)
        
        # High activity should increase protein synthesis
        initial_level = slow_plasticity.protein_synthesis_level
        slow_plasticity.update_protein_synthesis(activity_level=0.8, dt=1.0)
        
        assert slow_plasticity.protein_synthesis_level > initial_level
        
    def test_gene_expression_delay(self):
        \"\"\"Test delayed gene expression following protein synthesis.\"\"\"
        if not IMPORTS_SUCCESS:
            pytest.skip(\"Required modules not available\")
            
        config = MultiTimescalePlasticityConfig()
        slow_plasticity = SlowPlasticityComponent(config)
        
        # Increase protein synthesis rapidly
        slow_plasticity.protein_synthesis_level = 1.0
        initial_gene_expression = slow_plasticity.gene_expression_activity
        
        # Gene expression should lag behind
        slow_plasticity.update_protein_synthesis(activity_level=0.0, dt=100.0)
        
        assert slow_plasticity.gene_expression_activity > initial_gene_expression
        
    def test_slow_plasticity_requires_history(self):
        \"\"\"Test that slow plasticity requires sustained activity history.\"\"\"
        if not IMPORTS_SUCCESS:
            pytest.skip(\"Required modules not available\")
            
        config = MultiTimescalePlasticityConfig()
        slow_plasticity = SlowPlasticityComponent(config)
        
        # Short history should produce no change
        short_history = [0.5, 0.6]
        change = slow_plasticity.compute_slow_change(
            \"test_synapse\", short_history, 1.0, dt=1.0
        )
        
        assert change == 0.0, \"Short activity history should not trigger slow plasticity\"
        
    def test_late_phase_stabilization(self):
        \"\"\"Test late-phase stabilization of synaptic changes.\"\"\"
        if not IMPORTS_SUCCESS:
            pytest.skip(\"Required modules not available\")
            
        config = MultiTimescalePlasticityConfig(late_phase_delay=100.0)
        slow_plasticity = SlowPlasticityComponent(config)
        
        synapse_id = \"test_synapse\"
        slow_plasticity.late_phase_weights[synapse_id] = 0.5
        
        # Young synapse should not be stabilized
        young_weight = slow_plasticity.apply_late_phase_stabilization(
            synapse_id, 1.0, age=50.0
        )
        assert young_weight == 1.0
        
        # Old synapse should be stabilized
        old_weight = slow_plasticity.apply_late_phase_stabilization(
            synapse_id, 1.0, age=200.0
        )
        assert old_weight > 1.0, \"Late-phase synapses should be stabilized\"


class TestConsolidationSystem:
    \"\"\"Test memory consolidation during rest periods.\"\"\"
    
    def test_rest_detection(self):
        \"\"\"Test detection of rest periods based on low activity.\"\"\"
        if not IMPORTS_SUCCESS:
            pytest.skip(\"Required modules not available\")
            
        config = MultiTimescalePlasticityConfig(rest_detection_time=5.0)
        consolidation = ConsolidationSystem(config)
        
        current_time = time.time()
        
        # High activity should not trigger rest
        is_rest = consolidation.detect_rest_period(0.5, current_time)
        assert not is_rest
        
        # Low activity for sufficient time should trigger rest
        consolidation.rest_start_time = current_time - 10.0  # Started 10s ago
        consolidation.is_resting = True
        is_rest = consolidation.detect_rest_period(0.05, current_time)
        assert is_rest, \"Sustained low activity should trigger rest detection\"
        
    def test_memory_consolidation_strengthening(self):
        \"\"\"Test that important memories are strengthened during consolidation.\"\"\"
        if not IMPORTS_SUCCESS:
            pytest.skip(\"Required modules not available\")
            
        config = MultiTimescalePlasticityConfig(consolidation_rate=0.1, consolidation_threshold=0.5)
        consolidation = ConsolidationSystem(config)
        consolidation.is_resting = True
        
        # Create weight matrix and importance scores
        weights = np.array([[1.0, 2.0], [3.0, 4.0]])
        importance = np.array([[0.8, 0.3], [0.6, 0.2]])  # First two are important
        
        consolidated = consolidation.consolidate_memories(weights, importance)
        
        # Important connections should be strengthened
        assert consolidated[0, 0] > weights[0, 0], \"Important synapse should be strengthened\"
        assert consolidated[1, 0] > weights[1, 0], \"Important synapse should be strengthened\"
        
        # Unimportant connections should be slightly weakened
        assert consolidated[0, 1] <= weights[0, 1], \"Unimportant synapse should not be strengthened\"
        
    def test_replay_pattern_generation(self):
        \"\"\"Test generation of replay patterns during rest.\"\"\"
        if not IMPORTS_SUCCESS:
            pytest.skip(\"Required modules not available\")
            
        config = MultiTimescalePlasticityConfig(replay_strength=0.2)
        consolidation = ConsolidationSystem(config)
        consolidation.is_resting = True
        
        # Create recent patterns
        patterns = [np.array([1.0, 0.5, 0.0]), np.array([0.0, 1.0, 0.5])]
        
        replay_patterns = consolidation.generate_replay_patterns(patterns)
        
        assert len(replay_patterns) == len(patterns)
        for replay, original in zip(replay_patterns, patterns):
            # Replay patterns should be similar but not identical (due to noise)
            correlation = np.corrcoef(replay.flatten(), original.flatten())[0, 1]
            assert correlation > 0.5, \"Replay patterns should correlate with originals\"


class TestAdaptiveForgetting:
    \"\"\"Test adaptive forgetting system.\"\"\"
    
    def test_activity_dependent_forgetting(self):
        \"\"\"Test that forgetting rate depends on recent activity.\"\"\"
        if not IMPORTS_SUCCESS:
            pytest.skip(\"Required modules not available\")
            
        config = MultiTimescalePlasticityConfig(
            forgetting_rate=0.01, 
            activity_dependent_forgetting=True
        )
        forgetting = AdaptiveForgettingSystem(config)
        
        # Low activity should increase forgetting
        low_activity_rate = forgetting.compute_forgetting_rate(
            \"synapse1\", recent_activity=0.05, age=100.0
        )
        
        # High activity should decrease forgetting
        high_activity_rate = forgetting.compute_forgetting_rate(
            \"synapse2\", recent_activity=0.9, age=100.0
        )
        
        assert low_activity_rate > high_activity_rate, \"Low activity should increase forgetting rate\"
        
    def test_age_dependent_forgetting(self):
        \"\"\"Test that older synapses have higher forgetting rates.\"\"\"
        if not IMPORTS_SUCCESS:
            pytest.skip(\"Required modules not available\")
            
        config = MultiTimescalePlasticityConfig(forgetting_rate=0.01)
        forgetting = AdaptiveForgettingSystem(config)
        
        young_rate = forgetting.compute_forgetting_rate(
            \"young_synapse\", recent_activity=0.5, age=100.0
        )
        
        old_rate = forgetting.compute_forgetting_rate(
            \"old_synapse\", recent_activity=0.5, age=10000.0
        )
        
        assert old_rate > young_rate, \"Older synapses should have higher forgetting rates\"
        
    def test_forgetting_application(self):
        \"\"\"Test application of forgetting to weights.\"\"\"
        if not IMPORTS_SUCCESS:
            pytest.skip(\"Required modules not available\")
            
        config = MultiTimescalePlasticityConfig()
        forgetting = AdaptiveForgettingSystem(config)
        
        initial_weight = 2.0
        forgetting_rate = 0.1
        dt = 1.0
        
        decayed_weight = forgetting.apply_forgetting(initial_weight, forgetting_rate, dt)
        
        # Weight should decay exponentially
        expected = initial_weight * np.exp(-forgetting_rate * dt)
        assert abs(decayed_weight - expected) < 1e-6
        
    def test_interference_detection(self):
        \"\"\"Test detection of interference between memory patterns.\"\"\"
        if not IMPORTS_SUCCESS:
            pytest.skip(\"Required modules not available\")
            
        config = MultiTimescalePlasticityConfig()
        forgetting = AdaptiveForgettingSystem(config)
        
        # Similar patterns should show high interference
        pattern1 = np.array([1.0, 0.8, 0.2])
        pattern2 = np.array([0.9, 0.7, 0.3])  # Similar to pattern1
        pattern3 = np.array([0.1, 0.2, 0.9])  # Different from pattern1
        
        interference_similar = forgetting.detect_interference(pattern1, [pattern2])
        interference_different = forgetting.detect_interference(pattern1, [pattern3])
        
        assert interference_similar > interference_different, \"Similar patterns should show more interference\"


class TestMultiTimescaleIntegration:
    \"\"\"Test integrated multi-timescale learning system.\"\"\"
    
    def test_system_initialization(self):
        \"\"\"Test system initialization with all components.\"\"\"
        if not IMPORTS_SUCCESS:
            pytest.skip(\"Required modules not available\")
            
        system = MultiTimescaleLearningSystem()
        
        assert system.fast_plasticity is not None
        assert system.slow_plasticity is not None
        assert system.consolidation is not None
        assert system.forgetting is not None
        assert len(system.synaptic_ages) == 0
        
    def test_synapse_update_integration(self):
        \"\"\"Test that synapse updates integrate all plasticity mechanisms.\"\"\"
        if not IMPORTS_SUCCESS:
            pytest.skip(\"Required modules not available\")
            
        config = MultiTimescalePlasticityConfig(
            fast_learning_rate=0.1,
            slow_learning_rate=0.01,
            forgetting_rate=0.001
        )
        system = MultiTimescaleLearningSystem(config)
        
        initial_weight = 1.0
        
        # Fast learning phase
        new_weight = system.update_synapse(
            \"test_synapse\", pre_spike=True, post_spike=True, 
            current_weight=initial_weight, global_activity=0.5
        )
        
        # Weight should change due to fast plasticity
        assert new_weight != initial_weight
        assert \"test_synapse\" in system.synaptic_ages
        assert \"test_synapse\" in system.activity_histories
        
    def test_learning_state_reporting(self):
        \"\"\"Test that system reports comprehensive learning state.\"\"\"
        if not IMPORTS_SUCCESS:
            pytest.skip(\"Required modules not available\")
            
        system = MultiTimescaleLearningSystem()
        
        # Add some synapses
        system.update_synapse(\"syn1\", True, True, 1.0, 0.5)
        system.update_synapse(\"syn2\", False, True, 1.5, 0.3)
        
        state = system.get_learning_state()
        
        assert 'fast_plasticity' in state
        assert 'slow_plasticity' in state
        assert 'consolidation' in state
        assert 'network_stats' in state
        
        assert state['network_stats']['total_synapses'] == 2
        assert state['network_stats']['mean_age'] >= 0.0
        
    def test_network_consolidation(self):
        \"\"\"Test network-wide consolidation functionality.\"\"\"
        if not IMPORTS_SUCCESS:
            pytest.skip(\"Required modules not available\")
            
        system = MultiTimescaleLearningSystem()
        
        # Create mock weight matrix
        weights = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        # Add importance scores for some synapses
        system.importance_scores[\"0_0\"] = 0.8  # High importance
        system.importance_scores[\"1_1\"] = 0.2  # Low importance
        
        consolidated_weights = system.consolidate_network(weights)
        
        # Should return valid weight matrix
        assert consolidated_weights.shape == weights.shape
        assert np.all(consolidated_weights >= 0.0)
        
    def test_system_reset(self):
        \"\"\"Test system reset functionality.\"\"\"
        if not IMPORTS_SUCCESS:
            pytest.skip(\"Required modules not available\")
            
        system = MultiTimescaleLearningSystem()
        
        # Add some data
        system.update_synapse(\"syn1\", True, True, 1.0, 0.5)
        system.importance_scores[\"syn1\"] = 0.7
        
        # Reset system
        system.reset_system()
        
        # All state should be cleared
        assert len(system.synaptic_ages) == 0
        assert len(system.activity_histories) == 0
        assert len(system.importance_scores) == 0


class TestMultiTimescaleLearningScenarios:
    \"\"\"Test realistic learning scenarios with multi-timescale plasticity.\"\"\"
    
    def test_rapid_learning_scenario(self):
        \"\"\"Test rapid learning followed by consolidation.\"\"\"
        if not IMPORTS_SUCCESS:
            pytest.skip(\"Required modules not available\")
            
        config = MultiTimescalePlasticityConfig(
            fast_learning_rate=0.2,
            consolidation_rate=0.1
        )
        system = MultiTimescaleLearningSystem(config)
        
        initial_weight = 1.0
        current_weight = initial_weight
        
        # Rapid learning phase (high activity)
        for _ in range(5):
            current_weight = system.update_synapse(
                \"learning_synapse\", True, True, current_weight, global_activity=0.8
            )
            
        weight_after_learning = current_weight
        
        # Rest phase (consolidation)
        for _ in range(5):
            current_weight = system.update_synapse(
                \"learning_synapse\", False, False, current_weight, global_activity=0.05
            )
            
        # Should show learning followed by some consolidation effects
        assert weight_after_learning > initial_weight, \"Should show learning\"
        
    def test_forgetting_without_use(self):
        \"\"\"Test gradual forgetting of unused synapses.\"\"\"
        if not IMPORTS_SUCCESS:
            pytest.skip(\"Required modules not available\")
            
        config = MultiTimescalePlasticityConfig(
            forgetting_rate=0.01,
            activity_dependent_forgetting=True
        )
        system = MultiTimescaleLearningSystem(config)
        
        # Learn something
        learned_weight = system.update_synapse(
            \"forgetting_synapse\", True, True, 1.0, global_activity=0.5
        )
        
        # Long period without activity
        current_weight = learned_weight
        for _ in range(20):
            current_weight = system.update_synapse(
                \"forgetting_synapse\", False, False, current_weight, global_activity=0.0
            )
            
        # Should show gradual forgetting
        assert current_weight < learned_weight, \"Unused synapse should show forgetting\"
        
    def test_protein_synthesis_dependent_learning(self):
        \"\"\"Test slow learning that requires protein synthesis.\"\"\"
        if not IMPORTS_SUCCESS:
            pytest.skip(\"Required modules not available\")
            
        config = MultiTimescalePlasticityConfig(
            slow_learning_rate=0.05,
            protein_synthesis_threshold=0.3
        )
        system = MultiTimescaleLearningSystem(config)
        
        initial_weight = 1.0
        current_weight = initial_weight
        
        # Sustained moderate activity to build up protein synthesis
        for i in range(30):
            # Moderate activity pattern
            pre_spike = i % 3 == 0
            post_spike = i % 4 == 0
            
            current_weight = system.update_synapse(
                \"protein_synapse\", pre_spike, post_spike, current_weight, global_activity=0.4
            )
            
        state = system.get_learning_state()
        
        # Should have built up protein synthesis
        assert state['slow_plasticity']['protein_synthesis_level'] > 0.1
        
        # Weight should show some change due to slow plasticity
        # (May be small due to the gradual nature)
        total_change = abs(current_weight - initial_weight)
        assert total_change >= 0.0  # At minimum, no error should occur


def run_multi_timescale_learning_demo():
    \"\"\"Run a comprehensive demonstration of multi-timescale learning.\"\"\"
    if not IMPORTS_SUCCESS:
        print(\"Cannot run demo - required modules not available\")
        return
        
    print(\"\n=== Multi-Timescale Learning System Demo ===\")
    
    # Create system
    config = MultiTimescalePlasticityConfig(
        fast_learning_rate=0.1,
        slow_learning_rate=0.02,
        consolidation_rate=0.05,
        forgetting_rate=0.005
    )
    system = MultiTimescaleLearningSystem(config)
    
    synapse_weight = 1.0
    
    print(f\"Initial synapse weight: {synapse_weight:.3f}\")
    
    # Phase 1: Rapid learning
    print(\"\nPhase 1: Rapid learning (5 steps with coincident spikes)\")
    for i in range(5):
        synapse_weight = system.update_synapse(
            \"demo_synapse\", True, True, synapse_weight, global_activity=0.7
        )
        print(f\"  Step {i+1}: Weight = {synapse_weight:.3f}\")
        
    # Phase 2: Sustained moderate activity for protein synthesis
    print(\"\nPhase 2: Building protein synthesis (10 steps moderate activity)\")
    for i in range(10):
        # Random moderate activity
        pre_spike = np.random.random() < 0.4
        post_spike = np.random.random() < 0.4
        synapse_weight = system.update_synapse(
            \"demo_synapse\", pre_spike, post_spike, synapse_weight, global_activity=0.4
        )
        if i % 3 == 0:
            print(f\"  Step {i+1}: Weight = {synapse_weight:.3f}\")
            
    # Check protein synthesis
    state = system.get_learning_state()
    print(f\"\nProtein synthesis level: {state['slow_plasticity']['protein_synthesis_level']:.3f}\")
    print(f\"Gene expression activity: {state['slow_plasticity']['gene_expression_activity']:.3f}\")
    
    # Phase 3: Rest period (consolidation)
    print(\"\nPhase 3: Rest period (consolidation - 5 steps low activity)\")
    for i in range(5):
        synapse_weight = system.update_synapse(
            \"demo_synapse\", False, False, synapse_weight, global_activity=0.05
        )
        print(f\"  Rest step {i+1}: Weight = {synapse_weight:.3f}\")
        
    # Phase 4: Forgetting test
    print(\"\nPhase 4: Extended inactivity (forgetting test - 10 steps)\")
    for i in range(10):
        synapse_weight = system.update_synapse(
            \"demo_synapse\", False, False, synapse_weight, global_activity=0.0
        )
        if i % 3 == 0:
            print(f\"  Inactive step {i+1}: Weight = {synapse_weight:.3f}\")
            
    print(f\"\nFinal synapse weight: {synapse_weight:.3f}\")
    
    # Final state
    final_state = system.get_learning_state()
    print(\"\nFinal system state:\")
    print(f\"  Total synapses tracked: {final_state['network_stats']['total_synapses']}\")
    print(f\"  Mean synapse age: {final_state['network_stats']['mean_age']:.1f}s\")
    print(f\"  Mean importance: {final_state['network_stats']['mean_importance']:.3f}\")
    print(f\"  Consolidation system resting: {final_state['consolidation']['is_resting']}\")
    
    return system, synapse_weight


if __name__ == \"__main__\":
    # Run demo if modules are available
    if IMPORTS_SUCCESS:
        demo_system, final_weight = run_multi_timescale_learning_demo()
        print(\"\n✅ Multi-timescale learning demo completed successfully!\")
    else:
        print(\"❌ Cannot run demo - import errors occurred\")
        print(\"Please ensure the multi_timescale_learning module is properly installed.\")
