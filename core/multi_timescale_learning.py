#!/usr/bin/env python3
\"\"\"
Multi-Timescale Learning Implementation
======================================

Task 5: Extends the existing learning system to support multiple timescales:
- Fast plasticity (seconds): Immediate synaptic changes  
- Slow plasticity (minutes-hours): Protein synthesis-dependent changes
- Memory consolidation during rest periods
- Adaptive forgetting through gradual weight decay
\"\"\"

import numpy as np
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum

try:
    from core.learning import PlasticityRule, PlasticityConfig, PlasticityManager
    LEARNING_AVAILABLE = True
except ImportError:
    LEARNING_AVAILABLE = False
    print(\"Warning: Core learning module not available - using standalone implementation\")


class PlasticityTimescale(Enum):
    \"\"\"Timescales for different plasticity mechanisms.\"\"\"
    IMMEDIATE = \"immediate\"  # < 1 second
    FAST = \"fast\"  # 1-60 seconds
    SLOW = \"slow\"  # 1-60 minutes  
    LATE_PHASE = \"late_phase\"  # 1+ hours
    CONSOLIDATION = \"consolidation\"  # During rest/sleep


@dataclass
class MultiTimescalePlasticityConfig:
    \"\"\"Configuration for multi-timescale plasticity.\"\"\"
    
    # Fast plasticity parameters (seconds)
    fast_learning_rate: float = 0.1
    fast_decay_tau: float = 30.0  # 30 second decay
    fast_saturation_threshold: float = 0.8
    
    # Slow plasticity parameters (minutes-hours)
    slow_learning_rate: float = 0.01
    slow_decay_tau: float = 1800.0  # 30 minute decay
    protein_synthesis_threshold: float = 0.5
    late_phase_delay: float = 3600.0  # 1 hour delay
    
    # Consolidation parameters
    consolidation_rate: float = 0.05
    consolidation_threshold: float = 0.3
    rest_detection_time: float = 10.0  # 10s of low activity = rest
    replay_strength: float = 0.2
    
    # Forgetting parameters
    forgetting_rate: float = 0.001
    activity_dependent_forgetting: bool = True
    interference_strength: float = 0.1
    
    # Weight bounds
    weight_min: float = 0.0
    weight_max: float = 10.0


class FastPlasticityComponent:
    \"\"\"Fast plasticity occurring within seconds.\"\"\"
    
    def __init__(self, config: MultiTimescalePlasticityConfig):
        self.config = config
        self.fast_weights = {}  # Temporary weight changes
        self.activity_trace = 0.0
        self.last_update_time = 0.0
        
    def compute_fast_change(self, pre_spike: bool, post_spike: bool, 
                           current_weight: float, dt: float) -> float:
        \"\"\"Compute immediate weight change from fast plasticity.\"\"\"
        
        # Update activity trace
        self.activity_trace *= np.exp(-dt / 10.0)  # 10s trace
        if pre_spike or post_spike:
            self.activity_trace += 1.0
            
        # Fast Hebbian-like learning
        if pre_spike and post_spike:
            # LTP within 20ms window
            fast_change = self.config.fast_learning_rate * (1.0 - current_weight / self.config.weight_max)
        elif pre_spike:
            # Weak LTD
            fast_change = -self.config.fast_learning_rate * 0.1 * current_weight / self.config.weight_max
        else:
            fast_change = 0.0
            
        # Apply saturation
        if abs(fast_change) > self.config.fast_saturation_threshold:
            fast_change = np.sign(fast_change) * self.config.fast_saturation_threshold
            
        return fast_change
        
    def apply_fast_decay(self, weight: float, dt: float) -> float:
        \"\"\"Apply exponential decay to fast weight changes.\"\"\"
        decay_factor = np.exp(-dt / self.config.fast_decay_tau)
        return weight * decay_factor


class SlowPlasticityComponent:
    \"\"\"Slow plasticity dependent on protein synthesis.\"\"\"
    
    def __init__(self, config: MultiTimescalePlasticityConfig):
        self.config = config
        self.protein_synthesis_level = 0.0
        self.late_phase_weights = {}
        self.gene_expression_activity = 0.0
        self.synthesis_history = []
        
    def update_protein_synthesis(self, activity_level: float, dt: float):
        \"\"\"Update protein synthesis based on neural activity.\"\"\"
        
        # Protein synthesis triggered by sustained activity
        if activity_level > self.config.protein_synthesis_threshold:
            self.protein_synthesis_level += 0.1 * dt
        else:
            self.protein_synthesis_level *= np.exp(-dt / 600.0)  # 10min decay
            
        # Gene expression follows protein synthesis with delay
        synthesis_change = (self.protein_synthesis_level - self.gene_expression_activity) / 1800.0
        self.gene_expression_activity += synthesis_change * dt
        
        # Bounds
        self.protein_synthesis_level = np.clip(self.protein_synthesis_level, 0.0, 2.0)
        self.gene_expression_activity = np.clip(self.gene_expression_activity, 0.0, 2.0)
        
    def compute_slow_change(self, synapse_id: str, activity_history: List[float], 
                          current_weight: float, dt: float) -> float:
        \"\"\"Compute slow weight change from protein synthesis.\"\"\"
        
        # Require sustained activity for slow plasticity
        if len(activity_history) < 10:
            return 0.0
            
        mean_activity = np.mean(activity_history[-10:])
        activity_stability = 1.0 / (1.0 + np.var(activity_history[-10:]))
        
        # Slow plasticity depends on protein synthesis
        if self.gene_expression_activity > 0.5 and mean_activity > 0.3:
            # Late-phase LTP
            slow_change = (self.config.slow_learning_rate * 
                         self.gene_expression_activity * 
                         mean_activity * activity_stability)
        else:
            slow_change = 0.0
            
        return slow_change
        
    def apply_late_phase_stabilization(self, synapse_id: str, weight: float, age: float) -> float:
        \"\"\"Apply late-phase stabilization to prevent decay.\"\"\"
        
        if age > self.config.late_phase_delay and synapse_id in self.late_phase_weights:
            # Late-phase weights resist decay
            stabilization_factor = 1.0 + self.late_phase_weights[synapse_id] * 0.5
            return weight * stabilization_factor
        return weight


class ConsolidationSystem:
    \"\"\"Memory consolidation during rest periods.\"\"\"
    
    def __init__(self, config: MultiTimescalePlasticityConfig):
        self.config = config
        self.is_resting = False
        self.rest_start_time = 0.0
        self.important_memories = {}
        self.replay_patterns = []
        self.consolidation_strength = 0.0
        
    def detect_rest_period(self, global_activity: float, current_time: float) -> bool:
        \"\"\"Detect when the network is in a rest state.\"\"\"
        
        if global_activity < 0.1:  # Low activity threshold
            if not self.is_resting:
                self.rest_start_time = current_time
                self.is_resting = True
            elif current_time - self.rest_start_time > self.config.rest_detection_time:
                return True
        else:
            self.is_resting = False
            
        return False
        
    def consolidate_memories(self, weight_matrix: np.ndarray, 
                           importance_scores: np.ndarray) -> np.ndarray:
        \"\"\"Consolidate important memories during rest.\"\"\"
        
        if not self.is_resting:
            return weight_matrix
            
        # Strengthen important connections
        consolidated_weights = weight_matrix.copy()
        
        # Find highly active/important synapses
        important_mask = importance_scores > self.config.consolidation_threshold
        
        # Strengthen important connections
        consolidated_weights[important_mask] *= (1.0 + self.config.consolidation_rate)
        
        # Weaken unimportant connections slightly
        unimportant_mask = importance_scores < self.config.consolidation_threshold * 0.5
        consolidated_weights[unimportant_mask] *= (1.0 - self.config.consolidation_rate * 0.1)
        
        # Apply bounds
        consolidated_weights = np.clip(consolidated_weights, 
                                     self.config.weight_min, 
                                     self.config.weight_max)
        
        return consolidated_weights
        
    def generate_replay_patterns(self, recent_patterns: List[np.ndarray]) -> List[np.ndarray]:
        \"\"\"Generate replay patterns for memory consolidation.\"\"\"
        
        if not recent_patterns or not self.is_resting:
            return []
            
        # Select patterns based on novelty and strength
        replay_list = []
        
        for pattern in recent_patterns[-10:]:  # Recent patterns
            # Add noise for generalization
            noisy_pattern = pattern + np.random.normal(0, 0.1, pattern.shape)
            replay_list.append(noisy_pattern * self.config.replay_strength)
            
        return replay_list


class AdaptiveForgettingSystem:
    \"\"\"Adaptive forgetting to prevent catastrophic interference.\"\"\"
    
    def __init__(self, config: MultiTimescalePlasticityConfig):
        self.config = config
        self.usage_history = {}
        self.interference_detector = {}
        
    def compute_forgetting_rate(self, synapse_id: str, recent_activity: float, 
                              age: float, competing_activity: float = 0.0) -> float:
        \"\"\"Compute adaptive forgetting rate for a synapse.\"\"\"
        
        # Base forgetting rate
        base_rate = self.config.forgetting_rate
        
        # Activity-dependent forgetting
        if self.config.activity_dependent_forgetting:
            if recent_activity < 0.1:  # Unused synapses decay faster
                activity_factor = 2.0
            elif recent_activity > 0.8:  # Highly active synapses resist decay
                activity_factor = 0.5
            else:
                activity_factor = 1.0
        else:
            activity_factor = 1.0
            
        # Age-dependent forgetting (older memories fade faster)
        age_factor = 1.0 + age / 3600.0  # Increase with hours
        
        # Interference-based forgetting
        interference_factor = 1.0 + competing_activity * self.config.interference_strength
        
        return base_rate * activity_factor * age_factor * interference_factor
        
    def apply_forgetting(self, weight: float, forgetting_rate: float, dt: float) -> float:
        \"\"\"Apply gradual weight decay.\"\"\"
        decay_factor = np.exp(-forgetting_rate * dt)
        return weight * decay_factor
        
    def detect_interference(self, new_pattern: np.ndarray, 
                          stored_patterns: List[np.ndarray]) -> float:
        \"\"\"Detect interference between new and stored patterns.\"\"\"
        
        if not stored_patterns:
            return 0.0
            
        # Compute similarity to existing patterns
        similarities = []
        for stored in stored_patterns:
            if stored.shape == new_pattern.shape:
                similarity = np.corrcoef(new_pattern.flatten(), stored.flatten())[0, 1]
                similarities.append(abs(similarity))
                
        return max(similarities) if similarities else 0.0


class MultiTimescaleLearningSystem:
    \"\"\"Integrated multi-timescale learning system.\"\"\"
    
    def __init__(self, config: Optional[MultiTimescalePlasticityConfig] = None):
        self.config = config or MultiTimescalePlasticityConfig()
        
        # Initialize components
        self.fast_plasticity = FastPlasticityComponent(self.config)
        self.slow_plasticity = SlowPlasticityComponent(self.config)
        self.consolidation = ConsolidationSystem(self.config)
        self.forgetting = AdaptiveForgettingSystem(self.config)
        
        # State tracking
        self.synaptic_ages = {}
        self.activity_histories = {}
        self.importance_scores = {}
        self.last_update_time = time.time()
        
        print(\"Multi-Timescale Learning System initialized\")
        print(f\"  Fast plasticity: {self.config.fast_learning_rate} learning rate\")
        print(f\"  Slow plasticity: {self.config.slow_learning_rate} learning rate\")
        print(f\"  Consolidation: {self.config.consolidation_rate} rate\")
        print(f\"  Forgetting: {self.config.forgetting_rate} base rate\")
        
    def update_synapse(self, synapse_id: str, pre_spike: bool, post_spike: bool,
                      current_weight: float, global_activity: float = 0.0) -> float:
        \"\"\"Update synapse using multi-timescale plasticity.\"\"\"
        
        current_time = time.time()
        dt = current_time - self.last_update_time
        
        # Initialize synapse if new
        if synapse_id not in self.synaptic_ages:
            self.synaptic_ages[synapse_id] = 0.0
            self.activity_histories[synapse_id] = []
            self.importance_scores[synapse_id] = 0.0
            
        # Update age and activity history
        self.synaptic_ages[synapse_id] += dt
        activity = float(pre_spike or post_spike)
        self.activity_histories[synapse_id].append(activity)
        
        # Keep activity history bounded
        if len(self.activity_histories[synapse_id]) > 1000:
            self.activity_histories[synapse_id].pop(0)
            
        # 1. Fast plasticity (immediate)
        fast_change = self.fast_plasticity.compute_fast_change(
            pre_spike, post_spike, current_weight, dt
        )
        
        # 2. Slow plasticity (protein synthesis-dependent)
        self.slow_plasticity.update_protein_synthesis(activity, dt)
        slow_change = self.slow_plasticity.compute_slow_change(
            synapse_id, self.activity_histories[synapse_id], current_weight, dt
        )
        
        # 3. Apply weight changes
        new_weight = current_weight + fast_change + slow_change
        
        # 4. Late-phase stabilization
        new_weight = self.slow_plasticity.apply_late_phase_stabilization(
            synapse_id, new_weight, self.synaptic_ages[synapse_id]
        )
        
        # 5. Forgetting
        recent_activity = np.mean(self.activity_histories[synapse_id][-10:]) if len(self.activity_histories[synapse_id]) >= 10 else activity
        forgetting_rate = self.forgetting.compute_forgetting_rate(
            synapse_id, recent_activity, self.synaptic_ages[synapse_id]
        )
        new_weight = self.forgetting.apply_forgetting(new_weight, forgetting_rate, dt)
        
        # 6. Update importance score
        self.importance_scores[synapse_id] = 0.9 * self.importance_scores[synapse_id] + 0.1 * activity
        
        # 7. Check for consolidation during rest
        if self.consolidation.detect_rest_period(global_activity, current_time):
            # Consolidation effects are applied at network level
            pass
            
        # Apply bounds
        new_weight = np.clip(new_weight, self.config.weight_min, self.config.weight_max)
        
        self.last_update_time = current_time
        return new_weight
        
    def consolidate_network(self, weight_matrix: np.ndarray) -> np.ndarray:
        \"\"\"Apply consolidation to entire network during rest periods.\"\"\"
        
        # Convert importance scores to matrix format
        importance_matrix = np.zeros_like(weight_matrix)
        for synapse_id, importance in self.importance_scores.items():
            # Parse synapse_id to get indices (assuming format \"i_j\")
            try:
                i, j = map(int, synapse_id.split('_'))
                if i < importance_matrix.shape[0] and j < importance_matrix.shape[1]:
                    importance_matrix[i, j] = importance
            except (ValueError, IndexError):
                continue
                
        return self.consolidation.consolidate_memories(weight_matrix, importance_matrix)
        
    def get_learning_state(self) -> Dict[str, Any]:
        \"\"\"Get current state of multi-timescale learning system.\"\"\"
        
        return {
            'fast_plasticity': {
                'activity_trace': self.fast_plasticity.activity_trace,
                'num_fast_weights': len(self.fast_plasticity.fast_weights)
            },
            'slow_plasticity': {
                'protein_synthesis_level': self.slow_plasticity.protein_synthesis_level,
                'gene_expression_activity': self.slow_plasticity.gene_expression_activity,
                'num_late_phase_weights': len(self.slow_plasticity.late_phase_weights)
            },
            'consolidation': {
                'is_resting': self.consolidation.is_resting,
                'consolidation_strength': self.consolidation.consolidation_strength,
                'num_important_memories': len([s for s in self.importance_scores.values() if s > self.config.consolidation_threshold])
            },
            'network_stats': {
                'total_synapses': len(self.synaptic_ages),
                'mean_age': np.mean(list(self.synaptic_ages.values())) if self.synaptic_ages else 0.0,
                'mean_importance': np.mean(list(self.importance_scores.values())) if self.importance_scores else 0.0
            }
        }
        
    def reset_system(self):
        \"\"\"Reset the multi-timescale learning system.\"\"\"
        self.synaptic_ages = {}
        self.activity_histories = {}
        self.importance_scores = {}
        self.fast_plasticity = FastPlasticityComponent(self.config)
        self.slow_plasticity = SlowPlasticityComponent(self.config)
        self.consolidation = ConsolidationSystem(self.config)
        self.forgetting = AdaptiveForgettingSystem(self.config)
        self.last_update_time = time.time()


# Integration with existing plasticity system
if LEARNING_AVAILABLE:
    class MultiTimescalePlasticityRule(PlasticityRule):
        \"\"\"Integration of multi-timescale learning with existing plasticity system.\"\"\"
        
        def __init__(self, config):
            super().__init__(config)
            
            # Create multi-timescale config from existing config
            mt_config = MultiTimescalePlasticityConfig(
                fast_learning_rate=getattr(config, 'learning_rate', 0.01) * 10,
                slow_learning_rate=getattr(config, 'learning_rate', 0.01),
                weight_min=getattr(config, 'weight_min', 0.0),
                weight_max=getattr(config, 'weight_max', 10.0)
            )
            
            self.mt_system = MultiTimescaleLearningSystem(mt_config)
            
        def compute_weight_change(self, pre_activity: float, post_activity: float, 
                                current_weight: float, **kwargs) -> float:
            \"\"\"Compute weight change using multi-timescale plasticity.\"\"\"
            
            pre_spike = kwargs.get('pre_spike', pre_activity > 0.5)
            post_spike = kwargs.get('post_spike', post_activity > 0.5)
            synapse_id = kwargs.get('synapse_id', f\"{id(self)}_{hash((pre_activity, post_activity))}\")
            global_activity = kwargs.get('global_activity', 0.0)
            
            new_weight = self.mt_system.update_synapse(
                synapse_id, pre_spike, post_spike, current_weight, global_activity
            )
            
            return new_weight - current_weight


def create_multi_timescale_demo():
    \"\"\"Create a demonstration of multi-timescale learning.\"\"\"
    
    print(\"\n=== Multi-Timescale Learning Demonstration ===\")
    
    # Create system
    mt_system = MultiTimescaleLearningSystem()
    
    # Simulate learning over different timescales
    initial_weight = 1.0
    current_weight = initial_weight
    
    print(f\"\nSimulating multi-timescale learning...\")
    print(f\"Initial weight: {current_weight:.3f}\")
    
    # Phase 1: Fast learning (high activity)
    print(\"\nPhase 1: Fast learning (high activity)\")
    for i in range(10):
        current_weight = mt_system.update_synapse(
            \"demo_synapse\", True, True, current_weight, global_activity=0.8
        )
        if i % 3 == 0:
            print(f\"  Step {i+1}: Weight = {current_weight:.3f}\")
            
    # Phase 2: Rest period (consolidation)
    print(\"\nPhase 2: Rest period (consolidation)\")
    for i in range(5):
        current_weight = mt_system.update_synapse(
            \"demo_synapse\", False, False, current_weight, global_activity=0.05
        )
        
    print(f\"  After rest: Weight = {current_weight:.3f}\")
    
    # Phase 3: Slow learning (sustained activity)
    print(\"\nPhase 3: Slow learning (sustained moderate activity)\")
    for i in range(20):
        # Simulate protein synthesis building up
        spike_prob = 0.3
        pre_spike = np.random.random() < spike_prob
        post_spike = np.random.random() < spike_prob
        
        current_weight = mt_system.update_synapse(
            \"demo_synapse\", pre_spike, post_spike, current_weight, global_activity=0.3
        )
        
        if i % 5 == 0:
            print(f\"  Step {i+1}: Weight = {current_weight:.3f}\")
            
    # Get final state
    state = mt_system.get_learning_state()
    print(f\"\nFinal weight: {current_weight:.3f} (change: {current_weight - initial_weight:+.3f})\")
    print(f\"Protein synthesis level: {state['slow_plasticity']['protein_synthesis_level']:.3f}\")
    print(f\"Gene expression activity: {state['slow_plasticity']['gene_expression_activity']:.3f}\")
    
    return mt_system, current_weight


if __name__ == \"__main__\":
    # Run demonstration
    demo_system, final_weight = create_multi_timescale_demo()
    
    print(\"\n✅ Multi-timescale learning demonstration completed!\")
    print(\"\nKey features implemented:\")
    print(\"  • Fast plasticity (immediate synaptic changes)\")
    print(\"  • Slow plasticity (protein synthesis-dependent)\")
    print(\"  • Memory consolidation during rest periods\")
    print(\"  • Adaptive forgetting with activity dependence\")
    print(\"  • Integration with existing plasticity framework\")
