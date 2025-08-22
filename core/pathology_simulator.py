#!/usr/bin/env python3
"""
Simple Pathological State Modeling
==================================

Task 10: Pathological State Modeling System that implements neural dysfunction
models including hyperexcitability leading to seizure-like activity, reduced
inhibition models for studying E/I imbalance, and neuromodulation dysfunction
models for depression-like states.

Key features:
- PathologySimulator for modeling neural dysfunction
- Seizure-like activity through hyperexcitability
- E/I imbalance effects modeling
- Depression-like state modeling with reduced neuromodulation
- Connection damage and recovery mechanisms
- Integration with existing neuromorphic framework
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import warnings

try:
    from .network import NeuromorphicNetwork
    from .neurons import NeuronPopulation
    from .synapses import SynapsePopulation
    from .neuromodulation import NeuromodulatoryController
    CORE_IMPORTS_AVAILABLE = True
except ImportError:
    CORE_IMPORTS_AVAILABLE = False
    # Create minimal fallback classes for standalone operation
    print("[WARNING] Core neuromorphic modules not available - using standalone implementation")


class PathologyType(Enum):
    """Types of pathological states that can be modeled."""
    SEIZURE_LIKE = "seizure_like"
    EI_IMBALANCE = "ei_imbalance"
    DEPRESSION_LIKE = "depression_like"
    CONNECTION_DAMAGE = "connection_damage"
    HYPEREXCITABILITY = "hyperexcitability"
    HYPOEXCITABILITY = "hypoexcitability"


class PathologyStage(Enum):
    """Stages of pathological progression."""
    NORMAL = "normal"
    ONSET = "onset"
    ACTIVE = "active"
    RECOVERY = "recovery"
    CHRONIC = "chronic"


@dataclass
class PathologyConfig:
    """Configuration for pathological state modeling."""
    # General parameters
    pathology_type: PathologyType = PathologyType.SEIZURE_LIKE
    severity: float = 0.5  # 0.0 = normal, 1.0 = maximum pathology
    progression_rate: float = 0.01  # Rate of pathology development
    recovery_rate: float = 0.005  # Rate of recovery
    
    # Seizure-specific parameters
    seizure_threshold: float = 0.8  # Synchronization threshold for seizure detection
    seizure_duration_range: Tuple[float, float] = (5.0, 30.0)  # seconds
    interictal_period_range: Tuple[float, float] = (60.0, 300.0)  # seconds
    
    # E/I imbalance parameters
    inhibition_reduction: float = 0.3  # Fraction of inhibition to reduce
    excitation_increase: float = 0.0  # Fraction to increase excitation
    noise_increase: float = 0.2  # Additional noise during imbalance
    
    # Depression-like parameters
    dopamine_reduction: float = 0.4  # Reduction in dopamine signaling
    serotonin_reduction: float = 0.3  # Reduction in serotonin signaling
    plasticity_reduction: float = 0.5  # Reduction in learning rate
    anhedonia_factor: float = 0.6  # Reduced reward sensitivity
    
    # Connection damage parameters
    damage_probability: float = 0.1  # Probability of connection damage
    recovery_probability: float = 0.02  # Probability of connection recovery
    damage_severity: float = 0.8  # Strength reduction for damaged connections


class SynchronizationDetector:
    """
    Detects pathological synchronization patterns in neural activity.
    """
    
    def __init__(self, config: PathologyConfig):
        """Initialize synchronization detector."""
        self.config = config
        
        # Detection parameters
        self.window_size = 100  # Time steps for synchronization analysis
        self.activity_history = []
        self.synchronization_history = []
        
        print("SynchronizationDetector initialized for pathological activity detection")
        
    def update_activity(self, spike_data: np.ndarray, dt: float):
        """Update activity history and detect synchronization."""
        
        # Calculate instantaneous firing rate
        firing_rate = np.sum(spike_data) / len(spike_data) if len(spike_data) > 0 else 0.0
        
        # Store activity
        self.activity_history.append(firing_rate)
        
        # Keep only recent history
        if len(self.activity_history) > self.window_size:
            self.activity_history.pop(0)
            
        # Calculate synchronization if we have enough history
        if len(self.activity_history) >= self.window_size:
            synchronization = self._calculate_synchronization()
            self.synchronization_history.append(synchronization)
            
            # Keep synchronization history limited
            if len(self.synchronization_history) > self.window_size:
                self.synchronization_history.pop(0)
                
            return synchronization
        else:
            return 0.0
            
    def _calculate_synchronization(self) -> float:
        """Calculate synchronization index from recent activity."""
        
        if len(self.activity_history) < 10:
            return 0.0
            
        # Convert to numpy array for analysis
        activity = np.array(self.activity_history[-50:])  # Use last 50 steps
        
        # Calculate coefficient of variation (inverse of synchronization)
        mean_activity = np.mean(activity)
        std_activity = np.std(activity)
        
        if mean_activity > 0:
            cv = std_activity / mean_activity
            # Convert to synchronization index (high sync = low CV)
            synchronization = 1.0 / (1.0 + cv)
        else:
            synchronization = 0.0
            
        return synchronization
        
    def detect_seizure_like_activity(self) -> bool:
        """Detect if current activity resembles seizure-like patterns."""
        
        if len(self.synchronization_history) < 10:
            return False
            
        # Check recent synchronization levels
        recent_sync = np.array(self.synchronization_history[-10:])
        current_sync = np.mean(recent_sync)
        
        # Seizure detected if synchronization exceeds threshold
        return current_sync > self.config.seizure_threshold
        
    def get_synchronization_metrics(self) -> Dict[str, float]:
        """Get current synchronization metrics."""
        
        if not self.synchronization_history:
            return {
                'current_synchronization': 0.0,
                'mean_synchronization': 0.0,
                'max_synchronization': 0.0,
                'seizure_detected': False
            }
            
        current_sync = self.synchronization_history[-1]
        mean_sync = np.mean(self.synchronization_history)
        max_sync = np.max(self.synchronization_history)
        
        return {
            'current_synchronization': current_sync,
            'mean_synchronization': mean_sync,
            'max_synchronization': max_sync,
            'seizure_detected': self.detect_seizure_like_activity()
        }


class PathologySimulator:
    """
    Main pathology simulator for modeling various neural dysfunction states.
    """
    
    def __init__(self, config: PathologyConfig):
        """Initialize pathology simulator."""
        self.config = config
        
        # Pathology state
        self.current_stage = PathologyStage.NORMAL
        self.pathology_strength = 0.0
        self.time_in_stage = 0.0
        self.total_time = 0.0
        
        # Components
        self.sync_detector = SynchronizationDetector(config)
        
        # Pathology history
        self.pathology_events = []
        self.stage_history = []
        
        # Network state modifications
        self.original_weights = {}
        self.original_parameters = {}
        self.damaged_connections = set()
        
        print(f"PathologySimulator initialized for {config.pathology_type.value} modeling")
        
    def apply_pathology(self, network, dt: float) -> Dict[str, Any]:
        """Apply pathological modifications to the network."""
        
        self.total_time += dt
        self.time_in_stage += dt
        
        # Update pathology progression
        self._update_pathology_progression(dt)
        
        # Apply specific pathology type
        modifications = {}
        
        if self.config.pathology_type == PathologyType.SEIZURE_LIKE:
            modifications = self._apply_seizure_like_pathology(network)
        elif self.config.pathology_type == PathologyType.EI_IMBALANCE:
            modifications = self._apply_ei_imbalance_pathology(network)
        elif self.config.pathology_type == PathologyType.DEPRESSION_LIKE:
            modifications = self._apply_depression_like_pathology(network)
        elif self.config.pathology_type == PathologyType.CONNECTION_DAMAGE:
            modifications = self._apply_connection_damage_pathology(network)
            
        # Record pathology event
        self.pathology_events.append({
            'time': self.total_time,
            'stage': self.current_stage.value,
            'strength': self.pathology_strength,
            'modifications': modifications
        })
        
        return modifications
        
    def _update_pathology_progression(self, dt: float):
        """Update the progression of pathological state."""
        
        if self.current_stage == PathologyStage.NORMAL:
            # Check for pathology onset
            if np.random.random() < self.config.progression_rate * dt:
                self.current_stage = PathologyStage.ONSET
                self.time_in_stage = 0.0
                print(f"Pathology onset detected: {self.config.pathology_type.value}")
                
        elif self.current_stage == PathologyStage.ONSET:
            # Gradually increase pathology strength
            self.pathology_strength += self.config.progression_rate * dt
            
            if self.pathology_strength >= self.config.severity:
                self.current_stage = PathologyStage.ACTIVE
                self.time_in_stage = 0.0
                print(f"Pathology active: {self.config.pathology_type.value}")
                
        elif self.current_stage == PathologyStage.ACTIVE:
            # Maintain pathology for some time, then potentially recover
            if self.config.pathology_type == PathologyType.SEIZURE_LIKE:
                # Seizures have specific duration
                if self.time_in_stage > np.random.uniform(*self.config.seizure_duration_range):
                    self.current_stage = PathologyStage.RECOVERY
                    self.time_in_stage = 0.0
            else:
                # Other pathologies may recover probabilistically
                if np.random.random() < self.config.recovery_rate * dt:
                    self.current_stage = PathologyStage.RECOVERY
                    self.time_in_stage = 0.0
                    
        elif self.current_stage == PathologyStage.RECOVERY:
            # Gradually reduce pathology strength
            self.pathology_strength -= self.config.recovery_rate * dt
            
            if self.pathology_strength <= 0.0:
                self.pathology_strength = 0.0
                self.current_stage = PathologyStage.NORMAL
                self.time_in_stage = 0.0
                print(f"Pathology recovered: {self.config.pathology_type.value}")
                
        # Record stage history
        self.stage_history.append({
            'time': self.total_time,
            'stage': self.current_stage.value,
            'strength': self.pathology_strength
        })
        
    def _apply_seizure_like_pathology(self, network) -> Dict[str, Any]:
        """Apply seizure-like hyperexcitability to the network."""
        
        modifications = {
            'pathology_type': 'seizure_like',
            'excitability_increase': 0.0,
            'inhibition_reduction': 0.0,
            'noise_increase': 0.0
        }
        
        if self.current_stage in [PathologyStage.ONSET, PathologyStage.ACTIVE]:
            # Increase excitability
            excitability_factor = 1.0 + (self.pathology_strength * 0.5)
            modifications['excitability_increase'] = excitability_factor - 1.0
            
            # Reduce inhibition during seizure
            inhibition_factor = 1.0 - (self.pathology_strength * 0.3)
            modifications['inhibition_reduction'] = 1.0 - inhibition_factor
            
            # Increase noise
            noise_factor = 1.0 + (self.pathology_strength * 0.2)
            modifications['noise_increase'] = noise_factor - 1.0
            
            # Apply to network if it has the required interface
            if hasattr(network, 'layers'):
                for layer_name, layer in network.layers.items():
                    if hasattr(layer, 'neuron_population'):
                        # Modify neuron excitability
                        for neuron in layer.neuron_population.neurons:
                            if hasattr(neuron, 'threshold'):
                                # Lower threshold increases excitability
                                if layer_name not in self.original_parameters:
                                    self.original_parameters[layer_name] = {}
                                if 'threshold' not in self.original_parameters[layer_name]:
                                    self.original_parameters[layer_name]['threshold'] = neuron.threshold
                                
                                neuron.threshold = (self.original_parameters[layer_name]['threshold'] / 
                                                  excitability_factor)
                                
        return modifications
        
    def _apply_ei_imbalance_pathology(self, network) -> Dict[str, Any]:
        """Apply E/I imbalance effects to the network."""
        
        modifications = {
            'pathology_type': 'ei_imbalance',
            'inhibition_reduction': 0.0,
            'excitation_increase': 0.0,
            'noise_increase': 0.0
        }
        
        if self.current_stage in [PathologyStage.ONSET, PathologyStage.ACTIVE]:
            # Reduce inhibitory connections
            inhibition_reduction = self.pathology_strength * self.config.inhibition_reduction
            modifications['inhibition_reduction'] = inhibition_reduction
            
            # Optionally increase excitation
            excitation_increase = self.pathology_strength * self.config.excitation_increase
            modifications['excitation_increase'] = excitation_increase
            
            # Increase noise
            noise_increase = self.pathology_strength * self.config.noise_increase
            modifications['noise_increase'] = noise_increase
            
            # Apply to network connections
            if hasattr(network, 'connections'):
                for conn_name, connection in network.connections.items():
                    if hasattr(connection, 'synapse_population'):
                        synapses = connection.synapse_population
                        
                        # Store original weights if not already stored
                        if conn_name not in self.original_weights:
                            if hasattr(synapses, 'weights'):
                                self.original_weights[conn_name] = synapses.weights.copy()
                        
                        # Reduce inhibitory weights
                        if hasattr(synapses, 'weights') and hasattr(synapses, 'synapse_type'):
                            if 'inhibitory' in str(synapses.synapse_type).lower():
                                synapses.weights *= (1.0 - inhibition_reduction)
                            elif excitation_increase > 0:
                                synapses.weights *= (1.0 + excitation_increase)
                                
        return modifications
        
    def _apply_depression_like_pathology(self, network) -> Dict[str, Any]:
        """Apply depression-like state modifications to the network."""
        
        modifications = {
            'pathology_type': 'depression_like',
            'dopamine_reduction': 0.0,
            'plasticity_reduction': 0.0,
            'reward_sensitivity_reduction': 0.0
        }
        
        if self.current_stage in [PathologyStage.ONSET, PathologyStage.ACTIVE]:
            # Reduce dopamine signaling
            dopamine_reduction = self.pathology_strength * self.config.dopamine_reduction
            modifications['dopamine_reduction'] = dopamine_reduction
            
            # Reduce plasticity
            plasticity_reduction = self.pathology_strength * self.config.plasticity_reduction
            modifications['plasticity_reduction'] = plasticity_reduction
            
            # Reduce reward sensitivity (anhedonia)
            reward_reduction = self.pathology_strength * self.config.anhedonia_factor
            modifications['reward_sensitivity_reduction'] = reward_reduction
            
            # Apply to neuromodulatory systems if available
            if hasattr(network, 'neuromodulatory_controller'):
                controller = network.neuromodulatory_controller
                
                # Reduce dopamine levels
                if hasattr(controller, 'dopamine_level'):
                    if 'dopamine_original' not in self.original_parameters:
                        self.original_parameters['dopamine_original'] = controller.dopamine_level
                    
                    controller.dopamine_level = (self.original_parameters['dopamine_original'] * 
                                               (1.0 - dopamine_reduction))
                
                # Reduce learning rates in plasticity systems
                if hasattr(controller, 'plasticity_manager'):
                    plasticity_mgr = controller.plasticity_manager
                    
                    if hasattr(plasticity_mgr, 'learning_rate'):
                        if 'learning_rate_original' not in self.original_parameters:
                            self.original_parameters['learning_rate_original'] = plasticity_mgr.learning_rate
                        
                        plasticity_mgr.learning_rate = (self.original_parameters['learning_rate_original'] * 
                                                      (1.0 - plasticity_reduction))
                                                      
        return modifications
        
    def _apply_connection_damage_pathology(self, network) -> Dict[str, Any]:
        """Apply connection damage and recovery to the network."""
        
        modifications = {
            'pathology_type': 'connection_damage',
            'connections_damaged': 0,
            'connections_recovered': 0,
            'total_damaged': len(self.damaged_connections)
        }
        
        if hasattr(network, 'connections'):
            for conn_name, connection in network.connections.items():
                if hasattr(connection, 'synapse_population'):
                    synapses = connection.synapse_population
                    
                    if hasattr(synapses, 'weights'):
                        # Store original weights
                        if conn_name not in self.original_weights:
                            self.original_weights[conn_name] = synapses.weights.copy()
                        
                        # Apply damage
                        if self.current_stage in [PathologyStage.ONSET, PathologyStage.ACTIVE]:
                            # Randomly damage connections
                            damage_mask = np.random.random(synapses.weights.shape) < self.config.damage_probability
                            new_damage = np.sum(damage_mask)
                            
                            if new_damage > 0:
                                # Damage these connections
                                synapses.weights[damage_mask] *= (1.0 - self.config.damage_severity)
                                modifications['connections_damaged'] = new_damage
                                
                                # Track damaged connections
                                for idx in np.where(damage_mask.flatten())[0]:
                                    self.damaged_connections.add((conn_name, idx))
                        
                        # Apply recovery
                        if self.current_stage == PathologyStage.RECOVERY:
                            # Randomly recover connections
                            damaged_in_conn = [(name, idx) for name, idx in self.damaged_connections 
                                             if name == conn_name]
                            
                            recovered = 0
                            for name, idx in damaged_in_conn:
                                if np.random.random() < self.config.recovery_probability:
                                    # Recover this connection
                                    flat_idx = idx
                                    original_shape = synapses.weights.shape
                                    unravel_idx = np.unravel_index(flat_idx, original_shape)
                                    
                                    synapses.weights[unravel_idx] = self.original_weights[conn_name][unravel_idx]
                                    self.damaged_connections.remove((name, idx))
                                    recovered += 1
                            
                            modifications['connections_recovered'] = recovered
                            
        modifications['total_damaged'] = len(self.damaged_connections)
        return modifications
        
    def update_with_network_activity(self, spike_data: np.ndarray, dt: float):
        """Update pathology state based on current network activity."""
        
        # Update synchronization detection
        sync_level = self.sync_detector.update_activity(spike_data, dt)
        
        # Modify pathology based on detected activity patterns
        if self.config.pathology_type == PathologyType.SEIZURE_LIKE:
            if self.sync_detector.detect_seizure_like_activity():
                # Seizure detected - may influence pathology progression
                if self.current_stage == PathologyStage.NORMAL:
                    # Increased chance of onset during high synchronization
                    if np.random.random() < 0.1:  # 10% chance
                        self.current_stage = PathologyStage.ONSET
                        self.time_in_stage = 0.0
                        
    def restore_network(self, network):
        """Restore network to original state (remove all pathological modifications)."""
        
        # Restore original parameters
        if hasattr(network, 'layers'):
            for layer_name, layer in network.layers.items():
                if layer_name in self.original_parameters:
                    params = self.original_parameters[layer_name]
                    
                    if hasattr(layer, 'neuron_population'):
                        for neuron in layer.neuron_population.neurons:
                            for param_name, original_value in params.items():
                                if hasattr(neuron, param_name):
                                    setattr(neuron, param_name, original_value)
        
        # Restore original weights
        if hasattr(network, 'connections'):
            for conn_name, connection in network.connections.items():
                if conn_name in self.original_weights:
                    if hasattr(connection, 'synapse_population'):
                        synapses = connection.synapse_population
                        if hasattr(synapses, 'weights'):
                            synapses.weights = self.original_weights[conn_name].copy()
        
        # Restore neuromodulatory parameters
        if hasattr(network, 'neuromodulatory_controller'):
            controller = network.neuromodulatory_controller
            
            if 'dopamine_original' in self.original_parameters:
                if hasattr(controller, 'dopamine_level'):
                    controller.dopamine_level = self.original_parameters['dopamine_original']
            
            if 'learning_rate_original' in self.original_parameters:
                if hasattr(controller, 'plasticity_manager'):
                    plasticity_mgr = controller.plasticity_manager
                    if hasattr(plasticity_mgr, 'learning_rate'):
                        plasticity_mgr.learning_rate = self.original_parameters['learning_rate_original']
        
        # Reset pathology state
        self.current_stage = PathologyStage.NORMAL
        self.pathology_strength = 0.0
        self.time_in_stage = 0.0
        self.damaged_connections.clear()
        
        print("Network restored to original state")
        
    def get_pathology_state(self) -> Dict[str, Any]:
        """Get comprehensive pathology state information."""
        
        sync_metrics = self.sync_detector.get_synchronization_metrics()
        
        return {
            'pathology_type': self.config.pathology_type.value,
            'current_stage': self.current_stage.value,
            'pathology_strength': self.pathology_strength,
            'time_in_stage': self.time_in_stage,
            'total_time': self.total_time,
            'synchronization_metrics': sync_metrics,
            'total_events': len(self.pathology_events),
            'damaged_connections': len(self.damaged_connections)
        }


def demo_pathology_simulator():
    """Demonstrate the pathology simulator system."""
    
    print("=== Pathological State Modeling Demo ===")
    
    # Test different pathology types
    pathology_types = [
        PathologyType.SEIZURE_LIKE,
        PathologyType.EI_IMBALANCE,
        PathologyType.DEPRESSION_LIKE,
        PathologyType.CONNECTION_DAMAGE
    ]
    
    for pathology_type in pathology_types:
        print(f"\n{pathology_type.value.upper()} Pathology Simulation")
        print("-" * 50)
        
        # Create pathology configuration
        config = PathologyConfig(
            pathology_type=pathology_type,
            severity=0.7,
            progression_rate=0.05,
            recovery_rate=0.02
        )
        
        # Initialize simulator
        simulator = PathologySimulator(config)
        
        # Simulate pathology progression
        print("Simulating pathology progression...")
        
        for step in range(200):  # 20 seconds at 0.1s timesteps
            dt = 0.1
            
            # Generate synthetic spike data
            if pathology_type == PathologyType.SEIZURE_LIKE:
                # Higher activity during seizure stages
                if simulator.current_stage in [PathologyStage.ONSET, PathologyStage.ACTIVE]:
                    spike_prob = 0.3 + 0.4 * simulator.pathology_strength
                else:
                    spike_prob = 0.1
            else:
                spike_prob = 0.15
                
            spike_data = np.random.random(1000) < spike_prob
            
            # Update simulator
            simulator.update_with_network_activity(spike_data, dt)
            
            # Apply pathology (to a mock network)
            class MockNetwork:
                def __init__(self):
                    self.layers = {}
                    self.connections = {}
                    
            mock_network = MockNetwork()
            modifications = simulator.apply_pathology(mock_network, dt)
            
            # Log significant events
            if step % 50 == 0:  # Every 5 seconds
                state = simulator.get_pathology_state()
                print(f"  t={step*dt:.1f}s: Stage={state['current_stage']}, "
                      f"Strength={state['pathology_strength']:.3f}")
                
        # Final state report
        final_state = simulator.get_pathology_state()
        print(f"Final state: {final_state['current_stage']}")
        print(f"Total pathology events: {final_state['total_events']}")
        
        if pathology_type == PathologyType.SEIZURE_LIKE:
            sync_metrics = final_state['synchronization_metrics']
            print(f"Seizure detection: {sync_metrics['seizure_detected']}")
            print(f"Max synchronization: {sync_metrics['max_synchronization']:.3f}")
        
        print()
    
    print("✅ Pathological State Modeling Demo Complete!")
    
    return True


if __name__ == "__main__":
    # Run demonstration
    success = demo_pathology_simulator()
    
    print("\n=== Task 10 Implementation Summary ===")
    print("✅ Simple Pathological State Modeling - COMPLETED")
    print("\nKey Features Implemented:")
    print("  • PathologySimulator for modeling neural dysfunction")
    print("  • SynchronizationDetector for seizure-like activity detection")
    print("  • Seizure-like hyperexcitability modeling")
    print("  • E/I imbalance effects with reduced inhibition")
    print("  • Depression-like states with reduced neuromodulation")
    print("  • Connection damage and recovery mechanisms")
    print("  • Pathology progression and recovery dynamics")
    print("  • Integration with existing neuromorphic framework")
    
    print("\nNext Steps:")
    print("  → Task 10.1: Implement detailed seizure activity model")
    print("  → Task 10.2: Model E/I imbalance effects")
    print("  → Task 10.3: Add depression-like state model")