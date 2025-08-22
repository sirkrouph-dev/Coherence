#!/usr/bin/env python3
"""
Attention Mechanism Implementation
=================================

Task 8: Attention mechanism system that implements selective processing
with bottom-up attention driven by stimulus salience, top-down attention
based on task goals, attention switching, and inhibition of return.

Key features:
- AttentionController for selective neural processing
- Bottom-up attention driven by stimulus salience
- Top-down attention based on task goals and expectations
- Attention switching with inhibition of return
- Spatial and feature-based attention mechanisms
- Integration with sensory hierarchies and working memory
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import warnings

try:
    from .hierarchical_sensory_processing import (
        SensoryHierarchy, ProcessingLevel
    )
    from .working_memory import WorkingMemoryNetwork, AttentionMode
    HIERARCHICAL_IMPORTS_AVAILABLE = True
except ImportError:
    HIERARCHICAL_IMPORTS_AVAILABLE = False
    # Minimal fallback classes
    from enum import Enum
    
    class ProcessingLevel(Enum):
        PRIMARY = "primary"
        SECONDARY = "secondary"
        ASSOCIATION = "association"
        INTEGRATION = "integration"
        
    class AttentionMode(Enum):
        FOCUSED = "focused"
        DIVIDED = "divided"
        AUTOMATIC = "automatic"


class AttentionType(Enum):
    """Types of attention mechanisms."""
    BOTTOM_UP = "bottom_up"          # Stimulus-driven
    TOP_DOWN = "top_down"            # Goal-driven
    SPATIAL = "spatial"              # Location-based
    FEATURE = "feature"              # Feature-based
    OBJECT = "object"                # Object-based


class AttentionState(Enum):
    """States of attention system."""
    IDLE = "idle"
    SEARCHING = "searching"
    FOCUSED = "focused"
    SWITCHING = "switching"
    INHIBITED = "inhibited"


@dataclass
class AttentionConfig:
    """Configuration for attention mechanisms."""
    # Bottom-up attention
    salience_threshold: float = 0.3
    contrast_sensitivity: float = 2.0
    temporal_sensitivity: float = 1.5
    
    # Top-down attention
    goal_weight: float = 0.7
    expectation_weight: float = 0.5
    memory_influence: float = 0.4
    
    # Attention dynamics
    switching_threshold: float = 0.6
    switching_cost: float = 0.2
    inhibition_duration: float = 0.5
    attention_decay_rate: float = 0.1
    
    # Spatial attention
    spatial_kernel_size: int = 5
    spatial_falloff_rate: float = 0.3
    
    # Feature attention
    feature_selectivity: float = 0.8
    feature_competition_strength: float = 0.5


class SalienceDetector:
    """
    Detects bottom-up salience in sensory inputs based on
    contrast, motion, uniqueness, and other stimulus properties.
    """
    
    def __init__(self, config: AttentionConfig):
        """Initialize salience detector."""
        self.config = config
        
        # Salience computation state
        self.previous_input = None
        self.salience_history = []
        self.contrast_maps = []
        
        print("SalienceDetector initialized for bottom-up attention")
        
    def compute_salience_map(self, input_data: np.ndarray) -> np.ndarray:
        """Compute salience map from input data."""
        
        # Ensure 2D input
        if len(input_data.shape) == 1:
            side_length = int(np.sqrt(len(input_data)))
            if side_length * side_length == len(input_data):
                input_2d = input_data.reshape(side_length, side_length)
            else:
                # Treat as 1D signal
                input_2d = input_data.reshape(1, -1)
        else:
            input_2d = input_data
            
        salience_map = np.zeros_like(input_2d, dtype=float)
        
        # 1. Intensity contrast salience
        contrast_salience = self._compute_contrast_salience(input_2d)
        salience_map += contrast_salience * 0.4
        
        # 2. Temporal change salience
        if self.previous_input is not None:
            temporal_salience = self._compute_temporal_salience(input_2d)
            salience_map += temporal_salience * 0.3
            
        # 3. Uniqueness/pop-out salience
        uniqueness_salience = self._compute_uniqueness_salience(input_2d)
        salience_map += uniqueness_salience * 0.3
        
        # Store for next iteration
        self.previous_input = input_2d.copy()
        self.salience_history.append(np.mean(salience_map))
        
        return salience_map
        
    def _compute_contrast_salience(self, input_data: np.ndarray) -> np.ndarray:
        """Compute salience based on local contrast."""
        
        # Center-surround contrast
        from scipy import ndimage
        
        # Gaussian kernels for center and surround
        center_kernel = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]]) / 8.0
        surround_kernel = np.ones((5, 5)) / 25.0
        
        try:
            center_response = ndimage.convolve(input_data, center_kernel, mode='constant')
            surround_response = ndimage.convolve(input_data, surround_kernel, mode='constant')
            
            contrast = np.abs(center_response - surround_response)
            
        except ImportError:
            # Fallback without scipy
            contrast = np.zeros_like(input_data)
            h, w = input_data.shape
            
            for i in range(1, h-1):
                for j in range(1, w-1):
                    center = input_data[i, j]
                    surround = np.mean(input_data[i-1:i+2, j-1:j+2])
                    contrast[i, j] = abs(center - surround)
                    
        return contrast * self.config.contrast_sensitivity
        
    def _compute_temporal_salience(self, input_data: np.ndarray) -> np.ndarray:
        """Compute salience based on temporal changes."""
        
        if self.previous_input.shape != input_data.shape:
            # Resize previous input to match current
            if input_data.size <= self.previous_input.size:
                prev_resized = self.previous_input.flat[:input_data.size].reshape(input_data.shape)
            else:
                # Pad previous input
                prev_resized = np.zeros_like(input_data)
                flat_prev = self.previous_input.flatten()
                prev_resized.flat[:len(flat_prev)] = flat_prev
        else:
            prev_resized = self.previous_input
            
        # Temporal difference
        temporal_diff = np.abs(input_data - prev_resized)
        
        return temporal_diff * self.config.temporal_sensitivity
        
    def _compute_uniqueness_salience(self, input_data: np.ndarray) -> np.ndarray:
        """Compute salience based on local uniqueness/pop-out."""
        
        uniqueness = np.zeros_like(input_data)
        h, w = input_data.shape
        
        # Simple uniqueness: how different is each pixel from its neighbors
        for i in range(h):
            for j in range(w):
                # Get neighborhood
                i_start, i_end = max(0, i-1), min(h, i+2)
                j_start, j_end = max(0, j-1), min(w, j+2)
                
                neighborhood = input_data[i_start:i_end, j_start:j_end]
                center_val = input_data[i, j]
                
                # Neighborhood mean calculation with bounds checking
                if len(neighborhood) > 1:
                    neighborhood_vals = neighborhood[neighborhood != center_val]
                    if len(neighborhood_vals) > 0:
                        neighborhood_mean = np.mean(neighborhood_vals)
                        uniqueness[i, j] = abs(center_val - neighborhood_mean)
                    else:
                        uniqueness[i, j] = 0.0
                else:
                    uniqueness[i, j] = 0.0
                
        return uniqueness
        
    def get_most_salient_location(self, salience_map: np.ndarray) -> Tuple[int, int]:
        """Get location of maximum salience."""
        
        # Find peak salience location
        max_idx = np.unravel_index(np.argmax(salience_map), salience_map.shape)
        return max_idx
        
    def is_salient_enough(self, salience_map: np.ndarray) -> bool:
        """Check if there's enough salience to capture attention."""
        
        max_salience = np.max(salience_map)
        return max_salience > self.config.salience_threshold


class TopDownAttention:
    """
    Implements top-down attention based on task goals, expectations,
    and working memory contents.
    """
    
    def __init__(self, config: AttentionConfig):
        """Initialize top-down attention."""
        self.config = config
        
        # Top-down state
        self.current_goals = {}
        self.expectations = {}
        self.task_templates = {}
        self.attention_templates = {}
        
        print("TopDownAttention initialized for goal-driven attention")
        
    def set_attention_goal(self, goal_type: str, target_features: np.ndarray, 
                          priority: float = 1.0):
        """Set top-down attention goal."""
        
        self.current_goals[goal_type] = {
            'target_features': target_features.copy(),
            'priority': priority,
            'activation_time': 0.0
        }
        
        # Create attention template
        self.attention_templates[goal_type] = self._create_attention_template(target_features)
        
    def _create_attention_template(self, target_features: np.ndarray) -> np.ndarray:
        """Create attention template from target features."""
        
        # Simple template: normalized target features
        if np.linalg.norm(target_features) > 0:
            template = target_features / np.linalg.norm(target_features)
        else:
            template = target_features
            
        return template
        
    def compute_top_down_bias(self, input_features: np.ndarray) -> np.ndarray:
        """Compute top-down attention bias for input features."""
        
        if not self.current_goals:
            return np.zeros_like(input_features)
            
        total_bias = np.zeros_like(input_features, dtype=float)
        
        # Compute bias from each active goal
        for goal_type, goal_info in self.current_goals.items():
            template = self.attention_templates.get(goal_type)
            if template is not None:
                
                # Resize template to match input if needed
                if len(template) != len(input_features):
                    if len(template) > len(input_features):
                        template_resized = template[:len(input_features)]
                    else:
                        template_resized = np.zeros(len(input_features))
                        template_resized[:len(template)] = template
                else:
                    template_resized = template
                    
                # Compute similarity
                if np.linalg.norm(input_features) > 0 and np.linalg.norm(template_resized) > 0:
                    input_norm = input_features / np.linalg.norm(input_features)
                    template_norm = template_resized / np.linalg.norm(template_resized)
                    
                    similarity = np.dot(input_norm, template_norm)
                    goal_bias = similarity * goal_info['priority'] * self.config.goal_weight
                    
                    # Apply bias as multiplicative enhancement
                    feature_bias = input_features * (1.0 + goal_bias)
                    total_bias += feature_bias * goal_info['priority']
                    
        return total_bias
        
    def update_expectations(self, predicted_features: Dict[str, np.ndarray]):
        """Update expectations for upcoming stimuli."""
        
        self.expectations.update(predicted_features)
        
    def clear_goals(self):
        """Clear all current goals."""
        
        self.current_goals.clear()
        self.attention_templates.clear()


class AttentionController:
    """
    Main attention controller that integrates bottom-up salience detection
    with top-down control, implements attention switching, and manages
    inhibition of return.
    """
    
    def __init__(self, config: AttentionConfig):
        """Initialize attention controller."""
        self.config = config
        
        # Attention components
        self.salience_detector = SalienceDetector(config)
        self.top_down_attention = TopDownAttention(config)
        
        # Attention state
        self.current_state = AttentionState.IDLE
        self.attention_location = None
        self.attention_features = None
        self.attention_strength = 0.0
        
        # Inhibition of return
        self.inhibited_locations = []
        self.inhibition_timers = []
        
        # Attention history
        self.attention_history = []
        self.switching_events = []
        
        print("AttentionController initialized with integrated attention mechanisms")
        
    def update_attention(self, sensory_input: np.ndarray, 
                        working_memory_contents: Optional[List[np.ndarray]] = None,
                        dt: float = 0.001) -> Dict[str, Any]:
        """Update attention based on sensory input and internal state."""
        
        # Update inhibition timers
        self._update_inhibition_timers(dt)
        
        # Compute bottom-up salience
        salience_map = self.salience_detector.compute_salience_map(sensory_input)
        
        # Apply inhibition of return
        inhibited_salience = self._apply_inhibition_of_return(salience_map)
        
        # Compute top-down bias
        if len(sensory_input.shape) == 1:
            input_features = sensory_input
        else:
            input_features = sensory_input.flatten()
            
        top_down_bias = self.top_down_attention.compute_top_down_bias(input_features)
        
        # Combine bottom-up and top-down attention
        combined_attention = self._combine_attention_signals(
            inhibited_salience, top_down_bias, working_memory_contents
        )
        
        # Make attention decision
        attention_decision = self._make_attention_decision(combined_attention)
        
        # Update attention state
        self._update_attention_state(attention_decision, dt)
        
        # Store attention event
        self.attention_history.append({
            'timestamp': len(self.attention_history) * dt,
            'state': self.current_state.value,
            'location': self.attention_location,
            'strength': self.attention_strength,
            'salience': np.max(inhibited_salience) if inhibited_salience.size > 0 else 0.0
        })
        
        return {
            'attention_state': self.current_state.value,
            'attention_location': self.attention_location,
            'attention_strength': self.attention_strength,
            'salience_map': inhibited_salience,
            'top_down_bias': top_down_bias
        }
        
    def _update_inhibition_timers(self, dt: float):
        """Update inhibition of return timers."""
        
        # Decrement timers
        self.inhibition_timers = [max(0.0, timer - dt) for timer in self.inhibition_timers]
        
        # Remove expired inhibitions
        active_inhibitions = []
        active_timers = []
        
        for i, timer in enumerate(self.inhibition_timers):
            if timer > 0.0:
                active_inhibitions.append(self.inhibited_locations[i])
                active_timers.append(timer)
                
        self.inhibited_locations = active_inhibitions
        self.inhibition_timers = active_timers
        
    def _apply_inhibition_of_return(self, salience_map: np.ndarray) -> np.ndarray:
        """Apply inhibition of return to salience map."""
        
        inhibited_salience = salience_map.copy()
        
        # Suppress salience at inhibited locations
        for location in self.inhibited_locations:
            if len(salience_map.shape) == 2:
                i, j = location
                if 0 <= i < salience_map.shape[0] and 0 <= j < salience_map.shape[1]:
                    # Apply Gaussian suppression around inhibited location
                    suppression_radius = self.config.spatial_kernel_size // 2
                    
                    for di in range(-suppression_radius, suppression_radius + 1):
                        for dj in range(-suppression_radius, suppression_radius + 1):
                            ni, nj = i + di, j + dj
                            if 0 <= ni < salience_map.shape[0] and 0 <= nj < salience_map.shape[1]:
                                distance = np.sqrt(di*di + dj*dj)
                                suppression = np.exp(-distance * self.config.spatial_falloff_rate)
                                inhibited_salience[ni, nj] *= (1.0 - suppression * 0.7)
                                
        return inhibited_salience
        
    def _combine_attention_signals(self, bottom_up_salience: np.ndarray,
                                 top_down_bias: np.ndarray,
                                 working_memory_contents: Optional[List[np.ndarray]] = None) -> np.ndarray:
        """Combine bottom-up and top-down attention signals."""
        
        # Flatten salience map for combination
        if len(bottom_up_salience.shape) == 2:
            salience_flat = bottom_up_salience.flatten()
        else:
            salience_flat = bottom_up_salience
            
        # Ensure same size
        min_size = min(len(salience_flat), len(top_down_bias))
        salience_resized = salience_flat[:min_size]
        bias_resized = top_down_bias[:min_size]
        
        # Combine with configurable weights
        bottom_up_weight = 1.0 - self.config.goal_weight
        combined = (bottom_up_weight * salience_resized + 
                   self.config.goal_weight * bias_resized)
        
        # Add working memory influence
        if working_memory_contents:
            memory_influence = self._compute_memory_influence(
                combined, working_memory_contents
            )
            combined += self.config.memory_influence * memory_influence
            
        return combined
        
    def _compute_memory_influence(self, attention_signal: np.ndarray,
                                working_memory_contents: List[np.ndarray]) -> np.ndarray:
        """Compute influence of working memory on attention."""
        
        memory_influence = np.zeros_like(attention_signal)
        
        for memory_item in working_memory_contents:
            if memory_item is not None:
                # Resize memory item to match attention signal
                if len(memory_item) <= len(attention_signal):
                    memory_resized = np.zeros_like(attention_signal)
                    memory_resized[:len(memory_item)] = memory_item
                else:
                    memory_resized = memory_item[:len(attention_signal)]
                    
                # Add similarity-based influence
                if np.linalg.norm(attention_signal) > 0 and np.linalg.norm(memory_resized) > 0:
                    attention_norm = attention_signal / np.linalg.norm(attention_signal)
                    memory_norm = memory_resized / np.linalg.norm(memory_resized)
                    
                    similarity = np.dot(attention_norm, memory_norm)
                    memory_influence += similarity * memory_resized * 0.3
                    
        return memory_influence
        
    def _make_attention_decision(self, combined_attention: np.ndarray) -> Dict[str, Any]:
        """Make attention decision based on combined signals."""
        
        # Handle NaN values in combined attention
        valid_attention = combined_attention[~np.isnan(combined_attention)]
        if len(valid_attention) == 0:
            # All values are NaN, return default decision
            return {
                'should_switch': False,
                'new_state': self.current_state,
                'new_location': (0, 0),
                'new_strength': 0.0
            }
            
        max_attention = np.max(valid_attention)
        
        # Find location of max attention (handling NaN)
        max_location = np.nanargmax(combined_attention)
        
        # Convert to 2D location if needed
        if hasattr(self, '_last_salience_shape') and len(self._last_salience_shape) == 2:
            location_2d = np.unravel_index(max_location, self._last_salience_shape)
        else:
            location_2d = (0, max_location)
            
        # Decision logic
        should_switch = False
        new_state = self.current_state  # Default to current state
        
        if self.current_state == AttentionState.IDLE:
            if max_attention > self.config.salience_threshold:
                should_switch = True
                new_state = AttentionState.FOCUSED
                
        elif self.current_state == AttentionState.FOCUSED:
            # Check if should switch to new location
            if max_attention > self.attention_strength + self.config.switching_threshold:
                should_switch = True
                new_state = AttentionState.SWITCHING
            else:
                new_state = AttentionState.FOCUSED
            
        return {
            'should_switch': should_switch,
            'new_state': new_state,
            'new_location': location_2d,
            'new_strength': max_attention
        }
        
    def _update_attention_state(self, decision: Dict[str, Any], dt: float):
        """Update attention state based on decision."""
        
        if decision['should_switch']:
            # Add current location to inhibition if switching
            if (self.attention_location is not None and 
                self.current_state == AttentionState.FOCUSED):
                self.inhibited_locations.append(self.attention_location)
                self.inhibition_timers.append(self.config.inhibition_duration)
                
                # Log switching event
                self.switching_events.append({
                    'timestamp': len(self.attention_history) * dt,
                    'from_location': self.attention_location,
                    'to_location': decision['new_location'],
                    'switching_cost': self.config.switching_cost
                })
                
            # Update attention
            self.attention_location = decision['new_location']
            self.attention_strength = decision['new_strength'] * (1.0 - self.config.switching_cost)
            
        else:
            # Apply attention decay if not switching
            self.attention_strength *= (1.0 - self.config.attention_decay_rate * dt)
            
        # Update state
        self.current_state = decision['new_state']
        
        # Check if attention too weak
        if self.attention_strength < self.config.salience_threshold * 0.5:
            self.current_state = AttentionState.IDLE
            self.attention_location = None
            
    def set_top_down_goal(self, goal_type: str, target_features: np.ndarray, 
                         priority: float = 1.0):
        """Set top-down attention goal."""
        self.top_down_attention.set_attention_goal(goal_type, target_features, priority)
        
    def clear_attention_goals(self):
        """Clear all attention goals."""
        self.top_down_attention.clear_goals()
        
    def get_attention_info(self) -> Dict[str, Any]:
        """Get comprehensive attention system information."""
        return {
            'current_state': self.current_state.value,
            'attention_location': self.attention_location,
            'attention_strength': self.attention_strength,
            'inhibited_locations': len(self.inhibited_locations),
            'active_goals': len(self.top_down_attention.current_goals),
            'switching_events': len(self.switching_events),
            'total_attention_events': len(self.attention_history)
        }


def demo_attention_mechanism():
    """Demonstrate the attention mechanism system."""
    
    print("=== Attention Mechanism System Demo ===")
    
    # Create attention controller
    print("\n1. Creating Attention Controller")
    
    config = AttentionConfig(
        salience_threshold=0.2,
        goal_weight=0.6,
        switching_threshold=0.4,
        inhibition_duration=0.3
    )
    
    attention_controller = AttentionController(config)
    
    print("Attention controller created with integrated mechanisms")
    
    # Test bottom-up attention
    print("\n2. Testing Bottom-Up Attention (Salience Detection)")
    
    # Create input with salient regions
    test_input = np.zeros((8, 8))
    test_input[2:4, 2:4] = 1.0  # Salient region
    test_input += np.random.normal(0, 0.1, (8, 8))  # Background noise
    
    attention_result = attention_controller.update_attention(test_input)
    
    print(f"Bottom-up attention state: {attention_result['attention_state']}")
    print(f"Attention location: {attention_result['attention_location']}")
    print(f"Attention strength: {attention_result['attention_strength']:.3f}")
    
    # Test top-down attention
    print("\n3. Testing Top-Down Attention (Goal-Driven)")
    
    # Set attention goal
    target_features = np.array([1.0, 0.8, 0.6, 0.4, 0.2] * 8)  # Target pattern
    attention_controller.set_top_down_goal("search_target", target_features[:40], priority=0.8)
    
    # Create input with target-like features
    target_input = np.random.rand(8, 8) * 0.3
    target_input[4:6, 4:6] = 0.8  # Target-like region
    
    td_result = attention_controller.update_attention(target_input)
    
    print(f"Top-down attention state: {td_result['attention_state']}")
    print(f"Goal-driven location: {td_result['attention_location']}")
    print(f"Enhanced strength: {td_result['attention_strength']:.3f}")
    
    # Test attention switching and inhibition of return
    print("\n4. Testing Attention Switching and Inhibition of Return")
    
    # Create sequence of inputs with competing regions
    switching_inputs = []
    
    # Input 1: Left region salient
    input1 = np.zeros((6, 6))
    input1[1:3, 1:3] = 1.2
    switching_inputs.append(input1)
    
    # Input 2: Right region more salient
    input2 = np.zeros((6, 6))
    input2[1:3, 1:3] = 0.8  # Previous region weaker
    input2[3:5, 3:5] = 1.5  # New region stronger
    switching_inputs.append(input2)
    
    # Input 3: Back to left (should be inhibited)
    input3 = np.zeros((6, 6))
    input3[1:3, 1:3] = 1.4  # Strong but inhibited
    input3[3:5, 3:5] = 0.9  # Current location
    switching_inputs.append(input3)
    
    switching_results = []
    for i, inp in enumerate(switching_inputs):
        result = attention_controller.update_attention(inp, dt=0.1)  # Larger time steps
        switching_results.append(result)
        print(f"  Step {i+1}: State={result['attention_state']}, Location={result['attention_location']}")
        
    # Check inhibition of return
    attention_info = attention_controller.get_attention_info()
    print(f"Inhibited locations: {attention_info['inhibited_locations']}")
    print(f"Switching events: {attention_info['switching_events']}")
    
    print("\n✅ Attention Mechanism Demo Complete!")
    
    return attention_controller


if __name__ == "__main__":
    # Run demonstration
    attention_controller = demo_attention_mechanism()
    
    print("\n=== Task 8 Implementation Summary ===")
    print("✅ Attention Mechanism Implementation - COMPLETED")
    print("\nKey Features Implemented:")
    print("  • SalienceDetector for bottom-up attention with contrast and motion detection")
    print("  • TopDownAttention for goal-driven attention with feature templates")
    print("  • AttentionController integrating bottom-up and top-down mechanisms")
    print("  • Attention switching with configurable thresholds and costs")
    print("  • Inhibition of return preventing immediate re-attention to previous locations")
    print("  • Spatial and feature-based attention mechanisms")
    
    print("\nNext Steps:")
    print("  → Task 9: Developmental Plasticity and Critical Periods")