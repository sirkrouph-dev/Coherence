#!/usr/bin/env python3
"""
Working Memory System Implementation
===================================

Task 7: Working memory system that implements limited capacity memory (3-7 items),
persistent activity patterns for information maintenance, interference effects,
attention-based control, and decay/refreshing mechanisms.

Key features:
- WorkingMemoryNetwork with limited capacity (3-7 items)
- Persistent activity patterns for memory maintenance
- Interference effects between competing memories
- Attention-based control for memory management
- Decay and refreshing mechanisms for memory persistence
- Integration with existing neuromorphic infrastructure
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import warnings

try:
    from .network import NeuromorphicNetwork, NetworkLayer
    NETWORK_IMPORTS_AVAILABLE = True
except ImportError:
    NETWORK_IMPORTS_AVAILABLE = False


class MemoryState(Enum):
    """States of working memory items."""
    EMPTY = "empty"
    ENCODING = "encoding"
    MAINTENANCE = "maintenance"
    RETRIEVAL = "retrieval"
    DECAY = "decay"
    INTERFERENCE = "interference"


class AttentionMode(Enum):
    """Attention control modes for working memory."""
    FOCUSED = "focused"        # Single item focus
    DIVIDED = "divided"        # Multiple item attention
    AUTOMATIC = "automatic"    # Default maintenance
    REFRESH = "refresh"        # Active refreshing


@dataclass
class MemoryItem:
    """Individual item stored in working memory."""
    content: np.ndarray
    strength: float = 1.0
    age: float = 0.0
    interference_level: float = 0.0
    state: MemoryState = MemoryState.EMPTY
    encoding_time: float = 0.0
    last_refresh_time: float = 0.0
    priority: float = 1.0


@dataclass
class WorkingMemoryConfig:
    """Configuration for working memory system."""
    capacity: int = 7                    # Maximum number of items (3-7)
    decay_rate: float = 0.01            # Rate of memory decay per time step
    interference_strength: float = 0.1   # Strength of interference between items
    refresh_rate: float = 0.05          # Rate of active refresh
    attention_capacity: float = 1.0     # Total attention available
    encoding_threshold: float = 0.3     # Minimum strength for encoding
    retrieval_threshold: float = 0.2    # Minimum strength for retrieval
    maintenance_noise: float = 0.02     # Noise in maintenance activity


class WorkingMemoryBuffer:
    """
    Individual working memory buffer that maintains a single item
    through persistent neural activity.
    """
    
    def __init__(self, buffer_id: int, item_size: int, config: WorkingMemoryConfig):
        """Initialize working memory buffer."""
        self.buffer_id = buffer_id
        self.item_size = item_size
        self.config = config
        
        # Buffer state
        self.memory_item = MemoryItem(content=np.zeros(item_size))
        self.neural_activity = np.zeros(item_size)
        self.maintenance_weights = np.random.normal(0, 0.1, (item_size, item_size))
        
        # Activity history for analysis
        self.activity_history = []
        self.interference_history = []
        
        print(f"WorkingMemoryBuffer {buffer_id} initialized with size {item_size}")
        
    def encode_item(self, item_content: np.ndarray, strength: float = 1.0) -> bool:
        """Encode new item into buffer."""
        if self.memory_item.state != MemoryState.EMPTY and self.memory_item.strength > 0.1:
            # Buffer occupied, encoding fails
            return False
            
        # Encode new item
        self.memory_item.content = item_content.copy()
        self.memory_item.strength = strength
        self.memory_item.age = 0.0
        self.memory_item.state = MemoryState.ENCODING
        self.memory_item.encoding_time = 0.0
        self.memory_item.last_refresh_time = 0.0
        
        # Initialize neural activity
        self.neural_activity = item_content * strength
        
        return True
        
    def maintain_activity(self, dt: float = 0.001, attention_weight: float = 1.0):
        """Maintain persistent activity for stored item."""
        if self.memory_item.state == MemoryState.EMPTY:
            return
            
        # Update age
        self.memory_item.age += dt
        
        # Maintenance through recurrent activity
        if self.memory_item.state in [MemoryState.ENCODING, MemoryState.MAINTENANCE]:
            # Recurrent maintenance
            recurrent_input = np.dot(self.maintenance_weights, self.neural_activity)
            
            # Apply attention modulation
            maintenance_strength = attention_weight * 0.5
            
            # Update activity with decay and maintenance
            decay = self.config.decay_rate * dt
            maintenance = maintenance_strength * recurrent_input * dt
            noise = np.random.normal(0, self.config.maintenance_noise, self.item_size) * dt
            
            self.neural_activity = (
                self.neural_activity * (1 - decay) +  # Decay
                maintenance +                           # Maintenance
                noise                                   # Noise
            )
            
            # Clip activity to reasonable bounds
            self.neural_activity = np.clip(self.neural_activity, -2.0, 2.0)
            
            # Update item strength based on activity
            self.memory_item.strength = np.mean(np.abs(self.neural_activity))
            
            # State transitions
            if self.memory_item.state == MemoryState.ENCODING and self.memory_item.age > 0.1:
                self.memory_item.state = MemoryState.MAINTENANCE
                
            elif (self.memory_item.state == MemoryState.MAINTENANCE and 
                  self.memory_item.strength < self.config.retrieval_threshold):
                self.memory_item.state = MemoryState.DECAY
                
        # Store history
        self.activity_history.append({
            'time': self.memory_item.age,
            'strength': self.memory_item.strength,
            'state': self.memory_item.state,
            'activity': np.mean(np.abs(self.neural_activity))
        })
        
    def apply_interference(self, interfering_activity: np.ndarray):
        """Apply interference from other active buffers."""
        if self.memory_item.state in [MemoryState.MAINTENANCE, MemoryState.ENCODING]:
            # Compute interference based on similarity
            similarity = np.dot(self.neural_activity, interfering_activity) / (
                np.linalg.norm(self.neural_activity) * np.linalg.norm(interfering_activity) + 1e-8
            )
            
            interference_strength = self.config.interference_strength * abs(similarity)
            
            # Apply interference (reduces activity)
            interference_effect = interfering_activity * interference_strength
            self.neural_activity -= interference_effect
            
            # Update interference level
            self.memory_item.interference_level += interference_strength
            
            # Store interference event
            self.interference_history.append({
                'time': self.memory_item.age,
                'similarity': similarity,
                'interference_strength': interference_strength
            })
            
    def refresh_item(self, refresh_strength: float = 1.0):
        """Actively refresh the stored item."""
        if self.memory_item.state in [MemoryState.MAINTENANCE, MemoryState.DECAY]:
            # Boost activity toward original content
            target_activity = self.memory_item.content * refresh_strength
            refresh_rate = self.config.refresh_rate
            
            self.neural_activity += (target_activity - self.neural_activity) * refresh_rate
            
            # Update refresh time
            self.memory_item.last_refresh_time = self.memory_item.age
            
            # If strength was low, potentially rescue from decay
            if self.memory_item.state == MemoryState.DECAY:
                new_strength = np.mean(np.abs(self.neural_activity))
                if new_strength > self.config.retrieval_threshold:
                    self.memory_item.state = MemoryState.MAINTENANCE
                    
    def retrieve_item(self) -> Optional[np.ndarray]:
        """Retrieve item from buffer."""
        if (self.memory_item.state in [MemoryState.MAINTENANCE, MemoryState.ENCODING] and
            self.memory_item.strength > self.config.retrieval_threshold):
            
            self.memory_item.state = MemoryState.RETRIEVAL
            return self.neural_activity.copy()
        else:
            return None
            
    def clear_buffer(self):
        """Clear buffer contents."""
        self.memory_item = MemoryItem(content=np.zeros(self.item_size))
        self.neural_activity = np.zeros(self.item_size)
        
    def get_buffer_info(self) -> Dict[str, Any]:
        """Get information about buffer state."""
        return {
            'buffer_id': self.buffer_id,
            'state': self.memory_item.state.value,
            'strength': self.memory_item.strength,
            'age': self.memory_item.age,
            'interference_level': self.memory_item.interference_level,
            'activity_magnitude': np.mean(np.abs(self.neural_activity)),
            'last_refresh_time': self.memory_item.last_refresh_time
        }


class AttentionController:
    """
    Controls attention allocation across working memory buffers.
    """
    
    def __init__(self, config: WorkingMemoryConfig):
        """Initialize attention controller."""
        self.config = config
        
        # Attention state
        self.attention_weights = {}
        self.attention_mode = AttentionMode.AUTOMATIC
        self.focused_buffer = None
        self.attention_history = []
        
        print(f"AttentionController initialized with capacity {config.attention_capacity}")
        
    def allocate_attention(self, buffer_states: Dict[int, Dict[str, Any]]) -> Dict[int, float]:
        """Allocate attention across buffers based on their states and priorities."""
        
        active_buffers = {
            buf_id: state for buf_id, state in buffer_states.items()
            if state['state'] not in ['empty', 'decay']
        }
        
        if not active_buffers:
            return {}
            
        attention_allocation = {}
        
        if self.attention_mode == AttentionMode.FOCUSED and self.focused_buffer is not None:
            # Focus all attention on single buffer
            if self.focused_buffer in active_buffers:
                attention_allocation[self.focused_buffer] = self.config.attention_capacity
                
        elif self.attention_mode == AttentionMode.DIVIDED:
            # Divide attention equally among active buffers
            attention_per_buffer = self.config.attention_capacity / len(active_buffers)
            for buf_id in active_buffers:
                attention_allocation[buf_id] = attention_per_buffer
                
        elif self.attention_mode == AttentionMode.AUTOMATIC:
            # Allocate based on buffer priorities and states
            total_priority = sum(
                self._compute_buffer_priority(state) for state in active_buffers.values()
            )
            
            if total_priority > 0:
                for buf_id, state in active_buffers.items():
                    priority = self._compute_buffer_priority(state)
                    attention_weight = (priority / total_priority) * self.config.attention_capacity
                    attention_allocation[buf_id] = attention_weight
                    
        elif self.attention_mode == AttentionMode.REFRESH:
            # Prioritize buffers that need refreshing
            refresh_priorities = {}
            for buf_id, state in active_buffers.items():
                time_since_refresh = state['age'] - state.get('last_refresh_time', 0)
                refresh_priority = time_since_refresh * (1.0 - state['strength'])
                refresh_priorities[buf_id] = refresh_priority
                
            total_refresh_priority = sum(refresh_priorities.values())
            if total_refresh_priority > 0:
                for buf_id, priority in refresh_priorities.items():
                    attention_weight = (priority / total_refresh_priority) * self.config.attention_capacity
                    attention_allocation[buf_id] = attention_weight
                    
        # Store attention allocation
        self.attention_weights = attention_allocation
        self.attention_history.append({
            'mode': self.attention_mode.value,
            'allocation': attention_allocation.copy(),
            'focused_buffer': self.focused_buffer
        })
        
        return attention_allocation
        
    def _compute_buffer_priority(self, buffer_state: Dict[str, Any]) -> float:
        """Compute priority for attention allocation."""
        base_priority = 1.0
        
        # Prioritize encoding and maintenance states
        if buffer_state['state'] == 'encoding':
            base_priority *= 2.0
        elif buffer_state['state'] == 'maintenance':
            base_priority *= 1.0
        elif buffer_state['state'] == 'retrieval':
            base_priority *= 1.5
            
        # Prioritize weaker items (need more attention)
        strength_factor = max(0.1, 1.0 - buffer_state['strength'])
        base_priority *= strength_factor
        
        # Prioritize items with high interference
        interference_factor = 1.0 + buffer_state.get('interference_level', 0.0)
        base_priority *= interference_factor
        
        return base_priority
        
    def set_attention_mode(self, mode: AttentionMode, focused_buffer: Optional[int] = None):
        """Set attention mode and optionally focused buffer."""
        self.attention_mode = mode
        self.focused_buffer = focused_buffer
        
    def get_attention_info(self) -> Dict[str, Any]:
        """Get information about current attention state."""
        return {
            'mode': self.attention_mode.value,
            'focused_buffer': self.focused_buffer,
            'current_allocation': self.attention_weights.copy(),
            'total_attention': sum(self.attention_weights.values())
        }


class WorkingMemoryNetwork:
    """
    Complete working memory network with multiple buffers, attention control,
    and capacity limitations.
    """
    
    def __init__(self, config: WorkingMemoryConfig, item_size: int = 32):
        """Initialize working memory network."""
        self.config = config
        self.item_size = item_size
        
        # Create memory buffers
        self.buffers = {}
        for i in range(config.capacity):
            self.buffers[i] = WorkingMemoryBuffer(i, item_size, config)
            
        # Create attention controller
        self.attention_controller = AttentionController(config)
        
        # Network state
        self.global_time = 0.0
        self.operation_history = []
        self.performance_metrics = {}
        
        print(f"WorkingMemoryNetwork initialized: {config.capacity} buffers, item size {item_size}")
        
    def encode_item(self, item_content: np.ndarray, priority: float = 1.0) -> Optional[int]:
        """Encode new item into available buffer."""
        
        # Find available buffer
        available_buffers = [
            buf_id for buf_id, buffer in self.buffers.items()
            if buffer.memory_item.state == MemoryState.EMPTY or 
               buffer.memory_item.strength < self.config.encoding_threshold
        ]
        
        if not available_buffers:
            # No available buffers - capacity limit reached
            # Could implement replacement strategy here
            return None
            
        # Choose buffer (could be priority-based)
        chosen_buffer = available_buffers[0]  # Simple FIFO for now
        
        # Encode item
        success = self.buffers[chosen_buffer].encode_item(item_content, priority)
        
        if success:
            # Log operation
            self.operation_history.append({
                'time': self.global_time,
                'operation': 'encode',
                'buffer_id': chosen_buffer,
                'success': True
            })
            return chosen_buffer
        else:
            return None
            
    def retrieve_item(self, buffer_id: int) -> Optional[np.ndarray]:
        """Retrieve item from specific buffer."""
        if buffer_id not in self.buffers:
            return None
            
        retrieved_item = self.buffers[buffer_id].retrieve_item()
        
        # Log operation
        self.operation_history.append({
            'time': self.global_time,
            'operation': 'retrieve',
            'buffer_id': buffer_id,
            'success': retrieved_item is not None
        })
        
        return retrieved_item
        
    def refresh_buffer(self, buffer_id: int, refresh_strength: float = 1.0):
        """Actively refresh specific buffer."""
        if buffer_id in self.buffers:
            self.buffers[buffer_id].refresh_item(refresh_strength)
            
            # Log operation
            self.operation_history.append({
                'time': self.global_time,
                'operation': 'refresh',
                'buffer_id': buffer_id,
                'strength': refresh_strength
            })
            
    def update_network(self, dt: float = 0.001):
        """Update entire working memory network."""
        self.global_time += dt
        
        # Get buffer states
        buffer_states = {
            buf_id: buffer.get_buffer_info() 
            for buf_id, buffer in self.buffers.items()
        }
        
        # Allocate attention
        attention_allocation = self.attention_controller.allocate_attention(buffer_states)
        
        # Update each buffer
        for buf_id, buffer in self.buffers.items():
            attention_weight = attention_allocation.get(buf_id, 0.0)
            buffer.maintain_activity(dt, attention_weight)
            
        # Apply interference between buffers
        self._apply_inter_buffer_interference()
        
        # Update performance metrics
        self._update_performance_metrics()
        
    def _apply_inter_buffer_interference(self):
        """Apply interference between active buffers."""
        active_buffers = [
            (buf_id, buffer) for buf_id, buffer in self.buffers.items()
            if buffer.memory_item.state in [MemoryState.MAINTENANCE, MemoryState.ENCODING]
        ]
        
        # Apply pairwise interference
        for i, (buf_id1, buffer1) in enumerate(active_buffers):
            for j, (buf_id2, buffer2) in enumerate(active_buffers):
                if i != j:  # Don't interfere with self
                    buffer1.apply_interference(buffer2.neural_activity)
                    
    def _update_performance_metrics(self):
        """Update performance metrics."""
        active_items = sum(
            1 for buffer in self.buffers.values()
            if buffer.memory_item.state in [MemoryState.MAINTENANCE, MemoryState.ENCODING]
        )
        
        total_strength = sum(
            buffer.memory_item.strength for buffer in self.buffers.values()
            if buffer.memory_item.state != MemoryState.EMPTY
        )
        
        average_interference = np.mean([
            buffer.memory_item.interference_level for buffer in self.buffers.values()
            if buffer.memory_item.state != MemoryState.EMPTY
        ]) if active_items > 0 else 0.0
        
        self.performance_metrics = {
            'active_items': active_items,
            'capacity_utilization': active_items / self.config.capacity,
            'total_strength': total_strength,
            'average_interference': average_interference,
            'attention_allocation': self.attention_controller.attention_weights.copy()
        }
        
    def set_attention_mode(self, mode: AttentionMode, focused_buffer: Optional[int] = None):
        """Set attention mode for the network."""
        self.attention_controller.set_attention_mode(mode, focused_buffer)
        
    def get_network_state(self) -> Dict[str, Any]:
        """Get comprehensive network state information."""
        buffer_states = {
            buf_id: buffer.get_buffer_info() 
            for buf_id, buffer in self.buffers.items()
        }
        
        attention_info = self.attention_controller.get_attention_info()
        
        return {
            'global_time': self.global_time,
            'buffer_states': buffer_states,
            'attention_info': attention_info,
            'performance_metrics': self.performance_metrics.copy(),
            'config': {
                'capacity': self.config.capacity,
                'decay_rate': self.config.decay_rate,
                'interference_strength': self.config.interference_strength
            }
        }
        
    def clear_all_buffers(self):
        """Clear all memory buffers."""
        for buffer in self.buffers.values():
            buffer.clear_buffer()
            
        # Log operation
        self.operation_history.append({
            'time': self.global_time,
            'operation': 'clear_all'
        })


def demo_working_memory_system():
    """Demonstrate the working memory system functionality."""
    
    print("=== Working Memory System Demo ===")
    
    # Create working memory network
    print("\n1. Creating Working Memory Network")
    
    config = WorkingMemoryConfig(
        capacity=5,
        decay_rate=0.02,
        interference_strength=0.1,
        refresh_rate=0.1,
        attention_capacity=2.0
    )
    
    wm_network = WorkingMemoryNetwork(config, item_size=16)
    
    print(f"Created network with {config.capacity} buffers")
    
    # Test encoding multiple items
    print("\n2. Testing Item Encoding")
    
    # Create test items
    items = [
        np.array([1.0, 0.8, 0.2, 0.1] * 4),      # Item 1
        np.array([0.2, 1.0, 0.9, 0.1] * 4),      # Item 2  
        np.array([0.1, 0.2, 1.0, 0.8] * 4),      # Item 3
        np.array([0.8, 0.1, 0.2, 1.0] * 4),      # Item 4
    ]
    
    encoded_buffers = []
    for i, item in enumerate(items):
        buffer_id = wm_network.encode_item(item, priority=1.0)
        if buffer_id is not None:
            encoded_buffers.append(buffer_id)
            print(f"  Encoded item {i+1} in buffer {buffer_id}")
        else:
            print(f"  Failed to encode item {i+1} (capacity limit)")
            
    # Test maintenance over time
    print("\n3. Testing Memory Maintenance")
    
    # Set automatic attention mode
    wm_network.set_attention_mode(AttentionMode.AUTOMATIC)
    
    # Simulate time passage
    maintenance_steps = 100
    for step in range(maintenance_steps):
        wm_network.update_network(dt=0.01)  # 10ms steps
        
        if step % 25 == 0:  # Print every 250ms
            state = wm_network.get_network_state()
            active_items = state['performance_metrics']['active_items']
            avg_strength = state['performance_metrics']['total_strength'] / max(1, active_items)
            print(f"  t={step*10}ms: {active_items} active items, avg strength: {avg_strength:.3f}")
            
    # Test focused attention
    print("\n4. Testing Focused Attention")
    
    if encoded_buffers:
        focus_buffer = encoded_buffers[0]
        wm_network.set_attention_mode(AttentionMode.FOCUSED, focus_buffer)
        
        print(f"Focusing attention on buffer {focus_buffer}")
        
        # Get state before focused attention
        state_before = wm_network.get_network_state()
        strength_before = state_before['buffer_states'][focus_buffer]['strength']
        
        # Apply focused attention for some time
        for _ in range(50):
            wm_network.update_network(dt=0.01)
            
        state_after = wm_network.get_network_state()
        strength_after = state_after['buffer_states'][focus_buffer]['strength']
        
        print(f"  Buffer {focus_buffer} strength: {strength_before:.3f} → {strength_after:.3f}")
        
    # Test active refresh
    print("\n5. Testing Active Refresh")
    
    if encoded_buffers:
        refresh_buffer = encoded_buffers[1] if len(encoded_buffers) > 1 else encoded_buffers[0]
        
        # Let item decay first
        wm_network.set_attention_mode(AttentionMode.DIVIDED)
        for _ in range(100):
            wm_network.update_network(dt=0.01)
            
        # Get strength before refresh
        state_before_refresh = wm_network.get_network_state()
        strength_before_refresh = state_before_refresh['buffer_states'][refresh_buffer]['strength']
        
        # Apply refresh
        wm_network.refresh_buffer(refresh_buffer, refresh_strength=1.5)
        
        # Update a bit more
        for _ in range(10):
            wm_network.update_network(dt=0.01)
            
        state_after_refresh = wm_network.get_network_state()
        strength_after_refresh = state_after_refresh['buffer_states'][refresh_buffer]['strength']
        
        print(f"  Buffer {refresh_buffer} refresh: {strength_before_refresh:.3f} → {strength_after_refresh:.3f}")
        
    # Test retrieval
    print("\n6. Testing Item Retrieval")
    
    for buffer_id in encoded_buffers:
        retrieved = wm_network.retrieve_item(buffer_id)
        if retrieved is not None:
            similarity = np.dot(retrieved, items[buffer_id]) / (np.linalg.norm(retrieved) * np.linalg.norm(items[buffer_id]))
            print(f"  Retrieved from buffer {buffer_id}, similarity to original: {similarity:.3f}")
        else:
            print(f"  Failed to retrieve from buffer {buffer_id}")
            
    # Test interference effects
    print("\n7. Testing Interference Effects")
    
    # Clear network and encode similar items
    wm_network.clear_all_buffers()
    
    # Create similar items (should interfere more)
    similar_items = [
        np.array([1.0, 0.9, 0.8, 0.1] * 4),
        np.array([0.9, 1.0, 0.7, 0.2] * 4),      # Similar to first
        np.array([0.1, 0.2, 0.1, 1.0] * 4),      # Different
    ]
    
    for i, item in enumerate(similar_items):
        wm_network.encode_item(item)
        
    # Run for a while and measure interference
    wm_network.set_attention_mode(AttentionMode.DIVIDED)
    for _ in range(150):
        wm_network.update_network(dt=0.01)
        
    final_state = wm_network.get_network_state()
    
    print("Final interference levels:")
    for buf_id, buf_state in final_state['buffer_states'].items():
        if buf_state['state'] != 'empty':
            interference = buf_state['interference_level']
            print(f"  Buffer {buf_id}: interference level {interference:.3f}")
            
    print("\n✅ Working Memory System Demo Complete!")
    
    return wm_network


if __name__ == "__main__":
    # Run demonstration
    wm_network = demo_working_memory_system()
    
    print("\n=== Task 7 Implementation Summary ===")
    print("✅ Working Memory System Implementation - COMPLETED")
    print("\nKey Features Implemented:")
    print("  • WorkingMemoryBuffer with persistent activity maintenance")
    print("  • Limited capacity network (3-7 items) with interference effects")
    print("  • AttentionController with multiple attention modes")
    print("  • Decay and refreshing mechanisms for memory persistence")
    print("  • Inter-buffer interference based on content similarity")
    print("  • Attention-based control for memory management")
    
    print("\nNext Steps:")
    print("  → Task 8: Attention Mechanism Implementation")
    print("  → Task 9: Developmental Plasticity and Critical Periods")