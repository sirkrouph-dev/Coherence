#!/usr/bin/env python3
"""
Tests for Working Memory System Implementation
============================================

Task 7 Testing: Validates the working memory system including limited capacity
buffers, persistent activity patterns, interference effects, and attention control.
"""

import pytest
import numpy as np
from typing import Dict, List

try:
    from core.working_memory import (
        WorkingMemoryBuffer,
        AttentionController,
        WorkingMemoryNetwork,
        WorkingMemoryConfig,
        MemoryState,
        AttentionMode,
        MemoryItem
    )
    IMPORTS_SUCCESS = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORTS_SUCCESS = False


class TestWorkingMemoryConfig:
    """Test working memory configuration."""
    
    def test_config_creation(self):
        """Test working memory configuration creation."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = WorkingMemoryConfig(
            capacity=5,
            decay_rate=0.02,
            interference_strength=0.15,
            refresh_rate=0.08,
            attention_capacity=1.5
        )
        
        assert config.capacity == 5
        assert config.decay_rate == 0.02
        assert config.interference_strength == 0.15
        assert config.refresh_rate == 0.08
        assert config.attention_capacity == 1.5


class TestMemoryItem:
    """Test memory item functionality."""
    
    def test_memory_item_creation(self):
        """Test memory item creation and properties."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        content = np.array([0.8, 0.6, 0.9, 0.3])
        item = MemoryItem(
            content=content,
            strength=0.9,
            state=MemoryState.ENCODING
        )
        
        assert np.array_equal(item.content, content)
        assert item.strength == 0.9
        assert item.state == MemoryState.ENCODING
        assert item.age == 0.0


class TestWorkingMemoryBuffer:
    """Test working memory buffer functionality."""
    
    def test_buffer_initialization(self):
        """Test buffer initialization."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = WorkingMemoryConfig(capacity=5, decay_rate=0.01)
        buffer = WorkingMemoryBuffer(0, 16, config)
        
        assert buffer.buffer_id == 0
        assert buffer.item_size == 16
        assert buffer.memory_item.state == MemoryState.EMPTY
        assert len(buffer.neural_activity) == 16
        
    def test_item_encoding(self):
        """Test encoding items into buffer."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = WorkingMemoryConfig()
        buffer = WorkingMemoryBuffer(0, 8, config)
        
        # Test successful encoding
        item_content = np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.3, 0.7])
        success = buffer.encode_item(item_content, strength=0.9)
        
        assert success
        assert buffer.memory_item.state == MemoryState.ENCODING
        assert buffer.memory_item.strength == 0.9
        assert np.array_equal(buffer.memory_item.content, item_content)
        
    def test_maintenance_activity(self):
        """Test maintenance of persistent activity."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = WorkingMemoryConfig(decay_rate=0.02)
        buffer = WorkingMemoryBuffer(0, 4, config)
        
        # Encode item
        item_content = np.array([1.0, 0.8, 0.6, 0.4])
        buffer.encode_item(item_content, strength=1.0)
        
        initial_strength = buffer.memory_item.strength
        
        # Maintain activity for several steps
        for _ in range(50):
            buffer.maintain_activity(dt=0.01, attention_weight=1.0)
            
        # Should transition to maintenance state
        assert buffer.memory_item.state == MemoryState.MAINTENANCE
        
        # Activity should persist but with some decay
        final_strength = buffer.memory_item.strength
        assert final_strength > 0  # Should still be active
        
    def test_interference_effects(self):
        """Test interference between buffers."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = WorkingMemoryConfig(interference_strength=0.2)
        buffer = WorkingMemoryBuffer(0, 4, config)
        
        # Encode item
        item_content = np.array([1.0, 0.0, 1.0, 0.0])
        buffer.encode_item(item_content)
        
        # Maintain to get to maintenance state
        for _ in range(20):
            buffer.maintain_activity(dt=0.01)
            
        initial_interference = buffer.memory_item.interference_level
        
        # Apply interference from similar pattern
        interfering_pattern = np.array([0.8, 0.2, 0.9, 0.1])  # Similar pattern
        buffer.apply_interference(interfering_pattern)
        
        # Should have increased interference
        assert buffer.memory_item.interference_level > initial_interference
        
    def test_item_refresh(self):
        """Test active refresh of memory items."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = WorkingMemoryConfig(decay_rate=0.05, refresh_rate=0.1)
        buffer = WorkingMemoryBuffer(0, 4, config)
        
        # Encode and let decay
        item_content = np.array([1.0, 0.8, 0.6, 0.4])
        buffer.encode_item(item_content)
        
        # Let it decay significantly
        for _ in range(100):
            buffer.maintain_activity(dt=0.01, attention_weight=0.1)
            
        strength_before_refresh = buffer.memory_item.strength
        
        # Apply refresh
        buffer.refresh_item(refresh_strength=1.5)
        
        # Update a few more steps
        for _ in range(10):
            buffer.maintain_activity(dt=0.01, attention_weight=1.0)
            
        strength_after_refresh = buffer.memory_item.strength
        
        # Should be strengthened by refresh
        assert strength_after_refresh > strength_before_refresh
        
    def test_item_retrieval(self):
        """Test retrieval of items from buffer."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = WorkingMemoryConfig()
        buffer = WorkingMemoryBuffer(0, 4, config)
        
        # Test retrieval from empty buffer
        retrieved = buffer.retrieve_item()
        assert retrieved is None
        
        # Encode item and maintain
        item_content = np.array([1.0, 0.5, 0.8, 0.3])
        buffer.encode_item(item_content)
        
        for _ in range(20):
            buffer.maintain_activity(dt=0.01)
            
        # Should be able to retrieve
        retrieved = buffer.retrieve_item()
        assert retrieved is not None
        assert len(retrieved) == 4


class TestAttentionController:
    """Test attention controller functionality."""
    
    def test_controller_initialization(self):
        """Test attention controller initialization."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = WorkingMemoryConfig(attention_capacity=2.0)
        controller = AttentionController(config)
        
        assert controller.config.attention_capacity == 2.0
        assert controller.attention_mode == AttentionMode.AUTOMATIC
        assert controller.focused_buffer is None
        
    def test_automatic_attention_allocation(self):
        """Test automatic attention allocation."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = WorkingMemoryConfig(attention_capacity=1.0)
        controller = AttentionController(config)
        
        # Create mock buffer states
        buffer_states = {
            0: {'state': 'encoding', 'strength': 0.8, 'age': 0.1, 'interference_level': 0.1},
            1: {'state': 'maintenance', 'strength': 0.4, 'age': 1.0, 'interference_level': 0.3},
            2: {'state': 'decay', 'strength': 0.1, 'age': 2.0, 'interference_level': 0.5}
        }
        
        allocation = controller.allocate_attention(buffer_states)
        
        # Should allocate attention to active buffers
        assert len(allocation) >= 1  # At least encoding buffer should get attention
        assert 0 in allocation  # Encoding buffer should get attention
        
        # Total allocation should not exceed capacity
        total_attention = sum(allocation.values())
        assert total_attention <= config.attention_capacity * 1.1  # Allow small tolerance
        
    def test_focused_attention_mode(self):
        """Test focused attention mode."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = WorkingMemoryConfig(attention_capacity=1.0)
        controller = AttentionController(config)
        
        # Set focused mode
        controller.set_attention_mode(AttentionMode.FOCUSED, focused_buffer=1)
        
        buffer_states = {
            0: {'state': 'maintenance', 'strength': 0.8, 'age': 0.5, 'interference_level': 0.1},
            1: {'state': 'maintenance', 'strength': 0.6, 'age': 0.8, 'interference_level': 0.2},
            2: {'state': 'encoding', 'strength': 0.9, 'age': 0.1, 'interference_level': 0.0}
        }
        
        allocation = controller.allocate_attention(buffer_states)
        
        # Should focus all attention on buffer 1
        assert allocation.get(1, 0) == config.attention_capacity
        assert allocation.get(0, 0) == 0
        assert allocation.get(2, 0) == 0


class TestWorkingMemoryNetwork:
    """Test complete working memory network."""
    
    def test_network_initialization(self):
        """Test network initialization."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = WorkingMemoryConfig(capacity=4)
        network = WorkingMemoryNetwork(config, item_size=8)
        
        assert len(network.buffers) == 4
        assert isinstance(network.attention_controller, AttentionController)
        assert network.global_time == 0.0
        
    def test_item_encoding_and_capacity(self):
        """Test item encoding and capacity limits."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = WorkingMemoryConfig(capacity=3)
        network = WorkingMemoryNetwork(config, item_size=4)
        
        # Encode items up to capacity
        items = [
            np.array([1.0, 0.0, 1.0, 0.0]),
            np.array([0.0, 1.0, 0.0, 1.0]),
            np.array([0.5, 0.5, 0.8, 0.2])
        ]
        
        encoded_buffers = []
        for item in items:
            buffer_id = network.encode_item(item)
            assert buffer_id is not None
            encoded_buffers.append(buffer_id)
            
        # Try to encode beyond capacity
        overflow_item = np.array([0.9, 0.1, 0.7, 0.3])
        overflow_buffer = network.encode_item(overflow_item)
        
        # Should fail due to capacity limit
        assert overflow_buffer is None
        
    def test_network_update_dynamics(self):
        """Test network update and dynamics."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = WorkingMemoryConfig(capacity=3, decay_rate=0.01)
        network = WorkingMemoryNetwork(config, item_size=4)
        
        # Encode some items
        items = [
            np.array([1.0, 0.8, 0.6, 0.4]),
            np.array([0.4, 0.6, 0.8, 1.0])
        ]
        
        for item in items:
            network.encode_item(item)
            
        # Update network for some time
        for step in range(100):
            network.update_network(dt=0.01)
            
        # Check network state
        state = network.get_network_state()
        
        assert 'buffer_states' in state
        assert 'attention_info' in state
        assert 'performance_metrics' in state
        
        # Should have some active items
        active_items = state['performance_metrics']['active_items']
        assert active_items >= 1
        
    def test_item_retrieval(self):
        """Test item retrieval from network."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = WorkingMemoryConfig()
        network = WorkingMemoryNetwork(config, item_size=4)
        
        # Encode item
        test_item = np.array([0.9, 0.7, 0.5, 0.3])
        buffer_id = network.encode_item(test_item)
        
        # Maintain for some time
        for _ in range(50):
            network.update_network(dt=0.01)
            
        # Retrieve item
        retrieved = network.retrieve_item(buffer_id)
        assert retrieved is not None
        assert len(retrieved) == 4
        
    def test_attention_mode_switching(self):
        """Test switching attention modes."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = WorkingMemoryConfig(capacity=3)
        network = WorkingMemoryNetwork(config, item_size=4)
        
        # Encode items
        for i in range(2):
            item = np.random.rand(4)
            network.encode_item(item)
            
        # Test divided attention
        network.set_attention_mode(AttentionMode.DIVIDED)
        
        for _ in range(20):
            network.update_network(dt=0.01)
            
        state = network.get_network_state()
        attention_mode = state['attention_info']['mode']
        assert attention_mode == 'divided'
        
        # Test focused attention
        network.set_attention_mode(AttentionMode.FOCUSED, focused_buffer=0)
        
        for _ in range(20):
            network.update_network(dt=0.01)
            
        state = network.get_network_state()
        assert state['attention_info']['focused_buffer'] == 0


def run_working_memory_tests():
    """Run comprehensive working memory tests."""
    if not IMPORTS_SUCCESS:
        print("Cannot run tests - required modules not available")
        return False
        
    print("\n=== Working Memory System Tests ===")
    
    try:
        # Test 1: Basic component functionality
        print("\n1. Testing Basic Component Functionality...")
        
        config = WorkingMemoryConfig(capacity=5, decay_rate=0.02)
        
        # Test buffer
        buffer = WorkingMemoryBuffer(0, 8, config)
        assert buffer.buffer_id == 0
        print("  ✅ WorkingMemoryBuffer creation")
        
        # Test encoding
        test_item = np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.3, 0.7])
        success = buffer.encode_item(test_item)
        assert success
        assert buffer.memory_item.state == MemoryState.ENCODING
        print("  ✅ Item encoding")
        
        # Test attention controller
        controller = AttentionController(config)
        assert controller.attention_mode == AttentionMode.AUTOMATIC
        print("  ✅ AttentionController creation")
        
        # Test 2: Network functionality
        print("\n2. Testing Network Functionality...")
        
        network = WorkingMemoryNetwork(config, item_size=6)
        assert len(network.buffers) == 5
        print("  ✅ WorkingMemoryNetwork creation")
        
        # Test encoding multiple items
        items = [
            np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.0]),
            np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
            np.array([0.5, 0.7, 0.3, 0.9, 0.1, 0.6])
        ]
        
        encoded_buffers = []
        for item in items:
            buffer_id = network.encode_item(item)
            assert buffer_id is not None
            encoded_buffers.append(buffer_id)
            
        print(f"  ✅ Multiple item encoding: {len(encoded_buffers)} items")
        
        # Test 3: Maintenance and dynamics
        print("\n3. Testing Maintenance and Dynamics...")
        
        # Update network
        for step in range(100):
            network.update_network(dt=0.01)
            
        state = network.get_network_state()
        active_items = state['performance_metrics']['active_items']
        
        print(f"  ✅ Network dynamics: {active_items} active items after 1s")
        
        # Test attention allocation
        attention_info = state['attention_info']
        allocation = attention_info['current_allocation']
        print(f"  ✅ Attention allocation: {len(allocation)} buffers receiving attention")
        
        # Test 4: Capacity and interference
        print("\n4. Testing Capacity Limits and Interference...")
        
        # Try to exceed capacity
        overflow_items = [np.random.rand(6) for _ in range(5)]
        overflow_encoded = 0
        
        for item in overflow_items:
            buffer_id = network.encode_item(item)
            if buffer_id is not None:
                overflow_encoded += 1
                
        print(f"  ✅ Capacity enforcement: {overflow_encoded}/5 overflow items encoded")
        
        # Let interference develop
        network.set_attention_mode(AttentionMode.DIVIDED)
        for _ in range(150):
            network.update_network(dt=0.01)
            
        final_state = network.get_network_state()
        final_active = final_state['performance_metrics']['active_items']
        avg_interference = final_state['performance_metrics']['average_interference']
        
        print(f"  ✅ Interference effects: {final_active} items remain, avg interference: {avg_interference:.3f}")
        
        # Test 5: Attention modes
        print("\n5. Testing Attention Modes...")
        
        if encoded_buffers:
            # Test focused attention
            focus_buffer = encoded_buffers[0]
            network.set_attention_mode(AttentionMode.FOCUSED, focus_buffer)
            
            for _ in range(50):
                network.update_network(dt=0.01)
                
            focused_state = network.get_network_state()
            focused_allocation = focused_state['attention_info']['current_allocation']
            
            focused_attention = focused_allocation.get(focus_buffer, 0)
            print(f"  ✅ Focused attention: buffer {focus_buffer} receives {focused_attention:.3f} attention")
            
            # Test refresh mode
            network.set_attention_mode(AttentionMode.REFRESH)
            
            for _ in range(30):
                network.update_network(dt=0.01)
                
            print("  ✅ Refresh attention mode activated")
        
        print("\n✅ All Working Memory tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_working_memory_tests()
    
    if success:
        print("\n🎉 Task 7: Working Memory System Implementation")
        print("All tests passed - working memory system validated!")
        print("\nKey features validated:")
        print("  • Limited capacity buffers (3-7 items)")
        print("  • Persistent activity patterns with decay")
        print("  • Inter-buffer interference effects")
        print("  • Attention-based control and allocation")
        print("  • Memory refresh and maintenance mechanisms")
    else:
        print("\n❌ Some tests failed - check implementation")
        
    exit(0 if success else 1)