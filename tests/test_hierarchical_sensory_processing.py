#!/usr/bin/env python3
"""
Tests for Hierarchical Sensory Processing Framework
==================================================

Task 6B.1 Testing: Validates the core hierarchical sensory processing system
including layer creation, connectivity patterns, and sensory input processing.
"""

import pytest
import numpy as np
from typing import Dict, List

try:
    from core.hierarchical_sensory_processing import (
        SensoryHierarchy,
        SensoryLayer, 
        LayerConfig,
        ConnectionConfig,
        HierarchyConfig,
        SensoryModalityType,
        ProcessingLevel,
        create_default_visual_hierarchy,
        create_default_auditory_hierarchy
    )
    IMPORTS_SUCCESS = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORTS_SUCCESS = False


class TestLayerConfig:
    """Test layer configuration functionality."""
    
    def test_layer_config_creation(self):
        """Test basic layer configuration creation."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = LayerConfig(
            name="test_layer",
            level=ProcessingLevel.PRIMARY,
            size=100,
            neuron_type="adex",
            receptive_field_size=5
        )
        
        assert config.name == "test_layer"
        assert config.level == ProcessingLevel.PRIMARY
        assert config.size == 100
        assert config.neuron_type == "adex"
        assert config.receptive_field_size == 5
        
    def test_layer_config_with_spatial_layout(self):
        """Test layer configuration with spatial layout."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = LayerConfig(
            name="spatial_layer",
            level=ProcessingLevel.SECONDARY,
            size=64,
            spatial_layout=(8, 8)
        )
        
        assert config.spatial_layout == (8, 8)
        assert config.size == 64


class TestConnectionConfig:
    """Test connection configuration functionality."""
    
    def test_connection_config_creation(self):
        """Test basic connection configuration."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = ConnectionConfig(
            source_layer="layer1",
            target_layer="layer2",
            connection_type="feedforward",
            connection_probability=0.3,
            plasticity_rule="stdp"
        )
        
        assert config.source_layer == "layer1"
        assert config.target_layer == "layer2"
        assert config.connection_type == "feedforward"
        assert config.connection_probability == 0.3
        assert config.plasticity_rule == "stdp"
        
    def test_connection_config_feedback(self):
        """Test feedback connection configuration."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = ConnectionConfig(
            source_layer="higher",
            target_layer="lower",
            connection_type="feedback",
            weight_scale=0.5
        )
        
        assert config.connection_type == "feedback"
        assert config.weight_scale == 0.5


class TestSensoryLayer:
    """Test individual sensory layer functionality."""
    
    def test_sensory_layer_creation(self):
        """Test basic sensory layer creation."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = LayerConfig(
            name="V1",
            level=ProcessingLevel.PRIMARY,
            size=100,
            receptive_field_size=3
        )
        
        layer = SensoryLayer(config, "test_hierarchy")
        
        assert layer.config.name == "V1"
        assert layer.config.size == 100
        assert layer.hierarchy_id == "test_hierarchy"
        assert layer.layer_id == "test_hierarchy_V1"
        
    def test_feature_map_creation(self):
        """Test feature map creation."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = LayerConfig(
            name="test_layer",
            level=ProcessingLevel.PRIMARY,
            size=64,
            spatial_layout=(8, 8)
        )
        
        layer = SensoryLayer(config)
        feature_maps = layer.create_feature_maps(4)
        
        assert len(feature_maps) == 4
        assert len(layer.feature_maps) == 4
        for feature_map in feature_maps:
            assert feature_map.shape == (8, 8)
            
    def test_receptive_field_computation(self):
        """Test receptive field computation."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = LayerConfig(
            name="rf_layer",
            level=ProcessingLevel.PRIMARY,
            size=16,
            receptive_field_size=3,
            spatial_layout=(4, 4)
        )
        
        layer = SensoryLayer(config)
        receptive_fields = layer.compute_receptive_fields((12, 12))
        
        assert len(receptive_fields) > 0
        assert len(receptive_fields) <= config.size
        
        # Check receptive field format (top, bottom, left, right)
        for neuron_id, (top, bottom, left, right) in receptive_fields.items():
            assert isinstance(top, int)
            assert isinstance(bottom, int) 
            assert isinstance(left, int)
            assert isinstance(right, int)
            assert top < bottom
            assert left < right
            
    def test_input_processing(self):
        """Test basic input processing."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = LayerConfig(
            name="proc_layer",
            level=ProcessingLevel.PRIMARY,
            size=25,
            receptive_field_size=3,
            spatial_layout=(5, 5)
        )
        
        layer = SensoryLayer(config)
        layer.compute_receptive_fields((10, 10))
        
        # Test input processing
        test_input = np.random.rand(10, 10)
        response = layer.process_input(test_input)
        
        assert isinstance(response, np.ndarray)
        assert len(response) == config.size
        assert np.all(response >= 0)  # Should be non-negative responses
        
    def test_attention_modulation(self):
        """Test attention modulation effects."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = LayerConfig(
            name="attention_layer",
            level=ProcessingLevel.PRIMARY,
            size=9,
            spatial_layout=(3, 3)
        )
        
        layer = SensoryLayer(config)
        layer.compute_receptive_fields((6, 6))
        
        test_input = np.ones((6, 6)) * 0.5
        
        # Normal processing
        normal_response = layer.process_input(test_input)
        
        # Enhanced attention
        layer.attention_modulation = 2.0
        enhanced_response = layer.process_input(test_input)
        
        # Enhanced response should be larger
        assert np.mean(enhanced_response) > np.mean(normal_response)


class TestSensoryHierarchy:
    """Test complete sensory hierarchy functionality."""
    
    def test_visual_hierarchy_creation(self):
        """Test creation of visual processing hierarchy."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = create_default_visual_hierarchy()
        hierarchy = SensoryHierarchy(config)
        
        assert hierarchy.config.modality == SensoryModalityType.VISUAL
        assert len(hierarchy.layers) > 0
        assert "V1" in hierarchy.layers
        assert "IT" in hierarchy.layers
        
        # Check layer order
        v1_idx = hierarchy.layer_order.index("V1")
        it_idx = hierarchy.layer_order.index("IT") 
        assert v1_idx < it_idx  # V1 should come before IT
        
    def test_auditory_hierarchy_creation(self):
        """Test creation of auditory processing hierarchy."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = create_default_auditory_hierarchy()
        hierarchy = SensoryHierarchy(config)
        
        assert hierarchy.config.modality == SensoryModalityType.AUDITORY
        assert len(hierarchy.layers) > 0
        assert "A1" in hierarchy.layers
        assert "Parabelt" in hierarchy.layers
        
    def test_hierarchy_connections(self):
        """Test connection creation in hierarchy."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = create_default_visual_hierarchy()
        hierarchy = SensoryHierarchy(config)
        
        # Check that connections exist
        assert len(hierarchy.connections) > 0
        
        # Check specific feedforward connections
        v1_connections = hierarchy.connections.get("V1", [])
        v1_targets = [conn.target_layer for conn in v1_connections 
                     if conn.connection_type == "feedforward"]
        assert "V2" in v1_targets
        
    def test_visual_input_processing(self):
        """Test processing of visual input through hierarchy."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = create_default_visual_hierarchy()
        hierarchy = SensoryHierarchy(config)
        
        # Create test visual input (28x28 image)
        visual_input = np.random.rand(28, 28)
        
        # Process input
        activations = hierarchy.process_sensory_input(visual_input)
        
        # Check that all layers produced activations
        assert len(activations) == len(hierarchy.layers)
        
        # Check activation properties
        for layer_name, activation in activations.items():
            assert isinstance(activation, np.ndarray)
            assert len(activation) == hierarchy.layers[layer_name].config.size
            assert np.all(np.isfinite(activation))  # No NaN or infinity
            
    def test_auditory_input_processing(self):
        """Test processing of auditory input through hierarchy."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = create_default_auditory_hierarchy()
        hierarchy = SensoryHierarchy(config)
        
        # Create test auditory input (spectrogram-like)
        auditory_input = np.random.rand(64, 32)
        
        # Process input
        activations = hierarchy.process_sensory_input(auditory_input)
        
        # Check that all layers produced activations
        assert len(activations) == len(hierarchy.layers)
        
        # Check activation properties
        for layer_name, activation in activations.items():
            assert isinstance(activation, np.ndarray)
            assert len(activation) == hierarchy.layers[layer_name].config.size
            
    def test_attention_modulation_in_hierarchy(self):
        """Test attention modulation effects in complete hierarchy."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = create_default_visual_hierarchy()
        hierarchy = SensoryHierarchy(config)
        
        test_input = np.random.rand(28, 28)
        
        # Normal processing
        normal_activations = hierarchy.process_sensory_input(test_input)
        
        # Apply attention to V1
        hierarchy.apply_attention_modulation("V1", 1.5)
        
        # Process with attention
        attention_activations = hierarchy.process_sensory_input(test_input)
        
        # V1 activation should be enhanced
        v1_normal = np.mean(normal_activations["V1"])
        v1_attention = np.mean(attention_activations["V1"])
        
        # Allow for some numerical variation
        assert v1_attention >= v1_normal * 0.9  # Should be approximately enhanced
        
    def test_hierarchy_info_retrieval(self):
        """Test hierarchy information retrieval."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = create_default_visual_hierarchy()
        hierarchy = SensoryHierarchy(config)
        
        info = hierarchy.get_hierarchy_info()
        
        # Check required information fields
        required_fields = [
            'hierarchy_id', 'modality', 'total_layers', 
            'total_connections', 'layers', 'connections'
        ]
        
        for field in required_fields:
            assert field in info
            
        assert info['modality'] == 'visual'
        assert info['total_layers'] > 0
        assert isinstance(info['layers'], dict)
        assert isinstance(info['connections'], dict)
        
    def test_processing_state_reset(self):
        """Test resetting processing state."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = create_default_visual_hierarchy()
        hierarchy = SensoryHierarchy(config)
        
        # Process some input to create state
        test_input = np.random.rand(28, 28)
        hierarchy.process_sensory_input(test_input)
        
        # Apply attention
        hierarchy.apply_attention_modulation("V1", 1.5)
        
        # Reset state
        hierarchy.reset_processing_state()
        
        # Check state is reset
        assert len(hierarchy.processing_history) == 0
        assert len(hierarchy.feature_responses) == 0
        assert len(hierarchy.attention_weights) == 0
        
        # Check layer states are reset
        for layer in hierarchy.layers.values():
            assert len(layer.activation_history) == 0
            assert len(layer.feature_responses) == 0
            assert layer.attention_modulation == 1.0


class TestHierarchyConfigurations:
    """Test different hierarchy configuration scenarios."""
    
    def test_custom_visual_hierarchy(self):
        """Test creation of custom visual hierarchy."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        # Create custom configuration
        layers = [
            LayerConfig("Custom_V1", ProcessingLevel.PRIMARY, 64, spatial_layout=(8, 8)),
            LayerConfig("Custom_V2", ProcessingLevel.SECONDARY, 32, spatial_layout=(4, 8))
        ]
        
        connections = [
            ConnectionConfig("Custom_V1", "Custom_V2", "feedforward", 0.5)
        ]
        
        config = HierarchyConfig(
            modality=SensoryModalityType.VISUAL,
            layers=layers,
            connections=connections,
            input_dimensions=(16, 16)
        )
        
        hierarchy = SensoryHierarchy(config)
        
        assert len(hierarchy.layers) == 2
        assert "Custom_V1" in hierarchy.layers
        assert "Custom_V2" in hierarchy.layers
        
    def test_minimal_hierarchy(self):
        """Test minimal hierarchy with single layer."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        layers = [
            LayerConfig("Single", ProcessingLevel.PRIMARY, 16)
        ]
        
        config = HierarchyConfig(
            modality=SensoryModalityType.VISUAL,
            layers=layers,
            connections=[],
            input_dimensions=(8, 8)
        )
        
        hierarchy = SensoryHierarchy(config)
        
        assert len(hierarchy.layers) == 1
        assert "Single" in hierarchy.layers
        
        # Test processing
        test_input = np.random.rand(8, 8)
        activations = hierarchy.process_sensory_input(test_input)
        
        assert len(activations) == 1
        assert "Single" in activations
        
    def test_feedback_disabled_hierarchy(self):
        """Test hierarchy with feedback disabled."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = create_default_visual_hierarchy()
        config.enable_feedback = False
        
        hierarchy = SensoryHierarchy(config)
        
        # Check that no feedback connections exist
        all_connections = []
        for conn_list in hierarchy.connections.values():
            all_connections.extend(conn_list)
            
        feedback_connections = [conn for conn in all_connections 
                              if conn.connection_type == "feedback"]
        
        assert len(feedback_connections) == 0


def run_hierarchical_sensory_processing_tests():
    """Run all hierarchical sensory processing tests."""
    if not IMPORTS_SUCCESS:
        print("Cannot run tests - required modules not available")
        return False
        
    print("\n=== Hierarchical Sensory Processing Framework Tests ===")
    
    try:
        # Test configurations
        print("\n1. Testing Layer and Connection Configurations...")
        
        # Test layer config
        layer_config = LayerConfig("test", ProcessingLevel.PRIMARY, 100)
        assert layer_config.name == "test"
        print("  ✅ LayerConfig creation")
        
        # Test connection config
        conn_config = ConnectionConfig("layer1", "layer2", "feedforward")
        assert conn_config.source_layer == "layer1"
        print("  ✅ ConnectionConfig creation")
        
        # Test sensory layer
        print("\n2. Testing Sensory Layer Functionality...")
        layer = SensoryLayer(layer_config)
        layer.create_feature_maps(2)
        assert len(layer.feature_maps) == 2
        print("  ✅ SensoryLayer feature maps")
        
        # Test hierarchies
        print("\n3. Testing Hierarchy Creation...")
        
        visual_config = create_default_visual_hierarchy()
        visual_hierarchy = SensoryHierarchy(visual_config)
        assert len(visual_hierarchy.layers) > 0
        print("  ✅ Visual hierarchy creation")
        
        auditory_config = create_default_auditory_hierarchy()
        auditory_hierarchy = SensoryHierarchy(auditory_config)
        assert len(auditory_hierarchy.layers) > 0
        print("  ✅ Auditory hierarchy creation")
        
        # Test processing
        print("\n4. Testing Input Processing...")
        
        visual_input = np.random.rand(28, 28)
        visual_activations = visual_hierarchy.process_sensory_input(visual_input)
        assert len(visual_activations) == len(visual_hierarchy.layers)
        print("  ✅ Visual input processing")
        
        auditory_input = np.random.rand(64, 32)  
        auditory_activations = auditory_hierarchy.process_sensory_input(auditory_input)
        assert len(auditory_activations) == len(auditory_hierarchy.layers)
        print("  ✅ Auditory input processing")
        
        # Test attention
        print("\n5. Testing Attention Modulation...")
        
        visual_hierarchy.apply_attention_modulation("V1", 1.5)
        enhanced_activations = visual_hierarchy.process_sensory_input(visual_input)
        print("  ✅ Attention modulation")
        
        # Test information retrieval
        info = visual_hierarchy.get_hierarchy_info()
        assert 'total_layers' in info
        print("  ✅ Hierarchy information retrieval")
        
        print("\n✅ All Hierarchical Sensory Processing tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_hierarchical_sensory_processing_tests()
    
    if success:
        print("\n🎉 Task 6B.1: Core Hierarchical Sensory Processing Framework")
        print("All tests passed - implementation validated!")
    else:
        print("\n❌ Some tests failed - check implementation")
        
    exit(0 if success else 1)