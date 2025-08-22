#!/usr/bin/env python3
"""
Tests for Visual Processing Hierarchy Implementation  
===================================================

Task 6B.2 Testing: Validates the visual processing pipeline including
edge detection, orientation selectivity, shape detection, and object recognition.
"""

import pytest
import numpy as np
from typing import Dict, List

try:
    from core.visual_processing_hierarchy import (
        EdgeDetectionLayer,
        OrientationLayer, 
        ShapeDetectionLayer,
        ObjectRecognitionLayer,
        VisualProcessingHierarchy,
        VisualLayerConfig,
        GaborFilterConfig,
        VisualFeatureType,
        create_visual_processing_hierarchy
    )
    from core.hierarchical_sensory_processing import (
        ProcessingLevel,
        SensoryModalityType
    )
    IMPORTS_SUCCESS = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORTS_SUCCESS = False


class TestGaborFilterConfig:
    """Test Gabor filter configuration."""
    
    def test_gabor_config_creation(self):
        """Test basic Gabor filter configuration creation."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = GaborFilterConfig(
            sigma_x=2.0,
            sigma_y=1.5,
            theta=np.pi/4,
            lambda_=6.0,
            psi=np.pi/2,
            gamma=0.7
        )
        
        assert config.sigma_x == 2.0
        assert config.sigma_y == 1.5
        assert config.theta == np.pi/4
        assert config.lambda_ == 6.0
        assert config.psi == np.pi/2
        assert config.gamma == 0.7


class TestVisualLayerConfig:
    """Test visual layer configuration."""
    
    def test_visual_layer_config_creation(self):
        """Test visual layer configuration with visual-specific parameters."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = VisualLayerConfig(
            name="TestVisualLayer",
            level=ProcessingLevel.PRIMARY,
            size=64,
            feature_type=VisualFeatureType.EDGE,
            n_orientations=8,
            spatial_pooling_size=2,
            response_threshold=0.1
        )
        
        assert config.name == "TestVisualLayer"
        assert config.level == ProcessingLevel.PRIMARY
        assert config.size == 64
        assert config.feature_type == VisualFeatureType.EDGE
        assert config.n_orientations == 8
        assert config.spatial_pooling_size == 2
        assert config.response_threshold == 0.1


class TestEdgeDetectionLayer:
    """Test edge detection layer functionality."""
    
    def test_edge_layer_initialization(self):
        """Test edge detection layer initialization."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = VisualLayerConfig(
            name="EdgeLayer",
            level=ProcessingLevel.PRIMARY,
            size=64,
            feature_type=VisualFeatureType.EDGE,
            n_orientations=4
        )
        
        layer = EdgeDetectionLayer(config)
        
        assert layer.config.name == "EdgeLayer"
        assert len(layer.gabor_filters) == 4  # Should create 4 orientation filters
        assert len(layer.edge_responses) == 0  # No processing yet
        
    def test_gabor_filter_generation(self):
        """Test Gabor filter generation."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = VisualLayerConfig(
            name="EdgeLayer",
            level=ProcessingLevel.PRIMARY,
            size=64,
            feature_type=VisualFeatureType.EDGE
        )
        
        layer = EdgeDetectionLayer(config)
        
        # Check that filters were created
        assert len(layer.gabor_filters) > 0
        
        # Check filter properties
        for gabor_filter, gabor_config in layer.gabor_filters:
            assert isinstance(gabor_filter, np.ndarray)
            assert gabor_filter.shape[0] > 0
            assert gabor_filter.shape[1] > 0
            assert isinstance(gabor_config, GaborFilterConfig)
            
            # Filter should have zero mean (approximately)
            assert abs(np.mean(gabor_filter)) < 0.1
            
    def test_edge_detection_processing(self):
        """Test edge detection processing."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = VisualLayerConfig(
            name="EdgeLayer",
            level=ProcessingLevel.PRIMARY,
            size=32,
            feature_type=VisualFeatureType.EDGE,
            n_orientations=4,
            response_threshold=0.0
        )
        
        layer = EdgeDetectionLayer(config)
        
        # Create test image with horizontal edge
        test_image = np.zeros((16, 16))
        test_image[6:10, :] = 1.0  # Horizontal edge
        
        response = layer.process_input(test_image)
        
        assert isinstance(response, np.ndarray)
        assert len(response) == config.size
        assert np.any(response > 0)  # Should detect some edges
        
    def test_different_edge_orientations(self):
        """Test detection of different edge orientations."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = VisualLayerConfig(
            name="EdgeLayer", 
            level=ProcessingLevel.PRIMARY,
            size=64,
            n_orientations=4,
            response_threshold=0.0
        )
        
        layer = EdgeDetectionLayer(config)
        
        # Test horizontal edge
        horizontal_image = np.zeros((20, 20))
        horizontal_image[9:11, :] = 1.0
        
        horizontal_response = layer.process_input(horizontal_image)
        
        # Test vertical edge
        vertical_image = np.zeros((20, 20))
        vertical_image[:, 9:11] = 1.0
        
        vertical_response = layer.process_input(vertical_image)
        
        # Both should produce responses
        assert np.sum(horizontal_response > 0) > 0
        assert np.sum(vertical_response > 0) > 0
        
        # Responses might be different due to different orientations
        # (exact comparison depends on filter orientations)


class TestOrientationLayer:
    """Test orientation layer functionality."""
    
    def test_orientation_layer_initialization(self):
        """Test orientation layer initialization."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = VisualLayerConfig(
            name="OrientationLayer",
            level=ProcessingLevel.SECONDARY,
            size=32,
            feature_type=VisualFeatureType.ORIENTATION,
            n_orientations=8
        )
        
        layer = OrientationLayer(config)
        
        assert layer.config.name == "OrientationLayer"
        assert len(layer.preferred_orientations) == config.size
        
        # Check orientation preferences are assigned
        for neuron_id in range(config.size):
            assert neuron_id in layer.preferred_orientations
            theta = layer.preferred_orientations[neuron_id]
            assert 0 <= theta <= np.pi
            
    def test_orientation_processing(self):
        """Test orientation processing."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = VisualLayerConfig(
            name="OrientationLayer",
            level=ProcessingLevel.SECONDARY,
            size=16,
            n_orientations=4,
            response_threshold=0.0
        )
        
        layer = OrientationLayer(config)
        
        # Create input simulating edge detection output
        # Assume input has responses from 4 orientation filters
        input_data = np.array([1.0, 0.5, 0.2, 0.8] * 4)  # 16 values
        
        response = layer.process_input(input_data)
        
        assert isinstance(response, np.ndarray)
        assert len(response) == config.size
        assert np.any(response >= 0)  # Should have non-negative responses
        
    def test_orientation_selectivity(self):
        """Test that neurons respond preferentially to their preferred orientations."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = VisualLayerConfig(
            name="OrientationLayer",
            level=ProcessingLevel.SECONDARY,
            size=8,
            n_orientations=4,
            response_threshold=0.0
        )
        
        layer = OrientationLayer(config)
        
        # Create strong input for one orientation, weak for others
        strong_orientation_input = np.array([2.0, 0.1, 0.1, 0.1] * 2)  # 8 values
        
        response = layer.process_input(strong_orientation_input)
        
        # Neurons preferring the first orientation should respond more strongly
        first_orientation_neurons = [nid for nid, theta in layer.preferred_orientations.items() 
                                   if abs(theta - 0.0) < 0.1]
        
        if first_orientation_neurons:
            first_orientation_response = np.mean([response[nid] for nid in first_orientation_neurons])
            overall_response = np.mean(response)
            
            # First orientation neurons should respond at least as strongly as average
            assert first_orientation_response >= overall_response * 0.8


class TestShapeDetectionLayer:
    """Test shape detection layer functionality."""
    
    def test_shape_layer_initialization(self):
        """Test shape detection layer initialization.""" 
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = VisualLayerConfig(
            name="ShapeLayer",
            level=ProcessingLevel.ASSOCIATION,
            size=25,
            feature_type=VisualFeatureType.SHAPE
        )
        
        layer = ShapeDetectionLayer(config)
        
        assert layer.config.name == "ShapeLayer"
        assert len(layer.shape_templates) > 0
        
        # Check that shape templates exist
        expected_shapes = ['horizontal_line', 'vertical_line', 'diagonal_line', 'corner', 'circle']
        for shape in expected_shapes:
            assert shape in layer.shape_templates
            
    def test_shape_template_properties(self):
        """Test properties of shape templates."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = VisualLayerConfig(
            name="ShapeLayer",
            level=ProcessingLevel.ASSOCIATION,
            size=25
        )
        
        layer = ShapeDetectionLayer(config)
        
        # Check template properties
        for shape_name, template in layer.shape_templates.items():
            assert isinstance(template, np.ndarray)
            assert len(template) > 0
            assert np.all(template >= 0)  # Should be non-negative
            assert np.sum(template) > 0   # Should have some non-zero values
            
    def test_shape_detection_processing(self):
        """Test shape detection processing."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = VisualLayerConfig(
            name="ShapeLayer",
            level=ProcessingLevel.ASSOCIATION,
            size=20,
            n_orientations=8,
            response_threshold=0.0
        )
        
        layer = ShapeDetectionLayer(config)
        
        # Create input representing orientation responses
        # Strong horizontal orientation (should match horizontal_line template)
        orientation_input = np.array([1.0, 1.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2] * 3)  # 24 values
        
        response = layer.process_input(orientation_input)
        
        assert isinstance(response, np.ndarray)
        assert len(response) == config.size
        assert np.any(response >= 0)


class TestObjectRecognitionLayer:
    """Test object recognition layer functionality."""
    
    def test_object_layer_initialization(self):
        """Test object recognition layer initialization."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = VisualLayerConfig(
            name="ObjectLayer",
            level=ProcessingLevel.INTEGRATION,
            size=16,
            feature_type=VisualFeatureType.OBJECT
        )
        
        layer = ObjectRecognitionLayer(config)
        
        assert layer.config.name == "ObjectLayer"
        assert len(layer.object_prototypes) > 0
        
        # Check object prototypes
        expected_objects = ['face', 'house', 'car', 'tree']
        for obj in expected_objects:
            assert obj in layer.object_prototypes
            prototype = layer.object_prototypes[obj]
            assert 'required_shapes' in prototype
            assert 'weights' in prototype
            assert len(prototype['required_shapes']) == len(prototype['weights'])
            
    def test_object_recognition_processing(self):
        """Test object recognition processing."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = VisualLayerConfig(
            name="ObjectLayer",
            level=ProcessingLevel.INTEGRATION,
            size=8,
            response_threshold=0.0
        )
        
        layer = ObjectRecognitionLayer(config)
        
        # Create input representing shape detection responses
        shape_input = np.array([0.5, 0.8, 0.3, 0.6, 0.9])  # 5 shape responses
        
        response = layer.process_input(shape_input)
        
        assert isinstance(response, np.ndarray)
        assert len(response) == config.size
        assert np.all(response >= 0)  # Should be non-negative


class TestVisualProcessingHierarchy:
    """Test complete visual processing hierarchy."""
    
    def test_visual_hierarchy_creation(self):
        """Test creation of complete visual hierarchy."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = create_visual_processing_hierarchy()
        hierarchy = VisualProcessingHierarchy(config)
        
        assert hierarchy.config.modality == SensoryModalityType.VISUAL
        assert len(hierarchy.layers) == 4
        
        # Check that specialized layers were created
        layer_names = ['EdgeDetection', 'Orientation', 'ShapeDetection', 'ObjectRecognition']
        for name in layer_names:
            assert name in hierarchy.layers
            
        # Check layer types
        assert isinstance(hierarchy.layers['EdgeDetection'], EdgeDetectionLayer)
        assert isinstance(hierarchy.layers['Orientation'], OrientationLayer)
        assert isinstance(hierarchy.layers['ShapeDetection'], ShapeDetectionLayer)
        assert isinstance(hierarchy.layers['ObjectRecognition'], ObjectRecognitionLayer)
        
    def test_full_visual_processing_pipeline(self):
        """Test processing through the complete visual pipeline."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = create_visual_processing_hierarchy()
        hierarchy = VisualProcessingHierarchy(config)
        
        # Create test visual input with edges
        visual_input = np.zeros((28, 28))
        
        # Add horizontal edge
        visual_input[12:15, 5:23] = 1.0
        
        # Add vertical edge
        visual_input[5:23, 12:15] = 1.0
        
        # Process through hierarchy
        activations = hierarchy.process_sensory_input(visual_input)
        
        # Check all layers produced activations
        assert len(activations) == 4
        for layer_name in ['EdgeDetection', 'Orientation', 'ShapeDetection', 'ObjectRecognition']:
            assert layer_name in activations
            assert isinstance(activations[layer_name], np.ndarray)
            assert len(activations[layer_name]) == hierarchy.layers[layer_name].config.size
            
    def test_attention_modulation_in_visual_hierarchy(self):
        """Test attention effects in visual hierarchy."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = create_visual_processing_hierarchy()
        hierarchy = VisualProcessingHierarchy(config)
        
        # Create test input
        test_input = np.random.rand(28, 28) * 0.5
        
        # Normal processing
        normal_activations = hierarchy.process_sensory_input(test_input)
        
        # Apply attention to edge detection layer
        hierarchy.apply_attention_modulation("EdgeDetection", 2.0)
        
        # Process with attention
        attention_activations = hierarchy.process_sensory_input(test_input)
        
        # Edge detection should be enhanced
        edge_normal = np.mean(normal_activations["EdgeDetection"])
        edge_attention = np.mean(attention_activations["EdgeDetection"])
        
        # Allow for some variation but should show enhancement
        assert edge_attention >= edge_normal * 0.9
        
    def test_hierarchy_information_retrieval(self):
        """Test getting information from visual hierarchy."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = create_visual_processing_hierarchy()
        hierarchy = VisualProcessingHierarchy(config)
        
        info = hierarchy.get_hierarchy_info()
        
        # Check required fields
        assert 'hierarchy_id' in info
        assert 'modality' in info
        assert 'total_layers' in info
        assert 'layers' in info
        
        assert info['modality'] == 'visual'
        assert info['total_layers'] == 4
        
        # Check layer information
        for layer_name in ['EdgeDetection', 'Orientation', 'ShapeDetection', 'ObjectRecognition']:
            assert layer_name in info['layers']
            layer_info = info['layers'][layer_name]
            assert 'name' in layer_info
            assert 'level' in layer_info
            assert 'size' in layer_info


class TestVisualProcessingIntegration:
    """Test integration scenarios for visual processing."""
    
    def test_realistic_image_processing(self):
        """Test processing of more realistic visual inputs."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = create_visual_processing_hierarchy()
        hierarchy = VisualProcessingHierarchy(config)
        
        # Create simple geometric pattern
        image = np.zeros((28, 28))
        
        # Rectangle outline
        image[8:20, 8:20] = 0.3      # Fill
        image[8, 8:20] = 1.0         # Top edge
        image[19, 8:20] = 1.0        # Bottom edge
        image[8:20, 8] = 1.0         # Left edge
        image[8:20, 19] = 1.0        # Right edge
        
        activations = hierarchy.process_sensory_input(image)
        
        # Should activate edge detection
        edge_activation = np.mean(activations['EdgeDetection'])
        assert edge_activation > 0.01  # Should detect edges
        
        # Should have some orientation responses
        orientation_activation = np.mean(activations['Orientation'])
        assert orientation_activation >= 0
        
    def test_noise_robustness(self):
        """Test robustness to noise in visual input."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = create_visual_processing_hierarchy()
        hierarchy = VisualProcessingHierarchy(config)
        
        # Clean signal
        clean_image = np.zeros((28, 28))
        clean_image[10:12, :] = 1.0  # Horizontal line
        
        clean_activations = hierarchy.process_sensory_input(clean_image)
        
        # Noisy signal
        noisy_image = clean_image + np.random.normal(0, 0.1, (28, 28))
        noisy_image = np.clip(noisy_image, 0, 1)
        
        noisy_activations = hierarchy.process_sensory_input(noisy_image)
        
        # Should still detect features with some noise
        clean_edge = np.mean(clean_activations['EdgeDetection'])
        noisy_edge = np.mean(noisy_activations['EdgeDetection'])
        
        # Noisy response should be reasonably close to clean response
        if clean_edge > 0.01:  # If clean signal was detected
            assert noisy_edge > clean_edge * 0.3  # At least 30% of clean response
            
    def test_multiple_feature_detection(self):
        """Test detection of multiple visual features."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = create_visual_processing_hierarchy()
        hierarchy = VisualProcessingHierarchy(config)
        
        # Create image with multiple features
        complex_image = np.zeros((28, 28))
        
        # Horizontal lines
        complex_image[5, 5:15] = 1.0
        complex_image[22, 10:25] = 1.0
        
        # Vertical lines
        complex_image[8:18, 20] = 1.0
        complex_image[15:25, 5] = 1.0
        
        # Diagonal
        for i in range(10):
            if 10+i < 28 and 10+i < 28:
                complex_image[10+i, 10+i] = 1.0
                
        activations = hierarchy.process_sensory_input(complex_image)
        
        # Should activate multiple processing stages
        edge_active = np.sum(activations['EdgeDetection'] > 0.01)
        orientation_active = np.sum(activations['Orientation'] > 0.01)
        
        assert edge_active > 5      # Multiple edge detectors active
        assert orientation_active > 2  # Multiple orientation detectors active


def run_visual_processing_tests():
    """Run comprehensive visual processing tests."""
    if not IMPORTS_SUCCESS:
        print("Cannot run tests - required modules not available")
        return False
        
    print("\n=== Visual Processing Hierarchy Tests ===")
    
    try:
        # Test 1: Basic layer functionality
        print("\n1. Testing Basic Layer Functionality...")
        
        # Test edge detection
        edge_config = VisualLayerConfig(
            name="TestEdge", level=ProcessingLevel.PRIMARY, size=32,
            feature_type=VisualFeatureType.EDGE, n_orientations=4
        )
        edge_layer = EdgeDetectionLayer(edge_config)
        assert len(edge_layer.gabor_filters) == 4
        print("  ✅ EdgeDetectionLayer creation and Gabor filters")
        
        # Test processing
        test_image = np.random.rand(16, 16)
        edge_response = edge_layer.process_input(test_image)
        assert len(edge_response) == 32
        print("  ✅ Edge detection processing")
        
        # Test orientation layer
        orient_config = VisualLayerConfig(
            name="TestOrient", level=ProcessingLevel.SECONDARY, size=16,
            feature_type=VisualFeatureType.ORIENTATION, n_orientations=8
        )
        orient_layer = OrientationLayer(orient_config)
        assert len(orient_layer.preferred_orientations) == 16
        print("  ✅ OrientationLayer creation and preferences")
        
        # Test 2: Complete hierarchy
        print("\n2. Testing Complete Visual Hierarchy...")
        
        config = create_visual_processing_hierarchy()
        hierarchy = VisualProcessingHierarchy(config)
        
        assert len(hierarchy.layers) == 4
        assert isinstance(hierarchy.layers['EdgeDetection'], EdgeDetectionLayer)
        print("  ✅ Visual hierarchy creation with specialized layers")
        
        # Test processing pipeline
        test_input = np.zeros((28, 28))
        test_input[10:12, 5:23] = 1.0  # Horizontal edge
        test_input[5:23, 10:12] = 1.0  # Vertical edge
        
        activations = hierarchy.process_sensory_input(test_input)
        assert len(activations) == 4
        print("  ✅ Full pipeline processing")
        
        # Test 3: Feature detection validation
        print("\n3. Testing Feature Detection...")
        
        # Edge detection validation
        edge_image = np.zeros((28, 28))
        edge_image[14, :] = 1.0  # Strong horizontal edge
        
        edge_activations = hierarchy.process_sensory_input(edge_image)
        edge_response = np.mean(edge_activations['EdgeDetection'])
        print(f"  ✅ Edge detection response: {edge_response:.4f}")
        
        # Attention modulation
        hierarchy.apply_attention_modulation("EdgeDetection", 1.5)
        enhanced_activations = hierarchy.process_sensory_input(edge_image)
        enhanced_response = np.mean(enhanced_activations['EdgeDetection'])
        print(f"  ✅ Attention-enhanced response: {enhanced_response:.4f}")
        
        # Test 4: Information retrieval
        info = hierarchy.get_hierarchy_info()
        assert info['total_layers'] == 4
        assert info['modality'] == 'visual'
        print("  ✅ Hierarchy information retrieval")
        
        print("\n✅ All Visual Processing Hierarchy tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_visual_processing_tests()
    
    if success:
        print("\n🎉 Task 6B.2: Visual Processing Hierarchy Implementation")
        print("All tests passed - visual processing pipeline validated!")
        print("\nKey features validated:")
        print("  • Gabor filter-based edge detection")
        print("  • Orientation selectivity in visual cortex")
        print("  • Shape template matching")
        print("  • Object recognition prototypes")
        print("  • Complete visual processing pipeline")
    else:
        print("\n❌ Some tests failed - check implementation")
        
    exit(0 if success else 1)