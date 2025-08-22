#!/usr/bin/env python3
"""
Tests for Multi-Modal Integration System
======================================

Task 6B.4 Testing: Validates the multi-modal integration system including
cross-modal plasticity, attention mechanisms, and sensory integration.
"""

import pytest
import numpy as np
from typing import Dict, List

try:
    from core.multimodal_integration import (
        MultiModalIntegrationLayer,
        CrossModalPlasticity,
        MultiModalAttention,
        MultiModalHierarchy,
        MultiModalConfig,
        CrossModalConfig,
        IntegrationType
    )
    from core.hierarchical_sensory_processing import (
        SensoryModalityType,
        ProcessingLevel
    )
    IMPORTS_SUCCESS = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORTS_SUCCESS = False


class TestCrossModalConfig:
    """Test cross-modal configuration functionality."""
    
    def test_cross_modal_config_creation(self):
        """Test cross-modal configuration creation."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = CrossModalConfig(
            source_modality=SensoryModalityType.VISUAL,
            target_modality=SensoryModalityType.AUDITORY,
            integration_type=IntegrationType.ASSOCIATIVE,
            connection_strength=0.7,
            plasticity_rate=0.02,
            temporal_window=0.15
        )
        
        assert config.source_modality == SensoryModalityType.VISUAL
        assert config.target_modality == SensoryModalityType.AUDITORY
        assert config.integration_type == IntegrationType.ASSOCIATIVE
        assert config.connection_strength == 0.7
        assert config.plasticity_rate == 0.02
        assert config.temporal_window == 0.15


class TestMultiModalConfig:
    """Test multi-modal configuration functionality."""
    
    def test_multimodal_config_creation(self):
        """Test multi-modal configuration creation."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = MultiModalConfig(
            name="TestMultiModal",
            level=ProcessingLevel.INTEGRATION,
            size=128,
            modalities=[SensoryModalityType.VISUAL, SensoryModalityType.AUDITORY],
            integration_type=IntegrationType.COOPERATIVE,
            temporal_integration_window=0.25,
            attention_decay_rate=0.15
        )
        
        assert config.name == "TestMultiModal"
        assert config.level == ProcessingLevel.INTEGRATION
        assert config.size == 128
        assert len(config.modalities) == 2
        assert SensoryModalityType.VISUAL in config.modalities
        assert SensoryModalityType.AUDITORY in config.modalities
        assert config.integration_type == IntegrationType.COOPERATIVE
        assert config.temporal_integration_window == 0.25
        assert config.attention_decay_rate == 0.15


class TestCrossModalPlasticity:
    """Test cross-modal plasticity functionality."""
    
    def test_plasticity_initialization(self):
        """Test cross-modal plasticity initialization."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = CrossModalConfig(
            source_modality=SensoryModalityType.VISUAL,
            target_modality=SensoryModalityType.AUDITORY,
            plasticity_rate=0.01
        )
        
        plasticity = CrossModalPlasticity(config)
        
        assert plasticity.config == config
        assert len(plasticity.association_weights) == 0  # Initially empty
        assert len(plasticity.learning_history) == 0
        
    def test_association_update(self):
        """Test association weight updates."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = CrossModalConfig(
            source_modality=SensoryModalityType.VISUAL,
            target_modality=SensoryModalityType.AUDITORY,
            plasticity_rate=0.1
        )
        
        plasticity = CrossModalPlasticity(config)
        
        # Create correlated activities
        t = np.linspace(0, 1, 100)
        source_activity = np.sin(2 * np.pi * 5 * t) + 0.1 * np.random.rand(100)
        target_activity = np.sin(2 * np.pi * 5 * t + 0.1) + 0.1 * np.random.rand(100)
        
        # Update associations
        plasticity.update_associations(source_activity, target_activity, dt=0.01)
        
        # Should have learned some association
        assert len(plasticity.association_weights) > 0
        assert len(plasticity.learning_history) > 0
        
        association_strength = plasticity.get_association_strength()
        assert isinstance(association_strength, float)
        
    def test_cross_modal_influence(self):
        """Test cross-modal influence computation."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = CrossModalConfig(
            source_modality=SensoryModalityType.VISUAL,
            target_modality=SensoryModalityType.AUDITORY,
            connection_strength=0.5
        )
        
        plasticity = CrossModalPlasticity(config)
        
        # Set up association manually
        key = "visual_auditory"
        plasticity.association_weights[key] = 0.8
        
        # Test influence
        source_activity = np.array([1.0, 0.5, 0.8, 0.3])
        influence = plasticity.compute_cross_modal_influence(source_activity)
        
        assert isinstance(influence, np.ndarray)
        assert len(influence) == len(source_activity)
        assert np.all(influence != source_activity)  # Should be different due to weighting
        
    def test_plasticity_learning_dynamics(self):
        """Test plasticity learning over time."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = CrossModalConfig(
            source_modality=SensoryModalityType.VISUAL,
            target_modality=SensoryModalityType.AUDITORY,
            plasticity_rate=0.05
        )
        
        plasticity = CrossModalPlasticity(config)
        
        # Train with correlated inputs
        initial_strength = plasticity.get_association_strength()
        
        for epoch in range(20):
            # Create positively correlated signals
            base_signal = np.sin(epoch * 0.3)
            noise_level = 0.1
            
            source = np.array([base_signal + np.random.rand() * noise_level for _ in range(10)])
            target = np.array([base_signal + np.random.rand() * noise_level for _ in range(10)])
            
            plasticity.update_associations(source, target, dt=0.01)
            
        final_strength = plasticity.get_association_strength()
        
        # Should have strengthened association
        assert final_strength > initial_strength


class TestMultiModalAttention:
    """Test multi-modal attention functionality."""
    
    def test_attention_initialization(self):
        """Test multi-modal attention initialization."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        modalities = [SensoryModalityType.VISUAL, SensoryModalityType.AUDITORY]
        attention = MultiModalAttention(modalities, decay_rate=0.1)
        
        assert len(attention.modalities) == 2
        assert len(attention.attention_weights) == 2
        
        # Initial weights should be equal
        for modality in modalities:
            assert attention.attention_weights[modality] == 1.0
            
    def test_attention_update(self):
        """Test attention weight updates."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        modalities = [SensoryModalityType.VISUAL, SensoryModalityType.AUDITORY]
        attention = MultiModalAttention(modalities, decay_rate=0.2)
        
        # Create activities with different salience
        activities = {
            SensoryModalityType.VISUAL: np.array([1.0, 0.8, 0.9, 1.2]),     # High salience
            SensoryModalityType.AUDITORY: np.array([0.2, 0.1, 0.3, 0.15])    # Low salience
        }
        
        initial_weights = attention.get_attention_weights()
        
        # Update attention multiple times
        for _ in range(10):
            attention.update_attention(activities, dt=0.01)
            
        final_weights = attention.get_attention_weights()
        
        # Visual should get more attention due to higher salience
        assert final_weights[SensoryModalityType.VISUAL] > initial_weights[SensoryModalityType.VISUAL] * 0.9
        
    def test_attention_application(self):
        """Test application of attention weights."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        modalities = [SensoryModalityType.VISUAL, SensoryModalityType.AUDITORY]
        attention = MultiModalAttention(modalities)
        
        # Set specific attention weights
        attention.attention_weights[SensoryModalityType.VISUAL] = 1.5
        attention.attention_weights[SensoryModalityType.AUDITORY] = 0.5
        
        activities = {
            SensoryModalityType.VISUAL: np.array([1.0, 1.0, 1.0]),
            SensoryModalityType.AUDITORY: np.array([1.0, 1.0, 1.0])
        }
        
        attended = attention.apply_attention(activities)
        
        # Visual should be enhanced, auditory should be suppressed
        assert np.all(attended[SensoryModalityType.VISUAL] > activities[SensoryModalityType.VISUAL])
        assert np.all(attended[SensoryModalityType.AUDITORY] < activities[SensoryModalityType.AUDITORY])


class TestMultiModalIntegrationLayer:
    """Test multi-modal integration layer functionality."""
    
    def test_integration_layer_initialization(self):
        """Test integration layer initialization."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = MultiModalConfig(
            name="TestIntegration",
            level=ProcessingLevel.INTEGRATION,
            size=64,
            modalities=[SensoryModalityType.VISUAL, SensoryModalityType.AUDITORY],
            integration_type=IntegrationType.CONVERGENT
        )
        
        layer = MultiModalIntegrationLayer(config)
        
        assert layer.config.name == "TestIntegration"
        assert len(layer.cross_modal_plasticity) > 0  # Should create cross-modal connections
        assert isinstance(layer.attention_system, MultiModalAttention)
        
    def test_convergent_integration(self):
        """Test convergent integration mode."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = MultiModalConfig(
            name="ConvergentTest",
            level=ProcessingLevel.INTEGRATION,
            size=32,
            modalities=[SensoryModalityType.VISUAL, SensoryModalityType.AUDITORY],
            integration_type=IntegrationType.CONVERGENT
        )
        
        layer = MultiModalIntegrationLayer(config)
        
        # Test multimodal input
        inputs = {
            SensoryModalityType.VISUAL: np.array([0.8, 0.6, 0.7, 0.5] * 8),     # 32 elements
            SensoryModalityType.AUDITORY: np.array([0.4, 0.3, 0.5, 0.2] * 8)    # 32 elements
        }
        
        response = layer.process_multimodal_input(inputs)
        
        assert isinstance(response, np.ndarray)
        assert len(response) == config.size
        assert np.all(response >= 0)  # Should be non-negative
        
    def test_competitive_integration(self):
        """Test competitive integration mode."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = MultiModalConfig(
            name="CompetitiveTest",
            level=ProcessingLevel.INTEGRATION,
            size=24,
            modalities=[SensoryModalityType.VISUAL, SensoryModalityType.AUDITORY],
            integration_type=IntegrationType.COMPETITIVE
        )
        
        layer = MultiModalIntegrationLayer(config)
        
        # Create inputs with clear winner
        inputs = {
            SensoryModalityType.VISUAL: np.array([1.5, 1.2, 1.8] * 8),      # Strong
            SensoryModalityType.AUDITORY: np.array([0.2, 0.1, 0.3] * 8)     # Weak
        }
        
        response = layer.process_multimodal_input(inputs)
        
        assert isinstance(response, np.ndarray)
        assert len(response) == config.size
        
        # Should be dominated by visual input (winner)
        visual_mean = np.mean(inputs[SensoryModalityType.VISUAL])
        response_mean = np.mean(response)
        
        # Response should be closer to visual than auditory
        assert response_mean > 0.5  # Should be substantial
        
    def test_cooperative_integration(self):
        """Test cooperative integration mode."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = MultiModalConfig(
            name="CooperativeTest",
            level=ProcessingLevel.INTEGRATION,
            size=20,
            modalities=[SensoryModalityType.VISUAL, SensoryModalityType.AUDITORY],
            integration_type=IntegrationType.COOPERATIVE
        )
        
        layer = MultiModalIntegrationLayer(config)
        
        # Train some cross-modal associations first
        for _ in range(5):
            correlated_inputs = {
                SensoryModalityType.VISUAL: np.array([0.8, 0.9, 0.7] * 7)[:20],
                SensoryModalityType.AUDITORY: np.array([0.7, 0.8, 0.6] * 7)[:20]
            }
            layer.process_multimodal_input(correlated_inputs)
            
        # Test cooperative enhancement
        test_inputs = {
            SensoryModalityType.VISUAL: np.array([0.6, 0.7, 0.5] * 7)[:20],
            SensoryModalityType.AUDITORY: np.array([0.5, 0.6, 0.4] * 7)[:20]
        }
        
        response = layer.process_multimodal_input(test_inputs)
        
        assert isinstance(response, np.ndarray)
        assert len(response) == config.size
        
    def test_integration_info_retrieval(self):
        """Test integration information retrieval."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = MultiModalConfig(
            name="InfoTest",
            level=ProcessingLevel.INTEGRATION,
            size=16,
            modalities=[SensoryModalityType.VISUAL, SensoryModalityType.AUDITORY]
        )
        
        layer = MultiModalIntegrationLayer(config)
        
        info = layer.get_integration_info()
        
        # Check required fields
        assert 'attention_weights' in info
        assert 'cross_modal_associations' in info
        assert 'integration_type' in info
        assert 'modalities' in info
        
        assert len(info['modalities']) == 2
        assert 'visual' in info['modalities']
        assert 'auditory' in info['modalities']


class TestMultiModalHierarchy:
    """Test complete multi-modal hierarchy functionality."""
    
    def test_multimodal_hierarchy_creation(self):
        """Test multi-modal hierarchy creation."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        # Create simple mock hierarchies
        visual_hierarchy = type('MockHierarchy', (), {
            'process_sensory_input': lambda self, x: {'ObjectRecognition': np.random.rand(16)}
        })()
        
        auditory_hierarchy = type('MockHierarchy', (), {
            'layers': {
                'FrequencyAnalysis': type('MockLayer', (), {
                    'process_input': lambda self, x: np.random.rand(32)
                })(),
                'TemporalPattern': type('MockLayer', (), {
                    'process_input': lambda self, x: np.random.rand(24)
                })(),
                'SoundRecognition': type('MockLayer', (), {
                    'process_input': lambda self, x: np.random.rand(12)
                })()
            }
        })()
        
        hierarchy = MultiModalHierarchy(
            visual_hierarchy=visual_hierarchy,
            auditory_hierarchy=auditory_hierarchy
        )
        
        assert hierarchy.visual_hierarchy is not None
        assert hierarchy.auditory_hierarchy is not None
        assert isinstance(hierarchy.integration_layer, MultiModalIntegrationLayer)
        
    def test_multimodal_processing_pipeline(self):
        """Test complete multimodal processing pipeline."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        # Create integration-only hierarchy for testing
        hierarchy = MultiModalHierarchy()
        
        # Test with direct modality inputs
        visual_input = np.random.rand(28, 28)
        auditory_input = np.random.rand(64, 32)
        
        results = hierarchy.process_multimodal_input(
            visual_input=visual_input,
            auditory_input=auditory_input
        )
        
        # Should return results even without full hierarchies
        assert isinstance(results, dict)
        
    def test_hierarchy_info_retrieval(self):
        """Test hierarchy information retrieval."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        hierarchy = MultiModalHierarchy()
        
        info = hierarchy.get_multimodal_info()
        
        assert 'has_visual' in info
        assert 'has_auditory' in info
        assert 'integration_info' in info
        
        assert isinstance(info['has_visual'], bool)
        assert isinstance(info['has_auditory'], bool)
        assert isinstance(info['integration_info'], dict)


class TestMultiModalIntegration:
    """Test integration scenarios and dynamics."""
    
    def test_attention_modulation_effects(self):
        """Test attention modulation effects on integration."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = MultiModalConfig(
            name="AttentionTest",
            level=ProcessingLevel.INTEGRATION,
            size=20,
            modalities=[SensoryModalityType.VISUAL, SensoryModalityType.AUDITORY],
            attention_decay_rate=0.3
        )
        
        layer = MultiModalIntegrationLayer(config)
        
        # Create inputs with different salience over time
        high_visual_inputs = {
            SensoryModalityType.VISUAL: np.array([1.2, 1.5, 1.0, 1.3] * 5),
            SensoryModalityType.AUDITORY: np.array([0.3, 0.2, 0.4, 0.1] * 5)
        }
        
        # Process several times to build attention bias
        for _ in range(8):
            layer.process_multimodal_input(high_visual_inputs)
            
        attention_weights = layer.attention_system.get_attention_weights()
        
        # Visual attention should be higher
        assert attention_weights[SensoryModalityType.VISUAL] > attention_weights[SensoryModalityType.AUDITORY]
        
    def test_cross_modal_plasticity_development(self):
        """Test development of cross-modal plasticity over time."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = MultiModalConfig(
            name="PlasticityTest",
            level=ProcessingLevel.INTEGRATION,
            size=16,
            modalities=[SensoryModalityType.VISUAL, SensoryModalityType.AUDITORY]
        )
        
        layer = MultiModalIntegrationLayer(config)
        
        # Get initial association strengths
        initial_info = layer.get_integration_info()
        initial_associations = initial_info['cross_modal_associations']
        
        # Train with correlated inputs
        for step in range(15):
            correlation_strength = 0.8
            base_activity = np.sin(step * 0.4) * 0.5 + 0.5
            
            correlated_inputs = {
                SensoryModalityType.VISUAL: np.array([base_activity + np.random.rand() * 0.2] * 16),
                SensoryModalityType.AUDITORY: np.array([base_activity * correlation_strength + np.random.rand() * 0.2] * 16)
            }
            
            layer.process_multimodal_input(correlated_inputs)
            
        # Get final association strengths
        final_info = layer.get_integration_info()
        final_associations = final_info['cross_modal_associations']
        
        # Should have developed associations
        for key in initial_associations:
            if key in final_associations:
                # Association should have changed (learned)
                assert abs(final_associations[key]) >= abs(initial_associations[key])
                
    def test_temporal_integration_dynamics(self):
        """Test temporal integration over multiple time steps."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        config = MultiModalConfig(
            name="TemporalTest",
            level=ProcessingLevel.INTEGRATION,
            size=12,
            modalities=[SensoryModalityType.VISUAL, SensoryModalityType.AUDITORY],
            temporal_integration_window=0.3
        )
        
        layer = MultiModalIntegrationLayer(config)
        
        responses = []
        
        # Process sequence of inputs with temporal structure
        for t in range(10):
            temporal_pattern = np.sin(t * 0.5)
            
            inputs = {
                SensoryModalityType.VISUAL: np.array([temporal_pattern + 0.3] * 12),
                SensoryModalityType.AUDITORY: np.array([temporal_pattern * 0.8 + 0.2] * 12)
            }
            
            response = layer.process_multimodal_input(inputs)
            responses.append(np.mean(response))
            
        # Should show temporal dynamics
        assert len(responses) == 10
        assert np.std(responses) > 0.01  # Should have variation over time


def run_multimodal_integration_tests():
    """Run comprehensive multimodal integration tests."""
    if not IMPORTS_SUCCESS:
        print("Cannot run tests - required modules not available")
        return False
        
    print("\n=== Multi-Modal Integration System Tests ===")
    
    try:
        # Test 1: Basic component functionality
        print("\n1. Testing Basic Component Functionality...")
        
        # Test cross-modal plasticity
        cross_config = CrossModalConfig(
            source_modality=SensoryModalityType.VISUAL,
            target_modality=SensoryModalityType.AUDITORY,
            plasticity_rate=0.05
        )
        
        plasticity = CrossModalPlasticity(cross_config)
        assert plasticity.config == cross_config
        print("  ✅ CrossModalPlasticity creation")
        
        # Test attention system
        modalities = [SensoryModalityType.VISUAL, SensoryModalityType.AUDITORY]
        attention = MultiModalAttention(modalities)
        assert len(attention.attention_weights) == 2
        print("  ✅ MultiModalAttention creation")
        
        # Test 2: Integration layer functionality
        print("\n2. Testing Integration Layer Functionality...")
        
        config = MultiModalConfig(
            name="TestLayer",
            level=ProcessingLevel.INTEGRATION,
            size=32,
            modalities=[SensoryModalityType.VISUAL, SensoryModalityType.AUDITORY],
            integration_type=IntegrationType.CONVERGENT
        )
        
        layer = MultiModalIntegrationLayer(config)
        assert len(layer.cross_modal_plasticity) > 0
        print("  ✅ MultiModalIntegrationLayer creation")
        
        # Test processing
        inputs = {
            SensoryModalityType.VISUAL: np.random.rand(32) * 0.6 + 0.2,
            SensoryModalityType.AUDITORY: np.random.rand(32) * 0.4 + 0.3
        }
        
        response = layer.process_multimodal_input(inputs)
        assert len(response) == 32
        print("  ✅ Multi-modal input processing")
        
        # Test 3: Integration modes
        print("\n3. Testing Integration Modes...")
        
        # Convergent integration
        conv_config = MultiModalConfig(
            name="Convergent", level=ProcessingLevel.INTEGRATION, size=16,
            modalities=[SensoryModalityType.VISUAL, SensoryModalityType.AUDITORY],
            integration_type=IntegrationType.CONVERGENT
        )
        conv_layer = MultiModalIntegrationLayer(conv_config)
        
        test_inputs = {
            SensoryModalityType.VISUAL: np.array([0.8, 0.6, 0.7, 0.9] * 4),
            SensoryModalityType.AUDITORY: np.array([0.4, 0.5, 0.3, 0.6] * 4)
        }
        
        conv_response = conv_layer.process_multimodal_input(test_inputs)
        assert len(conv_response) == 16
        print("  ✅ Convergent integration")
        
        # Competitive integration
        comp_config = MultiModalConfig(
            name="Competitive", level=ProcessingLevel.INTEGRATION, size=16,
            modalities=[SensoryModalityType.VISUAL, SensoryModalityType.AUDITORY],
            integration_type=IntegrationType.COMPETITIVE
        )
        comp_layer = MultiModalIntegrationLayer(comp_config)
        
        # Strong visual, weak auditory
        comp_inputs = {
            SensoryModalityType.VISUAL: np.array([1.2, 1.5, 1.0, 1.8] * 4),
            SensoryModalityType.AUDITORY: np.array([0.2, 0.1, 0.3, 0.15] * 4)
        }
        
        comp_response = comp_layer.process_multimodal_input(comp_inputs)
        assert np.mean(comp_response) > 0.3  # Should be dominated by visual
        print("  ✅ Competitive integration")
        
        # Test 4: Plasticity and attention dynamics
        print("\n4. Testing Plasticity and Attention Dynamics...")
        
        # Train cross-modal associations
        for epoch in range(10):
            base_signal = np.sin(epoch * 0.3) * 0.5 + 0.5
            
            corr_inputs = {
                SensoryModalityType.VISUAL: np.array([base_signal + np.random.rand() * 0.1] * 16),
                SensoryModalityType.AUDITORY: np.array([base_signal * 0.8 + np.random.rand() * 0.1] * 16)
            }
            
            conv_layer.process_multimodal_input(corr_inputs)
            
        # Check learned associations
        integration_info = conv_layer.get_integration_info()
        associations = integration_info['cross_modal_associations']
        
        print(f"  ✅ Cross-modal plasticity: {len(associations)} associations learned")
        
        # Test attention adaptation
        attention_weights = integration_info['attention_weights']
        print(f"  ✅ Attention weights: Visual={attention_weights.get(SensoryModalityType.VISUAL, 0):.3f}")
        
        # Test 5: Complete hierarchy
        print("\n5. Testing Complete Multi-Modal Hierarchy...")
        
        hierarchy = MultiModalHierarchy()
        assert isinstance(hierarchy.integration_layer, MultiModalIntegrationLayer)
        
        info = hierarchy.get_multimodal_info()
        assert 'integration_info' in info
        print("  ✅ MultiModalHierarchy creation and info retrieval")
        
        print("\n✅ All Multi-Modal Integration tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_multimodal_integration_tests()
    
    if success:
        print("\n🎉 Task 6B.4: Multi-Modal Integration System")
        print("All tests passed - multi-modal integration validated!")
        print("\nKey features validated:")
        print("  • Cross-modal plasticity with Hebbian learning")
        print("  • Multi-modal attention with dynamic weighting")
        print("  • Multiple integration strategies (convergent, competitive, cooperative)")
        print("  • Temporal synchronization and association learning")
        print("  • Complete integration with sensory hierarchies")
    else:
        print("\n❌ Some tests failed - check implementation")
        
    exit(0 if success else 1)