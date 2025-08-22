#!/usr/bin/env python3
"""
Tests for Large-Scale Network Properties Validation
==================================================

This test suite validates that large-scale networks maintain biological
realism across different scales and configurations.
"""

import pytest
import numpy as np
import time
from typing import Dict, List

try:
    from core.large_scale_validation import LargeScaleNetworkValidator, BiologicalMetrics
    from core.brain_topology import BrainTopologyBuilder
    from core.gpu_scaling import LargeScaleNetworkBuilder
    IMPORTS_SUCCESS = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORTS_SUCCESS = False


class TestEIBalanceValidation:
    """Test E/I balance validation at different scales."""
    
    def test_ei_balance_small_network(self):
        """Test E/I balance validation for small networks."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        validator = LargeScaleNetworkValidator()
        
        # Test valid E/I ratio
        config = {
            'total_neurons': 1000,
            'ei_ratio': 0.8,
            'num_modules': 4
        }
        
        result = validator.validate_network_properties(config)
        assert result.metrics.ei_ratio == 0.8
        assert result.metrics.excitatory_fraction == 0.8
        assert result.metrics.inhibitory_fraction == 0.2
        
    def test_ei_balance_large_network(self):
        """Test E/I balance maintenance in large networks."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        validator = LargeScaleNetworkValidator()
        
        # Test large network E/I balance
        config = {
            'total_neurons': 100000,
            'ei_ratio': 0.8,
            'num_modules': 100,
            'total_connections': 200000,
            'small_world_metrics': {
                'clustering': 0.4,
                'avg_path_length': 3.0,
                'small_world_index': 2.0
            }
        }
        
        result = validator.validate_network_properties(config)
        assert result.metrics.ei_ratio == 0.8
        assert result.passes_validation
        
    def test_invalid_ei_ratio(self):
        """Test detection of invalid E/I ratios."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        validator = LargeScaleNetworkValidator()
        
        # Test invalid E/I ratio (too high)
        config = {
            'total_neurons': 5000,
            'ei_ratio': 0.95,  # Too high
            'num_modules': 10
        }
        
        result = validator.validate_network_properties(config)
        assert not result.passes_validation
        assert any("E/I ratio" in note for note in result.validation_notes)


class TestConnectionDensityValidation:
    """Test connection density validation."""
    
    def test_appropriate_density(self):
        """Test validation of appropriate connection density."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        validator = LargeScaleNetworkValidator()
        
        config = {
            'total_neurons': 10000,
            'ei_ratio': 0.8,
            'total_connections': 100000,  # 0.001 density - very sparse
            'num_modules': 20
        }
        
        result = validator.validate_network_properties(config)
        assert result.metrics.connection_density <= 0.05  # Should be sparse
        
    def test_excessive_density(self):
        """Test detection of excessive connection density."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        validator = LargeScaleNetworkValidator()
        
        config = {
            'total_neurons': 1000,
            'ei_ratio': 0.8,
            'total_connections': 100000,  # 0.1 density - too dense
            'num_modules': 5
        }
        
        result = validator.validate_network_properties(config)
        assert not result.passes_validation
        assert any("Density" in note for note in result.validation_notes)


class TestSmallWorldValidation:
    """Test small-world properties validation."""
    
    def test_valid_small_world_properties(self):
        """Test validation of proper small-world properties."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        validator = LargeScaleNetworkValidator()
        
        config = {
            'total_neurons': 25000,
            'ei_ratio': 0.8,
            'total_connections': 50000,
            'num_modules': 25,
            'small_world_metrics': {
                'clustering': 0.45,  # Good clustering
                'avg_path_length': 3.2,  # Short path length
                'small_world_index': 2.1,  # Strong small-world
                'degree_gamma': 2.2
            }
        }
        
        result = validator.validate_network_properties(config)
        assert result.metrics.clustering >= 0.3
        assert result.metrics.average_path_length <= 6.0
        assert result.metrics.small_world_index >= 1.5
        
    def test_invalid_small_world_properties(self):
        """Test detection of invalid small-world properties."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        validator = LargeScaleNetworkValidator()
        
        config = {
            'total_neurons': 15000,
            'ei_ratio': 0.8,
            'total_connections': 30000,
            'num_modules': 15,
            'small_world_metrics': {
                'clustering': 0.1,  # Too low clustering
                'avg_path_length': 8.0,  # Too long path length
                'small_world_index': 0.8,  # Not small-world
                'degree_gamma': 1.5
            }
        }
        
        result = validator.validate_network_properties(config)
        assert not result.passes_validation
        assert any("Clustering" in note or "Path length" in note or "SW index" in note 
                  for note in result.validation_notes)


class TestActivityPatternValidation:
    """Test neural activity pattern validation."""
    
    def test_realistic_firing_rates(self):
        """Test validation of realistic firing rates."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        validator = LargeScaleNetworkValidator()
        
        # Generate realistic spike data
        num_neurons = 1000
        simulation_time = 1000  # 1 second
        
        # Create spike times with realistic firing rates (1-20 Hz)
        spike_data = []
        for neuron in range(num_neurons):
            firing_rate = np.random.uniform(1, 20)  # Hz
            num_spikes = int(firing_rate * simulation_time / 1000)
            spike_times = np.random.uniform(0, simulation_time, num_spikes)
            spike_data.extend([(neuron, t) for t in spike_times])
            
        spike_array = np.array(spike_data)
        
        config = {
            'total_neurons': num_neurons,
            'ei_ratio': 0.8,
            'num_modules': 10
        }
        
        result = validator.validate_network_properties(config, spike_array)
        assert 0.5 <= result.metrics.firing_rate_mean <= 50.0
        assert result.metrics.firing_rate_std > 0.1
        
    def test_unrealistic_firing_rates(self):
        """Test detection of unrealistic firing rates."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        validator = LargeScaleNetworkValidator()
        
        # Generate unrealistic spike data (too high firing rate)
        num_neurons = 500
        simulation_time = 1000
        
        # Create excessive spikes (>100 Hz)
        spike_data = []
        for neuron in range(num_neurons):
            firing_rate = 150  # Way too high
            num_spikes = int(firing_rate * simulation_time / 1000)
            spike_times = np.random.uniform(0, simulation_time, num_spikes)
            spike_data.extend([(neuron, t) for t in spike_times])
            
        spike_array = np.array(spike_data)
        
        config = {
            'total_neurons': num_neurons,
            'ei_ratio': 0.8,
            'num_modules': 5
        }
        
        result = validator.validate_network_properties(config, spike_array)
        assert not result.passes_validation
        assert any("firing rate" in note for note in result.validation_notes)


class TestLargeScaleValidation:
    """Test validation across different network scales."""
    
    def test_modular_network_validation(self):
        """Test validation of networks with many modules."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        validator = LargeScaleNetworkValidator()
        
        # Large network with 1000 modules
        config = {
            'total_neurons': 1000000,  # 1M neurons
            'ei_ratio': 0.8,
            'total_connections': 20000000,  # 2% connectivity
            'num_modules': 1000,
            'small_world_metrics': {
                'clustering': 0.5,
                'avg_path_length': 4.2,
                'small_world_index': 1.8,
                'degree_gamma': 2.3
            }
        }
        
        result = validator.validate_network_properties(config)
        assert result.network_size == 1000000
        assert result.num_modules == 1000
        # Should maintain biological properties at scale
        assert result.metrics.ei_ratio == 0.8
        assert result.metrics.connection_density <= 0.05
        
    def test_scaling_performance(self):
        """Test validation performance across network scales."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        validator = LargeScaleNetworkValidator()
        
        # Test different network sizes
        network_sizes = [1000, 10000, 100000, 500000]
        
        for size in network_sizes:
            config = {
                'total_neurons': size,
                'ei_ratio': 0.8,
                'total_connections': int(size * 0.02),
                'num_modules': max(1, size // 1000),
                'small_world_metrics': {
                    'clustering': 0.4,
                    'avg_path_length': 3.5,
                    'small_world_index': 1.9
                }
            }
            
            start_time = time.time()
            result = validator.validate_network_properties(config)
            validation_time = time.time() - start_time
            
            # Validation should complete quickly even for large networks
            assert validation_time < 1.0  # Less than 1 second
            assert result.network_size == size
            

class TestBiologicalRealism:
    """Test overall biological realism validation."""
    
    def test_realistic_network_passes(self):
        """Test that biologically realistic networks pass validation."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        validator = LargeScaleNetworkValidator()
        
        # Create realistic network configuration
        config = {
            'total_neurons': 50000,
            'ei_ratio': 0.8,  # 80% excitatory
            'total_connections': 100000,  # 4% connectivity  
            'num_modules': 50,
            'small_world_metrics': {
                'clustering': 0.45,
                'avg_path_length': 3.2,
                'small_world_index': 2.0,
                'degree_gamma': 2.2
            }
        }
        
        # Add realistic activity data
        spike_data = self._generate_realistic_spikes(50000, 1000)
        
        result = validator.validate_network_properties(config, spike_data)
        assert result.passes_validation
        assert result.metrics.validation_score >= 0.7
        
    def test_unrealistic_network_fails(self):
        """Test that unrealistic networks fail validation."""
        if not IMPORTS_SUCCESS:
            pytest.skip("Required modules not available")
            
        validator = LargeScaleNetworkValidator()
        
        # Create unrealistic network configuration
        config = {
            'total_neurons': 10000,
            'ei_ratio': 0.95,  # Too many excitatory neurons
            'total_connections': 500000,  # Way too dense
            'num_modules': 5,
            'small_world_metrics': {
                'clustering': 0.1,  # Too low
                'avg_path_length': 10.0,  # Too high
                'small_world_index': 0.5,  # Not small-world
                'degree_gamma': 1.0
            }
        }
        
        result = validator.validate_network_properties(config)
        assert not result.passes_validation
        assert result.metrics.validation_score < 0.7
        
    def _generate_realistic_spikes(self, num_neurons: int, sim_time: int) -> np.ndarray:
        """Generate realistic spike data for testing."""
        spike_data = []
        
        for neuron in range(num_neurons):
            # Realistic firing rate (1-20 Hz, log-normal distribution)
            firing_rate = np.random.lognormal(1.5, 0.5)
            firing_rate = np.clip(firing_rate, 0.5, 25.0)
            
            num_spikes = int(firing_rate * sim_time / 1000)
            if num_spikes > 0:
                spike_times = np.sort(np.random.uniform(0, sim_time, num_spikes))
                for t in spike_times:
                    spike_data.append([neuron, t])
                    
        return np.array(spike_data) if spike_data else np.array([]).reshape(0, 2)


def test_integration_with_brain_topology():
    """Test integration with brain topology builder."""
    if not IMPORTS_SUCCESS:
        pytest.skip("Required modules not available")
        
    try:
        validator = LargeScaleNetworkValidator()
        builder = BrainTopologyBuilder()
        
        # Create brain-inspired network
        network_config = builder.create_cortical_network(
            size=10000,
            modules=20,
            connectivity_density=0.03,
            ei_ratio=0.8
        )
        
        # Validate the network
        result = validator.validate_network_properties(network_config)
        
        # Brain-inspired networks should pass validation
        assert result.network_size == 10000
        assert result.num_modules == 20
        assert result.metrics.ei_ratio == 0.8
        
    except ImportError:
        pytest.skip("BrainTopologyBuilder not available")


if __name__ == "__main__":
    # Run basic validation tests
    print("Running Large-Scale Network Validation Tests...")
    
    try:
        from core.large_scale_validation import validate_large_scale_networks
        results = validate_large_scale_networks()
        
        print(f"\nValidation completed for {len(results)} networks")
        successful = sum(1 for r in results if r.passes_validation)
        print(f"Biological validation: {successful}/{len(results)} networks passed")
        
    except Exception as e:
        print(f"Validation test failed: {str(e)}")
        import traceback
        traceback.print_exc()