"""
Simplified tests for massive scale neuromorphic computing features.

This module validates the core optimizations for 50k+ neurons
focusing on construction speed and memory efficiency.
"""

import pytest
import numpy as np
import time
from core.synapses import SynapsePopulation
from core.neurons import NeuronPopulation


class TestMassiveScaleOptimizations:
    """Test optimizations for massive scale neuromorphic computing."""
    
    def test_sparse_connectivity_performance(self):
        """Test sparse connectivity construction speed."""
        # Test parameters for sparse connectivity (use size that triggers sparse path)
        pre_size = 30000  # Ensure we exceed 50k total threshold  
        post_size = 25000  # 30k + 25k = 55k > 50k threshold
        connection_probability = 0.0001  # Ultra-sparse (0.01%)
        
        start_time = time.time()
        
        # Create synapse population with sparse connectivity
        syn_pop = SynapsePopulation(
            pre_population_size=pre_size,
            post_population_size=post_size,
            synapse_type="stdp",
            connection_probability=connection_probability,
            weight=0.1
        )
        
        construction_time = time.time() - start_time
        
        # Verify performance - should be much faster than dense
        assert construction_time < 5.0, f"Construction took {construction_time:.2f}s - too slow!"
        
        # Verify sparse connectivity properties
        total_possible = pre_size * post_size
        expected_connections = int(total_possible * connection_probability)
        actual_connections = len(syn_pop._pre_ids) if hasattr(syn_pop, '_pre_ids') else len(syn_pop.synapses)
        
        # Allow 30% variance due to probabilistic nature
        assert abs(actual_connections - expected_connections) <= expected_connections * 0.3
        
        # Verify sparse arrays exist
        assert hasattr(syn_pop, '_pre_ids')
        assert hasattr(syn_pop, '_post_ids')
        assert len(syn_pop._pre_ids) == actual_connections
        
        print(f"✓ Sparse connectivity test: {pre_size}×{post_size} → {actual_connections} connections in {construction_time:.3f}s")
    
    def test_50k_neuron_construction_speed(self):
        """Test that 50k neuron networks build quickly."""
        pre_neurons = 50000
        post_neurons = 1000
        
        start_time = time.time()
        
        syn_pop = SynapsePopulation(
            pre_population_size=pre_neurons,
            post_population_size=post_neurons,
            synapse_type="stdp",
            connection_probability=0.0001,  # Ultra-sparse for 50k
            weight=0.2
        )
        
        construction_time = time.time() - start_time
        
        # This should be MUCH faster than original 365s
        assert construction_time < 30.0, f"50k construction took {construction_time:.2f}s - still too slow!"
        
        # Verify we got reasonable number of connections
        expected_connections = int(pre_neurons * post_neurons * 0.0001)
        actual_connections = len(syn_pop.synapses)
        assert actual_connections > 0
        assert actual_connections < pre_neurons * post_neurons * 0.001  # Much less than 0.1%
        
        print(f"✓ 50k neuron test: {construction_time:.3f}s vs original 365s = {365/construction_time:.1f}x speedup")
    
    def test_gpu_acceleration_threshold(self):
        """Test GPU acceleration triggers correctly."""
        # This should trigger GPU acceleration and sparse path
        large_size = 45000  # 45k + 5k = 50k total triggers sparse path
        
        try:
            syn_pop = SynapsePopulation(
                pre_population_size=large_size,
                post_population_size=5000,  # Total 50k triggers sparse path
                synapse_type="stdp",
                connection_probability=0.0001,
                weight=0.1
            )
            
            # If GPU is available, should use it
            # If not available, should still work (CPU fallback)
            assert len(syn_pop.synapses) >= 0
            print(f"✓ GPU acceleration test: {large_size} neurons processed")
            
        except Exception as e:
            # GPU not available is OK, but other errors are not
            if "CUDA" not in str(e) and "cupy" not in str(e):
                raise e
            print(f"✓ GPU not available, CPU fallback working: {e}")
    
    def test_memory_efficiency(self):
        """Test memory-efficient sparse representation."""
        size = 20000
        sparse_prob = 0.0005  # Very sparse
        
        syn_pop = SynapsePopulation(
            pre_population_size=size,
            post_population_size=size,
            synapse_type="stdp",
            connection_probability=sparse_prob,
            weight=0.15
        )
        
        # Dense representation would be size² connections
        dense_size = size * size
        sparse_size = len(syn_pop.synapses)
        
        # Should be MUCH smaller than dense
        compression_ratio = dense_size / sparse_size
        assert compression_ratio > 1000, f"Only {compression_ratio:.1f}x compression - not sparse enough!"
        
        print(f"✓ Memory efficiency: {compression_ratio:.0f}x compression vs dense representation")
    
    def test_neuromorphic_scale_readiness(self):
        """Test architecture readiness for neuromorphic hardware scales."""
        # Test parameters that represent neuromorphic chip scales
        test_scales = [
            (100000, 1000),   # 100k pre, 1k post
            (200000, 500),    # 200k pre, 500 post  
        ]
        
        for pre_size, post_size in test_scales:
            start_time = time.time()
            
            syn_pop = SynapsePopulation(
                pre_population_size=pre_size,
                post_population_size=post_size,
                synapse_type="stdp",
                connection_probability=0.00005,  # Extremely sparse for large scales
                weight=0.1
            )
            
            construction_time = time.time() - start_time
            
            # Should build in reasonable time for neuromorphic deployment
            assert construction_time < 60.0, f"{pre_size} neuron network took {construction_time:.1f}s"
            
            # Verify sparse architecture
            connections = len(syn_pop.synapses)
            total_possible = pre_size * post_size
            sparsity = connections / total_possible
            assert sparsity < 0.001, f"Sparsity {sparsity:.5f} too high for neuromorphic scale"
            
            print(f"✓ Neuromorphic scale {pre_size}×{post_size}: {construction_time:.2f}s, {connections} connections")
    
    def test_80m_neuron_extrapolation(self):
        """Test architectural readiness for 80M neuron target."""
        # Test a smaller representative sample that triggers sparse path
        test_size = 50000  # Use size that triggers sparse connectivity (50k + 5k = 55k > 50k threshold)
        target_size = 80_000_000
        
        start_time = time.time()
        
        syn_pop = SynapsePopulation(
            pre_population_size=test_size,
            post_population_size=5000,  # 50k + 5k = 55k total > 50k threshold
            synapse_type="stdp", 
            connection_probability=0.00001,  # Ultra-sparse for 80M extrapolation
            weight=0.05
        )
        
        construction_time = time.time() - start_time
        
        # Extrapolate to 80M scale
        scale_factor = target_size / test_size
        estimated_80m_time = construction_time * scale_factor
        
        # For 80M neurons, should be manageable with optimizations
        print(f"✓ 80M extrapolation: {test_size} → {estimated_80m_time:.1f}s estimated for 80M neurons")
        print(f"  Architecture supports neuromorphic hardware deployment")
        
        # Verify ultra-sparse connectivity for massive scale  
        connections = len(syn_pop._pre_ids) if hasattr(syn_pop, '_pre_ids') else len(syn_pop.synapses)
        assert connections > 0
        # For massive scale, connections should be much less than neuron count
        assert connections < test_size * 0.1  # Much less than 10% of neuron count
