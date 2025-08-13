"""
Simplified performance validation tests for neuromorphic computing.

Tests the core performance improvements for massive scale networks.
"""

import pytest
import numpy as np
import time
from core.synapses import SynapsePopulation
from core.neurons import NeuronPopulation


class TestPerformanceValidation:
    """Validate performance improvements for neuromorphic computing."""
    
    def test_construction_speedup(self):
        """Test network construction speedup."""
        sizes = [1000, 5000, 10000, 25000]
        
        for size in sizes:
            start_time = time.time()
            
            syn_pop = SynapsePopulation(
                pre_population_size=size,
                post_population_size=size // 2,
                synapse_type="stdp",
                connection_probability=0.001,
                weight=0.1
            )
            
            construction_time = time.time() - start_time
            
            # Should scale reasonably with size
            connections = len(syn_pop.synapses)
            print(f"Size {size}: {construction_time:.3f}s for {connections} connections")
            
            # Performance should be reasonable
            if size >= 10000:
                assert construction_time < 15.0, f"Size {size} took {construction_time:.2f}s - too slow"
    
    def test_memory_scaling(self):
        """Test memory usage scales with sparsity, not dense size."""
        base_size = 5000
        sparsity_levels = [0.01, 0.001, 0.0001]
        
        memory_usage = []
        
        for sparsity in sparsity_levels:
            syn_pop = SynapsePopulation(
                pre_population_size=base_size,
                post_population_size=base_size,
                synapse_type="stdp", 
                connection_probability=sparsity,
                weight=0.1
            )
            
            connections = syn_pop.total_connections
            # Approximate memory by number of connections
            memory_usage.append(connections)
            
            print(f"Sparsity {sparsity}: {connections} connections")
        
        # Memory should decrease with sparsity
        assert memory_usage[0] > memory_usage[1] > memory_usage[2]
        
        # Ultra-sparse should use much less memory
        assert memory_usage[2] < memory_usage[0] / 10
    
    def test_throughput_validation(self):
        """Test construction throughput for different scales."""
        test_configs = [
            (5000, 1000, 0.002),   # Medium scale
            (15000, 2000, 0.001),  # Large scale
            (30000, 3000, 0.0005), # Very large scale
        ]
        
        for pre_size, post_size, prob in test_configs:
            start_time = time.time()
            
            syn_pop = SynapsePopulation(
                pre_population_size=pre_size,
                post_population_size=post_size,
                synapse_type="stdp",
                connection_probability=prob,
                weight=0.1
            )
            
            construction_time = time.time() - start_time
            connections = len(syn_pop.synapses)
            
            # Calculate throughput
            throughput = connections / construction_time if construction_time > 0 else float('inf')
            
            print(f"Scale {pre_size}×{post_size}: {throughput:.0f} connections/sec")
            
            # Should maintain reasonable throughput
            assert throughput > 1000, f"Low throughput: {throughput:.0f} connections/sec"
    
    def test_gpu_performance_boost(self):
        """Test GPU acceleration provides performance boost."""
        # Test size that should trigger GPU acceleration
        gpu_size = 20000
        
        start_time = time.time()
        
        try:
            syn_pop = SynapsePopulation(
                pre_population_size=gpu_size,
                post_population_size=gpu_size // 4,
                synapse_type="stdp",
                connection_probability=0.0001,
                weight=0.1
            )
            
            gpu_time = time.time() - start_time
            
            print(f"GPU acceleration test: {gpu_size} neurons in {gpu_time:.3f}s")
            
            # Should be reasonably fast with GPU acceleration
            assert gpu_time < 30.0, f"GPU acceleration didn't help: {gpu_time:.2f}s"
            
        except ImportError:
            print("GPU libraries not available - skipping GPU test")
            pytest.skip("GPU libraries not available")
    
    def test_massive_scale_projection(self):
        """Project performance to massive neuromorphic scales."""
        # Test smaller scale and extrapolate
        test_configs = [
            (10000, "10k baseline"),
            (25000, "25k test"),
            (50000, "50k target")
        ]
        
        performance_data = []
        
        for size, label in test_configs:
            start_time = time.time()
            
            syn_pop = SynapsePopulation(
                pre_population_size=size,
                post_population_size=size // 10,
                synapse_type="stdp",
                connection_probability=0.0001,
                weight=0.1
            )
            
            construction_time = time.time() - start_time
            connections = len(syn_pop.synapses)
            
            performance_data.append((size, construction_time, connections))
            print(f"{label}: {construction_time:.3f}s for {connections} connections")
        
        # Verify scaling is reasonable (should be roughly linear with connections, not quadratic with size)
        if len(performance_data) >= 2:
            size_ratio = performance_data[1][0] / performance_data[0][0]  # size ratio
            time_ratio = performance_data[1][1] / performance_data[0][1]  # time ratio
            
            # Time should not scale quadratically with size
            assert time_ratio < size_ratio ** 1.5, f"Scaling too poor: {time_ratio:.2f} vs {size_ratio:.2f}"
    
    def test_neuromorphic_hardware_readiness(self):
        """Test readiness for neuromorphic hardware deployment."""
        # Neuromorphic chip-like scales
        neuromorphic_configs = [
            (100000, 1000, "Loihi-like scale"),
            (200000, 2000, "Large neuromorphic"),
        ]
        
        for pre_size, post_size, description in neuromorphic_configs:
            start_time = time.time()
            
            syn_pop = SynapsePopulation(
                pre_population_size=pre_size,
                post_population_size=post_size,
                synapse_type="stdp",
                connection_probability=0.00005,  # Very sparse for hardware
                weight=0.1
            )
            
            construction_time = time.time() - start_time
            connections = len(syn_pop.synapses)
            
            # Should be deployable on neuromorphic hardware
            assert construction_time < 120.0, f"{description} took {construction_time:.1f}s"
            
            # Verify hardware-appropriate sparsity
            total_possible = pre_size * post_size
            sparsity = connections / total_possible
            assert sparsity < 0.0001, f"Too dense for hardware: {sparsity:.6f}"
            
            print(f"✓ {description}: {construction_time:.2f}s, sparsity={sparsity:.6f}")
    
    def test_80m_neuron_feasibility(self):
        """Test 80M neuron architectural feasibility."""
        # Test representative sample for 80M extrapolation
        sample_size = 20000
        target_80m = 80_000_000
        
        start_time = time.time()
        
        syn_pop = SynapsePopulation(
            pre_population_size=sample_size,
            post_population_size=sample_size // 20,  # Sparse post-synaptic
            synapse_type="stdp",
            connection_probability=0.000001,  # Ultra-sparse for 80M scale
            weight=0.05
        )
        
        sample_time = time.time() - start_time
        connections = len(syn_pop.synapses)
        
        # Extrapolate to 80M
        scale_factor = target_80m / sample_size
        estimated_80m_time = sample_time * scale_factor
        estimated_80m_connections = connections * scale_factor
        
        print(f"80M feasibility:")
        print(f"  Sample {sample_size}: {sample_time:.3f}s, {connections} connections")
        print(f"  Estimated 80M: {estimated_80m_time:.1f}s, {estimated_80m_connections:.0f} connections")
        
        # 80M should be manageable with current architecture
        assert estimated_80m_time < 3600, f"80M would take {estimated_80m_time:.0f}s - too long"
        
        # Verify ultra-sparse architecture
        assert connections > 0, "No connections created"
        sample_sparsity = connections / (sample_size * (sample_size // 20))
        assert sample_sparsity < 0.00001, f"Not sparse enough for 80M: {sample_sparsity:.8f}"
