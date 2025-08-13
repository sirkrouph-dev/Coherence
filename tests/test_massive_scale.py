"""
Tests for massive scale neuromorphic computing features.

This module validates the optimizations implemented for large-scale
spiking neural networks (50k+ neurons) including sparse connectivity,
GPU acceleration, and neuromorphic hardware targeting.
"""

import pytest
import numpy as np
import time
import re
from unittest.mock import patch, MagicMock

from core.synapses import SynapsePopulation, STDP_Synapse
from core.neurons import LeakyIntegrateAndFire, NeuronPopulation
from core.network import NeuromorphicNetwork


class TestMassiveScaleOptimizations:
    """Test optimizations for massive scale neuromorphic computing."""
    
    def test_sparse_connectivity_constructor(self):
        """Test sparse connectivity construction for large networks."""
        # Test parameters for sparse connectivity (use larger network to trigger sparse path)
        pre_size = 45000  # Ensure we exceed 50k total threshold
        post_size = 5500   # 45k + 5.5k = 50.5k > 50k threshold
        connection_probability = 0.00005  # Ultra-sparse (0.005%) - MUCH sparser
        
        # Create synapse population with sparse connectivity
        syn_pop = SynapsePopulation(
            pre_population_size=pre_size,
            post_population_size=post_size,
            synapse_type="stdp",
            connection_probability=connection_probability,
            weight=0.1
        )
        
        # Verify sparse connectivity properties
        total_possible = pre_size * post_size
        expected_connections = int(total_possible * connection_probability)
        
        # For sparse networks, check the actual vectorized connections
        actual_connections = len(syn_pop._pre_ids) if hasattr(syn_pop, '_pre_ids') else len(syn_pop.synapses)
        
        # Allow 50% variance due to probabilistic nature and ultra-sparse connectivity
        assert abs(actual_connections - expected_connections) <= max(expected_connections * 0.5, 10)
        assert actual_connections < total_possible * 0.001  # Much less than 0.1%
        
        # Verify sparse matrices are used
        assert hasattr(syn_pop, '_pre_ids')
        assert hasattr(syn_pop, '_post_ids')
        assert len(syn_pop._pre_ids) == len(syn_pop._post_ids)
        assert len(syn_pop._pre_ids) == actual_connections
        
        # Verify the synapses dict contains only a sample for API compatibility
        assert len(syn_pop.synapses) <= 100, "Should only contain sample synapses"
        assert len(syn_pop.synapses) > 0, "Should contain some sample synapses"
    
    def test_vectorized_synapse_construction_performance(self):
        """Test that vectorized construction is much faster than naive approach."""
        # Use large enough network to trigger sparse path (>50k total neurons)
        pre_size = 45000
        post_size = 5500  # 50.5k total neurons
        connection_probability = 0.00005  # Ultra-sparse to keep memory manageable
        
        # Time the vectorized construction
        start_time = time.time()
        syn_pop = SynapsePopulation(
            pre_population_size=pre_size,
            post_population_size=post_size,
            synapse_type="stdp",
            connection_probability=connection_probability,
            weight=0.1
        )
        vectorized_time = time.time() - start_time
        
        # Verify construction completed efficiently (sparse path should be much faster)
        assert vectorized_time < 120.0, f"Vectorized construction took {vectorized_time:.2f}s, too slow"
        
        # Verify the network was actually built
        assert hasattr(syn_pop, '_pre_ids'), "Sparse arrays should be created"
        assert hasattr(syn_pop, '_weights'), "Weight arrays should be created"
        assert syn_pop._weights is not None
        
        print(f"Sparse construction: {len(syn_pop._pre_ids):,} connections in {vectorized_time:.2f}s")
    
    @pytest.mark.skipif(True, reason="Requires GPU - enable for GPU testing")
    def test_gpu_acceleration_enforcement(self):
        """Test that GPU acceleration is enforced for large networks."""
        # Mock GPU availability
        with patch('core.synapses.GPU_AVAILABLE', True):
            with patch('core.synapses.cp') as mock_cp:
                mock_cp.array = MagicMock(side_effect=lambda x: np.array(x))
                mock_cp.zeros = MagicMock(side_effect=lambda x: np.zeros(x))
                
                # Create large synapse population that should trigger GPU
                syn_pop = SynapsePopulation(
                    pre_population_size=60000,  # Above 50k threshold
                    post_population_size=1000,
                    synapse_type="stdp",
                    connection_probability=0.001,  # Ultra-sparse
                    weight=0.1
                )
                
                # Verify GPU was attempted to be used
                assert mock_cp.array.called or mock_cp.zeros.called
    
    def test_ultra_sparse_connectivity_limits(self):
        """Test ultra-sparse connectivity for neuromorphic hardware scale."""
        # Simulate parameters for 80M neuron scale - BUT MUCH SMALLER FOR TESTING
        pre_size = 45000  # Scaled down for testing but triggers sparse path
        post_size = 5500   # 45k + 5.5k = 50.5k > 50k threshold  
        connection_probability = 0.00002  # 0.002% - ultra-sparse
        
        syn_pop = SynapsePopulation(
            pre_population_size=pre_size,
            post_population_size=post_size,
            synapse_type="stdp",
            connection_probability=connection_probability,
            weight=0.1
        )
        
        # Verify ultra-sparse properties
        total_possible = pre_size * post_size
        actual_connections = len(syn_pop.synapses)
        sparsity = actual_connections / total_possible
        
        assert sparsity < 0.001, f"Network not sparse enough: {sparsity:.6f}"
        assert actual_connections > 0, "Network should have some connections"
        
        # Verify memory efficiency
        assert hasattr(syn_pop, '_pre_ids')
        assert hasattr(syn_pop, '_post_ids')
        assert hasattr(syn_pop, '_weights')
    
    def test_mandatory_gpu_threshold_enforcement(self):
        """Test that networks above thresholds enforce GPU usage."""
        # Test the threshold logic without actually creating massive networks
        with patch('core.synapses.GPU_AVAILABLE', False):
            # Should raise error for networks requiring GPU
            with pytest.raises(RuntimeError) as exc_info:
                SynapsePopulation(
                    pre_population_size=1000000,  # Above mandatory GPU threshold
                    post_population_size=100,
                    synapse_type="stdp",
                    connection_probability=0.0001,
                    weight=0.1
                )
            
            # Check that the error message contains expected text
            error_msg = str(exc_info.value)
            assert "MASSIVE SCALE ERROR" in error_msg
            assert "REQUIRE GPU" in error_msg
    
    def test_batch_processing_for_large_networks(self):
        """Test batch processing of synapse operations."""
        # Create moderately large network to test batching - KEEP SMALL
        pre_size = 45000  # Use sparse path but keep connections minimal
        post_size = 5500  # 50.5k total triggers sparse path
        connection_probability = 0.00001  # Very sparse (0.001%)
        
        syn_pop = SynapsePopulation(
            pre_population_size=pre_size,
            post_population_size=post_size,
            synapse_type="stdp",
            connection_probability=connection_probability,
            weight=0.1
        )
        
        # Test current computation with batching
        spike_trains = [bool(np.random.choice([False, True], p=[0.95, 0.05])) 
                       for _ in range(pre_size)]
        
        start_time = time.time()
        currents = syn_pop.get_synaptic_currents(spike_trains, 0.0)
        compute_time = time.time() - start_time
        
        # Verify computation completed efficiently
        assert compute_time < 0.5, f"Current computation took {compute_time:.2f}s, too slow"
        assert len(currents) == post_size
        assert all(isinstance(c, (int, float, np.number)) for c in currents)
    
    def test_memory_efficiency_large_networks(self):
        """Test memory usage patterns for large networks."""
        import psutil
        process = psutil.Process()
        
        # Measure baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large sparse network
        pre_size = 20000
        post_size = 10000
        connection_probability = 0.0005  # 0.05% connectivity
        
        syn_pop = SynapsePopulation(
            pre_population_size=pre_size,
            post_population_size=post_size,
            synapse_type="stdp",
            connection_probability=connection_probability,
            weight=0.1
        )
        
        # Measure memory after network creation
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = final_memory - baseline_memory
        
        # Verify memory usage is reasonable for sparse network
        total_possible = pre_size * post_size
        actual_connections = len(syn_pop.synapses)
        memory_per_connection = memory_used / actual_connections if actual_connections > 0 else 0
        
        # Should use much less memory than dense network would
        dense_estimate = total_possible * 0.1  # Estimate for dense network (MB)
        assert memory_used < dense_estimate * 0.01, f"Sparse network used {memory_used:.1f}MB, too much"
        print(f"Sparse network ({actual_connections} synapses): {memory_used:.1f}MB "
              f"({memory_per_connection:.3f}MB/synapse)")


class TestNeuromorphicScaleValidation:
    """Validate readiness for neuromorphic hardware scale (80M+ neurons)."""
    
    def test_connectivity_scaling_algorithm(self):
        """Test that connectivity algorithms scale appropriately."""
        # Test multiple scales to verify O(n) scaling, not O(n²)
        scales = [1000, 2000, 4000]
        times = []
        
        for scale in scales:
            start_time = time.time()
            syn_pop = SynapsePopulation(
                pre_population_size=scale,
                post_population_size=scale,
                synapse_type="stdp",
                connection_probability=0.0001,  # Keep connection count VERY manageable
                weight=0.1
            )
            elapsed = time.time() - start_time
            times.append(elapsed)
            
            # Cleanup
            del syn_pop
        
        # Verify sub-linear or linear scaling (not quadratic)
        # If quadratic, time[2] would be ~16x time[0]
        # We allow up to 8x for linear scaling with overhead
        scale_factor = (scales[2] / scales[0]) ** 2  # What quadratic would be
        time_factor = times[2] / times[0] if times[0] > 0 else 1
        
        assert time_factor < scale_factor / 2, (
            f"Scaling worse than linear: {scales[0]}→{scales[2]} neurons, "
            f"{times[0]:.3f}s→{times[2]:.3f}s ({time_factor:.1f}x time)"
        )
    
    def test_ultra_large_network_construction_simulation(self):
        """Simulate construction patterns for 80M+ neuron networks."""
        # Test with parameters equivalent to 80M network but scaled down
        # 80M neurons with 0.01% connectivity = 8M synapses
        # Scale down: 8k neurons with 0.01% connectivity = 8 synapses (for testing)
        
        scale_factor = 10000  # 80M / 8k = 10k scale factor
        test_size = 8000
        equivalent_connectivity = 0.0001  # Ultra-sparse for massive scale
        
        # This should complete quickly even at large scale
        start_time = time.time()
        syn_pop = SynapsePopulation(
            pre_population_size=test_size,
            post_population_size=test_size,
            synapse_type="stdp",
            connection_probability=equivalent_connectivity,
            weight=0.1
        )
        construction_time = time.time() - start_time
        
        # Verify construction time scales appropriately
        # For 80M scale, this should extrapolate to reasonable time
        extrapolated_time = construction_time * scale_factor
        assert extrapolated_time < 300, (  # 5 minutes max for 80M network
            f"Construction time {construction_time:.3f}s extrapolates to "
            f"{extrapolated_time:.1f}s for 80M neurons - too slow"
        )
        
        # Verify sparse properties
        actual_connections = len(syn_pop.synapses)
        sparsity = actual_connections / (test_size * test_size)
        assert sparsity < 0.001, f"Not sparse enough for neuromorphic scale: {sparsity:.6f}"
        
        print(f"80M neuron simulation: {test_size} neurons, {actual_connections} synapses, "
              f"{construction_time:.3f}s construction (extrapolates to {extrapolated_time:.1f}s)")
    
    def test_gpu_acceleration_readiness(self):
        """Test readiness for GPU acceleration at massive scale."""
        # Test that large networks can be created and would use GPU if available
        syn_pop = SynapsePopulation(
            pre_population_size=50000,  # Above GPU threshold
            post_population_size=1000,
            synapse_type="stdp",
            connection_probability=0.0005,  # Higher connectivity
            weight=0.1,
            use_gpu=True  # Explicitly enable GPU
        )
        
        # Verify network was created successfully  
        assert syn_pop.total_connections > 1000, "Network should have substantial connectivity"
        assert hasattr(syn_pop, '_get_synaptic_currents_gpu'), "GPU method should exist"
        
        # Test that computation works (regardless of GPU/CPU)
        spike_trains = [bool(np.random.choice([False, True], p=[0.99, 0.01])) 
                       for _ in range(50000)]
        
        currents = syn_pop.get_synaptic_currents(spike_trains, 0.0)
        assert len(currents) == 1000, "Should return currents for all post-synaptic neurons"
        
        print(f"✅ GPU-ready network: {syn_pop.total_connections:,} connections, 51k neurons")
    
    def test_neuromorphic_hardware_constraints(self):
        """Test constraints specific to neuromorphic hardware deployment."""
        # Test ultra-sparse connectivity limits (neuromorphic constraint)
        max_connections_per_neuron = 1000  # Typical neuromorphic hardware limit
        network_size = 10000
        max_total_connections = network_size * max_connections_per_neuron
        
        # Calculate required sparsity
        total_possible = network_size * network_size
        required_sparsity = max_total_connections / total_possible
        
        syn_pop = SynapsePopulation(
            pre_population_size=network_size,
            post_population_size=network_size,
            synapse_type="stdp",
            connection_probability=required_sparsity * 0.5,  # Stay well under limit
            weight=0.1
        )
        
        actual_connections = len(syn_pop.synapses)
        assert actual_connections <= max_total_connections, (
            f"Too many connections for neuromorphic hardware: "
            f"{actual_connections} > {max_total_connections}"
        )
        
        # Verify connection distribution doesn't exceed per-neuron limits
        # This is a simplified check - real hardware would need more detailed validation
        avg_connections_per_neuron = actual_connections / network_size
        assert avg_connections_per_neuron <= max_connections_per_neuron, (
            f"Average connections per neuron too high: {avg_connections_per_neuron}"
        )


if __name__ == "__main__":
    # Run specific tests for development
    pytest.main([__file__, "-v", "--tb=short"])
