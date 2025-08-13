"""
Fast tests for massive scale neuromorphic computing features.

This module validates the optimizations implemented for large-scale
spiking neural networks with FAST execution times.
"""

import pytest
import numpy as np
import time
import re
from unittest.mock import patch, MagicMock

from core.synapses import SynapsePopulation


class TestMassiveScaleFast:
    """Fast tests for massive scale optimizations."""
    
    def test_sparse_path_trigger(self):
        """Test that sparse path is triggered correctly."""
        # Small test that triggers sparse path (55k total)
        pre_size = 50000
        post_size = 5000  # 50k + 5k = 55k > 50k threshold
        connection_probability = 0.00001  # Ultra-sparse (0.001%)
        
        start_time = time.time()
        syn_pop = SynapsePopulation(
            pre_population_size=pre_size,
            post_population_size=post_size,
            synapse_type="stdp",
            connection_probability=connection_probability,
            weight=0.1
        )
        construction_time = time.time() - start_time
        
        # Should be very fast with sparse path
        assert construction_time < 2.0, f"Sparse construction took {construction_time:.2f}s - too slow!"
        
        # Verify sparse arrays exist
        assert hasattr(syn_pop, '_pre_ids')
        assert hasattr(syn_pop, '_post_ids')
        assert hasattr(syn_pop, '_weights')
        
        # Verify reasonable number of connections
        connections = len(syn_pop._pre_ids)
        expected = int(pre_size * post_size * connection_probability)
        assert connections > 0
        assert connections < expected * 5  # Allow variance but not explosion
        
        print(f"✓ Sparse path: {connections:,} connections in {construction_time:.3f}s")
    
    def test_dense_vs_sparse_threshold(self):
        """Test that threshold correctly switches between dense and sparse."""
        # Test just below threshold (dense path)
        small_pre = 40000
        small_post = 9000  # 49k total < 50k threshold
        
        start_time = time.time()
        small_syn = SynapsePopulation(
            pre_population_size=small_pre,
            post_population_size=small_post,
            synapse_type="stdp",
            connection_probability=0.00001,  # Very sparse
            weight=0.1
        )
        dense_time = time.time() - start_time
        
        # Test just above threshold (sparse path)
        large_pre = 45000
        large_post = 5500  # 50.5k total > 50k threshold
        
        start_time = time.time()
        large_syn = SynapsePopulation(
            pre_population_size=large_pre,
            post_population_size=large_post,
            synapse_type="stdp",
            connection_probability=0.00001,  # Same sparsity
            weight=0.1
        )
        sparse_time = time.time() - start_time
        
        # Dense path should have individual synapse objects
        assert isinstance(small_syn.synapses, dict)
        assert len(small_syn.synapses) > 10  # Has actual synapse objects
        
        # Sparse path should have vectorized arrays
        assert hasattr(large_syn, '_pre_ids')
        assert hasattr(large_syn, '_post_ids')
        assert len(large_syn.synapses) <= 100  # Only sample objects
        
        print(f"✓ Dense ({small_pre}+{small_post}): {dense_time:.3f}s")
        print(f"✓ Sparse ({large_pre}+{large_post}): {sparse_time:.3f}s")
        print(f"✓ Threshold working correctly!")
    
    def test_gpu_threshold_mock(self):
        """Test GPU threshold without actually using GPU."""
        with patch('core.synapses.GPU_AVAILABLE', False):
            # Should raise error for mandatory GPU threshold
            with pytest.raises(RuntimeError) as exc_info:
                SynapsePopulation(
                    pre_population_size=1000000,
                    post_population_size=100,
                    synapse_type="stdp",
                    connection_probability=0.0001,
                    weight=0.1
                )
            # Check that the error message contains the expected text
            error_msg = str(exc_info.value)
            assert "MASSIVE SCALE ERROR" in error_msg
            assert "REQUIRE GPU" in error_msg
            assert "1,000,100" in error_msg
    
    def test_memory_efficiency_validation(self):
        """Test memory efficiency without huge allocations."""
        # Test that sparse representation uses much less memory
        import psutil
        process = psutil.Process()
        baseline = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create sparse network
        syn_pop = SynapsePopulation(
            pre_population_size=50000,
            post_population_size=5000,
            synapse_type="stdp",
            connection_probability=0.00001,  # Ultra-sparse
            weight=0.1
        )
        
        final = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = final - baseline
        
        # Should use reasonable memory
        assert memory_used < 100, f"Used {memory_used:.1f}MB - too much for sparse network"
        
        connections = len(syn_pop._pre_ids)
        print(f"✓ Memory test: {connections:,} connections used {memory_used:.1f}MB")
    
    def test_neuromorphic_scale_readiness(self):
        """Test readiness for 80M neuron scale with extrapolation."""
        # Test small representative sample
        test_size = 50000
        post_size = 5000
        connection_prob = 0.00001  # 0.001% - neuromorphic scale
        
        start_time = time.time()
        syn_pop = SynapsePopulation(
            pre_population_size=test_size,
            post_population_size=post_size,
            synapse_type="stdp",
            connection_probability=connection_prob,
            weight=0.1
        )
        construction_time = time.time() - start_time
        
        # Extrapolate to 80M scale
        scale_factor = 80_000_000 / test_size  # 1600x scale factor
        estimated_80m_time = construction_time * scale_factor
        
        # For 80M neurons, should be manageable
        assert estimated_80m_time < 3600, f"80M extrapolation: {estimated_80m_time:.1f}s too slow"
        
        connections = len(syn_pop._pre_ids)
        print(f"✓ 80M readiness: {test_size:,} neurons → {estimated_80m_time:.1f}s estimated for 80M")
        print(f"✓ {connections:,} connections built in {construction_time:.3f}s")
        print(f"✓ Architecture ready for neuromorphic hardware!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
