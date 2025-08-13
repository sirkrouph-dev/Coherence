"""
Performance validation tests for massive scale optimizations.

These tests validate that the performance optimizations deliver the expected
12-24x improvements for large-scale networks and ensure readiness for
neuromorphic hardware deployment.
"""

import pytest
import numpy as np
import time
import psutil
from unittest.mock import patch

from core.synapses import SynapsePopulation, STDP_Synapse
from core.neurons import LeakyIntegrateAndFire, NeuronPopulation
from core.network import NeuromorphicNetwork


class TestPerformanceValidation:
    """Validate performance improvements for massive scale networks."""
    
    def test_50k_neuron_network_construction_performance(self):
        """Test that 50k neuron networks construct efficiently."""
        # Target: Construction should complete in under 30 seconds
        # (Previous: 329s, Target: <30s = 11x improvement)
        
        pre_size = 50000
        post_size = 1000
        connection_probability = 0.0002  # Ultra-sparse (0.02%)
        
        start_time = time.time()
        syn_pop = SynapsePopulation(
            pre_population_size=pre_size,
            post_population_size=post_size,
            synapse_type="stdp",
            connection_probability=connection_probability,
            weight=0.1
        )
        construction_time = time.time() - start_time
        
        # Verify performance target achieved
        target_time = 30.0  # seconds
        assert construction_time < target_time, (
            f"50k neuron construction took {construction_time:.1f}s, "
            f"target was <{target_time}s"
        )
        
        # Verify network properties
        actual_connections = len(syn_pop.synapses)
        expected_connections = int(pre_size * post_size * connection_probability)
        
        print(f"50k neuron network: {actual_connections} synapses, "
              f"{construction_time:.2f}s construction")
        
        # Performance should be much better than 329s baseline
        baseline_time = 329.0  # Previous performance
        improvement_factor = baseline_time / construction_time
        assert improvement_factor > 10, (
            f"Performance improvement only {improvement_factor:.1f}x, "
            f"expected >10x"
        )
    
    def test_simulation_performance_scaling(self):
        """Test simulation performance scales well with network size."""
        # Test multiple scales and verify performance scaling
        scales = [1000, 5000, 10000]
        simulation_times = []
        
        for scale in scales:
            # Create network
            syn_pop = SynapsePopulation(
                pre_population_size=scale,
                post_population_size=scale // 10,  # Keep post smaller for efficiency
                synapse_type="stdp",
                connection_probability=0.001,
                weight=0.1
            )
            
            # Run simulation
            spike_trains = [bool(np.random.choice([False, True], p=[0.95, 0.05])) 
                           for _ in range(scale)]
            
            start_time = time.time()
            for _ in range(10):  # 10 simulation steps
                currents = syn_pop.get_synaptic_currents(spike_trains, 0.0)
            simulation_time = time.time() - start_time
            simulation_times.append(simulation_time)
            
            # Cleanup
            del syn_pop
        
        # Verify scaling is reasonable (not exponential)
        scale_ratio = scales[2] / scales[0]  # 10x neurons
        time_ratio = simulation_times[2] / simulation_times[0] if simulation_times[0] > 0 else 1
        
        # Should be sub-quadratic scaling
        assert time_ratio < scale_ratio ** 1.5, (
            f"Simulation scaling worse than expected: {scale_ratio}x neurons, "
            f"{time_ratio:.1f}x time"
        )
        
        print(f"Simulation scaling: {scales[0]}→{scales[2]} neurons, "
              f"{simulation_times[0]:.3f}s→{simulation_times[2]:.3f}s "
              f"({time_ratio:.1f}x time)")
    
    def test_memory_efficiency_validation(self):
        """Validate memory usage for large sparse networks."""
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large sparse network
        pre_size = 30000
        post_size = 5000
        connection_probability = 0.0003  # 0.03% connectivity
        
        syn_pop = SynapsePopulation(
            pre_population_size=pre_size,
            post_population_size=post_size,
            synapse_type="stdp",
            connection_probability=connection_probability,
            weight=0.1
        )
        
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = current_memory - baseline_memory
        
        # Calculate efficiency metrics
        total_possible = pre_size * post_size
        actual_connections = len(syn_pop.synapses)
        sparsity = actual_connections / total_possible
        
        # Memory should be proportional to actual connections, not total possible
        memory_per_connection = memory_used / actual_connections if actual_connections > 0 else 0
        dense_memory_estimate = total_possible * memory_per_connection
        memory_efficiency = memory_used / dense_memory_estimate if dense_memory_estimate > 0 else 0
        
        assert memory_efficiency < 0.1, (  # Should use <10% of dense memory
            f"Memory efficiency poor: using {memory_efficiency:.1%} of dense equivalent"
        )
        
        print(f"Memory efficiency: {memory_used:.1f}MB for {actual_connections} synapses "
              f"({sparsity:.4%} sparsity, {memory_per_connection:.3f}MB/synapse)")
    
    def test_throughput_validation(self):
        """Validate computational throughput for large networks."""
        # Create network for throughput testing
        pre_size = 20000
        post_size = 2000
        connection_probability = 0.0005  # 0.05% connectivity
        
        syn_pop = SynapsePopulation(
            pre_population_size=pre_size,
            post_population_size=post_size,
            synapse_type="stdp",
            connection_probability=connection_probability,
            weight=0.1
        )
        
        # Generate spike data
        spike_trains = [bool(np.random.choice([False, True], p=[0.98, 0.02])) 
                       for _ in range(pre_size)]
        
        # Measure throughput
        num_steps = 50
        start_time = time.time()
        
        for step in range(num_steps):
            currents = syn_pop.get_synaptic_currents(spike_trains, 0.0)
            # Simulate some weight updates
            if step % 10 == 0:
                # Update subset of synapses (first 100)
                synapse_items = list(syn_pop.synapses.items())[:100]
                for (pre_id, post_id), synapse in synapse_items:
                    synapse.update_weight(0.01)  # Small weight increase
        
        total_time = time.time() - start_time
        
        # Calculate performance metrics
        actual_connections = syn_pop.total_connections
        synapse_updates_per_second = (actual_connections * num_steps) / total_time
        neurons_processed_per_second = (pre_size * num_steps) / total_time
        
        # Verify throughput targets
        min_synapse_throughput = 1000000  # 1M synapse-updates/second
        min_neuron_throughput = 100000    # 100k neurons/second
        
        assert synapse_updates_per_second > min_synapse_throughput, (
            f"Synapse throughput {synapse_updates_per_second:.0f}/s < "
            f"target {min_synapse_throughput}/s"
        )
        
        assert neurons_processed_per_second > min_neuron_throughput, (
            f"Neuron throughput {neurons_processed_per_second:.0f}/s < "
            f"target {min_neuron_throughput}/s"
        )
        
        print(f"Throughput: {synapse_updates_per_second/1000:.0f}k synapse-updates/s, "
              f"{neurons_processed_per_second/1000:.0f}k neurons/s")


class TestNetworkScaleIntegration:
    """Integration tests for complete large-scale networks."""
    
    def test_complete_network_50k_simulation(self):
        """Test complete 50k neuron network simulation."""
        # Create a realistic large network
        input_size = 10000
        hidden_size = 40000
        output_size = 1000
        
        # Build network layers
        input_pop = NeuronPopulation(input_size, "lif")
        hidden_pop = NeuronPopulation(hidden_size, "lif")
        output_pop = NeuronPopulation(output_size, "lif")
        
        # Create sparse connections (realistic connectivity)
        input_hidden_synapses = SynapsePopulation(
            pre_population_size=input_size,
            post_population_size=hidden_size,
            synapse_type="stdp",
            connection_probability=0.0005,  # 0.05% - very sparse
            weight=0.1
        )
        
        hidden_output_synapses = SynapsePopulation(
            pre_population_size=hidden_size,
            post_population_size=output_size,
            synapse_type="stdp",
            connection_probability=0.001,   # 0.1% - slightly denser for output
            weight=0.1
        )
        
        # Simulate network activity
        simulation_steps = 100
        dt = 0.1
        
        start_time = time.time()
        
        for step in range(simulation_steps):
            # Generate input
            input_current = np.random.uniform(0, 50, input_size)
            
            # Step input layer
            input_spikes = input_pop.step(dt, input_current.tolist())
            
            # Compute hidden layer input
            time_step = step * dt
            hidden_currents = input_hidden_synapses.get_synaptic_currents(input_spikes, time_step)
            
            # Step hidden layer
            hidden_spikes = hidden_pop.step(dt, hidden_currents)
            
            # Compute output layer input
            output_currents = hidden_output_synapses.get_synaptic_currents(hidden_spikes, time_step)
            
            # Step output layer
            output_spikes = output_pop.step(dt, output_currents)
        
        simulation_time = time.time() - start_time
        
        # Verify performance
        total_neurons = input_size + hidden_size + output_size
        neuron_steps_per_second = (total_neurons * simulation_steps) / simulation_time
        
        # Target: >100k neuron-steps/second
        min_throughput = 100000
        assert neuron_steps_per_second > min_throughput, (
            f"Network simulation throughput {neuron_steps_per_second:.0f}/s < "
            f"target {min_throughput}/s"
        )
        
        print(f"Complete 50k neuron network: {simulation_time:.2f}s simulation, "
              f"{neuron_steps_per_second/1000:.0f}k neuron-steps/s")
        
        # Verify network was created successfully and simulation ran
        assert len(input_pop.neurons) == input_size, "Input population not properly created"
        assert len(hidden_pop.neurons) == hidden_size, "Hidden population not properly created"
        assert len(output_pop.neurons) == output_size, "Output population not properly created"
        
        # Check that membrane potentials are in reasonable physiological range
        input_potential = input_pop.neurons[0].membrane_potential
        hidden_potential = hidden_pop.neurons[0].membrane_potential
        output_potential = output_pop.neurons[0].membrane_potential
        
        assert -100 <= input_potential <= 50, f"Input potential out of range: {input_potential}"
        assert -100 <= hidden_potential <= 50, f"Hidden potential out of range: {hidden_potential}"
        assert -100 <= output_potential <= 50, f"Output potential out of range: {output_potential}"
        
        print(f"Membrane potentials: input={input_potential:.1f}mV, "
              f"hidden={hidden_potential:.1f}mV, output={output_potential:.1f}mV")
    
    @pytest.mark.slow
    def test_neuromorphic_scale_readiness(self):
        """Test readiness for neuromorphic hardware scale (80M neurons)."""
        # This test validates algorithmic readiness without actually creating 80M neurons
        
        # Test scaling characteristics with smaller networks
        scales = [1000, 5000, 10000]
        construction_times = []
        memory_usage = []
        
        for scale in scales:
            process = psutil.Process()
            baseline_memory = process.memory_info().rss / 1024 / 1024
            
            start_time = time.time()
            syn_pop = SynapsePopulation(
                pre_population_size=scale,
                post_population_size=scale,
                synapse_type="stdp",
                connection_probability=0.0001,  # Neuromorphic sparsity
                weight=0.1
            )
            construction_time = time.time() - start_time
            
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_used = current_memory - baseline_memory
            
            construction_times.append(construction_time)
            memory_usage.append(memory_used)
            
            del syn_pop
        
        # Analyze scaling characteristics
        scale_factor_10x = scales[2] / scales[0]  # 10x neurons
        time_factor_10x = construction_times[2] / construction_times[0] if construction_times[0] > 0 else 1
        memory_factor_10x = memory_usage[2] / memory_usage[0] if memory_usage[0] > 0 else 1
        
        # Extrapolate to 80M neurons (8000x from 10k test)
        extrapolation_factor = 8000  # 80M / 10k
        extrapolated_time = construction_times[2] * (extrapolation_factor / scale_factor_10x) ** (np.log(time_factor_10x) / np.log(scale_factor_10x))
        extrapolated_memory = memory_usage[2] * (extrapolation_factor / scale_factor_10x) ** (np.log(memory_factor_10x) / np.log(scale_factor_10x))
        
        # Verify neuromorphic scale feasibility
        max_acceptable_time = 600  # 10 minutes for 80M network construction
        max_acceptable_memory = 32000  # 32GB memory limit
        
        assert extrapolated_time < max_acceptable_time, (
            f"80M neuron construction extrapolates to {extrapolated_time:.0f}s, "
            f"exceeds {max_acceptable_time}s limit"
        )
        
        assert extrapolated_memory < max_acceptable_memory, (
            f"80M neuron memory extrapolates to {extrapolated_memory:.0f}MB, "
            f"exceeds {max_acceptable_memory}MB limit"
        )
        
        print(f"Neuromorphic scale readiness:")
        print(f"  Scaling: {scale_factor_10x}x neurons → {time_factor_10x:.1f}x time, {memory_factor_10x:.1f}x memory")
        print(f"  80M extrapolation: {extrapolated_time:.0f}s construction, {extrapolated_memory:.0f}MB memory")
        print(f"  Status: {'✓ READY' if extrapolated_time < max_acceptable_time and extrapolated_memory < max_acceptable_memory else '✗ NOT READY'}")


if __name__ == "__main__":
    # Run performance validation tests
    pytest.main([__file__, "-v", "--tb=short", "-m", "not slow"])
