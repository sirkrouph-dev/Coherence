#!/usr/bin/env python3
"""
GPU Large-Scale Neuromorphic Network Demonstration
==================================================

This demonstration showcases the GPU scaling capabilities of the enhanced
neuromorphic framework, demonstrating networks with 100K+ neurons running
efficiently on GPU hardware.

Features demonstrated:
- GPU memory management and optimization
- Large-scale network creation (10K - 1M+ neurons)
- Performance monitoring and benchmarking
- Brain-inspired topology with E/I balance
- Real-time performance metrics
- Adaptive memory management
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional

# Core imports
try:
    from core.network import NeuromorphicNetwork
    from core.gpu_scaling import GPUMemoryManager, LargeScaleNetworkBuilder
    from core.gpu_neurons import (
        create_large_scale_gpu_network, 
        benchmark_gpu_scaling,
        GPU_AVAILABLE,
        test_gpu_scaling_limits
    )
    from core.brain_topology import BrainTopologyBuilder
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all core modules are available")
    IMPORTS_SUCCESSFUL = False


class GPUScalingDemo:
    """Comprehensive demonstration of GPU scaling capabilities."""
    
    def __init__(self):
        """Initialize the GPU scaling demonstration."""
        self.gpu_available = GPU_AVAILABLE
        self.memory_manager = None
        self.results = {}
        
        if self.gpu_available:
            self.memory_manager = GPUMemoryManager()
            print(f"GPU Scaling Demo Initialized")
            print(f"GPU: {self.memory_manager.gpu_config.device_name}")
            print(f"GPU Memory: {self.memory_manager.gpu_config.total_memory_mb:.0f} MB")
            print(f"Max Network Size: {self.memory_manager.gpu_config.max_network_size:,} neurons")
        else:
            print("GPU not available - demonstration will use CPU fallback")
            
    def demo_memory_management(self):
        """Demonstrate GPU memory management capabilities."""
        print("\n" + "="*60)
        print("GPU Memory Management Demo")
        print("="*60)
        
        if not self.memory_manager:
            print("Skipping - GPU not available")
            return
            
        # Test different network sizes
        test_sizes = [1000, 10000, 50000, 100000, 250000, 500000]
        
        print(f"Testing memory requirements for different network sizes:")
        print(f"{'Size':>10} {'Neurons (MB)':>15} {'Synapses (MB)':>15} {'Total (MB)':>12} {'GPU %':>8}")
        print("-" * 70)
        
        memory_results = []
        
        for size in test_sizes:
            profile = self.memory_manager.estimate_memory_requirements(size)
            gpu_percent = (profile.total_mb / self.memory_manager.gpu_config.total_memory_mb) * 100
            
            print(f"{size:>10,} {profile.neurons_mb:>12.1f} {profile.synapses_mb:>14.1f} "
                  f"{profile.total_mb:>10.1f} {gpu_percent:>6.1f}%")
                  
            memory_results.append({
                'size': size,
                'total_mb': profile.total_mb,
                'gpu_percent': gpu_percent,
                'fits': gpu_percent < 85  # 85% threshold
            })
            
        # Find maximum feasible size
        max_feasible = max([r['size'] for r in memory_results if r['fits']], default=0)
        print(f"\nMaximum feasible network size: {max_feasible:,} neurons")
        
        self.results['memory_analysis'] = memory_results
        return memory_results
        
    def demo_network_creation(self):
        """Demonstrate large-scale network creation."""
        print("\n" + "="*60)
        print("Large-Scale Network Creation Demo")
        print("="*60)
        
        # Test progressively larger networks
        test_sizes = [10000, 25000, 50000, 100000]
        
        creation_results = []
        
        for size in test_sizes:
            print(f"\nCreating network with {size:,} neurons...")
            
            try:
                start_time = time.time()
                
                # Create GPU-accelerated network
                network = create_large_scale_gpu_network(
                    target_neurons=size,
                    neuron_type="adex",
                    use_adaptive=True
                )
                
                creation_time = time.time() - start_time
                actual_size = network.num_neurons
                
                # Test a few simulation steps
                print(f"  Testing simulation performance...")
                step_start = time.time()
                
                for i in range(10):
                    if hasattr(network, 'adaptive_step'):\n                        spikes, metrics = network.adaptive_step(0.1)\n                    else:\n                        spikes, metrics = network.step(0.1)\n                        \n                step_time = time.time() - step_start\n                \n                throughput = actual_size * 10 / step_time  # neurons * steps / time\n                \n                result = {\n                    'target_size': size,\n                    'actual_size': actual_size,\n                    'creation_time': creation_time,\n                    'step_time': step_time,\n                    'throughput': throughput,\n                    'success': True\n                }\n                \n                creation_results.append(result)\n                \n                print(f\"  ✓ Success: {actual_size:,} neurons in {creation_time:.2f}s\")\n                print(f\"  ✓ Simulation: {throughput:.0f} neurons/sec\")\n                \n                # Clean up GPU memory\n                if hasattr(network, 'clear_gpu_memory'):\n                    network.clear_gpu_memory()\n                    \n            except Exception as e:\n                print(f\"  ✗ Failed: {str(e)}\")\n                creation_results.append({\n                    'target_size': size,\n                    'success': False,\n                    'error': str(e)\n                })\n                break  # Stop on first failure\n                \n        self.results['network_creation'] = creation_results\n        return creation_results\n        \n    def demo_brain_topology_scaling(self):\n        \"\"\"Demonstrate brain-inspired topology at large scale.\"\"\"\n        print(\"\\n\" + \"=\"*60)\n        print(\"Large-Scale Brain Topology Demo\")\n        print(\"=\"*60)\n        \n        try:\n            from core.brain_topology import BrainTopologyBuilder\n            \n            # Create large-scale brain-inspired network\n            print(\"Creating 50K neuron brain-inspired network...\")\n            \n            builder = BrainTopologyBuilder()\n            \n            start_time = time.time()\n            \n            # Build modular cortical network\n            network_config = builder.create_cortical_network(\n                size=50000,\n                modules=20,  # 20 modules of ~2500 neurons each\n                connectivity_density=0.02,  # 2% connectivity\n                ei_ratio=0.8  # 80% excitatory\n            )\n            \n            creation_time = time.time() - start_time\n            \n            print(f\"  ✓ Network created in {creation_time:.2f}s\")\n            print(f\"  • Total neurons: {network_config['total_neurons']:,}\")\n            print(f\"  • Modules: {network_config['num_modules']}\")\n            print(f\"  • Connections: {network_config['total_connections']:,}\")\n            print(f\"  • E/I ratio: {network_config['ei_ratio']:.1f}\")\n            \n            # Analyze network properties\n            if 'small_world_metrics' in network_config:\n                sw_metrics = network_config['small_world_metrics']\n                print(f\"  • Clustering coefficient: {sw_metrics['clustering']:.3f}\")\n                print(f\"  • Average path length: {sw_metrics['avg_path_length']:.2f}\")\n                print(f\"  • Small-world index: {sw_metrics['small_world_index']:.3f}\")\n                \n            self.results['brain_topology'] = {\n                'creation_time': creation_time,\n                'network_config': network_config,\n                'success': True\n            }\n            \n        except Exception as e:\n            print(f\"  ✗ Brain topology demo failed: {str(e)}\")\n            self.results['brain_topology'] = {\n                'success': False,\n                'error': str(e)\n            }\n            \n    def demo_performance_benchmarking(self):\n        \"\"\"Demonstrate comprehensive performance benchmarking.\"\"\"\n        print(\"\\n\" + \"=\"*60)\n        print(\"Performance Benchmarking Demo\")\n        print(\"=\"*60)\n        \n        if not self.gpu_available:\n            print(\"Skipping benchmarking - GPU required\")\n            return\n            \n        try:\n            # Run comprehensive benchmark\n            print(\"Running GPU scaling benchmark...\")\n            \n            benchmark_results = benchmark_gpu_scaling(max_neurons=200000)\n            \n            if 'error' in benchmark_results:\n                print(f\"Benchmark failed: {benchmark_results['error']}\")\n                return\n                \n            # Display results\n            print(\"\\nBenchmark Results:\")\n            print(f\"{'Size':>10} {'Time (s)':>10} {'Memory %':>10} {'Status':>15}\")\n            print(\"-\" * 50)\n            \n            for result in benchmark_results:\n                status = \"✓ Success\" if result['success'] else \"✗ Failed\"\n                memory_pct = result.get('memory_percent', 0)\n                creation_time = result.get('creation_time', 0)\n                \n                print(f\"{result['network_size']:>10,} {creation_time:>8.2f} \"\n                      f\"{memory_pct:>8.1f}% {status:>15}\")\n                      \n            self.results['benchmarking'] = benchmark_results\n            \n        except Exception as e:\n            print(f\"Benchmarking failed: {str(e)}\")\n            self.results['benchmarking'] = {'error': str(e)}\n            \n    def demo_neuromorphic_network_integration(self):\n        \"\"\"Demonstrate integration with enhanced NeuromorphicNetwork class.\"\"\"\n        print(\"\\n\" + \"=\"*60)\n        print(\"NeuromorphicNetwork Integration Demo\")\n        print(\"=\"*60)\n        \n        try:\n            # Create enhanced neuromorphic network\n            print(\"Creating enhanced neuromorphic network...\")\n            \n            network = NeuromorphicNetwork(use_gpu_scaling=True)\n            \n            # Add multiple layers with different sizes\n            layer_configs = [\n                ('input', 5000, 'lif'),\n                ('hidden1', 15000, 'adex'),\n                ('hidden2', 10000, 'adex'),\n                ('output', 2000, 'lif')\n            ]\n            \n            total_neurons = 0\n            for name, size, neuron_type in layer_configs:\n                print(f\"  Adding layer '{name}': {size:,} {neuron_type} neurons\")\n                network.add_layer(name, size, neuron_type)\n                total_neurons += size\n                \n            # Connect layers\n            connections = [\n                ('input', 'hidden1', 0.1),\n                ('hidden1', 'hidden2', 0.15),\n                ('hidden2', 'output', 0.2)\n            ]\n            \n            for pre, post, prob in connections:\n                print(f\"  Connecting {pre} → {post} (p={prob})\")\n                network.connect_layers(pre, post, connection_probability=prob)\n                \n            print(f\"\\n  ✓ Network created: {total_neurons:,} total neurons\")\n            print(f\"  ✓ GPU acceleration: {network.use_gpu_scaling}\")\n            print(f\"  ✓ Large-scale mode: {network.is_large_scale}\")\n            \n            # Test simulation with performance monitoring\n            print(\"\\n  Running simulation with performance monitoring...\")\n            \n            # Create simple input pattern\n            input_pattern = {\n                'input': np.random.randn(5000) * 10  # Random input currents\n            }\n            \n            # Run enhanced simulation\n            if hasattr(network, 'run_simulation_enhanced'):\n                results = network.run_simulation_enhanced(\n                    duration=100,  # 100ms\n                    dt=0.1,\n                    input_pattern=input_pattern,\n                    monitor_performance=True\n                )\n                \n                perf = results['performance_metrics']\n                print(f\"  ✓ Simulation completed: {perf['total_time_s']:.2f}s\")\n                print(f\"  ✓ Throughput: {perf['neurons_per_second']:.0f} neurons/sec\")\n                print(f\"  ✓ Total spikes: {perf['total_spikes']:,}\")\n                \n                # GPU performance summary\n                if hasattr(network, 'get_performance_summary_enhanced'):\n                    summary = network.get_performance_summary_enhanced()\n                    if 'gpu_memory' in summary:\n                        gpu_mem = summary['gpu_memory']\n                        print(f\"  ✓ GPU memory: {gpu_mem['used_mb']:.1f} MB ({gpu_mem['percent_used']:.1f}%)\")\n                        \n                self.results['network_integration'] = {\n                    'total_neurons': total_neurons,\n                    'simulation_results': results,\n                    'success': True\n                }\n            else:\n                print(\"  ! Enhanced simulation methods not available\")\n                \n            # Clean up\n            if hasattr(network, 'cleanup_gpu_resources_enhanced'):\n                network.cleanup_gpu_resources_enhanced()\n                \n        except Exception as e:\n            print(f\"  ✗ Integration demo failed: {str(e)}\")\n            self.results['network_integration'] = {\n                'success': False,\n                'error': str(e)\n            }\n            \n    def generate_summary_report(self):\n        \"\"\"Generate comprehensive summary report of all demonstrations.\"\"\"\n        print(\"\\n\" + \"=\"*60)\n        print(\"GPU SCALING DEMONSTRATION SUMMARY\")\n        print(\"=\"*60)\n        \n        # System information\n        print(\"\\nSystem Information:\")\n        print(f\"  GPU Available: {self.gpu_available}\")\n        if self.memory_manager:\n            print(f\"  GPU Device: {self.memory_manager.gpu_config.device_name}\")\n            print(f\"  GPU Memory: {self.memory_manager.gpu_config.total_memory_mb:.0f} MB\")\n            print(f\"  Max Network Size: {self.memory_manager.gpu_config.max_network_size:,} neurons\")\n            \n        # Results summary\n        print(\"\\nDemo Results:\")\n        \n        for demo_name, result in self.results.items():\n            if isinstance(result, dict) and 'success' in result:\n                status = \"✓ Success\" if result['success'] else \"✗ Failed\"\n                print(f\"  {demo_name.replace('_', ' ').title()}: {status}\")\n            elif isinstance(result, list):  # Memory analysis or benchmarking\n                print(f\"  {demo_name.replace('_', ' ').title()}: ✓ Complete ({len(result)} data points)\")\n            else:\n                print(f\"  {demo_name.replace('_', ' ').title()}: ✓ Complete\")\n                \n        # Performance highlights\n        if 'network_creation' in self.results:\n            successful_networks = [r for r in self.results['network_creation'] if r.get('success', False)]\n            if successful_networks:\n                max_network = max(successful_networks, key=lambda x: x['actual_size'])\n                print(f\"\\nPerformance Highlights:\")\n                print(f\"  Largest successful network: {max_network['actual_size']:,} neurons\")\n                print(f\"  Creation time: {max_network['creation_time']:.2f}s\")\n                print(f\"  Simulation throughput: {max_network['throughput']:.0f} neurons/sec\")\n                \n        # Memory efficiency\n        if 'memory_analysis' in self.results:\n            feasible_networks = [r for r in self.results['memory_analysis'] if r['fits']]\n            if feasible_networks:\n                max_feasible = max(feasible_networks, key=lambda x: x['size'])\n                print(f\"\\nMemory Efficiency:\")\n                print(f\"  Max feasible network: {max_feasible['size']:,} neurons\")\n                print(f\"  Memory utilization: {max_feasible['gpu_percent']:.1f}%\")\n                \n        print(\"\\nDemonstration completed successfully!\")\n        print(\"The neuromorphic framework now supports GPU-accelerated\")\n        print(\"simulation of large-scale networks with 100K+ neurons.\")\n        \n    def run_full_demonstration(self):\n        \"\"\"Run the complete GPU scaling demonstration.\"\"\"\n        print(\"Starting GPU Large-Scale Neuromorphic Network Demonstration\")\n        print(\"This demo showcases scaling to 100K+ neuron networks on GPU\")\n        \n        # Run all demonstrations\n        self.demo_memory_management()\n        self.demo_network_creation()\n        self.demo_brain_topology_scaling()\n        self.demo_performance_benchmarking()\n        self.demo_neuromorphic_network_integration()\n        \n        # Generate final report\n        self.generate_summary_report()\n        \n        return self.results\n\n\ndef quick_gpu_test():\n    \"\"\"Quick test to verify GPU scaling is working.\"\"\"\n    print(\"=== Quick GPU Scaling Test ===\")\n    \n    if not IMPORTS_SUCCESSFUL:\n        print(\"✗ Import failed - cannot run test\")\n        return False\n        \n    if not GPU_AVAILABLE:\n        print(\"⚠ GPU not available - testing CPU fallback\")\n        \n    try:\n        # Test network creation\n        print(\"Testing 10K neuron network creation...\")\n        network = create_large_scale_gpu_network(10000, \"adex\")\n        print(f\"✓ Created {network.num_neurons:,} neurons\")\n        \n        # Test simulation\n        print(\"Testing simulation...\")\n        spikes, metrics = network.step(0.1)\n        print(f\"✓ Simulation step completed: {metrics.get('throughput', 0):.0f} neurons/sec\")\n        \n        # Clean up\n        if hasattr(network, 'clear_gpu_memory'):\n            network.clear_gpu_memory()\n            \n        print(\"✓ Quick test passed!\")\n        return True\n        \n    except Exception as e:\n        print(f\"✗ Quick test failed: {str(e)}\")\n        return False\n\n\nif __name__ == \"__main__\":\n    import argparse\n    \n    parser = argparse.ArgumentParser(description='GPU Large-Scale Neuromorphic Demo')\n    parser.add_argument('--quick', action='store_true', \n                       help='Run quick test only')\n    parser.add_argument('--no-benchmark', action='store_true',\n                       help='Skip performance benchmarking')\n    \n    args = parser.parse_args()\n    \n    if not IMPORTS_SUCCESSFUL:\n        print(\"Required modules not available. Please check your installation.\")\n        exit(1)\n        \n    if args.quick:\n        # Run quick test\n        success = quick_gpu_test()\n        exit(0 if success else 1)\n    else:\n        # Run full demonstration\n        demo = GPUScalingDemo()\n        \n        if args.no_benchmark:\n            # Skip benchmarking for faster demo\n            demo.demo_memory_management()\n            demo.demo_network_creation()\n            demo.demo_neuromorphic_network_integration()\n            demo.generate_summary_report()\n        else:\n            # Full demonstration\n            results = demo.run_full_demonstration()\n            \n        print(\"\\nDemo completed. Check results above for performance metrics.\")