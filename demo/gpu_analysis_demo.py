"""
GPU Performance Analysis and Scaling Demo
==========================================

This script demonstrates and analyzes GPU performance for neuromorphic computing,
showing how to scale from thousands to millions of neurons.
"""

import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import psutil

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.gpu_neurons import GPUNeuronPool, MultiGPUNeuronSystem, analyze_gpu_performance

# Try to import GPU monitoring tools
try:
    import GPUtil
    GPU_MONITOR_AVAILABLE = True
except ImportError:
    GPU_MONITOR_AVAILABLE = False
    print("⚠ GPUtil not available, limited GPU monitoring")

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except:
    NVML_AVAILABLE = False
    print("⚠ pynvml not available, limited GPU metrics")


class GPUPerformanceAnalyzer:
    """Comprehensive GPU performance analysis for neuromorphic systems."""
    
    def __init__(self):
        self.results = {}
        self.gpu_info = self._get_gpu_info()
        
    def _get_gpu_info(self) -> Dict:
        """Get detailed GPU information."""
        info = {
            "available": False,
            "devices": [],
            "cuda_available": False,
            "total_memory": 0,
        }
        
        if GPU_MONITOR_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                info["available"] = len(gpus) > 0
                
                for gpu in gpus:
                    device_info = {
                        "id": gpu.id,
                        "name": gpu.name,
                        "memory_total": gpu.memoryTotal,
                        "memory_used": gpu.memoryUsed,
                        "memory_free": gpu.memoryFree,
                        "utilization": gpu.load * 100,
                        "temperature": gpu.temperature,
                    }
                    info["devices"].append(device_info)
                    info["total_memory"] += gpu.memoryTotal
                    
            except Exception as e:
                print(f"Error getting GPU info: {e}")
        
        # Check CUDA availability
        try:
            import torch
            info["cuda_available"] = torch.cuda.is_available()
            if info["cuda_available"]:
                info["cuda_device_count"] = torch.cuda.device_count()
                info["cuda_device_name"] = torch.cuda.get_device_name(0)
        except:
            pass
        
        return info
    
    def print_system_info(self):
        """Print comprehensive system information."""
        print("\n" + "="*70)
        print("SYSTEM INFORMATION")
        print("="*70)
        
        # CPU Information
        print("\n📊 CPU Information:")
        print(f"  Cores (Physical): {psutil.cpu_count(logical=False)}")
        print(f"  Cores (Logical): {psutil.cpu_count(logical=True)}")
        print(f"  Current Usage: {psutil.cpu_percent(interval=1)}%")
        print(f"  Frequency: {psutil.cpu_freq().current:.0f} MHz")
        
        # Memory Information
        mem = psutil.virtual_memory()
        print(f"\n💾 Memory Information:")
        print(f"  Total: {mem.total / (1024**3):.1f} GB")
        print(f"  Available: {mem.available / (1024**3):.1f} GB")
        print(f"  Used: {mem.used / (1024**3):.1f} GB ({mem.percent}%)")
        
        # GPU Information
        print(f"\n🎮 GPU Information:")
        if self.gpu_info["available"]:
            for device in self.gpu_info["devices"]:
                print(f"  GPU {device['id']}: {device['name']}")
                print(f"    Memory: {device['memory_used']:.0f}/{device['memory_total']:.0f} MB")
                print(f"    Utilization: {device['utilization']:.1f}%")
                print(f"    Temperature: {device['temperature']}°C")
        else:
            print("  No GPU detected or monitoring unavailable")
        
        if self.gpu_info["cuda_available"]:
            print(f"  CUDA: ✓ Available ({self.gpu_info['cuda_device_name']})")
        else:
            print("  CUDA: ✗ Not available")
        
        print("="*70 + "\n")
    
    def benchmark_scaling(self, scales: List[int] = None) -> Dict:
        """
        Benchmark neuromorphic system at different scales.
        
        Args:
            scales: List of neuron counts to test
        
        Returns:
            Benchmark results
        """
        if scales is None:
            scales = [1000, 5000, 10000, 25000, 50000, 100000]
        
        print("\n" + "="*70)
        print("SCALING BENCHMARK")
        print("="*70)
        
        results = {}
        
        for num_neurons in scales:
            print(f"\n🧠 Testing {num_neurons:,} neurons...")
            
            # Monitor initial state
            initial_cpu = psutil.cpu_percent(interval=0.1)
            initial_mem = psutil.virtual_memory().percent
            
            try:
                # Create neuron pool
                start_time = time.time()
                pool = GPUNeuronPool(
                    num_neurons=num_neurons,
                    neuron_type="adex",
                    use_gpu=True,
                    batch_size=min(10000, num_neurons),
                    precision="float32"
                )
                init_time = time.time() - start_time
                
                # Run simulation
                simulation_duration = 100.0  # 100ms
                dt = 0.1
                num_steps = int(simulation_duration / dt)
                
                step_times = []
                spike_counts = []
                gpu_metrics = []
                
                print(f"  Running {num_steps} simulation steps...")
                
                for step in range(num_steps):
                    # Generate input current
                    I_syn = np.random.randn(num_neurons) * 10
                    
                    # Step simulation
                    step_start = time.time()
                    spikes, metrics = pool.step(dt, I_syn)
                    step_times.append(time.time() - step_start)
                    spike_counts.append(len(spikes))
                    gpu_metrics.append(metrics)
                    
                    # Progress indicator
                    if step % 100 == 0:
                        print(f"    Step {step}/{num_steps}", end="\r")
                
                # Get final statistics
                stats = pool.get_spike_statistics()
                
                # Monitor final state
                final_cpu = psutil.cpu_percent(interval=0.1)
                final_mem = psutil.virtual_memory().percent
                
                # Calculate performance metrics
                total_time = sum(step_times)
                mean_step_time = np.mean(step_times)
                std_step_time = np.std(step_times)
                
                results[num_neurons] = {
                    "success": True,
                    "initialization_time": init_time,
                    "total_simulation_time": total_time,
                    "mean_step_time": mean_step_time,
                    "std_step_time": std_step_time,
                    "min_step_time": np.min(step_times),
                    "max_step_time": np.max(step_times),
                    "neurons_per_second": num_neurons / mean_step_time,
                    "total_spikes": stats["total_spikes"],
                    "mean_spike_rate": stats["mean_spike_rate"],
                    "active_neurons": stats["active_neurons"],
                    "silent_neurons": stats["silent_neurons"],
                    "cpu_usage_delta": final_cpu - initial_cpu,
                    "memory_usage_delta": final_mem - initial_mem,
                }
                
                # Add GPU metrics if available
                if gpu_metrics and "gpu_memory_used_mb" in gpu_metrics[-1]:
                    results[num_neurons]["gpu_memory_mb"] = gpu_metrics[-1]["gpu_memory_used_mb"]
                    results[num_neurons]["gpu_utilization"] = gpu_metrics[-1].get("gpu_utilization", -1)
                
                # Print summary
                print(f"\n  ✅ Success!")
                print(f"    Initialization: {init_time:.3f}s")
                print(f"    Simulation: {total_time:.3f}s")
                print(f"    Throughput: {results[num_neurons]['neurons_per_second']:,.0f} neurons/sec")
                print(f"    Total spikes: {stats['total_spikes']:,}")
                print(f"    Active neurons: {stats['active_neurons']:,}/{num_neurons:,}")
                
                # Cleanup
                pool.clear_gpu_memory()
                
            except Exception as e:
                print(f"\n  ❌ Failed: {e}")
                results[num_neurons] = {
                    "success": False,
                    "error": str(e)
                }
        
        self.results["scaling_benchmark"] = results
        return results
    
    def test_precision_impact(self) -> Dict:
        """Test impact of different numerical precisions."""
        print("\n" + "="*70)
        print("PRECISION IMPACT ANALYSIS")
        print("="*70)
        
        precisions = ["float16", "float32", "float64"]
        num_neurons = 10000
        results = {}
        
        for precision in precisions:
            print(f"\n🔢 Testing {precision} precision...")
            
            try:
                # Create neuron pool
                pool = GPUNeuronPool(
                    num_neurons=num_neurons,
                    neuron_type="adex",
                    use_gpu=True,
                    precision=precision
                )
                
                # Run simulation
                step_times = []
                for _ in range(100):
                    I_syn = np.random.randn(num_neurons) * 10
                    start = time.time()
                    spikes, metrics = pool.step(0.1, I_syn)
                    step_times.append(time.time() - start)
                
                stats = pool.get_spike_statistics()
                
                results[precision] = {
                    "mean_step_time": np.mean(step_times),
                    "std_step_time": np.std(step_times),
                    "neurons_per_second": num_neurons / np.mean(step_times),
                    "total_spikes": stats["total_spikes"],
                    "gpu_memory_mb": metrics.get("gpu_memory_used_mb", 0),
                }
                
                print(f"  ✅ Throughput: {results[precision]['neurons_per_second']:,.0f} neurons/sec")
                print(f"     Memory: {results[precision]['gpu_memory_mb']:.1f} MB")
                
                pool.clear_gpu_memory()
                
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                results[precision] = {"error": str(e)}
        
        self.results["precision_impact"] = results
        return results
    
    def test_neuron_types(self) -> Dict:
        """Compare performance of different neuron types."""
        print("\n" + "="*70)
        print("NEURON TYPE COMPARISON")
        print("="*70)
        
        neuron_types = ["lif", "adex", "izhikevich"]
        num_neurons = 25000
        results = {}
        
        for neuron_type in neuron_types:
            print(f"\n🧬 Testing {neuron_type.upper()} neurons...")
            
            try:
                # Create neuron pool
                pool = GPUNeuronPool(
                    num_neurons=num_neurons,
                    neuron_type=neuron_type,
                    use_gpu=True
                )
                
                # Run simulation
                step_times = []
                for _ in range(100):
                    I_syn = np.random.randn(num_neurons) * 10
                    start = time.time()
                    spikes, metrics = pool.step(0.1, I_syn)
                    step_times.append(time.time() - start)
                
                stats = pool.get_spike_statistics()
                
                results[neuron_type] = {
                    "mean_step_time": np.mean(step_times),
                    "neurons_per_second": num_neurons / np.mean(step_times),
                    "total_spikes": stats["total_spikes"],
                    "active_neurons": stats["active_neurons"],
                }
                
                print(f"  ✅ Throughput: {results[neuron_type]['neurons_per_second']:,.0f} neurons/sec")
                print(f"     Active neurons: {stats['active_neurons']:,}/{num_neurons:,}")
                
                pool.clear_gpu_memory()
                
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                results[neuron_type] = {"error": str(e)}
        
        self.results["neuron_types"] = results
        return results
    
    def test_massive_scale(self) -> Dict:
        """Test massive scale simulation with multi-GPU system."""
        print("\n" + "="*70)
        print("MASSIVE SCALE TEST (1M+ NEURONS)")
        print("="*70)
        
        try:
            # Create multi-GPU system
            print("\n🚀 Initializing 1 million neuron system...")
            multi_system = MultiGPUNeuronSystem(
                total_neurons=1000000,
                neurons_per_gpu=100000,
                neuron_types=["adex", "lif", "izhikevich"]
            )
            
            # Run simulation
            print("\n⚡ Running simulation (50ms)...")
            results = multi_system.simulate(duration=50.0, dt=0.1)
            
            print("\n📊 Results:")
            print(f"  Total neurons: {results['total_neurons']:,}")
            print(f"  Total spikes: {results['total_spikes']:,}")
            print(f"  Simulation time: {results['simulation_time']:.2f}s")
            print(f"  Throughput: {results['neurons_per_second']:,.0f} neurons/sec")
            print(f"  Spike rate: {results['spikes_per_second']:,.0f} spikes/sec")
            
            if "total_gpu_memory_mb" in results:
                print(f"  GPU memory: {results['total_gpu_memory_mb']:.1f} MB")
            
            # Cleanup
            multi_system.cleanup()
            
            self.results["massive_scale"] = results
            return results
            
        except Exception as e:
            print(f"\n❌ Massive scale test failed: {e}")
            self.results["massive_scale"] = {"error": str(e)}
            return {"error": str(e)}
    
    def visualize_results(self):
        """Create visualization of benchmark results."""
        if "scaling_benchmark" not in self.results:
            print("No benchmark results to visualize")
            return
        
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)
        
        # Prepare data
        scaling_data = self.results["scaling_benchmark"]
        successful_runs = {k: v for k, v in scaling_data.items() if v.get("success", False)}
        
        if not successful_runs:
            print("No successful runs to visualize")
            return
        
        neurons = sorted(successful_runs.keys())
        throughput = [successful_runs[n]["neurons_per_second"] for n in neurons]
        spike_rates = [successful_runs[n]["mean_spike_rate"] for n in neurons]
        step_times = [successful_runs[n]["mean_step_time"] * 1000 for n in neurons]  # Convert to ms
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("GPU Neuromorphic System Performance Analysis", fontsize=16, fontweight='bold')
        
        # Plot 1: Throughput scaling
        ax1 = axes[0, 0]
        ax1.plot(neurons, throughput, 'b-o', linewidth=2, markersize=8)
        ax1.set_xlabel("Number of Neurons")
        ax1.set_ylabel("Throughput (neurons/second)")
        ax1.set_title("Computational Throughput Scaling")
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Step time
        ax2 = axes[0, 1]
        ax2.plot(neurons, step_times, 'r-s', linewidth=2, markersize=8)
        ax2.set_xlabel("Number of Neurons")
        ax2.set_ylabel("Mean Step Time (ms)")
        ax2.set_title("Simulation Step Time")
        ax2.set_xscale('log')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Spike activity
        ax3 = axes[1, 0]
        ax3.bar(range(len(neurons)), spike_rates, color='green', alpha=0.7)
        ax3.set_xlabel("Configuration")
        ax3.set_ylabel("Mean Spike Rate")
        ax3.set_title("Neural Activity")
        ax3.set_xticks(range(len(neurons)))
        ax3.set_xticklabels([f"{n//1000}k" for n in neurons], rotation=45)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Efficiency metrics
        ax4 = axes[1, 1]
        if any("gpu_memory_mb" in successful_runs[n] for n in neurons):
            memory_usage = [successful_runs[n].get("gpu_memory_mb", 0) for n in neurons]
            ax4.plot(neurons, memory_usage, 'purple', marker='D', linewidth=2, markersize=8)
            ax4.set_xlabel("Number of Neurons")
            ax4.set_ylabel("GPU Memory (MB)")
            ax4.set_title("GPU Memory Usage")
            ax4.set_xscale('log')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, "GPU Memory Data\nNot Available", 
                    ha='center', va='center', fontsize=12)
            ax4.set_title("GPU Memory Usage")
        
        plt.tight_layout()
        
        # Save figure
        output_path = "gpu_performance_analysis.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n📈 Visualization saved to: {output_path}")
        
        plt.show()
    
    def save_results(self, filename: str = "gpu_analysis_results.json"):
        """Save analysis results to JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\n💾 Results saved to: {filename}")
    
    def generate_report(self):
        """Generate comprehensive performance report."""
        print("\n" + "="*70)
        print("PERFORMANCE ANALYSIS REPORT")
        print("="*70)
        
        if "scaling_benchmark" in self.results:
            print("\n📊 Scaling Performance:")
            scaling_data = self.results["scaling_benchmark"]
            for neurons, data in scaling_data.items():
                if data.get("success", False):
                    print(f"  {neurons:,} neurons: {data['neurons_per_second']:,.0f} neurons/sec")
        
        if "precision_impact" in self.results:
            print("\n🔢 Precision Impact:")
            for precision, data in self.results["precision_impact"].items():
                if "error" not in data:
                    print(f"  {precision}: {data['neurons_per_second']:,.0f} neurons/sec")
        
        if "neuron_types" in self.results:
            print("\n🧬 Neuron Type Performance:")
            for ntype, data in self.results["neuron_types"].items():
                if "error" not in data:
                    print(f"  {ntype.upper()}: {data['neurons_per_second']:,.0f} neurons/sec")
        
        if "massive_scale" in self.results:
            print("\n🚀 Massive Scale Capability:")
            data = self.results["massive_scale"]
            if "error" not in data:
                print(f"  Successfully simulated {data['total_neurons']:,} neurons")
                print(f"  Throughput: {data['neurons_per_second']:,.0f} neurons/sec")
        
        print("\n" + "="*70)


def main():
    """Main demonstration function."""
    print("\n🧠 GPU-ACCELERATED NEUROMORPHIC SYSTEM ANALYSIS")
    print("="*70)
    
    # Initialize analyzer
    analyzer = GPUPerformanceAnalyzer()
    
    # Print system information
    analyzer.print_system_info()
    
    # Run benchmarks
    print("\n🏃 Running performance benchmarks...")
    
    # 1. Scaling benchmark
    scales = [1000, 5000, 10000, 25000, 50000, 100000, 250000, 500000]
    analyzer.benchmark_scaling(scales)
    
    # 2. Precision impact test
    analyzer.test_precision_impact()
    
    # 3. Neuron type comparison
    analyzer.test_neuron_types()
    
    # 4. Massive scale test
    analyzer.test_massive_scale()
    
    # Generate visualizations
    analyzer.visualize_results()
    
    # Generate report
    analyzer.generate_report()
    
    # Save results
    analyzer.save_results()
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
