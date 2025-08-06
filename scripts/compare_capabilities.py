"""
Compare Neuromorphic System Capabilities
=======================================

This script compares the capabilities of Jetson Nano vs Desktop GPU
for neuromorphic computing, explaining the neuron count differences.
"""

import os
import sys

import psutil

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compare_system_capabilities():
    """Compare Jetson Nano vs Desktop GPU capabilities."""

    print("=== Neuromorphic System Capability Comparison ===")
    print()

    # Jetson Nano specifications
    print("JETSON NANO 8GB:")
    print("  • CPU: ARM Cortex-A57 quad-core @ 1.43 GHz")
    print("  • GPU: Maxwell-based 128 CUDA cores")
    print("  • Memory: 4GB LPDDR4 (shared CPU/GPU)")
    print("  • Power: 10W TDP")
    print("  • Storage: 16GB eMMC")
    print("  • Typical neuron capacity: 1,000 - 5,000 neurons")
    print("  • Typical synapse capacity: 10,000 - 50,000 synapses")
    print("  • Memory per neuron: ~1KB")
    print("  • Memory per synapse: ~0.5KB")
    print("  • Estimated memory usage: 2-4GB for 5k neurons")
    print()

    # Desktop GPU specifications (example: RTX 3080)
    print("DESKTOP GPU (RTX 3080 example):")
    print("  • CPU: Intel/AMD multi-core @ 3-4 GHz")
    print("  • GPU: Ampere-based 8,704 CUDA cores")
    print("  • Memory: 10GB GDDR6X (dedicated GPU)")
    print("  • Power: 320W TDP")
    print("  • Storage: SSD/NVMe")
    print("  • Typical neuron capacity: 50,000 - 500,000 neurons")
    print("  • Typical synapse capacity: 5,000,000 - 50,000,000 synapses")
    print("  • Memory per neuron: ~1KB")
    print("  • Memory per synapse: ~0.5KB")
    print("  • Estimated memory usage: 8-10GB for 50k neurons")
    print()

    # Current system info
    print("YOUR CURRENT SYSTEM:")
    cpu_count = os.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    print(f"  • CPU Cores: {cpu_count}")
    print(f"  • Total RAM: {memory_gb:.1f} GB")
    print(f"  • Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    print()

    # Calculate theoretical capacities
    available_ram_gb = psutil.virtual_memory().available / (1024**3)
    neuron_memory_kb = 1
    synapse_memory_kb = 0.5

    # Calculate capacity based on available RAM (using 80% to be safe)
    safe_ram_gb = available_ram_gb * 0.8
    max_neurons_ram = int((safe_ram_gb * 1024**3) / (neuron_memory_kb * 1024))
    max_synapses_ram = int((safe_ram_gb * 1024**3) / (synapse_memory_kb * 1024))

    print("THEORETICAL CAPACITIES (RAM-based):")
    print(f"  • Max neurons (RAM only): {max_neurons_ram:,}")
    print(f"  • Max synapses (RAM only): {max_synapses_ram:,}")
    print(
        f"  • Memory usage for 50k neurons: {(50000 * neuron_memory_kb) / 1024:.1f} GB"
    )
    print(
        f"  • Memory usage for 50k neurons + synapses: {((50000 * neuron_memory_kb) + (50000 * 100 * synapse_memory_kb)) / 1024:.1f} GB"
    )
    print()

    # GPU detection
    try:
        import cupy as cp

        print("GPU DETECTION:")
        print("  • CuPy available: YES")
        gpu_memory = cp.cuda.runtime.memGetInfo()
        gpu_total_gb = gpu_memory[1] / (1024**3)
        gpu_free_gb = gpu_memory[0] / (1024**3)
        print(f"  • GPU Memory Total: {gpu_total_gb:.1f} GB")
        print(f"  • GPU Memory Free: {gpu_free_gb:.1f} GB")

        # Calculate GPU-based capacity
        safe_gpu_gb = gpu_free_gb * 0.8
        max_neurons_gpu = int((safe_gpu_gb * 1024**3) / (neuron_memory_kb * 1024))
        max_synapses_gpu = int((safe_gpu_gb * 1024**3) / (synapse_memory_kb * 1024))

        print(f"  • Max neurons (GPU): {max_neurons_gpu:,}")
        print(f"  • Max synapses (GPU): {max_synapses_gpu:,}")

        # Combined capacity
        total_max_neurons = max_neurons_ram + max_neurons_gpu
        print(f"  • Combined max neurons: {total_max_neurons:,}")

    except ImportError:
        print("GPU DETECTION:")
        print("  • CuPy available: NO")
        print("  • GPU acceleration not available")
        print("  • Using CPU-only mode")
    except Exception as e:
        print(f"GPU DETECTION:")
        print(f"  • Error detecting GPU: {e}")
    print()

    # Why the difference?
    print("WHY THE DIFFERENCE?")
    print(
        "  1. Memory: Desktop GPUs have 8-24GB dedicated VRAM vs 4GB shared on Jetson"
    )
    print("  2. Processing: Desktop CPUs are 10-50x faster than ARM Cortex-A57")
    print("  3. GPU Cores: Desktop GPUs have 1000-10000x more CUDA cores")
    print("  4. Power: Desktop systems have 30-50x more power budget")
    print("  5. Cooling: Desktop systems have active cooling vs passive on Jetson")
    print("  6. Architecture: Desktop systems use x86-64 vs ARM64 on Jetson")
    print()

    # Recommendations
    print("RECOMMENDATIONS:")
    if "max_neurons_gpu" in locals() and max_neurons_gpu >= 50000:
        print("  ✓ Your system can handle 50k+ neurons!")
        print("  ✓ Use GPU acceleration for best performance")
        print("  ✓ Consider running even larger networks (100k+ neurons)")
    elif max_neurons_ram >= 50000:
        print("  ✓ Your system can handle 50k+ neurons with RAM!")
        print("  ✓ Install CuPy for GPU acceleration")
        print("  ✓ Consider adding more RAM for larger networks")
    else:
        print("  ⚠ Your system may struggle with 50k neurons")
        print("  ⚠ Consider reducing network size or upgrading hardware")
        print("  ⚠ Install CuPy for GPU acceleration if available")
    print()

    print("NEXT STEPS:")
    print("  1. Install GPU requirements: pip install -r requirements_gpu.txt")
    print("  2. Run GPU demo: python demo/gpu_large_scale_demo.py")
    print("  3. Test your system's capacity")
    print("  4. Scale up to 50k+ neurons on your GPU!")


if __name__ == "__main__":
    compare_system_capabilities()
