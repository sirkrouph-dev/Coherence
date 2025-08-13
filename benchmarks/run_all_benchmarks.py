"""
NeuroMorph Master Benchmark Suite
Comprehensive performance testing for all components
"""

import os
import sys
import time
import importlib.util
from pathlib import Path


class BenchmarkRunner:
    """Master benchmark runner for NeuroMorph."""
    
    def __init__(self):
        self.benchmark_dir = Path(__file__).parent
        self.results = {}
        
    def run_all_benchmarks(self):
        """Run all available benchmarks."""
        print("=" * 80)
        print("NEUROMORPH COMPREHENSIVE BENCHMARK SUITE")
        print("=" * 80)
        
        benchmarks = [
            ("CPU Performance", "cpu_performance_benchmark.py"),
            ("High Performance Architecture", "high_performance_benchmark.py"),
            ("GPU RTX 3060", "gpu_benchmark_rtx3060.py"),
            ("Performance Benchmarks", "performance_benchmarks.py"),
        ]
        
        for name, filename in benchmarks:
            print(f"\n🔬 Running {name} Benchmark...")
            print("-" * 60)
            
            try:
                result = self._run_benchmark(filename)
                self.results[name] = {"status": "success", "result": result}
                print(f"✅ {name} benchmark completed successfully")
                
            except Exception as e:
                self.results[name] = {"status": "failed", "error": str(e)}
                print(f"❌ {name} benchmark failed: {e}")
                
            print()
        
        self._print_summary()
    
    def _run_benchmark(self, filename):
        """Run a specific benchmark file."""
        file_path = self.benchmark_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {filename}")
        
        # Load and execute the benchmark module
        spec = importlib.util.spec_from_file_location("benchmark", file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load benchmark module: {filename}")
            
        benchmark_module = importlib.util.module_from_spec(spec)
        
        # Add the benchmarks directory to path
        sys.path.insert(0, str(self.benchmark_dir.parent))
        
        try:
            # Capture output by temporarily redirecting stdout
            import io
            import contextlib
            
            output = io.StringIO()
            
            with contextlib.redirect_stdout(output):
                spec.loader.exec_module(benchmark_module)
            
            return output.getvalue()
        finally:
            # Remove from path
            if str(self.benchmark_dir.parent) in sys.path:
                sys.path.remove(str(self.benchmark_dir.parent))
    
    def _print_summary(self):
        """Print benchmark summary."""
        print("=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)
        
        total = len(self.results)
        successful = sum(1 for r in self.results.values() if r["status"] == "success")
        failed = total - successful
        
        print(f"Total Benchmarks: {total}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print()
        
        for name, result in self.results.items():
            status_icon = "✅" if result["status"] == "success" else "❌"
            print(f"{status_icon} {name}: {result['status']}")
            if result["status"] == "failed":
                print(f"   Error: {result['error']}")
        
        print()
        success_rate = (successful / total) * 100
        print(f"Success Rate: {success_rate:.1f}%")
        
        if success_rate == 100:
            print("🎉 All benchmarks passed! NeuroMorph is performing excellently.")
        elif success_rate >= 75:
            print("🎯 Most benchmarks passed. Check failed ones for optimization opportunities.")
        else:
            print("⚠️  Multiple benchmark failures. System needs attention.")


def main():
    """Main benchmark runner."""
    runner = BenchmarkRunner()
    
    print("NeuroMorph Benchmark Suite")
    print("=========================")
    print()
    
    # Quick system info
    try:
        import psutil
        import platform
        
        print(f"System: {platform.system()} {platform.release()}")
        print(f"CPU: {platform.processor()}")
        print(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f}GB")
        
        # GPU info if available
        try:
            import cupy as cp
            if cp.cuda.is_available():
                device_info = cp.cuda.Device()
                free_mem, total_mem = device_info.mem_info
                device_name = cp.cuda.runtime.getDeviceProperties(device_info.id)['name'].decode()
                print(f"GPU: {device_name} ({total_mem/(1024**3):.1f}GB)")
            else:
                print("GPU: Not available")
        except ImportError:
            print("GPU: CuPy not installed")
            
    except ImportError:
        print("System info: psutil not available")
    
    print()
    
    # Run benchmarks
    runner.run_all_benchmarks()


if __name__ == "__main__":
    main()
