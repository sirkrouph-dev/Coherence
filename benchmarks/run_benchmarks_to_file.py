#!/usr/bin/env python3
"""
Benchmark Runner with File Output
=================================

Runs neuromorphic benchmarks and writes results to files.
"""

import subprocess
import sys
import os
import time
from datetime import datetime

def run_benchmark_to_file(benchmark_script, output_file, args=""):
    """Run a benchmark script and capture output to file."""
    print(f"Running {benchmark_script}...")
    
    try:
        # Set environment for proper imports
        env = os.environ.copy()
        project_root = os.path.dirname(os.getcwd())  # Parent of benchmarks directory
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = f"{project_root}{os.pathsep}{env['PYTHONPATH']}"
        else:
            env['PYTHONPATH'] = project_root
        env['PYTHONIOENCODING'] = 'utf-8'
        
        # Run benchmark
        command = [sys.executable, benchmark_script] + args.split() if args else [sys.executable, benchmark_script]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            env=env
        )
        
        # Write results
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"=== Benchmark Results for {benchmark_script} ===\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write(f"Return Code: {result.returncode}\n\n")
            f.write("=== STDOUT ===\n")
            f.write(result.stdout)
            f.write("\n\n=== STDERR ===\n")
            f.write(result.stderr)
            
        print(f"Results written to {output_file}")
        print(f"Return code: {result.returncode}")
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"=== Benchmark Results for {benchmark_script} ===\n")
            f.write(f"Timestamp: {datetime.now()}\n\n")
            f.write("ERROR: Benchmark timed out after 5 minutes\n")
        print(f"Benchmark {benchmark_script} timed out")
        return False
        
    except Exception as e:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"=== Benchmark Results for {benchmark_script} ===\n")
            f.write(f"Timestamp: {datetime.now()}\n\n")
            f.write(f"ERROR: {str(e)}\n")
        print(f"Error running {benchmark_script}: {e}")
        return False

def main():
    """Run benchmarks."""
    print("=== Neuromorphic Framework Benchmark Runner ===")
    
    # Change to benchmarks directory
    benchmarks_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(benchmarks_dir)
    
    benchmarks = [
        ("quick_benchmark.py", "benchmark_results_quick.txt", "--sizes=100,500,1000 --max-seconds=60"),
        ("cpu_performance_benchmark.py", "benchmark_results_cpu.txt", ""),
        ("performance_benchmarks.py", "benchmark_results_performance.txt", ""),
    ]
    
    results = {}
    
    for script, output_file, args in benchmarks:
        if os.path.exists(script):
            success = run_benchmark_to_file(script, output_file, args)
            results[script] = success
        else:
            print(f"Benchmark {script} not found")
            results[script] = False
    
    # Summary
    print("\n=== Benchmark Summary ===")
    total = len(results)
    passed = sum(results.values())
    
    for script, success in results.items():
        status = "PASSED" if success else "FAILED"
        print(f"{script}: {status}")
    
    print(f"\nOverall: {passed}/{total} benchmarks passed")
    
    # Write summary
    with open("benchmark_summary.txt", 'w', encoding='utf-8') as f:
        f.write("=== Neuromorphic Framework Benchmark Summary ===\n")
        f.write(f"Timestamp: {datetime.now()}\n\n")
        f.write(f"Total benchmarks: {total}\n")
        f.write(f"Passed: {passed}\n")
        f.write(f"Failed: {total - passed}\n\n")
        
        for script, success in results.items():
            status = "PASSED" if success else "FAILED"
            f.write(f"{script}: {status}\n")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)