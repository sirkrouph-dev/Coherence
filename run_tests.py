#!/usr/bin/env python3
"""
Test Runner for Neuromorphic Framework
=====================================

Runs all tests and writes results to files for verification.
"""

import subprocess
import sys
import os
from datetime import datetime

def run_test_file(test_file, output_file):
    """Run a test file and capture output."""
    print(f"Running {test_file}...")
    
    try:
        # Run the test file directly with project root in PYTHONPATH
        env = os.environ.copy()
        project_root = os.getcwd()
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = f"{project_root}{os.pathsep}{env['PYTHONPATH']}"
        else:
            env['PYTHONPATH'] = project_root
        
        # Set UTF-8 encoding for Windows console
        env['PYTHONIOENCODING'] = 'utf-8'
            
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
            env=env
        )
        
        with open(output_file, 'w') as f:
            f.write(f"=== Test Results for {test_file} ===\n")
            f.write(f"Timestamp: {datetime.now()}\n\n")
            f.write(f"Return Code: {result.returncode}\n\n")
            f.write("=== STDOUT ===\n")
            f.write(result.stdout)
            f.write("\n\n=== STDERR ===\n")
            f.write(result.stderr)
            
        print(f"Results written to {output_file}")
        print(f"Return code: {result.returncode}")
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        with open(output_file, 'w') as f:
            f.write(f"=== Test Results for {test_file} ===\n")
            f.write(f"Timestamp: {datetime.now()}\n\n")
            f.write("ERROR: Test timed out after 120 seconds\n")
        print(f"Test {test_file} timed out")
        return False
        
    except Exception as e:
        with open(output_file, 'w') as f:
            f.write(f"=== Test Results for {test_file} ===\n")
            f.write(f"Timestamp: {datetime.now()}\n\n")
            f.write(f"ERROR: {str(e)}\n")
        print(f"Error running {test_file}: {e}")
        return False

def main():
    """Run all tests."""
    print("=== Neuromorphic Framework Test Runner ===")
    
    test_files = [
        ("tests/test_working_memory.py", "test_results_working_memory.txt"),
        ("tests/test_developmental_plasticity.py", "test_results_developmental_plasticity.txt"),
        ("tests/test_attention_mechanism.py", "test_results_attention_mechanism.txt")
    ]
    
    results = {}
    
    for test_file, output_file in test_files:
        if os.path.exists(test_file):
            success = run_test_file(test_file, output_file)
            results[test_file] = success
        else:
            print(f"Test file {test_file} not found")
            results[test_file] = False
    
    # Summary
    print("\n=== Test Summary ===")
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_file, success in results.items():
        status = "PASSED" if success else "FAILED"
        print(f"{test_file}: {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} test files passed")
    
    # Write summary
    with open("test_summary.txt", 'w') as f:
        f.write("=== Neuromorphic Framework Test Summary ===\n")
        f.write(f"Timestamp: {datetime.now()}\n\n")
        f.write(f"Total test files: {total_tests}\n")
        f.write(f"Passed: {passed_tests}\n")
        f.write(f"Failed: {total_tests - passed_tests}\n\n")
        
        for test_file, success in results.items():
            status = "PASSED" if success else "FAILED"
            f.write(f"{test_file}: {status}\n")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)