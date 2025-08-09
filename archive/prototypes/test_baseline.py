#!/usr/bin/env python
"""
Comprehensive baseline test suite for the Neuromorphic System
Tests imports, basic functionality, and documents failures
"""

import sys
import traceback
import time
from datetime import datetime
import json

# Test results collector
test_results = {
    "timestamp": datetime.now().isoformat(),
    "passed": [],
    "failed": [],
    "import_errors": [],
    "runtime_errors": [],
    "coverage": {}
}

def test_import(module_path, class_name=None):
    """Test if a module/class can be imported"""
    try:
        if class_name:
            exec(f"from {module_path} import {class_name}")
            test_results["passed"].append(f"Import: {module_path}.{class_name}")
            print(f"✅ Import SUCCESS: {module_path}.{class_name}")
            return True
        else:
            exec(f"import {module_path}")
            test_results["passed"].append(f"Import: {module_path}")
            print(f"✅ Import SUCCESS: {module_path}")
            return True
    except Exception as e:
        error_info = {
            "module": module_path,
            "class": class_name,
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        test_results["import_errors"].append(error_info)
        print(f"❌ Import FAILED: {module_path}.{class_name if class_name else ''}")
        print(f"   Error: {str(e)}")
        return False

def run_test(test_name, test_func):
    """Run a test function and capture results"""
    print(f"\n🧪 Testing: {test_name}")
    try:
        start_time = time.time()
        result = test_func()
        elapsed = time.time() - start_time
        test_results["passed"].append(test_name)
        print(f"✅ PASSED: {test_name} (took {elapsed:.3f}s)")
        return True
    except Exception as e:
        error_info = {
            "test": test_name,
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        test_results["runtime_errors"].append(error_info)
        print(f"❌ FAILED: {test_name}")
        print(f"   Error: {str(e)}")
        print(f"   Stack trace:\n{traceback.format_exc()}")
        return False

def main():
    print("="*60)
    print("🚀 NEUROMORPHIC SYSTEM BASELINE TEST SUITE")
    print("="*60)
    
    # Test 1: Core module imports
    print("\n📦 TESTING CORE MODULE IMPORTS")
    print("-"*40)
    
    core_imports = [
        ("core.neurons", "LIFNeuron"),
        ("core.neurons", "IzhikevichNeuron"),
        ("core.synapses", "Synapse"),
        ("core.synapses", "STDPSynapse"),
        ("core.network", "NeuronGroup"),
        ("core.network", "Network"),
        ("core.encoding", "RateEncoder"),
        ("core.encoding", "PoissonEncoder"),
        ("core.encoding", "TemporalEncoder"),
        ("core.neuromodulation", "Neuromodulator"),
        ("core.neuromodulation", "DopamineModulator"),
        ("core.logging_utils", "SimulationLogger"),
        ("core.enhanced_logging", "EnhancedLogger"),
        ("core.enhanced_encoding", "MultiModalEncoder"),
        ("core.task_complexity", "TaskComplexity"),
        ("core.robustness_testing", "RobustnessFramework"),
    ]
    
    for module, cls in core_imports:
        test_import(module, cls)
    
    # Test 2: API imports
    print("\n📦 TESTING API MODULE IMPORTS")
    print("-"*40)
    
    api_imports = [
        ("api.neuromorphic_api", "NeuromorphicAPI"),
        ("api.advanced_api", "AdvancedNeuromorphicAPI"),
    ]
    
    for module, cls in api_imports:
        test_import(module, cls)
    
    # Test 3: Visualization imports
    print("\n📦 TESTING VISUALIZATION MODULE IMPORTS")
    print("-"*40)
    
    viz_imports = [
        ("visualization.plotters", "NetworkVisualizer"),
        ("visualization.real_time", "RealTimeMonitor"),
    ]
    
    for module, cls in viz_imports:
        test_import(module, cls)
    
    # Test 4: RateEncoder functionality
    print("\n🔧 TESTING RATEENCODER FUNCTIONALITY")
    print("-"*40)
    
    def test_rate_encoder():
        from core.encoding import RateEncoder
        import numpy as np
        
        encoder = RateEncoder(max_rate=100.0)
        
        # Test with different input values
        test_values = [0.0, 0.5, 1.0]
        for val in test_values:
            spikes = encoder.encode(val, duration=100, dt=1.0)
            print(f"   Input={val}: Generated {len(spikes)} spikes")
            assert isinstance(spikes, (list, np.ndarray)), f"Expected list/array, got {type(spikes)}"
        
        return True
    
    run_test("RateEncoder Basic Functionality", test_rate_encoder)
    
    # Test 5: Network creation
    print("\n🔧 TESTING NETWORK CREATION")
    print("-"*40)
    
    def test_network_creation():
        from core.network import Network
        from core.neurons import LIFNeuron
        
        network = Network()
        
        # Add neurons
        for i in range(5):
            neuron = LIFNeuron(neuron_id=f"test_{i}")
            network.add_neuron(neuron)
        
        print(f"   Created network with {len(network.neurons)} neurons")
        assert len(network.neurons) == 5, f"Expected 5 neurons, got {len(network.neurons)}"
        
        return True
    
    run_test("Network Creation", test_network_creation)
    
    # Test 6: Synapse connectivity
    print("\n🔧 TESTING SYNAPSE CONNECTIVITY")
    print("-"*40)
    
    def test_synapse():
        from core.neurons import LIFNeuron
        from core.synapses import Synapse
        
        pre = LIFNeuron(neuron_id="pre")
        post = LIFNeuron(neuron_id="post")
        
        synapse = Synapse(
            pre_neuron=pre,
            post_neuron=post,
            weight=0.5,
            delay=1.0
        )
        
        # Test spike transmission
        pre.spike_times = [10.0]
        synapse.transmit(current_time=11.0, dt=1.0)
        
        print(f"   Created synapse with weight={synapse.weight}, delay={synapse.delay}")
        return True
    
    run_test("Synapse Connectivity", test_synapse)
    
    # Test 7: Enhanced system features
    print("\n🔧 TESTING ENHANCED FEATURES")
    print("-"*40)
    
    def test_enhanced_features():
        from core.enhanced_logging import EnhancedLogger
        from core.task_complexity import TaskComplexity
        
        # Test logger
        logger = EnhancedLogger("test_logger")
        logger.log_info("SYSTEM", "Test message")
        
        # Test task complexity
        task = TaskComplexity(level="basic")
        config = task.get_configuration()
        
        print(f"   Task complexity level: {task.level}")
        print(f"   Configuration keys: {list(config.keys())}")
        
        return True
    
    run_test("Enhanced Features", test_enhanced_features)
    
    # Generate summary report
    print("\n" + "="*60)
    print("📊 TEST RESULTS SUMMARY")
    print("="*60)
    
    total_tests = len(test_results["passed"]) + len(test_results["import_errors"]) + len(test_results["runtime_errors"])
    
    print(f"\n✅ Passed: {len(test_results['passed'])}/{total_tests}")
    print(f"❌ Failed: {len(test_results['import_errors']) + len(test_results['runtime_errors'])}/{total_tests}")
    
    if test_results["import_errors"]:
        print(f"\n⚠️ Import Errors: {len(test_results['import_errors'])}")
        for error in test_results["import_errors"]:
            print(f"   - {error['module']}.{error['class'] or ''}: {error['error'].split('(')[0]}")
    
    if test_results["runtime_errors"]:
        print(f"\n⚠️ Runtime Errors: {len(test_results['runtime_errors'])}")
        for error in test_results["runtime_errors"]:
            print(f"   - {error['test']}: {error['error'].split('(')[0]}")
    
    # Calculate coverage estimate
    test_results["coverage"]["modules_tested"] = len(set([e.get("module", "") for e in test_results["import_errors"]] + 
                                                          [p.split(":")[1].strip().split(".")[0] if ":" in p else "" for p in test_results["passed"]]))
    test_results["coverage"]["tests_run"] = total_tests
    test_results["coverage"]["pass_rate"] = len(test_results["passed"]) / total_tests * 100 if total_tests > 0 else 0
    
    print(f"\n📈 Coverage Statistics:")
    print(f"   - Modules tested: {test_results['coverage']['modules_tested']}")
    print(f"   - Tests run: {test_results['coverage']['tests_run']}")
    print(f"   - Pass rate: {test_results['coverage']['pass_rate']:.1f}%")
    
    # Save detailed results to file
    with open("test_baseline_results.json", "w") as f:
        json.dump(test_results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: test_baseline_results.json")
    
    # Specific focus on RateEncoder issues
    print("\n" + "="*60)
    print("🔍 RATEENCODER SPECIFIC ANALYSIS")
    print("="*60)
    
    rate_encoder_issues = [e for e in test_results["import_errors"] if "RateEncoder" in str(e)]
    rate_encoder_issues.extend([e for e in test_results["runtime_errors"] if "RateEncoder" in str(e)])
    
    if rate_encoder_issues:
        print("⚠️ RateEncoder Issues Found:")
        for issue in rate_encoder_issues:
            print(f"\n{issue}")
    else:
        print("✅ No RateEncoder-specific issues detected")
    
    return len(test_results["import_errors"]) == 0 and len(test_results["runtime_errors"]) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
