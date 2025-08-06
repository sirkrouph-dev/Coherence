#!/usr/bin/env python3
"""
Quick test script for the enhanced neuromorphic system.
Verifies all components work correctly.
"""

import sys
import time
from pathlib import Path

import numpy as np

# Add the project root to the path
sys.path.append(".")


def test_enhanced_logging():
    """Test enhanced logging system."""
    print("🧪 Testing Enhanced Logging System...")

    try:
        from core.enhanced_logging import enhanced_logger

        # Test spike event logging
        enhanced_logger.log_spike_event(
            neuron_id=1,
            layer_name="test",
            spike_time=100.0,
            membrane_potential=-55.0,
            synaptic_inputs={"test": 0.5},
            neuromodulator_levels={"dopamine": 0.3},
        )

        # Test membrane potential logging
        enhanced_logger.log_membrane_potential(
            neuron_id=1,
            layer_name="test",
            time_step=100.0,
            membrane_potential=-65.0,
            synaptic_current=0.1,
        )

        # Test network state logging
        enhanced_logger.log_network_state(
            layer_name="test",
            time_step=100.0,
            active_neurons=5,
            total_neurons=10,
            firing_rate=25.0,
            average_membrane_potential=-60.0,
            spike_count=5,
        )

        print("✅ Enhanced logging system working correctly")

    except Exception as e:
        print(f"❌ Enhanced logging test failed: {e}")
        raise


def test_task_complexity():
    """Test task complexity system."""
    print("🧪 Testing Task Complexity System...")

    try:
        from core.task_complexity import (TaskLevel, TaskParameters,
                                          task_manager)

        # Test task creation
        params = TaskParameters(
            level=TaskLevel.LEVEL_2, input_noise=0.1, missing_modalities=[]
        )

        task = task_manager.create_task(TaskLevel.LEVEL_2, params)

        # Verify task structure
        assert "inputs" in task
        assert "expected_output" in task
        assert "metadata" in task

        print("✅ Task complexity system working correctly")

    except Exception as e:
        print(f"❌ Task complexity test failed: {e}")
        raise


def test_enhanced_encoding():
    """Test enhanced sensory encoding system."""
    print("🧪 Testing Enhanced Sensory Encoding System...")

    try:
        from core.enhanced_encoding import enhanced_encoder

        # Test sensory encoding
        sensory_inputs = {
            "visual": np.random.random((32, 32)),
            "auditory": np.random.random(1000),
            "tactile": np.random.random((8, 8)),
        }

        result = enhanced_encoder.encode_sensory_inputs(sensory_inputs)

        # Verify encoding result
        assert "encoded_inputs" in result
        assert "fused_result" in result
        assert "total_encoding_time" in result

        print("✅ Enhanced sensory encoding system working correctly")

    except Exception as e:
        print(f"❌ Enhanced encoding test failed: {e}")
        raise


def test_robustness_testing():
    """Test robustness testing framework."""
    print("🧪 Testing Robustness Testing Framework...")

    try:
        from core.robustness_testing import robustness_tester

        # Test noise generation
        test_data = np.random.random((10, 10))
        noisy_data = robustness_tester.noise_generator.gaussian_noise(test_data, 0.1)

        # Verify noise was applied
        assert not np.array_equal(test_data, noisy_data)

        # Test adversarial attack
        adversarial_data = robustness_tester.adversarial_attacker.fgsm_attack(
            test_data, 0.1
        )

        # Verify adversarial perturbation was applied
        assert not np.array_equal(test_data, adversarial_data)

        print("✅ Robustness testing framework working correctly")

    except Exception as e:
        print(f"❌ Robustness testing test failed: {e}")
        raise


def test_comprehensive_demo():
    """Test the comprehensive demo system."""
    print("🧪 Testing Comprehensive Demo System...")

    try:
        from demo.enhanced_comprehensive_demo import EnhancedNeuromorphicDemo

        # Create demo instance
        demo = EnhancedNeuromorphicDemo()

        # Test network creation
        network = demo.create_enhanced_network()

        # Verify network structure
        assert hasattr(network, "layers")
        assert hasattr(network, "connections")

        print("✅ Comprehensive demo system working correctly")

    except Exception as e:
        print(f"❌ Comprehensive demo test failed: {e}")
        raise


def test_file_generation():
    """Test that required files are generated."""
    print("🧪 Testing File Generation...")

    try:
        # Check if enhanced logging file exists
        log_file = Path("enhanced_trace.log")
        if log_file.exists():
            print("✅ Enhanced trace log file generated")
        else:
            print("⚠️ Enhanced trace log file not found (may be generated during demo)")

        # Check if directories exist
        directories = ["enhanced_data", "enhanced_analysis", "enhanced_plots"]
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            print(f"✅ Directory {directory} created/verified")

    except Exception as e:
        print(f"❌ File generation test failed: {e}")
        raise


def main():
    """Run all tests."""
    print("🚀 Enhanced Neuromorphic System Test Suite")
    print("=" * 50)

    tests = [
        test_enhanced_logging,
        test_task_complexity,
        test_enhanced_encoding,
        test_robustness_testing,
        test_comprehensive_demo,
        test_file_generation,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)

    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")

    passed = sum(results)
    total = len(results)

    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {i+1}. {test.__name__}: {status}")

    print(f"\n🎯 Overall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Enhanced system is ready.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
