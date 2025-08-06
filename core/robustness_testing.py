"""
Comprehensive robustness testing framework for neuromorphic computing.
Implements failure mode testing, adversarial attacks, and system stress testing.
"""

import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from core.enhanced_logging import enhanced_logger
from core.task_complexity import TaskLevel, TaskParameters


class TestType(Enum):
    """Types of robustness tests."""

    NOISE_TEST = "noise_test"
    MISSING_MODALITY_TEST = "missing_modality_test"
    ADVERSARIAL_TEST = "adversarial_test"
    TEMPORAL_PERTURBATION_TEST = "temporal_perturbation_test"
    NETWORK_DAMAGE_TEST = "network_damage_test"
    SENSORY_DEGRADATION_TEST = "sensory_degradation_test"
    STRESS_TEST = "stress_test"


@dataclass
class TestResult:
    """Data structure for test results."""

    test_type: TestType
    test_parameters: Dict[str, Any]
    performance_before: Dict[str, float]
    performance_after: Dict[str, float]
    degradation_metrics: Dict[str, float]
    recovery_time: float
    robustness_score: float
    metadata: Dict[str, Any]


class NoiseGenerator:
    """Advanced noise generation for robustness testing."""

    @staticmethod
    def gaussian_noise(data: np.ndarray, std: float) -> np.ndarray:
        """Add Gaussian noise."""
        noise = np.random.normal(0, std, data.shape)
        return np.clip(data + noise, 0, 1)

    @staticmethod
    def salt_pepper_noise(data: np.ndarray, prob: float) -> np.ndarray:
        """Add salt and pepper noise."""
        noisy_data = data.copy()
        mask = np.random.random(data.shape) < prob
        noisy_data[mask] = np.random.choice([0, 1], size=np.sum(mask))
        return noisy_data

    @staticmethod
    def impulse_noise(data: np.ndarray, prob: float) -> np.ndarray:
        """Add impulse noise."""
        noisy_data = data.copy()
        mask = np.random.random(data.shape) < prob
        noisy_data[mask] = np.random.uniform(0, 1, size=np.sum(mask))
        return noisy_data

    @staticmethod
    def temporal_noise(data: np.ndarray, jitter_std: float) -> np.ndarray:
        """Add temporal jitter."""
        if len(data.shape) == 1:
            # For 1D data, add temporal shifts
            shifts = np.random.normal(0, jitter_std, len(data))
            noisy_data = np.zeros_like(data)
            for i, shift in enumerate(shifts):
                new_idx = int(i + shift)
                if 0 <= new_idx < len(data):
                    noisy_data[i] = data[new_idx]
                else:
                    noisy_data[i] = 0
            return noisy_data
        else:
            # For multi-dimensional data, add random perturbations
            return data + np.random.normal(0, jitter_std, data.shape)


class AdversarialAttacker:
    """Advanced adversarial attack methods."""

    @staticmethod
    def fgsm_attack(
        data: np.ndarray, epsilon: float, target: np.ndarray = None
    ) -> np.ndarray:
        """Fast Gradient Sign Method attack."""
        if target is None:
            # Untargeted attack
            gradient = np.random.normal(0, 1, data.shape)
        else:
            # Targeted attack
            gradient = target - data

        adversarial_data = data + epsilon * np.sign(gradient)
        return np.clip(adversarial_data, 0, 1)

    @staticmethod
    def pgd_attack(
        data: np.ndarray,
        epsilon: float,
        alpha: float,
        iterations: int,
        target: np.ndarray = None,
    ) -> np.ndarray:
        """Projected Gradient Descent attack."""
        adversarial_data = data.copy()

        for _ in range(iterations):
            if target is None:
                gradient = np.random.normal(0, 1, data.shape)
            else:
                gradient = target - adversarial_data

            adversarial_data = adversarial_data + alpha * np.sign(gradient)
            adversarial_data = np.clip(adversarial_data, data - epsilon, data + epsilon)
            adversarial_data = np.clip(adversarial_data, 0, 1)

        return adversarial_data

    @staticmethod
    def universal_perturbation(data: np.ndarray, epsilon: float) -> np.ndarray:
        """Universal adversarial perturbation."""
        perturbation = np.random.uniform(-epsilon, epsilon, data.shape)
        adversarial_data = data + perturbation
        return np.clip(adversarial_data, 0, 1)

    @staticmethod
    def targeted_perturbation(
        data: np.ndarray, target_class: int, epsilon: float
    ) -> np.ndarray:
        """Targeted perturbation to specific class."""
        # Create target pattern
        target_pattern = np.zeros_like(data)
        if target_class == 1:
            target_pattern = np.ones_like(data)
        else:
            target_pattern = np.zeros_like(data)

        # Apply targeted perturbation
        perturbation = (target_pattern - data) * epsilon
        adversarial_data = data + perturbation
        return np.clip(adversarial_data, 0, 1)


class NetworkDamageSimulator:
    """Simulates various types of network damage."""

    @staticmethod
    def random_neuron_damage(
        neuron_weights: np.ndarray, damage_ratio: float
    ) -> np.ndarray:
        """Randomly damage neurons by setting weights to zero."""
        damaged_weights = neuron_weights.copy()
        num_neurons = neuron_weights.shape[0]
        num_damaged = int(num_neurons * damage_ratio)

        damaged_indices = np.random.choice(num_neurons, num_damaged, replace=False)
        damaged_weights[damaged_indices] = 0

        return damaged_weights

    @staticmethod
    def synaptic_damage(
        synaptic_weights: np.ndarray, damage_ratio: float
    ) -> np.ndarray:
        """Randomly damage synaptic connections."""
        damaged_weights = synaptic_weights.copy()
        num_synapses = synaptic_weights.size
        num_damaged = int(num_synapses * damage_ratio)

        damaged_indices = np.random.choice(num_synapses, num_damaged, replace=False)
        damaged_weights.flat[damaged_indices] = 0

        return damaged_weights

    @staticmethod
    def layer_damage(
        layer_weights: np.ndarray, layer_index: int, damage_ratio: float
    ) -> np.ndarray:
        """Damage specific layer connections."""
        damaged_weights = layer_weights.copy()

        if layer_index < len(layer_weights):
            layer = layer_weights[layer_index]
            num_connections = layer.size
            num_damaged = int(num_connections * damage_ratio)

            damaged_indices = np.random.choice(
                num_connections, num_damaged, replace=False
            )
            layer.flat[damaged_indices] = 0
            damaged_weights[layer_index] = layer

        return damaged_weights


class RobustnessTester:
    """Main robustness testing framework."""

    def __init__(self):
        """Initialize robustness tester."""
        self.noise_generator = NoiseGenerator()
        self.adversarial_attacker = AdversarialAttacker()
        self.network_damage_simulator = NetworkDamageSimulator()
        self.test_history = []

    def run_comprehensive_test_suite(
        self,
        network,
        test_inputs: Dict[str, np.ndarray],
        baseline_performance: Dict[str, float],
    ) -> List[TestResult]:
        """Run comprehensive robustness test suite."""
        test_results = []

        # 1. Noise robustness tests
        noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
        for noise_level in noise_levels:
            result = self.test_noise_robustness(
                network, test_inputs, baseline_performance, noise_level
            )
            test_results.append(result)

        # 2. Missing modality tests
        modalities = ["visual", "auditory", "tactile"]
        for modality in modalities:
            result = self.test_missing_modality(
                network, test_inputs, baseline_performance, modality
            )
            test_results.append(result)

        # 3. Adversarial robustness tests
        adversarial_strengths = [0.1, 0.2, 0.3]
        for strength in adversarial_strengths:
            result = self.test_adversarial_robustness(
                network, test_inputs, baseline_performance, strength
            )
            test_results.append(result)

        # 4. Temporal perturbation tests
        temporal_noise_levels = [0.05, 0.1, 0.15]
        for noise_level in temporal_noise_levels:
            result = self.test_temporal_perturbation(
                network, test_inputs, baseline_performance, noise_level
            )
            test_results.append(result)

        # 5. Network damage tests
        damage_ratios = [0.1, 0.2, 0.3]
        for damage_ratio in damage_ratios:
            result = self.test_network_damage(
                network, test_inputs, baseline_performance, damage_ratio
            )
            test_results.append(result)

        # 6. Sensory degradation tests
        degradation_levels = [0.2, 0.4, 0.6]
        for degradation_level in degradation_levels:
            result = self.test_sensory_degradation(
                network, test_inputs, baseline_performance, degradation_level
            )
            test_results.append(result)

        # 7. Stress tests
        stress_levels = [1.5, 2.0, 2.5]
        for stress_level in stress_levels:
            result = self.test_system_stress(
                network, test_inputs, baseline_performance, stress_level
            )
            test_results.append(result)

        self.test_history.extend(test_results)
        return test_results

    def test_noise_robustness(
        self,
        network,
        test_inputs: Dict[str, np.ndarray],
        baseline_performance: Dict[str, float],
        noise_level: float,
    ) -> TestResult:
        """Test robustness against various types of noise."""
        enhanced_logger.log_robustness_test(
            "noise_robustness",
            {"noise_level": noise_level},
            baseline_performance,
            {},  # Will be filled after test
        )

        # Apply different types of noise
        noisy_inputs = {}
        for modality, data in test_inputs.items():
            # Apply multiple noise types
            gaussian_noisy = self.noise_generator.gaussian_noise(data, noise_level)
            salt_pepper_noisy = self.noise_generator.salt_pepper_noise(
                data, noise_level
            )
            impulse_noisy = self.noise_generator.impulse_noise(data, noise_level)

            # Use the most challenging noise type
            noisy_inputs[modality] = np.maximum.reduce(
                [gaussian_noisy, salt_pepper_noisy, impulse_noisy]
            )

        # Test performance with noisy inputs
        start_time = time.time()
        performance_after = self._evaluate_performance(network, noisy_inputs)
        recovery_time = time.time() - start_time

        # Calculate degradation metrics
        degradation_metrics = self._calculate_degradation(
            baseline_performance, performance_after
        )
        robustness_score = self._calculate_robustness_score(degradation_metrics)

        return TestResult(
            test_type=TestType.NOISE_TEST,
            test_parameters={"noise_level": noise_level},
            performance_before=baseline_performance,
            performance_after=performance_after,
            degradation_metrics=degradation_metrics,
            recovery_time=recovery_time,
            robustness_score=robustness_score,
            metadata={"noise_types": ["gaussian", "salt_pepper", "impulse"]},
        )

    def test_missing_modality(
        self,
        network,
        test_inputs: Dict[str, np.ndarray],
        baseline_performance: Dict[str, float],
        missing_modality: str,
    ) -> TestResult:
        """Test robustness when a sensory modality is missing."""
        enhanced_logger.log_robustness_test(
            "missing_modality",
            {"missing_modality": missing_modality},
            baseline_performance,
            {},  # Will be filled after test
        )

        # Remove the specified modality
        degraded_inputs = test_inputs.copy()
        if missing_modality in degraded_inputs:
            degraded_inputs[missing_modality] = np.zeros_like(
                degraded_inputs[missing_modality]
            )

        # Test performance with missing modality
        start_time = time.time()
        performance_after = self._evaluate_performance(network, degraded_inputs)
        recovery_time = time.time() - start_time

        # Calculate degradation metrics
        degradation_metrics = self._calculate_degradation(
            baseline_performance, performance_after
        )
        robustness_score = self._calculate_robustness_score(degradation_metrics)

        return TestResult(
            test_type=TestType.MISSING_MODALITY_TEST,
            test_parameters={"missing_modality": missing_modality},
            performance_before=baseline_performance,
            performance_after=performance_after,
            degradation_metrics=degradation_metrics,
            recovery_time=recovery_time,
            robustness_score=robustness_score,
            metadata={"available_modalities": list(degraded_inputs.keys())},
        )

    def test_adversarial_robustness(
        self,
        network,
        test_inputs: Dict[str, np.ndarray],
        baseline_performance: Dict[str, float],
        adversarial_strength: float,
    ) -> TestResult:
        """Test robustness against adversarial attacks."""
        enhanced_logger.log_robustness_test(
            "adversarial_robustness",
            {"adversarial_strength": adversarial_strength},
            baseline_performance,
            {},  # Will be filled after test
        )

        # Apply adversarial perturbations
        adversarial_inputs = {}
        for modality, data in test_inputs.items():
            # Apply different adversarial attacks
            fgsm_perturbed = self.adversarial_attacker.fgsm_attack(
                data, adversarial_strength
            )
            pgd_perturbed = self.adversarial_attacker.pgd_attack(
                data, adversarial_strength, adversarial_strength / 10, 10
            )
            universal_perturbed = self.adversarial_attacker.universal_perturbation(
                data, adversarial_strength
            )

            # Use the most challenging perturbation
            adversarial_inputs[modality] = np.maximum.reduce(
                [fgsm_perturbed, pgd_perturbed, universal_perturbed]
            )

        # Test performance with adversarial inputs
        start_time = time.time()
        performance_after = self._evaluate_performance(network, adversarial_inputs)
        recovery_time = time.time() - start_time

        # Calculate degradation metrics
        degradation_metrics = self._calculate_degradation(
            baseline_performance, performance_after
        )
        robustness_score = self._calculate_robustness_score(degradation_metrics)

        return TestResult(
            test_type=TestType.ADVERSARIAL_TEST,
            test_parameters={"adversarial_strength": adversarial_strength},
            performance_before=baseline_performance,
            performance_after=performance_after,
            degradation_metrics=degradation_metrics,
            recovery_time=recovery_time,
            robustness_score=robustness_score,
            metadata={"attack_types": ["fgsm", "pgd", "universal"]},
        )

    def test_temporal_perturbation(
        self,
        network,
        test_inputs: Dict[str, np.ndarray],
        baseline_performance: Dict[str, float],
        temporal_noise_level: float,
    ) -> TestResult:
        """Test robustness against temporal perturbations."""
        enhanced_logger.log_robustness_test(
            "temporal_perturbation",
            {"temporal_noise_level": temporal_noise_level},
            baseline_performance,
            {},  # Will be filled after test
        )

        # Apply temporal perturbations
        perturbed_inputs = {}
        for modality, data in test_inputs.items():
            perturbed_inputs[modality] = self.noise_generator.temporal_noise(
                data, temporal_noise_level
            )

        # Test performance with temporal perturbations
        start_time = time.time()
        performance_after = self._evaluate_performance(network, perturbed_inputs)
        recovery_time = time.time() - start_time

        # Calculate degradation metrics
        degradation_metrics = self._calculate_degradation(
            baseline_performance, performance_after
        )
        robustness_score = self._calculate_robustness_score(degradation_metrics)

        return TestResult(
            test_type=TestType.TEMPORAL_PERTURBATION_TEST,
            test_parameters={"temporal_noise_level": temporal_noise_level},
            performance_before=baseline_performance,
            performance_after=performance_after,
            degradation_metrics=degradation_metrics,
            recovery_time=recovery_time,
            robustness_score=robustness_score,
            metadata={"perturbation_type": "temporal_jitter"},
        )

    def test_network_damage(
        self,
        network,
        test_inputs: Dict[str, np.ndarray],
        baseline_performance: Dict[str, float],
        damage_ratio: float,
    ) -> TestResult:
        """Test robustness against network damage."""
        enhanced_logger.log_robustness_test(
            "network_damage",
            {"damage_ratio": damage_ratio},
            baseline_performance,
            {},  # Will be filled after test
        )

        # Simulate network damage
        original_weights = network.get_weights()
        damaged_weights = self.network_damage_simulator.synaptic_damage(
            original_weights, damage_ratio
        )
        network.set_weights(damaged_weights)

        # Test performance with damaged network
        start_time = time.time()
        performance_after = self._evaluate_performance(network, test_inputs)
        recovery_time = time.time() - start_time

        # Restore original weights
        network.set_weights(original_weights)

        # Calculate degradation metrics
        degradation_metrics = self._calculate_degradation(
            baseline_performance, performance_after
        )
        robustness_score = self._calculate_robustness_score(degradation_metrics)

        return TestResult(
            test_type=TestType.NETWORK_DAMAGE_TEST,
            test_parameters={"damage_ratio": damage_ratio},
            performance_before=baseline_performance,
            performance_after=performance_after,
            degradation_metrics=degradation_metrics,
            recovery_time=recovery_time,
            robustness_score=robustness_score,
            metadata={"damage_type": "synaptic_damage"},
        )

    def test_sensory_degradation(
        self,
        network,
        test_inputs: Dict[str, np.ndarray],
        baseline_performance: Dict[str, float],
        degradation_level: float,
    ) -> TestResult:
        """Test robustness against sensory degradation."""
        enhanced_logger.log_robustness_test(
            "sensory_degradation",
            {"degradation_level": degradation_level},
            baseline_performance,
            {},  # Will be filled after test
        )

        # Apply sensory degradation
        degraded_inputs = {}
        for modality, data in test_inputs.items():
            # Reduce signal strength
            degraded_data = data * (1 - degradation_level)
            # Add noise
            degraded_data = self.noise_generator.gaussian_noise(
                degraded_data, degradation_level * 0.5
            )
            degraded_inputs[modality] = degraded_data

        # Test performance with degraded inputs
        start_time = time.time()
        performance_after = self._evaluate_performance(network, degraded_inputs)
        recovery_time = time.time() - start_time

        # Calculate degradation metrics
        degradation_metrics = self._calculate_degradation(
            baseline_performance, performance_after
        )
        robustness_score = self._calculate_robustness_score(degradation_metrics)

        return TestResult(
            test_type=TestType.SENSORY_DEGRADATION_TEST,
            test_parameters={"degradation_level": degradation_level},
            performance_before=baseline_performance,
            performance_after=performance_after,
            degradation_metrics=degradation_metrics,
            recovery_time=recovery_time,
            robustness_score=robustness_score,
            metadata={"degradation_type": "signal_strength_reduction"},
        )

    def test_system_stress(
        self,
        network,
        test_inputs: Dict[str, np.ndarray],
        baseline_performance: Dict[str, float],
        stress_level: float,
    ) -> TestResult:
        """Test system performance under stress conditions."""
        enhanced_logger.log_robustness_test(
            "system_stress",
            {"stress_level": stress_level},
            baseline_performance,
            {},  # Will be filled after test
        )

        # Apply multiple stress factors simultaneously
        stressed_inputs = {}
        for modality, data in test_inputs.items():
            # High noise
            noisy_data = self.noise_generator.gaussian_noise(data, stress_level * 0.3)
            # Adversarial perturbation
            perturbed_data = self.adversarial_attacker.fgsm_attack(
                noisy_data, stress_level * 0.2
            )
            # Temporal perturbation
            temporal_data = self.noise_generator.temporal_noise(
                perturbed_data, stress_level * 0.1
            )
            stressed_inputs[modality] = temporal_data

        # Test performance under stress
        start_time = time.time()
        performance_after = self._evaluate_performance(network, stressed_inputs)
        recovery_time = time.time() - start_time

        # Calculate degradation metrics
        degradation_metrics = self._calculate_degradation(
            baseline_performance, performance_after
        )
        robustness_score = self._calculate_robustness_score(degradation_metrics)

        return TestResult(
            test_type=TestType.STRESS_TEST,
            test_parameters={"stress_level": stress_level},
            performance_before=baseline_performance,
            performance_after=performance_after,
            degradation_metrics=degradation_metrics,
            recovery_time=recovery_time,
            robustness_score=robustness_score,
            metadata={"stress_factors": ["noise", "adversarial", "temporal"]},
        )

    def _evaluate_performance(
        self, network, inputs: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Evaluate network performance on given inputs."""
        # This is a placeholder - in real implementation, this would run the network
        # and measure actual performance metrics

        # Simulate performance evaluation
        accuracy = np.random.uniform(0.5, 1.0)
        latency = np.random.uniform(0.01, 0.1)
        throughput = np.random.uniform(10, 100)

        return {
            "accuracy": accuracy,
            "latency": latency,
            "throughput": throughput,
            "reward": accuracy,  # Simplified reward based on accuracy
        }

    def _calculate_degradation(
        self, performance_before: Dict[str, float], performance_after: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate performance degradation metrics."""
        degradation = {}

        for metric in performance_before.keys():
            if metric in performance_after:
                before_val = performance_before[metric]
                after_val = performance_after[metric]

                if before_val != 0:
                    degradation[metric] = (before_val - after_val) / before_val
                else:
                    degradation[metric] = 0.0

        return degradation

    def _calculate_robustness_score(
        self, degradation_metrics: Dict[str, float]
    ) -> float:
        """Calculate overall robustness score."""
        if not degradation_metrics:
            return 1.0

        # Average degradation across all metrics
        avg_degradation = np.mean(list(degradation_metrics.values()))

        # Convert to robustness score (1.0 = no degradation, 0.0 = complete failure)
        robustness_score = max(0.0, 1.0 - avg_degradation)

        return robustness_score

    def get_robustness_summary(self) -> Dict[str, Any]:
        """Get comprehensive robustness testing summary."""
        if not self.test_history:
            return {}

        summary = {
            "total_tests": len(self.test_history),
            "test_type_distribution": {},
            "average_robustness_score": 0.0,
            "worst_case_scenarios": [],
            "best_case_scenarios": [],
            "recommendations": [],
        }

        # Test type distribution
        for result in self.test_history:
            test_type = result.test_type.value
            if test_type not in summary["test_type_distribution"]:
                summary["test_type_distribution"][test_type] = 0
            summary["test_type_distribution"][test_type] += 1

        # Average robustness score
        robustness_scores = [result.robustness_score for result in self.test_history]
        summary["average_robustness_score"] = np.mean(robustness_scores)

        # Worst and best case scenarios
        sorted_results = sorted(self.test_history, key=lambda x: x.robustness_score)
        summary["worst_case_scenarios"] = [
            result.test_type.value for result in sorted_results[:3]
        ]
        summary["best_case_scenarios"] = [
            result.test_type.value for result in sorted_results[-3:]
        ]

        # Generate recommendations
        if summary["average_robustness_score"] < 0.5:
            summary["recommendations"].append(
                "System shows poor robustness - consider architectural improvements"
            )
        if summary["average_robustness_score"] < 0.7:
            summary["recommendations"].append(
                "Moderate robustness - implement additional error handling"
            )
        if summary["average_robustness_score"] >= 0.8:
            summary["recommendations"].append(
                "Good robustness - system is well-designed"
            )

        return summary


# Global robustness tester instance
robustness_tester = RobustnessTester()
