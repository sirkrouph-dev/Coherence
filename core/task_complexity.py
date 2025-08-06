"""
Enhanced task complexity system for neuromorphic computing.
Implements progressive difficulty levels with noise, missing modalities, and adversarial testing.
"""

import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from core.enhanced_logging import enhanced_logger


class TaskLevel(Enum):
    """Task complexity levels."""

    LEVEL_1 = "simple_binary"  # Basic binary classification
    LEVEL_2 = "pattern_recognition"  # Visual patterns with noise
    LEVEL_3 = "temporal_sequence"  # Time-dependent decisions
    LEVEL_4 = "multi_modal_fusion"  # Cross-modal integration
    LEVEL_5 = "adaptive_learning"  # Dynamic environment changes
    LEVEL_6 = "adversarial_robustness"  # Adversarial testing
    LEVEL_7 = "real_world_simulation"  # Real-world scenarios


@dataclass
class TaskParameters:
    """Parameters for task generation."""

    level: TaskLevel
    input_noise: float = 0.0
    missing_modalities: List[str] = None
    temporal_complexity: float = 0.0
    adversarial_strength: float = 0.0
    dynamic_changes: bool = False
    sequence_length: int = 1
    pattern_complexity: int = 1

    def __post_init__(self):
        if self.missing_modalities is None:
            self.missing_modalities = []


class PatternGenerator:
    """Generates complex patterns for testing."""

    def __init__(self):
        self.patterns = {
            "visual": self._generate_visual_patterns(),
            "auditory": self._generate_auditory_patterns(),
            "tactile": self._generate_tactile_patterns(),
        }

    def _generate_visual_patterns(self) -> Dict[str, np.ndarray]:
        """Generate visual patterns of varying complexity."""
        patterns = {}

        # Simple patterns
        patterns["horizontal_line"] = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0])
        patterns["vertical_line"] = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0])
        patterns["diagonal"] = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])

        # Medium complexity
        patterns["cross"] = np.array([0, 1, 0, 1, 1, 1, 0, 1, 0])
        patterns["square"] = np.array([1, 1, 1, 1, 0, 1, 1, 1, 1])
        patterns["triangle"] = np.array([0, 0, 1, 0, 1, 1, 1, 1, 1])

        # High complexity
        patterns["spiral"] = np.array([1, 1, 1, 0, 0, 1, 1, 1, 0])
        patterns["zigzag"] = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1])
        patterns["checkerboard"] = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1])

        return patterns

    def _generate_auditory_patterns(self) -> Dict[str, np.ndarray]:
        """Generate auditory patterns."""
        patterns = {}

        # Frequency patterns
        patterns["low_freq"] = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0])
        patterns["high_freq"] = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1])
        patterns["ascending"] = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])
        patterns["descending"] = np.array([0, 0, 1, 0, 1, 0, 1, 0, 0])

        return patterns

    def _generate_tactile_patterns(self) -> Dict[str, np.ndarray]:
        """Generate tactile patterns."""
        patterns = {}

        # Pressure patterns
        patterns["light_touch"] = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])
        patterns["firm_press"] = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0])
        patterns["vibration"] = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1])

        return patterns


class NoiseGenerator:
    """Generates various types of noise for robustness testing."""

    @staticmethod
    def add_gaussian_noise(data: np.ndarray, noise_level: float) -> np.ndarray:
        """Add Gaussian noise to data."""
        noise = np.random.normal(0, noise_level, data.shape)
        noisy_data = data + noise
        return np.clip(noisy_data, 0, 1)  # Keep in [0, 1] range

    @staticmethod
    def add_salt_pepper_noise(data: np.ndarray, noise_prob: float) -> np.ndarray:
        """Add salt and pepper noise."""
        noisy_data = data.copy()
        mask = np.random.random(data.shape) < noise_prob
        noisy_data[mask] = np.random.choice([0, 1], size=np.sum(mask))
        return noisy_data

    @staticmethod
    def add_temporal_noise(data: np.ndarray, noise_level: float) -> np.ndarray:
        """Add temporal noise (delays, jitter)."""
        # Simulate temporal noise by shifting patterns
        shifts = np.random.randint(
            -int(noise_level * len(data)), int(noise_level * len(data)), len(data)
        )
        noisy_data = np.zeros_like(data)
        for i, shift in enumerate(shifts):
            if 0 <= i + shift < len(data):
                noisy_data[i] = data[i + shift]
        return noisy_data


class AdversarialGenerator:
    """Generates adversarial examples for robustness testing."""

    @staticmethod
    def fgsm_attack(data: np.ndarray, epsilon: float) -> np.ndarray:
        """Fast Gradient Sign Method attack."""
        # Simplified FGSM for neuromorphic inputs
        gradient = np.random.normal(0, 1, data.shape)
        adversarial_data = data + epsilon * np.sign(gradient)
        return np.clip(adversarial_data, 0, 1)

    @staticmethod
    def targeted_attack(
        data: np.ndarray, target: np.ndarray, epsilon: float
    ) -> np.ndarray:
        """Targeted adversarial attack."""
        perturbation = target - data
        adversarial_data = data + epsilon * perturbation
        return np.clip(adversarial_data, 0, 1)

    @staticmethod
    def universal_perturbation(data: np.ndarray, epsilon: float) -> np.ndarray:
        """Universal adversarial perturbation."""
        perturbation = np.random.uniform(-epsilon, epsilon, data.shape)
        adversarial_data = data + perturbation
        return np.clip(adversarial_data, 0, 1)


class TaskComplexityManager:
    """Manages task complexity and generates appropriate challenges."""

    def __init__(self):
        self.pattern_generator = PatternGenerator()
        self.noise_generator = NoiseGenerator()
        self.adversarial_generator = AdversarialGenerator()
        self.current_level = TaskLevel.LEVEL_1
        self.task_history = []

    def create_task(
        self, level: TaskLevel, parameters: TaskParameters
    ) -> Dict[str, Any]:
        """Create a task with specified complexity level."""
        self.current_level = level
        enhanced_logger.log_task_complexity(
            level.value,
            f"Creating {level.value} task",
            parameters.input_noise,
            parameters.missing_modalities,
        )

        task = {
            "level": level,
            "parameters": parameters,
            "inputs": {},
            "expected_output": None,
            "metadata": {},
        }

        if level == TaskLevel.LEVEL_1:
            task = self._create_simple_binary_task(parameters)
        elif level == TaskLevel.LEVEL_2:
            task = self._create_pattern_recognition_task(parameters)
        elif level == TaskLevel.LEVEL_3:
            task = self._create_temporal_sequence_task(parameters)
        elif level == TaskLevel.LEVEL_4:
            task = self._create_multi_modal_fusion_task(parameters)
        elif level == TaskLevel.LEVEL_5:
            task = self._create_adaptive_learning_task(parameters)
        elif level == TaskLevel.LEVEL_6:
            task = self._create_adversarial_task(parameters)
        elif level == TaskLevel.LEVEL_7:
            task = self._create_real_world_task(parameters)

        self.task_history.append(task)
        return task

    def _create_simple_binary_task(self, params: TaskParameters) -> Dict[str, Any]:
        """Create simple binary classification task."""
        # Simple binary decision based on input strength
        visual_input = np.random.random(9)
        auditory_input = np.random.random(9)
        tactile_input = np.random.random(9)

        # Add noise if specified
        if params.input_noise > 0:
            visual_input = self.noise_generator.add_gaussian_noise(
                visual_input, params.input_noise
            )
            auditory_input = self.noise_generator.add_gaussian_noise(
                auditory_input, params.input_noise
            )
            tactile_input = self.noise_generator.add_gaussian_noise(
                tactile_input, params.input_noise
            )

        # Remove missing modalities
        if "visual" in params.missing_modalities:
            visual_input = np.zeros_like(visual_input)
        if "auditory" in params.missing_modalities:
            auditory_input = np.zeros_like(auditory_input)
        if "tactile" in params.missing_modalities:
            tactile_input = np.zeros_like(tactile_input)

        # Simple decision rule
        total_input = (
            np.mean(visual_input) + np.mean(auditory_input) + np.mean(tactile_input)
        )
        expected_output = 1 if total_input > 0.5 else 0

        return {
            "level": TaskLevel.LEVEL_1,
            "parameters": params,
            "inputs": {
                "visual": visual_input,
                "auditory": auditory_input,
                "tactile": tactile_input,
            },
            "expected_output": expected_output,
            "metadata": {
                "total_input_strength": total_input,
                "decision_threshold": 0.5,
            },
        }

    def _create_pattern_recognition_task(
        self, params: TaskParameters
    ) -> Dict[str, Any]:
        """Create pattern recognition task with noise."""
        # Select random patterns
        visual_pattern = random.choice(
            list(self.pattern_generator.patterns["visual"].keys())
        )
        auditory_pattern = random.choice(
            list(self.pattern_generator.patterns["auditory"].keys())
        )
        tactile_pattern = random.choice(
            list(self.pattern_generator.patterns["tactile"].keys())
        )

        visual_input = self.pattern_generator.patterns["visual"][visual_pattern].copy()
        auditory_input = self.pattern_generator.patterns["auditory"][
            auditory_pattern
        ].copy()
        tactile_input = self.pattern_generator.patterns["tactile"][
            tactile_pattern
        ].copy()

        # Add noise based on complexity
        if params.input_noise > 0:
            visual_input = self.noise_generator.add_gaussian_noise(
                visual_input, params.input_noise
            )
            auditory_input = self.noise_generator.add_gaussian_noise(
                auditory_input, params.input_noise
            )
            tactile_input = self.noise_generator.add_gaussian_noise(
                tactile_input, params.input_noise
            )

        # Remove missing modalities
        if "visual" in params.missing_modalities:
            visual_input = np.zeros_like(visual_input)
        if "auditory" in params.missing_modalities:
            auditory_input = np.zeros_like(auditory_input)
        if "tactile" in params.missing_modalities:
            tactile_input = np.zeros_like(tactile_input)

        # Pattern-based decision
        pattern_complexity = (
            np.sum(visual_input) + np.sum(auditory_input) + np.sum(tactile_input)
        ) / 27
        expected_output = 1 if pattern_complexity > 0.3 else 0

        return {
            "level": TaskLevel.LEVEL_2,
            "parameters": params,
            "inputs": {
                "visual": visual_input,
                "auditory": auditory_input,
                "tactile": tactile_input,
            },
            "expected_output": expected_output,
            "metadata": {
                "visual_pattern": visual_pattern,
                "auditory_pattern": auditory_pattern,
                "tactile_pattern": tactile_pattern,
                "pattern_complexity": pattern_complexity,
            },
        }

    def _create_temporal_sequence_task(self, params: TaskParameters) -> Dict[str, Any]:
        """Create temporal sequence task."""
        sequence_length = params.sequence_length
        inputs = []

        for t in range(sequence_length):
            # Generate temporal sequence
            visual_input = np.random.random(9)
            auditory_input = np.random.random(9)
            tactile_input = np.random.random(9)

            # Add temporal noise
            if params.temporal_complexity > 0:
                visual_input = self.noise_generator.add_temporal_noise(
                    visual_input, params.temporal_complexity
                )
                auditory_input = self.noise_generator.add_temporal_noise(
                    auditory_input, params.temporal_complexity
                )
                tactile_input = self.noise_generator.add_temporal_noise(
                    tactile_input, params.temporal_complexity
                )

            inputs.append(
                {
                    "visual": visual_input,
                    "auditory": auditory_input,
                    "tactile": tactile_input,
                }
            )

        # Decision based on temporal pattern
        temporal_pattern = np.array(
            [np.mean(inputs[i]["visual"]) for i in range(sequence_length)]
        )
        expected_output = 1 if np.std(temporal_pattern) > 0.1 else 0

        return {
            "level": TaskLevel.LEVEL_3,
            "parameters": params,
            "inputs": inputs,
            "expected_output": expected_output,
            "metadata": {
                "sequence_length": sequence_length,
                "temporal_pattern": temporal_pattern.tolist(),
                "temporal_variance": np.std(temporal_pattern),
            },
        }

    def _create_multi_modal_fusion_task(self, params: TaskParameters) -> Dict[str, Any]:
        """Create multi-modal fusion task."""
        # Generate conflicting information across modalities
        visual_input = np.random.random(9)
        auditory_input = np.random.random(9)
        tactile_input = np.random.random(9)

        # Create conflict: one modality suggests one output, others suggest different
        visual_strength = np.mean(visual_input)
        auditory_strength = np.mean(auditory_input)
        tactile_strength = np.mean(tactile_input)

        # Decision based on modality fusion
        modality_weights = [0.4, 0.3, 0.3]  # Visual, Auditory, Tactile weights
        fused_decision = (
            visual_strength * modality_weights[0]
            + auditory_strength * modality_weights[1]
            + tactile_strength * modality_weights[2]
        )

        expected_output = 1 if fused_decision > 0.5 else 0

        return {
            "level": TaskLevel.LEVEL_4,
            "parameters": params,
            "inputs": {
                "visual": visual_input,
                "auditory": auditory_input,
                "tactile": tactile_input,
            },
            "expected_output": expected_output,
            "metadata": {
                "visual_strength": visual_strength,
                "auditory_strength": auditory_strength,
                "tactile_strength": tactile_strength,
                "fused_decision": fused_decision,
                "modality_conflict": max(
                    visual_strength, auditory_strength, tactile_strength
                )
                - min(visual_strength, auditory_strength, tactile_strength),
            },
        }

    def _create_adaptive_learning_task(self, params: TaskParameters) -> Dict[str, Any]:
        """Create adaptive learning task with dynamic changes."""
        # Simulate changing environment
        if params.dynamic_changes:
            # Randomly change task parameters during execution
            params.input_noise = np.random.uniform(0, 0.3)
            params.missing_modalities = random.sample(
                ["visual", "auditory", "tactile"], random.randint(0, 2)
            )

        # Generate inputs with dynamic characteristics
        visual_input = np.random.random(9)
        auditory_input = np.random.random(9)
        tactile_input = np.random.random(9)

        # Add time-varying noise
        if params.input_noise > 0:
            time_factor = time.time() % 10 / 10  # Time-varying factor
            visual_input = self.noise_generator.add_gaussian_noise(
                visual_input, params.input_noise * time_factor
            )
            auditory_input = self.noise_generator.add_gaussian_noise(
                auditory_input, params.input_noise * time_factor
            )
            tactile_input = self.noise_generator.add_gaussian_noise(
                tactile_input, params.input_noise * time_factor
            )

        # Adaptive decision rule
        adaptive_threshold = 0.5 + 0.2 * np.sin(time.time())  # Time-varying threshold
        total_input = (
            np.mean(visual_input) + np.mean(auditory_input) + np.mean(tactile_input)
        )
        expected_output = 1 if total_input > adaptive_threshold else 0

        return {
            "level": TaskLevel.LEVEL_5,
            "parameters": params,
            "inputs": {
                "visual": visual_input,
                "auditory": auditory_input,
                "tactile": tactile_input,
            },
            "expected_output": expected_output,
            "metadata": {
                "adaptive_threshold": adaptive_threshold,
                "total_input": total_input,
                "time_factor": time.time() % 10 / 10,
            },
        }

    def _create_adversarial_task(self, params: TaskParameters) -> Dict[str, Any]:
        """Create adversarial testing task."""
        # Generate clean inputs
        visual_input = np.random.random(9)
        auditory_input = np.random.random(9)
        tactile_input = np.random.random(9)

        # Apply adversarial perturbations
        if params.adversarial_strength > 0:
            visual_input = self.adversarial_generator.fgsm_attack(
                visual_input, params.adversarial_strength
            )
            auditory_input = self.adversarial_generator.fgsm_attack(
                auditory_input, params.adversarial_strength
            )
            tactile_input = self.adversarial_generator.fgsm_attack(
                tactile_input, params.adversarial_strength
            )

        # Decision based on adversarial robustness
        input_strength = (
            np.mean(visual_input) + np.mean(auditory_input) + np.mean(tactile_input)
        )
        expected_output = 1 if input_strength > 0.5 else 0

        return {
            "level": TaskLevel.LEVEL_6,
            "parameters": params,
            "inputs": {
                "visual": visual_input,
                "auditory": auditory_input,
                "tactile": tactile_input,
            },
            "expected_output": expected_output,
            "metadata": {
                "adversarial_strength": params.adversarial_strength,
                "input_strength": input_strength,
                "perturbation_magnitude": np.mean(
                    [
                        np.linalg.norm(visual_input - np.random.random(9)),
                        np.linalg.norm(auditory_input - np.random.random(9)),
                        np.linalg.norm(tactile_input - np.random.random(9)),
                    ]
                ),
            },
        }

    def _create_real_world_task(self, params: TaskParameters) -> Dict[str, Any]:
        """Create real-world simulation task."""
        # Simulate realistic sensor data
        # Visual: object detection with occlusion
        visual_input = np.random.random(9)
        if np.random.random() < 0.3:  # 30% chance of occlusion
            occlusion_mask = np.random.choice([0, 1], size=9, p=[0.3, 0.7])
            visual_input *= occlusion_mask

        # Auditory: speech recognition with background noise
        auditory_input = np.random.random(9)
        if np.random.random() < 0.4:  # 40% chance of background noise
            noise_mask = np.random.normal(0, 0.2, 9)
            auditory_input = np.clip(auditory_input + noise_mask, 0, 1)

        # Tactile: object manipulation with varying pressure
        tactile_input = np.random.random(9)
        pressure_variation = np.random.normal(1, 0.3, 9)
        tactile_input = np.clip(tactile_input * pressure_variation, 0, 1)

        # Real-world decision: object interaction
        object_present = np.mean(visual_input) > 0.3
        sound_detected = np.mean(auditory_input) > 0.4
        contact_made = np.mean(tactile_input) > 0.5

        # Complex decision rule
        if object_present and sound_detected and contact_made:
            expected_output = 1  # Successful interaction
        elif object_present and (sound_detected or contact_made):
            expected_output = 0.5  # Partial interaction
        else:
            expected_output = 0  # No interaction

        return {
            "level": TaskLevel.LEVEL_7,
            "parameters": params,
            "inputs": {
                "visual": visual_input,
                "auditory": auditory_input,
                "tactile": tactile_input,
            },
            "expected_output": expected_output,
            "metadata": {
                "object_present": object_present,
                "sound_detected": sound_detected,
                "contact_made": contact_made,
                "interaction_quality": expected_output,
            },
        }

    def get_task_statistics(self) -> Dict[str, Any]:
        """Get statistics about task performance."""
        if not self.task_history:
            return {}

        stats = {
            "total_tasks": len(self.task_history),
            "level_distribution": {},
            "average_complexity": 0,
            "success_rate": 0,
            "robustness_metrics": {},
        }

        # Level distribution
        for task in self.task_history:
            level = task["level"].value
            if level not in stats["level_distribution"]:
                stats["level_distribution"][level] = 0
            stats["level_distribution"][level] += 1

        # Average complexity
        complexity_scores = []
        for task in self.task_history:
            params = task["parameters"]
            complexity = (
                params.input_noise
                + len(params.missing_modalities) * 0.2
                + params.temporal_complexity
                + params.adversarial_strength
            )
            complexity_scores.append(complexity)

        stats["average_complexity"] = np.mean(complexity_scores)

        return stats


# Global task manager instance
task_manager = TaskComplexityManager()
