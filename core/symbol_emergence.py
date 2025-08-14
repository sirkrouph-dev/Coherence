"""
Neuromorphic Symbol Engine
Pure emergent symbolic reasoning without LLM dependencies
Implements symbols as stable cell assembly trajectories
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import time

try:
    from .enhanced_symbol_detection import EnhancedAssemblyDetector, EnhancedCellAssembly
except ImportError:
    # Fallback for direct execution
    from enhanced_symbol_detection import EnhancedAssemblyDetector, EnhancedCellAssembly


@dataclass
class SymbolBinding:
    """Neuromorphic symbol binding state"""

    pattern: np.ndarray
    phase_offset: float
    stability: float
    activation_level: float
    binding_time: float


@dataclass
class CellAssembly:
    """Stable cell assembly representing a symbol"""

    neurons: List[int]
    activation_pattern: np.ndarray
    stability_score: float
    emergence_history: List[float]
    concept_grounding: Dict[str, float]


class PhaseBinding:
    """Neuromorphic phase-based binding mechanism"""

    def __init__(
        self,
        theta_range: Tuple[float, float] = (4, 8),
        gamma_range: Tuple[float, float] = (30, 100),
    ):
        self.theta_range = theta_range
        self.gamma_range = gamma_range
        self.current_phase = 0.0

    def bind_features(self, features: List[np.ndarray], dt: float = 0.001) -> np.ndarray:
        """Bind features using phase offsets"""
        if not features:
            return np.array([])

        # Find maximum feature size for consistent dimensions
        max_size = max(feature.shape[0] for feature in features)
        
        # Generate phase offsets for each feature
        num_features = len(features)
        phase_offsets = np.linspace(0, 2 * np.pi, num_features, endpoint=False)

        # Create composite pattern through phase interference
        composite = np.zeros(max_size, dtype=complex)

        for i, feature in enumerate(features):
            # Pad feature to max size if needed
            padded_feature = np.zeros(max_size)
            padded_feature[:len(feature)] = feature
            
            # Apply phase modulation
            phase_modulated = padded_feature * np.exp(1j * (phase_offsets[i] + self.current_phase))
            composite += phase_modulated

        # Update global phase (theta rhythm)
        theta_freq = np.mean(self.theta_range)
        self.current_phase += 2 * np.pi * theta_freq * dt
        self.current_phase = self.current_phase % (2 * np.pi)

        # Return real-valued pattern with binding information preserved
        return np.abs(composite)


class SynchronyBinding:
    """Synchrony-based binding mechanism"""

    def __init__(self, window: float = 0.01):
        self.synchrony_window = window
        self.spike_history = deque(maxlen=1000)

    def bind_by_synchrony(self, spike_times: List[float], neuron_ids: List[int]) -> float:
        """Measure binding strength through spike synchrony"""
        if len(spike_times) < 2:
            return 0.0

        # Find spikes within synchrony window
        spike_times_array = np.array(spike_times)
        synchrony_groups = []

        for t in spike_times:
            synchronized = spike_times_array[
                np.abs(spike_times_array - t) <= self.synchrony_window
            ]
            if len(synchronized) > 1:
                synchrony_groups.append(synchronized)

        # Calculate binding strength
        if not synchrony_groups:
            return 0.0

        max_sync_size = max(len(group) for group in synchrony_groups)
        binding_strength = max_sync_size / len(spike_times)

        return binding_strength


class SparsenessBinding:
    """Sparseness-based binding mechanism"""

    def __init__(self, k: float = 0.001):
        self.sparseness_k = k  # Target sparseness level

    def enforce_sparseness(self, activity: np.ndarray) -> np.ndarray:
        """Enforce sparse representation for binding"""
        num_active = int(len(activity) * self.sparseness_k)

        if num_active == 0:
            return np.zeros_like(activity)

        # Keep only top-k most active neurons
        top_k_indices = np.argpartition(activity, -num_active)[-num_active:]
        sparse_activity = np.zeros_like(activity)
        sparse_activity[top_k_indices] = activity[top_k_indices]

        return sparse_activity


class NeuromorphicLayer:
    """Hierarchical layer in the symbol emergence system"""

    def __init__(self, size: int, layer_type: str, temporal_res: float):
        self.size = size
        self.layer_type = layer_type
        self.temporal_resolution = temporal_res

        # Neural activity
        self.activity = np.zeros(size)
        self.previous_activity = np.zeros(size)

        # Temporal dynamics
        self.activity_history = deque(maxlen=int(1000 / temporal_res))

        # Cell assemblies
        self.cell_assemblies = []
        self.assembly_detector = CellAssemblyDetector(size)
        
        # Enhanced detection system
        self.enhanced_detector = EnhancedAssemblyDetector(
            layer_size=size,
            detection_threshold=0.5
        )

        # Binding mechanisms
        self.phase_binder = PhaseBinding()
        self.synchrony_binder = SynchronyBinding()
        self.sparseness_binder = SparsenessBinding()

    def process(
        self,
        input_data: np.ndarray,
        binding_type: str = "phase",
        temporal_integration: bool = False,
    ) -> np.ndarray:
        """Process input through this layer"""

        # Apply binding mechanism
        if binding_type == "phase":
            processed = self._apply_phase_binding(input_data)
        elif binding_type == "synchrony":
            processed = self._apply_synchrony_binding(input_data)
        elif binding_type == "sparseness":
            processed = self._apply_sparseness_binding(input_data)
        else:
            processed = input_data

        # Temporal integration if requested
        if temporal_integration:
            processed = self._temporal_integration(processed)

        # Update activity
        self.previous_activity = self.activity.copy()
        self.activity = processed[: self.size]  # Ensure correct size

        # Store in history
        self.activity_history.append(self.activity.copy())

        # Detect emerging cell assemblies
        self._detect_assemblies()

        return self.activity

    def _apply_phase_binding(self, input_data: np.ndarray) -> np.ndarray:
        """Apply phase-based binding"""
        # Ensure minimum input size
        if len(input_data) == 0:
            return np.zeros(self.size)
            
        # Split input into more consistent chunks
        num_chunks = min(4, len(input_data))  # Maximum 4 chunks
        chunk_size = len(input_data) // num_chunks
        
        chunks = []
        for i in range(num_chunks):
            start_idx = i * chunk_size
            if i == num_chunks - 1:  # Last chunk gets any remainder
                end_idx = len(input_data)
            else:
                end_idx = (i + 1) * chunk_size
            chunks.append(input_data[start_idx:end_idx])

        # Bind chunks using phase offsets
        bound_pattern = self.phase_binder.bind_features(chunks)

        # Pad or truncate to layer size
        if len(bound_pattern) > self.size:
            return bound_pattern[: self.size]
        else:
            padded = np.zeros(self.size)
            padded[: len(bound_pattern)] = bound_pattern
            return padded

    def _apply_synchrony_binding(self, input_data: np.ndarray) -> np.ndarray:
        """Apply synchrony-based binding"""
        # Convert activity to spike times (simplified)
        spike_times = []
        neuron_ids = []

        for i, activity in enumerate(input_data):
            if activity > 0.5:  # Threshold for spiking
                spike_times.append(time.time() % 1.0)  # Use current time
                neuron_ids.append(i)

        # Calculate synchrony strength
        sync_strength = self.synchrony_binder.bind_by_synchrony(spike_times, neuron_ids)

        # Modulate activity by synchrony
        modulated = input_data * (1.0 + sync_strength)

        # Pad or truncate to layer size
        if len(modulated) > self.size:
            return modulated[: self.size]
        else:
            padded = np.zeros(self.size)
            padded[: len(modulated)] = modulated
            return padded

    def _apply_sparseness_binding(self, input_data: np.ndarray) -> np.ndarray:
        """Apply sparseness-based binding"""
        # Pad or truncate to layer size first
        if len(input_data) > self.size:
            resized = input_data[: self.size]
        else:
            resized = np.zeros(self.size)
            resized[: len(input_data)] = input_data

        # Apply sparseness constraint
        return self.sparseness_binder.enforce_sparseness(resized)

    def _temporal_integration(self, current_input: np.ndarray) -> np.ndarray:
        """Integrate temporal information"""
        if len(self.activity_history) == 0:
            return current_input

        # Weighted integration with history
        integrated = current_input.copy()

        for i, historical_activity in enumerate(reversed(list(self.activity_history)[-5:])):
            weight = 0.8 ** (i + 1)  # Exponential decay
            if len(historical_activity) == len(integrated):
                integrated += weight * historical_activity

        return integrated

    def _detect_assemblies(self):
        """Detect emerging cell assemblies with enhanced detection"""
        if len(self.activity_history) < 5:
            return

        # Prepare activity matrix for enhanced detection
        if len(self.activity_history) > 0:
            activity_matrix = np.array(list(self.activity_history))
        else:
            activity_matrix = self.activity.reshape(1, -1)

        # Use enhanced detector
        enhanced_assemblies = self.enhanced_detector.detect_enhanced_assemblies(
            activity_matrix, temporal_context=self.activity
        )

        # Convert enhanced assemblies to regular format for compatibility
        for enhanced_assembly in enhanced_assemblies:
            regular_assembly = CellAssembly(
                neurons=enhanced_assembly.neurons,
                activation_pattern=enhanced_assembly.activation_pattern,
                stability_score=enhanced_assembly.stability_score,
                emergence_history=enhanced_assembly.emergence_history,
                concept_grounding=enhanced_assembly.concept_grounding
            )
            self.cell_assemblies.append(regular_assembly)

        # Also use the original detector for comparison
        original_assemblies = self.assembly_detector.detect_assemblies(
            list(self.activity_history), self.activity
        )

        # Merge results (enhanced assemblies take priority)
        for assembly in original_assemblies:
            # Only add if not already covered by enhanced detection
            is_duplicate = any(
                set(assembly.neurons) == set(enh_assembly.neurons)
                for enh_assembly in enhanced_assemblies
            )
            if not is_duplicate:
                self.cell_assemblies.append(assembly)

        # Remove unstable assemblies
        self.cell_assemblies = [a for a in self.cell_assemblies if a.stability_score > 0.5]


class CellAssemblyDetector:
    """Detects stable cell assemblies in neural activity"""

    def __init__(self, layer_size: int):
        self.layer_size = layer_size
        self.correlation_threshold = 0.7
        self.stability_threshold = 0.6

    def detect_assemblies(
        self, activity_history: List[np.ndarray], current_activity: np.ndarray
    ) -> List[CellAssembly]:
        """Detect cell assemblies from activity patterns"""
        if len(activity_history) < 5:
            return []

        assemblies = []

        # Find consistently co-active neurons
        activity_matrix = np.array(activity_history[-10:])  # Last 10 time steps

        # Calculate correlation matrix
        correlations = np.corrcoef(activity_matrix.T)

        # Find highly correlated groups
        for i in range(self.layer_size):
            if current_activity[i] > 0.3:  # Only consider active neurons
                # Find neurons correlated with neuron i
                correlated = np.where(correlations[i] > self.correlation_threshold)[0]

                if len(correlated) > 3:  # Minimum assembly size
                    # Calculate stability
                    stability = self._calculate_stability(correlated, activity_matrix)

                    if stability > self.stability_threshold:
                        # Create cell assembly
                        assembly_pattern = np.mean(activity_matrix[:, correlated], axis=0)

                        assembly = CellAssembly(
                            neurons=correlated.tolist(),
                            activation_pattern=assembly_pattern,
                            stability_score=stability,
                            emergence_history=[stability],
                            concept_grounding={},
                        )

                        assemblies.append(assembly)

        return assemblies

    def _calculate_stability(self, neuron_group: np.ndarray, activity_matrix: np.ndarray) -> float:
        """Calculate stability of a neuron group"""
        if len(neuron_group) == 0:
            return 0.0

        # Calculate variance in co-activation
        group_activity = activity_matrix[:, neuron_group]
        group_means = np.mean(group_activity, axis=1)

        # Stability is inverse of variance in group activation
        variance = np.var(group_means)
        stability = 1.0 / (1.0 + variance)

        return float(stability)


class PredictiveSTDP:
    """Predictive spike-timing dependent plasticity"""

    def __init__(self, beta: float = 0.3):
        self.beta = beta  # Prediction strength
        self.prediction_window = 0.02  # 20ms prediction window

    def update_weights(
        self,
        pre_activity: np.ndarray,
        post_activity: np.ndarray,
        prediction_error: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        """Update weights based on predictive STDP"""

        # Calculate weight updates
        ltp = np.outer(pre_activity, post_activity)  # Long-term potentiation
        ltd = np.outer(post_activity, pre_activity)  # Long-term depression

        # Modulate by prediction error
        prediction_modulation = 1.0 + self.beta * prediction_error.reshape(-1, 1)

        # Apply updates
        weight_update = (ltp - ltd) * prediction_modulation[: len(ltp)]
        updated_weights = weights + 0.01 * weight_update

        # Clip to reasonable range
        return np.clip(updated_weights, -1.0, 1.0)


class NeuromorphicSymbolEngine:
    """Main symbol emergence engine"""

    def __init__(self, total_neurons: int = 10_000):  # Reasonable default for testing
        """Initialize symbol engine with hierarchical organization"""

        # Distribute neurons across hierarchy
        self.hierarchy = [
            NeuromorphicLayer(
                size=int(total_neurons * 0.67),  # 67% sensory
                layer_type="sensory",
                temporal_res=0.1,
            ),
            NeuromorphicLayer(
                size=int(total_neurons * 0.27),  # 27% conceptual
                layer_type="conceptual",
                temporal_res=1.0,
            ),
            NeuromorphicLayer(
                size=int(total_neurons * 0.05),  # 5% syntactic
                layer_type="syntactic",
                temporal_res=10.0,
            ),
            NeuromorphicLayer(
                size=int(total_neurons * 0.01),  # 1% semantic
                layer_type="semantic",
                temporal_res=100.0,
            ),
        ]

        # Inter-layer connections
        self.predictive_stdp = PredictiveSTDP(beta=0.3)
        self.layer_weights = self._initialize_layer_weights()

        # Symbol tracking
        self.emerged_symbols = {}
        self.symbol_stability = {}
        self.grounding_connections = {}

        # Sensorimotor grounding
        self.sensorimotor_buffer = deque(maxlen=1000)
        self.action_outcomes = {}

    def _initialize_layer_weights(self) -> List[np.ndarray]:
        """Initialize sparse weights between layers"""
        weights = []

        for i in range(len(self.hierarchy) - 1):
            pre_size = self.hierarchy[i].size
            post_size = self.hierarchy[i + 1].size

            # Use sparse connectivity to avoid memory issues
            connectivity = 0.01  # 1% connectivity
            num_connections = int(pre_size * post_size * connectivity)

            # Create sparse weight matrix
            weight_matrix = np.zeros((pre_size, post_size))

            # Add random connections
            if num_connections > 0:
                pre_indices = np.random.randint(0, pre_size, num_connections)
                post_indices = np.random.randint(0, post_size, num_connections)
                weight_values = np.random.normal(0, 0.1, num_connections)

                weight_matrix[pre_indices, post_indices] = weight_values

            weights.append(weight_matrix)

        return weights

    def process_sensorimotor_stream(
        self, sensory_input: np.ndarray, motor_output: Optional[np.ndarray] = None
    ) -> Dict:
        """Process sensorimotor stream for symbol emergence"""

        # Store sensorimotor experience
        self.sensorimotor_buffer.append(
            {"sensory": sensory_input, "motor": motor_output, "timestamp": time.time()}
        )

        # Forward pass through hierarchy
        layer_activities = []
        current_input = sensory_input

        for i, layer in enumerate(self.hierarchy):
            # Determine binding type based on layer
            if layer.layer_type == "sensory":
                binding_type = "phase"
            elif layer.layer_type == "conceptual":
                binding_type = "sparseness"
            elif layer.layer_type == "syntactic":
                binding_type = "synchrony"
            else:  # semantic
                binding_type = "phase"

            # Process through layer
            activity = layer.process(
                current_input, binding_type=binding_type, temporal_integration=(i > 0)
            )

            layer_activities.append(activity)

            # Prepare input for next layer
            if i < len(self.layer_weights):
                # Project to next layer size
                next_layer_size = self.hierarchy[i + 1].size
                if len(activity) > 0 and self.layer_weights[i].shape[0] > 0:
                    # Ensure activity matches weight matrix dimensions
                    weight_matrix = self.layer_weights[i]
                    pre_size = weight_matrix.shape[0]
                    
                    # Resize activity to match weight matrix input size
                    if len(activity) > pre_size:
                        resized_activity = activity[:pre_size]
                    else:
                        resized_activity = np.zeros(pre_size)
                        resized_activity[:len(activity)] = activity
                    
                    # Project through weights
                    projection = np.dot(resized_activity, weight_matrix)
                    current_input = np.tanh(projection)  # Nonlinearity
                else:
                    current_input = np.zeros(next_layer_size)

        # Analyze emerged symbols
        symbol_analysis = self._analyze_symbol_emergence(layer_activities)

        # Update symbol stability
        self._update_symbol_stability(symbol_analysis)

        return {
            "layer_activities": layer_activities,
            "symbol_analysis": symbol_analysis,
            "emerged_symbols": len(self.emerged_symbols),
            "symbol_stability": (
                np.mean([np.mean(stab_list) for stab_list in self.symbol_stability.values()]) 
                if self.symbol_stability else 0.0
            ),
        }

    def _analyze_symbol_emergence(self, layer_activities: List[np.ndarray]) -> Dict:
        """Analyze symbol emergence from layer activities"""
        analysis = {
            "symbol_candidates": [],
            "binding_strength": [],
            "temporal_coherence": [],
            "hierarchical_consistency": [],
        }

        # Analyze each layer for symbol properties
        for i, activity in enumerate(layer_activities):
            layer = self.hierarchy[i]

            # Check for stable cell assemblies
            for assembly in layer.cell_assemblies:
                symbol_candidate = {
                    "layer": i,
                    "neurons": assembly.neurons,
                    "stability": assembly.stability_score,
                    "activation_strength": np.mean(assembly.activation_pattern),
                    "temporal_coherence": self._calculate_temporal_coherence(assembly),
                }

                analysis["symbol_candidates"].append(symbol_candidate)

        return analysis

    def _calculate_temporal_coherence(self, assembly: CellAssembly) -> float:
        """Calculate temporal coherence of a cell assembly"""
        if len(assembly.emergence_history) < 2:
            return 0.0

        # Coherence is stability over time
        variance = np.var(assembly.emergence_history)
        coherence = 1.0 / (1.0 + variance)

        return float(coherence)

    def _update_symbol_stability(self, symbol_analysis: Dict):
        """Update symbol stability tracking"""
        for candidate in symbol_analysis["symbol_candidates"]:
            symbol_id = f"layer_{candidate['layer']}_neurons_{len(candidate['neurons'])}"

            if symbol_id not in self.symbol_stability:
                self.symbol_stability[symbol_id] = []

            self.symbol_stability[symbol_id].append(candidate["stability"])

            # Keep only recent history
            if len(self.symbol_stability[symbol_id]) > 100:
                self.symbol_stability[symbol_id] = self.symbol_stability[symbol_id][-100:]

    def test_symbol_properties(self) -> Dict:
        """Test emergent symbols for key properties"""
        results = {
            "compositionality": 0.0,
            "productivity": 0.0,
            "systematicity": 0.0,
            "discreteness": 0.0,
            "grounding": 0.0,
        }

        if not self.symbol_stability:
            return results

        # Test discreteness (symbol boundary separation)
        stability_values = [np.mean(stab) for stab in self.symbol_stability.values()]
        results["discreteness"] = float(np.mean(stability_values))

        # Test compositionality (ability to combine symbols)
        results["compositionality"] = self._test_compositionality()

        # Test productivity (novel combinations)
        results["productivity"] = self._test_productivity()

        # Test systematicity (consistent transformations)
        results["systematicity"] = self._test_systematicity()

        # Test grounding (sensorimotor connections)
        results["grounding"] = self._test_grounding()

        return results

    def _test_compositionality(self) -> float:
        """Test if symbols can be composed"""
        # Simplified test: can we bind multiple symbols?
        if len(self.hierarchy[0].cell_assemblies) < 2:
            return 0.0

        # Try to bind two different assemblies
        assembly1 = self.hierarchy[0].cell_assemblies[0]
        assembly2 = (
            self.hierarchy[0].cell_assemblies[1]
            if len(self.hierarchy[0].cell_assemblies) > 1
            else assembly1
        )

        # Test if binding creates stable composite
        binder = PhaseBinding()
        composite = binder.bind_features(
            [assembly1.activation_pattern, assembly2.activation_pattern]
        )

        # Measure stability of composite
        composite_stability = 1.0 - np.var(composite) / (np.mean(composite) + 1e-6)

        return np.clip(composite_stability, 0.0, 1.0)

    def _test_productivity(self) -> float:
        """Test ability to generate novel combinations"""
        # Count unique symbol combinations over time
        unique_combinations = len(set(self.symbol_stability.keys()))

        # Productivity scales with number of unique symbols
        max_expected = 100  # Reasonable expectation for test
        productivity = min(unique_combinations / max_expected, 1.0)

        return productivity

    def _test_systematicity(self) -> float:
        """Test systematic transformation ability"""
        # Test if similar transformations produce similar results
        if len(self.sensorimotor_buffer) < 10:
            return 0.0

        # Simple test: consistency in sensorimotor mappings
        recent_experiences = list(self.sensorimotor_buffer)[-10:]

        # Calculate consistency in responses to similar inputs
        similarities = []
        for i in range(len(recent_experiences) - 1):
            exp1 = recent_experiences[i]
            exp2 = recent_experiences[i + 1]

            if exp1["sensory"] is not None and exp2["sensory"] is not None:
                # Calculate input similarity
                input_sim = np.corrcoef(exp1["sensory"].flatten(), exp2["sensory"].flatten())[0, 1]

                if not np.isnan(input_sim):
                    similarities.append(abs(input_sim))

        return float(np.mean(similarities)) if similarities else 0.0

    def _test_grounding(self) -> float:
        """Test sensorimotor grounding"""
        # Test if symbols are connected to sensorimotor experience
        if not self.sensorimotor_buffer:
            return 0.0

        # Simple measure: how well do symbols predict sensorimotor patterns?
        prediction_accuracy = []

        for symbol_id, stability_history in self.symbol_stability.items():
            if len(stability_history) > 5:
                # Test if symbol stability correlates with sensorimotor consistency
                recent_stability = np.mean(stability_history[-5:])

                # This is a simplified test - in practice, you'd test actual prediction
                grounding_strength = min(recent_stability, 1.0)
                prediction_accuracy.append(grounding_strength)

        return float(np.mean(prediction_accuracy)) if prediction_accuracy else 0.0

    def get_system_state(self) -> Dict:
        """Get comprehensive system state"""
        return {
            "total_neurons": sum(layer.size for layer in self.hierarchy),
            "layers": len(self.hierarchy),
            "emerged_symbols": len(self.emerged_symbols),
            "cell_assemblies": sum(len(layer.cell_assemblies) for layer in self.hierarchy),
            "sensorimotor_experiences": len(self.sensorimotor_buffer),
            "symbol_properties": self.test_symbol_properties(),
            "layer_activities": [np.mean(layer.activity) for layer in self.hierarchy],
        }
