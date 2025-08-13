"""
Memory subsystems for neuromorphic computing.
Implements short-term (working) and long-term memory abstractions
backed by recurrent connections and weight consolidation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import time

from core.neurons import AdaptiveExponentialIntegrateAndFire
from core.synapses import STDP_Synapse, SynapseType
from core.network import NeuromorphicNetwork
from core.logging_utils import neuromorphic_logger


class MemoryType(Enum):
    """Types of memory systems."""

    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


@dataclass
class MemoryTrace:
    """Represents a memory trace in the system."""

    pattern: np.ndarray
    timestamp: float
    strength: float
    access_count: int = 0
    last_access: Optional[float] = None
    memory_type: MemoryType = MemoryType.SHORT_TERM
    metadata: Dict[str, Any] = field(default_factory=dict)


class RecurrentMemoryNetwork:
    """
    Recurrent neural network for memory storage and retrieval.
    Uses recurrent connections to maintain persistent activity patterns.
    """

    def __init__(
        self,
        n_neurons: int = 100,
        n_recurrent: int = 50,
        sparsity: float = 0.2,
        recurrent_strength: float = 2.0,
        tau_decay: float = 100.0,  # ms
    ):
        """
        Initialize recurrent memory network.

        Args:
            n_neurons: Number of neurons in the network
            n_recurrent: Number of recurrent connections per neuron
            sparsity: Connection sparsity
            recurrent_strength: Strength of recurrent connections
            tau_decay: Time constant for activity decay
        """
        self.n_neurons = n_neurons
        self.n_recurrent = n_recurrent
        self.sparsity = sparsity
        self.recurrent_strength = recurrent_strength
        self.tau_decay = tau_decay

        # Initialize neurons
        self.neurons = [
            AdaptiveExponentialIntegrateAndFire(i) for i in range(n_neurons)
        ]

        # Initialize recurrent connectivity matrix
        self.connectivity = self._create_recurrent_connections()

        # Activity state
        self.activity = np.zeros(n_neurons)
        self.sustained_activity = np.zeros(n_neurons)

    def _create_recurrent_connections(self) -> np.ndarray:
        """Create sparse recurrent connectivity matrix."""
        W = np.zeros((self.n_neurons, self.n_neurons))

        for i in range(self.n_neurons):
            # Select random targets for recurrent connections
            n_connections = min(self.n_recurrent, self.n_neurons - 1)
            targets = np.random.choice(
                [j for j in range(self.n_neurons) if j != i],
                size=n_connections,
                replace=False,
            )

            # Set connection weights with some variability
            for j in targets:
                if np.random.random() < self.sparsity:
                    W[i, j] = self.recurrent_strength * (0.5 + np.random.random())

        return W

    def encode_pattern(self, pattern: np.ndarray) -> np.ndarray:
        """
        Encode input pattern into network activity.

        Args:
            pattern: Input pattern to encode

        Returns:
            Encoded activity pattern
        """
        # Ensure pattern matches network size
        if len(pattern) != self.n_neurons:
            pattern = np.resize(pattern, self.n_neurons)

        # Initialize activity with input pattern
        self.activity = pattern.copy()

        # Allow recurrent dynamics to shape the representation
        for _ in range(10):  # Settling iterations
            self._update_recurrent_activity(dt=1.0)

        return self.activity.copy()

    def _update_recurrent_activity(self, dt: float = 1.0):
        """Update network activity through recurrent connections."""
        # Compute recurrent input
        recurrent_input = np.dot(self.connectivity.T, self.activity)

        # Update activity with decay and recurrent input
        decay = -self.activity / self.tau_decay
        self.activity += dt * (decay + recurrent_input * 0.1)

        # Apply activation function (rectified linear)
        self.activity = np.maximum(0, self.activity)

        # Maintain sustained activity for working memory
        self.sustained_activity = 0.9 * self.sustained_activity + 0.1 * self.activity

    def retrieve_pattern(self, cue: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Retrieve stored pattern from network.

        Args:
            cue: Optional partial cue for pattern completion

        Returns:
            Retrieved pattern
        """
        if cue is not None:
            # Use cue to seed retrieval, ensure it matches network size
            if len(cue) != self.n_neurons:
                # Pad or resize cue to match network dimensions
                full_cue = np.zeros(self.n_neurons)
                full_cue[: min(len(cue), self.n_neurons)] = cue[
                    : min(len(cue), self.n_neurons)
                ]
                self.activity = full_cue
            else:
                self.activity = cue.copy()

        # Let recurrent dynamics complete the pattern
        for _ in range(20):  # Retrieval iterations
            self._update_recurrent_activity(dt=1.0)

        return self.activity.copy()


class WeightConsolidation:
    """
    Implements synaptic weight consolidation for long-term memory.
    Uses mechanisms inspired by synaptic tagging and capture.
    """

    def __init__(
        self,
        consolidation_threshold: float = 0.7,
        decay_rate: float = 0.01,
        protection_factor: float = 0.9,
    ):
        """
        Initialize weight consolidation system.

        Args:
            consolidation_threshold: Threshold for consolidation
            decay_rate: Rate of weight decay without consolidation
            protection_factor: Protection from decay for consolidated weights
        """
        self.consolidation_threshold = consolidation_threshold
        self.decay_rate = decay_rate
        self.protection_factor = protection_factor

        # Track consolidation state
        self.consolidation_tags = {}
        self.consolidation_strength = {}
        self.importance_scores = {}

    def tag_for_consolidation(self, synapse_id: int, importance: float):
        """
        Tag synapse for potential consolidation.

        Args:
            synapse_id: ID of synapse to tag
            importance: Importance score for consolidation
        """
        self.consolidation_tags[synapse_id] = time.time()
        self.importance_scores[synapse_id] = importance

        # Trigger consolidation if importance exceeds threshold
        if importance > self.consolidation_threshold:
            self._consolidate_synapse(synapse_id)

    def _consolidate_synapse(self, synapse_id: int):
        """Consolidate synapse weights for long-term storage."""
        self.consolidation_strength[synapse_id] = self.importance_scores.get(
            synapse_id, 0.5
        )

        neuromorphic_logger.log_info(
            f"Consolidated synapse {synapse_id} with strength "
            f"{self.consolidation_strength[synapse_id]:.3f}"
        )

    def apply_decay(self, weights: Dict[int, float]) -> Dict[int, float]:
        """
        Apply decay to weights based on consolidation state.

        Args:
            weights: Dictionary of synapse weights

        Returns:
            Updated weights after decay
        """
        updated_weights = {}

        for syn_id, weight in weights.items():
            if syn_id in self.consolidation_strength:
                # Protected weight - slower decay
                protection = (
                    self.consolidation_strength[syn_id] * self.protection_factor
                )
                decay = self.decay_rate * (1 - protection)
            else:
                # Unprotected weight - normal decay
                decay = self.decay_rate

            updated_weights[syn_id] = weight * (1 - decay)

        return updated_weights

    def get_consolidation_status(self) -> Dict[str, Any]:
        """Get current consolidation status."""
        return {
            "n_consolidated": len(self.consolidation_strength),
            "n_tagged": len(self.consolidation_tags),
            "avg_strength": (
                np.mean(list(self.consolidation_strength.values()))
                if self.consolidation_strength
                else 0
            ),
        }


class ShortTermMemory:
    """
    Short-term (working) memory implementation.
    Uses sustained neural activity through recurrent connections.
    """

    def __init__(
        self,
        capacity: int = 7,  # Miller's magic number
        duration: float = 20000.0,  # ms (20 seconds)
        n_neurons_per_item: int = 50,
    ):
        """
        Initialize short-term memory.

        Args:
            capacity: Maximum number of items to store
            duration: Duration of memory retention (ms)
            n_neurons_per_item: Number of neurons per memory item
        """
        self.capacity = capacity
        self.duration = duration
        self.n_neurons_per_item = n_neurons_per_item

        # Initialize recurrent networks for each memory slot
        self.memory_slots = [
            RecurrentMemoryNetwork(
                n_neurons=n_neurons_per_item,
                recurrent_strength=2.5,
                tau_decay=duration / 10,
            )
            for _ in range(capacity)
        ]

        # Track memory contents
        self.memory_traces: List[Optional[MemoryTrace]] = [None] * capacity
        self.current_time = 0.0

    def store(self, pattern: np.ndarray, metadata: Optional[Dict] = None) -> bool:
        """
        Store pattern in short-term memory.

        Args:
            pattern: Pattern to store
            metadata: Optional metadata for the memory

        Returns:
            True if stored successfully, False if memory full
        """
        # Find available slot or replace oldest
        slot_idx = self._find_available_slot()
        if slot_idx is None:
            return False

        # Encode pattern in recurrent network
        encoded = self.memory_slots[slot_idx].encode_pattern(pattern)

        # Create memory trace
        trace = MemoryTrace(
            pattern=pattern,
            timestamp=self.current_time,
            strength=1.0,
            memory_type=MemoryType.SHORT_TERM,
            metadata=metadata or {},
        )

        self.memory_traces[slot_idx] = trace

        neuromorphic_logger.log_info(
            f"Stored pattern in STM slot {slot_idx}, " f"shape: {pattern.shape}"
        )

        return True

    def retrieve(self, cue: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Retrieve pattern from short-term memory.

        Args:
            cue: Optional partial cue for retrieval

        Returns:
            Retrieved pattern or None if not found
        """
        if cue is not None:
            # Find best matching slot
            best_match = None
            best_similarity = -1

            for idx, trace in enumerate(self.memory_traces):
                if trace is None:
                    continue

                # Compute similarity with cue
                # Ensure same dimensions for correlation
                cue_flat = cue.flatten()
                pattern_flat = trace.pattern.flatten()
                min_len = min(len(cue_flat), len(pattern_flat))
                similarity = np.corrcoef(cue_flat[:min_len], pattern_flat[:min_len])[
                    0, 1
                ]

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = idx

            if best_match is not None and best_similarity > 0.5:
                # Retrieve from best matching slot
                retrieved = self.memory_slots[best_match].retrieve_pattern(cue)
                self.memory_traces[best_match].access_count += 1
                self.memory_traces[best_match].last_access = self.current_time
                return retrieved

        # Return most recent if no cue provided
        for idx in reversed(range(self.capacity)):
            if self.memory_traces[idx] is not None:
                retrieved = self.memory_slots[idx].retrieve_pattern()
                self.memory_traces[idx].access_count += 1
                self.memory_traces[idx].last_access = self.current_time
                return retrieved

        return None

    def _find_available_slot(self) -> Optional[int]:
        """Find available memory slot."""
        # Check for empty slots
        for idx, trace in enumerate(self.memory_traces):
            if trace is None:
                return idx

        # Find oldest slot to replace
        oldest_idx = 0
        oldest_time = float("inf")

        for idx, trace in enumerate(self.memory_traces):
            if trace.timestamp < oldest_time:
                oldest_time = trace.timestamp
                oldest_idx = idx

        return oldest_idx

    def update(self, dt: float):
        """Update memory state and apply decay."""
        self.current_time += dt

        # Apply decay to memory traces
        for idx, trace in enumerate(self.memory_traces):
            if trace is None:
                continue

            # Calculate age
            age = self.current_time - trace.timestamp

            # Apply exponential decay
            trace.strength = np.exp(-age / self.duration)

            # Remove if strength below threshold
            if trace.strength < 0.1:
                self.memory_traces[idx] = None
                neuromorphic_logger.log_debug(f"STM slot {idx} decayed below threshold")

            # Update recurrent activity
            self.memory_slots[idx]._update_recurrent_activity(dt)


class LongTermMemory:
    """
    Long-term memory implementation.
    Uses weight consolidation and structural changes.
    """

    def __init__(
        self,
        n_neurons: int = 1000,
        consolidation_rate: float = 0.1,
        retrieval_threshold: float = 0.6,
    ):
        """
        Initialize long-term memory.

        Args:
            n_neurons: Number of neurons in LTM network
            consolidation_rate: Rate of memory consolidation
            retrieval_threshold: Threshold for successful retrieval
        """
        self.n_neurons = n_neurons
        self.consolidation_rate = consolidation_rate
        self.retrieval_threshold = retrieval_threshold

        # Initialize neural network for LTM
        self.network = NeuromorphicNetwork()
        self._build_ltm_network()

        # Weight consolidation system
        self.consolidator = WeightConsolidation(
            consolidation_threshold=0.7,
            decay_rate=0.001,  # Slower decay for LTM
        )

        # Memory storage
        self.memory_traces: List[MemoryTrace] = []
        self.memory_weights: Dict[int, np.ndarray] = {}

    def _build_ltm_network(self):
        """Build the LTM neural network architecture."""
        # Create a single layer for LTM storage
        self.network.add_layer("ltm_layer", self.n_neurons, neuron_type="adex")

        # Add recurrent connections within the layer for associative memory
        self.network.connect_layers(
            "ltm_layer",
            "ltm_layer",
            synapse_type="stdp",
            connection_probability=0.1,
            A_plus=0.005,  # Slower learning for LTM
            A_minus=0.005,
        )

        # Store neurons and synapses as lists for compatibility
        self.neurons = []
        self.synapses = []

    def consolidate(self, pattern: np.ndarray, importance: float = 0.5) -> bool:
        """
        Consolidate pattern into long-term memory.

        Args:
            pattern: Pattern to consolidate
            importance: Importance score for consolidation priority

        Returns:
            True if consolidation successful
        """
        # Create memory trace
        trace = MemoryTrace(
            pattern=pattern,
            timestamp=time.time(),
            strength=importance,
            memory_type=MemoryType.LONG_TERM,
        )

        # Encode pattern in network weights
        weight_pattern = self._encode_in_weights(pattern)

        # Tag synapses for consolidation based on importance
        for syn_id, weight in enumerate(weight_pattern.flatten()):
            if abs(weight) > 0.1:  # Only consolidate significant weights
                self.consolidator.tag_for_consolidation(
                    syn_id, importance * abs(weight)
                )

        # Store memory trace and weights
        trace_id = len(self.memory_traces)
        self.memory_traces.append(trace)
        self.memory_weights[trace_id] = weight_pattern

        neuromorphic_logger.log_info(
            f"Consolidated pattern to LTM with importance {importance:.2f}, "
            f"trace_id: {trace_id}"
        )

        return True

    def _encode_in_weights(self, pattern: np.ndarray) -> np.ndarray:
        """Encode pattern in synaptic weights."""
        # Reshape pattern to match network dimensions
        pattern_size = min(len(pattern), self.n_neurons)

        # Create weight matrix using outer product (Hebbian-like)
        pattern_neurons = pattern[:pattern_size]
        weight_matrix = np.outer(pattern_neurons, pattern_neurons)

        # Normalize weights
        weight_matrix = weight_matrix / (np.max(np.abs(weight_matrix)) + 1e-6)

        return weight_matrix

    def retrieve(self, cue: np.ndarray) -> Optional[np.ndarray]:
        """
        Retrieve pattern from long-term memory.

        Args:
            cue: Partial cue for retrieval

        Returns:
            Retrieved pattern or None if not found
        """
        best_match = None
        best_similarity = 0

        for trace_id, trace in enumerate(self.memory_traces):
            # Compute similarity between cue and stored pattern
            cue_flat = cue.flatten()[: len(trace.pattern.flatten())]
            pattern_flat = trace.pattern.flatten()

            if len(cue_flat) != len(pattern_flat):
                cue_flat = np.resize(cue_flat, len(pattern_flat))

            similarity = np.corrcoef(cue_flat, pattern_flat)[0, 1]

            # Weight similarity by trace strength
            weighted_similarity = similarity * trace.strength

            if weighted_similarity > best_similarity:
                best_similarity = weighted_similarity
                best_match = trace_id

        if best_match is not None and best_similarity > self.retrieval_threshold:
            # Reconstruct pattern from weights
            retrieved = self._reconstruct_from_weights(
                self.memory_weights[best_match], cue
            )

            # Update access statistics
            self.memory_traces[best_match].access_count += 1
            self.memory_traces[best_match].last_access = time.time()

            neuromorphic_logger.log_info(
                f"Retrieved pattern from LTM, trace_id: {best_match}, "
                f"similarity: {best_similarity:.3f}"
            )

            return retrieved

        return None

    def _reconstruct_from_weights(
        self, weight_matrix: np.ndarray, cue: np.ndarray
    ) -> np.ndarray:
        """Reconstruct pattern from weight matrix using cue."""
        # Use cue to activate network and reconstruct full pattern
        cue_neurons = cue.flatten()[: weight_matrix.shape[0]]

        # Iterative reconstruction (similar to Hopfield dynamics)
        state = cue_neurons.copy()
        for _ in range(10):
            # Update state based on weights
            activation = np.dot(weight_matrix, state)
            state = np.tanh(activation)  # Nonlinear activation

        return state

    def maintain(self, dt: float):
        """Maintain long-term memories through consolidation and decay."""
        # Apply gradual consolidation
        for trace_id, trace in enumerate(self.memory_traces):
            if trace_id in self.memory_weights:
                # Strengthen frequently accessed memories
                if trace.access_count > 0:
                    trace.strength = min(
                        1.0, trace.strength + self.consolidation_rate * dt
                    )

                # Apply slow decay to unused memories
                age = time.time() - trace.last_access if trace.last_access else 0
                if age > 86400:  # More than 1 day
                    trace.strength *= 1 - 0.001 * dt

        # Apply weight decay with protection for consolidated synapses
        # Note: Weight decay is handled internally by the network now
        # This is a placeholder for future integration with the consolidator
        pass


class IntegratedMemorySystem:
    """
    Integrated memory system combining STM and LTM.
    Handles transfer between memory systems and coordinated retrieval.
    """

    def __init__(
        self,
        stm_capacity: int = 7,
        ltm_size: int = 1000,
        consolidation_threshold: float = 0.7,
    ):
        """
        Initialize integrated memory system.

        Args:
            stm_capacity: Capacity of short-term memory
            ltm_size: Size of long-term memory network
            consolidation_threshold: Threshold for STM to LTM transfer
        """
        self.stm = ShortTermMemory(capacity=stm_capacity)
        self.ltm = LongTermMemory(n_neurons=ltm_size)
        self.consolidation_threshold = consolidation_threshold

        # Track memory statistics
        self.stats = {
            "stm_stores": 0,
            "ltm_consolidations": 0,
            "successful_retrievals": 0,
            "failed_retrievals": 0,
        }

    def store(self, pattern: np.ndarray, duration: str = "short") -> bool:
        """
        Store pattern in appropriate memory system.

        Args:
            pattern: Pattern to store
            duration: "short" or "long" term storage

        Returns:
            True if storage successful
        """
        if duration == "short":
            success = self.stm.store(pattern)
            if success:
                self.stats["stm_stores"] += 1
            return success
        else:
            # Direct consolidation to LTM
            success = self.ltm.consolidate(pattern, importance=0.8)
            if success:
                self.stats["ltm_consolidations"] += 1
            return success

    def retrieve(self, cue: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Retrieve pattern from memory system.

        Args:
            cue: Optional retrieval cue

        Returns:
            Retrieved pattern or None
        """
        # Try STM first (more recent)
        pattern = self.stm.retrieve(cue)
        if pattern is not None:
            self.stats["successful_retrievals"] += 1
            return pattern

        # Fall back to LTM
        if cue is not None:
            pattern = self.ltm.retrieve(cue)
            if pattern is not None:
                self.stats["successful_retrievals"] += 1
                return pattern

        self.stats["failed_retrievals"] += 1
        return None

    def consolidate_stm_to_ltm(self):
        """Transfer important STM contents to LTM."""
        for trace in self.stm.memory_traces:
            if trace is None:
                continue

            # Check if trace should be consolidated
            importance = trace.strength * (1 + 0.1 * trace.access_count)

            if importance > self.consolidation_threshold:
                self.ltm.consolidate(trace.pattern, importance)
                self.stats["ltm_consolidations"] += 1
                neuromorphic_logger.log_info(
                    f"Consolidated STM trace to LTM, " f"importance: {importance:.3f}"
                )

    def update(self, dt: float):
        """Update memory systems."""
        self.stm.update(dt)
        self.ltm.maintain(dt)

        # Periodic consolidation check
        if np.random.random() < 0.01:  # 1% chance per update
            self.consolidate_stm_to_ltm()

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        return {
            **self.stats,
            "stm_utilization": sum(1 for t in self.stm.memory_traces if t is not None)
            / self.stm.capacity,
            "ltm_traces": len(self.ltm.memory_traces),
            "consolidation_status": self.ltm.consolidator.get_consolidation_status(),
        }
