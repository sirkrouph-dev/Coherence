"""
Logging utilities for the neuromorphic system.
Provides a simplified logger interface for neural activity tracking.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional


class NeuromorphicLogger:
    """Basic neuromorphic logger for system components."""

    def __init__(self, log_file: str = "neuromorphic.log", level: int = logging.DEBUG):
        """
        Initialize neuromorphic logger.

        Args:
            log_file: Path to log file
            level: Logging level
        """
        self.log_file = log_file
        self.level = level
        self.setup_logger()

    def setup_logger(self):
        """Setup the logger with basic formatting."""
        # Create logger
        self.logger = logging.getLogger("neuromorphic_system")
        self.logger.setLevel(self.level)

        # Clear existing handlers
        self.logger.handlers.clear()

        # Create file handler
        file_handler = logging.FileHandler(self.log_file, mode="w")
        file_handler.setLevel(self.level)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)

        # Create console handler for important messages
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)

        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def log_neuron_activity(
        self,
        neuron_id: int,
        layer_name: str,
        membrane_potential: float,
        spiked: bool,
        adaptation_current: float = 0.0,
    ):
        """
        Log neuron activity.

        Args:
            neuron_id: ID of the neuron
            layer_name: Name of the layer
            membrane_potential: Current membrane potential
            spiked: Whether the neuron spiked
            adaptation_current: Current adaptation value
        """
        if spiked:
            self.logger.debug(
                f"SPIKE: Neuron {neuron_id} in {layer_name} fired! "
                f"V={membrane_potential:.2f}mV, I_w={adaptation_current:.3f}"
            )
        else:
            self.logger.debug(
                f"ACTIVITY: Neuron {neuron_id} in {layer_name} "
                f"V={membrane_potential:.2f}mV, I_w={adaptation_current:.3f}"
            )

    def log_synapse_activity(
        self,
        synapse_id: int,
        pre_neuron: int,
        post_neuron: int,
        weight: float,
        weight_change: float = 0.0,
    ):
        """
        Log synapse activity.

        Args:
            synapse_id: ID of the synapse
            pre_neuron: Presynaptic neuron ID
            post_neuron: Postsynaptic neuron ID
            weight: Current synaptic weight
            weight_change: Change in weight
        """
        if weight_change != 0:
            self.logger.debug(
                f"SYNAPSE: Synapse {synapse_id} ({pre_neuron}->{post_neuron}) "
                f"weight changed by {weight_change:.4f} to {weight:.4f}"
            )
        else:
            self.logger.debug(
                f"SYNAPSE: Synapse {synapse_id} ({pre_neuron}->{post_neuron}) "
                f"weight={weight:.4f}"
            )

    def log_network_activity(
        self,
        network_name: str,
        time_step: float,
        active_neurons: int,
        total_neurons: int,
        average_firing_rate: float = 0.0,
    ):
        """
        Log network-level activity.

        Args:
            network_name: Name of the network
            time_step: Current simulation time
            active_neurons: Number of active neurons
            total_neurons: Total number of neurons
            average_firing_rate: Average firing rate
        """
        self.logger.info(
            f"NETWORK: {network_name} at t={time_step:.1f}ms - "
            f"Active: {active_neurons}/{total_neurons} neurons, "
            f"Avg rate: {average_firing_rate:.2f}Hz"
        )

    def log_learning_event(
        self,
        learning_rule: str,
        layer_name: str,
        parameter: str,
        old_value: float,
        new_value: float,
    ):
        """
        Log learning events.

        Args:
            learning_rule: Name of the learning rule
            layer_name: Name of the layer
            parameter: Parameter being modified
            old_value: Previous value
            new_value: New value
        """
        self.logger.debug(
            f"LEARNING: {learning_rule} in {layer_name} - "
            f"{parameter}: {old_value:.4f} -> {new_value:.4f}"
        )

    def log_error(self, component: str, error_message: str):
        """
        Log error messages.

        Args:
            component: Component where error occurred
            error_message: Error description
        """
        self.logger.error(f"ERROR in {component}: {error_message}")

    def log_warning(self, component: str, warning_message: str):
        """
        Log warning messages.

        Args:
            component: Component generating warning
            warning_message: Warning description
        """
        self.logger.warning(f"WARNING in {component}: {warning_message}")

    def log_info(self, message: str):
        """
        Log general info messages.

        Args:
            message: Info message
        """
        self.logger.info(message)

    def log_debug(self, message: str):
        """
        Log debug messages.

        Args:
            message: Debug message
        """
        self.logger.debug(message)

    def log_system_event(self, event_type: str, metadata: Dict[str, Any]):
        """
        Log system-level events.

        Args:
            event_type: Type of system event
            metadata: Event metadata
        """
        self.logger.info(f"SYSTEM_EVENT: {event_type} - {metadata}")


# Create global logger instance
neuromorphic_logger = NeuromorphicLogger()


class TrainingTracker:
    """Tracks training progress and metrics."""

    def __init__(self):
        """Initialize training tracker."""
        self.epochs = []
        self.losses = []
        self.accuracies = []
        self.learning_rates = []
        self.start_time = None
        self.current_epoch = 0

    def start_training(self):
        """Mark the start of training."""
        self.start_time = time.time() if "time" in globals() else 0
        neuromorphic_logger.log_info("Training started")

    def log_epoch(self, epoch: int, loss: float, accuracy: float, learning_rate: float):
        """Log metrics for an epoch."""
        self.current_epoch = epoch
        self.epochs.append(epoch)
        self.losses.append(loss)
        self.accuracies.append(accuracy)
        self.learning_rates.append(learning_rate)

        neuromorphic_logger.log_info(
            f"Epoch {epoch}: Loss={loss:.4f}, Accuracy={accuracy:.4f}, LR={learning_rate:.6f}"
        )

    def end_training(self):
        """Mark the end of training."""
        if self.start_time:
            duration = (time.time() if "time" in globals() else 0) - self.start_time
            neuromorphic_logger.log_info(
                f"Training completed in {duration:.2f} seconds"
            )
        else:
            neuromorphic_logger.log_info("Training completed")

    def get_summary(self) -> Dict[str, Any]:
        """Get training summary."""
        return {
            "total_epochs": len(self.epochs),
            "final_loss": self.losses[-1] if self.losses else None,
            "final_accuracy": self.accuracies[-1] if self.accuracies else None,
            "best_accuracy": max(self.accuracies) if self.accuracies else None,
            "best_loss": min(self.losses) if self.losses else None,
        }


def trace_function(func):
    """Decorator to trace function calls."""

    def wrapper(*args, **kwargs):
        neuromorphic_logger.log_debug(
            f"Calling {func.__name__} with args={args}, kwargs={kwargs}"
        )
        result = func(*args, **kwargs)
        neuromorphic_logger.log_debug(f"{func.__name__} returned {result}")
        return result

    return wrapper
