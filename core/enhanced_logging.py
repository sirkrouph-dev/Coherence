"""
Enhanced logging system for comprehensive neuromorphic system monitoring.
Captures dynamic neural activity, spike timing, synaptic changes, and system performance.
"""

import inspect
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


@dataclass
class SpikeEvent:
    """Data structure for spike events."""

    neuron_id: int
    layer_name: str
    spike_time: float
    membrane_potential: float
    synaptic_inputs: Dict[str, float]
    neuromodulator_levels: Dict[str, float]


@dataclass
class MembranePotentialEvent:
    """Data structure for membrane potential changes."""

    neuron_id: int
    layer_name: str
    time_step: float
    membrane_potential: float
    synaptic_current: float
    adaptation_current: float
    refractory_time: float


@dataclass
class SynapticWeightEvent:
    """Data structure for synaptic weight changes."""

    synapse_id: int
    pre_neuron_id: int
    post_neuron_id: int
    old_weight: float
    new_weight: float
    weight_change: float
    learning_rule: str
    time_step: float


@dataclass
class NetworkStateEvent:
    """Data structure for network state snapshots."""

    time_step: float
    layer_name: str
    active_neurons: int
    total_neurons: int
    firing_rate: float
    average_membrane_potential: float
    spike_count: int


class EnhancedNeuromorphicLogger:
    """Enhanced logger with comprehensive neural dynamics tracking."""

    def __init__(
        self,
        log_file: str = "enhanced_trace.log",
        data_dir: str = "neural_data",
        level: int = logging.DEBUG,
    ):
        """
        Initialize enhanced neuromorphic logger.

        Args:
            log_file: Path to log file
            data_dir: Directory for storing neural data
            level: Logging level
        """
        self.log_file = log_file
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.level = level

        # Data storage for analysis
        self.spike_events: List[SpikeEvent] = []
        self.membrane_events: List[MembranePotentialEvent] = []
        self.synaptic_events: List[SynapticWeightEvent] = []
        self.network_events: List[NetworkStateEvent] = []
        self.performance_metrics: List[Dict[str, Any]] = []

        # Real-time tracking
        self.current_time = 0.0
        self.spike_count = 0
        self.firing_rates = {}
        self.membrane_potentials = {}

        self.setup_logger()

    def setup_logger(self):
        """Setup the logger with detailed formatting."""
        # Create logger
        self.logger = logging.getLogger("enhanced_neuromorphic_system")
        self.logger.setLevel(self.level)

        # Clear existing handlers
        self.logger.handlers.clear()

        # Create file handler with detailed formatting
        file_handler = logging.FileHandler(self.log_file, mode="w")
        file_handler.setLevel(self.level)

        # Create detailed formatter
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(funcName)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
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

    def log_spike_event(
        self,
        neuron_id: int,
        layer_name: str,
        spike_time: float,
        membrane_potential: float,
        synaptic_inputs: Dict[str, float],
        neuromodulator_levels: Dict[str, float],
    ):
        """Log detailed spike event with all context."""
        event = SpikeEvent(
            neuron_id=neuron_id,
            layer_name=layer_name,
            spike_time=spike_time,
            membrane_potential=membrane_potential,
            synaptic_inputs=synaptic_inputs,
            neuromodulator_levels=neuromodulator_levels,
        )

        self.spike_events.append(event)
        self.spike_count += 1

        # Update firing rate tracking
        if layer_name not in self.firing_rates:
            self.firing_rates[layer_name] = []
        self.firing_rates[layer_name].append(spike_time)

        # Log to file
        self.logger.debug(
            f"SPIKE: Neuron {neuron_id} in {layer_name} at {spike_time:.3f}ms, "
            f"V={membrane_potential:.2f}mV"
        )

    def log_membrane_potential(
        self,
        neuron_id: int,
        layer_name: str,
        time_step: float,
        membrane_potential: float,
        synaptic_current: float,
        adaptation_current: float = 0.0,
        refractory_time: float = 0.0,
    ):
        """Log membrane potential changes with full context."""
        event = MembranePotentialEvent(
            neuron_id=neuron_id,
            layer_name=layer_name,
            time_step=time_step,
            membrane_potential=membrane_potential,
            synaptic_current=synaptic_current,
            adaptation_current=adaptation_current,
            refractory_time=refractory_time,
        )

        self.membrane_events.append(event)

        # Track membrane potential history
        key = f"{layer_name}_{neuron_id}"
        if key not in self.membrane_potentials:
            self.membrane_potentials[key] = []
        self.membrane_potentials[key].append((time_step, membrane_potential))

        # Log significant changes
        if len(self.membrane_potentials[key]) > 1:
            prev_potential = self.membrane_potentials[key][-2][1]
            if abs(membrane_potential - prev_potential) > 5.0:  # Significant change
                self.logger.debug(
                    f"MEMBRANE: Neuron {neuron_id} in {layer_name} "
                    f"V: {prev_potential:.2f} -> {membrane_potential:.2f}mV"
                )

    def log_synaptic_weight_change(
        self,
        synapse_id: int,
        pre_neuron_id: int,
        post_neuron_id: int,
        old_weight: float,
        new_weight: float,
        learning_rule: str,
        time_step: float,
    ):
        """Log synaptic weight changes with learning context."""
        weight_change = new_weight - old_weight
        event = SynapticWeightEvent(
            synapse_id=synapse_id,
            pre_neuron_id=pre_neuron_id,
            post_neuron_id=post_neuron_id,
            old_weight=old_weight,
            new_weight=new_weight,
            weight_change=weight_change,
            learning_rule=learning_rule,
            time_step=time_step,
        )

        self.synaptic_events.append(event)

        # Log significant weight changes
        if abs(weight_change) > 0.01:  # Significant change
            self.logger.debug(
                f"SYNAPSE: {synapse_id} ({pre_neuron_id}->{post_neuron_id}) "
                f"w: {old_weight:.4f} -> {new_weight:.4f} "
                f"({weight_change:+.4f}) via {learning_rule}"
            )

    def log_network_state(
        self,
        layer_name: str,
        time_step: float,
        active_neurons: int,
        total_neurons: int,
        firing_rate: float,
        average_membrane_potential: float,
        spike_count: int,
    ):
        """Log network state snapshot."""
        event = NetworkStateEvent(
            time_step=time_step,
            layer_name=layer_name,
            active_neurons=active_neurons,
            total_neurons=total_neurons,
            firing_rate=firing_rate,
            average_membrane_potential=average_membrane_potential,
            spike_count=spike_count,
        )

        self.network_events.append(event)

        # Log network statistics
        self.logger.info(
            f"NETWORK: {layer_name} - Active: {active_neurons}/{total_neurons} "
            f"({active_neurons/total_neurons*100:.1f}%), "
            f"Firing Rate: {firing_rate:.2f}Hz, "
            f"Avg V: {average_membrane_potential:.2f}mV"
        )

    def log_performance_metrics(self, metrics: Dict[str, Any]):
        """Log comprehensive performance metrics."""
        self.performance_metrics.append({"timestamp": time.time(), "metrics": metrics})

        # Log key metrics
        if "simulation_time" in metrics:
            self.logger.info(
                f"PERFORMANCE: Simulation time: {metrics['simulation_time']:.4f}s"
            )
        if "reward" in metrics:
            self.logger.info(f"PERFORMANCE: Reward: {metrics['reward']:.4f}")
        if "accuracy" in metrics:
            self.logger.info(f"PERFORMANCE: Accuracy: {metrics['accuracy']:.4f}")

    def log_task_complexity(
        self,
        task_level: str,
        task_description: str,
        input_noise: float = 0.0,
        missing_modalities: List[str] = None,
    ):
        """Log task complexity and parameters."""
        self.logger.info(f"TASK: Level {task_level} - {task_description}")
        if input_noise > 0:
            self.logger.info(f"TASK: Input noise: {input_noise*100:.1f}%")
        if missing_modalities:
            self.logger.info(f"TASK: Missing modalities: {missing_modalities}")

    def log_sensory_encoding(
        self,
        modality: str,
        input_data: np.ndarray,
        encoded_spikes: np.ndarray,
        encoding_time: float,
    ):
        """Log sensory encoding details."""
        self.logger.debug(
            f"SENSORY: {modality} encoding - "
            f"Input shape: {input_data.shape}, "
            f"Spikes: {np.sum(encoded_spikes)}, "
            f"Time: {encoding_time:.4f}s"
        )

    def log_robustness_test(
        self,
        test_type: str,
        test_parameters: Dict[str, Any],
        performance_before: Dict[str, float],
        performance_after: Dict[str, float],
    ):
        """Log robustness testing results."""
        self.logger.info(f"ROBUSTNESS: {test_type} test")
        self.logger.info(f"ROBUSTNESS: Parameters: {test_parameters}")
        self.logger.info(f"ROBUSTNESS: Performance before: {performance_before}")
        self.logger.info(f"ROBUSTNESS: Performance after: {performance_after}")

        # Calculate degradation
        if "accuracy" in performance_before and "accuracy" in performance_after:
            degradation = performance_before["accuracy"] - performance_after["accuracy"]
            self.logger.info(f"ROBUSTNESS: Accuracy degradation: {degradation:.4f}")

    def save_neural_data(self, filename: str = None):
        """Save all neural data to files for analysis."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"neural_data_{timestamp}"

        data_path = self.data_dir / filename

        # Save spike events
        spike_data = [asdict(event) for event in self.spike_events]
        with open(data_path.with_suffix(".spikes.json"), "w") as f:
            json.dump(spike_data, f, indent=2)

        # Save membrane potential events
        membrane_data = [asdict(event) for event in self.membrane_events]
        with open(data_path.with_suffix(".membrane.json"), "w") as f:
            json.dump(membrane_data, f, indent=2)

        # Save synaptic weight events
        synaptic_data = [asdict(event) for event in self.synaptic_events]
        with open(data_path.with_suffix(".synapses.json"), "w") as f:
            json.dump(synaptic_data, f, indent=2)

        # Save network state events
        network_data = [asdict(event) for event in self.network_events]
        with open(data_path.with_suffix(".network.json"), "w") as f:
            json.dump(network_data, f, indent=2)

        # Save performance metrics
        with open(data_path.with_suffix(".performance.json"), "w") as f:
            json.dump(self.performance_metrics, f, indent=2)

        self.logger.info(f"Neural data saved to {data_path}")

    def generate_analysis_plots(self, save_dir: str = "analysis_plots"):
        """Generate comprehensive analysis plots."""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)

        # Spike raster plot
        self._plot_spike_raster(save_path / "spike_raster.png")

        # Membrane potential evolution
        self._plot_membrane_potentials(save_path / "membrane_potentials.png")

        # Firing rate analysis
        self._plot_firing_rates(save_path / "firing_rates.png")

        # Synaptic weight evolution
        self._plot_synaptic_weights(save_path / "synaptic_weights.png")

        # Network state evolution
        self._plot_network_state(save_path / "network_state.png")

        self.logger.info(f"Analysis plots saved to {save_path}")

    def _plot_spike_raster(self, save_path: Path):
        """Generate spike raster plot."""
        if not self.spike_events:
            return

        plt.figure(figsize=(12, 8))

        # Group spikes by layer
        layers = {}
        for event in self.spike_events:
            if event.layer_name not in layers:
                layers[event.layer_name] = []
            layers[event.layer_name].append((event.neuron_id, event.spike_time))

        colors = ["red", "blue", "green", "orange", "purple"]
        y_offset = 0

        for i, (layer_name, spikes) in enumerate(layers.items()):
            if spikes:
                neuron_ids, spike_times = zip(*spikes)
                plt.scatter(
                    spike_times,
                    [nid + y_offset for nid in neuron_ids],
                    c=colors[i % len(colors)],
                    alpha=0.7,
                    s=10,
                    label=layer_name,
                )
                y_offset += max(neuron_ids) + 10

        plt.xlabel("Time (ms)")
        plt.ylabel("Neuron ID")
        plt.title("Spike Raster Plot")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_membrane_potentials(self, save_path: Path):
        """Generate membrane potential evolution plot."""
        if not self.membrane_events:
            return

        plt.figure(figsize=(12, 8))

        # Sample a few neurons for clarity
        sampled_neurons = {}
        for event in self.membrane_events:
            key = f"{event.layer_name}_{event.neuron_id}"
            if key not in sampled_neurons:
                sampled_neurons[key] = []
            sampled_neurons[key].append((event.time_step, event.membrane_potential))

        # Plot first 10 neurons
        colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(sampled_neurons))))

        for i, (neuron_key, data) in enumerate(list(sampled_neurons.items())[:10]):
            if data:
                times, potentials = zip(*data)
                plt.plot(
                    times,
                    potentials,
                    color=colors[i],
                    alpha=0.7,
                    label=f"Neuron {neuron_key}",
                )

        plt.xlabel("Time (ms)")
        plt.ylabel("Membrane Potential (mV)")
        plt.title("Membrane Potential Evolution")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_firing_rates(self, save_path: Path):
        """Generate firing rate analysis plot."""
        if not self.firing_rates:
            return

        plt.figure(figsize=(12, 8))

        for layer_name, spike_times in self.firing_rates.items():
            if spike_times:
                # Calculate firing rate over time windows
                time_windows = np.arange(0, max(spike_times), 100)  # 100ms windows
                firing_rates = []

                for window_start in time_windows:
                    window_end = window_start + 100
                    spikes_in_window = sum(
                        1 for t in spike_times if window_start <= t < window_end
                    )
                    firing_rate = spikes_in_window / 0.1  # Hz
                    firing_rates.append(firing_rate)

                plt.plot(time_windows, firing_rates, label=layer_name, linewidth=2)

        plt.xlabel("Time (ms)")
        plt.ylabel("Firing Rate (Hz)")
        plt.title("Firing Rate Evolution")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_synaptic_weights(self, save_path: Path):
        """Generate synaptic weight evolution plot."""
        if not self.synaptic_events:
            return

        plt.figure(figsize=(12, 8))

        # Group by synapse
        synapses = {}
        for event in self.synaptic_events:
            key = f"{event.pre_neuron_id}->{event.post_neuron_id}"
            if key not in synapses:
                synapses[key] = []
            synapses[key].append((event.time_step, event.new_weight))

        # Plot first 10 synapses
        colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(synapses))))

        for i, (synapse_key, data) in enumerate(list(synapses.items())[:10]):
            if data:
                times, weights = zip(*data)
                plt.plot(
                    times,
                    weights,
                    color=colors[i],
                    alpha=0.7,
                    label=f"Synapse {synapse_key}",
                )

        plt.xlabel("Time (ms)")
        plt.ylabel("Synaptic Weight")
        plt.title("Synaptic Weight Evolution")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_network_state(self, save_path: Path):
        """Generate network state evolution plot."""
        if not self.network_events:
            return

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

        # Group by layer
        layers = {}
        for event in self.network_events:
            if event.layer_name not in layers:
                layers[event.layer_name] = []
            layers[event.layer_name].append(event)

        colors = plt.cm.tab10(np.linspace(0, 1, len(layers)))

        for i, (layer_name, events) in enumerate(layers.items()):
            times = [e.time_step for e in events]
            active_ratios = [e.active_neurons / e.total_neurons for e in events]
            firing_rates = [e.firing_rate for e in events]
            avg_potentials = [e.average_membrane_potential for e in events]

            ax1.plot(
                times, active_ratios, color=colors[i], label=layer_name, linewidth=2
            )
            ax2.plot(
                times, firing_rates, color=colors[i], label=layer_name, linewidth=2
            )
            ax3.plot(
                times, avg_potentials, color=colors[i], label=layer_name, linewidth=2
            )

        ax1.set_ylabel("Active Neuron Ratio")
        ax1.set_title("Network Activity Evolution")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.set_ylabel("Firing Rate (Hz)")
        ax2.set_title("Network Firing Rate Evolution")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3.set_xlabel("Time (ms)")
        ax3.set_ylabel("Average Membrane Potential (mV)")
        ax3.set_title("Network Membrane Potential Evolution")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get comprehensive summary statistics."""
        summary = {
            "total_spikes": len(self.spike_events),
            "total_synapses_updated": len(self.synaptic_events),
            "total_network_states": len(self.network_events),
            "simulation_duration": self.current_time if self.current_time > 0 else 0,
            "layers": {},
            "performance_metrics": (
                self.performance_metrics[-1] if self.performance_metrics else {}
            ),
        }

        # Layer-specific statistics
        for event in self.network_events:
            layer = event.layer_name
            if layer not in summary["layers"]:
                summary["layers"][layer] = {
                    "total_neurons": event.total_neurons,
                    "max_active_neurons": 0,
                    "avg_firing_rate": 0,
                    "avg_membrane_potential": 0,
                    "state_count": 0,
                }

            layer_stats = summary["layers"][layer]
            layer_stats["max_active_neurons"] = max(
                layer_stats["max_active_neurons"], event.active_neurons
            )
            layer_stats["avg_firing_rate"] += event.firing_rate
            layer_stats["avg_membrane_potential"] += event.average_membrane_potential
            layer_stats["state_count"] += 1

        # Calculate averages
        for layer_stats in summary["layers"].values():
            if layer_stats["state_count"] > 0:
                layer_stats["avg_firing_rate"] /= layer_stats["state_count"]
                layer_stats["avg_membrane_potential"] /= layer_stats["state_count"]

        return summary

    def log_sensory_encoding(
        self,
        modality: str,
        sensory_data: np.ndarray,
        num_spikes: int,
        encoding_time: float,
    ):
        """Log sensory encoding event."""
        self.logger.debug(
            f"SENSORY_ENCODING: {modality} - "
            f"Data shape: {sensory_data.shape}, "
            f"Spikes generated: {num_spikes}, "
            f"Encoding time: {encoding_time:.4f}s"
        )

        # Store for analysis
        encoding_event = {
            "modality": modality,
            "data_shape": sensory_data.shape,
            "num_spikes": num_spikes,
            "encoding_time": encoding_time,
            "timestamp": self.current_time,
        }
        # You could store this in a list for later analysis

    def log_task_complexity(
        self,
        level: str,
        description: str,
        noise_level: float,
        missing_modalities: List[str],
    ):
        """Log task complexity information."""
        self.logger.info(
            f"TASK_COMPLEXITY: Level={level}, "
            f"Description={description}, "
            f"Noise={noise_level:.2f}, "
            f"Missing modalities={missing_modalities}"
        )

        # Store for analysis
        task_event = {
            "level": level,
            "description": description,
            "noise_level": noise_level,
            "missing_modalities": missing_modalities,
            "timestamp": self.current_time,
        }
        # You could store this in a list for later analysis


# Global logger instance
enhanced_logger = EnhancedNeuromorphicLogger()
