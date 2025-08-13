"""
Comprehensive Enhanced Neuromorphic System Demo.
Demonstrates all enhanced capabilities including dynamic logging, task complexity,
sensory encoding, and robustness testing.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from core.enhanced_encoding import enhanced_encoder
# Import enhanced components
from core.enhanced_logging import enhanced_logger
from core.network import NeuromorphicNetwork
from core.neuromodulation import NeuromodulatorType, NeuromodulatorySystem
from core.neurons import NeuronFactory, NeuronPopulation
from core.robustness_testing import TestType, robustness_tester
from core.synapses import STDP_Synapse, SynapseFactory
from core.task_complexity import TaskLevel, TaskParameters, task_manager


class EnhancedNeuromorphicDemo:
    """Comprehensive demo of enhanced neuromorphic system capabilities."""

    def __init__(self):
        """Initialize the enhanced demo system."""
        self.network = None
        self.neuromodulation = None
        self.training_history = []
        self.robustness_results = []
        self.analysis_results = {}

        # Setup directories
        self.setup_directories()

    def setup_directories(self):
        """Setup directories for data and analysis."""
        Path("enhanced_data").mkdir(exist_ok=True)
        Path("enhanced_analysis").mkdir(exist_ok=True)
        Path("enhanced_plots").mkdir(exist_ok=True)

    def create_enhanced_network(self) -> NeuromorphicNetwork:
        """Create an enhanced neuromorphic network."""
        print("🔧 Creating Enhanced Neuromorphic Network...")

        # Define layer sizes
        input_size = 30
        hidden_size = 20
        output_size = 10

        # Create network
        network = NeuromorphicNetwork()

        # Add layers with enhanced parameters
        network.add_layer(
            "input",
            input_size,
            "lif",
            tau_m=15.0,
            v_thresh=-55.0,
            refractory_period=2.0,
        )
        network.add_layer(
            "hidden",
            hidden_size,
            "adex",
            tau_m=20.0,
            v_thresh=-55.0,
            delta_t=2.0,
            tau_w=144.0,
            a=4.0,
            b=0.0805,
        )
        network.add_layer(
            "output",
            output_size,
            "lif",
            tau_m=15.0,
            v_thresh=-55.0,
            refractory_period=2.0,
        )

        # Connect layers with STDP synapses
        network.connect_layers(
            "input",
            "hidden",
            synapse_type="stdp",
            connection_probability=0.3,
            A_plus=0.01,
            A_minus=0.01,
        )
        network.connect_layers(
            "hidden",
            "output",
            synapse_type="stdp",
            connection_probability=0.3,
            A_plus=0.01,
            A_minus=0.01,
        )

        # Initialize neuromodulation system
        self.neuromodulation = NeuromodulatorySystem(NeuromodulatorType.DOPAMINE)

        print(f"✅ Enhanced Network Created:")
        print(f"   - Input neurons: {input_size}")
        print(f"   - Hidden neurons: {hidden_size}")
        print(f"   - Output neurons: {output_size}")
        print(f"   - Total connections: {len(network.connections)}")

        return network

    def run_progressive_task_complexity_demo(self):
        """Demonstrate progressive task complexity."""
        print("\n🎯 Running Progressive Task Complexity Demo...")

        self.network = self.create_enhanced_network()

        # Test all task complexity levels
        for level in TaskLevel:
            print(f"\n📊 Testing {level.value}...")

            # Create task parameters
            params = TaskParameters(
                level=level,
                input_noise=0.1 if level.value != "simple_binary" else 0.0,
                missing_modalities=(
                    [] if level.value != "multi_modal_fusion" else ["auditory"]
                ),
                temporal_complexity=0.2 if level.value == "temporal_sequence" else 0.0,
                adversarial_strength=(
                    0.1 if level.value == "adversarial_robustness" else 0.0
                ),
                dynamic_changes=level.value == "adaptive_learning",
                sequence_length=5 if level.value == "temporal_sequence" else 1,
            )

            # Create and execute task
            task = task_manager.create_task(level, params)
            performance = self.execute_task(task)

            # Log results
            enhanced_logger.log_performance_metrics(
                {
                    "task_level": level.value,
                    "reward": performance["reward"],
                    "accuracy": performance["accuracy"],
                    "latency": performance["latency"],
                }
            )

            print(
                f"   ✅ {level.value}: Reward={performance['reward']:.4f}, "
                f"Accuracy={performance['accuracy']:.4f}"
            )

    def run_enhanced_sensory_encoding_demo(self):
        """Demonstrate enhanced sensory encoding capabilities."""
        print("\n👁️ Running Enhanced Sensory Encoding Demo...")

        # Generate synthetic sensory data
        visual_data = np.random.random((32, 32))
        auditory_data = np.random.random(4410)  # 0.1 seconds at 44.1kHz
        tactile_data = np.random.random((8, 8))

        sensory_inputs = {
            "visual": visual_data,
            "auditory": auditory_data,
            "tactile": tactile_data,
        }

        # Encode sensory inputs
        print("🔄 Encoding sensory inputs...")
        encoding_result = enhanced_encoder.encode_sensory_inputs(sensory_inputs)

        print(f"✅ Encoding completed:")
        print(f"   - Modalities encoded: {encoding_result['modalities_encoded']}")
        print(
            f"   - Total encoding time: {encoding_result['total_encoding_time']:.4f}s"
        )
        print(
            f"   - Fusion quality: {encoding_result['fused_result']['fusion_quality']:.4f}"
        )

        # Analyze encoding statistics
        encoding_stats = enhanced_encoder.get_encoding_statistics()
        print(f"📊 Encoding Statistics:")
        print(f"   - Visual features: {encoding_stats['visual_features']}")
        print(f"   - Auditory bands: {encoding_stats['auditory_bands']}")
        print(f"   - Tactile sensors: {encoding_stats['tactile_sensors']}")

    def run_comprehensive_robustness_testing(self):
        """Run comprehensive robustness testing."""
        print("\n🛡️ Running Comprehensive Robustness Testing...")

        # Create test inputs
        test_inputs = {
            "visual": np.random.random((32, 32)),
            "auditory": np.random.random(1000),
            "tactile": np.random.random((8, 8)),
        }

        # Establish baseline performance
        print("📈 Establishing baseline performance...")
        baseline_performance = self._evaluate_baseline_performance(test_inputs)

        print(f"✅ Baseline Performance:")
        print(f"   - Accuracy: {baseline_performance['accuracy']:.4f}")
        print(f"   - Latency: {baseline_performance['latency']:.4f}s")
        print(f"   - Throughput: {baseline_performance['throughput']:.1f} ops/s")

        # Run comprehensive robustness tests
        print("🧪 Running robustness tests...")
        test_results = robustness_tester.run_comprehensive_test_suite(
            self.network, test_inputs, baseline_performance
        )

        # Analyze results
        self.robustness_results = test_results
        robustness_summary = robustness_tester.get_robustness_summary()

        print(f"📊 Robustness Testing Summary:")
        print(f"   - Total tests: {robustness_summary['total_tests']}")
        print(
            f"   - Average robustness score: {robustness_summary['average_robustness_score']:.4f}"
        )
        print(
            f"   - Worst case scenarios: {robustness_summary['worst_case_scenarios']}"
        )
        print(f"   - Best case scenarios: {robustness_summary['best_case_scenarios']}")

        # Generate recommendations
        if robustness_summary["recommendations"]:
            print("💡 Recommendations:")
            for rec in robustness_summary["recommendations"]:
                print(f"   - {rec}")

    def run_dynamic_logging_demo(self):
        """Demonstrate dynamic logging capabilities."""
        print("\n📝 Running Dynamic Logging Demo...")

        # Simulate neural activity with dynamic logging
        print("🧠 Simulating neural activity with enhanced logging...")

        for trial in range(5):
            print(f"   Trial {trial + 1}/5...")

            # Simulate spike events
            for neuron_id in range(10):
                if np.random.random() < 0.3:  # 30% chance of spike
                    spike_time = trial * 100 + np.random.random() * 100
                    membrane_potential = -55.0 + np.random.random() * 20

                    enhanced_logger.log_spike_event(
                        neuron_id=neuron_id,
                        layer_name="hidden",
                        spike_time=spike_time,
                        membrane_potential=membrane_potential,
                        synaptic_inputs={"excitatory": np.random.random()},
                        neuromodulator_levels={"dopamine": np.random.random()},
                    )

            # Simulate membrane potential changes
            for neuron_id in range(10):
                enhanced_logger.log_membrane_potential(
                    neuron_id=neuron_id,
                    layer_name="hidden",
                    time_step=trial * 100,
                    membrane_potential=-65.0 + np.random.random() * 30,
                    synaptic_current=np.random.random() * 0.1,
                    adaptation_current=np.random.random() * 0.05,
                )

            # Simulate synaptic weight changes
            for synapse_id in range(5):
                enhanced_logger.log_synaptic_weight_change(
                    synapse_id=synapse_id,
                    pre_neuron_id=np.random.randint(0, 10),
                    post_neuron_id=np.random.randint(0, 10),
                    old_weight=np.random.random(),
                    new_weight=np.random.random(),
                    learning_rule="stdp",
                    time_step=trial * 100,
                )

            # Log network state
            enhanced_logger.log_network_state(
                layer_name="hidden",
                time_step=trial * 100,
                active_neurons=np.random.randint(5, 15),
                total_neurons=20,
                firing_rate=np.random.random() * 50,
                average_membrane_potential=-60.0 + np.random.random() * 20,
                spike_count=np.random.randint(5, 15),
            )

        print("✅ Dynamic logging completed!")

    def run_comprehensive_analysis(self):
        """Run comprehensive analysis of all results."""
        print("\n📊 Running Comprehensive Analysis...")

        # Save neural data
        enhanced_logger.save_neural_data("comprehensive_demo")

        # Generate analysis plots
        enhanced_logger.generate_analysis_plots("enhanced_analysis")

        # Get summary statistics
        summary_stats = enhanced_logger.get_summary_statistics()

        print("📈 Analysis Results:")
        print(f"   - Total spikes: {summary_stats['total_spikes']}")
        print(f"   - Synapses updated: {summary_stats['total_synapses_updated']}")
        print(f"   - Network states: {summary_stats['total_network_states']}")
        print(f"   - Simulation duration: {summary_stats['simulation_duration']:.2f}ms")

        # Layer-specific analysis
        if summary_stats["layers"]:
            print("   - Layer Analysis:")
            for layer_name, layer_stats in summary_stats["layers"].items():
                print(
                    f"     {layer_name}: {layer_stats['total_neurons']} neurons, "
                    f"avg firing rate: {layer_stats['avg_firing_rate']:.2f}Hz"
                )

        # Save comprehensive report
        self.save_comprehensive_report(summary_stats)

    def execute_task(self, task: Dict[str, Any]) -> Dict[str, float]:
        """Execute a task and return performance metrics."""
        # Simulate task execution
        inputs = task["inputs"]
        expected_output = task["expected_output"]

        # Simulate network processing
        processing_time = np.random.uniform(0.01, 0.1)
        time.sleep(processing_time)

        # Simulate performance metrics
        accuracy = np.random.uniform(0.7, 1.0)
        if task["level"] in [TaskLevel.LEVEL_6, TaskLevel.LEVEL_7]:
            accuracy *= 0.8  # Reduced accuracy for complex tasks

        reward = accuracy
        latency = processing_time

        return {
            "accuracy": accuracy,
            "reward": reward,
            "latency": latency,
            "throughput": 1.0 / latency,
        }

    def _evaluate_baseline_performance(
        self, test_inputs: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Evaluate baseline performance on test inputs."""
        # Simulate baseline performance evaluation
        return {
            "accuracy": np.random.uniform(0.8, 0.95),
            "latency": np.random.uniform(0.01, 0.05),
            "throughput": np.random.uniform(50, 100),
            "reward": np.random.uniform(0.8, 0.95),
        }

    def save_comprehensive_report(self, summary_stats: Dict[str, Any]):
        """Save comprehensive analysis report."""
        report = {
            "timestamp": time.time(),
            "summary_statistics": summary_stats,
            "robustness_results": [
                result.__dict__ for result in self.robustness_results
            ],
            "task_complexity_stats": task_manager.get_task_statistics(),
            "encoding_statistics": enhanced_encoder.get_encoding_statistics(),
            "system_metadata": {
                "enhanced_logging": True,
                "task_complexity": True,
                "sensory_encoding": True,
                "robustness_testing": True,
                "dynamic_analysis": True,
            },
        }

        report_path = Path("enhanced_data/comprehensive_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        print(f"📄 Comprehensive report saved to {report_path}")

    def run_full_demo(self):
        """Run the complete enhanced neuromorphic system demo."""
        print("🚀 Starting Enhanced Neuromorphic System Demo")
        print("=" * 60)

        try:
            # 1. Progressive Task Complexity Demo
            self.run_progressive_task_complexity_demo()

            # 2. Enhanced Sensory Encoding Demo
            self.run_enhanced_sensory_encoding_demo()

            # 3. Comprehensive Robustness Testing
            self.run_comprehensive_robustness_testing()

            # 4. Dynamic Logging Demo
            self.run_dynamic_logging_demo()

            # 5. Comprehensive Analysis
            self.run_comprehensive_analysis()

            print("\n" + "=" * 60)
            print("✅ Enhanced Neuromorphic System Demo Completed Successfully!")
            print("\n📁 Generated Files:")
            print("   - enhanced_data/comprehensive_report.json")
            print("   - enhanced_analysis/ (analysis plots)")
            print("   - neural_data/ (neural activity data)")
            print("   - enhanced_trace.log (detailed logging)")

        except Exception as e:
            print(f"❌ Error during demo: {str(e)}")
            enhanced_logger.logger.error(f"Demo error: {str(e)}")

    def generate_demo_visualizations(self):
        """Generate comprehensive visualizations for the demo."""
        print("\n🎨 Generating Demo Visualizations...")

        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Enhanced Neuromorphic System Demo Results", fontsize=16)

        # 1. Task Complexity Performance
        task_levels = [level.value for level in TaskLevel]
        task_performance = [0.85, 0.82, 0.78, 0.75, 0.72, 0.68, 0.65]  # Simulated data

        axes[0, 0].bar(task_levels, task_performance, color="skyblue", alpha=0.7)
        axes[0, 0].set_title("Task Complexity Performance")
        axes[0, 0].set_ylabel("Performance Score")
        axes[0, 0].tick_params(axis="x", rotation=45)

        # 2. Robustness Testing Results
        test_types = [
            "Noise",
            "Missing Modality",
            "Adversarial",
            "Temporal",
            "Network Damage",
        ]
        robustness_scores = [0.85, 0.78, 0.72, 0.80, 0.75]  # Simulated data

        axes[0, 1].bar(test_types, robustness_scores, color="lightgreen", alpha=0.7)
        axes[0, 1].set_title("Robustness Testing Results")
        axes[0, 1].set_ylabel("Robustness Score")
        axes[0, 1].tick_params(axis="x", rotation=45)

        # 3. Sensory Encoding Quality
        modalities = ["Visual", "Auditory", "Tactile"]
        encoding_quality = [0.92, 0.88, 0.85]  # Simulated data

        axes[0, 2].pie(
            encoding_quality,
            labels=modalities,
            autopct="%1.1f%%",
            colors=["lightcoral", "lightblue", "lightyellow"],
        )
        axes[0, 2].set_title("Sensory Encoding Quality")

        # 4. Neural Activity Timeline
        time_points = np.linspace(0, 500, 50)
        spike_counts = np.random.poisson(5, 50)  # Simulated spike data

        axes[1, 0].plot(time_points, spike_counts, "r-", linewidth=2)
        axes[1, 0].set_title("Neural Activity Timeline")
        axes[1, 0].set_xlabel("Time (ms)")
        axes[1, 0].set_ylabel("Spike Count")

        # 5. Synaptic Weight Evolution
        weight_time = np.linspace(0, 100, 20)
        weight_values = np.random.normal(0.5, 0.1, 20)  # Simulated weight data

        axes[1, 1].plot(weight_time, weight_values, "b-", linewidth=2)
        axes[1, 1].set_title("Synaptic Weight Evolution")
        axes[1, 1].set_xlabel("Time (ms)")
        axes[1, 1].set_ylabel("Weight Value")

        # 6. System Performance Metrics
        metrics = ["Accuracy", "Latency", "Throughput", "Robustness"]
        values = [0.85, 0.03, 85.0, 0.78]  # Simulated metrics

        axes[1, 2].bar(
            metrics, values, color=["gold", "silver", "bronze", "purple"], alpha=0.7
        )
        axes[1, 2].set_title("System Performance Metrics")
        axes[1, 2].set_ylabel("Score")
        axes[1, 2].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(
            "enhanced_plots/comprehensive_demo_results.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print(
            "✅ Demo visualizations saved to enhanced_plots/comprehensive_demo_results.png"
        )


def main():
    """Main function to run the enhanced neuromorphic system demo."""
    print("🧠 Enhanced Neuromorphic Computing System")
    print("=" * 50)

    # Create and run demo
    demo = EnhancedNeuromorphicDemo()

    # Run full demo
    demo.run_full_demo()

    # Generate visualizations
    demo.generate_demo_visualizations()

    print(
        "\n🎉 Demo completed! Check the generated files and plots for detailed results."
    )


if __name__ == "__main__":
    main()
