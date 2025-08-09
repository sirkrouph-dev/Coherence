#!/usr/bin/env python3
"""
STDP Learning Example with Pre/Post Spike Protocols
====================================================

This example demonstrates:
1. Setting up a simple two-neuron system with STDP
2. Creating controlled pre/post spike timing protocols
3. Measuring weight changes based on spike timing
4. Visualizing the STDP learning window
5. Testing different plasticity rules

Key concepts illustrated:
- Spike-Timing-Dependent Plasticity (STDP)
- Long-Term Potentiation (LTP) and Depression (LTD)
- Hebbian learning principles
- Reward-modulated STDP
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.learning import (
    STDPRule, HebbianRule, RewardModulatedSTDP, 
    PlasticityConfig, PlasticityManager
)


def create_spike_protocol(timing_differences: List[float], 
                         base_time: float = 50.0) -> Tuple[List[float], List[float]]:
    """
    Create pre and post spike times with specific timing differences.
    
    Args:
        timing_differences: List of time differences (post_time - pre_time)
        base_time: Base time for first pre spike
        
    Returns:
        Tuple of (pre_spike_times, post_spike_times)
    """
    pre_spikes = []
    post_spikes = []
    
    current_time = base_time
    
    for dt in timing_differences:
        pre_spikes.append(current_time)
        post_spikes.append(current_time + dt)
        current_time += 100.0  # Space out spike pairs
        
    return pre_spikes, post_spikes


def simulate_stdp_protocol(pre_spikes: List[float], 
                          post_spikes: List[float],
                          config: PlasticityConfig = None) -> Dict[str, Any]:
    """
    Simulate STDP with given spike protocol.
    
    Args:
        pre_spikes: Presynaptic spike times
        post_spikes: Postsynaptic spike times
        config: Plasticity configuration
        
    Returns:
        Dictionary with results
    """
    if config is None:
        config = PlasticityConfig(
            learning_rate=0.01,
            tau_plus=20.0,
            tau_minus=20.0,
            A_plus=0.01,
            A_minus=0.01,
            weight_min=0.0,
            weight_max=1.0
        )
    
    # Initialize STDP rule
    stdp = STDPRule(config)
    
    # Initial weight
    weight = 0.5
    weight_history = [weight]
    
    # Simulation parameters
    dt = 0.1  # ms
    total_time = max(max(pre_spikes, default=0), max(post_spikes, default=0)) + 50.0
    
    # Convert spike times to indices
    pre_spike_indices = [int(t / dt) for t in pre_spikes]
    post_spike_indices = [int(t / dt) for t in post_spikes]
    
    # Run simulation
    for t_idx in range(int(total_time / dt)):
        # Check for spikes
        pre_spike = t_idx in pre_spike_indices
        post_spike = t_idx in post_spike_indices
        
        # Update weight
        weight = stdp.update_weight(
            weight,
            pre_activity=1.0 if pre_spike else 0.0,
            post_activity=1.0 if post_spike else 0.0,
            dt=dt,
            pre_spike=pre_spike,
            post_spike=post_spike
        )
        
        weight_history.append(weight)
    
    return {
        'final_weight': weight,
        'weight_history': weight_history,
        'initial_weight': 0.5,
        'weight_change': weight - 0.5,
        'pre_spikes': pre_spikes,
        'post_spikes': post_spikes
    }


def measure_stdp_window(time_differences: np.ndarray,
                       config: PlasticityConfig = None) -> np.ndarray:
    """
    Measure the STDP learning window.
    
    Args:
        time_differences: Array of timing differences to test
        config: Plasticity configuration
        
    Returns:
        Array of weight changes
    """
    weight_changes = []
    
    for dt in time_differences:
        # Create single spike pair
        if dt >= 0:
            pre_spikes = [50.0]
            post_spikes = [50.0 + dt]
        else:
            pre_spikes = [50.0 - dt]
            post_spikes = [50.0]
        
        # Run simulation
        results = simulate_stdp_protocol(pre_spikes, post_spikes, config)
        weight_changes.append(results['weight_change'])
    
    return np.array(weight_changes)


def test_hebbian_learning():
    """
    Test Hebbian learning with correlated activity.
    
    Returns:
        Dictionary with results
    """
    print("\nTesting Hebbian Learning...")
    print("-" * 40)
    
    config = PlasticityConfig(
        learning_rate=0.01,
        hebbian_threshold=0.3,
        hebbian_decay=0.99,
        weight_min=0.0,
        weight_max=1.0
    )
    
    hebbian = HebbianRule(config)
    
    # Test different correlation levels
    correlations = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    weight_changes = []
    
    initial_weight = 0.5
    
    for correlation in correlations:
        weight = initial_weight
        
        # Simulate 100 time steps
        for _ in range(100):
            pre_activity = np.random.random()
            # Correlated post activity
            post_activity = correlation * pre_activity + (1 - correlation) * np.random.random()
            
            weight = hebbian.update_weight(
                weight,
                pre_activity,
                post_activity
            )
        
        weight_changes.append(weight - initial_weight)
        print(f"  Correlation={correlation:.1f}: Δw={weight-initial_weight:+.4f}")
    
    return {
        'correlations': correlations,
        'weight_changes': weight_changes
    }


def test_reward_modulated_stdp():
    """
    Test reward-modulated STDP.
    
    Returns:
        Dictionary with results
    """
    print("\nTesting Reward-Modulated STDP...")
    print("-" * 40)
    
    config = PlasticityConfig(
        learning_rate=0.01,
        tau_plus=20.0,
        tau_minus=20.0,
        A_plus=0.01,
        A_minus=0.01,
        reward_decay=0.9,
        reward_sensitivity=2.0,
        dopamine_time_constant=200.0
    )
    
    rstdp = RewardModulatedSTDP(config)
    
    # Test with different reward schedules
    scenarios = [
        ("No reward", 0.0),
        ("Small reward", 0.5),
        ("Large reward", 1.0),
        ("Punishment", -0.5)
    ]
    
    results = {}
    
    for scenario_name, reward in scenarios:
        # Set reward
        rstdp.set_reward(reward)
        
        # Create spike protocol (pre before post - normally LTP)
        pre_spikes = [50.0, 150.0, 250.0]
        post_spikes = [60.0, 160.0, 260.0]  # 10ms after pre
        
        weight = 0.5
        weight_history = [weight]
        
        # Simulate
        dt = 1.0
        for t in range(300):
            pre_spike = any(abs(t - spike_t) < dt for spike_t in pre_spikes)
            post_spike = any(abs(t - spike_t) < dt for spike_t in post_spikes)
            
            weight = rstdp.update_weight(
                weight,
                pre_activity=1.0 if pre_spike else 0.0,
                post_activity=1.0 if post_spike else 0.0,
                dt=dt,
                pre_spike=pre_spike,
                post_spike=post_spike
            )
            
            weight_history.append(weight)
        
        results[scenario_name] = {
            'reward': reward,
            'final_weight': weight,
            'weight_change': weight - 0.5,
            'weight_history': weight_history
        }
        
        print(f"  {scenario_name} (r={reward:+.1f}): Δw={weight-0.5:+.4f}")
    
    return results


def visualize_stdp_window(time_differences: np.ndarray,
                         weight_changes: np.ndarray):
    """
    Visualize the STDP learning window.
    
    Args:
        time_differences: Timing differences
        weight_changes: Corresponding weight changes
    """
    plt.figure(figsize=(10, 6))
    
    # Separate LTP and LTD
    ltp_mask = time_differences > 0
    ltd_mask = time_differences < 0
    
    # Plot LTP (post after pre)
    plt.plot(time_differences[ltp_mask], weight_changes[ltp_mask], 
             'r-', linewidth=2, label='LTP (Post after Pre)')
    
    # Plot LTD (pre after post)
    plt.plot(time_differences[ltd_mask], weight_changes[ltd_mask],
             'b-', linewidth=2, label='LTD (Pre after Post)')
    
    # Add zero lines
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    # Labels and formatting
    plt.xlabel('Spike Timing Difference Δt = t_post - t_pre (ms)', fontsize=12)
    plt.ylabel('Weight Change Δw', fontsize=12)
    plt.title('STDP Learning Window', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Add annotations
    plt.text(20, max(weight_changes) * 0.8, 'Potentiation\n(Causal)', 
             ha='center', fontsize=10, color='red')
    plt.text(-20, min(weight_changes) * 0.8, 'Depression\n(Anti-causal)', 
             ha='center', fontsize=10, color='blue')
    
    plt.tight_layout()
    plt.show()


def visualize_spike_protocol(results: Dict[str, Any], title: str = "Spike Protocol"):
    """
    Visualize a spike protocol and weight evolution.
    
    Args:
        results: Results from simulate_stdp_protocol
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot spikes
    pre_spikes = results['pre_spikes']
    post_spikes = results['post_spikes']
    
    # Pre spikes
    for spike_time in pre_spikes:
        ax1.axvline(spike_time, ymin=0.1, ymax=0.4, color='blue', linewidth=2)
    ax1.text(10, 0.25, 'Pre', fontsize=10, color='blue')
    
    # Post spikes
    for spike_time in post_spikes:
        ax1.axvline(spike_time, ymin=0.6, ymax=0.9, color='red', linewidth=2)
    ax1.text(10, 0.75, 'Post', fontsize=10, color='red')
    
    # Timing differences
    for pre, post in zip(pre_spikes, post_spikes):
        dt = post - pre
        mid_time = (pre + post) / 2
        ax1.annotate(f'Δt={dt:.1f}ms', xy=(mid_time, 0.5), 
                    ha='center', fontsize=8)
        ax1.plot([pre, post], [0.4, 0.6], 'k--', alpha=0.3)
    
    ax1.set_ylim(0, 1)
    ax1.set_ylabel('Neuron', fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot weight evolution
    weight_history = results['weight_history']
    time_points = np.arange(len(weight_history)) * 0.1  # dt = 0.1ms
    
    ax2.plot(time_points, weight_history, 'g-', linewidth=2)
    ax2.axhline(y=results['initial_weight'], color='k', linestyle='--', 
                alpha=0.3, label='Initial weight')
    ax2.fill_between(time_points, results['initial_weight'], weight_history,
                     alpha=0.3, color='green')
    
    # Mark spike times
    for spike_time in pre_spikes:
        ax2.axvline(spike_time, color='blue', alpha=0.2, linestyle=':')
    for spike_time in post_spikes:
        ax2.axvline(spike_time, color='red', alpha=0.2, linestyle=':')
    
    ax2.set_xlabel('Time (ms)', fontsize=12)
    ax2.set_ylabel('Synaptic Weight', fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # Add final weight change annotation
    final_change = results['weight_change']
    ax2.text(0.98, 0.95, f'Δw = {final_change:+.4f}',
            transform=ax2.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()


def run_comprehensive_stdp_test():
    """
    Run a comprehensive test of STDP with various timing protocols.
    """
    print("=" * 60)
    print("Comprehensive STDP Learning Test")
    print("=" * 60)
    
    # Test 1: Classic STDP window
    print("\n1. Measuring STDP Learning Window...")
    print("-" * 40)
    
    time_differences = np.linspace(-50, 50, 101)
    weight_changes = measure_stdp_window(time_differences)
    
    print(f"  Max LTP: {max(weight_changes):+.4f} at Δt={time_differences[np.argmax(weight_changes)]:.1f}ms")
    print(f"  Max LTD: {min(weight_changes):+.4f} at Δt={time_differences[np.argmin(weight_changes)]:.1f}ms")
    
    visualize_stdp_window(time_differences, weight_changes)
    
    # Test 2: Specific spike protocols
    print("\n2. Testing Specific Spike Protocols...")
    print("-" * 40)
    
    protocols = [
        ("Strong LTP", [10, 10, 10]),      # Repeated causal pairing
        ("Strong LTD", [-10, -10, -10]),   # Repeated anti-causal pairing
        ("Mixed", [10, -10, 20, -20]),     # Mixed timing
        ("Neutral", [0, 0, 0])             # Simultaneous spikes
    ]
    
    for protocol_name, timing_diffs in protocols:
        print(f"\n  Protocol: {protocol_name}")
        pre_spikes, post_spikes = create_spike_protocol(timing_diffs)
        results = simulate_stdp_protocol(pre_spikes, post_spikes)
        print(f"    Initial weight: {results['initial_weight']:.3f}")
        print(f"    Final weight: {results['final_weight']:.3f}")
        print(f"    Change: {results['weight_change']:+.4f}")
        
        visualize_spike_protocol(results, f"STDP Protocol: {protocol_name}")
    
    return {
        'time_differences': time_differences,
        'weight_changes': weight_changes,
        'protocols': protocols
    }


def test_plasticity_manager():
    """
    Test the PlasticityManager with multiple rules.
    """
    print("\n" + "=" * 60)
    print("Testing Plasticity Manager")
    print("=" * 60)
    
    # Create manager
    manager = PlasticityManager()
    
    # Test different rules
    print("\nAvailable plasticity rules:")
    for rule_name in manager.rules.keys():
        print(f"  - {rule_name}")
    
    # Activate specific rules
    manager.activate_rule('stdp')
    manager.activate_rule('homeostatic')
    
    print(f"\nActive rules: {manager.active_rules}")
    
    # Create a small weight matrix
    weights = np.random.uniform(0.3, 0.7, (5, 5))
    pre_activity = np.random.random(5)
    post_activity = np.random.random(5)
    
    print("\nInitial weights:")
    print(weights)
    
    # Update weights
    updated_weights = manager.update_weights(
        weights, pre_activity, post_activity
    )
    
    print("\nUpdated weights:")
    print(updated_weights)
    
    print("\nWeight changes:")
    print(updated_weights - weights)
    
    # Get statistics
    stats = manager.get_statistics()
    print(f"\nManager statistics:")
    print(f"  Active rules: {stats['active_rules']}")
    print(f"  Learning rate: {stats['config']['learning_rate']}")


def main():
    """Main function to run all examples."""
    
    # Run comprehensive STDP test
    stdp_results = run_comprehensive_stdp_test()
    
    # Test Hebbian learning
    hebbian_results = test_hebbian_learning()
    
    # Visualize Hebbian results
    plt.figure(figsize=(10, 6))
    plt.plot(hebbian_results['correlations'], 
             hebbian_results['weight_changes'], 
             'o-', linewidth=2, markersize=8)
    plt.xlabel('Pre-Post Correlation', fontsize=12)
    plt.ylabel('Weight Change', fontsize=12)
    plt.title('Hebbian Learning: Weight Change vs Correlation', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Test reward-modulated STDP
    rstdp_results = test_reward_modulated_stdp()
    
    # Visualize reward modulation
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, (scenario_name, results) in enumerate(rstdp_results.items()):
        ax = axes[idx]
        time_points = np.arange(len(results['weight_history']))
        ax.plot(time_points, results['weight_history'], linewidth=2)
        ax.set_title(f'{scenario_name} (r={results["reward"]:+.1f})', fontsize=12)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Weight')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.5, color='k', linestyle='--', alpha=0.3)
        
        # Add weight change annotation
        ax.text(0.98, 0.95, f'Δw = {results["weight_change"]:+.4f}',
               transform=ax.transAxes, ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Reward-Modulated STDP', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Test plasticity manager
    test_plasticity_manager()
    
    print("\n" + "=" * 60)
    print("All STDP learning examples completed!")
    print("=" * 60)
    
    return {
        'stdp': stdp_results,
        'hebbian': hebbian_results,
        'rstdp': rstdp_results
    }


if __name__ == "__main__":
    all_results = main()
    
    print("\n\nKey Takeaways:")
    print("-" * 40)
    print("1. STDP implements temporal causality in learning")
    print("2. Pre→Post spike order leads to LTP (strengthening)")
    print("3. Post→Pre spike order leads to LTD (weakening)")
    print("4. Hebbian learning strengthens correlated activity")
    print("5. Reward signals can modulate plasticity")
    print("\nExperiment with different parameters to explore learning dynamics!")
