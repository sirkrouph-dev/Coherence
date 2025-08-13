"""
Test and demonstration of learning and plasticity mechanisms.
Shows STDP, Hebbian, reward-modulated learning, and custom plasticity rules.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.learning import (
    PlasticityManager, PlasticityConfig, PlasticityType,
    STDPRule, HebbianRule, BCMRule, RewardModulatedSTDP,
    TripletSTDP, HomeostaticPlasticity, CustomPlasticityRule
)


def test_stdp_learning():
    """Test STDP learning with different spike timing patterns."""
    print("\n=== Testing STDP Learning ===")
    
    # Create configuration
    config = PlasticityConfig(
        learning_rate=0.01,
        tau_plus=20.0,
        tau_minus=20.0,
        A_plus=0.01,
        A_minus=0.012,
        weight_min=0.0,
        weight_max=5.0
    )
    
    # Initialize manager and activate STDP
    manager = PlasticityManager(config)
    manager.activate_rule('stdp')
    
    # Create weight matrix (10 pre x 10 post neurons)
    weights = np.random.uniform(1.0, 2.0, (10, 10))
    initial_weights = weights.copy()
    
    # Simulate spike patterns
    n_timesteps = 100
    dt = 1.0  # ms
    
    weight_history = []
    
    for t in range(n_timesteps):
        # Generate spike patterns
        pre_spikes = np.random.random(10) < 0.05  # 5% spike probability
        post_spikes = np.random.random(10) < 0.05
        
        # Convert to activity (simplified)
        pre_activity = pre_spikes.astype(float)
        post_activity = post_spikes.astype(float)
        
        # Update weights
        weights = manager.update_weights(
            weights, pre_activity, post_activity,
            dt=dt, pre_spike=pre_spikes, post_spike=post_spikes
        )
        
        weight_history.append(weights[0, 0])  # Track one synapse
    
    print(f"Initial weight[0,0]: {initial_weights[0, 0]:.4f}")
    print(f"Final weight[0,0]: {weights[0, 0]:.4f}")
    print(f"Mean weight change: {np.mean(weights - initial_weights):.4f}")
    
    # Plot weight evolution
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(weight_history)
    plt.xlabel('Time (ms)')
    plt.ylabel('Weight')
    plt.title('STDP: Single Synapse Weight Evolution')
    
    plt.subplot(1, 2, 2)
    plt.hist((weights - initial_weights).flatten(), bins=30)
    plt.xlabel('Weight Change')
    plt.ylabel('Count')
    plt.title('STDP: Weight Change Distribution')
    plt.tight_layout()
    plt.show()


def test_hebbian_learning():
    """Test Hebbian learning with correlated activity."""
    print("\n=== Testing Hebbian Learning ===")
    
    config = PlasticityConfig(
        learning_rate=0.01,
        hebbian_threshold=0.3,
        hebbian_decay=0.99,
        weight_min=0.0,
        weight_max=5.0
    )
    
    manager = PlasticityManager(config)
    manager.activate_rule('hebbian')
    
    weights = np.ones((5, 5)) * 2.0
    initial_weights = weights.copy()
    
    # Create correlated activity patterns
    n_patterns = 50
    for pattern in range(n_patterns):
        # Create correlated pre and post activity
        base_pattern = np.random.random(5)
        pre_activity = base_pattern + np.random.normal(0, 0.1, 5)
        post_activity = base_pattern + np.random.normal(0, 0.1, 5)
        
        # Normalize to [0, 1]
        pre_activity = np.clip(pre_activity, 0, 1)
        post_activity = np.clip(post_activity, 0, 1)
        
        weights = manager.update_weights(weights, pre_activity, post_activity)
    
    print(f"Initial mean weight: {np.mean(initial_weights):.4f}")
    print(f"Final mean weight: {np.mean(weights):.4f}")
    print(f"Weight standard deviation: {np.std(weights):.4f}")
    
    # Visualize weight matrix
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(initial_weights, cmap='hot', vmin=0, vmax=5)
    plt.colorbar()
    plt.title('Initial Weights')
    
    plt.subplot(1, 2, 2)
    plt.imshow(weights, cmap='hot', vmin=0, vmax=5)
    plt.colorbar()
    plt.title('Final Weights (Hebbian)')
    plt.tight_layout()
    plt.show()


def test_reward_modulated_learning():
    """Test reward-modulated STDP learning."""
    print("\n=== Testing Reward-Modulated STDP ===")
    
    config = PlasticityConfig(
        learning_rate=0.01,
        reward_decay=0.9,
        reward_sensitivity=2.0,
        dopamine_time_constant=200.0,
        weight_min=0.0,
        weight_max=5.0
    )
    
    manager = PlasticityManager(config)
    manager.activate_rule('rstdp')
    
    weights = np.ones((3, 3)) * 1.5
    initial_weights = weights.copy()
    
    # Simulate learning with rewards
    n_trials = 20
    trial_length = 50
    
    reward_history = []
    weight_means = []
    
    for trial in range(n_trials):
        # Determine if this trial gets reward (based on some criterion)
        trial_reward = 1.0 if trial % 3 == 0 else -0.5
        manager.set_reward(trial_reward)
        reward_history.append(trial_reward)
        
        for t in range(trial_length):
            # Generate activity
            pre_activity = np.random.random(3) * 0.5
            post_activity = np.random.random(3) * 0.5
            
            # Add spikes
            pre_spikes = np.random.random(3) < 0.1
            post_spikes = np.random.random(3) < 0.1
            
            weights = manager.update_weights(
                weights, pre_activity, post_activity,
                dt=1.0, pre_spike=pre_spikes, post_spike=post_spikes
            )
        
        weight_means.append(np.mean(weights))
    
    print(f"Initial mean weight: {np.mean(initial_weights):.4f}")
    print(f"Final mean weight: {np.mean(weights):.4f}")
    print(f"Total reward: {sum(reward_history):.2f}")
    
    # Plot results
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(reward_history, 'o-')
    plt.xlabel('Trial')
    plt.ylabel('Reward')
    plt.title('Reward Schedule')
    
    plt.subplot(1, 3, 2)
    plt.plot(weight_means)
    plt.xlabel('Trial')
    plt.ylabel('Mean Weight')
    plt.title('Weight Evolution')
    
    plt.subplot(1, 3, 3)
    plt.imshow(weights - initial_weights, cmap='RdBu', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Weight Changes')
    plt.tight_layout()
    plt.show()


def test_bcm_learning():
    """Test BCM learning with sliding threshold."""
    print("\n=== Testing BCM Learning ===")
    
    config = PlasticityConfig(
        learning_rate=0.001,
        bcm_threshold=0.5,
        bcm_time_constant=1000.0,
        weight_min=0.0,
        weight_max=3.0
    )
    
    # Create BCM rule directly
    bcm_rule = BCMRule(config)
    
    # Single synapse test
    weight = 1.0
    weight_history = [weight]
    threshold_history = [bcm_rule.sliding_threshold]
    
    for t in range(500):
        # Varying activity levels
        pre_activity = 0.5 + 0.3 * np.sin(t * 0.05)
        post_activity = 0.4 + 0.4 * np.sin(t * 0.03)
        
        weight = bcm_rule.update_weight(weight, pre_activity, post_activity)
        weight_history.append(weight)
        threshold_history.append(bcm_rule.sliding_threshold)
    
    print(f"Initial weight: 1.0")
    print(f"Final weight: {weight:.4f}")
    print(f"Final threshold: {bcm_rule.sliding_threshold:.4f}")
    
    # Plot BCM dynamics
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(weight_history, label='Weight')
    plt.plot(threshold_history, label='Threshold', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.title('BCM Learning Dynamics')
    
    plt.subplot(1, 2, 2)
    plt.plot(bcm_rule.activity_history)
    plt.xlabel('Time (recent)')
    plt.ylabel('Post-synaptic Activity')
    plt.title('Activity History')
    plt.tight_layout()
    plt.show()


def test_custom_plasticity():
    """Test custom user-defined plasticity rule."""
    print("\n=== Testing Custom Plasticity Rule ===")
    
    # Define a custom calcium-based plasticity rule
    def calcium_plasticity(pre_activity, post_activity, current_weight, state, config, **kwargs):
        """Custom calcium-based plasticity rule."""
        # Initialize calcium level if needed
        if 'calcium' not in state:
            state['calcium'] = 0.0
        
        # Update calcium based on coincident activity
        calcium_influx = pre_activity * post_activity * 2.0
        state['calcium'] = state['calcium'] * 0.95 + calcium_influx  # Decay + influx
        
        # Plasticity depends on calcium level
        if state['calcium'] > 1.0:  # High calcium -> LTP
            delta_w = config.learning_rate * (state['calcium'] - 1.0) * 0.1
        elif state['calcium'] < 0.5:  # Low calcium -> LTD
            delta_w = -config.learning_rate * (0.5 - state['calcium']) * 0.05
        else:  # Medium calcium -> no change
            delta_w = 0.0
        
        return delta_w
    
    # Create manager with custom rule
    config = PlasticityConfig(learning_rate=0.1, weight_min=0.0, weight_max=5.0)
    manager = PlasticityManager(config)
    manager.add_custom_rule('calcium', calcium_plasticity)
    manager.activate_rule('calcium')
    
    # Test the custom rule
    weights = np.ones((2, 2)) * 2.0
    initial_weights = weights.copy()
    
    calcium_history = []
    
    for t in range(200):
        # Create activity patterns
        if t < 50:  # Low activity
            pre_activity = np.random.random(2) * 0.2
            post_activity = np.random.random(2) * 0.2
        elif t < 100:  # High correlated activity
            pre_activity = np.random.random(2) * 0.8 + 0.2
            post_activity = pre_activity + np.random.normal(0, 0.1, 2)
        elif t < 150:  # Medium activity
            pre_activity = np.random.random(2) * 0.5
            post_activity = np.random.random(2) * 0.5
        else:  # Low activity again
            pre_activity = np.random.random(2) * 0.2
            post_activity = np.random.random(2) * 0.2
        
        weights = manager.update_weights(weights, pre_activity, post_activity)
        
        # Track calcium for one synapse (for visualization)
        if 'calcium' in manager.rules['calcium'].state:
            calcium_history.append(manager.rules['calcium'].state['calcium'])
    
    print(f"Initial weight[0,0]: {initial_weights[0, 0]:.4f}")
    print(f"Final weight[0,0]: {weights[0, 0]:.4f}")
    
    # Visualize custom rule behavior
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(calcium_history)
    plt.axhline(y=1.0, color='r', linestyle='--', label='LTP threshold')
    plt.axhline(y=0.5, color='b', linestyle='--', label='LTD threshold')
    plt.xlabel('Time')
    plt.ylabel('Calcium Level')
    plt.legend()
    plt.title('Custom Rule: Calcium Dynamics')
    
    plt.subplot(1, 2, 2)
    plt.imshow(weights - initial_weights, cmap='RdBu', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Weight Changes (Custom Rule)')
    plt.tight_layout()
    plt.show()


def test_config_loading():
    """Test loading and saving configuration files."""
    print("\n=== Testing Configuration Loading/Saving ===")
    
    # Load from YAML
    yaml_path = Path('../configs/learning_config.yaml')
    if yaml_path.exists():
        manager = PlasticityManager()
        manager.load_config(yaml_path, format='yaml')
        print(f"Loaded YAML config - Learning rate: {manager.config.learning_rate}")
        print(f"Available rules: {manager.get_statistics()['available_rules']}")
    
    # Load from JSON
    json_path = Path('../configs/learning_config.json')
    if json_path.exists():
        manager = PlasticityManager()
        manager.load_config(json_path, format='json')
        print(f"Loaded JSON config - Learning rate: {manager.config.learning_rate}")
    
    # Create and save new config
    new_config = PlasticityConfig(
        learning_rate=0.02,
        tau_plus=15.0,
        tau_minus=25.0,
        weight_min=0.1,
        weight_max=8.0
    )
    
    # Save to temporary files
    temp_yaml = Path('temp_config.yaml')
    temp_json = Path('temp_config.json')
    
    new_config.to_yaml(temp_yaml)
    new_config.to_json(temp_json)
    print(f"Saved configuration to {temp_yaml} and {temp_json}")
    
    # Clean up
    import os
    if temp_yaml.exists():
        os.remove(temp_yaml)
    if temp_json.exists():
        os.remove(temp_json)


def test_multiple_rules():
    """Test combining multiple plasticity rules."""
    print("\n=== Testing Multiple Plasticity Rules ===")
    
    config = PlasticityConfig(
        learning_rate=0.01,
        weight_min=0.0,
        weight_max=5.0,
        target_rate=5.0  # For homeostatic plasticity
    )
    
    manager = PlasticityManager(config)
    
    # Activate multiple rules
    manager.activate_rule('stdp')
    manager.activate_rule('homeostatic')
    
    print(f"Active rules: {manager.active_rules}")
    
    weights = np.ones((3, 3)) * 2.0
    initial_weights = weights.copy()
    
    # Simulate with both rules active
    for t in range(100):
        pre_spikes = np.random.random(3) < 0.1
        post_spikes = np.random.random(3) < 0.08  # Slightly lower rate
        
        pre_activity = pre_spikes.astype(float)
        post_activity = post_spikes.astype(float)
        
        weights = manager.update_weights(
            weights, pre_activity, post_activity,
            dt=1.0, pre_spike=pre_spikes, post_spike=post_spikes
        )
    
    print(f"Weight change with STDP + Homeostatic: {np.mean(weights - initial_weights):.4f}")
    
    # Get statistics
    stats = manager.get_statistics()
    print(f"Statistics: {len(stats['weight_histories'])} rules with history")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Learning and Plasticity Mechanisms")
    print("=" * 60)
    
    # Run tests
    test_stdp_learning()
    test_hebbian_learning()
    test_reward_modulated_learning()
    test_bcm_learning()
    test_custom_plasticity()
    test_config_loading()
    test_multiple_rules()
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
