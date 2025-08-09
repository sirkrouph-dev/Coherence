# Learning and Plasticity Module Documentation

## Overview

The learning and plasticity module provides a comprehensive framework for implementing synaptic plasticity mechanisms in spiking neural networks. It supports various biologically-inspired learning rules and allows for custom user-defined plasticity mechanisms.

## Features

### Built-in Plasticity Rules

1. **STDP (Spike-Timing-Dependent Plasticity)**
   - Classical STDP with exponential windows
   - Configurable LTP/LTD time constants and amplitudes
   - Weight-dependent scaling

2. **Hebbian Learning**
   - "Cells that fire together, wire together"
   - Correlation-based weight updates
   - Includes decay term for stability

3. **BCM (Bienenstock-Cooper-Munro)**
   - Sliding threshold mechanism
   - Prevents runaway potentiation/depression
   - Adapts to maintain stable activity

4. **Reward-Modulated STDP**
   - Combines timing-based and reward-based plasticity
   - Eligibility traces for credit assignment
   - Dopamine-like neuromodulation

5. **Triplet STDP**
   - More accurate modeling of experimental data
   - Fast and slow traces for pre/post activity
   - Captures frequency-dependent effects

6. **Homeostatic Plasticity**
   - Maintains stable firing rates
   - Slow adaptation to target rates
   - Prevents network instability

### Custom Plasticity Rules

Users can define their own plasticity rules by:
- Implementing a function with the required signature
- Extending the `PlasticityRule` base class
- Using the `CustomPlasticityRule` wrapper

## Usage

### Basic Example

```python
from core.learning import PlasticityManager, PlasticityConfig

# Create configuration
config = PlasticityConfig(
    learning_rate=0.01,
    tau_plus=20.0,
    tau_minus=20.0,
    weight_min=0.0,
    weight_max=10.0
)

# Initialize manager
manager = PlasticityManager(config)

# Activate learning rules
manager.activate_rule('stdp')
manager.activate_rule('homeostatic')

# Update weights based on activity
weights = manager.update_weights(
    weights,
    pre_activity,
    post_activity,
    dt=1.0,
    pre_spike=pre_spikes,
    post_spike=post_spikes
)
```

### Configuration via YAML

```yaml
# learning_config.yaml
learning_rate: 0.01
weight_min: 0.0
weight_max: 10.0

# STDP parameters
tau_plus: 20.0
tau_minus: 20.0
A_plus: 0.01
A_minus: 0.012

# Hebbian parameters
hebbian_threshold: 0.5
hebbian_decay: 0.99

# Reward modulation
reward_sensitivity: 1.0
dopamine_time_constant: 200.0
```

Load configuration:
```python
manager = PlasticityManager()
manager.load_config('learning_config.yaml', format='yaml')
```

### Configuration via JSON

```json
{
  "learning_rate": 0.01,
  "weight_min": 0.0,
  "weight_max": 10.0,
  "tau_plus": 20.0,
  "tau_minus": 20.0,
  "hebbian_threshold": 0.5
}
```

Load configuration:
```python
manager.load_config('learning_config.json', format='json')
```

### Custom Plasticity Rule

```python
def my_custom_rule(pre_activity, post_activity, current_weight, state, config, **kwargs):
    """
    Custom plasticity rule example.
    
    Args:
        pre_activity: Presynaptic activity level
        post_activity: Postsynaptic activity level
        current_weight: Current synaptic weight
        state: Persistent state dictionary for this rule
        config: PlasticityConfig instance
        **kwargs: Additional parameters (e.g., voltage, calcium)
    
    Returns:
        delta_w: Weight change
    """
    # Initialize state variables
    if 'my_variable' not in state:
        state['my_variable'] = 0.0
    
    # Compute weight change
    correlation = pre_activity * post_activity
    delta_w = config.learning_rate * correlation
    
    # Update state
    state['my_variable'] += correlation
    
    return delta_w

# Add to manager
manager.add_custom_rule('my_rule', my_custom_rule)
manager.activate_rule('my_rule')
```

### Reward-Modulated Learning

```python
# Create manager with reward-modulated STDP
manager = PlasticityManager(config)
manager.activate_rule('rstdp')

# During simulation
for trial in range(n_trials):
    # Compute reward based on performance
    reward = compute_reward(performance)
    
    # Set reward signal
    manager.set_reward(reward)
    
    # Update weights with reward modulation
    weights = manager.update_weights(weights, pre_activity, post_activity)
```

## Integration with Synapses

The learning module integrates seamlessly with the synapse models:

```python
from core.synapses import STDP_Synapse

# Create synapse with integrated plasticity
synapse = STDP_Synapse(
    synapse_id=0,
    pre_neuron_id=1,
    post_neuron_id=2,
    tau_stdp=20.0,
    A_plus=0.01,
    A_minus=0.012
)

# The synapse automatically has a PlasticityManager
# configured for STDP learning
```

## Advanced Features

### Multiple Active Rules

Multiple plasticity rules can be active simultaneously:

```python
manager.activate_rule('stdp')
manager.activate_rule('homeostatic')
manager.activate_rule('bcm')

# All active rules will be applied during weight updates
weights = manager.update_weights(weights, pre_activity, post_activity)
```

### Rule-Specific Updates

Apply specific rules selectively:

```python
# Update only with STDP
weights = manager.update_weights(
    weights, pre_activity, post_activity,
    rule_name='stdp'
)
```

### Statistics and Monitoring

Get statistics about plasticity:

```python
stats = manager.get_statistics()
print(f"Active rules: {stats['active_rules']}")
print(f"Weight histories: {stats['weight_histories']}")
```

## Parameters Reference

### Common Parameters
- `learning_rate`: Base learning rate for all rules
- `weight_min`: Minimum allowed weight
- `weight_max`: Maximum allowed weight
- `enabled`: Whether plasticity is enabled

### STDP Parameters
- `tau_plus`: LTP time constant (ms)
- `tau_minus`: LTD time constant (ms)
- `A_plus`: LTP amplitude
- `A_minus`: LTD amplitude

### Hebbian Parameters
- `hebbian_threshold`: Correlation threshold for potentiation
- `hebbian_decay`: Weight decay factor

### BCM Parameters
- `bcm_threshold`: Initial sliding threshold
- `bcm_time_constant`: Threshold adaptation time constant

### Reward Modulation Parameters
- `reward_decay`: Eligibility trace decay rate
- `reward_sensitivity`: Sensitivity to reward signals
- `dopamine_time_constant`: Dopamine trace decay time constant

### Homeostatic Parameters
- `target_rate`: Target firing rate (Hz)
- `homeostatic_time_constant`: Adaptation time constant

## Examples

Complete examples are available in `examples/test_learning.py`:

```bash
python examples/test_learning.py
```

This will demonstrate:
- STDP learning with spike patterns
- Hebbian learning with correlated activity
- Reward-modulated learning
- BCM dynamics
- Custom plasticity rules
- Configuration loading/saving
- Multiple rule combinations

## Best Practices

1. **Choose appropriate learning rates**: Start with small values (0.001-0.01) and adjust based on network behavior

2. **Set weight bounds**: Always define `weight_min` and `weight_max` to prevent unbounded growth

3. **Combine complementary rules**: Use STDP for fast learning with homeostatic plasticity for stability

4. **Monitor weight distributions**: Track weight statistics to detect pathological learning

5. **Use configuration files**: Store learning parameters in YAML/JSON for reproducibility

6. **Test custom rules thoroughly**: Validate custom plasticity rules on simple networks first

## Troubleshooting

### Weights saturating at bounds
- Reduce learning rate
- Adjust LTP/LTD balance (A_plus vs A_minus)
- Add homeostatic plasticity

### Unstable learning
- Check weight bounds
- Add decay terms
- Use BCM or homeostatic plasticity for stability

### No learning occurring
- Verify plasticity is enabled
- Check that rules are activated
- Ensure activity levels are sufficient

## References

- Bi, G. Q., & Poo, M. M. (1998). Synaptic modifications in cultured hippocampal neurons: dependence on spike timing, synaptic strength, and postsynaptic cell type. Journal of neuroscience, 18(24), 10464-10472.

- Bienenstock, E. L., Cooper, L. N., & Munro, P. W. (1982). Theory for the development of neuron selectivity: orientation specificity and binocular interaction in visual cortex. Journal of Neuroscience, 2(1), 32-48.

- Pfister, J. P., & Gerstner, W. (2006). Triplets of spikes in a model of spike timing-dependent plasticity. Journal of Neuroscience, 26(38), 9673-9682.

- Turrigiano, G. G. (2008). The self-tuning neuron: synaptic scaling of excitatory synapses. Cell, 135(3), 422-435.
