# Learning and Plasticity

Comprehensive guide to implementing learning mechanisms and synaptic plasticity in neuromorphic networks.

## Overview

Synaptic plasticity is the biological basis of learning and memory. This tutorial covers the various plasticity mechanisms implemented in the system, from basic Hebbian learning to complex neuromodulated plasticity.

## Spike-Timing-Dependent Plasticity (STDP)

### Basic STDP

STDP modifies synaptic weights based on the relative timing of pre- and post-synaptic spikes:

```python
from core.synapses import STDP_Synapse, SynapseType

# Create STDP synapse
synapse = STDP_Synapse(
    synapse_id=0,
    pre_neuron_id=0,
    post_neuron_id=1,
    weight=1.0,
    synapse_type=SynapseType.EXCITATORY,
    tau_stdp=20.0,  # Time constant
    A_plus=0.01,    # LTP amplitude
    A_minus=0.01    # LTD amplitude
)

# Simulate spike timing
print(f"Initial weight: {synapse.weight}")

# Pre before post → LTP (strengthening)
synapse.pre_spike(t=10.0)
synapse.post_spike(t=12.0)
print(f"After LTP: {synapse.weight}")

# Post before pre → LTD (weakening)
synapse.post_spike(t=20.0)
synapse.pre_spike(t=22.0)
print(f"After LTD: {synapse.weight}")
```

### Asymmetric STDP

Different time constants for potentiation and depression:

```python
from core.learning import AsymmetricSTDP

# Create asymmetric STDP rule
plasticity = AsymmetricSTDP(
    tau_plus=17.0,   # LTP time constant
    tau_minus=34.0,  # LTD time constant
    A_plus=0.005,
    A_minus=0.006,
    weight_min=0.0,
    weight_max=5.0
)

# Apply to synapses
synapse.set_plasticity_rule(plasticity)
```

### Triplet STDP

Considers triplets of spikes for more accurate biological modeling:

```python
from core.learning import TripletSTDP

# Create triplet STDP rule
plasticity = TripletSTDP(
    tau_plus=16.8,
    tau_minus=33.7,
    tau_x=101.0,
    tau_y=125.0,
    A2_plus=0.0046,
    A3_plus=0.0091,
    A2_minus=0.0057,
    A3_minus=0.0015
)

# Track spike triplets
plasticity.add_pre_spike(t=10.0)
plasticity.add_post_spike(t=12.0)
plasticity.add_pre_spike(t=14.0)  # Triplet formed

weight_change = plasticity.compute_weight_change()
print(f"Weight change from triplet: {weight_change}")
```

## Short-Term Plasticity (STP)

### Depression and Facilitation

STP causes temporary changes in synaptic efficacy:

```python
from core.synapses import ShortTermPlasticitySynapse

# Create STP synapse with depression
depressing_synapse = ShortTermPlasticitySynapse(
    synapse_id=0,
    pre_neuron_id=0,
    post_neuron_id=1,
    weight=2.0,
    U=0.5,           # Utilization parameter
    tau_dep=100.0,   # Depression time constant
    tau_fac=50.0     # Facilitation time constant
)

# Simulate repeated stimulation
for t in range(0, 100, 10):
    efficacy = depressing_synapse.process_spike(t)
    print(f"Time {t}ms: Efficacy = {efficacy:.3f}")
```

### Mixed STP

Combining depression and facilitation:

```python
# Create synapse with mixed dynamics
mixed_synapse = ShortTermPlasticitySynapse(
    synapse_id=1,
    pre_neuron_id=0,
    post_neuron_id=1,
    weight=1.0,
    U=0.2,          # Lower initial utilization
    tau_dep=200.0,  # Slower depression
    tau_fac=500.0   # Strong facilitation
)

# Different firing patterns produce different effects
# High frequency → facilitation dominates
for t in range(0, 50, 5):
    efficacy = mixed_synapse.process_spike(t)
    print(f"High freq {t}ms: {efficacy:.3f}")

mixed_synapse.reset()

# Low frequency → depression dominates  
for t in range(0, 500, 50):
    efficacy = mixed_synapse.process_spike(t)
    print(f"Low freq {t}ms: {efficacy:.3f}")
```

## Reward-Modulated Learning

### Basic Reward-Modulated STDP

Combining STDP with reward signals:

```python
from core.synapses import RSTDP_Synapse

# Create reward-modulated synapse
rstdp_synapse = RSTDP_Synapse(
    synapse_id=0,
    pre_neuron_id=0,
    post_neuron_id=1,
    weight=1.0,
    learning_rate=0.01,
    tau_eligibility=1000.0  # Eligibility trace decay
)

# Regular STDP creates eligibility trace
rstdp_synapse.pre_spike(t=10.0)
rstdp_synapse.post_spike(t=12.0)

# Reward signal modulates weight change
rstdp_synapse.apply_reward(reward=1.0, t=15.0)
print(f"Weight after reward: {rstdp_synapse.weight}")

# Punishment (negative reward)
rstdp_synapse.apply_reward(reward=-0.5, t=20.0)
print(f"Weight after punishment: {rstdp_synapse.weight}")
```

### Three-Factor Learning Rule

Combining pre-synaptic, post-synaptic, and modulatory signals:

```python
from core.learning import ThreeFactorRule

# Create three-factor learning rule
learning_rule = ThreeFactorRule(
    learning_rate=0.01,
    tau_eligibility=500.0,
    tau_dopamine=200.0,
    baseline_dopamine=0.1
)

# Simulate learning episode
# 1. Coincident pre/post activity
learning_rule.update_eligibility(pre_spike=True, post_spike=True, t=10.0)

# 2. Delayed reward signal
learning_rule.set_dopamine(level=1.0, t=100.0)

# 3. Compute weight update
weight_change = learning_rule.compute_update(t=150.0)
print(f"Three-factor weight change: {weight_change}")
```

## Homeostatic Plasticity

### Synaptic Scaling

Maintains stable firing rates:

```python
from core.learning import SynapticScaling

# Create homeostatic controller
homeostasis = SynapticScaling(
    target_rate=5.0,  # Hz
    tau_homeostasis=10000.0,  # Slow time constant
    scaling_factor=0.001
)

# Monitor firing rate and adjust weights
current_rate = 10.0  # Too high
scaling = homeostasis.compute_scaling(current_rate)
print(f"Scaling factor: {scaling}")  # < 1.0 to reduce activity

# Apply to all synapses
for synapse in synapses:
    synapse.weight *= scaling
```

### Intrinsic Plasticity

Adjusts neuron excitability:

```python
from core.learning import IntrinsicPlasticity

# Create intrinsic plasticity mechanism
ip = IntrinsicPlasticity(
    target_rate=5.0,
    learning_rate=0.001,
    tau_avg=1000.0
)

# Adjust neuron threshold based on activity
neuron_rate = 2.0  # Too low
threshold_change = ip.compute_threshold_adjustment(neuron_rate)
print(f"Threshold adjustment: {threshold_change}")  # Negative to increase excitability
```

## Structural Plasticity

### Synapse Formation and Elimination

Dynamic network topology:

```python
from core.learning import StructuralPlasticity

# Configure structural plasticity
structural = StructuralPlasticity(
    formation_rate=0.001,
    elimination_rate=0.0005,
    max_synapses_per_neuron=100,
    activity_threshold=1.0
)

# Check if new synapse should form
pre_activity = 5.0  # Hz
post_activity = 4.0  # Hz
distance = 0.1  # Normalized distance

form_probability = structural.formation_probability(
    pre_activity, post_activity, distance
)
print(f"Formation probability: {form_probability}")

# Check if synapse should be eliminated
synapse_weight = 0.01  # Very weak
synapse_age = 10000  # Old synapse

eliminate_probability = structural.elimination_probability(
    synapse_weight, synapse_age
)
print(f"Elimination probability: {eliminate_probability}")
```

## Learning in Networks

### Supervised Learning

Training networks with target patterns:

```python
from api.neuromorphic_api import NeuromorphicAPI
import numpy as np

# Create network
api = NeuromorphicAPI()
api.create_network()

# Build architecture
api.add_sensory_layer("input", 100, "rate")
api.add_processing_layer("hidden", 50, "adex")
api.add_motor_layer("output", 10)

# Enable supervised learning
api.connect_layers("input", "hidden", "feedforward", 
                  synapse_type="stdp", learning_enabled=True)
api.connect_layers("hidden", "output", "feedforward",
                  synapse_type="rstdp", learning_enabled=True)

# Training loop
for epoch in range(100):
    # Generate input pattern
    input_pattern = np.random.rand(100)
    input_spikes = [(i, t) for i, rate in enumerate(input_pattern)
                   for t in np.random.exponential(1000/rate, 10)]
    
    # Define target output
    target = np.zeros(10)
    target[epoch % 10] = 1.0
    
    # Run simulation with supervision
    results = api.run_simulation(
        duration=100.0,
        external_inputs={"input": input_spikes},
        target_outputs={"output": target},
        learning_rate=0.01
    )
    
    # Compute error
    output_rates = results['layer_rates']['output']
    error = np.mean((output_rates - target) ** 2)
    print(f"Epoch {epoch}: MSE = {error:.4f}")
```

### Unsupervised Learning

Self-organizing networks:

```python
# Create self-organizing map
api = NeuromorphicAPI()
api.create_network()

# Build SOM architecture
api.add_sensory_layer("input", 100, "rate")
api.add_processing_layer("som", 10*10, "lif")  # 10x10 grid

# Lateral connections for competition
api.connect_layers("input", "som", "all_to_all",
                  synapse_type="stdp", learning_enabled=True)
api.connect_layers("som", "som", "lateral_inhibition",
                  synapse_type="static", weight=-2.0)

# Training with random patterns
for iteration in range(1000):
    # Random input pattern
    pattern = np.random.rand(100)
    input_spikes = encode_pattern(pattern)
    
    # Run with learning
    results = api.run_simulation(
        duration=50.0,
        external_inputs={"input": input_spikes},
        learning_rate=0.1 * np.exp(-iteration/200)  # Decreasing rate
    )
    
    # Winner neuron strengthens connections
    winner = np.argmax(results['layer_rates']['som'])
    print(f"Iteration {iteration}: Winner = {winner}")
```

### Reinforcement Learning

Learning from rewards:

```python
# Create reinforcement learning network
api = NeuromorphicAPI()
api.create_network()

# Actor-critic architecture
api.add_sensory_layer("state", 50, "population")
api.add_processing_layer("critic", 20, "adex")
api.add_processing_layer("actor", 30, "adex")
api.add_motor_layer("action", 4)

# Configure reward-modulated connections
api.connect_layers("state", "critic", "feedforward",
                  synapse_type="rstdp")
api.connect_layers("state", "actor", "feedforward",
                  synapse_type="rstdp")
api.connect_layers("actor", "action", "feedforward",
                  synapse_type="rstdp")

# Learning episode
state = get_environment_state()
for step in range(100):
    # Encode state
    state_spikes = encode_state(state)
    
    # Get action from network
    results = api.run_simulation(
        duration=20.0,
        external_inputs={"state": state_spikes}
    )
    
    # Execute action
    action = decode_action(results['layer_spike_times']['action'])
    next_state, reward = environment.step(action)
    
    # Apply reward signal
    api.apply_neuromodulation(
        dopamine_level=reward,
        duration=10.0
    )
    
    # Update weights based on reward
    api.update_weights(learning_rate=0.01)
    
    state = next_state
    print(f"Step {step}: Reward = {reward:.2f}")
```

## Advanced Topics

### Meta-Plasticity

Plasticity of plasticity:

```python
from core.learning import MetaPlasticity

# Create meta-plasticity controller
meta = MetaPlasticity(
    initial_learning_rate=0.01,
    meta_learning_rate=0.001,
    tau_meta=10000.0
)

# Adapt learning rate based on performance
performance_history = [0.5, 0.6, 0.65, 0.7, 0.72]
new_learning_rate = meta.adapt_learning_rate(performance_history)
print(f"Adapted learning rate: {new_learning_rate}")
```

### Consolidation

Memory consolidation and transfer:

```python
from core.memory import MemoryConsolidation

# Create consolidation system
consolidation = MemoryConsolidation(
    fast_learning_rate=0.1,
    slow_learning_rate=0.001,
    transfer_rate=0.01,
    consolidation_threshold=0.5
)

# Fast learning during experience
fast_weights = consolidation.fast_learn(experience_data)

# Slow consolidation during rest
slow_weights = consolidation.consolidate(fast_weights)

# Retrieve consolidated memory
memory = consolidation.retrieve(cue)
```

## Best Practices

1. **Choose appropriate plasticity rules**
   - STDP for temporal pattern learning
   - STP for dynamic filtering
   - RSTDP for goal-directed learning

2. **Balance plasticity and stability**
   - Use homeostatic mechanisms
   - Implement weight bounds
   - Consider meta-plasticity

3. **Optimize learning rates**
   - Start with small rates (0.001-0.01)
   - Use adaptive/decaying schedules
   - Different rates for different layers

4. **Monitor learning progress**
   - Track weight distributions
   - Measure performance metrics
   - Detect pathological states

## Next Steps

- Explore [Neuromodulation](04_neuromodulation.md)
- Learn about [Edge Deployment](05_edge_deployment.md)
- Build [Complex Networks](06_complex_networks.md)

---

*← [Sensory Encoding](02_sensory_encoding.md) | [Neuromodulation →](04_neuromodulation.md)*
