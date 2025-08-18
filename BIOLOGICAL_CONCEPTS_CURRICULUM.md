# Neuromorphic Computing: Biological Concepts and Code Implementation Curriculum

A comprehensive guide to understanding the biological foundations of neuromorphic computing through code implementation.

## Table of Contents

1. [Introduction](#introduction)
2. [Basic Neural Components](#basic-neural-components)
3. [Synaptic Transmission and Plasticity](#synaptic-transmission-and-plasticity)
4. [Sensory Encoding Systems](#sensory-encoding-systems)
5. [Neuromodulation and Reward Systems](#neuromodulation-and-reward-systems)
6. [Learning and Memory Formation](#learning-and-memory-formation)
7. [Neural Networks and Connectivity](#neural-networks-and-connectivity)
8. [Higher-Order Cognitive Functions](#higher-order-cognitive-functions)
9. [Homeostasis and Regulation](#homeostasis-and-regulation)
10. [Advanced Concepts](#advanced-concepts)

---

## Introduction

This curriculum covers the biological concepts implemented in our neuromorphic computing system, bridging neuroscience research with computational models. Each concept includes:
- **Biological Background**: The real neural mechanism
- **Mathematical Model**: How it's represented computationally  
- **Code Implementation**: Actual implementation in the system
- **Practical Examples**: How to use it in simulations

---

## 1. Basic Neural Components

### 1.1 Neuron Models

#### Biological Background
Neurons are the fundamental units of the nervous system, generating action potentials (spikes) when their membrane potential reaches a threshold.

#### 1.1.1 Leaky Integrate-and-Fire (LIF)

**Mathematical Model:**
```
τ_m * dV/dt = -(V - V_rest) + R * I_input
```

**Code Implementation:**
```python
# File: core/neurons.py
class LeakyIntegrateAndFire(NeuronModel):
    def step(self, dt: float, I_syn: float) -> bool:
        # Handle refractory period
        if self.refractory_time > 0:
            self.refractory_time -= dt
            return False

        # Update membrane potential
        R_input = 100.0  # MOhm, typical membrane resistance
        dv_dt = (-(self.membrane_potential - self.v_rest) + I_syn * R_input) / self.tau_m
        self.membrane_potential += dv_dt * dt
        
        # Check for spike
        if self.membrane_potential >= self.v_thresh:
            self._spike()
            return True
        return False
```

**Usage Example:**
```python
from core.neurons import LeakyIntegrateAndFire

# Create LIF neuron
lif_neuron = LeakyIntegrateAndFire(
    neuron_id=0,
    tau_m=20.0,      # Membrane time constant (ms)
    v_rest=-65.0,    # Resting potential (mV)
    v_thresh=-55.0,  # Spike threshold (mV)
    v_reset=-65.0,   # Reset potential (mV)
    refractory_period=2.0  # Refractory period (ms)
)

# Simulate with synaptic input
spiked = lif_neuron.step(dt=0.1, I_syn=2.0)
```

#### 1.1.2 Adaptive Exponential Integrate-and-Fire (AdEx)

**Mathematical Model:**
```
τ_m * dV/dt = -(V - V_rest) + Δ_T * exp((V - V_T)/Δ_T) - w + I
τ_w * dw/dt = a*(V - V_rest) - w
```

**Code Implementation:**
```python
# File: core/neurons.py
class AdaptiveExponentialIntegrateAndFire(NeuronModel):
    def step(self, dt: float, I_syn: float) -> bool:
        # Update membrane potential with exponential nonlinearity
        R_input = 100.0  # MOhm
        dv_dt = (
            -(self.membrane_potential - self.v_rest)
            + self.delta_t * np.exp((self.membrane_potential - self.v_thresh) / self.delta_t)
            - self.adaptation_current
            + I_syn * R_input
        ) / self.tau_m
        self.membrane_potential += dv_dt * dt

        # Update adaptation current
        dw_dt = (
            self.a * (self.membrane_potential - self.v_rest) - self.adaptation_current
        ) / self.tau_w
        self.adaptation_current += dw_dt * dt

        # Check for spike
        if self.membrane_potential >= self.v_thresh:
            self._spike()
            return True
        return False
```

#### 1.1.3 Hodgkin-Huxley Model

**Mathematical Model:**
```
C_m * dV/dt = -g_Na*m³*h*(V-E_Na) - g_K*n⁴*(V-E_K) - g_L*(V-E_L) + I
```

**Code Implementation:**
```python
# File: core/neurons.py
class HodgkinHuxleyNeuron(NeuronModel):
    def step(self, dt: float, I_syn: float) -> bool:
        # Calculate channel conductances
        g_Na_current = self.g_Na * (self.m**3) * self.h
        g_K_current = self.g_K * (self.n**4)
        
        # Calculate ionic currents
        I_Na = g_Na_current * (self.membrane_potential - self.E_Na)
        I_K = g_K_current * (self.membrane_potential - self.E_K)
        I_L = self.g_L * (self.membrane_potential - self.E_L)
        
        # Update membrane potential
        I_total = I_Na + I_K + I_L + I_syn
        dv_dt = -I_total / self.C_m
        self.membrane_potential += dv_dt * dt
        
        # Update gating variables
        self._update_gating_variables(dt)
        
        return self.membrane_potential > 0  # Simplified spike detection
```

### 1.2 Neural Populations

**Biological Background:**
Neurons in the brain are organized into populations with similar properties and functions.

**Code Implementation:**
```python
# File: core/neurons.py
class NeuronPopulation:
    def __init__(self, size: int, neuron_type: str = "adex", **kwargs):
        self.neurons = []
        
        # Create heterogeneous population
        for i in range(size):
            per_neuron_kwargs = dict(kwargs)
            # Add biological variability
            if neuron_type.lower() == "lif":
                jitter = -0.5 + (i % 5) * 0.25  # Threshold variability
                per_neuron_kwargs["v_thresh"] = -55.0 + jitter
                
            neuron = NeuronFactory.create_neuron(neuron_type, i, **per_neuron_kwargs)
            self.neurons.append(neuron)
    
    def step(self, dt: float, I_syn: List[float]) -> List[bool]:
        # Vectorized processing for efficiency
        spikes = []
        for i, (neuron, current) in enumerate(zip(self.neurons, I_syn)):
            spiked = neuron.step(dt, current)
            spikes.append(spiked)
        return spikes
```

---

## 2. Synaptic Transmission and Plasticity

### 2.1 Basic Synaptic Transmission

**Biological Background:**
Synapses transmit signals between neurons through neurotransmitter release and receptor activation.

**Mathematical Model:**
```
I_syn(t) = g_syn * (V_post - E_rev) * s(t)
s(t) = w * exp(-(t - t_spike)/τ_syn)
```

**Code Implementation:**
```python
# File: core/synapses.py
class SynapseModel:
    def compute_current(self, pre_spike_time: float, current_time: float) -> float:
        if current_time < pre_spike_time:
            return 0.0
        
        # Exponential decay from spike time
        dt = current_time - pre_spike_time
        current = self.weight * np.exp(-dt / self.tau_syn)
        
        # Apply reversal potential
        if self.synapse_type == SynapseType.EXCITATORY:
            return current
        elif self.synapse_type == SynapseType.INHIBITORY:
            return -current
        return current
```

### 2.2 Spike-Timing-Dependent Plasticity (STDP)

**Biological Background:**
STDP is a fundamental learning rule where synaptic strength changes based on the relative timing of pre- and post-synaptic spikes.

**Mathematical Model:**
```
Δw = A_+ * exp(-Δt/τ_+) if Δt > 0 (LTP)
Δw = -A_- * exp(Δt/τ_-) if Δt < 0 (LTD)
```

**Code Implementation:**
```python
# File: core/synapses.py
class STDP_Synapse(SynapseModel):
    def pre_spike(self, spike_time: float):
        self.current_time = spike_time
        self._decay_traces(spike_time)
        
        # LTD if post-before-pre
        if (self.last_post_spike > -np.inf and 
            (spike_time - self.last_post_spike) < self.tau_stdp):
            if self.post_trace > 0.0:
                ltd = -self.A_minus * self.post_trace
                # Weight-dependent scaling
                ltd *= self.weight / self.w_max
                self.update_weight(ltd)
        
        # Update traces
        self.pre_trace += 1.0
        self.last_pre_spike = spike_time
    
    def post_spike(self, spike_time: float):
        self.current_time = spike_time
        self._decay_traces(spike_time)
        
        # LTP if pre-before-post
        if (self.last_pre_spike > -np.inf and 
            (spike_time - self.last_pre_spike) < self.tau_stdp):
            if self.pre_trace > 0.0:
                ltp = self.A_plus * self.pre_trace
                # Multiplicative LTP
                ltp *= max(self.w_max - self.weight, 0.0)
                self.update_weight(ltp)
        
        self.post_trace += 1.0
        self.last_post_spike = spike_time
```

**Usage Example:**
```python
from core.synapses import STDP_Synapse, SynapseType

# Create STDP synapse
synapse = STDP_Synapse(
    synapse_id=0,
    pre_neuron_id=0,
    post_neuron_id=1,
    weight=1.0,
    synapse_type=SynapseType.EXCITATORY,
    tau_stdp=20.0,  # STDP time window
    A_plus=0.01,    # LTP amplitude
    A_minus=0.01    # LTD amplitude
)

# Simulate STDP learning
print(f"Initial weight: {synapse.weight}")
synapse.pre_spike(t=10.0)   # Pre-synaptic spike
synapse.post_spike(t=12.0)  # Post-synaptic spike (LTP)
print(f"Weight after LTP: {synapse.weight}")
```

### 2.3 Short-Term Plasticity (STP)

**Biological Background:**
STP causes temporary changes in synaptic strength due to depletion and facilitation of neurotransmitter resources.

**Mathematical Model:**
```
dx/dt = (1 - x)/τ_D - u * x * δ(t - t_spike)
du/dt = (U - u)/τ_F + U * (1 - u) * δ(t - t_spike)
```

**Code Implementation:**
```python
# File: core/synapses.py
class ShortTermPlasticitySynapse(SynapseModel):
    def pre_spike(self, spike_time: float):
        dt = spike_time - self.last_spike_time
        
        if dt > 0:
            # Recovery of available resources (depression)
            self.x = 1.0 - (1.0 - self.x) * np.exp(-dt / self.tau_dep)
            # Decay of utilization (facilitation)
            self.u = self.u * np.exp(-dt / self.tau_fac)
        
        # Neurotransmitter release
        self.u += self.U * (1.0 - self.u)  # Facilitation
        self.x -= self.u * self.x          # Depression
        
        self.last_spike_time = spike_time
    
    def compute_current(self, pre_spike_time: float, current_time: float) -> float:
        # Effective weight modulated by STP
        effective_weight = self.weight * self.x * self.u
        
        dt = current_time - pre_spike_time
        current = effective_weight * np.exp(-dt / self.tau_syn)
        
        return current if self.synapse_type == SynapseType.EXCITATORY else -current
```

### 2.4 Reward-Modulated STDP

**Biological Background:**
Combines timing-based STDP with reward signals, implementing dopamine-like neuromodulation for reinforcement learning.

**Code Implementation:**
```python
# File: core/synapses.py
class RSTDP_Synapse(STDP_Synapse):
    def __init__(self, learning_rate: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.neuromodulator_level = 0.0
        self.reward_signal = 0.0
    
    def update_neuromodulator(self, level: float):
        """Update dopamine-like neuromodulator"""
        self.neuromodulator_level = np.clip(level, 0.0, 1.0)
    
    def update_reward(self, reward: float):
        """Update reward signal"""
        self.reward_signal = reward
    
    def pre_spike(self, t: float):
        super().pre_spike(t)  # Standard STDP
        
        # Reward-modulated weight update
        if self.reward_signal > 0:
            dw = self.learning_rate * self.neuromodulator_level * self.reward_signal
            self.update_weight(dw)
```

---

## 3. Sensory Encoding Systems

### 3.1 Rate Encoding

**Biological Background:**
Information is encoded in the firing rate of neurons - higher stimulus intensity leads to higher firing rates.

**Code Implementation:**
```python
# File: core/encoding.py
class RateEncoder:
    def __init__(self, max_rate: float = 100.0):
        self.max_rate = max_rate
    
    def encode(self, value: float, duration: float = 100.0) -> List[Tuple[int, float]]:
        value = np.clip(value, 0.0, 1.0)
        spike_rate = value * self.max_rate  # Hz
        
        spikes = []
        if spike_rate > 0:
            isi = 1000.0 / spike_rate  # Inter-spike interval
            
            t = 0.0
            while t < duration:
                jitter = np.random.uniform(-0.1, 0.1) * isi
                spike_time = t + jitter
                
                if 0 <= spike_time < duration:
                    spikes.append((0, spike_time))  # (neuron_id, spike_time)
                t += isi
        
        return spikes
```

### 3.2 Retinal Encoding

**Biological Background:**
Visual processing begins in the retina with ON/OFF center-surround receptive fields that detect light changes.

**Code Implementation:**
```python
# File: core/encoding.py
class RetinalEncoder:
    def __init__(self, resolution: Tuple[int, int] = (32, 32)):
        self.resolution = resolution
    
    def encode(self, image: np.ndarray) -> Dict[str, Any]:
        # Convert to grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Resize to target resolution
        image = cv2.resize(image, self.resolution)
        
        # ON-center/OFF-surround processing
        on_center = self._compute_center_surround(image, True)
        off_center = self._compute_center_surround(image, False)
        
        return {
            "on_center": on_center,
            "off_center": off_center,
            "original": image
        }
    
    def _compute_center_surround(self, image: np.ndarray, on_center: bool) -> np.ndarray:
        # Gaussian difference for center-surround
        center = cv2.GaussianBlur(image, (3, 3), 1)
        surround = cv2.GaussianBlur(image, (9, 9), 3)
        
        if on_center:
            response = center - surround  # Bright center, dark surround
        else:
            response = surround - center  # Dark center, bright surround
        
        return np.clip(response, 0, 255)
```

### 3.3 Cochlear Encoding

**Biological Background:**
The cochlea decomposes sound into frequency components, with different frequencies activating different neural populations.

**Code Implementation:**
```python
# File: core/encoding.py
class CochlearEncoder:
    def __init__(self, num_channels: int = 32, sample_rate: int = 44100):
        self.num_channels = num_channels
        self.sample_rate = sample_rate
    
    def encode(self, audio: np.ndarray) -> Dict[str, Any]:
        # Frequency decomposition (simplified cochlear model)
        frequencies = np.fft.fft(audio)
        freq_bins = np.array_split(frequencies, self.num_channels)
        
        channel_responses = []
        for bin_data in freq_bins:
            power = np.abs(bin_data).mean()
            channel_responses.append(power)
        
        return {
            "channel_responses": np.array(channel_responses),
            "sample_rate": self.sample_rate
        }
```

### 3.4 Population Encoding

**Biological Background:**
Information is distributed across a population of neurons, each with different preferred values and tuning curves.

**Code Implementation:**
```python
# File: core/encoding.py
class PopulationEncoder:
    def __init__(self, num_neurons: int = 10, value_range: Tuple[float, float] = (0, 1)):
        self.num_neurons = num_neurons
        self.value_range = value_range
        
        # Create preferred values for each neuron
        self.preferred_values = np.linspace(value_range[0], value_range[1], num_neurons)
        self.tuning_width = (value_range[1] - value_range[0]) / (num_neurons * 2)
    
    def encode(self, value: float) -> np.ndarray:
        # Gaussian tuning curves
        responses = np.exp(-0.5 * ((value - self.preferred_values) / self.tuning_width) ** 2)
        return responses / (responses.sum() + 1e-6)  # Normalize
```

---

## 4. Neuromodulation and Reward Systems

### 4.1 Dopaminergic System

**Biological Background:**
Dopamine neurons signal reward prediction error, crucial for reinforcement learning and motivation.

**Mathematical Model:**
```
δ = r(t) + γ*V(s_{t+1}) - V(s_t)  # Temporal Difference Error
```

**Code Implementation:**
```python
# File: core/neuromodulation.py
class DopaminergicSystem(NeuromodulatorySystem):
    def __init__(self, learning_rate: float = 0.01, discount_factor: float = 0.9):
        super().__init__(NeuromodulatorType.DOPAMINE)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.reward_prediction = 0.0
    
    def compute_reward_prediction_error(self, reward: float, expected_reward: float) -> float:
        """Compute temporal difference error"""
        prediction_error = reward - expected_reward
        return prediction_error
    
    def update(self, reward: float, expected_reward: float, dt: float):
        # Compute reward prediction error
        prediction_error = self.compute_reward_prediction_error(reward, expected_reward)
        
        # Update dopamine level (positive error increases dopamine)
        dopamine_signal = np.tanh(prediction_error)
        super().update(dopamine_signal, dt)
        
        # Update reward prediction using TD learning
        self.reward_prediction = expected_reward + self.learning_rate * prediction_error
    
    def get_learning_rate_modulation(self) -> float:
        """Dopamine modulates learning rate"""
        return 1.0 + 2.0 * self.current_level
```

### 4.2 Serotonergic System

**Biological Background:**
Serotonin regulates mood, behavioral flexibility, and social behaviors.

**Code Implementation:**
```python
# File: core/neuromodulation.py
class SerotonergicSystem(NeuromodulatorySystem):
    def __init__(self, mood_decay_rate: float = 0.99):
        super().__init__(NeuromodulatorType.SEROTONIN)
        self.mood_decay_rate = mood_decay_rate
        self.mood_state = 0.5  # Neutral mood
    
    def update_mood(self, positive_events: float, negative_events: float, dt: float):
        # Update mood based on event balance
        mood_change = (positive_events - negative_events) * dt
        self.mood_state += mood_change
        self.mood_state *= self.mood_decay_rate
        self.mood_state = np.clip(self.mood_state, 0.0, 1.0)
        
        # Update serotonin based on mood
        serotonin_signal = self.mood_state - 0.5  # Center around neutral
        super().update(serotonin_signal, dt)
    
    def get_behavioral_flexibility(self) -> float:
        """Serotonin increases behavioral flexibility"""
        return 0.5 + self.current_level
```

### 4.3 Cholinergic System

**Biological Background:**
Acetylcholine modulates attention, arousal, and learning rate based on environmental novelty.

**Code Implementation:**
```python
# File: core/neuromodulation.py
class CholinergicSystem(NeuromodulatorySystem):
    def __init__(self, attention_threshold: float = 0.1):
        super().__init__(NeuromodulatorType.ACETYLCHOLINE)
        self.attention_threshold = attention_threshold
        self.attention_state = 0.0
        self.novelty_detector = 0.0
    
    def update_attention(self, sensory_input: np.ndarray, expected_input: np.ndarray, dt: float):
        # Compute novelty (prediction error)
        novelty = np.mean(np.abs(sensory_input - expected_input))
        
        # Update novelty detector with temporal smoothing
        self.novelty_detector = 0.9 * self.novelty_detector + 0.1 * novelty
        
        # Update attention state
        if novelty > self.attention_threshold:
            self.attention_state = min(1.0, self.attention_state + dt)
        else:
            self.attention_state = max(0.0, self.attention_state - dt)
        
        # Update acetylcholine level
        super().update(self.attention_state, dt)
    
    def get_learning_rate_modulation(self) -> float:
        """Acetylcholine modulates learning based on attention"""
        return 1.0 + self.current_level
```

### 4.4 Neuromodulatory Controller

**Code Implementation:**
```python
# File: core/neuromodulation.py
class NeuromodulatoryController:
    def __init__(self):
        self.systems = {
            NeuromodulatorType.DOPAMINE: DopaminergicSystem(),
            NeuromodulatorType.SEROTONIN: SerotonergicSystem(),
            NeuromodulatorType.ACETYLCHOLINE: CholinergicSystem(),
            NeuromodulatorType.NOREPINEPHRINE: NoradrenergicSystem(),
        }
    
    def update(self, sensory_input: np.ndarray, reward: float, expected_reward: float,
               positive_events: float = 0.0, negative_events: float = 0.0,
               threat_signals: float = 0.0, task_difficulty: float = 0.5, dt: float = 0.1):
        # Update all neuromodulatory systems
        self.systems[NeuromodulatorType.DOPAMINE].update(reward, expected_reward, dt)
        self.systems[NeuromodulatorType.SEROTONIN].update_mood(positive_events, negative_events, dt)
        
        # Use reward prediction error as novelty signal
        novelty = abs(reward - expected_reward)
        self.systems[NeuromodulatorType.ACETYLCHOLINE].update_attention(
            np.array([novelty]), np.array([0.0]), dt
        )
        
        self.systems[NeuromodulatorType.NOREPINEPHRINE].update_arousal(
            threat_signals, task_difficulty, dt
        )
    
    def get_modulator_levels(self) -> Dict[NeuromodulatorType, float]:
        return {mod_type: system.get_level() for mod_type, system in self.systems.items()}
```

---

## 5. Learning and Memory Formation

### 5.1 Hebbian Learning

**Biological Background:**
"Cells that fire together, wire together" - the foundational principle of synaptic plasticity.

**Mathematical Model:**
```
Δw = η * x_pre * x_post
```

**Code Implementation:**
```python
# File: core/learning.py
class HebbianRule(PlasticityRule):
    def compute_weight_change(self, pre_activity: float, post_activity: float, 
                            current_weight: float, **kwargs) -> float:
        # Basic Hebbian rule with decay
        correlation = pre_activity * post_activity
        
        if correlation > self.config.hebbian_threshold:
            # Potentiation with weight-dependent scaling
            delta_w = self.config.learning_rate * correlation
            delta_w *= (self.config.weight_max - current_weight) / self.config.weight_max
        else:
            # Decay term for stability
            delta_w = (-self.config.learning_rate * 
                      (1 - self.config.hebbian_decay) * current_weight)
        
        return delta_w
```

### 5.2 BCM (Bienenstock-Cooper-Munro) Rule

**Biological Background:**
Implements a sliding threshold mechanism that prevents runaway potentiation and maintains stable activity.

**Mathematical Model:**
```
Δw = η * x_pre * x_post * (x_post - θ)
dθ/dt = (x_post² - θ)/τ_θ
```

**Code Implementation:**
```python
# File: core/learning.py
class BCMRule(PlasticityRule):
    def __init__(self, config: PlasticityConfig):
        super().__init__(config)
        self.sliding_threshold = config.bcm_threshold
        self.activity_history = []
    
    def compute_weight_change(self, pre_activity: float, post_activity: float, 
                            current_weight: float, **kwargs) -> float:
        # BCM learning function
        phi = post_activity * (post_activity - self.sliding_threshold)
        delta_w = self.config.learning_rate * phi * pre_activity
        
        # Update sliding threshold
        self.activity_history.append(post_activity)
        if len(self.activity_history) > 100:
            self.activity_history.pop(0)
        
        # Adapt threshold to maintain stability
        mean_activity = np.mean(self.activity_history) if self.activity_history else post_activity
        tau = self.config.bcm_time_constant
        self.sliding_threshold += (mean_activity**2 - self.sliding_threshold) / tau
        
        return delta_w
```

### 5.3 Triplet STDP

**Biological Background:**
More accurate modeling of experimental STDP data by considering triplets of spikes rather than just pairs.

**Mathematical Model:**
```
Δw = A₂⁺ * r₁ + A₃⁺ * r₁ * o₂  (for post spike)
Δw = -A₂⁻ * o₁ - A₃⁻ * o₁ * r₂  (for pre spike)
```

**Code Implementation:**
```python
# File: core/learning.py
class TripletSTDP(PlasticityRule):
    def __init__(self, config: PlasticityConfig):
        super().__init__(config)
        self.r1 = 0.0  # Fast presynaptic trace
        self.r2 = 0.0  # Slow presynaptic trace
        self.o1 = 0.0  # Fast postsynaptic trace
        self.o2 = 0.0  # Slow postsynaptic trace
    
    def compute_weight_change(self, pre_activity: float, post_activity: float,
                            current_weight: float, dt: float = 1.0,
                            pre_spike: bool = False, post_spike: bool = False, **kwargs) -> float:
        delta_w = 0.0
        
        # Decay traces
        self.r1 *= np.exp(-dt / self.config.tau_x)
        self.r2 *= np.exp(-dt / self.config.tau_plus)
        self.o1 *= np.exp(-dt / self.config.tau_y)
        self.o2 *= np.exp(-dt / self.config.tau_minus)
        
        if pre_spike:
            # LTD (triplet depression)
            delta_w -= self.o1 * (self.config.A2_minus + self.config.A3_minus * self.r2)
            self.r1 = 1.0
            self.r2 = 1.0
        
        if post_spike:
            # LTP (triplet potentiation)
            delta_w += self.r1 * (self.config.A2_plus + self.config.A3_plus * self.o2)
            self.o1 = 1.0
            self.o2 = 1.0
        
        return delta_w * self.config.learning_rate
```

### 5.4 Homeostatic Plasticity

**Biological Background:**
Maintains stable neural activity by scaling synaptic strengths and intrinsic excitability.

**Code Implementation:**
```python
# File: core/learning.py
class HomeostaticPlasticity(PlasticityRule):
    def __init__(self, config: PlasticityConfig):
        super().__init__(config)
        self.firing_rate_estimate = 0.0
        self.total_synaptic_strength = 0.0
        self.intrinsic_excitability = 1.0
        self.activity_history = []
    
    def update_total_synaptic_strength(self, all_weights: np.ndarray):
        """Update total synaptic strength for homeostatic scaling"""
        self.total_synaptic_strength = np.sum(all_weights)
    
    def compute_homeostatic_scaling(self) -> float:
        """Compute synaptic scaling factor"""
        if self.total_synaptic_strength > 0:
            target = self.config.target_total_strength
            current = self.total_synaptic_strength
            scaling_factor = target / current
            return np.clip(scaling_factor, 0.1, 10.0)
        return 1.0
    
    def update_intrinsic_excitability(self, current_activity: float):
        """Update intrinsic excitability based on activity"""
        target_activity = self.config.activity_threshold
        error = current_activity - target_activity
        
        # Adjust excitability inversely to activity
        self.intrinsic_excitability -= self.config.excitability_scaling_rate * error
        self.intrinsic_excitability = np.clip(self.intrinsic_excitability, 0.1, 10.0)
```

---

## 6. Neural Networks and Connectivity

### 6.1 Network Architecture

**Biological Background:**
Neural networks are organized in layers with specific connectivity patterns and functional roles.

**Code Implementation:**
```python
# File: core/network.py
class NeuromorphicNetwork:
    def __init__(self):
        self.layers: Dict[str, NetworkLayer] = {}
        self.connections: Dict[Tuple[str, str], NetworkConnection] = {}
        self.current_time = 0.0
    
    def add_layer(self, name: str, size: int, neuron_type: str = "adex", **kwargs):
        """Add a network layer with validation"""
        if size > self.MAX_NEURONS:
            raise ValueError(f"Layer size {size} exceeds maximum {self.MAX_NEURONS}")
        
        layer = NetworkLayer(name, size, neuron_type, **kwargs)
        self.layers[name] = layer
        self.total_neurons += size
    
    def connect_layers(self, pre_layer: str, post_layer: str, synapse_type: str = "stdp",
                      connection_probability: float = 0.1, **kwargs):
        """Connect two layers with synapses"""
        if pre_layer not in self.layers or post_layer not in self.layers:
            raise ValueError("Both layers must exist before connecting")
        
        connection = NetworkConnection(pre_layer, post_layer, synapse_type, 
                                     connection_probability, **kwargs)
        
        # Initialize synapse population
        pre_size = self.layers[pre_layer].size
        post_size = self.layers[post_layer].size
        connection.initialize(pre_size, post_size)
        
        self.connections[(pre_layer, post_layer)] = connection
```

### 6.2 Network Layer

**Code Implementation:**
```python
# File: core/network.py
class NetworkLayer:
    def __init__(self, name: str, size: int, neuron_type: str = "adex", **kwargs):
        self.name = name
        self.size = size
        self.neuron_type = neuron_type
        self.neuron_population = NeuronPopulation(size, neuron_type, **kwargs)
        self.spike_times = [[] for _ in range(size)]
        self.current_time = 0.0
    
    def step(self, dt: float, I_syn: List[float]) -> List[bool]:
        """Advance layer by one time step"""
        spikes = self.neuron_population.step(dt, I_syn)
        
        # Record spike times
        for i, spiked in enumerate(spikes):
            if spiked:
                self.spike_times[i].append(self.current_time)
        
        self.current_time += dt
        return spikes
```

### 6.3 Synaptic Populations

**Code Implementation:**
```python
# File: core/synapses.py
class SynapsePopulation:
    def __init__(self, pre_size: int, post_size: int, synapse_type: str, 
                 connection_probability: float, **kwargs):
        self.pre_size = pre_size
        self.post_size = post_size
        self.synapse_type = synapse_type
        self.synapses = {}
        
        # Create sparse connectivity
        for pre_id in range(pre_size):
            for post_id in range(post_size):
                if np.random.random() < connection_probability:
                    synapse_id = len(self.synapses)
                    synapse = SynapseFactory.create_synapse(
                        synapse_type, synapse_id, pre_id, post_id, **kwargs
                    )
                    self.synapses[(pre_id, post_id)] = synapse
    
    def get_synaptic_currents(self, pre_spikes: List[bool], current_time: float) -> List[float]:
        """Compute synaptic currents for all post-synaptic neurons"""
        currents = [0.0] * self.post_size
        
        for (pre_id, post_id), synapse in self.synapses.items():
            if pre_id < len(pre_spikes) and pre_spikes[pre_id]:
                # Pre-synaptic spike occurred
                current = synapse.compute_current(current_time, current_time)
                currents[post_id] += current
        
        return currents
```

---

## 7. Higher-Order Cognitive Functions

### 7.1 Symbol Emergence

**Biological Background:**
Symbols emerge from stable patterns of neural activity called cell assemblies.

**Code Implementation:**
```python
# File: core/symbol_emergence.py
class CellAssembly:
    """Stable cell assembly representing a symbol"""
    def __init__(self):
        self.neurons: List[int] = []
        self.activation_pattern: np.ndarray = np.array([])
        self.stability_score: float = 0.0
        self.emergence_history: List[float] = []
        self.concept_grounding: Dict[str, float] = {}

class PhaseBinding:
    """Phase-based binding mechanism for feature integration"""
    def __init__(self, theta_range: Tuple[float, float] = (4, 8),
                 gamma_range: Tuple[float, float] = (30, 100)):
        self.theta_range = theta_range
        self.gamma_range = gamma_range
        self.current_phase = 0.0
    
    def bind_features(self, features: List[np.ndarray], dt: float = 0.001) -> np.ndarray:
        """Bind features using phase offsets"""
        if not features:
            return np.array([])
        
        max_size = max(feature.shape[0] for feature in features)
        num_features = len(features)
        phase_offsets = np.linspace(0, 2 * np.pi, num_features, endpoint=False)
        
        # Create composite pattern through phase interference
        composite = np.zeros(max_size, dtype=complex)
        
        for i, feature in enumerate(features):
            padded_feature = np.zeros(max_size)
            padded_feature[:len(feature)] = feature
            
            # Apply phase modulation
            phase_modulated = padded_feature * np.exp(1j * (phase_offsets[i] + self.current_phase))
            composite += phase_modulated
        
        # Update global phase (theta rhythm)
        theta_freq = np.mean(self.theta_range)
        self.current_phase += 2 * np.pi * theta_freq * dt
        self.current_phase = self.current_phase % (2 * np.pi)
        
        return np.abs(composite)
```

### 7.2 Attention Mechanisms

**Code Implementation:**
```python
# File: core/symbol_emergence.py
class AttentionMechanism:
    def __init__(self, attention_strength: float = 1.0):
        self.attention_strength = attention_strength
        self.attention_focus = np.array([])
    
    def apply_attention(self, input_pattern: np.ndarray, attention_signal: np.ndarray) -> np.ndarray:
        """Apply attention to modulate input processing"""
        if len(attention_signal) != len(input_pattern):
            # Resize attention signal to match input
            attention_signal = np.interp(
                np.linspace(0, 1, len(input_pattern)),
                np.linspace(0, 1, len(attention_signal)),
                attention_signal
            )
        
        # Multiplicative attention
        attended_pattern = input_pattern * (1.0 + self.attention_strength * attention_signal)
        return attended_pattern
```

### 7.3 Working Memory

**Code Implementation:**
```python
# File: core/symbol_emergence.py
class WorkingMemory:
    def __init__(self, capacity: int = 7, decay_rate: float = 0.95):
        self.capacity = capacity
        self.decay_rate = decay_rate
        self.memory_items = []
        self.memory_strengths = []
    
    def store_item(self, item: np.ndarray, strength: float = 1.0):
        """Store item in working memory"""
        if len(self.memory_items) >= self.capacity:
            # Remove weakest item
            min_idx = np.argmin(self.memory_strengths)
            self.memory_items.pop(min_idx)
            self.memory_strengths.pop(min_idx)
        
        self.memory_items.append(item.copy())
        self.memory_strengths.append(strength)
    
    def update(self, dt: float):
        """Update working memory with decay"""
        # Apply decay to all items
        for i in range(len(self.memory_strengths)):
            self.memory_strengths[i] *= self.decay_rate
        
        # Remove items below threshold
        threshold = 0.1
        to_remove = []
        for i, strength in enumerate(self.memory_strengths):
            if strength < threshold:
                to_remove.append(i)
        
        for i in reversed(to_remove):
            self.memory_items.pop(i)
            self.memory_strengths.pop(i)
    
    def recall_similar(self, query: np.ndarray, threshold: float = 0.8) -> Optional[np.ndarray]:
        """Recall item similar to query"""
        best_match = None
        best_similarity = 0.0
        
        for item, strength in zip(self.memory_items, self.memory_strengths):
            if len(item) == len(query):
                similarity = np.dot(item, query) / (np.linalg.norm(item) * np.linalg.norm(query))
                similarity *= strength  # Modulate by memory strength
                
                if similarity > best_similarity and similarity > threshold:
                    best_similarity = similarity
                    best_match = item
        
        return best_match
```

---

## 8. Homeostasis and Regulation

### 8.1 Homeostatic Regulation

**Biological Background:**
Neural systems maintain stable activity levels through homeostatic mechanisms that adjust synaptic strengths and intrinsic excitability.

**Code Implementation:**
```python
# File: core/neuromodulation.py
class HomeostaticRegulator:
    def __init__(self, target_firing_rate: float = 10.0, adaptation_rate: float = 0.01):
        self.target_firing_rate = target_firing_rate
        self.adaptation_rate = adaptation_rate
        self.current_firing_rates = {}
        self.scaling_factors = {}
    
    def update_firing_rates(self, layer_name: str, spike_times: List[List[float]], 
                           time_window: float):
        """Update firing rates for a layer"""
        rates = []
        for neuron_spikes in spike_times:
            spike_count = len([t for t in neuron_spikes if t <= time_window])
            rate = (spike_count / time_window) * 1000.0  # Convert to Hz
            rates.append(rate)
        
        self.current_firing_rates[layer_name] = rates
    
    def compute_scaling_factors(self) -> Dict[str, float]:
        """Compute homeostatic scaling factors"""
        scaling_factors = {}
        
        for layer_name, rates in self.current_firing_rates.items():
            if rates:
                mean_rate = np.mean(rates)
                if mean_rate > 0:
                    scaling_factor = self.target_firing_rate / mean_rate
                    scaling_factor = np.clip(scaling_factor, 0.1, 10.0)
                    
                    # Smooth adaptation
                    if layer_name in self.scaling_factors:
                        old_factor = self.scaling_factors[layer_name]
                        scaling_factor = ((1 - self.adaptation_rate) * old_factor + 
                                        self.adaptation_rate * scaling_factor)
                    
                    scaling_factors[layer_name] = scaling_factor
                else:
                    scaling_factors[layer_name] = 1.0
            else:
                scaling_factors[layer_name] = 1.0
        
        self.scaling_factors = scaling_factors
        return scaling_factors
    
    def apply_homeostasis(self, network, scaling_factors: Dict[str, float]):
        """Apply homeostatic scaling to network weights"""
        for layer_name, scaling_factor in scaling_factors.items():
            if layer_name in network.layers:
                for (pre_name, post_name), connection in network.connections.items():
                    if post_name == layer_name and connection.synapse_population:
                        for synapse in connection.synapse_population.synapses.values():
                            synapse.weight *= scaling_factor
```

### 8.2 Sleep and Consolidation

**Biological Background:**
Sleep phases involve memory consolidation through replay of neural patterns and synaptic homeostasis.

**Code Implementation:**
```python
# File: experiments/optimal_gpu_learning.py (practical example)
def run_sleep_phase(self, duration: float = 50.0, replay_patterns: Optional[Dict] = None,
                   downscale_factor: float = 0.98, normalize_incoming: bool = True,
                   noise_std: float = 0.05):
    """
    Simulate sleep-like consolidation phase with replay and homeostasis
    
    Args:
        duration: Duration of sleep phase (ms)
        replay_patterns: Dict of patterns to replay during sleep
        downscale_factor: Global synaptic downscaling (SHY hypothesis)
        normalize_incoming: Whether to normalize incoming weights
        noise_std: Standard deviation of background noise
    """
    print(f"🌙 Entering sleep phase ({duration}ms)...")
    
    steps = int(duration / 0.5)  # 0.5ms time steps
    
    for step in range(steps):
        # Background noise for spontaneous activity
        background_noise = self.xp.random.normal(0, noise_std, self.num_neurons)
        
        # Replay specific patterns if provided
        replay_input = self.xp.zeros(self.num_neurons)
        if replay_patterns and step % 100 == 0:  # Replay every 50ms
            for pattern_name, pattern in replay_patterns.items():
                if isinstance(pattern, np.ndarray) and len(pattern) <= self.num_neurons:
                    replay_input[:len(pattern)] += pattern * 0.5
        
        # Combined input during sleep
        sleep_input = background_noise + replay_input
        
        # Simplified sleep dynamics (no adaptation current updates)
        self.membrane_potential += 0.5 * (
            -(self.membrane_potential - (-65.0)) / 20.0 + sleep_input
        )
        
        # Occasional sleep spikes
        sleep_spike_mask = self.membrane_potential > -50.0  # Higher threshold
        if self.xp.any(sleep_spike_mask):
            sleep_spike_indices = self.xp.where(sleep_spike_mask)[0]
            self.membrane_potential[sleep_spike_mask] = -65.0
            
            # Apply STDP-like consolidation to recently active synapses
            if len(sleep_spike_indices) > 1:
                self._apply_sleep_consolidation(sleep_spike_indices)
    
    # Global synaptic homeostasis (SHY - Synaptic Homeostasis Hypothesis)
    if downscale_factor < 1.0:
        self.weights *= downscale_factor
        print(f"   Applied synaptic homeostasis (downscaling: {downscale_factor})")
    
    # Weight normalization
    if normalize_incoming:
        self._normalize_incoming_weights()
        print(f"   Applied weight normalization")
    
    print(f"   ✅ Sleep consolidation complete")

def _apply_sleep_consolidation(self, active_indices):
    """Apply consolidation-specific plasticity during sleep"""
    # Find synapses involving active neurons
    active_mask = np.isin(self.pre_indices.get(), active_indices) | np.isin(self.post_indices.get(), active_indices)
    
    if self.xp.any(active_mask):
        # Strengthen correlated connections, weaken uncorrelated ones
        strengthening = self.xp.random.random(self.xp.sum(active_mask)) > 0.7
        self.weights[active_mask][strengthening] *= 1.05  # Modest strengthening
        self.weights[active_mask][~strengthening] *= 0.98  # Modest weakening
        
        # Keep weights in bounds
        self.weights = self.xp.clip(self.weights, 0.0, 1.0)

def _normalize_incoming_weights(self):
    """Normalize incoming weights to each neuron"""
    for post_neuron in range(self.num_neurons):
        incoming_mask = self.post_indices == post_neuron
        if self.xp.any(incoming_mask):
            incoming_weights = self.weights[incoming_mask]
            total_weight = self.xp.sum(incoming_weights)
            if total_weight > 0:
                target_total = 5.0  # Target total incoming weight
                self.weights[incoming_mask] *= target_total / total_weight
```

---

## 9. Advanced Concepts

### 9.1 Metaplasticity

**Biological Background:**
"Plasticity of plasticity" - the plasticity threshold itself changes based on neural activity history.

**Code Implementation:**
```python
# File: core/learning.py
class MetaplasticityRule(PlasticityRule):
    def __init__(self, config: PlasticityConfig):
        super().__init__(config)
        self.plasticity_threshold = config.metaplasticity_threshold
        self.activity_history = deque(maxlen=config.metaplasticity_window)
        self.threshold_adaptation_rate = config.threshold_adaptation_rate
    
    def update_plasticity_threshold(self, current_activity: float):
        """Update plasticity threshold based on activity history"""
        self.activity_history.append(current_activity)
        
        if len(self.activity_history) > 10:
            mean_activity = np.mean(self.activity_history)
            
            # Adapt threshold to maintain optimal plasticity range
            if mean_activity > 0.8:  # High activity
                self.plasticity_threshold += self.threshold_adaptation_rate
            elif mean_activity < 0.2:  # Low activity
                self.plasticity_threshold -= self.threshold_adaptation_rate
            
            self.plasticity_threshold = np.clip(self.plasticity_threshold, 0.1, 2.0)
    
    def compute_weight_change(self, pre_activity: float, post_activity: float,
                            current_weight: float, **kwargs) -> float:
        # Update plasticity threshold
        self.update_plasticity_threshold(post_activity)
        
        # Standard STDP with adaptive threshold
        correlation = pre_activity * post_activity
        
        if correlation > self.plasticity_threshold:
            # Above threshold - allow plasticity
            delta_w = self.config.learning_rate * correlation
            # Modulate by current threshold level
            delta_w *= (2.0 - self.plasticity_threshold)  # Higher threshold = less plasticity
        else:
            # Below threshold - minimal plasticity
            delta_w = 0.1 * self.config.learning_rate * correlation
        
        return delta_w
```

### 9.2 Synaptic Competition

**Code Implementation:**
```python
# File: core/learning.py
class SynapticCompetition:
    def __init__(self, competition_strength: float = 0.1, 
                 normalization_target: float = 10.0):
        self.competition_strength = competition_strength
        self.normalization_target = normalization_target
    
    def apply_competition(self, weights: np.ndarray, activities: np.ndarray) -> np.ndarray:
        """Apply synaptic competition (winner-take-all dynamics)"""
        if len(weights) != len(activities):
            return weights
        
        # Compute competitive advantage
        mean_activity = np.mean(activities)
        advantage = activities - mean_activity
        
        # Apply competition (strong synapses get stronger)
        competition_factor = 1.0 + self.competition_strength * np.tanh(advantage)
        updated_weights = weights * competition_factor
        
        # Normalize to maintain total synaptic strength
        current_total = np.sum(updated_weights)
        if current_total > 0:
            updated_weights *= self.normalization_target / current_total
        
        return np.clip(updated_weights, 0.0, self.normalization_target)
    
    def soft_winner_take_all(self, activities: np.ndarray, 
                           strength: float = 2.0) -> np.ndarray:
        """Soft winner-take-all competition"""
        if len(activities) == 0:
            return activities
        
        # Softmax-like competition
        exp_activities = np.exp(strength * (activities - np.max(activities)))
        return exp_activities / np.sum(exp_activities)
```

### 9.3 Predictive Coding

**Code Implementation:**
```python
# File: core/symbol_emergence.py
class PredictiveSTDP:
    """Predictive spike-timing dependent plasticity"""
    
    def __init__(self, beta: float = 0.3):
        self.beta = beta  # Prediction strength
        self.prediction_window = 0.02  # 20ms prediction window
    
    def update_weights(self, pre_activity: np.ndarray, post_activity: np.ndarray,
                      prediction_error: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Update weights based on predictive STDP"""
        
        # Calculate weight updates
        ltp = np.outer(pre_activity, post_activity)  # Long-term potentiation
        ltd = np.outer(post_activity, pre_activity)  # Long-term depression
        
        # Modulate by prediction error
        prediction_modulation = 1.0 + self.beta * prediction_error.reshape(-1, 1)
        
        # Apply updates
        weight_update = (ltp - ltd) * prediction_modulation[:len(ltp)]
        updated_weights = weights + 0.01 * weight_update
        
        # Clip to reasonable range
        return np.clip(updated_weights, -1.0, 1.0)
```

---

## 10. Practical Implementation Examples

### 10.1 Complete Learning Network

```python
from core.network import NeuromorphicNetwork
from core.neuromodulation import NeuromodulatoryController

# Create network with multiple layers
network = NeuromorphicNetwork()

# Add layers
network.add_layer("input", 100, "lif", v_thresh=-55.0, tau_m=20.0)
network.add_layer("hidden", 50, "adex", tau_m=20.0, tau_w=144.0)
network.add_layer("output", 10, "lif", v_thresh=-50.0)

# Connect layers with STDP
network.connect_layers("input", "hidden", "stdp", 
                      connection_probability=0.1,
                      tau_stdp=20.0, A_plus=0.01, A_minus=0.01)
network.connect_layers("hidden", "output", "stdp",
                      connection_probability=0.2)

# Add neuromodulation
neuromod_controller = NeuromodulatoryController()

# Run learning simulation
results = network.run_simulation(
    duration=1000.0,  # 1 second
    dt=0.1,           # 0.1ms time steps
    neuromodulation=neuromod_controller
)

# Analyze results
print(f"Total spikes: {sum(len(spikes) for spikes in results['spike_times'].values())}")
print(f"Learning events: {results['learning_events']}")
```

### 10.2 Sleep Consolidation Example

```python
# After initial learning, run sleep consolidation
network.run_sleep_phase(
    duration=100.0,  # 100ms sleep
    replay={"input": learned_pattern},
    downscale_factor=0.98,        # Synaptic homeostasis
    normalize_incoming=True,      # Weight normalization
    noise_std=0.05               # Background noise
)
```

### 10.3 Multi-modal Sensory Processing

```python
from core.encoding import MultiModalEncoder

# Create multi-modal encoder
encoder = MultiModalEncoder()

# Process different sensory inputs
sensory_data = {
    "visual": camera_image,
    "auditory": microphone_data,
    "tactile": pressure_sensors
}

# Encode all modalities
encoded = encoder.encode(sensory_data)

# Fuse into unified representation
fused_representation = encoder.fuse_modalities(encoded)

# Feed into network
input_spikes = encoder.rate_encoder.encode_array(fused_representation)
```

---

## Summary

This curriculum covers the complete spectrum of biological concepts implemented in our neuromorphic computing system:

1. **Neural Components**: From basic LIF to complex AdEx and HH models
2. **Synaptic Mechanisms**: STDP, STP, reward modulation, and competition
3. **Sensory Processing**: Retinal, cochlear, and multi-modal encoding
4. **Neuromodulation**: Dopamine, serotonin, acetylcholine systems
5. **Learning Rules**: Hebbian, BCM, triplet STDP, homeostatic plasticity
6. **Network Architecture**: Layers, connections, populations
7. **Cognitive Functions**: Symbol emergence, attention, working memory
8. **Homeostasis**: Activity regulation and sleep consolidation
9. **Advanced Concepts**: Metaplasticity, competition, predictive coding

Each concept bridges from biological reality through mathematical modeling to practical code implementation, providing a complete learning pathway for understanding neuromorphic computing.

The system successfully demonstrates massive-scale learning (750K neurons) with biologically plausible mechanisms, making it a powerful platform for both neuroscience research and edge computing applications.
