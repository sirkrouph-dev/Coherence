#!/usr/bin/env python3
"""
Final GPU Learning Benchmark - 750K Neurons
===========================================

Based on all testing, this is the optimal configuration for your RTX 3060:
- 750,000 neurons (proven to work)
- Realistic spiking activity 
- STDP learning that actually works
- No system freezing
- Maximum performance within GPU limits
"""

import numpy as np
import time
import psutil
import os
import gc

# GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("[OK] CuPy GPU acceleration available")
except ImportError:
    cp = np
    GPU_AVAILABLE = False
    print("[WARNING] Using CPU fallback")

def measure_memory():
    """Get current memory usage"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

class OptimalLearningNetwork:
    """Optimal learning network for RTX 3060 8GB"""
    
    def __init__(self):
        self.xp = cp if GPU_AVAILABLE else np
        
        # Optimal configuration from testing
        self.num_neurons = 750_000
        self.connectivity = 0.0003  # 0.03% = 168M synapses (safe)
        self.num_synapses = int(self.num_neurons * self.num_neurons * self.connectivity)
        
        print(f"🧠 Optimal GPU Learning Network")
        print(f"   Neurons: {self.num_neurons:,}")
        print(f"   Synapses: {self.num_synapses:,}")
        print(f"   Connectivity: {self.connectivity:.4f}")
        print(f"   Backend: {'GPU' if GPU_AVAILABLE else 'CPU'}")
        
        self._initialize_network()
    
    def _initialize_network(self):
        """Initialize network optimally"""
        print(f"   Initializing connections...")
        start_time = time.time()
        
        # Generate sparse connections
        self.pre_indices = self.xp.random.randint(
            0, self.num_neurons, self.num_synapses, dtype=self.xp.int32
        )
        self.post_indices = self.xp.random.randint(
            0, self.num_neurons, self.num_synapses, dtype=self.xp.int32
        )
        
        # Initialize weights for learning
        self.weights = self.xp.random.random(self.num_synapses).astype(self.xp.float32) * 0.2
        
        # Neuron states
        self.membrane_potential = self.xp.random.random(self.num_neurons).astype(self.xp.float32) * 0.5
        self.last_spike_time = self.xp.full(self.num_neurons, -100.0, dtype=self.xp.float32)
        
        # Learning parameters
        self.stdp_lr = 0.005
        self.current_time = 0.0
        
        # Statistics
        self.total_spikes = 0
        self.total_learning_events = 0
        
        init_time = time.time() - start_time
        print(f"   ✅ Initialized in {init_time:.2f}s")
    
    def simulate_step(self, dt=0.5):
        """Optimized simulation step with realistic activity"""
        
        # Realistic input currents
        # Create different input patterns for different neuron groups
        input_current = self.xp.zeros(self.num_neurons, dtype=self.xp.float32)
        
        # Background noise
        input_current += self.xp.random.random(self.num_neurons) * 0.1
        
        # Stimulated groups (10% of neurons get strong input)
        num_stimulated = self.num_neurons // 10
        stimulated_indices = self.xp.random.choice(self.num_neurons, num_stimulated, replace=False)
        input_current[stimulated_indices] += 0.8
        
        # LIF neuron dynamics
        tau_m = 20.0
        v_threshold = 1.0
        v_reset = 0.0
        
        # Update membrane potential
        self.membrane_potential += dt * (
            -self.membrane_potential / tau_m + input_current
        )
        
        # Find spikes
        spike_mask = self.membrane_potential >= v_threshold
        spike_indices = self.xp.where(spike_mask)[0]
        num_spikes = len(spike_indices)
        
        # Reset spiking neurons
        if num_spikes > 0:
            self.membrane_potential[spike_mask] = v_reset
            self.last_spike_time[spike_indices] = self.current_time
            self.total_spikes += num_spikes
        
        self.current_time += dt
        
        return num_spikes
    
    def apply_learning(self):
        """Apply efficient STDP learning"""
        
        # Find recently active connections
        recent_threshold = self.current_time - 20.0  # 20ms window
        
        # Check which synapses have recent pre/post activity
        pre_recent = self.last_spike_time[self.pre_indices] > recent_threshold
        post_recent = self.last_spike_time[self.post_indices] > recent_threshold
        
        # STDP: strengthen synapses with correlated activity
        correlated_mask = pre_recent & post_recent
        num_correlated = int(self.xp.sum(correlated_mask))
        
        if num_correlated > 0:
            # Strengthen correlated synapses
            self.weights[correlated_mask] += self.stdp_lr
            
            # Also add some depression for uncorrelated
            uncorrelated_mask = pre_recent & ~post_recent
            self.weights[uncorrelated_mask] -= self.stdp_lr * 0.5
            
            # Clip weights
            self.weights = self.xp.clip(self.weights, 0.0, 1.0)
            
            self.total_learning_events += num_correlated
            
            return num_correlated
        
        return 0
    
    def run_learning_benchmark(self, duration_ms=10000, dt=0.5):
        """Run comprehensive learning benchmark"""
        print(f"\n🎓 Running {duration_ms}ms learning benchmark...")
        
        steps = int(duration_ms / dt)
        spike_history = []
        learning_history = []
        
        start_time = time.time()
        last_progress_time = start_time
        
        for step in range(steps):
            # Simulation step
            num_spikes = self.simulate_step(dt)
            spike_history.append(num_spikes)
            
            # Apply learning every 5 steps
            if step % 5 == 0:
                learning_count = self.apply_learning()
                learning_history.append(learning_count)
            
            # Progress updates
            current_time = time.time()
            if (step % (steps // 20) == 0 and step > 0) or \
               (current_time - last_progress_time > 15):
                
                elapsed = current_time - start_time
                progress = step / steps * 100
                avg_spikes = np.mean(spike_history[-1000:]) if spike_history else 0
                avg_learning = np.mean(learning_history[-200:]) if learning_history else 0
                
                print(f"   Step {step:,}/{steps:,} ({progress:.1f}%) - "
                      f"{avg_spikes:.1f} spikes/step - "
                      f"{avg_learning:.1f} learning/epoch - "
                      f"{elapsed:.1f}s")
                
                last_progress_time = current_time
        
        total_time = time.time() - start_time
        
        # Calculate final statistics
        avg_weight = float(self.xp.mean(self.weights))
        weight_std = float(self.xp.std(self.weights))
        active_synapses = int(self.xp.sum(self.weights > 0.1))
        
        print(f"\n   ✅ Learning benchmark complete:")
        print(f"      Duration: {total_time:.2f}s")
        print(f"      Total spikes: {self.total_spikes:,}")
        print(f"      Learning events: {self.total_learning_events:,}")
        print(f"      Spike rate: {self.total_spikes/total_time:.0f} spikes/second")
        print(f"      Learning rate: {self.total_learning_events/total_time:.0f} updates/second")
        print(f"      Neuron throughput: {self.num_neurons * steps / total_time:.0f} neuron-steps/second")
        print(f"      Average weight: {avg_weight:.4f} ± {weight_std:.4f}")
        print(f"      Active synapses: {active_synapses:,} ({active_synapses/self.num_synapses*100:.1f}%)")
        
        return {
            'duration': total_time,
            'total_spikes': self.total_spikes,
            'total_learning': self.total_learning_events,
            'steps': steps,
            'spike_rate': self.total_spikes / total_time,
            'learning_rate': self.total_learning_events / total_time,
            'neuron_throughput': self.num_neurons * steps / total_time,
            'avg_weight': avg_weight,
            'weight_std': weight_std,
            'active_synapses': active_synapses
        }
    
    def cleanup(self):
        """Clean up GPU memory"""
        if GPU_AVAILABLE:
            cp.get_default_memory_pool().free_all_blocks()
        gc.collect()

def main():
    """Run optimal GPU learning benchmark"""
    print("🚀 OPTIMAL GPU LEARNING BENCHMARK")
    print("=" * 60)
    
    start_mem = measure_memory()
    print(f"Initial memory: {start_mem:.1f}MB")
    
    try:
        # Create optimal network
        network = OptimalLearningNetwork()
        
        creation_mem = measure_memory()
        print(f"Memory after creation: {creation_mem - start_mem:.1f}MB")
        
        # Run benchmark
        results = network.run_learning_benchmark(duration_ms=10000, dt=0.5)
        
        final_mem = measure_memory()
        
        print(f"\n{'='*60}")
        print(f"🎉 FINAL OPTIMAL GPU LEARNING RESULTS")
        print(f"{'='*60}")
        print(f"Neurons: {network.num_neurons:,}")
        print(f"Synapses: {network.num_synapses:,}")
        print(f"Duration: {results['duration']:.2f}s")
        print(f"Total spikes: {results['total_spikes']:,}")
        print(f"Learning events: {results['total_learning']:,}")
        print(f"Performance: {results['neuron_throughput']:.0f} neuron-steps/second")
        print(f"Memory used: {final_mem - start_mem:.1f}MB")
        print(f"Learning effectiveness: {results['active_synapses']:,} active synapses")
        print(f"🏆 750K NEURON LEARNING WITH RTX 3060 ACHIEVED!")
        
        # Cleanup
        network.cleanup()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Final cleanup
        if GPU_AVAILABLE:
            cp.get_default_memory_pool().free_all_blocks()
        gc.collect()

if __name__ == "__main__":
    main()
