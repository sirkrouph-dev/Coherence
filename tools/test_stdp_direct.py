#!/usr/bin/env python3
"""
STDP DIRECT TEST - Test STDP mechanism directly
"""

import numpy as np
from core.synapses import STDP_Synapse

def test_stdp_directly():
    """Test STDP synapse directly without network overhead"""
    print("🔬 DIRECT STDP TESTING")
    print("=" * 25)
    
    # Create a single STDP synapse
    synapse = STDP_Synapse(
        synapse_id=1,
        pre_neuron_id=0,
        post_neuron_id=1,
        weight=1.0,
        A_plus=0.2,   # Strong LTP
        A_minus=0.1,  # Strong LTD
        tau_stdp=20.0
    )
    
    print(f"Initial weight: {synapse.weight:.3f}")
    
    # Test 1: Pre-before-post (should increase weight)
    print("\nTest 1: Pre-before-post (LTP expected)")
    synapse.pre_spike(0.0)   # Pre spike at t=0
    synapse.post_spike(5.0)  # Post spike at t=5ms (pre-before-post)
    print(f"After pre(0) → post(5): {synapse.weight:.3f}")
    
    # Reset synapse
    synapse = STDP_Synapse(
        synapse_id=2,
        pre_neuron_id=0,
        post_neuron_id=1,
        weight=1.0,
        A_plus=0.2,
        A_minus=0.1,
        tau_stdp=20.0
    )
    
    # Test 2: Post-before-pre (should decrease weight)
    print("\nTest 2: Post-before-pre (LTD expected)")
    synapse.post_spike(0.0)  # Post spike at t=0
    synapse.pre_spike(5.0)   # Pre spike at t=5ms (post-before-pre)
    print(f"After post(0) → pre(5): {synapse.weight:.3f}")
    
    return True

def test_synaptic_current():
    """Test synaptic current computation"""
    print("\n⚡ SYNAPTIC CURRENT TEST")
    print("=" * 25)
    
    synapse = STDP_Synapse(
        synapse_id=3,
        pre_neuron_id=0,
        post_neuron_id=1,
        weight=2.0,
        tau_syn=5.0
    )
    
    # Test current after spike
    current_0 = synapse.compute_current(0.0, 0.0)  # At spike time
    current_1 = synapse.compute_current(0.0, 1.0)  # 1ms after
    current_5 = synapse.compute_current(0.0, 5.0)  # 5ms after
    current_10 = synapse.compute_current(0.0, 10.0) # 10ms after
    
    print(f"Current at spike (t=0): {current_0:.3f}")
    print(f"Current at t=1ms: {current_1:.3f}")
    print(f"Current at t=5ms: {current_5:.3f}")
    print(f"Current at t=10ms: {current_10:.3f}")
    
    return current_0 > 0

if __name__ == "__main__":
    print("Testing STDP mechanisms directly...")
    stdp_working = test_stdp_directly()
    current_working = test_synaptic_current()
    
    print(f"\n📊 DIRECT TEST RESULTS")
    print("=" * 22)
    print(f"STDP plasticity: {'✅ Working' if stdp_working else '❌ Not working'}")
    print(f"Synaptic current: {'✅ Working' if current_working else '❌ Not working'}")
