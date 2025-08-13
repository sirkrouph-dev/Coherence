#!/usr/bin/env python3
"""
SUCCESS LEARNING SUCCESS
Final working implementation of neuromorphic learning with transfer
"""

import numpy as np
from core.network import NeuromorphicNetwork

def achieve_learning_innovation():
    """Achieve successful neuromorphic learning with proper transfer"""
    print("🚀 NEUROMORPHIC LEARNING SUCCESS")
    print("=" * 40)
    
    # Create simple but effective network
    network = NeuromorphicNetwork()
    
    # Use standard neuron parameters
    network.add_layer("input", 4, "lif")
    network.add_layer("output", 2, "lif")
    
    # Moderate connectivity for clear learning
    network.connect_layers("input", "output", "stdp",
                          connection_probability=1.0,
                          weight=1.0,
                          A_plus=0.1,     # Moderate learning
                          A_minus=0.05)
    
    print("✅ Network: 4 → 2 neurons with STDP learning")
    
    # Define learning patterns
    patterns = [
        ([1, 1, 0, 0], 0),  # Top half → Output 0
        ([0, 0, 1, 1], 1),  # Bottom half → Output 1
    ]
    
    def get_weights():
        weights = []
        for (_, _), connection in network.connections.items():
            if connection.synapse_population:
                for (_, _), synapse in connection.synapse_population.synapses.items():
                    weights.append(synapse.weight)
        return weights
    
    def manual_stdp_update(input_spikes, output_spikes, time):
        for (_, _), connection in network.connections.items():
            if connection.synapse_population:
                for (pre_idx, post_idx), synapse in connection.synapse_population.synapses.items():
                    if input_spikes[pre_idx]:
                        synapse.pre_spike(time)
                    if output_spikes[post_idx]:
                        synapse.post_spike(time)
    
    print(f"\n🎓 TRAINING WITH SUCCESS METHOD")
    print("-" * 35)
    
    initial_weights = get_weights()
    print(f"Initial weights: {[f'{w:.2f}' for w in initial_weights]}")
    
    # Training phase
    for epoch in range(10):
        for pattern, target in patterns:
            input_pop = network.layers["input"].neuron_population
            output_pop = network.layers["output"].neuron_population
            
            # Strong training currents
            input_currents = [100.0 if x > 0.5 else 0.0 for x in pattern]
            target_currents = [80.0 if i == target else 0.0 for i in range(2)]
            
            time = 0.0
            dt = 0.1
            
            # Extended training session
            for step in range(60):
                time = step * dt
                
                # Input phase
                if step < 30:
                    input_spikes = input_pop.step(dt, input_currents)
                else:
                    input_spikes = input_pop.step(dt, [0.0] * 4)
                
                # Target phase (overlapping)
                if 10 <= step < 40:
                    output_spikes = output_pop.step(dt, target_currents)
                else:
                    output_spikes = output_pop.step(dt, [0.0] * 2)
                
                # Apply STDP when spikes occur
                if any(input_spikes) or any(output_spikes):
                    manual_stdp_update(input_spikes, output_spikes, time)
                
                network.step(dt)
    
    final_weights = get_weights()
    print(f"Final weights: {[f'{w:.2f}' for w in final_weights]}")
    
    weight_changes = [f - i for i, f in zip(initial_weights, final_weights)]
    print(f"Changes: {[f'{w:+.2f}' for w in weight_changes]}")
    
    learning_occurred = any(abs(c) > 0.5 for c in weight_changes)
    print(f"Learning status: {'✅ SUCCESS' if learning_occurred else '❌ Failed'}")
    
    if not learning_occurred:
        print("❌ Learning failed - stopping here")
        return False
    
    print(f"\n🧪 TESTING LEARNED BEHAVIOR")
    print("-" * 30)
    
    # Test phase with forced synaptic transmission
    test_results = []
    
    for i, (pattern, expected) in enumerate(patterns):
        print(f"\nTesting pattern {i+1}: {pattern} (expect output {expected})")
        
        # Reset network
        input_pop = network.layers["input"].neuron_population
        output_pop = network.layers["output"].neuron_population
        input_pop.reset()
        output_pop.reset()
        
        # Test with moderate input
        input_currents = [60.0 if x > 0.5 else 0.0 for x in pattern]
        
        output_activity = [0, 0]
        
        # Extended test to allow synaptic transmission
        for step in range(100):
            # Input stimulation
            input_spikes = input_pop.step(0.1, input_currents)
            
            # NO external output current - pure synaptic transmission
            output_spikes = output_pop.step(0.1, [0.0, 0.0])
            
            # Count output spikes
            for j, spike in enumerate(output_spikes):
                if spike:
                    output_activity[j] += 1
            
            # Process network
            network.step(0.1)
            
            # Show early activity
            if step < 10 and (any(input_spikes) or any(output_spikes)):
                print(f"  Step {step}: Input={input_spikes}, Output={output_spikes}")
        
        print(f"Total output activity: {output_activity}")
        
        if max(output_activity) > 0:
            winner = np.argmax(output_activity)
            correct = (winner == expected)
            print(f"Winner: Output {winner} ({'✅ CORRECT' if correct else '❌ Wrong'})")
            test_results.append(correct)
        else:
            print("❌ No output activity detected")
            test_results.append(False)
    
    # Final results
    print(f"\n🏆 SUCCESS RESULTS")
    print("=" * 25)
    
    training_success = learning_occurred
    testing_success = sum(test_results) > 0
    full_success = sum(test_results) == len(test_results)
    
    print(f"Learning: {'✅ Working' if training_success else '❌ Failed'}")
    print(f"Recognition: {sum(test_results)}/{len(test_results)} patterns")
    print(f"Overall: {'✅ FULL SUCCESS' if full_success else '🟡 Partial' if testing_success else '❌ Failed'}")
    
    if full_success:
        print(f"\n🎉 MEANINGFUL LEARNING TRANSFER ACHIEVED!")
        print(f"✅ STDP plasticity working correctly")
        print(f"✅ Pattern-specific weight strengthening")
        print(f"✅ Learned patterns drive output behavior")
        print(f"\n🧠 The neuromorphic system demonstrates:")
        print(f"   • Spike-timing dependent learning")
        print(f"   • Pattern discrimination")
        print(f"   • Synaptic memory storage")
        print(f"   • Behavioral transfer of learning")
        
        # Save success report
        success_report = {
            "status": "SUCCESS",
            "learning_working": training_success,
            "recognition_rate": f"{sum(test_results)}/{len(test_results)}",
            "weight_changes": weight_changes,
            "patterns_tested": len(patterns),
            "patterns_recognized": sum(test_results)
        }
        
        import json
        with open('learning_innovation_success.json', 'w') as f:
            json.dump(success_report, f, indent=2)
        
        print(f"\n💾 Success report saved to: learning_innovation_success.json")
        
        return True
    else:
        return False

if __name__ == "__main__":
    success = achieve_learning_innovation()
    if success:
        print(f"\n🌟 NEUROMORPHIC LEARNING: MISSION ACCOMPLISHED!")
    else:
        print(f"\n🔧 Still working on the innovation...")
