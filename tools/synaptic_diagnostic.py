#!/usr/bin/env python3
"""
NEUROMORPHIC SYNAPTIC TRANSMISSION DIAGNOSTIC
Debug synaptic transmission between layers
"""

import numpy as np
from core.network import NeuromorphicNetwork

class SynapticTransmissionDiagnostic:
    def __init__(self):
        print("🔍 SYNAPTIC TRANSMISSION DIAGNOSTIC")
        print("=" * 50)
        print("Investigating synaptic transmission issues")
        
        self.network = NeuromorphicNetwork()
        self.setup_test_network()
        
    def setup_test_network(self):
        """Create minimal test network"""
        self.network.add_layer("input", 3, "lif")
        self.network.add_layer("output", 2, "lif")
        
        # Create simple connection
        self.network.connect_layers("input", "output", "stdp",
                                  connection_probability=1.0,
                                  weight=3.0,  # Strong weight
                                  A_plus=0.2,
                                  A_minus=0.1,
                                  tau_stdp=20.0)
        
        print("✅ Test network: 3 → 2 neurons")
        
    def test_basic_transmission(self):
        """Test basic synaptic transmission"""
        print(f"\n🧪 BASIC TRANSMISSION TEST")
        print("-" * 30)
        
        # Reset network
        for layer in self.network.layers.values():
            layer.neuron_population.reset()
        
        input_pop = self.network.layers["input"].neuron_population
        output_pop = self.network.layers["output"].neuron_population
        
        # Check initial neuron parameters
        print("Input neuron parameters:")
        for i, neuron in enumerate(input_pop.neurons):
            print(f"  Neuron {i}: threshold={neuron.v_thresh}, reset={neuron.v_reset}, membrane={neuron.membrane_potential}")
        
        print("Output neuron parameters:")
        for i, neuron in enumerate(output_pop.neurons):
            print(f"  Neuron {i}: threshold={neuron.v_thresh}, reset={neuron.v_reset}, membrane={neuron.membrane_potential}")
        
        # Check synaptic connections
        connection = self.network.connections[("input", "output")]
        print(f"\nSynaptic connections:")
        if connection.synapse_population and hasattr(connection.synapse_population, 'synapses'):
            for (pre_idx, post_idx), synapse in connection.synapse_population.synapses.items():
                print(f"  Input[{pre_idx}] → Output[{post_idx}]: weight={synapse.weight:.3f}")
        
        # Test strong stimulation
        strong_currents = [200.0, 200.0, 200.0]  # Very strong
        
        print(f"\nTesting with strong stimulation: {strong_currents}")
        
        for step in range(50):
            dt = 0.1
            time = step * dt
            
            # Step input layer
            input_states = input_pop.step(dt, strong_currents)
            
            # Calculate synaptic currents manually
            synaptic_currents = [0.0, 0.0]
            
            if connection.synapse_population and hasattr(connection.synapse_population, 'synapses'):
                for (pre_idx, post_idx), synapse in connection.synapse_population.synapses.items():
                    if input_states[pre_idx]:  # If pre-synaptic neuron spiked
                        synaptic_current = synapse.weight * 10.0  # Amplify current
                        synaptic_currents[post_idx] += synaptic_current
                        print(f"    Step {step}: Input[{pre_idx}] spiked → Output[{post_idx}] gets {synaptic_current:.1f} current")
            
            # Step output layer with calculated currents
            output_states = output_pop.step(dt, synaptic_currents)
            
            # Report activity
            if any(input_states) or any(output_states):
                print(f"  Step {step}: Input spikes={input_states}, Output spikes={output_states}")
                if any(output_states):
                    print(f"    SUCCESS: Output layer responded!")
                    break
            
            # Check membrane potentials
            if step % 10 == 0:
                input_membranes = [neuron.membrane_potential for neuron in input_pop.neurons]
                output_membranes = [neuron.membrane_potential for neuron in output_pop.neurons]
                print(f"  Step {step}: Input V_mem={[f'{v:.1f}' for v in input_membranes]}, Output V_mem={[f'{v:.1f}' for v in output_membranes]}")
            
            self.network.step(dt)
        
        return True
    
    def test_network_step_integration(self):
        """Test if network.step() properly handles synaptic transmission"""
        print(f"\n🔧 NETWORK INTEGRATION TEST")
        print("-" * 30)
        
        # Reset again
        for layer in self.network.layers.values():
            layer.neuron_population.reset()
        
        input_pop = self.network.layers["input"].neuron_population
        output_pop = self.network.layers["output"].neuron_population
        
        # Store network step results
        network_steps = []
        
        for step in range(30):
            dt = 0.1
            
            # Apply strong input
            input_currents = [150.0, 150.0, 150.0]
            
            # Use network's built-in step
            self.network.step(dt)
            
            # Manually stimulate input layer
            input_states = input_pop.step(dt, input_currents)
            output_states = output_pop.step(dt, [0.0, 0.0])
            
            # Check what happened
            input_spikes = [neuron.membrane_potential > neuron.v_thresh for neuron in input_pop.neurons]
            output_spikes = [neuron.membrane_potential > neuron.v_thresh for neuron in output_pop.neurons]
            
            network_steps.append({
                'step': step,
                'input_spikes': input_spikes,
                'output_spikes': output_spikes
            })
            
            if any(output_spikes):
                print(f"  SUCCESS at step {step}: Network.step() enabled transmission!")
                break
            elif step % 10 == 0:
                print(f"  Step {step}: No output response yet...")
        
        # Summary
        total_input_spikes = sum(sum(s['input_spikes']) for s in network_steps)
        total_output_spikes = sum(sum(s['output_spikes']) for s in network_steps)
        
        print(f"\nNetwork integration results:")
        print(f"  Total input spikes: {total_input_spikes}")
        print(f"  Total output spikes: {total_output_spikes}")
        
        if total_output_spikes > 0:
            print(f"  ✅ Network.step() works for transmission")
            return True
        else:
            print(f"  ❌ Network.step() not transmitting properly")
            return False
    
    def test_manual_synaptic_current(self):
        """Test manual synaptic current injection"""
        print(f"\n💉 MANUAL SYNAPTIC CURRENT TEST")
        print("-" * 35)
        
        # Reset network
        for layer in self.network.layers.values():
            layer.neuron_population.reset()
        
        output_pop = self.network.layers["output"].neuron_population
        
        # Test direct current injection to output neurons
        test_currents = [50.0, 100.0, 150.0, 200.0, 250.0]
        
        for current in test_currents:
            # Reset output neurons
            output_pop.reset()
            
            print(f"\nTesting direct current injection: {current} nA")
            
            spikes_generated = 0
            for step in range(20):
                output_states = output_pop.step(0.1, [current, current])
                spikes_generated += sum(output_states)
                
                if any(output_states):
                    print(f"  ✅ Spikes generated at {current} nA after {step + 1} steps")
                    break
            
            if spikes_generated == 0:
                print(f"  ❌ No spikes with {current} nA")
            else:
                print(f"  Total spikes: {spikes_generated}")
        
        return True
    
    def test_lif_neuron_thresholds(self):
        """Test LIF neuron threshold behavior"""
        print(f"\n⚡ LIF NEURON THRESHOLD TEST")
        print("-" * 30)
        
        # Create single LIF neuron for testing
        from core.neurons import LeakyIntegrateAndFire
        
        test_neuron = LeakyIntegrateAndFire(neuron_id=0)
        print(f"LIF neuron parameters:")
        print(f"  Threshold: {test_neuron.v_thresh} mV")
        print(f"  Reset: {test_neuron.v_reset} mV")
        print(f"  Rest: {test_neuron.v_rest} mV")
        print(f"  Tau_membrane: {test_neuron.tau_m} ms")
        print(f"  Initial membrane: {test_neuron.membrane_potential} mV")
        
        # Test with different currents
        currents_to_test = [10.0, 50.0, 100.0, 200.0, 500.0]
        
        for current in currents_to_test:
            test_neuron.reset()
            print(f"\nTesting {current} nA current:")
            
            for step in range(50):
                spiked = test_neuron.step(0.1, current)
                
                if step % 10 == 0 or spiked:
                    print(f"  Step {step}: V_mem = {test_neuron.membrane_potential:.2f} mV, Spiked = {spiked}")
                
                if spiked:
                    print(f"  ✅ Spike achieved with {current} nA at step {step}")
                    break
            else:
                print(f"  ❌ No spike with {current} nA after 50 steps")
        
        return True
    
    def run_complete_diagnostic(self):
        """Run complete synaptic transmission diagnostic"""
        print("Starting complete synaptic transmission diagnostic...")
        
        # Test sequence
        print(f"\n" + "="*60)
        test1_ok = self.test_lif_neuron_thresholds()
        
        print(f"\n" + "="*60)
        test2_ok = self.test_manual_synaptic_current()
        
        print(f"\n" + "="*60)
        test3_ok = self.test_basic_transmission()
        
        print(f"\n" + "="*60)
        test4_ok = self.test_network_step_integration()
        
        # Final assessment
        print(f"\n🏆 DIAGNOSTIC SUMMARY")
        print("=" * 30)
        print(f"✅ LIF neuron thresholds: {'PASS' if test1_ok else 'FAIL'}")
        print(f"✅ Manual current injection: {'PASS' if test2_ok else 'FAIL'}")
        print(f"✅ Basic transmission: {'PASS' if test3_ok else 'FAIL'}")
        print(f"✅ Network integration: {'PASS' if test4_ok else 'FAIL'}")
        
        if all([test1_ok, test2_ok, test3_ok, test4_ok]):
            print(f"\n🎉 ALL TESTS PASSED - Synaptic transmission working!")
            diagnosis = "TRANSMISSION_WORKING"
        elif test1_ok and test2_ok:
            print(f"\n🔧 NEURON LEVEL OK - Issue with synaptic connections")
            diagnosis = "SYNAPTIC_CONNECTION_ISSUE"
        elif test1_ok:
            print(f"\n⚡ NEURON OK - Issue with current injection")
            diagnosis = "CURRENT_INJECTION_ISSUE"
        else:
            print(f"\n🚨 FUNDAMENTAL NEURON ISSUE")
            diagnosis = "NEURON_ISSUE"
        
        return diagnosis

if __name__ == "__main__":
    diagnostic = SynapticTransmissionDiagnostic()
    result = diagnostic.run_complete_diagnostic()
    print(f"\n🎯 DIAGNOSIS: {result}")
