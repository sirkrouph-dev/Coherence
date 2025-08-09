import sys, numpy as np
sys.path.append('.')
from core.neurons import NeuronPopulation
from core.synapses import SynapsePopulation

input_size=10; output_size=5
inp=NeuronPopulation(input_size, neuron_type='lif')
out=NeuronPopulation(output_size, neuron_type='lif')
syn=SynapsePopulation(input_size, output_size, 'stdp', connection_probability=0.5, weight=5.0)

def run(pattern):
    inp.reset(); out.reset()
    dt=0.1
    counts=np.zeros(output_size)
    for i in range(200):
        t=i*dt
        pre=inp.step(dt, pattern)
        cur=syn.get_synaptic_currents(pre, t)
        post=out.step(dt, cur)
        syn.update_weights(pre, post, t)
        counts+=post
    return counts

p0=np.zeros(input_size); p0[0:3]=50
p1=np.zeros(input_size); p1[3:6]=50
r0=run(p0)
r1=run(p1)
print('r0', r0)
print('r1', r1)
print('equal', np.array_equal(r0, r1))
