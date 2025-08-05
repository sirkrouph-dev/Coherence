"""
Core neuromorphic programming system components.
"""

from .neurons import (
    NeuronModel,
    AdaptiveExponentialIntegrateAndFire,
    HodgkinHuxleyNeuron,
    LeakyIntegrateAndFire,
    NeuronFactory,
    NeuronPopulation
)

from .synapses import (
    SynapseType,
    SynapseModel,
    STDP_Synapse,
    ShortTermPlasticitySynapse,
    NeuromodulatorySynapse,
    SynapseFactory,
    SynapsePopulation
)

from .network import (
    NetworkLayer,
    NetworkConnection,
    NeuromorphicNetwork,
    EventDrivenSimulator,
    NetworkBuilder
)

from .encoding import (
    SensoryEncoder,
    RetinalEncoder,
    CochlearEncoder,
    SomatosensoryEncoder,
    MultiModalEncoder,
    RateEncoder
)

from .neuromodulation import (
    NeuromodulatorType,
    NeuromodulatorySystem,
    DopaminergicSystem,
    SerotonergicSystem,
    CholinergicSystem,
    NoradrenergicSystem,
    NeuromodulatoryController,
    HomeostaticRegulator,
    RewardSystem
)

__all__ = [
    # Neurons
    'NeuronModel',
    'AdaptiveExponentialIntegrateAndFire',
    'HodgkinHuxleyNeuron',
    'LeakyIntegrateAndFire',
    'NeuronFactory',
    'NeuronPopulation',
    
    # Synapses
    'SynapseType',
    'SynapseModel',
    'STDP_Synapse',
    'ShortTermPlasticitySynapse',
    'NeuromodulatorySynapse',
    'SynapseFactory',
    'SynapsePopulation',
    
    # Network
    'NetworkLayer',
    'NetworkConnection',
    'NeuromorphicNetwork',
    'EventDrivenSimulator',
    'NetworkBuilder',
    
    # Encoding
    'SensoryEncoder',
    'RetinalEncoder',
    'CochlearEncoder',
    'SomatosensoryEncoder',
    'MultiModalEncoder',
    'RateEncoder',
    
    # Neuromodulation
    'NeuromodulatorType',
    'NeuromodulatorySystem',
    'DopaminergicSystem',
    'SerotonergicSystem',
    'CholinergicSystem',
    'NoradrenergicSystem',
    'NeuromodulatoryController',
    'HomeostaticRegulator',
    'RewardSystem'
] 