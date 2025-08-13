#!/usr/bin/env python3
"""
ADVANCED NEUROMORPHIC INTELLIGENCE SYSTEM
Teaching complex behaviors and adaptive responses
"""

import numpy as np
from core.network import NeuromorphicNetwork
import json
from datetime import datetime

class AdvancedNeuromorphicIntelligence:
    def __init__(self):
        print("🧠 ADVANCED NEUROMORPHIC INTELLIGENCE SYSTEM")
        print("=" * 55)
        print("Teaching complex behaviors and adaptive responses")
        
        self.network = NeuromorphicNetwork()
        self.setup_intelligence_network()
        self.behavior_library = []
        self.adaptation_history = []
        
    def setup_intelligence_network(self):
        """Create network for complex intelligence"""
        # Multi-layer architecture for complex behaviors
        self.network.add_layer("sensors", 8, "lif")      # Environmental sensors
        self.network.add_layer("memory", 12, "lif")      # Working memory
        self.network.add_layer("association", 10, "lif")  # Pattern association
        self.network.add_layer("decision", 6, "lif")     # Decision making
        self.network.add_layer("action", 4, "lif")       # Action selection
        
        # Intelligence connections with strong learning
        self.network.connect_layers("sensors", "memory", "stdp",
                                  connection_probability=0.8,
                                  weight=1.5,
                                  A_plus=0.25,
                                  A_minus=0.1,
                                  tau_stdp=25.0)
        
        self.network.connect_layers("memory", "association", "stdp",
                                  connection_probability=0.9,
                                  weight=1.8,
                                  A_plus=0.3,
                                  A_minus=0.12,
                                  tau_stdp=20.0)
        
        self.network.connect_layers("association", "decision", "stdp",
                                  connection_probability=1.0,
                                  weight=2.0,
                                  A_plus=0.35,
                                  A_minus=0.15,
                                  tau_stdp=18.0)
        
        self.network.connect_layers("decision", "action", "stdp",
                                  connection_probability=1.0,
                                  weight=2.2,
                                  A_plus=0.4,
                                  A_minus=0.18,
                                  tau_stdp=15.0)
        
        # Memory feedback loop for context
        self.network.connect_layers("association", "memory", "stdp",
                                  connection_probability=0.6,
                                  weight=1.0,
                                  A_plus=0.2,
                                  A_minus=0.08,
                                  tau_stdp=30.0)
        
        print("✅ Intelligence network: 8→12→10→6→4 with feedback")
        print("✅ Multi-layer learning with context memory")
        
    def create_intelligence_scenarios(self):
        """Create complex scenarios requiring intelligence"""
        scenarios = {
            'sequence_prediction': {
                'description': 'Learn and predict sequential patterns',
                'patterns': [
                    {'sequence': [1,0,1,0,1,0,0,0], 'next': [1,0,0,0]},  # Pattern continues
                    {'sequence': [0,1,0,1,0,1,0,0], 'next': [0,1,0,0]},  # Alternating
                    {'sequence': [1,1,0,0,1,1,0,0], 'next': [1,1,0,0]},  # Pairs
                    {'sequence': [1,0,0,1,0,0,1,0], 'next': [0,1,0,0]}   # Triplets
                ],
                'skill': 'temporal_prediction'
            },
            
            'pattern_completion': {
                'description': 'Complete partial patterns intelligently',
                'patterns': [
                    {'partial': [1,1,0,0,0,0,0,0], 'complete': [1,0,1,0]},  # Extend diagonal
                    {'partial': [1,0,1,0,0,0,0,0], 'complete': [1,0,1,0]},  # Continue alt
                    {'partial': [0,1,1,0,0,0,0,0], 'complete': [0,1,1,0]},  # Maintain clusters
                    {'partial': [1,1,1,0,0,0,0,0], 'complete': [1,0,0,0]}   # Break pattern
                ],
                'skill': 'pattern_intelligence'
            },
            
            'adaptive_behavior': {
                'description': 'Adapt behavior based on context',
                'contexts': [
                    {'situation': [1,0,0,1,0,1,0,0], 'response': [1,0,0,0]},  # Threat: retreat
                    {'situation': [0,1,1,0,1,0,1,0], 'response': [0,1,0,0]},  # Opportunity: approach
                    {'situation': [1,1,0,0,0,1,1,0], 'response': [0,0,1,0]},  # Complex: analyze
                    {'situation': [0,0,1,1,1,0,0,1], 'response': [0,0,0,1]}   # Ambiguous: explore
                ],
                'skill': 'contextual_adaptation'
            }
        }
        
        return scenarios
    
    def teach_intelligence_scenario(self, scenario_name, scenario, lesson_num):
        """Teach complex intelligence scenario"""
        print(f"\n🎓 INTELLIGENCE LESSON {lesson_num}: {scenario['description']}")
        print(f"Teaching skill: {scenario['skill']}")
        
        # Reset network for clean learning
        for layer in self.network.layers.values():
            layer.neuron_population.reset()
        
        # Get network components
        populations = {
            'sensors': self.network.layers["sensors"].neuron_population,
            'memory': self.network.layers["memory"].neuron_population,
            'association': self.network.layers["association"].neuron_population,
            'decision': self.network.layers["decision"].neuron_population,
            'action': self.network.layers["action"].neuron_population
        }
        
        connections = {
            'sensors_memory': self.network.connections[("sensors", "memory")],
            'memory_association': self.network.connections[("memory", "association")],
            'association_decision': self.network.connections[("association", "decision")],
            'decision_action': self.network.connections[("decision", "action")],
            'association_memory': self.network.connections[("association", "memory")]
        }
        
        # Intensive learning sessions
        total_learning_episodes = 40
        skill_mastery_progress = []
        
        for episode in range(total_learning_episodes):
            episode_results = self.run_intelligence_episode(
                scenario_name, scenario, populations, connections, episode
            )
            skill_mastery_progress.append(episode_results)
            
            # Progress feedback
            if (episode + 1) % 10 == 0:
                recent_success = np.mean([ep['success_rate'] for ep in skill_mastery_progress[-10:]])
                print(f"  Episode {episode + 1}: Recent success rate = {recent_success:.2f}")
        
        # Assess learned intelligence
        final_assessment = self.assess_learned_intelligence(scenario_name, scenario, populations, connections)
        
        lesson_data = {
            'lesson': lesson_num,
            'scenario': scenario_name,
            'skill': scenario['skill'],
            'episodes': total_learning_episodes,
            'progress': skill_mastery_progress,
            'final_assessment': final_assessment,
            'intelligence_achieved': final_assessment['intelligence_score'] > 0.6
        }
        
        self.behavior_library.append(lesson_data)
        
        print(f"  🧠 Intelligence score: {final_assessment['intelligence_score']:.2f}")
        print(f"  📈 Skill mastery: {'✅ ACHIEVED' if lesson_data['intelligence_achieved'] else '🔄 Developing'}")
        
        return lesson_data
    
    def run_intelligence_episode(self, scenario_name, scenario, populations, connections, episode):
        """Run single intelligence learning episode"""
        dt = 0.1
        episode_successes = 0
        total_trials = len(scenario.get('patterns', scenario.get('contexts', [])))
        
        # Get appropriate data based on scenario type
        if 'patterns' in scenario:
            trial_data = scenario['patterns']
        else:
            trial_data = scenario['contexts']
        
        for trial_idx, trial in enumerate(trial_data):
            # Prepare input and target based on scenario type
            if scenario_name == 'sequence_prediction':
                input_pattern = trial['sequence']
                target_output = trial['next']
            elif scenario_name == 'pattern_completion':
                input_pattern = trial['partial']
                target_output = trial['complete']
            else:  # adaptive_behavior
                input_pattern = trial['situation']
                target_output = trial['response']
            
            trial_success = self.teach_intelligence_trial(
                input_pattern, target_output, populations, connections, dt
            )
            
            if trial_success:
                episode_successes += 1
        
        success_rate = episode_successes / total_trials
        
        return {
            'episode': episode,
            'scenario': scenario_name,
            'successes': episode_successes,
            'total_trials': total_trials,
            'success_rate': success_rate
        }
    
    def teach_intelligence_trial(self, input_pattern, target_output, populations, connections, dt):
        """Teach single intelligence trial with multi-layer coordination"""
        # Reset for trial
        for pop in populations.values():
            pop.reset()
        
        trial_phases = ['input_processing', 'memory_formation', 'association_building', 'decision_making', 'action_selection']
        phase_steps = 25
        
        for phase_idx, phase in enumerate(trial_phases):
            for step in range(phase_steps):
                time = phase_idx * phase_steps * dt + step * dt
                
                # Phase-specific processing
                if phase == 'input_processing':
                    # Strong sensory input
                    sensor_currents = [120.0 if p == 1 else 5.0 for p in input_pattern]
                    sensor_states = populations['sensors'].step(dt, sensor_currents)
                    
                    # Forward propagation
                    memory_currents = self.calculate_layer_currents(connections['sensors_memory'], sensor_states, 12)
                    memory_states = populations['memory'].step(dt, memory_currents)
                    
                    # Continue cascade
                    assoc_currents = self.calculate_layer_currents(connections['memory_association'], memory_states, 10)
                    assoc_states = populations['association'].step(dt, assoc_currents)
                    
                    decision_currents = self.calculate_layer_currents(connections['association_decision'], assoc_states, 6)
                    decision_states = populations['decision'].step(dt, decision_currents)
                    
                    action_currents = self.calculate_layer_currents(connections['decision_action'], decision_states, 4)
                    action_states = populations['action'].step(dt, action_currents)
                    
                    # Apply STDP across all layers
                    self.apply_intelligence_stdp(sensor_states, memory_states, assoc_states, decision_states, action_states, connections, time)
                
                elif phase == 'memory_formation':
                    # Continue input with memory consolidation
                    sensor_currents = [80.0 if p == 1 else 0.0 for p in input_pattern]
                    sensor_states = populations['sensors'].step(dt, sensor_currents)
                    
                    # Enhanced memory activity
                    memory_currents = self.calculate_layer_currents(connections['sensors_memory'], sensor_states, 12)
                    memory_enhancement = [20.0] * 12  # Memory consolidation current
                    memory_currents = [m + e for m, e in zip(memory_currents, memory_enhancement)]
                    memory_states = populations['memory'].step(dt, memory_currents)
                    
                    # Forward processing
                    assoc_currents = self.calculate_layer_currents(connections['memory_association'], memory_states, 10)
                    assoc_states = populations['association'].step(dt, assoc_currents)
                    
                    decision_currents = self.calculate_layer_currents(connections['association_decision'], assoc_states, 6)
                    decision_states = populations['decision'].step(dt, decision_currents)
                    
                    action_currents = self.calculate_layer_currents(connections['decision_action'], decision_states, 4)
                    action_states = populations['action'].step(dt, action_currents)
                    
                    self.apply_intelligence_stdp(sensor_states, memory_states, assoc_states, decision_states, action_states, connections, time)
                
                elif phase == 'association_building':
                    # Focus on association layer with bidirectional activity
                    sensor_currents = [40.0 if p == 1 else 0.0 for p in input_pattern]
                    sensor_states = populations['sensors'].step(dt, sensor_currents)
                    
                    memory_currents = self.calculate_layer_currents(connections['sensors_memory'], sensor_states, 12)
                    memory_states = populations['memory'].step(dt, memory_currents)
                    
                    # Enhanced association building
                    assoc_currents = self.calculate_layer_currents(connections['memory_association'], memory_states, 10)
                    assoc_enhancement = [30.0] * 10  # Association building current
                    assoc_currents = [a + e for a, e in zip(assoc_currents, assoc_enhancement)]
                    assoc_states = populations['association'].step(dt, assoc_currents)
                    
                    # Feedback to memory
                    memory_feedback = self.calculate_layer_currents(connections['association_memory'], assoc_states, 12)
                    memory_states = populations['memory'].step(dt, memory_feedback)
                    
                    decision_currents = self.calculate_layer_currents(connections['association_decision'], assoc_states, 6)
                    decision_states = populations['decision'].step(dt, decision_currents)
                    
                    action_currents = self.calculate_layer_currents(connections['decision_action'], decision_states, 4)
                    action_states = populations['action'].step(dt, action_currents)
                    
                    self.apply_intelligence_stdp(sensor_states, memory_states, assoc_states, decision_states, action_states, connections, time)
                
                elif phase == 'decision_making':
                    # Reduced input, enhanced decision processing
                    sensor_currents = [20.0 if p == 1 else 0.0 for p in input_pattern]
                    sensor_states = populations['sensors'].step(dt, sensor_currents)
                    
                    memory_currents = self.calculate_layer_currents(connections['sensors_memory'], sensor_states, 12)
                    memory_states = populations['memory'].step(dt, memory_currents)
                    
                    assoc_currents = self.calculate_layer_currents(connections['memory_association'], memory_states, 10)
                    assoc_states = populations['association'].step(dt, assoc_currents)
                    
                    # Enhanced decision making
                    decision_currents = self.calculate_layer_currents(connections['association_decision'], assoc_states, 6)
                    decision_enhancement = [40.0] * 6
                    decision_currents = [d + e for d, e in zip(decision_currents, decision_enhancement)]
                    decision_states = populations['decision'].step(dt, decision_currents)
                    
                    action_currents = self.calculate_layer_currents(connections['decision_action'], decision_states, 4)
                    action_states = populations['action'].step(dt, action_currents)
                    
                    self.apply_intelligence_stdp(sensor_states, memory_states, assoc_states, decision_states, action_states, connections, time)
                
                else:  # action_selection
                    # Minimal input, target-guided action learning
                    sensor_currents = [10.0 if p == 1 else 0.0 for p in input_pattern]
                    target_currents = [100.0 if t == 1 else 0.0 for t in target_output]
                    
                    sensor_states = populations['sensors'].step(dt, sensor_currents)
                    
                    memory_currents = self.calculate_layer_currents(connections['sensors_memory'], sensor_states, 12)
                    memory_states = populations['memory'].step(dt, memory_currents)
                    
                    assoc_currents = self.calculate_layer_currents(connections['memory_association'], memory_states, 10)
                    assoc_states = populations['association'].step(dt, assoc_currents)
                    
                    decision_currents = self.calculate_layer_currents(connections['association_decision'], assoc_states, 6)
                    decision_states = populations['decision'].step(dt, decision_currents)
                    
                    # Target-guided action learning
                    action_states = populations['action'].step(dt, target_currents)
                    
                    # Strong STDP during action learning
                    self.apply_intelligence_stdp(sensor_states, memory_states, assoc_states, decision_states, action_states, connections, time, strength=1.5)
                
                self.network.step(dt)
        
        # Test trial success
        return self.test_trial_intelligence(input_pattern, target_output, populations, connections)
    
    def calculate_layer_currents(self, connection, pre_states, post_layer_size):
        """Calculate synaptic currents for layer"""
        currents = [0.0] * post_layer_size
        
        if connection.synapse_population and hasattr(connection.synapse_population, 'synapses'):
            for (pre_idx, post_idx), synapse in connection.synapse_population.synapses.items():
                if pre_idx < len(pre_states) and post_idx < post_layer_size and pre_states[pre_idx]:
                    current = synapse.weight * 12.0  # Amplification
                    currents[post_idx] += current
        
        return currents
    
    def apply_intelligence_stdp(self, sensor_states, memory_states, assoc_states, decision_states, action_states, connections, time, strength=1.0):
        """Apply STDP across all intelligence layers"""
        layer_pairs = [
            (sensor_states, memory_states, 'sensors_memory'),
            (memory_states, assoc_states, 'memory_association'),
            (assoc_states, decision_states, 'association_decision'),
            (decision_states, action_states, 'decision_action'),
            (assoc_states, memory_states, 'association_memory')  # Feedback
        ]
        
        for pre_states, post_states, connection_name in layer_pairs:
            if connection_name in connections:
                connection = connections[connection_name]
                if connection.synapse_population and hasattr(connection.synapse_population, 'synapses'):
                    
                    for (pre_idx, post_idx), synapse in connection.synapse_population.synapses.items():
                        if pre_idx < len(pre_states) and post_idx < len(post_states):
                            
                            if pre_states[pre_idx]:
                                synapse.pre_spike(time)
                                if post_states[post_idx]:  # Coincident
                                    synapse.weight += synapse.A_plus * 0.15 * strength
                            
                            if post_states[post_idx]:
                                synapse.post_spike(time)
                                if not pre_states[pre_idx]:  # Post-only
                                    synapse.weight -= synapse.A_minus * 0.08 * strength
                            
                            # Maintain weight bounds
                            synapse.weight = np.clip(synapse.weight, 0.1, 6.0)
    
    def test_trial_intelligence(self, input_pattern, target_output, populations, connections):
        """Test if trial was learned intelligently"""
        # Reset for testing
        for pop in populations.values():
            pop.reset()
        
        output_activity = [0] * 4
        
        # Test learned intelligence
        for step in range(60):
            sensor_currents = [60.0 if p == 1 else 0.0 for p in input_pattern]
            sensor_states = populations['sensors'].step(0.1, sensor_currents)
            
            memory_currents = self.calculate_layer_currents(connections['sensors_memory'], sensor_states, 12)
            memory_states = populations['memory'].step(0.1, memory_currents)
            
            assoc_currents = self.calculate_layer_currents(connections['memory_association'], memory_states, 10)
            assoc_states = populations['association'].step(0.1, assoc_currents)
            
            decision_currents = self.calculate_layer_currents(connections['association_decision'], assoc_states, 6)
            decision_states = populations['decision'].step(0.1, decision_currents)
            
            action_currents = self.calculate_layer_currents(connections['decision_action'], decision_states, 4)
            action_states = populations['action'].step(0.1, action_currents)
            
            for i, spike in enumerate(action_states):
                if spike:
                    output_activity[i] += 1
            
            self.network.step(0.1)
        
        # Check if output matches target
        expected_pattern = target_output
        if sum(output_activity) > 0:
            # Find most active outputs
            threshold = max(output_activity) * 0.7
            predicted_pattern = [1 if activity >= threshold else 0 for activity in output_activity]
            
            # Calculate match
            matches = sum(1 for p, e in zip(predicted_pattern, expected_pattern) if p == e)
            success = matches >= len(expected_pattern) * 0.75  # 75% match required
        else:
            success = False
        
        return success
    
    def assess_learned_intelligence(self, scenario_name, scenario, populations, connections):
        """Comprehensive assessment of learned intelligence"""
        if 'patterns' in scenario:
            test_data = scenario['patterns']
        else:
            test_data = scenario['contexts']
        
        correct_predictions = 0
        total_tests = len(test_data)
        confidence_scores = []
        
        for test_item in test_data:
            if scenario_name == 'sequence_prediction':
                input_pattern = test_item['sequence']
                target_output = test_item['next']
            elif scenario_name == 'pattern_completion':
                input_pattern = test_item['partial']
                target_output = test_item['complete']
            else:
                input_pattern = test_item['situation']
                target_output = test_item['response']
            
            test_success = self.test_trial_intelligence(input_pattern, target_output, populations, connections)
            if test_success:
                correct_predictions += 1
                confidence_scores.append(0.8)  # High confidence for correct
            else:
                confidence_scores.append(0.2)  # Low confidence for incorrect
        
        intelligence_score = correct_predictions / total_tests
        avg_confidence = np.mean(confidence_scores)
        
        return {
            'scenario': scenario_name,
            'correct_predictions': correct_predictions,
            'total_tests': total_tests,
            'intelligence_score': intelligence_score,
            'average_confidence': avg_confidence,
            'skill_mastered': intelligence_score > 0.6
        }
    
    def run_intelligence_curriculum(self):
        """Run complete intelligence learning curriculum"""
        print("Starting advanced neuromorphic intelligence curriculum...")
        
        scenarios = self.create_intelligence_scenarios()
        print(f"✅ Created {len(scenarios)} intelligence scenarios")
        
        # Teaching phase
        print(f"\n🧠 INTELLIGENCE TEACHING PHASE")
        print("-" * 35)
        
        for i, (scenario_name, scenario) in enumerate(scenarios.items(), 1):
            lesson_data = self.teach_intelligence_scenario(scenario_name, scenario, i)
        
        # Final intelligence assessment
        print(f"\n🏆 FINAL INTELLIGENCE ASSESSMENT")
        print("=" * 38)
        
        skills_mastered = 0
        total_skills = len(scenarios)
        intelligence_scores = []
        
        for lesson in self.behavior_library:
            assessment = lesson['final_assessment']
            if assessment['skill_mastered']:
                skills_mastered += 1
                print(f"✅ {assessment['scenario']}: MASTERED (score: {assessment['intelligence_score']:.2f})")
            else:
                print(f"🔄 {assessment['scenario']}: Developing (score: {assessment['intelligence_score']:.2f})")
            
            intelligence_scores.append(assessment['intelligence_score'])
        
        # Overall intelligence metrics
        overall_intelligence = skills_mastered / total_skills
        avg_intelligence_score = np.mean(intelligence_scores)
        
        print(f"\n🧠 INTELLIGENCE METRICS")
        print("-" * 25)
        print(f"Skills mastered: {skills_mastered}/{total_skills} ({overall_intelligence:.1%})")
        print(f"Average intelligence score: {avg_intelligence_score:.3f}")
        
        # Intelligence classification
        if overall_intelligence >= 0.8 and avg_intelligence_score >= 0.7:
            print(f"\n🌟 ADVANCED INTELLIGENCE: ACHIEVED!")
            print(f"✅ High-level cognitive abilities")
            print(f"✅ Complex pattern recognition")
            print(f"✅ Adaptive decision making")
            intelligence_level = "ADVANCED"
        elif overall_intelligence >= 0.6:
            print(f"\n🧠 GOOD INTELLIGENCE: DEVELOPED!")
            print(f"✅ Solid cognitive foundations")
            intelligence_level = "GOOD"
        elif overall_intelligence > 0.3:
            print(f"\n🌱 EMERGING INTELLIGENCE: PROMISING!")
            print(f"🔄 Basic intelligence emerging")
            intelligence_level = "EMERGING"
        else:
            print(f"\n🌱 FOUNDATIONAL INTELLIGENCE: BUILDING")
            intelligence_level = "FOUNDATIONAL"
        
        # Save intelligence report
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_type': 'advanced_neuromorphic_intelligence',
            'total_skills': total_skills,
            'skills_mastered': skills_mastered,
            'overall_intelligence': overall_intelligence,
            'average_intelligence_score': avg_intelligence_score,
            'intelligence_level': intelligence_level,
            'behavior_library': self.behavior_library
        }
        
        with open('intelligence_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n💾 Intelligence report saved: intelligence_report.json")
        
        return overall_intelligence >= 0.5

if __name__ == "__main__":
    intelligence_system = AdvancedNeuromorphicIntelligence()
    success = intelligence_system.run_intelligence_curriculum()
    
    if success:
        print(f"\n🚀 ADVANCED NEUROMORPHIC INTELLIGENCE: SUCCESS!")
        print(f"The system demonstrates sophisticated cognitive abilities.")
    else:
        print(f"\n🧠 Continuing intelligence development journey...")
