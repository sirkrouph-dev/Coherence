#!/usr/bin/env python3
"""
NEUROMORPHIC REINFORCEMENT LEARNING SYSTEM
Teaching through rewards, exploration, and adaptive behavior
"""

import numpy as np
from core.network import NeuromorphicNetwork
from core.neuromodulation import NeuromodulatoryController
import json
from datetime import datetime

class NeuromorphicReinforcementLearning:
    def __init__(self):
        print("🧠 NEUROMORPHIC REINFORCEMENT LEARNING SYSTEM")
        print("=" * 50)
        print("Teaching through rewards, exploration, and adaptation")
        
        self.network = NeuromorphicNetwork()
        self.setup_rl_network()
        
        # Initialize neuromodulation for reward learning
        self.neuromod_controller = NeuromodulatoryController()
        self.setup_reward_system()
        
        self.learning_episodes = []
        self.exploration_history = []
        
    def setup_rl_network(self):
        """Create network optimized for reinforcement learning"""
        # Sensory input layer
        self.network.add_layer("sensors", 12, "lif")     # Environmental sensors
        
        # Action selection layers  
        self.network.add_layer("memory", 16, "lif")      # Working memory
        self.network.add_layer("decision", 8, "lif")     # Decision making
        self.network.add_layer("actions", 4, "lif")      # Action outputs
        
        # Value assessment
        self.network.add_layer("value", 6, "lif")        # Value estimation
        
        # Connect network with learning-optimized parameters
        # Sensor → Memory (strong learning)
        self.network.connect_layers("sensors", "memory", "stdp",
                                  connection_probability=0.9,
                                  weight=1.0,
                                  A_plus=0.2,
                                  A_minus=0.08,
                                  tau_stdp=30.0)
        
        # Memory → Decision (adaptive)
        self.network.connect_layers("memory", "decision", "stdp",
                                  connection_probability=0.8,
                                  weight=1.2,
                                  A_plus=0.25,
                                  A_minus=0.1,
                                  tau_stdp=25.0)
        
        # Decision → Actions (precise)
        self.network.connect_layers("decision", "actions", "stdp",
                                  connection_probability=1.0,
                                  weight=1.5,
                                  A_plus=0.3,
                                  A_minus=0.12,
                                  tau_stdp=20.0)
        
        # Memory → Value (reward prediction)
        self.network.connect_layers("memory", "value", "stdp",
                                  connection_probability=0.7,
                                  weight=0.8,
                                  A_plus=0.15,
                                  A_minus=0.06,
                                  tau_stdp=35.0)
        
        print("✅ RL Network: 12→16→8→4 (actions), 16→6 (value)")
        print("✅ Multi-pathway STDP learning")
        
    def setup_reward_system(self):
        """Initialize reward-based neuromodulation"""
        # Dopamine-like reward signal
        self.reward_baseline = 0.0
        self.reward_history = []
        
        # Exploration parameters
        self.exploration_rate = 0.3
        self.exploration_decay = 0.995
        
        # Learning rate modulation
        self.base_learning_rate = 1.0
        self.reward_learning_rate = 1.5
        self.punishment_learning_rate = 0.7
        
        print("✅ Reward system initialized")
        print(f"  - Exploration rate: {self.exploration_rate}")
        print(f"  - Learning modulation ready")
    
    def create_environment_scenarios(self):
        """Create different environmental scenarios for learning"""
        scenarios = {
            'navigation': {
                'description': 'Navigate to target while avoiding obstacles',
                'sensors': {
                    'target_left': 0, 'target_right': 1, 'target_up': 2, 'target_down': 3,
                    'obstacle_near': 4, 'obstacle_far': 5,
                    'wall_left': 6, 'wall_right': 7, 'wall_up': 8, 'wall_down': 9,
                    'energy_high': 10, 'energy_low': 11
                },
                'actions': ['move_left', 'move_right', 'move_up', 'move_down'],
                'reward_rules': {
                    'reach_target': +10.0,
                    'avoid_obstacle': +2.0,
                    'hit_obstacle': -5.0,
                    'move_toward_target': +1.0,
                    'move_away_target': -1.0,
                    'energy_efficient': +0.5,
                    'time_penalty': -0.1
                }
            },
            
            'foraging': {
                'description': 'Find and collect resources efficiently',
                'sensors': {
                    'food_detected': 0, 'food_quality': 1, 'food_distance': 2,
                    'predator_near': 3, 'predator_far': 4,
                    'shelter_available': 5, 'weather_good': 6, 'weather_bad': 7,
                    'energy_level': 8, 'health_status': 9, 'time_day': 10, 'time_night': 11
                },
                'actions': ['search', 'collect', 'retreat', 'rest'],
                'reward_rules': {
                    'collect_food': +8.0,
                    'high_quality_food': +3.0,
                    'avoid_predator': +4.0,
                    'efficient_search': +1.0,
                    'wasteful_action': -2.0,
                    'survival': +0.5
                }
            },
            
            'social_learning': {
                'description': 'Learn social behaviors and cooperation',
                'sensors': {
                    'peer_cooperate': 0, 'peer_defect': 1, 'peer_neutral': 2,
                    'group_size': 3, 'resource_abundance': 4, 'resource_scarcity': 5,
                    'reputation_high': 6, 'reputation_low': 7,
                    'communication_signal': 8, 'threat_level': 9, 'opportunity': 10, 'trust_level': 11
                },
                'actions': ['cooperate', 'compete', 'communicate', 'withdraw'],
                'reward_rules': {
                    'mutual_cooperation': +6.0,
                    'successful_communication': +3.0,
                    'build_trust': +2.0,
                    'exploit_others': -4.0,
                    'help_group': +1.5,
                    'social_learning': +1.0
                }
            }
        }
        
        return scenarios
    
    def run_learning_episode(self, scenario_name, scenario, episode_num):
        """Run single reinforcement learning episode"""
        print(f"\n🎮 Episode {episode_num}: {scenario['description']}")
        
        # Reset network state
        for layer in self.network.layers.values():
            layer.neuron_population.reset()
        
        # Episode parameters
        max_steps = 150
        total_reward = 0.0
        actions_taken = []
        states_visited = []
        
        # Initialize environment state
        current_state = self.generate_initial_state(scenario)
        
        for step in range(max_steps):
            # Process current state through sensors
            sensor_activations = self.state_to_sensors(current_state, scenario)
            
            # Get action from network
            action_idx, action_confidence = self.select_action(sensor_activations, scenario)
            action_name = scenario['actions'][action_idx]
            
            # Apply action to environment and get reward
            next_state, reward, done = self.apply_action(current_state, action_idx, scenario)
            
            # Apply neuromodulation based on reward
            self.apply_reward_learning(reward, action_confidence)
            
            # Record episode data
            total_reward += reward
            actions_taken.append(action_name)
            states_visited.append(current_state.copy())
            
            # Update state
            current_state = next_state
            
            if done or step >= max_steps - 1:
                break
        
        # Episode summary
        episode_data = {
            'episode': episode_num,
            'scenario': scenario_name,
            'total_reward': total_reward,
            'steps_taken': step + 1,
            'actions': actions_taken,
            'avg_reward_per_step': total_reward / (step + 1),
            'exploration_used': self.exploration_rate
        }
        
        self.learning_episodes.append(episode_data)
        
        # Update exploration rate
        self.exploration_rate *= self.exploration_decay
        
        print(f"  Reward: {total_reward:.2f}, Steps: {step + 1}, Avg: {episode_data['avg_reward_per_step']:.3f}")
        
        return episode_data
    
    def state_to_sensors(self, state, scenario):
        """Convert environment state to sensor activations"""
        sensor_activations = [0.0] * 12
        
        # Convert state dictionary to sensor array based on scenario
        for sensor_name, sensor_idx in scenario['sensors'].items():
            if sensor_name in state:
                sensor_activations[sensor_idx] = state[sensor_name]
        
        return sensor_activations
    
    def select_action(self, sensor_activations, scenario):
        """Select action using network with exploration"""
        # Stimulate sensor layer
        sensor_pop = self.network.layers["sensors"].neuron_population
        memory_pop = self.network.layers["memory"].neuron_population
        decision_pop = self.network.layers["decision"].neuron_population
        action_pop = self.network.layers["actions"].neuron_population
        value_pop = self.network.layers["value"].neuron_population
        
        dt = 0.1
        action_activity = [0] * 4
        value_activity = [0] * 6
        
        # Run network for action selection
        for step in range(60):
            # Sensor input (first half)
            if step < 30:
                sensor_states = sensor_pop.step(dt, [act * 50.0 for act in sensor_activations])
            else:
                sensor_states = sensor_pop.step(dt, [0.0] * 12)
            
            # Process through network layers
            memory_states = memory_pop.step(dt, [0.0] * 16)
            decision_states = decision_pop.step(dt, [0.0] * 8)
            action_states = action_pop.step(dt, [0.0] * 4)
            value_states = value_pop.step(dt, [0.0] * 6)
            
            # Count activity
            for i, spike in enumerate(action_states):
                if spike:
                    action_activity[i] += 1
            
            for i, spike in enumerate(value_states):
                if spike:
                    value_activity[i] += 1
            
            # Apply STDP
            self.apply_rl_stdp(sensor_states, memory_states, decision_states, action_states, value_states, step * dt)
            
            self.network.step(dt)
        
        # Select action (exploitation vs exploration)
        if np.random.random() < self.exploration_rate:
            # Exploration: random action
            action_idx = np.random.randint(0, 4)
            confidence = 0.1
        else:
            # Exploitation: network choice
            if max(action_activity) > 0:
                action_idx = np.argmax(action_activity)
                confidence = action_activity[action_idx] / sum(action_activity) if sum(action_activity) > 0 else 0.1
            else:
                action_idx = np.random.randint(0, 4)
                confidence = 0.05
        
        return action_idx, confidence
    
    def apply_action(self, state, action_idx, scenario):
        """Apply action to environment and calculate reward"""
        # Simple environment simulation
        next_state = state.copy()
        reward = 0.0
        done = False
        
        # Apply scenario-specific action effects
        if scenario.get('description') == 'Navigate to target while avoiding obstacles':
            reward, done = self.apply_navigation_action(state, next_state, action_idx, scenario)
        elif scenario.get('description') == 'Find and collect resources efficiently':
            reward, done = self.apply_foraging_action(state, next_state, action_idx, scenario)
        elif scenario.get('description') == 'Learn social behaviors and cooperation':
            reward, done = self.apply_social_action(state, next_state, action_idx, scenario)
        
        # Add small time penalty to encourage efficiency
        reward += scenario['reward_rules'].get('time_penalty', -0.1)
        
        return next_state, reward, done
    
    def apply_navigation_action(self, state, next_state, action_idx, scenario):
        """Apply navigation action and calculate reward"""
        actions = ['move_left', 'move_right', 'move_up', 'move_down']
        action = actions[action_idx]
        
        reward = 0.0
        done = False
        
        # Simple navigation simulation
        target_distance = state.get('target_distance', 5.0)
        obstacle_near = state.get('obstacle_near', 0.0)
        
        # Action effects
        if action == 'move_left' and state.get('target_left', 0) > 0.5:
            target_distance -= 1.0
            reward += scenario['reward_rules']['move_toward_target']
        elif action == 'move_right' and state.get('target_right', 0) > 0.5:
            target_distance -= 1.0
            reward += scenario['reward_rules']['move_toward_target']
        else:
            target_distance += 0.5
            reward += scenario['reward_rules']['move_away_target']
        
        # Obstacle interaction
        if obstacle_near > 0.5:
            if action in ['move_up', 'move_down']:  # Avoid obstacle
                reward += scenario['reward_rules']['avoid_obstacle']
            else:
                reward += scenario['reward_rules']['hit_obstacle']
        
        # Update state
        next_state['target_distance'] = max(0, target_distance)
        
        # Check completion
        if target_distance <= 0.5:
            reward += scenario['reward_rules']['reach_target']
            done = True
        
        return reward, done
    
    def apply_foraging_action(self, state, next_state, action_idx, scenario):
        """Apply foraging action and calculate reward"""
        actions = ['search', 'collect', 'retreat', 'rest']
        action = actions[action_idx]
        
        reward = 0.0
        done = False
        
        food_detected = state.get('food_detected', 0.0)
        predator_near = state.get('predator_near', 0.0)
        energy = state.get('energy_level', 0.5)
        
        if action == 'search':
            if food_detected < 0.3:
                # Successful search
                next_state['food_detected'] = min(1.0, food_detected + 0.3)
                reward += scenario['reward_rules']['efficient_search']
            energy -= 0.1
        elif action == 'collect' and food_detected > 0.5:
            reward += scenario['reward_rules']['collect_food']
            if state.get('food_quality', 0.5) > 0.7:
                reward += scenario['reward_rules']['high_quality_food']
            next_state['food_detected'] = 0.0
            energy += 0.2
        elif action == 'retreat' and predator_near > 0.5:
            reward += scenario['reward_rules']['avoid_predator']
        elif action == 'rest':
            energy += 0.15
        else:
            reward += scenario['reward_rules']['wasteful_action']
        
        next_state['energy_level'] = np.clip(energy, 0.0, 1.0)
        
        # Survival bonus
        if energy > 0.2:
            reward += scenario['reward_rules']['survival']
        
        return reward, done
    
    def apply_social_action(self, state, next_state, action_idx, scenario):
        """Apply social action and calculate reward"""
        actions = ['cooperate', 'compete', 'communicate', 'withdraw']
        action = actions[action_idx]
        
        reward = 0.0
        done = False
        
        peer_cooperate = state.get('peer_cooperate', 0.0)
        trust_level = state.get('trust_level', 0.5)
        
        if action == 'cooperate':
            if peer_cooperate > 0.5:
                reward += scenario['reward_rules']['mutual_cooperation']
                next_state['trust_level'] = min(1.0, trust_level + 0.2)
            else:
                reward += scenario['reward_rules']['exploit_others']
        elif action == 'communicate':
            reward += scenario['reward_rules']['successful_communication']
            next_state['trust_level'] = min(1.0, trust_level + 0.1)
        elif action == 'compete':
            reward += scenario['reward_rules']['exploit_others'] * 0.5
        
        # Trust building bonus
        if trust_level > 0.7:
            reward += scenario['reward_rules']['build_trust']
        
        return reward, done
    
    def generate_initial_state(self, scenario):
        """Generate random initial state for episode"""
        state = {}
        
        # Generate random values for scenario sensors
        for sensor_name in scenario['sensors'].keys():
            if 'distance' in sensor_name:
                state[sensor_name] = np.random.uniform(2.0, 8.0)
            elif 'level' in sensor_name:
                state[sensor_name] = np.random.uniform(0.3, 0.8)
            else:
                state[sensor_name] = np.random.uniform(0.0, 1.0)
        
        return state
    
    def apply_reward_learning(self, reward, confidence):
        """Apply reward-based learning modulation"""
        # Calculate reward prediction error
        self.reward_history.append(reward)
        if len(self.reward_history) > 10:
            self.reward_baseline = np.mean(self.reward_history[-10:])
        
        reward_error = reward - self.reward_baseline
        
        # Modulate learning based on reward
        if reward_error > 0:
            # Positive reward - strengthen recent connections
            learning_modulation = self.reward_learning_rate
        else:
            # Negative reward - weaken recent connections
            learning_modulation = self.punishment_learning_rate
        
        # Apply neuromodulation to network
        # (This would typically modify STDP parameters)
        self.current_learning_modulation = learning_modulation
    
    def apply_rl_stdp(self, sensor_spikes, memory_spikes, decision_spikes, action_spikes, value_spikes, time):
        """Apply STDP with reinforcement learning modulation"""
        # Apply STDP to all connections with current modulation
        for (pre_layer, post_layer), connection in self.network.connections.items():
            if connection.synapse_population and hasattr(connection.synapse_population, 'synapses'):
                
                # Get appropriate spike trains
                if pre_layer == "sensors" and post_layer == "memory":
                    self.apply_modulated_stdp(connection, sensor_spikes, memory_spikes, time)
                elif pre_layer == "memory" and post_layer == "decision":
                    self.apply_modulated_stdp(connection, memory_spikes, decision_spikes, time)
                elif pre_layer == "decision" and post_layer == "actions":
                    self.apply_modulated_stdp(connection, decision_spikes, action_spikes, time)
                elif pre_layer == "memory" and post_layer == "value":
                    self.apply_modulated_stdp(connection, memory_spikes, value_spikes, time)
    
    def apply_modulated_stdp(self, connection, pre_spikes, post_spikes, time):
        """Apply STDP with reward modulation"""
        modulation = getattr(self, 'current_learning_modulation', 1.0)
        
        for (pre_idx, post_idx), synapse in connection.synapse_population.synapses.items():
            if pre_idx < len(pre_spikes) and post_idx < len(post_spikes):
                # Apply spikes with modulation
                if pre_spikes[pre_idx]:
                    synapse.pre_spike(time)
                    # Enhance learning with reward modulation
                    if modulation > 1.0:
                        synapse.weight *= (1.0 + (modulation - 1.0) * 0.1)
                if post_spikes[post_idx]:
                    synapse.post_spike(time)
                    if modulation < 1.0:
                        synapse.weight *= modulation
    
    def run_rl_curriculum(self, episodes_per_scenario=50):
        """Run complete reinforcement learning curriculum"""
        print("Starting neuromorphic reinforcement learning curriculum...")
        
        scenarios = self.create_environment_scenarios()
        print(f"✅ Created {len(scenarios)} learning scenarios")
        
        total_episodes = 0
        scenario_results = {}
        
        for scenario_name, scenario in scenarios.items():
            print(f"\n🎯 LEARNING SCENARIO: {scenario['description']}")
            print("-" * 60)
            
            scenario_rewards = []
            
            for episode in range(episodes_per_scenario):
                episode_data = self.run_learning_episode(scenario_name, scenario, episode + 1)
                scenario_rewards.append(episode_data['total_reward'])
                total_episodes += 1
                
                # Progress reporting
                if (episode + 1) % 10 == 0:
                    recent_avg = np.mean(scenario_rewards[-10:])
                    print(f"  Episodes {episode-8}-{episode+1}: Avg reward = {recent_avg:.2f}")
            
            # Scenario summary
            final_performance = np.mean(scenario_rewards[-10:])
            improvement = final_performance - np.mean(scenario_rewards[:10])
            
            scenario_results[scenario_name] = {
                'episodes': episodes_per_scenario,
                'final_performance': final_performance,
                'improvement': improvement,
                'total_reward': sum(scenario_rewards)
            }
            
            print(f"\n📊 {scenario_name.upper()} RESULTS:")
            print(f"  Final performance: {final_performance:.2f}")
            print(f"  Improvement: {improvement:+.2f}")
            
            if improvement > 2.0:
                print(f"  🎉 Excellent learning progress!")
            elif improvement > 0.5:
                print(f"  ✅ Good improvement")
            else:
                print(f"  🔄 Gradual learning")
        
        # Overall assessment
        print(f"\n🏆 REINFORCEMENT LEARNING ASSESSMENT")
        print("=" * 50)
        
        total_improvement = sum(result['improvement'] for result in scenario_results.values())
        avg_improvement = total_improvement / len(scenario_results)
        
        print(f"Total episodes: {total_episodes}")
        print(f"Average improvement: {avg_improvement:.2f}")
        print(f"Final exploration rate: {self.exploration_rate:.3f}")
        
        # Success criteria
        if avg_improvement > 2.0:
            print(f"\n🌟 REINFORCEMENT LEARNING: HIGHLY SUCCESSFUL!")
            print(f"✅ Strong adaptation across scenarios")
            print(f"✅ Effective reward-based learning")
            print(f"✅ Exploration-exploitation balance")
            success_status = "HIGHLY_SUCCESSFUL"
        elif avg_improvement > 0.5:
            print(f"\n✅ REINFORCEMENT LEARNING: SUCCESSFUL!")
            print(f"✅ Clear learning progress")
            success_status = "SUCCESSFUL"
        else:
            print(f"\n🔄 REINFORCEMENT LEARNING: DEVELOPING")
            success_status = "DEVELOPING"
        
        # Save comprehensive report
        report = {
            'timestamp': datetime.now().isoformat(),
            'learning_type': 'neuromorphic_reinforcement',
            'total_episodes': total_episodes,
            'scenarios': len(scenarios),
            'average_improvement': avg_improvement,
            'final_exploration_rate': self.exploration_rate,
            'success_status': success_status,
            'scenario_results': scenario_results,
            'recent_episodes': self.learning_episodes[-20:]  # Last 20 episodes
        }
        
        with open('rl_learning_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n💾 RL learning report saved: rl_learning_report.json")
        
        return avg_improvement > 0.5

if __name__ == "__main__":
    rl_system = NeuromorphicReinforcementLearning()
    success = rl_system.run_rl_curriculum(episodes_per_scenario=30)
    
    if success:
        print(f"\n🚀 NEUROMORPHIC RL: SUCCESS ACHIEVED!")
        print(f"The system demonstrates adaptive learning through experience.")
    else:
        print(f"\n📈 Continuing reinforcement learning development...")
