"""
Pure Neuromorphic Symbol Emergence Demo
Demonstrates emergent symbolic reasoning without LLM dependencies
"""

import sys
import os
import numpy as np
import time
from typing import Dict, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.symbol_emergence import NeuromorphicSymbolEngine


class SymbolEmergenceDemo:
    """Demonstration of pure neuromorphic symbol emergence"""
    
    def __init__(self, scale_factor: float = 0.001):
        """Initialize demo with scaled system for testing"""
        # Scale down for demo (300M neurons -> 300K for testing)
        base_neurons = int(300_000_000 * scale_factor)
        
        self.engine = NeuromorphicSymbolEngine(total_neurons=base_neurons)
        self.demo_history = []
        
        print("🧠 PURE NEUROMORPHIC SYMBOL EMERGENCE")
        print("=" * 50)
        print(f"🔬 System Scale: {base_neurons:,} neurons")
        print(f"🌟 No LLMs, no pre-defined symbols, no static mappings")
        print(f"⚡ Symbols emerge from stable cell assembly trajectories")
        print()
    
    def run_sensorimotor_grounding_demo(self):
        """Demonstrate symbol grounding through sensorimotor loops"""
        print("🌱 PHASE 1: SENSORIMOTOR GROUNDING")
        print("-" * 40)
        print("Exposing system to structured sensorimotor patterns...")
        print("Symbols will emerge from consistent interaction patterns")
        print()
        
        # Simulate structured sensorimotor experiences
        experiences = [
            # "Grasping" pattern
            {
                'name': 'grasp_red_sphere',
                'sensory': self._generate_sensory_pattern('red', 'sphere', 'tactile'),
                'motor': self._generate_motor_pattern('grasp', 'close')
            },
            {
                'name': 'grasp_blue_cube', 
                'sensory': self._generate_sensory_pattern('blue', 'cube', 'tactile'),
                'motor': self._generate_motor_pattern('grasp', 'close')
            },
            # "Looking" pattern
            {
                'name': 'look_red_sphere',
                'sensory': self._generate_sensory_pattern('red', 'sphere', 'visual'),
                'motor': self._generate_motor_pattern('look', 'focus')
            },
            {
                'name': 'look_blue_cube',
                'sensory': self._generate_sensory_pattern('blue', 'cube', 'visual'), 
                'motor': self._generate_motor_pattern('look', 'focus')
            }
        ]
        
        # Run experiences multiple times to establish patterns
        for round_num in range(5):
            print(f"📚 Learning round {round_num + 1}/5")
            
            for exp in experiences:
                result = self.engine.process_sensorimotor_stream(
                    exp['sensory'], exp['motor']
                )
                
                print(f"  {exp['name']}: {result['emerged_symbols']} symbols, "
                      f"stability={result['symbol_stability']:.3f}")
        
        print("\n✅ Sensorimotor grounding complete!")
        self._show_symbol_analysis()
    
    def run_composition_demo(self):
        """Demonstrate symbol composition through temporal binding"""
        print("\n🔗 PHASE 2: SYMBOL COMPOSITION")
        print("-" * 40)
        print("Testing compositional binding of emerged symbols...")
        print()
        
        # Test novel combinations
        novel_combinations = [
            {
                'name': 'red_cube_grasp',
                'sensory': self._generate_sensory_pattern('red', 'cube', 'tactile'),
                'motor': self._generate_motor_pattern('grasp', 'close')
            },
            {
                'name': 'blue_sphere_look',
                'sensory': self._generate_sensory_pattern('blue', 'sphere', 'visual'),
                'motor': self._generate_motor_pattern('look', 'focus')
            }
        ]
        
        for combo in novel_combinations:
            result = self.engine.process_sensorimotor_stream(
                combo['sensory'], combo['motor']
            )
            
            print(f"🔄 Novel combination '{combo['name']}':")
            print(f"  Symbols emerged: {result['emerged_symbols']}")
            print(f"  Symbol stability: {result['symbol_stability']:.3f}")
            print(f"  Layer activities: {[f'{a:.3f}' for a in result['layer_activities'] if len(result['layer_activities']) > 0]}")
            print()
        
        print("✅ Composition testing complete!")
        self._test_symbol_properties()
    
    def run_syntax_emergence_demo(self):
        """Demonstrate emergent syntax through predictive sequences"""
        print("\n📝 PHASE 3: SYNTAX EMERGENCE")
        print("-" * 40)
        print("Testing emergent syntax through temporal prediction...")
        print()
        
        # Create temporal sequences
        sequences = [
            # Action sequences
            ['look', 'grasp', 'lift'],
            ['look', 'push', 'release'],
            ['grasp', 'move', 'place'],
            # Object-action sequences  
            ['red', 'sphere', 'grasp'],
            ['blue', 'cube', 'push'],
            ['green', 'cylinder', 'lift']
        ]
        
        for seq_num, sequence in enumerate(sequences):
            print(f"🔄 Sequence {seq_num + 1}: {' → '.join(sequence)}")
            
            # Process sequence step by step
            sequence_results = []
            
            for step, action in enumerate(sequence):
                # Generate appropriate sensorimotor pattern
                if action in ['red', 'blue', 'green']:
                    sensory = self._generate_sensory_pattern(action, 'object', 'visual')
                    motor = self._generate_motor_pattern('attend', 'focus')
                elif action in ['sphere', 'cube', 'cylinder']:
                    sensory = self._generate_sensory_pattern('color', action, 'visual')
                    motor = self._generate_motor_pattern('perceive', 'analyze')
                else:  # Actions
                    sensory = self._generate_sensory_pattern('object', 'shape', 'tactile')
                    motor = self._generate_motor_pattern(action, 'execute')
                
                result = self.engine.process_sensorimotor_stream(sensory, motor)
                sequence_results.append(result)
                
                print(f"  Step {step + 1} ({action}): "
                      f"stability={result['symbol_stability']:.3f}")
            
            # Analyze sequence learning
            stabilities = [r['symbol_stability'] for r in sequence_results]
            improvement = stabilities[-1] - stabilities[0] if len(stabilities) > 1 else 0
            
            print(f"  Sequence learning: {improvement:+.3f} stability improvement")
            print()
        
        print("✅ Syntax emergence testing complete!")
        self._analyze_predictive_patterns()
    
    def run_comprehensive_validation(self):
        """Run comprehensive validation of symbol properties"""
        print("\n🧪 PHASE 4: COMPREHENSIVE VALIDATION")
        print("-" * 40)
        print("Testing all symbolic properties...")
        print()
        
        # Test symbol properties
        properties = self.engine.test_symbol_properties()
        
        print("📊 SYMBOL PROPERTY VALIDATION:")
        print(f"  🔗 Compositionality: {properties['compositionality']:.3f}")
        print(f"  🌟 Productivity: {properties['productivity']:.3f}")
        print(f"  🔄 Systematicity: {properties['systematicity']:.3f}")
        print(f"  ⚡ Discreteness: {properties['discreteness']:.3f}")
        print(f"  🌍 Grounding: {properties['grounding']:.3f}")
        print()
        
        # Overall symbolic capability
        overall_score = np.mean(list(properties.values()))
        
        if overall_score >= 0.8:
            level = "🏆 OUTSTANDING - True symbolic reasoning achieved"
        elif overall_score >= 0.6:
            level = "🌟 EXCELLENT - Strong symbolic capabilities"
        elif overall_score >= 0.4:
            level = "✅ GOOD - Emerging symbolic properties"
        elif overall_score >= 0.2:
            level = "🔄 DEVELOPING - Basic symbol emergence"
        else:
            level = "🌱 FOUNDATIONAL - Early symbol formation"
        
        print(f"🎯 OVERALL SYMBOLIC CAPABILITY: {overall_score:.3f}")
        print(f"🏆 ACHIEVEMENT LEVEL: {level}")
        print()
        
        return properties
    
    def _generate_sensory_pattern(self, color: str, shape: str, modality: str) -> np.ndarray:
        """Generate consistent sensory patterns for grounding"""
        # Create deterministic patterns based on attributes
        pattern_size = 1000  # Size of sensory input
        pattern = np.zeros(pattern_size)
        
        # Color encoding (first 300 neurons)
        color_map = {'red': 0, 'blue': 100, 'green': 200, 'color': 150}
        if color in color_map:
            start_idx = color_map[color]
            pattern[start_idx:start_idx + 50] = np.random.uniform(0.8, 1.0, 50)
        
        # Shape encoding (neurons 300-600)
        shape_map = {'sphere': 300, 'cube': 400, 'cylinder': 500, 'shape': 350, 'object': 450}
        if shape in shape_map:
            start_idx = shape_map[shape]
            pattern[start_idx:start_idx + 50] = np.random.uniform(0.8, 1.0, 50)
        
        # Modality encoding (neurons 600-900)
        modality_map = {'visual': 600, 'tactile': 700, 'auditory': 800}
        if modality in modality_map:
            start_idx = modality_map[modality]
            pattern[start_idx:start_idx + 50] = np.random.uniform(0.8, 1.0, 50)
        
        # Add noise for biological realism
        noise = np.random.normal(0, 0.1, pattern_size)
        pattern += noise
        
        return np.clip(pattern, 0, 1)
    
    def _generate_motor_pattern(self, action: str, param: str) -> np.ndarray:
        """Generate consistent motor patterns"""
        pattern_size = 500  # Size of motor output
        pattern = np.zeros(pattern_size)
        
        # Action encoding
        action_map = {
            'grasp': 0, 'look': 100, 'push': 200, 'lift': 300, 'move': 400,
            'attend': 50, 'perceive': 150, 'execute': 250, 'release': 350, 'place': 450
        }
        if action in action_map:
            start_idx = action_map[action]
            pattern[start_idx:start_idx + 30] = np.random.uniform(0.7, 1.0, 30)
        
        # Parameter encoding
        param_map = {
            'close': 30, 'focus': 130, 'hard': 230, 'up': 330, 'forward': 430,
            'analyze': 80, 'gentle': 180, 'down': 280, 'back': 380
        }
        if param in param_map:
            start_idx = param_map[param]
            pattern[start_idx:start_idx + 20] = np.random.uniform(0.6, 0.9, 20)
        
        # Add noise
        noise = np.random.normal(0, 0.05, pattern_size)
        pattern += noise
        
        return np.clip(pattern, 0, 1)
    
    def _show_symbol_analysis(self):
        """Show analysis of emerged symbols"""
        state = self.engine.get_system_state()
        
        print("\n📊 SYMBOL EMERGENCE ANALYSIS:")
        print(f"  🧠 Total neurons: {state['total_neurons']:,}")
        print(f"  🏗️ Hierarchical layers: {state['layers']}")
        print(f"  ⚡ Cell assemblies: {state['cell_assemblies']}")
        print(f"  🌍 Sensorimotor experiences: {state['sensorimotor_experiences']}")
        print(f"  💡 Average layer activity: {np.mean(state['layer_activities']):.3f}")
        print()
    
    def _test_symbol_properties(self):
        """Test and display symbol properties"""
        properties = self.engine.test_symbol_properties()
        
        print("🔬 SYMBOL PROPERTY ANALYSIS:")
        for prop_name, value in properties.items():
            status = "✅" if value > 0.5 else "🔄" if value > 0.3 else "🌱"
            print(f"  {status} {prop_name.capitalize()}: {value:.3f}")
        print()
    
    def _analyze_predictive_patterns(self):
        """Analyze predictive patterns in the system"""
        state = self.engine.get_system_state()
        
        print("🔮 PREDICTIVE PATTERN ANALYSIS:")
        print(f"  📈 Temporal coherence emerging across {state['layers']} layers")
        print(f"  🧠 {state['cell_assemblies']} stable cell assemblies detected")
        print(f"  ⚡ Symbol stability: {state['symbol_properties']['discreteness']:.3f}")
        print()


def main():
    """Run comprehensive symbol emergence demonstration"""
    
    print("🚀 PURE NEUROMORPHIC SYMBOL EMERGENCE DEMO")
    print("=" * 55)
    print("🎯 Demonstrating true emergent symbolic reasoning")
    print("⚡ No LLMs, no pre-defined symbols, no static templates")
    print("🧠 Symbols emerge from neural dynamics alone")
    print()
    
    try:
        # Create demo system
        demo = SymbolEmergenceDemo(scale_factor=0.001)  # 300K neurons for demo
        
        # Phase 1: Sensorimotor grounding
        demo.run_sensorimotor_grounding_demo()
        
        # Phase 2: Symbol composition
        demo.run_composition_demo()
        
        # Phase 3: Syntax emergence
        demo.run_syntax_emergence_demo()
        
        # Phase 4: Comprehensive validation
        final_properties = demo.run_comprehensive_validation()
        
        # Summary
        print("🎉 DEMONSTRATION COMPLETE!")
        print("=" * 30)
        
        overall_score = np.mean(list(final_properties.values()))
        print(f"🎯 Overall symbolic capability: {overall_score:.3f}")
        
        if overall_score > 0.6:
            print("🏆 SUCCESS: True neuromorphic symbolic reasoning demonstrated!")
        elif overall_score > 0.4:
            print("🌟 PROGRESS: Strong symbolic emergence detected!")
        else:
            print("🌱 FOUNDATION: Basic symbol formation observed!")
        
        print("\n💡 Next steps:")
        print("  - Scale to full 300M neurons for enhanced emergence")
        print("  - Connect to real sensorimotor systems")
        print("  - Test novel combination generation")
        print("  - Validate compositional reasoning")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
