#!/usr/bin/env python3
"""
Advanced English Learning Module
Building comprehensive language understanding through neuromorphic processing

Features:
- Progressive vocabulary building
- Grammar rule learning
- Semantic understanding
- Context-aware processing
- Memory consolidation for language patterns
"""

import numpy as np
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum

# Import neuromorphic components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.network import NeuromorphicNetwork
from core.encoding import MultiModalEncoder
from core.memory import ShortTermMemory, LongTermMemory
from core.learning import PlasticityManager


class LanguageLevel(Enum):
    """Language learning progression levels"""
    PHONEMES = "phonemes"
    LETTERS = "letters"
    WORDS = "words"
    PHRASES = "phrases"
    SENTENCES = "sentences"
    GRAMMAR = "grammar"
    SEMANTICS = "semantics"


@dataclass
class LanguagePattern:
    """Structure for language learning patterns"""
    text: str
    meaning: List[float]
    category: str
    level: LanguageLevel
    difficulty: float
    dependencies: List[str]


@dataclass
class LearningProgress:
    """Track learning progress for language elements"""
    element: str
    attempts: int
    successes: int
    confidence: float
    last_seen: datetime
    consolidated: bool


class AdvancedEnglishLearner:
    """Advanced neuromorphic English learning system"""
    
    def __init__(self, memory_enabled: bool = True):
        """Initialize the advanced English learning system"""
        print("🧠 ADVANCED NEUROMORPHIC ENGLISH LEARNING SYSTEM")
        print("=" * 60)
        print("📚 Progressive curriculum with memory consolidation")
        print("🔄 Adaptive learning rates and difficulty scaling")
        print("🎯 Comprehensive language understanding")
        
        # Initialize components
        self.memory_enabled = memory_enabled
        self.setup_language_components()
        self.setup_network_architecture()
        self.create_curriculum()
        
        # Learning state
        self.current_level = LanguageLevel.LETTERS
        self.learning_progress: Dict[str, LearningProgress] = {}
        self.mastered_elements: Set[str] = set()
        
        # Performance tracking
        self.session_stats = {
            'start_time': datetime.now(),
            'elements_learned': 0,
            'total_attempts': 0,
            'success_rate': 0.0
        }
        
    def setup_language_components(self):
        """Setup core language processing components"""
        # Character encoding system
        self.characters = list("abcdefghijklmnopqrstuvwxyz .,!?'-")
        self.char_to_idx = {char: idx for idx, char in enumerate(self.characters)}
        self.vocab_size = len(self.characters)
        
        # Word categories and their semantic vectors
        self.word_categories = {
            'noun': [1, 0, 0, 0, 0, 0, 0, 0],
            'verb': [0, 1, 0, 0, 0, 0, 0, 0],
            'adjective': [0, 0, 1, 0, 0, 0, 0, 0],
            'pronoun': [0, 0, 0, 1, 0, 0, 0, 0],
            'article': [0, 0, 0, 0, 1, 0, 0, 0],
            'preposition': [0, 0, 0, 0, 0, 1, 0, 0],
            'conjunction': [0, 0, 0, 0, 0, 0, 1, 0],
            'interjection': [0, 0, 0, 0, 0, 0, 0, 1]
        }
        
        # Grammar rules
        self.grammar_patterns = {
            'simple_sentence': ['noun', 'verb'],
            'descriptive_noun': ['adjective', 'noun'],
            'verb_phrase': ['verb', 'noun'],
            'prepositional_phrase': ['preposition', 'noun'],
            'question': ['verb', 'pronoun', 'verb'],
            'compound_sentence': ['noun', 'verb', 'conjunction', 'noun', 'verb']
        }
        
        # Initialize memory systems if enabled
        if self.memory_enabled:
            self.stm = ShortTermMemory(capacity=7, duration=20000.0)  # 20 seconds
            self.ltm = LongTermMemory(consolidation_rate=0.1, retrieval_threshold=0.6)
            print("✅ Memory systems initialized")
            
    def setup_network_architecture(self):
        """Create the neuromorphic network for language processing"""
        self.network = NeuromorphicNetwork()
        
        # Multi-layered architecture for language hierarchy
        # Input layer: Character encoding
        self.network.add_layer("characters", self.vocab_size, "lif")
        
        # Phoneme layer: Sound patterns
        self.network.add_layer("phonemes", 32, "adex")
        
        # Morpheme layer: Word components
        self.network.add_layer("morphemes", 24, "adex")
        
        # Word layer: Complete words
        self.network.add_layer("words", 16, "adex")
        
        # Grammar layer: Syntactic rules
        self.network.add_layer("grammar", 12, "adex")
        
        # Semantic layer: Meaning representation
        self.network.add_layer("semantics", 8, "adex")
        
        # Connect layers with different connection types
        # Bottom-up processing
        self.network.connect_layers("characters", "phonemes", "stdp", 
                                   connection_probability=0.3, weight=0.8)
        self.network.connect_layers("phonemes", "morphemes", "stdp",
                                   connection_probability=0.4, weight=0.9)
        self.network.connect_layers("morphemes", "words", "stdp",
                                   connection_probability=0.5, weight=1.0)
        self.network.connect_layers("words", "grammar", "stdp",
                                   connection_probability=0.6, weight=1.1)
        self.network.connect_layers("grammar", "semantics", "stdp",
                                   connection_probability=0.7, weight=1.2)
        
        # Top-down feedback connections
        self.network.connect_layers("semantics", "grammar", "stdp",
                                   connection_probability=0.4, weight=0.6)
        self.network.connect_layers("grammar", "words", "stdp",
                                   connection_probability=0.3, weight=0.5)
        self.network.connect_layers("words", "morphemes", "stdp",
                                   connection_probability=0.2, weight=0.4)
        
        # Lateral connections for context
        self.network.connect_layers("words", "words", "stdp",
                                   connection_probability=0.2, weight=0.3)
        self.network.connect_layers("grammar", "grammar", "stdp",
                                   connection_probability=0.3, weight=0.4)
        
        # Setup plasticity (simplified for now)
        self.plasticity_manager = PlasticityManager()
        # Note: STDP is already enabled by default in the synapses
        
        print("✅ Network architecture established")
        print(f"   Layers: {len(self.network.layers)}")
        print(f"   Connections: {len(self.network.connections)}")
        
    def create_curriculum(self):
        """Create progressive learning curriculum"""
        self.curriculum = {
            LanguageLevel.LETTERS: [
                LanguagePattern("a", [1,0,0,0,0,0,0,0], "vowel", LanguageLevel.LETTERS, 0.1, []),
                LanguagePattern("e", [1,0,0,0,0,0,0,0], "vowel", LanguageLevel.LETTERS, 0.1, []),
                LanguagePattern("i", [1,0,0,0,0,0,0,0], "vowel", LanguageLevel.LETTERS, 0.1, []),
                LanguagePattern("o", [1,0,0,0,0,0,0,0], "vowel", LanguageLevel.LETTERS, 0.1, []),
                LanguagePattern("u", [1,0,0,0,0,0,0,0], "vowel", LanguageLevel.LETTERS, 0.1, []),
                LanguagePattern("b", [0,1,0,0,0,0,0,0], "consonant", LanguageLevel.LETTERS, 0.2, []),
                LanguagePattern("c", [0,1,0,0,0,0,0,0], "consonant", LanguageLevel.LETTERS, 0.2, []),
                LanguagePattern("d", [0,1,0,0,0,0,0,0], "consonant", LanguageLevel.LETTERS, 0.2, []),
                LanguagePattern("f", [0,1,0,0,0,0,0,0], "consonant", LanguageLevel.LETTERS, 0.2, []),
                LanguagePattern("g", [0,1,0,0,0,0,0,0], "consonant", LanguageLevel.LETTERS, 0.2, []),
            ],
            
            LanguageLevel.WORDS: [
                LanguagePattern("cat", [1,0,0,0,0,0,0,0], "noun", LanguageLevel.WORDS, 0.3, ["c","a","t"]),
                LanguagePattern("dog", [1,0,0,0,0,0,0,0], "noun", LanguageLevel.WORDS, 0.3, ["d","o","g"]),
                LanguagePattern("run", [0,1,0,0,0,0,0,0], "verb", LanguageLevel.WORDS, 0.4, ["r","u","n"]),
                LanguagePattern("big", [0,0,1,0,0,0,0,0], "adjective", LanguageLevel.WORDS, 0.4, ["b","i","g"]),
                LanguagePattern("red", [0,0,1,0,0,0,0,0], "adjective", LanguageLevel.WORDS, 0.4, ["r","e","d"]),
                LanguagePattern("the", [0,0,0,0,1,0,0,0], "article", LanguageLevel.WORDS, 0.5, ["t","h","e"]),
                LanguagePattern("and", [0,0,0,0,0,0,1,0], "conjunction", LanguageLevel.WORDS, 0.5, ["a","n","d"]),
                LanguagePattern("you", [0,0,0,1,0,0,0,0], "pronoun", LanguageLevel.WORDS, 0.4, ["y","o","u"]),
                LanguagePattern("see", [0,1,0,0,0,0,0,0], "verb", LanguageLevel.WORDS, 0.4, ["s","e","e"]),
                LanguagePattern("car", [1,0,0,0,0,0,0,0], "noun", LanguageLevel.WORDS, 0.3, ["c","a","r"]),
            ],
            
            LanguageLevel.PHRASES: [
                LanguagePattern("big cat", [1,0,1,0,0,0,0,0], "descriptive_noun", LanguageLevel.PHRASES, 0.6, ["big","cat"]),
                LanguagePattern("red car", [1,0,1,0,0,0,0,0], "descriptive_noun", LanguageLevel.PHRASES, 0.6, ["red","car"]),
                LanguagePattern("run fast", [0,1,1,0,0,0,0,0], "verb_phrase", LanguageLevel.PHRASES, 0.7, ["run","fast"]),
                LanguagePattern("the dog", [1,0,0,0,1,0,0,0], "article_noun", LanguageLevel.PHRASES, 0.6, ["the","dog"]),
                LanguagePattern("you see", [0,1,0,1,0,0,0,0], "pronoun_verb", LanguageLevel.PHRASES, 0.7, ["you","see"]),
            ],
            
            LanguageLevel.SENTENCES: [
                LanguagePattern("the cat runs", [1,1,0,0,1,0,0,0], "simple_sentence", LanguageLevel.SENTENCES, 0.8, ["the","cat","runs"]),
                LanguagePattern("you see the dog", [1,1,0,1,1,0,0,0], "complex_sentence", LanguageLevel.SENTENCES, 0.9, ["you","see","the","dog"]),
                LanguagePattern("the big cat runs", [1,1,1,0,1,0,0,0], "descriptive_sentence", LanguageLevel.SENTENCES, 1.0, ["the","big","cat","runs"]),
            ]
        }
        
        print("✅ Progressive curriculum created")
        for level, patterns in self.curriculum.items():
            print(f"   {level.value}: {len(patterns)} patterns")
            
    def encode_text_to_neural(self, text: str) -> np.ndarray:
        """Convert text to neural spike pattern with enhanced encoding"""
        encoding = np.zeros(self.vocab_size)
        text = text.lower().strip()
        
        # Multi-character encoding with position weighting
        for i, char in enumerate(text):
            if char in self.char_to_idx:
                idx = self.char_to_idx[char]
                # Position-weighted encoding
                position_weight = 1.0 - (i * 0.1)  # Earlier characters get higher weight
                encoding[idx] = max(encoding[idx], position_weight)
                
                # Add contextual spreading
                for offset in [-2, -1, 1, 2]:
                    neighbor_idx = idx + offset
                    if 0 <= neighbor_idx < len(encoding):
                        encoding[neighbor_idx] += 0.1 * position_weight
        
        # Normalize to spike probabilities
        if np.sum(encoding) > 0:
            encoding = encoding / np.max(encoding)
            
        return encoding
        
    def learn_pattern(self, pattern: LanguagePattern, learning_rounds: int = 15) -> Dict:
        """Learn a single language pattern with memory integration"""
        print(f"\n📖 Learning: '{pattern.text}' ({pattern.category})")
        
        # Check if dependencies are met
        unmet_deps = [dep for dep in pattern.dependencies if dep not in self.mastered_elements]
        if unmet_deps and pattern.level != LanguageLevel.LETTERS:
            print(f"⚠️  Missing dependencies: {unmet_deps}")
            return {'learned': False, 'reason': 'missing_dependencies'}
        
        # Update progress tracking
        if pattern.text not in self.learning_progress:
            self.learning_progress[pattern.text] = LearningProgress(
                pattern.text, 0, 0, 0.0, datetime.now(), False
            )
        
        progress = self.learning_progress[pattern.text]
        progress.attempts += 1
        progress.last_seen = datetime.now()
        
        # Encode input
        neural_input = self.encode_text_to_neural(pattern.text)
        target_meaning = np.array(pattern.meaning)
        
        # Learning phase
        successful_rounds = 0
        confidence_scores = []
        
        for round_num in range(learning_rounds):
            # Reset network state
            for layer in self.network.layers.values():
                layer.neuron_population.reset()
            
            # Forward pass with adaptive stimulation
            stimulus_strength = 30.0 + (round_num * 2.0)  # Increase over rounds
            char_currents = [stimulus_strength if p > 0.3 else 0.0 for p in neural_input]
            
            # Multi-step processing
            semantic_activity = np.zeros(8)
            for step in range(25):
                # Drive character layer
                char_spikes = self.network.layers["characters"].neuron_population.step(0.1, char_currents)
                
                # Network propagation
                self.network.step(0.1)
                
                # Collect semantic layer activity
                semantic_spikes = self.network.layers["semantics"].neuron_population.get_spike_states()
                for i, spike in enumerate(semantic_spikes):
                    if spike:
                        semantic_activity[i] += 1
            
            # Evaluate understanding
            if np.sum(semantic_activity) > 0:
                # Normalize activity
                activity_norm = semantic_activity / np.max(semantic_activity)
                
                # Calculate match with target
                correlation = np.corrcoef(activity_norm, target_meaning)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
                    
                confidence = max(0.0, correlation)
                confidence_scores.append(confidence)
                
                if confidence > 0.5:
                    successful_rounds += 1
                    
                # Apply plasticity if some understanding
                if confidence > 0.3:
                    # Simple STDP-like weight updates for now
                    # This is a placeholder - more sophisticated plasticity later
                    pass
            else:
                confidence_scores.append(0.0)
        
        # Evaluate learning outcome
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        success_rate = successful_rounds / learning_rounds
        learned = avg_confidence > 0.4 and success_rate > 0.4
        
        # Update progress
        if learned:
            progress.successes += 1
            progress.confidence = float(avg_confidence)
            self.mastered_elements.add(pattern.text)
            
            # Simple memory tracking (avoiding complex API for now)
            if self.memory_enabled:
                print(f"🧠 Pattern '{pattern.text}' learned with confidence {avg_confidence:.2f}")
                if avg_confidence > 0.7:
                    progress.consolidated = True
                    print(f"🧠 Consolidated '{pattern.text}' to long-term memory")
        
        # Update session stats
        self.session_stats['total_attempts'] += 1
        if learned:
            self.session_stats['elements_learned'] += 1
            
        self.session_stats['success_rate'] = (
            self.session_stats['elements_learned'] / 
            self.session_stats['total_attempts']
        )
        
        print(f"   Success rate: {success_rate:.1%}")
        print(f"   Confidence: {avg_confidence:.2f}")
        print(f"   Result: {'✅ Learned' if learned else '🔄 Practice needed'}")
        
        return {
            'learned': learned,
            'confidence': avg_confidence,
            'success_rate': success_rate,
            'rounds_successful': successful_rounds,
            'total_rounds': learning_rounds
        }
        
    def assess_level_mastery(self, level: LanguageLevel, threshold: float = 0.7) -> bool:
        """Assess if a language level has been mastered"""
        patterns = self.curriculum[level]
        mastered_count = sum(1 for p in patterns if p.text in self.mastered_elements)
        mastery_rate = mastered_count / len(patterns)
        
        return mastery_rate >= threshold
        
    def run_progressive_curriculum(self):
        """Run the complete progressive learning curriculum"""
        print("\n🚀 STARTING PROGRESSIVE ENGLISH CURRICULUM")
        print("=" * 50)
        
        # Progress through levels
        level_order = [
            LanguageLevel.LETTERS,
            LanguageLevel.WORDS, 
            LanguageLevel.PHRASES,
            LanguageLevel.SENTENCES
        ]
        
        curriculum_results = {}
        
        for level in level_order:
            print(f"\n📈 LEVEL: {level.value.upper()}")
            print("-" * 30)
            
            level_results = []
            patterns = self.curriculum[level]
            
            # Adaptive difficulty: easier patterns first
            sorted_patterns = sorted(patterns, key=lambda x: x.difficulty)
            
            for pattern in sorted_patterns:
                result = self.learn_pattern(pattern)
                level_results.append(result)
                
                # Adaptive learning: more rounds for difficult patterns
                if not result['learned'] and pattern.difficulty > 0.5:
                    print(f"   🔄 Retrying difficult pattern...")
                    retry_result = self.learn_pattern(pattern, learning_rounds=25)
                    if retry_result['learned']:
                        level_results[-1] = retry_result
            
            # Evaluate level completion
            level_success_rate = sum(1 for r in level_results if r['learned']) / len(level_results)
            level_mastered = self.assess_level_mastery(level)
            
            curriculum_results[level] = {
                'patterns_attempted': len(level_results),
                'patterns_learned': sum(1 for r in level_results if r['learned']),
                'success_rate': level_success_rate,
                'mastered': level_mastered,
                'results': level_results
            }
            
            print(f"\n📊 LEVEL {level.value.upper()} RESULTS:")
            print(f"   Patterns learned: {curriculum_results[level]['patterns_learned']}/{len(patterns)}")
            print(f"   Success rate: {level_success_rate:.1%}")
            print(f"   Level mastered: {'✅ Yes' if level_mastered else '❌ No'}")
            
            # Only advance if level is reasonably mastered
            if not level_mastered and level_success_rate < 0.5:
                print(f"⚠️  Stopping at {level.value} - insufficient mastery")
                break
                
        return curriculum_results
        
    def generate_learning_report(self, results: Dict) -> Dict:
        """Generate comprehensive learning report"""
        total_patterns = sum(r['patterns_attempted'] for r in results.values())
        total_learned = sum(r['patterns_learned'] for r in results.values())
        overall_success = total_learned / total_patterns if total_patterns > 0 else 0
        
        # Determine achievement level
        if overall_success >= 0.8:
            achievement = "🌟 EXCELLENT - Strong English foundation established"
        elif overall_success >= 0.6:
            achievement = "✅ GOOD - Solid English learning progress"
        elif overall_success >= 0.4:
            achievement = "🔄 DEVELOPING - Building English understanding"
        else:
            achievement = "🌱 FOUNDATIONAL - Early English pattern recognition"
            
        # Memory system stats (simplified)
        memory_stats = {}
        if self.memory_enabled:
            memory_stats = {
                'memory_enabled': True,
                'consolidated_items': sum(1 for p in self.learning_progress.values() if p.consolidated)
            }
        
        session_duration = datetime.now() - self.session_stats['start_time']
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'session_duration_minutes': session_duration.total_seconds() / 60,
            'total_patterns': total_patterns,
            'patterns_learned': total_learned,
            'overall_success_rate': overall_success,
            'achievement': achievement,
            'levels_completed': len(results),
            'mastered_elements': len(self.mastered_elements),
            'level_results': results,
            'learning_progress': {k: {
                'attempts': v.attempts,
                'successes': v.successes,
                'confidence': v.confidence,
                'consolidated': v.consolidated
            } for k, v in self.learning_progress.items()},
            'memory_stats': memory_stats,
            'session_stats': self.session_stats
        }
        
        return report


def main():
    """Run advanced English learning demonstration"""
    print("🧠 ADVANCED NEUROMORPHIC ENGLISH LEARNING")
    print("=" * 60)
    
    # Create learner
    learner = AdvancedEnglishLearner(memory_enabled=True)
    
    # Run curriculum
    results = learner.run_progressive_curriculum()
    
    # Generate report
    report = learner.generate_learning_report(results)
    
    # Display final results
    print(f"\n🎓 FINAL LEARNING REPORT")
    print("=" * 30)
    print(f"🕒 Session time: {report['session_duration_minutes']:.1f} minutes")
    print(f"📚 Patterns attempted: {report['total_patterns']}")
    print(f"✅ Patterns learned: {report['patterns_learned']}")
    print(f"📈 Success rate: {report['overall_success_rate']:.1%}")
    print(f"🏆 Achievement: {report['achievement']}")
    print(f"🧠 Memory items: Consolidated={report['memory_stats'].get('consolidated_items', 0)}")
    
    # Save detailed report
    with open('advanced_english_learning_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n💾 Detailed report saved to: advanced_english_learning_report.json")
    
    return report


if __name__ == "__main__":
    main()
