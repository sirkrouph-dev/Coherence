#!/usr/bin/env python3
"""
NEUROMORPHIC ENGLISH LANGUAGE LEARNING SYSTEM
Teaching English patterns, words, and basic language structure
"""

import numpy as np
from core.network import NeuromorphicNetwork
import json
from datetime import datetime

class NeuromorphicEnglishLearning:
    def __init__(self):
        print("🗣️ NEUROMORPHIC ENGLISH LANGUAGE LEARNING SYSTEM")
        print("=" * 55)
        print("Teaching English patterns, words, and language structure")
        
        self.network = NeuromorphicNetwork()
        self.setup_language_network()
        
        # Language encoding system
        self.char_to_index = {}
        self.index_to_char = {}
        self.setup_character_encoding()
        
        self.language_progress = []
        
    def setup_character_encoding(self):
        """Create character to neural encoding system"""
        # Basic English alphabet + space + punctuation
        characters = "abcdefghijklmnopqrstuvwxyz .,!?"
        
        for i, char in enumerate(characters):
            self.char_to_index[char] = i
            self.index_to_char[i] = char
        
        self.vocab_size = len(characters)
        print(f"✅ Character vocabulary: {self.vocab_size} characters")
        print(f"  Characters: {list(characters)}")
        
    def setup_language_network(self):
        """Create network optimized for language learning"""
        # Language processing architecture
        self.network.add_layer("characters", 32, "lif")    # Character input (expanded)
        self.network.add_layer("phonemes", 24, "lif")      # Sound patterns
        self.network.add_layer("syllables", 16, "lif")     # Syllable recognition
        self.network.add_layer("words", 12, "lif")         # Word formation
        self.network.add_layer("meaning", 8, "lif")        # Semantic meaning
        
        # Language learning connections
        self.network.connect_layers("characters", "phonemes", "stdp",
                                  connection_probability=0.8,
                                  weight=1.5,
                                  A_plus=0.3,
                                  A_minus=0.12,
                                  tau_stdp=25.0)
        
        self.network.connect_layers("phonemes", "syllables", "stdp",
                                  connection_probability=0.9,
                                  weight=1.8,
                                  A_plus=0.35,
                                  A_minus=0.15,
                                  tau_stdp=20.0)
        
        self.network.connect_layers("syllables", "words", "stdp",
                                  connection_probability=1.0,
                                  weight=2.0,
                                  A_plus=0.4,
                                  A_minus=0.18,
                                  tau_stdp=18.0)
        
        self.network.connect_layers("words", "meaning", "stdp",
                                  connection_probability=1.0,
                                  weight=2.2,
                                  A_plus=0.45,
                                  A_minus=0.2,
                                  tau_stdp=15.0)
        
        # Feedback connections for context
        self.network.connect_layers("words", "syllables", "stdp",
                                  connection_probability=0.6,
                                  weight=1.0,
                                  A_plus=0.2,
                                  A_minus=0.08,
                                  tau_stdp=30.0)
        
        print("✅ Language network: 32→24→16→12→8 with feedback")
        print("✅ Character→Phoneme→Syllable→Word→Meaning pipeline")
        
    def create_english_curriculum(self):
        """Create progressive English learning curriculum"""
        curriculum = {
            'letters': {
                'description': 'Individual letter recognition',
                'lessons': [
                    {'input': 'a', 'meaning': [1,0,0,0,0,0,0,0], 'category': 'vowel'},
                    {'input': 'e', 'meaning': [1,0,0,0,0,0,0,0], 'category': 'vowel'},
                    {'input': 'i', 'meaning': [1,0,0,0,0,0,0,0], 'category': 'vowel'},
                    {'input': 'b', 'meaning': [0,1,0,0,0,0,0,0], 'category': 'consonant'},
                    {'input': 'c', 'meaning': [0,1,0,0,0,0,0,0], 'category': 'consonant'},
                    {'input': 'd', 'meaning': [0,1,0,0,0,0,0,0], 'category': 'consonant'}
                ]
            },
            
            'simple_words': {
                'description': 'Basic 2-3 letter words',
                'lessons': [
                    {'input': 'cat', 'meaning': [0,0,1,0,0,0,0,0], 'category': 'animal'},
                    {'input': 'dog', 'meaning': [0,0,1,0,0,0,0,0], 'category': 'animal'},
                    {'input': 'car', 'meaning': [0,0,0,1,0,0,0,0], 'category': 'object'},
                    {'input': 'red', 'meaning': [0,0,0,0,1,0,0,0], 'category': 'color'},
                    {'input': 'big', 'meaning': [0,0,0,0,0,1,0,0], 'category': 'size'},
                    {'input': 'run', 'meaning': [0,0,0,0,0,0,1,0], 'category': 'action'}
                ]
            },
            
            'word_patterns': {
                'description': 'Common English patterns',
                'lessons': [
                    {'input': 'the', 'meaning': [0,0,0,0,0,0,0,1], 'category': 'article'},
                    {'input': 'and', 'meaning': [0,0,0,0,0,0,0,1], 'category': 'conjunction'},
                    {'input': 'can', 'meaning': [0,0,0,0,0,0,1,0], 'category': 'modal'},
                    {'input': 'see', 'meaning': [0,0,0,0,0,0,1,0], 'category': 'action'},
                    {'input': 'you', 'meaning': [1,0,0,0,0,0,0,0], 'category': 'pronoun'},
                    {'input': 'are', 'meaning': [0,0,0,0,0,0,1,0], 'category': 'verb'}
                ]
            },
            
            'simple_phrases': {
                'description': 'Basic English phrases',
                'lessons': [
                    {'input': 'i am', 'meaning': [1,0,0,1,0,0,0,0], 'category': 'identity'},
                    {'input': 'you are', 'meaning': [1,0,0,1,0,0,0,0], 'category': 'identity'},
                    {'input': 'can see', 'meaning': [0,0,0,0,0,0,1,1], 'category': 'ability'},
                    {'input': 'red car', 'meaning': [0,0,0,1,1,0,0,0], 'category': 'description'},
                    {'input': 'big dog', 'meaning': [0,0,1,0,0,1,0,0], 'category': 'description'}
                ]
            }
        }
        
        return curriculum
    
    def encode_text_to_neural(self, text):
        """Convert text to neural spike patterns"""
        text = text.lower().strip()
        
        # Create distributed representation
        neural_pattern = [0.0] * 32
        
        for i, char in enumerate(text[:8]):  # Max 8 characters for network size
            if char in self.char_to_index:
                char_idx = self.char_to_index[char]
                # Distribute character across multiple neurons
                base_neuron = (char_idx * 4) % 32
                for offset in range(4):
                    if base_neuron + offset < 32:
                        neural_pattern[base_neuron + offset] = 1.0
        
        return neural_pattern
    
    def teach_english_lesson(self, lesson_data, lesson_num, category):
        """Teach single English lesson"""
        print(f"\n📚 ENGLISH LESSON {lesson_num}: {lesson_data['input']}")
        print(f"Category: {lesson_data['category']}")
        
        # Reset network
        for layer in self.network.layers.values():
            layer.neuron_population.reset()
        
        # Get network populations
        populations = {
            'characters': self.network.layers["characters"].neuron_population,
            'phonemes': self.network.layers["phonemes"].neuron_population,
            'syllables': self.network.layers["syllables"].neuron_population,
            'words': self.network.layers["words"].neuron_population,
            'meaning': self.network.layers["meaning"].neuron_population
        }
        
        connections = {
            'char_phon': self.network.connections[("characters", "phonemes")],
            'phon_syll': self.network.connections[("phonemes", "syllables")],
            'syll_word': self.network.connections[("syllables", "words")],
            'word_mean': self.network.connections[("words", "meaning")],
            'word_syll_fb': self.network.connections[("words", "syllables")]
        }
        
        # Encode input text
        neural_input = self.encode_text_to_neural(lesson_data['input'])
        target_meaning = lesson_data['meaning']
        
        # Intensive language learning
        learning_rounds = 25
        successful_rounds = 0
        
        for round_num in range(learning_rounds):
            round_success = self.language_learning_round(
                neural_input, target_meaning, populations, connections, round_num
            )
            
            if round_success:
                successful_rounds += 1
            
            # Progress check
            if (round_num + 1) % 5 == 0:
                test_result = self.test_language_understanding(
                    lesson_data['input'], target_meaning, populations, connections
                )
                success_marker = "✅" if test_result['understood'] else "❌"
                print(f"  Round {round_num + 1}: {success_marker} Understanding = {test_result['confidence']:.2f}")
        
        # Final test
        final_test = self.comprehensive_language_test(
            lesson_data['input'], target_meaning, populations, connections
        )
        
        lesson_progress = {
            'lesson': lesson_num,
            'text': lesson_data['input'],
            'category': lesson_data['category'],
            'learning_rounds': learning_rounds,
            'successful_rounds': successful_rounds,
            'final_test': final_test
        }
        
        self.language_progress.append(lesson_progress)
        
        print(f"  📊 Success: {successful_rounds}/{learning_rounds} rounds")
        print(f"  🧠 Final understanding: {'✅ LEARNED' if final_test['understood'] else '❌ Needs more practice'}")
        
        return lesson_progress
    
    def language_learning_round(self, neural_input, target_meaning, populations, connections, round_num):
        """Single round of language learning"""
        dt = 0.1
        
        # Multi-phase language processing
        phases = ['encoding', 'processing', 'understanding', 'meaning_formation']
        
        for phase_idx, phase in enumerate(phases):
            for step in range(20):
                time = round_num * 80 * dt + phase_idx * 20 * dt + step * dt
                
                if phase == 'encoding':
                    # Strong character input
                    char_currents = [120.0 if p > 0.5 else 5.0 for p in neural_input]
                    char_states = populations['characters'].step(dt, char_currents)
                    
                    # Forward propagation
                    phon_currents = self.calculate_language_currents(connections['char_phon'], char_states, 24)
                    phon_states = populations['phonemes'].step(dt, phon_currents)
                    
                    syll_currents = self.calculate_language_currents(connections['phon_syll'], phon_states, 16)
                    syll_states = populations['syllables'].step(dt, syll_currents)
                    
                    word_currents = self.calculate_language_currents(connections['syll_word'], syll_states, 12)
                    word_states = populations['words'].step(dt, word_currents)
                    
                    mean_currents = self.calculate_language_currents(connections['word_mean'], word_states, 8)
                    mean_states = populations['meaning'].step(dt, mean_currents)
                    
                elif phase == 'processing':
                    # Continue input with processing enhancement
                    char_currents = [80.0 if p > 0.5 else 0.0 for p in neural_input]
                    char_states = populations['characters'].step(dt, char_currents)
                    
                    phon_currents = self.calculate_language_currents(connections['char_phon'], char_states, 24)
                    phon_enhancement = [15.0] * 24  # Phoneme processing boost
                    phon_currents = [p + e for p, e in zip(phon_currents, phon_enhancement)]
                    phon_states = populations['phonemes'].step(dt, phon_currents)
                    
                    syll_currents = self.calculate_language_currents(connections['phon_syll'], phon_states, 16)
                    syll_states = populations['syllables'].step(dt, syll_currents)
                    
                    word_currents = self.calculate_language_currents(connections['syll_word'], syll_states, 12)
                    word_states = populations['words'].step(dt, word_currents)
                    
                    mean_currents = self.calculate_language_currents(connections['word_mean'], word_states, 8)
                    mean_states = populations['meaning'].step(dt, mean_currents)
                    
                elif phase == 'understanding':
                    # Reduced input, enhanced word processing
                    char_currents = [40.0 if p > 0.5 else 0.0 for p in neural_input]
                    char_states = populations['characters'].step(dt, char_currents)
                    
                    phon_currents = self.calculate_language_currents(connections['char_phon'], char_states, 24)
                    phon_states = populations['phonemes'].step(dt, phon_currents)
                    
                    syll_currents = self.calculate_language_currents(connections['phon_syll'], phon_states, 16)
                    syll_states = populations['syllables'].step(dt, syll_currents)
                    
                    # Enhanced word understanding
                    word_currents = self.calculate_language_currents(connections['syll_word'], syll_states, 12)
                    word_enhancement = [25.0] * 12
                    word_currents = [w + e for w, e in zip(word_currents, word_enhancement)]
                    word_states = populations['words'].step(dt, word_currents)
                    
                    # Feedback to syllables
                    syll_feedback = self.calculate_language_currents(connections['word_syll_fb'], word_states, 16)
                    syll_states = populations['syllables'].step(dt, syll_feedback)
                    
                    mean_currents = self.calculate_language_currents(connections['word_mean'], word_states, 8)
                    mean_states = populations['meaning'].step(dt, mean_currents)
                    
                else:  # meaning_formation
                    # Minimal input, target-guided meaning learning
                    char_currents = [20.0 if p > 0.5 else 0.0 for p in neural_input]
                    target_currents = [100.0 if m > 0.5 else 0.0 for m in target_meaning]
                    
                    char_states = populations['characters'].step(dt, char_currents)
                    
                    phon_currents = self.calculate_language_currents(connections['char_phon'], char_states, 24)
                    phon_states = populations['phonemes'].step(dt, phon_currents)
                    
                    syll_currents = self.calculate_language_currents(connections['phon_syll'], phon_states, 16)
                    syll_states = populations['syllables'].step(dt, syll_currents)
                    
                    word_currents = self.calculate_language_currents(connections['syll_word'], syll_states, 12)
                    word_states = populations['words'].step(dt, word_currents)
                    
                    # Target-guided meaning learning
                    mean_states = populations['meaning'].step(dt, target_currents)
                
                # Apply language STDP
                self.apply_language_stdp(char_states, phon_states, syll_states, word_states, mean_states, connections, time)
                
                self.network.step(dt)
        
        # Test round success
        return self.test_round_language_learning(neural_input, target_meaning, populations, connections)
    
    def calculate_language_currents(self, connection, pre_states, post_size):
        """Calculate synaptic currents for language processing"""
        currents = [0.0] * post_size
        
        if connection.synapse_population and hasattr(connection.synapse_population, 'synapses'):
            for (pre_idx, post_idx), synapse in connection.synapse_population.synapses.items():
                if pre_idx < len(pre_states) and post_idx < post_size and pre_states[pre_idx]:
                    current = synapse.weight * 10.0  # Language processing amplification
                    currents[post_idx] += current
        
        return currents
    
    def apply_language_stdp(self, char_states, phon_states, syll_states, word_states, mean_states, connections, time):
        """Apply STDP for language learning"""
        layer_connections = [
            (char_states, phon_states, 'char_phon'),
            (phon_states, syll_states, 'phon_syll'),
            (syll_states, word_states, 'syll_word'),
            (word_states, mean_states, 'word_mean'),
            (word_states, syll_states, 'word_syll_fb')  # Feedback
        ]
        
        for pre_states, post_states, conn_name in layer_connections:
            if conn_name in connections:
                connection = connections[conn_name]
                if connection.synapse_population and hasattr(connection.synapse_population, 'synapses'):
                    
                    for (pre_idx, post_idx), synapse in connection.synapse_population.synapses.items():
                        if pre_idx < len(pre_states) and post_idx < len(post_states):
                            
                            if pre_states[pre_idx]:
                                synapse.pre_spike(time)
                                if post_states[post_idx]:  # Language co-activation
                                    synapse.weight += synapse.A_plus * 0.18
                            
                            if post_states[post_idx]:
                                synapse.post_spike(time)
                                if not pre_states[pre_idx]:  # Unrelated activation
                                    synapse.weight -= synapse.A_minus * 0.09
                            
                            # Language weight bounds
                            synapse.weight = np.clip(synapse.weight, 0.1, 5.0)
    
    def test_round_language_learning(self, neural_input, target_meaning, populations, connections):
        """Test if language learning occurred this round"""
        # Quick test
        for pop in populations.values():
            pop.reset()
        
        meaning_activity = [0] * 8
        
        for step in range(30):
            char_currents = [60.0 if p > 0.5 else 0.0 for p in neural_input]
            char_states = populations['characters'].step(0.1, char_currents)
            
            phon_currents = self.calculate_language_currents(connections['char_phon'], char_states, 24)
            phon_states = populations['phonemes'].step(0.1, phon_currents)
            
            syll_currents = self.calculate_language_currents(connections['phon_syll'], phon_states, 16)
            syll_states = populations['syllables'].step(0.1, syll_currents)
            
            word_currents = self.calculate_language_currents(connections['syll_word'], syll_states, 12)
            word_states = populations['words'].step(0.1, word_currents)
            
            mean_currents = self.calculate_language_currents(connections['word_mean'], word_states, 8)
            mean_states = populations['meaning'].step(0.1, mean_currents)
            
            for i, spike in enumerate(mean_states):
                if spike:
                    meaning_activity[i] += 1
            
            self.network.step(0.1)
        
        # Check if meaning matches target
        if sum(meaning_activity) > 0:
            predicted = [1 if a > 0 else 0 for a in meaning_activity]
            matches = sum(1 for p, t in zip(predicted, target_meaning) if p == t)
            success = matches >= len(target_meaning) * 0.75
        else:
            success = False
        
        return success
    
    def test_language_understanding(self, text, target_meaning, populations, connections):
        """Test language understanding"""
        neural_input = self.encode_text_to_neural(text)
        
        for pop in populations.values():
            pop.reset()
        
        meaning_activity = [0] * 8
        
        for step in range(50):
            char_currents = [50.0 if p > 0.5 else 0.0 for p in neural_input]
            char_states = populations['characters'].step(0.1, char_currents)
            
            phon_currents = self.calculate_language_currents(connections['char_phon'], char_states, 24)
            phon_states = populations['phonemes'].step(0.1, phon_currents)
            
            syll_currents = self.calculate_language_currents(connections['phon_syll'], phon_states, 16)
            syll_states = populations['syllables'].step(0.1, syll_currents)
            
            word_currents = self.calculate_language_currents(connections['syll_word'], syll_states, 12)
            word_states = populations['words'].step(0.1, word_currents)
            
            mean_currents = self.calculate_language_currents(connections['word_mean'], word_states, 8)
            mean_states = populations['meaning'].step(0.1, mean_currents)
            
            for i, spike in enumerate(mean_states):
                if spike:
                    meaning_activity[i] += 1
            
            self.network.step(0.1)
        
        if sum(meaning_activity) > 0:
            # Calculate understanding confidence
            predicted = [a / max(meaning_activity) if max(meaning_activity) > 0 else 0 for a in meaning_activity]
            target_norm = [t for t in target_meaning]
            
            similarity = sum(min(p, t) for p, t in zip(predicted, target_norm))
            confidence = similarity / sum(target_norm) if sum(target_norm) > 0 else 0
            understood = confidence > 0.5
        else:
            confidence = 0.0
            understood = False
        
        return {
            'text': text,
            'understood': understood,
            'confidence': confidence,
            'meaning_activity': meaning_activity
        }
    
    def comprehensive_language_test(self, text, target_meaning, populations, connections):
        """Comprehensive language understanding test"""
        neural_input = self.encode_text_to_neural(text)
        
        for pop in populations.values():
            pop.reset()
        
        meaning_activity = [0] * 8
        
        # Extended test
        for step in range(80):
            char_currents = [40.0 if p > 0.5 else 0.0 for p in neural_input]
            char_states = populations['characters'].step(0.1, char_currents)
            
            phon_currents = self.calculate_language_currents(connections['char_phon'], char_states, 24)
            phon_states = populations['phonemes'].step(0.1, phon_currents)
            
            syll_currents = self.calculate_language_currents(connections['phon_syll'], phon_states, 16)
            syll_states = populations['syllables'].step(0.1, syll_currents)
            
            word_currents = self.calculate_language_currents(connections['syll_word'], syll_states, 12)
            word_states = populations['words'].step(0.1, word_currents)
            
            mean_currents = self.calculate_language_currents(connections['word_mean'], word_states, 8)
            mean_states = populations['meaning'].step(0.1, mean_currents)
            
            for i, spike in enumerate(mean_states):
                if spike:
                    meaning_activity[i] += 1
            
            self.network.step(0.1)
        
        # Detailed analysis
        total_meaning_activity = sum(meaning_activity)
        
        if total_meaning_activity > 0:
            confidence = sum(min(meaning_activity[i], target_meaning[i] * 10) for i in range(8)) / max(10, total_meaning_activity)
            understood = confidence > 0.4
        else:
            confidence = 0.0
            understood = False
        
        return {
            'text': text,
            'understood': understood,
            'confidence': confidence,
            'meaning_activity': meaning_activity,
            'total_activity': total_meaning_activity
        }
    
    def run_english_learning_curriculum(self):
        """Run complete English learning curriculum"""
        print("Starting neuromorphic English language learning...")
        
        curriculum = self.create_english_curriculum()
        print(f"✅ Created English curriculum with {len(curriculum)} stages")
        
        # Progressive language learning
        lesson_num = 1
        
        for stage_name, stage in curriculum.items():
            print(f"\n🎓 LANGUAGE STAGE: {stage['description']}")
            print("-" * 50)
            
            for lesson_data in stage['lessons']:
                lesson_progress = self.teach_english_lesson(lesson_data, lesson_num, stage_name)
                lesson_num += 1
        
        # Final English assessment
        print(f"\n🗣️ FINAL ENGLISH LANGUAGE ASSESSMENT")
        print("=" * 42)
        
        words_learned = 0
        total_lessons = len(self.language_progress)
        confidence_scores = []
        
        for progress in self.language_progress:
            final_test = progress['final_test']
            if final_test['understood']:
                words_learned += 1
                confidence_scores.append(final_test['confidence'])
                print(f"✅ '{final_test['text']}': UNDERSTOOD (confidence {final_test['confidence']:.2f})")
            else:
                print(f"❌ '{final_test['text']}': Needs more practice")
        
        # Language learning metrics
        language_success_rate = words_learned / total_lessons
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
        print(f"\n📊 ENGLISH LEARNING RESULTS")
        print("-" * 30)
        print(f"Words/phrases learned: {words_learned}/{total_lessons} ({language_success_rate:.1%})")
        print(f"Average confidence: {avg_confidence:.3f}")
        
        # English language assessment
        if language_success_rate >= 0.6 and avg_confidence >= 0.5:
            print(f"\n🌟 ENGLISH LEARNING: HIGHLY SUCCESSFUL!")
            print(f"✅ Strong language pattern recognition")
            print(f"✅ Good semantic understanding")
            print(f"✅ Neuromorphic language processing working")
            language_level = "ADVANCED_ENGLISH"
        elif language_success_rate >= 0.4:
            print(f"\n✅ ENGLISH LEARNING: SUCCESSFUL!")
            print(f"✅ Basic language patterns learned")
            language_level = "BASIC_ENGLISH"
        elif language_success_rate > 0.2:
            print(f"\n🟡 ENGLISH LEARNING: EMERGING!")
            print(f"🔄 Some language recognition developing")
            language_level = "EMERGING_ENGLISH"
        else:
            print(f"\n🌱 ENGLISH LEARNING: FOUNDATIONAL")
            print(f"🔄 Building language foundations")
            language_level = "FOUNDATIONAL_ENGLISH"
        
        # Save English learning report
        report = {
            'timestamp': datetime.now().isoformat(),
            'learning_type': 'neuromorphic_english_language',
            'total_lessons': total_lessons,
            'words_learned': words_learned,
            'language_success_rate': language_success_rate,
            'average_confidence': avg_confidence,
            'language_level': language_level,
            'vocabulary_size': self.vocab_size,
            'language_progress': self.language_progress
        }
        
        with open('english_learning_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n💾 English learning report saved: english_learning_report.json")
        
        return language_success_rate >= 0.3

if __name__ == "__main__":
    english_system = NeuromorphicEnglishLearning()
    success = english_system.run_english_learning_curriculum()
    
    if success:
        print(f"\n🗣️ NEUROMORPHIC ENGLISH LEARNING: SUCCESS!")
        print(f"The system demonstrates language learning capabilities!")
    else:
        print(f"\n📚 Continuing English language development...")
