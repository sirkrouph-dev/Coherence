#!/usr/bin/env python3
"""
Interactive Neuromorphic Conversation System
Real-time conversation with advanced language understanding

Features:
- Real-time sentence processing
- Context-aware responses  
- Grammar pattern recognition
- Memory-based conversations
- Learning from interactions
"""

import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.comprehensive_sentence_learning import AdvancedSentenceLearner


class InteractiveConversationSystem:
    """Interactive neuromorphic conversation system"""
    
    def __init__(self):
        """Initialize interactive conversation system"""
        print("🤖 NEUROMORPHIC CONVERSATION SYSTEM")
        print("=" * 50)
        print("💬 Real-time sentence understanding")
        print("🧠 Context-aware responses")
        print("📚 Learning from conversations")
        print("🔗 Memory-based interactions")
        
        # Initialize sentence learner
        self.learner = AdvancedSentenceLearner()
        
        # Conversation state
        self.conversation_history = []
        self.current_context = {
            'topic': None,
            'mood': 'neutral',
            'known_facts': set(),
            'user_preferences': {}
        }
        
        # Response templates by category
        self.response_templates = {
            'greeting': [
                "Hello! It's great to talk with you.",
                "Hi there! How can I help you today?",
                "Hello! I'm excited to learn from our conversation."
            ],
            'question': [
                "That's an interesting question. Let me think about it.",
                "I'm learning about that topic. Can you tell me more?",
                "That makes me curious. What do you think?",
                "I'd like to understand better. Could you explain?"
            ],
            'emotion': [
                "I can sense the emotion in what you're saying.",
                "Emotions are important. How does that make you feel?",
                "I'm learning to understand feelings better."
            ],
            'learning': [
                "I'm learning something new from what you said.",
                "That adds to my understanding. Thank you.",
                "I'll remember that for our future conversations."
            ],
            'default': [
                "That's fascinating. Tell me more.",
                "I'm processing what you said and learning from it.",
                "Your words are helping me understand language better."
            ]
        }
        
        print("\\n✅ Interactive system ready!")
        print("Type 'exit' to end the conversation\\n")
        
    def analyze_user_input(self, user_input: str) -> Dict:
        """Analyze user input for content and intent"""
        
        # Parse sentence structure
        structure = self.learner.parse_sentence(user_input)
        
        # Detect intent categories
        intent = self.detect_intent(user_input, structure)
        
        # Extract key information
        entities = self.extract_entities(structure)
        
        # Assess emotional content
        emotion = self.detect_emotion(user_input, structure)
        
        return {
            'structure': structure,
            'intent': intent,
            'entities': entities,
            'emotion': emotion,
            'complexity': structure.complexity_score
        }
        
    def detect_intent(self, text: str, structure) -> str:
        """Detect user intent from text and structure"""
        
        words = text.lower().split()
        
        # Greeting detection
        if any(word in words for word in ['hello', 'hi', 'hey', 'greetings']):
            return 'greeting'
            
        # Question detection
        if any(word in words for word in ['what', 'who', 'where', 'when', 'why', 'how', '?']):
            return 'question'
            
        # Learning/teaching detection
        if any(word in words for word in ['learn', 'teach', 'explain', 'understand']):
            return 'learning'
            
        # Emotional content
        if any(word in words for word in ['feel', 'happy', 'sad', 'angry', 'excited', 'worried']):
            return 'emotion'
            
        return 'statement'
        
    def extract_entities(self, structure) -> Dict:
        """Extract important entities from sentence structure"""
        
        entities = {
            'nouns': [],
            'actions': [],
            'descriptors': [],
            'locations': []
        }
        
        for i, word in enumerate(structure.words):
            word_type = structure.word_types[i]
            
            if word_type.value == 'noun':
                entities['nouns'].append(word)
            elif word_type.value == 'verb':
                entities['actions'].append(word)
            elif word_type.value == 'adjective':
                entities['descriptors'].append(word)
            elif word_type.value == 'preposition' and i + 1 < len(structure.words):
                entities['locations'].append(f"{word} {structure.words[i+1]}")
                
        return entities
        
    def detect_emotion(self, text: str, structure) -> str:
        """Detect emotional content in user input"""
        
        positive_words = ['happy', 'good', 'great', 'love', 'like', 'excited', 'wonderful']
        negative_words = ['sad', 'bad', 'hate', 'angry', 'worried', 'terrible', 'awful']
        
        words = text.lower().split()
        
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
            
    def generate_contextual_response(self, analysis: Dict) -> str:
        """Generate contextually appropriate response"""
        
        intent = analysis['intent']
        emotion = analysis['emotion']
        entities = analysis['entities']
        
        # Update context with new information
        self.update_conversation_context(analysis)
        
        # Choose response category
        if intent == 'greeting':
            response_category = 'greeting'
        elif intent == 'question':
            response_category = 'question'
        elif intent == 'emotion' or emotion != 'neutral':
            response_category = 'emotion'
        elif intent == 'learning':
            response_category = 'learning'
        else:
            response_category = 'default'
            
        # Select and personalize response
        base_response = np.random.choice(self.response_templates[response_category])
        
        # Add contextual elements
        personalized_response = self.personalize_response(base_response, analysis)
        
        return personalized_response
        
    def personalize_response(self, base_response: str, analysis: Dict) -> str:
        """Add personal context to response"""
        
        entities = analysis['entities']
        
        # Add specific references to mentioned topics
        if entities['nouns']:
            main_noun = entities['nouns'][0]
            if main_noun in ['book', 'books']:
                base_response += f" Books are fascinating to learn about."
            elif main_noun in ['cat', 'dog', 'animal']:
                base_response += f" Animals are interesting creatures."
            elif main_noun in ['food', 'eat']:
                base_response += f" Food and nutrition are important topics."
                
        # Add action-based responses
        if entities['actions']:
            main_action = entities['actions'][0]
            if main_action in ['run', 'walk', 'move']:
                base_response += f" Movement and activity are good for health."
            elif main_action in ['read', 'learn', 'study']:
                base_response += f" Learning is one of my favorite activities."
                
        return base_response
        
    def update_conversation_context(self, analysis: Dict):
        """Update conversation context with new information"""
        
        # Update current topic
        if analysis['entities']['nouns']:
            self.current_context['topic'] = analysis['entities']['nouns'][0]
            
        # Update mood
        self.current_context['mood'] = analysis['emotion']
        
        # Add new facts
        for noun in analysis['entities']['nouns']:
            self.current_context['known_facts'].add(noun)
            
        # Track preferences based on positive language
        if analysis['emotion'] == 'positive' and analysis['entities']['nouns']:
            for noun in analysis['entities']['nouns']:
                if noun not in self.current_context['user_preferences']:
                    self.current_context['user_preferences'][noun] = 1
                else:
                    self.current_context['user_preferences'][noun] += 1
                    
    def run_conversation(self):
        """Run interactive conversation loop"""
        
        print("🤖: Hello! I'm a neuromorphic AI learning to understand language.")
        print("🤖: I can discuss topics, answer questions, and learn from our conversation.")
        print("🤖: What would you like to talk about?\\n")
        
        conversation_count = 0
        
        while True:
            # Get user input
            try:
                user_input = input("👤: ").strip()
            except KeyboardInterrupt:
                print("\\n\\n🤖: Thank you for the conversation! Goodbye!")
                break
                
            if not user_input:
                continue
                
            if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                print("🤖: Thank you for the wonderful conversation! I learned a lot.")
                self.save_conversation_summary()
                break
                
            conversation_count += 1
            
            # Analyze user input
            print("🧠 [Analyzing...]", end="", flush=True)
            analysis = self.analyze_user_input(user_input)
            
            # Learn from the input
            learning_result = self.learner.learn_sentence(user_input, learning_rounds=15)
            
            # Generate response
            response = self.generate_contextual_response(analysis)
            
            # Clear analysis indicator and show response
            print("\\r" + " " * 20 + "\\r", end="", flush=True)
            print(f"🤖: {response}")
            
            # Store conversation turn
            self.conversation_history.append({
                'turn': conversation_count,
                'user_input': user_input,
                'analysis': {
                    'intent': analysis['intent'],
                    'emotion': analysis['emotion'],
                    'entities': analysis['entities'],
                    'grammar_pattern': analysis['structure'].grammar_pattern.value
                },
                'learning_success': learning_result['learned'],
                'response': response,
                'timestamp': datetime.now().isoformat()
            })
            
            # Show learning progress occasionally
            if conversation_count % 3 == 0:
                self.show_learning_progress()
                
            print()  # Add spacing
            
    def show_learning_progress(self):
        """Show current learning progress"""
        
        total_sentences = len(self.conversation_history)
        learned_sentences = sum(1 for turn in self.conversation_history 
                               if turn['learning_success'])
        
        if total_sentences > 0:
            success_rate = learned_sentences / total_sentences
            print(f"\\n📈 Learning Progress: {learned_sentences}/{total_sentences} sentences learned ({success_rate:.1%})")
            
            # Show current context
            if self.current_context['topic']:
                print(f"🎯 Current Topic: {self.current_context['topic']}")
            if self.current_context['user_preferences']:
                top_interest = max(self.current_context['user_preferences'].items(), 
                                 key=lambda x: x[1])
                print(f"💡 You seem interested in: {top_interest[0]}")
                
    def save_conversation_summary(self):
        """Save conversation summary and learning statistics"""
        
        summary = {
            'session_info': {
                'start_time': self.conversation_history[0]['timestamp'] if self.conversation_history else None,
                'end_time': datetime.now().isoformat(),
                'total_turns': len(self.conversation_history)
            },
            'learning_statistics': {
                'sentences_learned': sum(1 for turn in self.conversation_history if turn['learning_success']),
                'grammar_patterns_encountered': list(set(turn['analysis']['grammar_pattern'] 
                                                        for turn in self.conversation_history)),
                'emotions_detected': list(set(turn['analysis']['emotion'] 
                                            for turn in self.conversation_history)),
                'intents_processed': list(set(turn['analysis']['intent'] 
                                            for turn in self.conversation_history))
            },
            'conversation_context': self.current_context,
            'conversation_history': self.conversation_history
        }
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_session_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
            
        print(f"\\n💾 Conversation saved: {filename}")
        
        # Show final statistics
        total_turns = len(self.conversation_history)
        learned_count = summary['learning_statistics']['sentences_learned']
        
        print(f"\\n📊 CONVERSATION SUMMARY")
        print(f"   Total turns: {total_turns}")
        print(f"   Sentences learned: {learned_count}/{total_turns}")
        print(f"   Grammar patterns: {len(summary['learning_statistics']['grammar_patterns_encountered'])}")
        print(f"   Topics discussed: {len(self.current_context['known_facts'])}")
        
        if total_turns > 0:
            success_rate = learned_count / total_turns
            print(f"   Learning success: {success_rate:.1%}")
            
            if success_rate >= 0.8:
                print("   🌟 Excellent conversation and learning!")
            elif success_rate >= 0.6:
                print("   ✅ Good learning progress!")
            else:
                print("   🌱 Building language understanding!")


def main():
    """Run interactive conversation system"""
    
    try:
        # Create and run conversation system
        conversation_system = InteractiveConversationSystem()
        conversation_system.run_conversation()
        
    except Exception as e:
        print(f"\\n❌ Error: {e}")
        print("Conversation system encountered an issue.")
        
    return conversation_system


if __name__ == "__main__":
    system = main()
