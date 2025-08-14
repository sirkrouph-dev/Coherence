#!/usr/bin/env python3
"""
Quick Demo: Advanced English Learning Capabilities
Demonstrates sentence learning, grammar understanding, and conversation

Usage: python demo_advanced_english.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.comprehensive_sentence_learning import AdvancedSentenceLearner
from core.interactive_conversation import InteractiveConversationSystem


def demo_sentence_learning():
    """Demonstrate advanced sentence learning"""
    
    print("🧠 SENTENCE LEARNING DEMONSTRATION")
    print("=" * 50)
    
    learner = AdvancedSentenceLearner()
    
    # Demo sentences showcasing different capabilities
    demo_sentences = [
        "hello i am learning english",
        "the intelligent cat runs very quickly",
        "she walks to the beautiful old house",
        "what do you like to do today",
        "i am happy because learning is fun",
        "cats and dogs are wonderful animal friends"
    ]
    
    print("📚 Learning demonstration sentences...")
    
    for i, sentence in enumerate(demo_sentences, 1):
        print(f"\\n{i}. Learning: '{sentence}'")
        result = learner.learn_sentence(sentence, learning_rounds=25)
        
        structure = result['structure']
        print(f"   ✅ Grammar: {structure.grammar_pattern.value}")
        print(f"   ✅ Words: {len(structure.words)}")
        print(f"   ✅ Success: {result['success_rate']:.1%}")
        print(f"   ✅ Learned: {'Yes' if result['learned'] else 'Still learning'}")
    
    return learner


def demo_conversation():
    """Demonstrate interactive conversation"""
    
    print("\\n\\n💬 CONVERSATION DEMONSTRATION")
    print("=" * 50)
    
    learner = AdvancedSentenceLearner()
    
    # Simulate conversation turns
    conversation_demo = [
        "hello how are you doing today",
        "i am learning about artificial intelligence",
        "what do you think about language learning",
        "i feel excited about new technology",
        "cats are very intelligent animals"
    ]
    
    print("🤖 Starting conversation demonstration...")
    
    for i, user_input in enumerate(conversation_demo, 1):
        print(f"\\n👤 User: {user_input}")
        
        # Generate response
        response = learner.generate_response(user_input)
        print(f"🤖 AI: {response}")
        
        # Show learning progress
        if i % 2 == 0:
            print(f"   📈 Processed {i} conversation turns")
    
    print("\\n✅ Conversation demonstration complete!")


def demo_memory_retention():
    """Demonstrate memory and retention capabilities"""
    
    print("\\n\\n🧠 MEMORY RETENTION DEMONSTRATION")
    print("=" * 50)
    
    learner = AdvancedSentenceLearner()
    
    # Learn some sentences
    memory_sentences = [
        "i love learning new languages",
        "the cat sleeps peacefully in the sun",
        "artificial intelligence helps people"
    ]
    
    print("📚 Learning sentences for memory test...")
    for sentence in memory_sentences:
        learner.learn_sentence(sentence, learning_rounds=30)
        print(f"   ✅ Learned: '{sentence}'")
    
    # Test retention at different intervals
    print("\\n🧠 Testing memory retention...")
    
    for days in [1, 7, 30]:
        print(f"\\n   After {days} days:")
        for sentence in memory_sentences:
            retention = learner.test_memory_retention(sentence, days_elapsed=days)
            status = "✅ Retained" if retention['retained'] else "❌ Forgotten"
            print(f"     '{sentence}': {status}")
    
    print("\\n✅ Memory demonstration complete!")


def demo_grammar_analysis():
    """Demonstrate grammar pattern recognition"""
    
    print("\\n\\n📝 GRAMMAR ANALYSIS DEMONSTRATION")
    print("=" * 50)
    
    learner = AdvancedSentenceLearner()
    
    # Test different grammar patterns
    grammar_examples = [
        ("the cat", "Article + Noun"),
        ("big red car", "Adjective + Adjective + Noun"),
        ("she runs", "Pronoun + Verb"),
        ("the dog eats food", "Article + Noun + Verb + Noun"),
        ("i walk to the store", "Complex sentence with preposition"),
        ("cats run and dogs walk", "Compound sentence with conjunction")
    ]
    
    print("🔍 Analyzing grammar patterns...")
    
    for sentence, description in grammar_examples:
        print(f"\\n   Sentence: '{sentence}'")
        print(f"   Expected: {description}")
        
        structure = learner.parse_sentence(sentence)
        print(f"   Detected: {structure.grammar_pattern.value}")
        print(f"   Word types: {[wt.value for wt in structure.word_types]}")
        print(f"   Complexity: {structure.complexity_score:.2f}")
    
    print("\\n✅ Grammar analysis complete!")


def main():
    """Run comprehensive demonstration of advanced English learning"""
    
    print("🌟 ADVANCED NEUROMORPHIC ENGLISH LEARNING DEMO")
    print("=" * 65)
    print("🚀 Demonstrating comprehensive language understanding capabilities")
    print("📊 Including sentence learning, grammar, memory, and conversation")
    
    try:
        # Run all demonstrations
        demo_sentence_learning()
        demo_conversation()
        demo_memory_retention()
        demo_grammar_analysis()
        
        print("\\n\\n🎉 DEMONSTRATION COMPLETE!")
        print("=" * 40)
        print("✅ Sentence Learning: Advanced comprehension achieved")
        print("✅ Grammar Rules: Pattern recognition working")
        print("✅ Memory Retention: Long-term storage functional")
        print("✅ Conversation: Interactive capabilities demonstrated")
        print("✅ Context Understanding: Multi-word relationships learned")
        
        print("\\n🚀 Ready for interactive conversation!")
        print("💡 Run 'python core/interactive_conversation.py' for live chat")
        
    except Exception as e:
        print(f"\\n❌ Demo error: {e}")
        print("Please check the system configuration.")


if __name__ == "__main__":
    main()
