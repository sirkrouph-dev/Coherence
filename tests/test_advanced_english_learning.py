#!/usr/bin/env python3
"""
Comprehensive Advanced English Learning Test Suite
Testing all advanced language capabilities

Test Categories:
- Sentence Comprehension
- Grammar Rule Learning
- Context Understanding  
- Memory Consolidation
- Conversation Capabilities
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.comprehensive_sentence_learning import AdvancedSentenceLearner
from core.interactive_conversation import InteractiveConversationSystem
import json
from datetime import datetime


def test_sentence_comprehension():
    """Test advanced sentence comprehension capabilities"""
    
    print("🧪 TESTING SENTENCE COMPREHENSION")
    print("=" * 40)
    
    learner = AdvancedSentenceLearner()
    
    # Test sentences of varying complexity
    test_sentences = [
        # Simple sentences
        ("cat runs", "simple"),
        ("dog eats", "simple"),
        
        # Complex sentences  
        ("the big cat runs quickly", "complex"),
        ("she walks to the house", "complex"),
        
        # Compound sentences
        ("the cat runs and the dog walks", "compound"),
        ("i eat food but she drinks water", "compound"),
        
        # Questions
        ("what do you like", "question"),
        ("where is the book", "question")
    ]
    
    results = []
    
    for sentence, expected_type in test_sentences:
        print(f"\\n🔍 Testing: '{sentence}'")
        
        # Learn the sentence
        result = learner.learn_sentence(sentence, learning_rounds=30)
        
        # Check if it learned successfully
        learned = result['learned']
        success_rate = result['success_rate']
        
        print(f"   Result: {'✅ PASS' if learned else '❌ FAIL'} ({success_rate:.1%})")
        
        results.append({
            'sentence': sentence,
            'expected_type': expected_type,
            'learned': learned,
            'success_rate': success_rate
        })
    
    # Calculate overall performance
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r['learned'])
    overall_success = passed_tests / total_tests
    
    print(f"\\n📊 SENTENCE COMPREHENSION RESULTS")
    print(f"   Tests passed: {passed_tests}/{total_tests} ({overall_success:.1%})")
    
    return {
        'test_name': 'sentence_comprehension',
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'success_rate': overall_success,
        'details': results
    }


def test_grammar_rules():
    """Test grammar rule learning and recognition"""
    
    print("\\n🧪 TESTING GRAMMAR RULE LEARNING")
    print("=" * 40)
    
    learner = AdvancedSentenceLearner()
    
    # Test different grammar patterns
    grammar_tests = [
        # Noun phrases
        ("the cat", "noun_phrase"),
        ("big dog", "noun_phrase"),
        
        # Verb phrases  
        ("runs quickly", "verb_phrase"),
        ("eats food", "verb_phrase"),
        
        # Simple sentences
        ("cat runs", "simple_sentence"),
        ("dog eats", "simple_sentence"),
        
        # Complex sentences
        ("the big cat runs", "complex_sentence"),
        ("she walks slowly", "complex_sentence")
    ]
    
    results = []
    grammar_patterns_learned = set()
    
    for sentence, expected_grammar in grammar_tests:
        print(f"\\n🔍 Testing grammar: '{sentence}'")
        
        # Parse and learn
        structure = learner.parse_sentence(sentence)
        result = learner.learn_sentence(sentence, learning_rounds=25)
        
        # Check grammar recognition
        detected_grammar = structure.grammar_pattern.value
        learned = result['learned']
        
        print(f"   Expected: {expected_grammar}")
        print(f"   Detected: {detected_grammar}")
        print(f"   Learned: {'✅ YES' if learned else '❌ NO'}")
        
        if learned:
            grammar_patterns_learned.add(detected_grammar)
        
        results.append({
            'sentence': sentence,
            'expected_grammar': expected_grammar,
            'detected_grammar': detected_grammar,
            'learned': learned,
            'grammar_correct': expected_grammar in detected_grammar or detected_grammar in expected_grammar
        })
    
    # Calculate performance
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r['learned'])
    grammar_accuracy = sum(1 for r in results if r['grammar_correct']) / total_tests
    
    print(f"\\n📊 GRAMMAR RULE RESULTS")
    print(f"   Tests passed: {passed_tests}/{total_tests}")
    print(f"   Grammar accuracy: {grammar_accuracy:.1%}")
    print(f"   Patterns learned: {len(grammar_patterns_learned)}")
    
    return {
        'test_name': 'grammar_rules',
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'grammar_accuracy': grammar_accuracy,
        'patterns_learned': len(grammar_patterns_learned),
        'details': results
    }


def test_context_understanding():
    """Test multi-word semantic relationships and context"""
    
    print("\\n🧪 TESTING CONTEXT UNDERSTANDING")
    print("=" * 40)
    
    learner = AdvancedSentenceLearner()
    
    # Test context-dependent understanding
    context_tests = [
        # Related sentence pairs
        ("the cat is big", "the big cat runs"),
        ("i like books", "the book is good"),
        ("she walks fast", "fast walking is good"),
        ("dogs eat food", "the dog eats quickly")
    ]
    
    results = []
    
    for sentence1, sentence2 in context_tests:
        print(f"\\n🔍 Testing context: '{sentence1}' + '{sentence2}'")
        
        # Learn first sentence
        result1 = learner.learn_sentence(sentence1, learning_rounds=20)
        
        # Learn second sentence (should benefit from context)
        result2 = learner.learn_sentence(sentence2, learning_rounds=20)
        
        # Check if context helped
        context_benefit = result2['success_rate'] >= result1['success_rate']
        both_learned = result1['learned'] and result2['learned']
        
        print(f"   Sentence 1: {'✅' if result1['learned'] else '❌'} ({result1['success_rate']:.1%})")
        print(f"   Sentence 2: {'✅' if result2['learned'] else '❌'} ({result2['success_rate']:.1%})")
        print(f"   Context benefit: {'✅' if context_benefit else '❌'}")
        
        results.append({
            'sentence_pair': (sentence1, sentence2),
            'both_learned': both_learned,
            'context_benefit': context_benefit,
            'success_rates': (result1['success_rate'], result2['success_rate'])
        })
    
    # Calculate performance
    total_pairs = len(results)
    successful_pairs = sum(1 for r in results if r['both_learned'])
    context_benefits = sum(1 for r in results if r['context_benefit'])
    
    print(f"\\n📊 CONTEXT UNDERSTANDING RESULTS")
    print(f"   Successful pairs: {successful_pairs}/{total_pairs}")
    print(f"   Context benefits: {context_benefits}/{total_pairs}")
    
    return {
        'test_name': 'context_understanding',
        'total_pairs': total_pairs,
        'successful_pairs': successful_pairs,
        'context_benefits': context_benefits,
        'details': results
    }


def test_memory_consolidation():
    """Test long-term memory retention"""
    
    print("\\n🧪 TESTING MEMORY CONSOLIDATION")
    print("=" * 40)
    
    learner = AdvancedSentenceLearner()
    
    # Learn base sentences
    base_sentences = [
        "the cat runs fast",
        "i like reading books", 
        "she walks to school",
        "dogs are good friends"
    ]
    
    print("📚 Learning base sentences...")
    for sentence in base_sentences:
        learner.learn_sentence(sentence, learning_rounds=30)
    
    # Test retention at different time intervals
    retention_tests = [
        (1, "1 day"),
        (3, "3 days"), 
        (7, "1 week"),
        (14, "2 weeks")
    ]
    
    results = []
    
    for days, description in retention_tests:
        print(f"\\n🧠 Testing retention after {description}")
        
        retention_results = []
        for sentence in base_sentences:
            retention_result = learner.test_memory_retention(sentence, days_elapsed=days)
            retention_results.append(retention_result)
            
        retained_count = sum(1 for r in retention_results if r['retained'])
        retention_rate = retained_count / len(base_sentences)
        
        print(f"   Retained: {retained_count}/{len(base_sentences)} ({retention_rate:.1%})")
        
        results.append({
            'days_elapsed': days,
            'description': description,
            'retained_count': retained_count,
            'retention_rate': retention_rate,
            'details': retention_results
        })
    
    print(f"\\n📊 MEMORY CONSOLIDATION RESULTS")
    for result in results:
        print(f"   {result['description']}: {result['retention_rate']:.1%}")
    
    return {
        'test_name': 'memory_consolidation',
        'base_sentences': len(base_sentences),
        'retention_tests': results
    }


def test_conversation_capabilities():
    """Test interactive conversation capabilities"""
    
    print("\\n🧪 TESTING CONVERSATION CAPABILITIES")
    print("=" * 40)
    
    learner = AdvancedSentenceLearner()
    
    # Test different conversation scenarios
    conversation_tests = [
        # Greetings
        ("hello how are you", "greeting"),
        ("hi there friend", "greeting"),
        
        # Questions
        ("what do you like", "question"),
        ("where is the book", "question"),
        
        # Emotions
        ("i am very happy", "emotion"),
        ("she feels sad today", "emotion"),
        
        # Statements
        ("the weather is nice", "statement"),
        ("cats are interesting animals", "statement")
    ]
    
    results = []
    
    for user_input, expected_category in conversation_tests:
        print(f"\\n💬 Testing: '{user_input}'")
        
        # Generate response
        response = learner.generate_response(user_input)
        
        # Check if response is appropriate (non-empty and contextual)
        response_appropriate = (len(response) > 10 and 
                              any(word in response.lower() for word in 
                                  ['interesting', 'understand', 'learn', 'feel', 'think', 'hello']))
        
        print(f"   Response: '{response}'")
        print(f"   Appropriate: {'✅' if response_appropriate else '❌'}")
        
        results.append({
            'user_input': user_input,
            'expected_category': expected_category,
            'response': response,
            'appropriate': response_appropriate
        })
    
    # Calculate performance
    total_tests = len(results)
    appropriate_responses = sum(1 for r in results if r['appropriate'])
    
    print(f"\\n📊 CONVERSATION CAPABILITIES RESULTS")
    print(f"   Appropriate responses: {appropriate_responses}/{total_tests} ({appropriate_responses/total_tests:.1%})")
    
    return {
        'test_name': 'conversation_capabilities',
        'total_tests': total_tests,
        'appropriate_responses': appropriate_responses,
        'success_rate': appropriate_responses / total_tests,
        'details': results
    }


def run_comprehensive_test_suite():
    """Run all advanced language learning tests"""
    
    print("🧪 COMPREHENSIVE ADVANCED ENGLISH LEARNING TEST SUITE")
    print("=" * 65)
    print("🔬 Testing all advanced language capabilities")
    print("📊 Generating comprehensive performance report")
    
    # Run all tests
    test_results = []
    
    try:
        # 1. Sentence Comprehension
        result1 = test_sentence_comprehension()
        test_results.append(result1)
        
        # 2. Grammar Rules
        result2 = test_grammar_rules()
        test_results.append(result2)
        
        # 3. Context Understanding
        result3 = test_context_understanding()
        test_results.append(result3)
        
        # 4. Memory Consolidation
        result4 = test_memory_consolidation()
        test_results.append(result4)
        
        # 5. Conversation Capabilities
        result5 = test_conversation_capabilities()
        test_results.append(result5)
        
    except Exception as e:
        print(f"\\n❌ Test error: {e}")
        return None
    
    # Calculate overall performance
    total_tests = sum(r.get('total_tests', 0) for r in test_results if 'total_tests' in r)
    total_passed = sum(r.get('passed_tests', 0) for r in test_results if 'passed_tests' in r)
    overall_success = total_passed / total_tests if total_tests > 0 else 0
    
    # Determine achievement level
    if overall_success >= 0.85:
        achievement = "🏆 OUTSTANDING - Advanced language mastery"
    elif overall_success >= 0.70:
        achievement = "🌟 EXCELLENT - Strong language understanding"
    elif overall_success >= 0.55:
        achievement = "✅ GOOD - Solid language capabilities"
    elif overall_success >= 0.40:
        achievement = "🔄 DEVELOPING - Building language skills"
    else:
        achievement = "🌱 FOUNDATIONAL - Early language learning"
    
    # Generate comprehensive report
    report = {
        'test_suite_info': {
            'name': 'Comprehensive Advanced English Learning Test',
            'timestamp': datetime.now().isoformat(),
            'total_test_categories': len(test_results)
        },
        'overall_performance': {
            'total_tests': total_tests,
            'total_passed': total_passed,
            'overall_success_rate': overall_success,
            'achievement_level': achievement
        },
        'detailed_results': test_results,
        'summary_by_category': {
            result['test_name']: {
                'success_rate': result.get('success_rate', result.get('passed_tests', 0) / result.get('total_tests', 1)),
                'key_metrics': {k: v for k, v in result.items() 
                              if k not in ['test_name', 'details']}
            }
            for result in test_results
        }
    }
    
    # Display final results
    print(f"\\n🎓 COMPREHENSIVE TEST RESULTS")
    print("=" * 45)
    print(f"📊 Overall Performance: {total_passed}/{total_tests} ({overall_success:.1%})")
    print(f"🏆 Achievement Level: {achievement}")
    
    print(f"\\n📋 Results by Category:")
    for result in test_results:
        category = result['test_name'].replace('_', ' ').title()
        if 'success_rate' in result:
            rate = result['success_rate']
        elif 'passed_tests' in result and 'total_tests' in result:
            rate = result['passed_tests'] / result['total_tests']
        else:
            rate = 0.0
        print(f"   {category}: {rate:.1%}")
    
    # Save comprehensive report
    with open('comprehensive_test_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\\n💾 Comprehensive test report saved: comprehensive_test_report.json")
    
    return report


def main():
    """Run comprehensive test suite"""
    
    try:
        report = run_comprehensive_test_suite()
        
        if report:
            print(f"\\n✅ All tests completed successfully!")
            print(f"🎯 Final Achievement: {report['overall_performance']['achievement_level']}")
        else:
            print(f"\\n❌ Test suite encountered errors")
            
    except Exception as e:
        print(f"\\n❌ Test suite error: {e}")
        
    return report


if __name__ == "__main__":
    main()
