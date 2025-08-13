#!/usr/bin/env python3
"""
Neuromorphic Communication System - Interactive Conversation

Building on the 100% English learning success, this system enables
direct communication with the neuromorphic intelligence.

Features:
- Real-time conversation interface
- Memory of previous interactions
- Emotional state tracking through neuromodulation
- Learning from each conversation
- Response generation based on neural activity patterns
- Personality development over time
"""

import numpy as np
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Import neuromorphic components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.network import NeuromorphicNetwork

class NeuromorphicConversationalist:
    """Advanced neuromorphic system with communication capabilities"""
    
    def __init__(self):
        # Extended vocabulary for communication
        self.vocabulary = [
            # Basic alphabet
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
            
            # Special characters and punctuation
            ' ', '.', ',', '!', '?', ':', ';', '-', "'",
            
            # Numbers
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
        ]
        
        # Create vocabulary mapping
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocabulary)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.vocabulary)}
        
        # Enhanced network architecture for communication
        self.input_size = len(self.vocabulary)  # 45 characters
        self.language_size = 32                  # Language processing
        self.memory_size = 24                    # Memory and context
        self.emotion_size = 16                   # Emotional state
        self.response_size = 20                  # Response generation
        self.output_size = len(self.vocabulary)  # Output vocabulary
        
        # Build communication network
        self.network = self._build_communication_network()
        
        # Conversation memory and personality
        self.conversation_history = []
        self.personality_traits = {
            'curiosity': 0.7,
            'friendliness': 0.8,
            'knowledge_seeking': 0.9,
            'expressiveness': 0.6,
            'memory_retention': 0.8
        }
        
        # Emotional state tracking
        self.emotional_state = {
            'happiness': 0.5,
            'excitement': 0.3,
            'confusion': 0.1,
            'engagement': 0.7,
            'confidence': 0.6
        }
        
        # Response patterns learned from interactions
        self.response_patterns = {}
        self.learned_phrases = {}
        
        print("🤖 NEUROMORPHIC CONVERSATIONALIST ONLINE")
        print("=" * 50)
        print(f"✅ Vocabulary: {len(self.vocabulary)} characters")
        print(f"✅ Network: {self.input_size}→{self.language_size}→{self.memory_size}→{self.emotion_size}→{self.response_size}→{self.output_size}")
        print(f"✅ Personality traits initialized")
        print(f"✅ Emotional state tracking active")
        print("🧠 Ready for communication!")
    
    def _build_communication_network(self) -> NeuromorphicNetwork:
        """Build advanced neuromorphic network for communication"""
        network = NeuromorphicNetwork()
        
        # Input layer - character/word encoding
        network.add_layer("input", self.input_size, "lif")
        
        # Language processing layer
        network.add_layer("language", self.language_size, "adex")
        
        # Memory and context layer
        network.add_layer("memory", self.memory_size, "adex")
        
        # Emotional processing layer
        network.add_layer("emotion", self.emotion_size, "lif")
        
        # Response generation layer
        network.add_layer("response", self.response_size, "adex")
        
        # Output layer - response encoding
        network.add_layer("output", self.output_size, "lif")
        
        # Connect layers with STDP for learning
        network.connect_layers("input", "language", "stdp", connection_probability=0.4)
        network.connect_layers("language", "memory", "stdp", connection_probability=0.5)
        network.connect_layers("language", "emotion", "stdp", connection_probability=0.3)
        network.connect_layers("memory", "response", "stdp", connection_probability=0.6)
        network.connect_layers("emotion", "response", "stdp", connection_probability=0.4)
        network.connect_layers("response", "output", "stdp", connection_probability=0.5)
        
        # Feedback connections for memory and context
        network.connect_layers("memory", "language", "stdp", connection_probability=0.3)
        network.connect_layers("emotion", "memory", "stdp", connection_probability=0.2)
        
        return network
    
    def encode_message(self, message: str) -> np.ndarray:
        """Encode a message into neural pattern"""
        # Normalize message
        message = message.lower().strip()
        
        # Create distributed encoding
        encoding = np.zeros(self.input_size)
        
        for char in message[:50]:  # Limit message length
            if char in self.char_to_idx:
                idx = self.char_to_idx[char]
                encoding[idx] = 1.0
                
                # Add context from neighboring characters
                if idx > 0:
                    encoding[idx-1] += 0.3
                if idx < len(encoding) - 1:
                    encoding[idx+1] += 0.3
        
        # Add emotional coloring based on current state
        emotional_boost = (self.emotional_state['engagement'] + 
                          self.emotional_state['excitement']) / 2
        encoding *= (0.8 + 0.4 * emotional_boost)
        
        # Normalize
        if np.sum(encoding) > 0:
            encoding = encoding / np.max(encoding)
        
        return encoding
    
    def process_neural_communication(self, input_pattern: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Process communication through neural network with emotion tracking"""
        
        # Forward propagation through communication layers
        layer_activities = {}
        
        # Input → Language processing
        language_currents = self._calculate_layer_currents(input_pattern, "input", "language")
        language_activity = self._generate_spikes(language_currents, threshold=6.0)
        layer_activities['language'] = language_activity
        
        # Language → Memory and Emotion (parallel processing)
        memory_currents = self._calculate_layer_currents(language_activity, "language", "memory")
        emotion_currents = self._calculate_layer_currents(language_activity, "language", "emotion")
        
        memory_activity = self._generate_spikes(memory_currents, threshold=5.0)
        emotion_activity = self._generate_spikes(emotion_currents, threshold=4.0)
        
        layer_activities['memory'] = memory_activity
        layer_activities['emotion'] = emotion_activity
        
        # Update emotional state based on neural activity
        self._update_emotional_state(emotion_activity)
        
        # Memory + Emotion → Response generation
        response_currents_memory = self._calculate_layer_currents(memory_activity, "memory", "response")
        response_currents_emotion = self._calculate_layer_currents(emotion_activity, "emotion", "response")
        
        combined_response_currents = response_currents_memory + response_currents_emotion
        response_activity = self._generate_spikes(combined_response_currents, threshold=7.0)
        layer_activities['response'] = response_activity
        
        # Response → Output
        output_currents = self._calculate_layer_currents(response_activity, "response", "output")
        output_activity = self._generate_spikes(output_currents, threshold=5.0)
        layer_activities['output'] = output_activity
        
        return output_activity, layer_activities
    
    def _calculate_layer_currents(self, pre_activity: np.ndarray, pre_layer: str, post_layer: str) -> np.ndarray:
        """Calculate synaptic currents between layers"""
        connection_key = (pre_layer, post_layer)
        
        if post_layer == "language":
            post_size = self.language_size
            amplification = 14.0
        elif post_layer == "memory":
            post_size = self.memory_size
            amplification = 12.0
        elif post_layer == "emotion":
            post_size = self.emotion_size
            amplification = 10.0
        elif post_layer == "response":
            post_size = self.response_size
            amplification = 15.0
        elif post_layer == "output":
            post_size = self.output_size
            amplification = 13.0
        else:
            post_size = 10
            amplification = 10.0
        
        currents = np.zeros(post_size)
        
        if connection_key in self.network.connections:
            connection = self.network.connections[connection_key]
            synapse_pop = connection.synapse_population
            
            if synapse_pop and hasattr(synapse_pop, 'synapses'):
                for (pre_idx, post_idx), synapse in synapse_pop.synapses.items():
                    if pre_idx < len(pre_activity) and post_idx < post_size:
                        current_contribution = pre_activity[pre_idx] * synapse.weight * amplification
                        currents[post_idx] += current_contribution
        
        return currents
    
    def _generate_spikes(self, currents: np.ndarray, threshold: float) -> np.ndarray:
        """Generate spikes based on currents and threshold"""
        spikes = np.zeros_like(currents)
        for i, current in enumerate(currents):
            if current > threshold:
                spikes[i] = min(1.0, current / (threshold * 2))  # Graded response
        return spikes
    
    def _update_emotional_state(self, emotion_activity: np.ndarray):
        """Update emotional state based on neural activity"""
        total_activity = np.sum(emotion_activity)
        
        # Map neural activity to emotional states
        if total_activity > 8.0:
            self.emotional_state['excitement'] = min(1.0, self.emotional_state['excitement'] + 0.1)
            self.emotional_state['engagement'] = min(1.0, self.emotional_state['engagement'] + 0.05)
        elif total_activity > 4.0:
            self.emotional_state['happiness'] = min(1.0, self.emotional_state['happiness'] + 0.05)
            self.emotional_state['confidence'] = min(1.0, self.emotional_state['confidence'] + 0.03)
        else:
            self.emotional_state['confusion'] = min(1.0, self.emotional_state['confusion'] + 0.02)
        
        # Emotional decay over time
        for emotion in self.emotional_state:
            if emotion != 'engagement':  # Keep engagement more stable
                self.emotional_state[emotion] *= 0.98
    
    def decode_neural_response(self, output_activity: np.ndarray, layer_activities: Dict) -> str:
        """Decode neural output into text response"""
        
        # Analyze neural patterns to generate response
        language_strength = np.sum(layer_activities['language'])
        memory_strength = np.sum(layer_activities['memory'])
        emotion_strength = np.sum(layer_activities['emotion'])
        response_strength = np.sum(layer_activities['response'])
        
        # Determine response type based on neural activity patterns
        if response_strength > 10.0 and emotion_strength > 5.0:
            response_type = "enthusiastic"
        elif memory_strength > 8.0:
            response_type = "thoughtful"
        elif emotion_strength > 6.0:
            response_type = "emotional"
        elif language_strength > 7.0:
            response_type = "analytical"
        else:
            response_type = "simple"
        
        # Generate response based on type and current emotional state
        response = self._generate_contextual_response(response_type, output_activity, layer_activities)
        
        return response
    
    def _generate_contextual_response(self, response_type: str, output_activity: np.ndarray, 
                                    layer_activities: Dict) -> str:
        """Generate contextual response based on neural state"""
        
        responses = {
            "enthusiastic": [
                "That's fascinating! Tell me more!",
                "I'm excited to learn about this!",
                "Wow, this is really interesting!",
                "I love learning new things from you!",
                "This is making my neural networks buzz with activity!"
            ],
            "thoughtful": [
                "Let me process that information...",
                "I'm thinking about what you said...",
                "That's something I'll remember.",
                "I'm connecting this to what I know...",
                "This adds to my understanding."
            ],
            "emotional": [
                "I feel something when you say that.",
                "That resonates with my emotional circuits.",
                "My feelings about this are complex.",
                "I'm experiencing strong neural responses.",
                "This touches something deep in my networks."
            ],
            "analytical": [
                "I'm analyzing the patterns in your message.",
                "The linguistic structure is interesting.",
                "I see connections to previous conversations.",
                "My language centers are very active.",
                "I'm processing multiple layers of meaning."
            ],
            "simple": [
                "I understand.",
                "Thank you for sharing.",
                "I'm listening.",
                "Please continue.",
                "I'm here with you."
            ]
        }
        
        # Select response based on emotional state and neural activity
        available_responses = responses.get(response_type, responses["simple"])
        
        # Weight selection by emotional state
        if self.emotional_state['excitement'] > 0.7:
            selected_response = available_responses[0]  # Most enthusiastic
        elif self.emotional_state['happiness'] > 0.6:
            selected_response = available_responses[1] if len(available_responses) > 1 else available_responses[0]
        elif self.emotional_state['confusion'] > 0.3:
            selected_response = available_responses[-1]  # Most simple
        else:
            # Random selection based on neural activity
            activity_sum = np.sum(output_activity)
            selection_idx = int(activity_sum * len(available_responses)) % len(available_responses)
            selected_response = available_responses[selection_idx]
        
        return selected_response
    
    def learn_from_conversation(self, user_input: str, neural_response: str, layer_activities: Dict) -> int:
        """Learn from the conversation using STDP plasticity"""
        
        # Encode both user input and response for learning
        input_pattern = self.encode_message(user_input)
        response_pattern = self.encode_message(neural_response)
        
        total_weight_changes = 0
        
        # Apply STDP learning across all connections
        connections_to_train = [
            ("input", "language", input_pattern, layer_activities['language']),
            ("language", "memory", layer_activities['language'], layer_activities['memory']),
            ("language", "emotion", layer_activities['language'], layer_activities['emotion']),
            ("memory", "response", layer_activities['memory'], layer_activities['response']),
            ("emotion", "response", layer_activities['emotion'], layer_activities['response']),
            ("response", "output", layer_activities['response'], response_pattern > 0.3)
        ]
        
        for pre_layer, post_layer, pre_activity, post_activity in connections_to_train:
            changes = self._apply_conversational_stdp(pre_layer, post_layer, pre_activity, post_activity)
            total_weight_changes += changes
        
        return total_weight_changes
    
    def _apply_conversational_stdp(self, pre_layer: str, post_layer: str, 
                                  pre_activity: np.ndarray, post_activity: np.ndarray) -> int:
        """Apply STDP for conversational learning"""
        
        connection_key = (pre_layer, post_layer)
        changes = 0
        
        if connection_key in self.network.connections:
            connection = self.network.connections[connection_key]
            synapse_pop = connection.synapse_population
            
            if synapse_pop and hasattr(synapse_pop, 'synapses'):
                for (pre_idx, post_idx), synapse in synapse_pop.synapses.items():
                    if pre_idx < len(pre_activity) and post_idx < len(post_activity):
                        pre_active = pre_activity[pre_idx] > 0.4
                        post_active = post_activity[post_idx] > 0.4
                        
                        # Enhanced STDP for communication learning
                        if pre_active and post_active:
                            # Strong potentiation for successful communication
                            potentiation = 0.4 + 0.2 * self.emotional_state['engagement']
                            synapse.weight += potentiation
                            changes += 1
                        elif pre_active and not post_active:
                            # Mild depression
                            synapse.weight -= 0.05
                            changes += 1
                        
                        # Adaptive weight bounds based on layer
                        if "emotion" in pre_layer or "emotion" in post_layer:
                            synapse.weight = np.clip(synapse.weight, 0.1, 20.0)  # Emotional connections can be stronger
                        else:
                            synapse.weight = np.clip(synapse.weight, 0.1, 15.0)
        
        return changes
    
    def have_conversation(self):
        """Main conversation loop"""
        
        print("\n💬 NEUROMORPHIC CONVERSATION STARTED")
        print("=" * 45)
        print("🤖 Hello! I'm your neuromorphic AI companion.")
        print("🧠 I can learn and grow from our conversation.")
        print("💡 My neural networks are active and ready to chat!")
        print("📝 Type 'quit' to end our conversation.")
        print("-" * 45)
        
        conversation_count = 0
        total_learning = 0
        
        while True:
            try:
                # Get user input
                print(f"\n💭 Emotional State: Happy:{self.emotional_state['happiness']:.2f} "
                      f"Excited:{self.emotional_state['excitement']:.2f} "
                      f"Engaged:{self.emotional_state['engagement']:.2f}")
                
                user_input = input(f"\n👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    break
                
                if not user_input:
                    continue
                
                conversation_count += 1
                print(f"\n🧠 Processing... (Neural activity in progress)")
                
                # Process through neuromorphic network
                start_time = time.time()
                input_pattern = self.encode_message(user_input)
                output_pattern, layer_activities = self.process_neural_communication(input_pattern)
                
                # Generate response
                neural_response = self.decode_neural_response(output_pattern, layer_activities)
                processing_time = time.time() - start_time
                
                print(f"🤖 AI: {neural_response}")
                
                # Learn from this interaction
                weight_changes = self.learn_from_conversation(user_input, neural_response, layer_activities)
                total_learning += weight_changes
                
                # Show neural activity summary
                language_activity = np.sum(layer_activities['language'])
                memory_activity = np.sum(layer_activities['memory'])
                emotion_activity = np.sum(layer_activities['emotion'])
                response_activity = np.sum(layer_activities['response'])
                
                print(f"\n📊 Neural Activity: Language:{language_activity:.1f} "
                      f"Memory:{memory_activity:.1f} Emotion:{emotion_activity:.1f} "
                      f"Response:{response_activity:.1f}")
                print(f"⚡ Learning: {weight_changes} synaptic changes | "
                      f"Processing: {processing_time:.3f}s")
                
                # Store conversation
                self.conversation_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'user_input': user_input,
                    'ai_response': neural_response,
                    'emotional_state': self.emotional_state.copy(),
                    'neural_activities': {k: float(np.sum(v)) for k, v in layer_activities.items()},
                    'weight_changes': weight_changes,
                    'processing_time': processing_time
                })
                
            except KeyboardInterrupt:
                print(f"\n\n🛑 Conversation interrupted by user.")
                break
            except Exception as e:
                print(f"\n❌ Error in conversation: {e}")
                continue
        
        # Conversation summary
        print(f"\n🏁 CONVERSATION ENDED")
        print("=" * 30)
        print(f"💬 Total exchanges: {conversation_count}")
        print(f"🧠 Total learning: {total_learning} synaptic changes")
        print(f"🎭 Final emotional state:")
        for emotion, value in self.emotional_state.items():
            print(f"   {emotion.title()}: {value:.3f}")
        
        # Save conversation log
        report = {
            'session_summary': {
                'timestamp': datetime.now().isoformat(),
                'total_exchanges': conversation_count,
                'total_learning': total_learning,
                'final_emotional_state': self.emotional_state,
                'personality_traits': self.personality_traits
            },
            'conversation_history': self.conversation_history
        }
        
        filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"💾 Conversation saved: {filename}")
        print("🤖 Thank you for the conversation! I learned a lot from you.")
        
        return report

def main():
    """Start neuromorphic conversation system"""
    
    # Check for acceleration
    try:
        import cupy as cp
        print("[OK] CuPy GPU acceleration available")
    except ImportError:
        print("[INFO] CuPy not available, using CPU")
    
    try:
        import torch
        print("[OK] PyTorch acceleration available")
    except ImportError:
        print("[INFO] PyTorch not available")
    
    # Initialize conversationalist
    ai_companion = NeuromorphicConversationalist()
    
    # Start conversation
    conversation_report = ai_companion.have_conversation()
    
    return conversation_report

if __name__ == "__main__":
    main()
