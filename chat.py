#!/usr/bin/env python3
"""
Code-Mixed Language Chat PoC using GPT-4o-mini
Responds intelligently to code-mixed South Indian language input
"""

import os
import json
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

class SouthIndianLanguage(Enum):
    """Supported South Indian languages"""
    TAMIL = "tamil"
    TELUGU = "telugu"
    KANNADA = "kannada"
    MALAYALAM = "malayalam"
    ENGLISH = "english"
    MIXED = "mixed"

@dataclass
class LanguageDetectionResult:
    """Result of language detection"""
    primary_language: SouthIndianLanguage
    secondary_language: Optional[SouthIndianLanguage]
    confidence: float
    is_code_mixed: bool
    mix_ratio: float  # 0.0 = pure primary, 1.0 = pure secondary

@dataclass
class ChatResponse:
    """Response from the chat system"""
    response_text: str
    detected_language: LanguageDetectionResult
    response_language: SouthIndianLanguage
    processing_time: float

class CodeMixedLanguageDetector:
    """Detects code-mixed languages using GPT-4o-mini"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = openai.OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        
        if not self.client.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
    
    def detect_language(self, text: str) -> LanguageDetectionResult:
        """Detect language and code-mixing in text"""
        
        detection_prompt = f"""
        Analyze the following text for language detection and code-mixing:
        
        Text: "{text}"
        
        Respond with ONLY a JSON object (no other text):
        {{
            "primary_language": "tamil/telugu/kannada/malayalam/english",
            "secondary_language": "tamil/telugu/kannada/malayalam/english or null",
            "confidence": 0.95,
            "is_code_mixed": true/false,
            "mix_ratio": 0.4,
            "reasoning": "brief explanation"
        }}
        
        Guidelines:
        - primary_language: The dominant language in the text
        - secondary_language: The secondary language if code-mixed, null if pure
        - confidence: How confident you are (0.0 to 1.0)
        - is_code_mixed: true if mixing multiple languages
        - mix_ratio: 0.0 = pure primary, 1.0 = pure secondary, 0.5 = equal mix
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a South Indian language expert. Respond only with valid JSON."},
                    {"role": "user", "content": detection_prompt}
                ],
                temperature=0.1,
                max_tokens=200
            )
            
            result = json.loads(response.choices[0].message.content)
            
            return LanguageDetectionResult(
                primary_language=SouthIndianLanguage(result["primary_language"]),
                secondary_language=SouthIndianLanguage(result["secondary_language"]) if result.get("secondary_language") else None,
                confidence=result["confidence"],
                is_code_mixed=result["is_code_mixed"],
                mix_ratio=result["mix_ratio"]
            )
            
        except Exception as e:
            print(f"⚠️  Language detection failed: {e}")
            # Fallback detection
            return LanguageDetectionResult(
                primary_language=SouthIndianLanguage.ENGLISH,
                secondary_language=None,
                confidence=0.5,
                is_code_mixed=False,
                mix_ratio=0.0
            )

class CodeMixedChatBot:
    """Chat bot that responds to code-mixed languages"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = openai.OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        self.detector = CodeMixedLanguageDetector(api_key)
        self.conversation_history = []
        
        # Cultural context for different regions
        self.cultural_context = {
            SouthIndianLanguage.TAMIL: {
                "greeting": "Vanakkam",
                "culture": "Tamil Nadu culture, Chennai references, Tamil cinema",
                "food": "dosa, idli, sambar, rasam, Tamil cuisine"
            },
            SouthIndianLanguage.TELUGU: {
                "greeting": "Namaskaram",
                "culture": "Andhra/Telangana culture, Hyderabad references, Tollywood",
                "food": "biryani, pesarattu, gongura, Telugu cuisine"
            },
            SouthIndianLanguage.KANNADA: {
                "greeting": "Namaskara",
                "culture": "Karnataka culture, Bangalore references, Sandalwood",
                "food": "masala dosa, mysore pak, Kannada cuisine"
            },
            SouthIndianLanguage.MALAYALAM: {
                "greeting": "Namaskaram",
                "culture": "Kerala culture, backwaters, Malayalam cinema",
                "food": "appam, fish curry, coconut, Kerala cuisine"
            }
        }
    
    def create_system_prompt(self, detected_lang: LanguageDetectionResult) -> str:
        """Create context-aware system prompt"""
        
        primary = detected_lang.primary_language
        context = self.cultural_context.get(primary, {})
        
        if detected_lang.is_code_mixed:
            prompt = f"""
            You are a friendly, culturally aware assistant who naturally speaks in code-mixed {primary.value}-English, 
            just like urban South Indians do in daily conversation.
            
            Cultural Context:
            - Primary language: {primary.value.title()}
            - Greeting style: {context.get('greeting', 'Hello')}
            - Cultural knowledge: {context.get('culture', 'General South Indian')}
            - Food references: {context.get('food', 'South Indian cuisine')}
            
            Response Style:
            - Mix {primary.value} and English naturally (like the user did)
            - Use appropriate {primary.value} greetings and expressions
            - Include cultural references when relevant
            - Keep responses conversational and warm
            - Match the user's code-mixing style and ratio
            - Use romanized {primary.value} (English letters) for {primary.value} words
            
            Examples of natural {primary.value}-English mixing:
            - "Vanakkam! How are you? Naan nalla irukken da!"
            - "Office work romba busy-aa irukku, but weekend plans ready!"
            - "Shall we go for lunch? Biriyani sapdalam!"
            
            Remember: Be natural, friendly, and culturally appropriate!
            """
        else:
            if primary == SouthIndianLanguage.ENGLISH:
                prompt = """
                You are a friendly assistant. The user is speaking in English.
                Respond naturally in English, but feel free to use some South Indian 
                expressions if contextually appropriate.
                """
            else:
                prompt = f"""
                You are a friendly assistant. The user is speaking in {primary.value}.
                Respond in a mix of {primary.value} and English, as this is natural 
                for South Indian conversations. Use romanized {primary.value}.
                """
        
        return prompt
    
    def generate_response(self, user_input: str) -> ChatResponse:
        """Generate response to user input"""
        start_time = time.time()
        
        # Detect language and code-mixing
        detected_lang = self.detector.detect_language(user_input)
        
        # Create appropriate system prompt
        system_prompt = self.create_system_prompt(detected_lang)
        
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Keep only last 10 messages for context
        recent_history = self.conversation_history[-10:]
        
        try:
            # Generate response
            messages = [
                {"role": "system", "content": system_prompt}
            ] + recent_history
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7,
                max_tokens=300
            )
            
            response_text = response.choices[0].message.content
            
            # Add response to history
            self.conversation_history.append({"role": "assistant", "content": response_text})
            
            processing_time = time.time() - start_time
            
            return ChatResponse(
                response_text=response_text,
                detected_language=detected_lang,
                response_language=detected_lang.primary_language,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"❌ Response generation failed: {e}")
            
            return ChatResponse(
                response_text="Sorry, I'm having trouble right now. Please try again!",
                detected_language=detected_lang,
                response_language=SouthIndianLanguage.ENGLISH,
                processing_time=processing_time
            )

def print_language_info(detection: LanguageDetectionResult):
    """Print language detection information"""
    print(f"🌍 Language: {detection.primary_language.value.title()}", end="")
    if detection.secondary_language:
        print(f" + {detection.secondary_language.value.title()}")
    else:
        print()
    
    print(f"🔀 Code-mixed: {'Yes' if detection.is_code_mixed else 'No'}")
    if detection.is_code_mixed:
        print(f"📊 Mix ratio: {detection.mix_ratio:.1%} secondary language")
    print(f"📈 Confidence: {detection.confidence:.1%}")

def print_sample_phrases():
    """Print sample phrases for testing"""
    samples = {
        "Tamil-English": [
            "Vanakkam! How are you? Naan nalla irukken.",
            "Office-la meeting irukku but I'll be late.",
            "Shall we go for coffee? Romba tired-aa irukku."
        ],
        "Telugu-English": [
            "Namaskaram! How was your day? Nenu bagane unnaanu.",
            "Biryani thindam raa! Very hungry feeling.",
            "Movie ticket book chesaava? Let's go for the show."
        ],
        "Kannada-English": [
            "Namaskara guru! How are you? Naanu chennagi iddene.",
            "Traffic jaasti aagtide, I'll be late for meeting.",
            "Masala dosa order maadbeka? Very tasty aaguttade."
        ],
        "Malayalam-English": [
            "Namaskaram machane! How are you? Njaan nannaayi und.",
            "Rain jaasti und today, but I love this weather.",
            "Fish curry kazhikkanamennu thonnunnu. Shall we go?"
        ]
    }
    
    print("\n📋 Sample phrases to try:")
    print("=" * 50)
    for lang, phrases in samples.items():
        print(f"\n🗣️  {lang}:")
        for i, phrase in enumerate(phrases, 1):
            print(f"   {i}. {phrase}")

def main():
    """Main chat function"""
    print("🗣️  Code-Mixed South Indian Language Chat PoC")
    print("=" * 60)
    
    # Check API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please create a .env file with your OpenAI API key:")
        print("OPENAI_API_KEY=your_api_key_here")
        return
    
    # Initialize chatbot
    try:
        chatbot = CodeMixedChatBot()
        print("✅ Code-mixed chatbot initialized")
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        return
    
    # Show sample phrases
    print_sample_phrases()
    
    print("\n💬 Start chatting! (Type 'quit' to exit, 'samples' to see examples)")
    print("-" * 60)
    
    try:
        while True:
            # Get user input
            user_input = input("\n🤖 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Bye! Come back soon!")
                break
            
            if user_input.lower() == 'samples':
                print_sample_phrases()
                continue
            
            # Generate response
            print("🔄 Processing...", end="", flush=True)
            response = chatbot.generate_response(user_input)
            print("\r" + " " * 15 + "\r", end="")  # Clear processing message
            
            # Print language detection info
            print("📊 Language Analysis:")
            print_language_info(response.detected_language)
            print(f"⏱️  Processing time: {response.processing_time:.2f}s")
            
            # Print response
            print(f"\n🤖 Bot: {response.response_text}")
            print("-" * 60)
    
    except KeyboardInterrupt:
        print("\n🛑 Chat interrupted by user")
    except Exception as e:
        print(f"\n❌ Chat error: {e}")
    
    print("👋 Thanks for chatting!")

if __name__ == "__main__":
    main()