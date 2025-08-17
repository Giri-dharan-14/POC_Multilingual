#!/usr/bin/env python3
"""
Code-Mixed South Indian Language TTS PoC using OpenAI TTS
Generates and plays code-mixed speech for Tamil, Telugu, Kannada, Malayalam
"""

import os
import time
import tempfile
import subprocess
import platform
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv
import openai

# Audio playback
try:
    import pygame
    pygame.mixer.init()
except ImportError:
    print("Missing pygame. Install with: pip install pygame")
    exit(1)

# Load environment variables
load_dotenv()

class SouthIndianLanguage(Enum):
    """Supported South Indian languages"""
    TAMIL = "tamil"
    TELUGU = "telugu"
    KANNADA = "kannada"
    MALAYALAM = "malayalam"
    MIXED = "mixed"

class TTSVoice(Enum):
    """Available OpenAI TTS voices"""
    ALLOY = "alloy"
    ECHO = "echo"
    FABLE = "fable"
    ONYX = "onyx"
    NOVA = "nova"
    SHIMMER = "shimmer"

@dataclass
class CodeMixedPhrase:
    """A phrase with language mixing information"""
    text: str
    primary_language: SouthIndianLanguage
    secondary_language: Optional[SouthIndianLanguage]
    mix_ratio: float  # 0.0 = pure primary, 1.0 = pure secondary
    description: str

class CodeMixedGenerator:
    """Generates code-mixed South Indian phrases"""
    
    def __init__(self):
        self.sample_phrases = {
            SouthIndianLanguage.TAMIL: [
                CodeMixedPhrase(
                    "Vanakkam anna! How are you? Naan nalla irukken, thanks for asking.",
                    SouthIndianLanguage.TAMIL, None, 0.4,
                    "Tamil greeting with English conversation"
                ),
                CodeMixedPhrase(
                    "Office-la meeting irukku, but naan late-aa vandhuruven. Sorry",
                    SouthIndianLanguage.TAMIL, None, 0.6,
                    "Tamil-English work context"
                ),
                CodeMixedPhrase(
                    "Enna bro, weekend plans edhuvum irukka? Let's go for a movie.",
                    SouthIndianLanguage.TAMIL, None, 0.5,
                    "Casual Tamil-English mixing"
                ),
                CodeMixedPhrase(
                    "Coffee kudikkalam-aa? I'm feeling very tired today.",
                    SouthIndianLanguage.TAMIL, None, 0.4,
                    "Tamil suggestion with English explanation"
                ),
            ],
            
            SouthIndianLanguage.TELUGU: [
                CodeMixedPhrase(
                    "Namaskaram! How are you doing? Nenu bagane unnaanu, thank you.",
                    SouthIndianLanguage.TELUGU, None, 0.4,
                    "Telugu greeting with English"
                ),
                CodeMixedPhrase(
                    "Office work chaala busy ga undi. But weekend-lo relax cheyaali.",
                    SouthIndianLanguage.TELUGU, None, 0.6,
                    "Telugu-English work talk"
                ),
                CodeMixedPhrase(
                    "Biryani thindam raa! I'm very hungry, chala rojula nundi waiting.",
                    SouthIndianLanguage.TELUGU, None, 0.5,
                    "Food context mixing"
                ),
                CodeMixedPhrase(
                    "Movie ekkada chudaali? Let's book tickets online, convenient ga untundi.",
                    SouthIndianLanguage.TELUGU, None, 0.5,
                    "Entertainment planning"
                ),
            ],
            
            SouthIndianLanguage.KANNADA: [
                CodeMixedPhrase(
                    "Namaskara guru! How are you? Naanu chennagi iddene, thanks for asking.",
                    SouthIndianLanguage.KANNADA, None, 0.4,
                    "Kannada greeting with English"
                ),
                CodeMixedPhrase(
                    "Traffic jaasti aagtide. I'll be late for the meeting, sorry guru.",
                    SouthIndianLanguage.KANNADA, None, 0.5,
                    "Traffic situation mixing"
                ),
                CodeMixedPhrase(
                    "Masala dosa thinbekku anstide. Let's go to that new restaurant.",
                    SouthIndianLanguage.KANNADA, None, 0.4,
                    "Food craving expression"
                ),
                CodeMixedPhrase(
                    "Weekend plans yenu guru? I want to go to Mysore, scenic aagide.",
                    SouthIndianLanguage.KANNADA, None, 0.5,
                    "Weekend planning"
                ),
            ],
            
            SouthIndianLanguage.MALAYALAM: [
                CodeMixedPhrase(
                    "Namaskaram machane! How are you? Njaan nannaayi und, thanks.",
                    SouthIndianLanguage.MALAYALAM, None, 0.4,
                    "Malayalam greeting with English"
                ),
                CodeMixedPhrase(
                    "Monsoon kaaranam rain jaasti und. But I love this weather, romantic aanu.",
                    SouthIndianLanguage.MALAYALAM, None, 0.5,
                    "Weather talk mixing"
                ),
                CodeMixedPhrase(
                    "Fish curry kazhikkanamennu thonnunnu. Let's go to that coastal restaurant.",
                    SouthIndianLanguage.MALAYALAM, None, 0.4,
                    "Food preference"
                ),
                CodeMixedPhrase(
                    "Backwaters-il boat ride cheyyaam. I heard it's very peaceful there.",
                    SouthIndianLanguage.MALAYALAM, None, 0.5,
                    "Tourism activity"
                ),
            ],
        }
    
    def get_phrases(self, language: SouthIndianLanguage) -> List[CodeMixedPhrase]:
        """Get sample phrases for a language"""
        return self.sample_phrases.get(language, [])
    
    def generate_custom_phrase(self, base_text: str, language: SouthIndianLanguage) -> CodeMixedPhrase:
        """Generate a custom code-mixed phrase"""
        return CodeMixedPhrase(
            text=base_text,
            primary_language=language,
            secondary_language=None,
            mix_ratio=0.3,  # Estimate
            description="Custom user input"
        )

class CodeMixedTTS:
    """Text-to-Speech for code-mixed South Indian languages"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = openai.OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        
        if not self.client.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
    
    def enhance_text_for_tts(self, phrase: CodeMixedPhrase) -> str:
        """
        Enhance text for better TTS pronunciation using GPT-4o-mini
        """
        enhancement_prompt = f"""
        Convert this code-mixed {phrase.primary_language.value} text to be more TTS-friendly while preserving the natural code-mixing:
        
        Original: "{phrase.text}"
        
        Instructions:
        1. Keep the natural code-mixing as is
        2. Add phonetic hints for non-English words in parentheses if needed
        3. Adjust spelling of regional words to help English TTS pronounce better
        4. Keep the conversational tone intact
        5. Don't over-modify - maintain authenticity
        
        Enhanced text for TTS:
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a linguistic expert in South Indian languages and code-mixing. Help optimize text for TTS while preserving natural speech patterns."},
                    {"role": "user", "content": enhancement_prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            enhanced_text = response.choices[0].message.content.strip()
            # Remove any quotes or extra formatting
            enhanced_text = enhanced_text.replace('"', '').replace('Enhanced text for TTS:', '').strip()
            
            return enhanced_text
        
        except Exception as e:
            print(f"⚠️  Enhancement failed: {e}")
            return phrase.text  # Fallback to original
    
    def generate_speech(self, phrase: CodeMixedPhrase, voice: TTSVoice = TTSVoice.NOVA, enhance: bool = True) -> str:
        """
        Generate speech from code-mixed text
        
        Returns path to generated audio file
        """
        print(f"🔄 Generating speech for: {phrase.description}")
        
        # Enhance text for better pronunciation if requested
        text_to_speak = self.enhance_text_for_tts(phrase) if enhance else phrase.text
        
        if enhance and text_to_speak != phrase.text:
            print(f"📝 Enhanced text: {text_to_speak}")
        
        try:
            response = self.client.audio.speech.create(
                model="tts-1-hd",  # High quality model
                voice=voice.value,
                input=text_to_speak,
                speed=0.9  # Slightly slower for better clarity
            )
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            temp_file.write(response.content)
            temp_file.close()
            
            return temp_file.name
        
        except Exception as e:
            print(f"❌ TTS generation failed: {e}")
            return None

class AudioPlayer:
    """Simple audio player using pygame"""
    
    @staticmethod
    def play_audio(file_path: str):
        """Play audio file"""
        try:
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
                
        except Exception as e:
            print(f"❌ Audio playback failed: {e}")
    
    @staticmethod
    def cleanup():
        """Clean up pygame resources"""
        pygame.mixer.quit()

def print_phrase_info(phrase: CodeMixedPhrase):
    """Pretty print phrase information"""
    print("\n" + "="*60)
    print("🗣️  CODE-MIXED PHRASE")
    print("="*60)
    print(f"📝 Text: {phrase.text}")
    print(f"🌍 Primary Language: {phrase.primary_language.value.title()}")
    print(f"🔀 Mix Ratio: {phrase.mix_ratio:.1%} English mixing")
    print(f"📖 Description: {phrase.description}")
    print("="*60)

def main():
    """Main PoC function"""
    print("🗣️  Code-Mixed South Indian TTS PoC")
    print("=" * 50)
    
    # Check API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please create a .env file with your OpenAI API key:")
        print("OPENAI_API_KEY=your_api_key_here")
        return
    
    # Initialize components
    try:
        generator = CodeMixedGenerator()
        tts = CodeMixedTTS()
        player = AudioPlayer()
        print("✅ TTS system initialized")
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        return
    
    try:
        while True:
            print("\n" + "-"*50)
            print("Language Options:")
            print("1. Tamil (தமிழ்)")
            print("2. Telugu (తెలుగు)")
            print("3. Kannada (ಕನ್ನಡ)")
            print("4. Malayalam (മലയാളം)")
            print("5. Custom text input")
            print("6. Voice comparison")
            print("7. Exit")
            
            choice = input("\nSelect language (1-7): ").strip()
            
            language_map = {
                "1": SouthIndianLanguage.TAMIL,
                "2": SouthIndianLanguage.TELUGU,
                "3": SouthIndianLanguage.KANNADA,
                "4": SouthIndianLanguage.MALAYALAM,
            }
            
            if choice in language_map:
                language = language_map[choice]
                phrases = generator.get_phrases(language)
                
                print(f"\n📋 Sample {language.value.title()} phrases:")
                for i, phrase in enumerate(phrases, 1):
                    print(f"{i}. {phrase.text}")
                
                phrase_choice = input(f"\nSelect phrase (1-{len(phrases)}): ").strip()
                
                try:
                    phrase_idx = int(phrase_choice) - 1
                    if 0 <= phrase_idx < len(phrases):
                        selected_phrase = phrases[phrase_idx]
                        print_phrase_info(selected_phrase)
                        
                        print("🔄 Generating speech...")
                        audio_file = tts.generate_speech(selected_phrase)
                        
                        if audio_file:
                            print("🔊 Playing audio...")
                            player.play_audio(audio_file)
                            os.unlink(audio_file)  # Clean up
                            print("✅ Playback complete!")
                        
                    else:
                        print("❌ Invalid phrase number")
                        
                except ValueError:
                    print("❌ Please enter a valid number")
            
            elif choice == "5":
                # Custom text input
                print("\n📝 Custom Text Input")
                custom_text = input("Enter your code-mixed text: ").strip()
                
                if custom_text:
                    print("\nSelect primary language:")
                    print("1. Tamil  2. Telugu  3. Kannada  4. Malayalam")
                    lang_choice = input("Choice (1-4): ").strip()
                    
                    if lang_choice in language_map:
                        language = language_map[lang_choice]
                        custom_phrase = generator.generate_custom_phrase(custom_text, language)
                        
                        print_phrase_info(custom_phrase)
                        
                        print("🔄 Generating speech...")
                        audio_file = tts.generate_speech(custom_phrase)
                        
                        if audio_file:
                            print("🔊 Playing audio...")
                            player.play_audio(audio_file)
                            os.unlink(audio_file)
                            print("✅ Playback complete!")
                    else:
                        print("❌ Invalid language choice")
                else:
                    print("❌ No text entered")
            
            elif choice == "6":
                # Voice comparison
                print("\n🎭 Voice Comparison")
                sample_text = "Vanakkam! How are you? Naan nalla irukken, thanks!"
                sample_phrase = CodeMixedPhrase(sample_text, SouthIndianLanguage.TAMIL, None, 0.4, "Voice comparison sample")
                
                voices = [TTSVoice.ALLOY, TTSVoice.NOVA, TTSVoice.SHIMMER]
                
                for voice in voices:
                    print(f"\n🔊 Playing with {voice.value} voice...")
                    audio_file = tts.generate_speech(sample_phrase, voice, enhance=False)
                    
                    if audio_file:
                        player.play_audio(audio_file)
                        os.unlink(audio_file)
                        input("Press Enter for next voice...")
                
                print("✅ Voice comparison complete!")
            
            elif choice == "7":
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice. Please enter 1-7.")
    
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    
    finally:
        player.cleanup()
        print("🧹 Cleaned up resources")

if __name__ == "__main__":
    main()