#!/usr/bin/env python3
"""
Simple Language Detection PoC using GPT-4o-mini
Records audio from microphone and detects the spoken language
"""

import os
import time
import json
import tempfile
import threading
from typing import Dict, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import openai

# Audio recording libraries
try:
    import pyaudio
    import wave
except ImportError:
    print("Missing audio libraries. Install with: pip install pyaudio")
    exit(1)

# Load environment variables
load_dotenv()

@dataclass
class LanguageDetectionResult:
    """Result of language detection"""
    detected_language: str
    confidence: float
    transcription: str
    processing_time: float
    raw_response: Dict

class AudioRecorder:
    """Simple audio recorder using PyAudio"""
    
    def __init__(self, sample_rate=16000, chunk_size=1024, channels=1):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.format = pyaudio.paInt16
        self.audio = pyaudio.PyAudio()
        self.recording = False
        self.frames = []
        
    def start_recording(self):
        """Start recording audio"""
        self.frames = []
        self.recording = True
        
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        print("🎤 Recording... Press ENTER to stop")
        
        # Record in a separate thread
        self.record_thread = threading.Thread(target=self._record_audio)
        self.record_thread.start()
    
    def _record_audio(self):
        """Internal method to record audio frames"""
        while self.recording:
            try:
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                self.frames.append(data)
            except Exception as e:
                print(f"Recording error: {e}")
                break
    
    def stop_recording(self) -> str:
        """Stop recording and save to temporary file"""
        self.recording = False
        
        if hasattr(self, 'record_thread'):
            self.record_thread.join()
        
        self.stream.stop_stream()
        self.stream.close()
        
        # Save to temporary WAV file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        
        with wave.open(temp_file.name, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(self.frames))
        
        return temp_file.name
    
    def cleanup(self):
        """Clean up PyAudio resources"""
        self.audio.terminate()

class LanguageDetector:
    """Language detection using OpenAI GPT-4o-mini with Whisper"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = openai.OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        
        if not self.client.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
    
    def detect_language(self, audio_file_path: str) -> LanguageDetectionResult:
        """
        Detect language from audio file using OpenAI
        
        Args:
            audio_file_path: Path to audio file
            
        Returns:
            LanguageDetectionResult with detection results
        """
        start_time = time.time()
        
        try:
            # Step 1: Transcribe audio using Whisper
            print("🔄 Transcribing audio...")
            with open(audio_file_path, 'rb') as audio_file:
                transcription_response = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",  # Get language detection info
                    language=None  # Auto-detect language
                )
            
            # Step 2: Use GPT-4o-mini for detailed language analysis
            print("🔄 Analyzing language with GPT-4o-mini...")
            
            analysis_prompt = f"""
            Analyze the following transcribed text and provide detailed language detection:
            
            Text: "{transcription_response.text}"
            
            Please respond with a JSON object containing:
            {{
                "primary_language": "language name",
                "language_code": "ISO 639-1 code", 
                "confidence": 0.95,
                "is_code_mixed": false,
                "secondary_languages": [],
                "script_type": "latin/devanagari/tamil/etc",
                "region_dialect": "if detectable",
                "reasoning": "brief explanation"
            }}
            
            Supported languages include: English, Hindi, Tamil, Telugu, Kannada, Malayalam, Bengali, Gujarati, Marathi, Punjabi, and others.
            """
            
            gpt_response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a language detection expert. Respond only with valid JSON."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.1,
                max_tokens=300
            )
            
            # Parse GPT response
            try:
                analysis = json.loads(gpt_response.choices[0].message.content)
            except json.JSONDecodeError:
                # Fallback if GPT doesn't return valid JSON
                analysis = {
                    "primary_language": transcription_response.language or "unknown",
                    "language_code": transcription_response.language or "unk",
                    "confidence": 0.7,
                    "is_code_mixed": False,
                    "secondary_languages": [],
                    "script_type": "unknown",
                    "region_dialect": "unknown",
                    "reasoning": "Fallback due to JSON parsing error"
                }
            
            processing_time = time.time() - start_time
            
            return LanguageDetectionResult(
                detected_language=analysis["primary_language"],
                confidence=analysis["confidence"],
                transcription=transcription_response.text,
                processing_time=processing_time,
                raw_response={
                    "whisper_response": transcription_response.model_dump(),
                    "gpt_analysis": analysis
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"❌ Error during language detection: {e}")
            
            return LanguageDetectionResult(
                detected_language="error",
                confidence=0.0,
                transcription="",
                processing_time=processing_time,
                raw_response={"error": str(e)}
            )

def print_results(result: LanguageDetectionResult):
    """Pretty print the detection results"""
    print("\n" + "="*60)
    print("🎯 LANGUAGE DETECTION RESULTS")
    print("="*60)
    print(f"📝 Transcription: {result.transcription}")
    print(f"🌍 Detected Language: {result.detected_language}")
    print(f"📊 Confidence: {result.confidence:.2%}")
    print(f"⏱️  Processing Time: {result.processing_time:.2f}s")
    
    if "gpt_analysis" in result.raw_response:
        analysis = result.raw_response["gpt_analysis"]
        print(f"🔤 Script Type: {analysis.get('script_type', 'unknown')}")
        print(f"🗺️  Region/Dialect: {analysis.get('region_dialect', 'unknown')}")
        print(f"🔀 Code-Mixed: {analysis.get('is_code_mixed', False)}")
        if analysis.get('secondary_languages'):
            print(f"🌐 Secondary Languages: {', '.join(analysis['secondary_languages'])}")
        print(f"💡 Analysis: {analysis.get('reasoning', 'No reasoning provided')}")
    
    print("="*60)

def main():
    """Main PoC function"""
    print("🎤 Language Detection PoC with GPT-4o-mini")
    print("=" * 50)
    
    # Check API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please create a .env file with your OpenAI API key:")
        print("OPENAI_API_KEY=your_api_key_here")
        return
    
    # Initialize components
    try:
        recorder = AudioRecorder()
        detector = LanguageDetector()
        print("✅ Audio recorder and language detector initialized")
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        return
    
    try:
        while True:
            print("\n" + "-"*50)
            print("Options:")
            print("1. Record and detect language")
            print("2. Test with sample phrases")
            print("3. Exit")
            
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == "1":
                # Record audio and detect language
                recorder.start_recording()
                
                # Wait for user to press Enter
                input()  # This blocks until Enter is pressed
                
                print("🛑 Stopping recording...")
                audio_file = recorder.stop_recording()
                
                try:
                    # Detect language
                    result = detector.detect_language(audio_file)
                    print_results(result)
                    
                finally:
                    # Clean up temporary file
                    os.unlink(audio_file)
            
            elif choice == "2":
                # Test with sample phrases (you'd need to create sample audio files)
                print("📋 Sample test not implemented yet.")
                print("💡 To test, use option 1 and speak these phrases:")
                print("   • English: 'Hello, how are you today?'")
                print("   • Tamil: 'வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?'")
                print("   • Hindi: 'नमस्ते, आप कैसे हैं?'")
                print("   • Telugu: 'నమస్కారం, మీరు ఎలా ఉన్నారు?'")
            
            elif choice == "3":
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
    
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    
    finally:
        recorder.cleanup()
        print("🧹 Cleaned up resources")

if __name__ == "__main__":
    main()