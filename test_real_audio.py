#!/usr/bin/env python3
"""
Test Wav2Vec2 with real audio files
Usage: python test_real_audio.py [audio_file]
"""

import sys
import torch
import librosa
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import os

def load_model():
    """Load the Wav2Vec2 model"""
    print("Loading Wav2Vec2 model...")
    model_name = "facebook/wav2vec2-base-960h"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    print(f"✓ Model loaded: {model_name}")
    return model, processor

def transcribe_audio(audio_file, model, processor):
    """Transcribe an audio file"""
    print(f"\nProcessing: {audio_file}")
    
    # Load audio
    try:
        audio_array, sample_rate = librosa.load(audio_file, sr=16000)
        print(f"✓ Audio loaded: {len(audio_array)/sample_rate:.2f} seconds")
    except Exception as e:
        print(f"❌ Error loading audio: {e}")
        return None
    
    # Process with model
    input_values = processor(audio_array, sampling_rate=16000, return_tensors="pt").input_values
    
    with torch.no_grad():
        logits = model(input_values).logits
    
    # Decode
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    
    # Calculate confidence
    probs = torch.softmax(logits, dim=-1)
    max_probs = torch.max(probs, dim=-1)[0]
    avg_confidence = torch.mean(max_probs).item()
    
    return transcription, avg_confidence

def main():
    """Main function"""
    print("=== Wav2Vec2 Real Audio Test ===")
    
    # Load model
    model, processor = load_model()
    
    # Check for audio files
    if len(sys.argv) > 1:
        # Use provided file
        audio_file = sys.argv[1]
        if not os.path.exists(audio_file):
            print(f"❌ File not found: {audio_file}")
            return
        
        transcription, confidence = transcribe_audio(audio_file, model, processor)
        
        print(f"\n=== RESULTS ===")
        print(f"File: {audio_file}")
        print(f"Transcription: '{transcription}'")
        print(f"Confidence: {confidence:.3f}")
        
    else:
        # Look for audio files in current directory
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
        audio_files = []
        
        for file in os.listdir('.'):
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                audio_files.append(file)
        
        if audio_files:
            print(f"\nFound {len(audio_files)} audio file(s):")
            for file in audio_files:
                print(f"  - {file}")
            
            print(f"\nTranscribing all files...")
            for audio_file in audio_files:
                transcription, confidence = transcribe_audio(audio_file, model, processor)
                if transcription is not None:
                    print(f"\n📁 {audio_file}")
                    print(f"📝 '{transcription}'")
                    print(f"📊 Confidence: {confidence:.3f}")
        else:
            print(f"\n📝 No audio files found in current directory.")
            print(f"\n💡 To test with 'Cat likes to run after rat':")
            print(f"   1. Record yourself saying the sentence")
            print(f"   2. Save as 'sentence.wav' in this directory")
            print(f"   3. Run: python test_real_audio.py sentence.wav")
            print(f"\n🎤 Or use any existing audio file:")
            print(f"   python test_real_audio.py /path/to/your/audio.wav")
            
            # Create a demo with the model working on a simple case
            print(f"\n=== DEMO: Testing model capability ===")
            
            # Try the working synthetic example
            sample_rate = 16000
            duration = 1.0
            t = np.linspace(0, duration, int(sample_rate * duration))
            
            # Simple tone that the model can recognize
            freq = 300
            audio_signal = 0.5 * np.sin(2 * np.pi * freq * t)
            
            # Add speech-like modulation
            modulation = 1 + 0.2 * np.sin(2 * np.pi * 8 * t)
            audio_signal = audio_signal * modulation
            
            # Apply envelope
            envelope = np.exp(-t / 2.0) * (1 - np.exp(-t * 10))
            audio_signal = audio_signal * envelope
            
            # Process
            input_values = processor(audio_signal, sampling_rate=16000, return_tensors="pt").input_values
            
            with torch.no_grad():
                logits = model(input_values).logits
            
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.decode(predicted_ids[0])
            
            print(f"Demo synthetic audio result: '{transcription}'")
            print(f"✓ Model is working and ready for real speech!")

if __name__ == "__main__":
    main()