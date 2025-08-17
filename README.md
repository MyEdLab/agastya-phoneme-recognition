# Phoneme Recognition App

Unified web application for phoneme-level speech recognition using Wav2Vec2.

## 🎯 Features

### 📊 Post-Recording Analysis
- **Record audio** directly in browser or upload files  
- **Complete phoneme analysis** with timing and confidence
- **IPA transcription** with detailed breakdown
- **Interactive phoneme grid** with visual display

### ⚡ Real-time Streaming
- **Live phoneme detection** as you speak
- **WebSocket streaming** for low-latency processing
- **Real-time IPA transcription** updates
- **Confidence-based phoneme bubbles** with color coding
- **Session statistics** (phonemes/second, total count)

## 🚀 Quick Start

1. **Setup environment:**
   ```bash
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run the web app:**
   ```bash
   python phoneme_app.py
   ```

3. **Open browser:** http://localhost:5001

## 📁 Files

### Core Application
- `phoneme_app.py` - Main Flask web application
- `templates/index.html` - Web UI
- `phoneme_vocab.json` - Phoneme vocabulary (392 phonemes)
- `requirements.txt` - Python dependencies

### Utilities
- `extract_phoneme_vocab.py` - Downloads phoneme vocabulary from HuggingFace
- `test_real_audio.py` - Command-line tool for testing audio files

### Sample Data
- `sample.mp3` - Test audio file ("SHE SELLS SEA SHELLS")

## 🎤 Usage

### Mode Selection
The app has two modes accessible via tabs:
- **📊 Post-Recording Analysis** - Analyze complete recordings
- **⚡ Real-time Streaming** - Live phoneme detection

### Post-Recording Mode
1. Click "Start Recording" or "Upload Audio File"
2. Speak clearly into microphone or select audio file
3. Click "Stop" then "Analyze" (for recordings)
4. View complete phoneme breakdown with IPA transcription

### Real-time Mode  
1. Switch to "Real-time Streaming" tab
2. Click "Start Real-time"
3. Speak and see live phonemes appear as bubbles
4. View real-time IPA transcription and statistics

## 🔬 Technical Details

- **Model:** `facebook/wav2vec2-lv-60-espeak-cv-ft`
- **Vocabulary:** 392 eSpeak phonemes
- **Input:** 16kHz audio
- **Output:** IPA phoneme transcription with timing and confidence

## 📊 Example Output

**Input:** "She sells sea shells"
**Phonemes:** `ʃ i s a l s iː ʃ a n s`
**IPA:** `/ʃ i s a l s iː ʃ a n s/`

## 🛠️ Troubleshooting

- Ensure microphone permissions are granted
- Use Chrome/Firefox for best browser compatibility
- Audio files should be clear speech for best results