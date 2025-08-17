#!/usr/bin/env python3
"""
Simple Phoneme Recognition Web App (Post-recording only)
- No real-time streaming, just file/recording analysis
- English-only phoneme filtering
- Clean and simple architecture
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Dict, List, Tuple

import librosa
import numpy as np
import torch
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from transformers import AutoFeatureExtractor, Wav2Vec2ForCTC

# ------------------------------
# App Configuration
# ------------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = 'phoneme_secret_key'
CORS(app)

CKPT = "facebook/wav2vec2-lv-60-espeak-cv-ft"
SAMPLE_RATE = 16000

# Device selection (GPU -> MPS -> CPU)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# Model state
model: Wav2Vec2ForCTC | None = None
feature_extractor: AutoFeatureExtractor | None = None
id2label: Dict[int, str] | None = None
blank_id: int = 0

# English phoneme filter (can be overridden per request)
DEFAULT_ENGLISH_ONLY = True
ENGLISH_PHONEMES = {
    # Consonants
    "p", "b", "t", "d", "k", "ɡ", "g",
    "f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h",
    "tʃ", "dʒ", "m", "n", "ŋ", "l", "ɹ", "r", "j", "w",
    # Vowels
    "i", "ɪ", "e", "ɛ", "æ", "ɑ", "ɔ", "o", "ʊ", "u",
    "ʌ", "ə", "ɚ", "ɝ",
    # Diphthongs
    "eɪ", "aɪ", "ɔɪ", "aʊ", "oʊ", "ɪə", "eə", "ʊə",
    # With length markers
    "iː", "uː", "ɑː", "ɔː", "ɜː", "aː", "eː", "oː",
    # Syllabic consonants
    "l̩", "n̩", "m̩", "r̩",
    # Special
    "<pad>", "<s>", "</s>", "<unk>"
}

# ------------------------------
# Helper Functions
# ------------------------------

def is_english_phoneme(phoneme, english_only=None):
    """Check if phoneme is English"""
    if english_only is None:
        english_only = DEFAULT_ENGLISH_ONLY
    if not english_only:
        return True
    base = phoneme.replace('ˈ', '').replace('ˌ', '')
    return base in ENGLISH_PHONEMES or phoneme in ENGLISH_PHONEMES

def normalize_seq(seq: str) -> str:
    """Normalize phoneme sequence for display"""
    toks = [t for t in seq.split() if t not in ("<pad>", "<s>", "</s>", "<unk>")]
    # Optional: map 'r' to 'ɹ' for IPA correctness
    return " ".join(toks)

def preprocess_audio(x: np.ndarray) -> np.ndarray:
    """Rescale to mono float32 in [-1, 1] and peak-normalize to ~0.95."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = np.mean(x, axis=0)
    m = float(np.max(np.abs(x)) or 1.0)
    return 0.95 * (x / m)

# ------------------------------
# Model Functions
# ------------------------------

def logits_to_tokens(
    logits: torch.Tensor,  # [1, T, V]
    length_seconds: float,
    english_only: bool = None,
) -> List[Dict]:
    """Convert model logits to phoneme tokens with timing"""
    assert model is not None and id2label is not None
    
    with torch.no_grad():
        logp = torch.log_softmax(logits, dim=-1)[0]  # [T, V]
    ids = torch.argmax(logits, dim=-1)[0].tolist()   # [T]
    
    T = logp.size(0)
    if T == 0:
        return []
    sec_per_frame = length_seconds / float(T)
    
    tokens: List[Dict] = []
    i = 0
    while i < T:
        tid = ids[i]
        j = i + 1
        # skip blanks
        if tid == blank_id:
            i = j
            continue
        # extend span of same tid
        while j < T and ids[j] == tid:
            j += 1
        # compute mean probability over the span as confidence
        conf = float(torch.exp(logp[i:j, tid]).mean().item())
        # map id -> label
        label = id2label.get(tid, f"<{tid}>")
        # Filter non-English phonemes if enabled
        if is_english_phoneme(label, english_only):
            tokens.append({
                "id": tid,
                "phoneme": label,
                "confidence": conf,
                "start": i * sec_per_frame,
                "end": j * sec_per_frame,
            })
        i = j
    
    return tokens

def run_model_on_audio(wave: np.ndarray, english_only: bool = None) -> Tuple[str, List[Dict]]:
    """Run inference on audio, return (decoded_string, token_list)"""
    assert model is not None and feature_extractor is not None
    wave = preprocess_audio(wave)
    
    inputs = feature_extractor(wave, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    input_values = inputs["input_values"].to(torch.float32)
    length_seconds = len(wave) / float(SAMPLE_RATE)
    
    with torch.no_grad():
        logits = model(input_values.to(DEVICE)).logits.cpu()
    
    tokens = logits_to_tokens(logits, length_seconds, english_only)
    decoded = " ".join(t["phoneme"] for t in tokens)
    return decoded, tokens

def load_phoneme_model():
    """Load the phoneme model and vocabulary"""
    global model, feature_extractor, id2label, blank_id
    print("Loading phoneme model...")
    
    # Load model and feature extractor
    fx = AutoFeatureExtractor.from_pretrained(CKPT)
    m = Wav2Vec2ForCTC.from_pretrained(CKPT)
    m.to(DEVICE).eval()
    
    # Load vocabulary - prioritize external file which has full phoneme set
    if os.path.exists('phoneme_vocab.json'):
        with open('phoneme_vocab.json', 'r') as f:
            vocab = json.load(f)
        id2 = {v: k for k, v in vocab.items()}  # swap key-value for id->label mapping
        print(f"✓ Loaded vocabulary from phoneme_vocab.json")
    else:
        # Fallback to model config (usually incomplete for this model)
        cfg_map = getattr(m.config, 'id2label', None)
        if isinstance(cfg_map, dict) and cfg_map:
            id2 = {int(k): v for k, v in cfg_map.items()}
            print(f"⚠️  Using model config vocabulary (may be incomplete)")
        else:
            raise RuntimeError("No vocabulary found. Please run extract_phoneme_vocab.py first.")
    
    # CTC blank (pad usually 0)
    b_id = getattr(m.config, 'pad_token_id', None)
    if b_id is None:
        b_id = 0
    
    # Commit globals
    globals()['feature_extractor'] = fx
    globals()['model'] = m
    globals()['id2label'] = id2
    globals()['blank_id'] = int(b_id)
    
    # Count English phonemes if filter is enabled
    if DEFAULT_ENGLISH_ONLY:
        english_count = sum(1 for p in id2.values() if is_english_phoneme(p))
        print(f"✓ Loaded {CKPT} on {DEVICE}")
        print(f"✓ Vocabulary: {english_count} English phonemes (filtered from {len(id2)} total)")
    else:
        print(f"✓ Loaded {CKPT} on {DEVICE} | {len(id2)} phonemes total")

# ------------------------------
# Flask Routes
# ------------------------------

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/model_status')
def model_status():
    """Check model loading status"""
    english_count = 0
    if id2label and DEFAULT_ENGLISH_ONLY:
        english_count = sum(1 for p in id2label.values() if is_english_phoneme(p))
    
    status = {
        'model_loaded': model is not None,
        'vocab_loaded': id2label is not None and len(id2label) > 0,
        'device': str(DEVICE),
        'vocab_size': english_count if DEFAULT_ENGLISH_ONLY else (len(id2label) if id2label else 0),
        'english_only': DEFAULT_ENGLISH_ONLY,
        'total_vocab_size': len(id2label) if id2label else 0,
    }
    return jsonify(status)

@app.route('/analyze', methods=['POST'])
def analyze_audio():
    """Analyze uploaded audio file"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get english_only parameter from form data
        english_only = request.form.get('english_only', 'true').lower() == 'true'
        
        # Save and load audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            audio_file.save(tmp.name)
            wave, sr = librosa.load(tmp.name, sr=SAMPLE_RATE)
        os.unlink(tmp.name)
        
        # Run inference
        decoded, tokens = run_model_on_audio(wave, english_only)
        norm = normalize_seq(decoded)
        avg_conf = float(np.mean([t['confidence'] for t in tokens]) if tokens else 0.0)
        
        return jsonify({
            'phonemes': tokens,
            'sequence_raw': decoded,
            'sequence_normalized': norm,
            'sequence': norm,  # For backward compatibility
            'ipa_transcription': f"/{norm}/",
            'avg_confidence': avg_conf,
            'total_phonemes': len(tokens),
            'audio_duration': float(len(wave) / SAMPLE_RATE),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ------------------------------
# Main Entry Point
# ------------------------------

if __name__ == '__main__':
    load_phoneme_model()
    
    print("\n🎤 Simple Phoneme Recognition App")
    print("=" * 60)
    print("✓ Model loaded successfully")
    print("✓ English-only filter:", "ENABLED" if DEFAULT_ENGLISH_ONLY else "DISABLED")
    print("✓ Server starting...")
    print("\n📱 Open your browser and go to: http://localhost:5001")
    print("\n🎯 Features:")
    print("   - Browser audio recording")
    print("   - Audio file upload")
    print("   - English phoneme filtering")
    print("   - IPA transcription")
    
    app.run(debug=True, host='0.0.0.0', port=5001)