#!/usr/bin/env python3
"""
Unified Phoneme Recognition Web App
Both post-recording and real-time phoneme analysis in one app
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import torch
import librosa
import numpy as np
import json
import os
import tempfile
import base64
import io
import threading
import time
from collections import deque
from transformers import Wav2Vec2ForCTC, AutoConfig
# Phoneme processor is now integrated directly

app = Flask(__name__)
app.config['SECRET_KEY'] = 'phoneme_secret_key'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global variables for model
model = None
config = None
id_to_phoneme = None
is_realtime_active = False

class RealTimePhonemeProcessor:
    def __init__(self, sample_rate=16000, chunk_duration=0.5, overlap=0.1):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration  
        self.overlap = overlap  
        self.chunk_size = int(sample_rate * chunk_duration)
        self.overlap_size = int(sample_rate * overlap)
        self.buffer = deque(maxlen=int(sample_rate * 2))
        
        # Essential filtering parameters only
        self.min_confidence = 0.45  # Balanced threshold
        self.min_phoneme_duration = 0.05  # 50ms minimum (less aggressive)
        
    def preprocess_audio(self, audio_data):
        """Enhanced audio preprocessing"""
        if len(audio_data) == 0:
            return audio_data
            
        # Convert to numpy if needed
        if isinstance(audio_data, list):
            audio_data = np.array(audio_data, dtype=np.float32)
        
        # Noise reduction - remove very low amplitude noise
        noise_floor = np.percentile(np.abs(audio_data), 10)
        audio_data = np.where(np.abs(audio_data) < noise_floor * 2, 
                             audio_data * 0.1, audio_data)
        
        # Normalize with soft limiting
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / (max_val + 1e-8) * 0.8
        
        # Light low-pass filtering to remove high-frequency noise
        if len(audio_data) > 3:
            kernel = np.array([0.25, 0.5, 0.25])
            audio_data = np.convolve(audio_data, kernel, mode='same')
        
        return audio_data
        
    def add_audio(self, audio_data):
        """Add new audio data to the buffer with preprocessing"""
        processed_audio = self.preprocess_audio(audio_data)
        self.buffer.extend(processed_audio)
    
    def get_chunk_for_analysis(self):
        """Get a chunk of audio for analysis"""
        if len(self.buffer) >= self.chunk_size:
            chunk = np.array(list(self.buffer)[-self.chunk_size:])
            return chunk
        return None
    
    def filter_repeated_phonemes(self, phonemes):
        """Basic filtering of obvious repeats only"""
        if not phonemes:
            return phonemes
            
        filtered = []
        
        for phoneme in phonemes:
            # Only filter if same phoneme appears very close in time
            if filtered:
                time_since_last = phoneme['timestamp'] - filtered[-1]['timestamp']
                # Only filter if it's the exact same phoneme very close together
                if (filtered[-1]['phoneme'] == phoneme['phoneme'] and 
                    time_since_last < self.min_phoneme_duration):
                    continue  # Skip this obvious repeat
            
            filtered.append(phoneme)
                
        return filtered
    
    def apply_basic_cleanup(self, phonemes):
        """Apply only essential cleanup - no overfitted corrections"""
        if not phonemes:
            return phonemes
            
        # Just return phonemes as-is, trusting the model output
        # Only basic cleanup that's clearly beneficial
        cleaned = []
        
        for phoneme in phonemes:
            # Skip obvious artifacts (very short, very low confidence)
            if (phoneme['confidence'] > 0.3 and 
                phoneme['phoneme'] not in ['<pad>', '<s>', '</s>', '<unk>']):
                cleaned.append(phoneme)
            
        return cleaned
    
    def remove_boundary_artifacts(self, phonemes):
        """Remove only clear artifacts at boundaries"""
        if len(phonemes) <= 1:
            return phonemes
            
        # Only remove if very low confidence AND very short
        cleaned = phonemes[:]
        
        # Remove first phoneme only if clearly spurious
        if (len(cleaned) > 1 and 
            cleaned[0]['confidence'] < 0.35 and
            cleaned[1]['timestamp'] - cleaned[0]['timestamp'] < 0.03):
            cleaned = cleaned[1:]
            
        # Remove last phoneme only if clearly spurious  
        if (len(cleaned) > 1 and 
            cleaned[-1]['confidence'] < 0.35 and
            cleaned[-1]['timestamp'] - cleaned[-2]['timestamp'] < 0.03):
            cleaned = cleaned[:-1]
        
        return cleaned
    
    def process_chunk(self, chunk, model, config, id_to_phoneme):
        """Process a chunk with filtering and corrections"""
        try:
            audio_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                outputs = model(audio_tensor)
                logits = outputs.logits
            
            predicted_ids = torch.argmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)
            max_probs = torch.max(probs, dim=-1)[0]
            
            sequence = predicted_ids.squeeze().tolist()
            confidences = max_probs.squeeze().tolist()
            
            # Initial filtering with higher confidence threshold
            raw_phonemes = []
            prev_id = -1
            
            for i, (token_id, confidence) in enumerate(zip(sequence, confidences)):
                if (token_id != prev_id and 
                    token_id != 0 and 
                    confidence > self.min_confidence):
                    
                    phoneme = id_to_phoneme.get(token_id, f"<{token_id}>")
                    if phoneme not in ['<pad>', '<s>', '</s>', '<unk>']:
                        raw_phonemes.append({
                            'phoneme': phoneme,
                            'confidence': float(confidence),
                            'timestamp': time.time(),
                            'frame_index': i
                        })
                prev_id = token_id
            
            # Apply only essential, non-overfitted processing
            filtered_phonemes = self.filter_repeated_phonemes(raw_phonemes)
            cleaned_phonemes = self.apply_basic_cleanup(filtered_phonemes)
            final_phonemes = self.remove_boundary_artifacts(cleaned_phonemes)
            
            return final_phonemes
            
        except Exception as e:
            print(f"Error processing chunk: {e}")
            return []

# Global processor
processor = RealTimePhonemeProcessor()

def load_phoneme_model():
    """Load the phoneme model and vocabulary"""
    global model, config, id_to_phoneme
    
    print("Loading phoneme model...")
    model_name = "facebook/wav2vec2-lv-60-espeak-cv-ft"
    
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    model.eval()  # Optimization for inference
    
    # Load vocabulary
    if os.path.exists('phoneme_vocab.json'):
        with open('phoneme_vocab.json', 'r') as f:
            vocab = json.load(f)
        id_to_phoneme = {v: k for k, v in vocab.items()}
        print(f"✓ Model loaded with {len(vocab)} phonemes")
    else:
        print("⚠️  Phoneme vocabulary not found, run extract_phoneme_vocab.py first")
        id_to_phoneme = {}

def process_audio_for_phonemes(audio_data, sample_rate=16000):
    """Process audio and return phoneme analysis"""
    global model, config, id_to_phoneme
    
    temp_processor = RealTimePhonemeProcessor()
    
    try:
        # Preprocess entire audio
        processed_audio = temp_processor.preprocess_audio(audio_data)
        
        audio_tensor = torch.tensor(processed_audio, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(audio_tensor)
            logits = outputs.logits
        
        predicted_ids = torch.argmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        max_probs = torch.max(probs, dim=-1)[0]
        
        sequence = predicted_ids.squeeze().tolist()
        confidences = max_probs.squeeze().tolist()
        
        # Extract phonemes with improved filtering
        raw_phonemes = []
        prev_id = -1
        
        for i, (token_id, confidence) in enumerate(zip(sequence, confidences)):
            if (token_id != prev_id and 
                token_id != 0 and 
                confidence > temp_processor.min_confidence):
                
                phoneme = id_to_phoneme.get(token_id, f"<{token_id}>")
                if phoneme not in ['<pad>', '<s>', '</s>', '<unk>']:
                    frame_duration = len(processed_audio) / sample_rate / len(sequence)
                    timestamp = i * frame_duration
                    
                    raw_phonemes.append({
                        'id': token_id,
                        'phoneme': phoneme,
                        'confidence': float(confidence),
                        'timestamp': float(timestamp),
                        'frame_index': i
                    })
            prev_id = token_id
        
        # Apply only essential processing
        filtered_phonemes = temp_processor.filter_repeated_phonemes(raw_phonemes)
        cleaned_phonemes = temp_processor.apply_basic_cleanup(filtered_phonemes)
        final_phonemes = temp_processor.remove_boundary_artifacts(cleaned_phonemes)
        
        # Create summary
        phoneme_sequence = [p['phoneme'] for p in final_phonemes]
        avg_confidence = np.mean([p['confidence'] for p in final_phonemes]) if final_phonemes else 0
        
        return {
            'phonemes': final_phonemes,
            'sequence': ' '.join(phoneme_sequence),
            'ipa_transcription': '/' + ' '.join(phoneme_sequence) + '/',
            'avg_confidence': float(avg_confidence),
            'total_phonemes': len(final_phonemes),
            'audio_duration': float(len(processed_audio) / sample_rate)
        }
    
    except Exception as e:
        return {'error': str(e)}

@app.route('/')
def index():
    """Main page with both modes"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_audio():
    """Analyze uploaded audio file (post-recording mode)"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            audio_file.save(tmp_file.name)
            audio_data, sr = librosa.load(tmp_file.name, sr=16000)
            os.unlink(tmp_file.name)
        
        result = process_audio_for_phonemes(audio_data, sr)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_blob', methods=['POST'])
def analyze_blob():
    """Analyze audio blob from browser recording (post-recording mode)"""
    try:
        data = request.get_json()
        
        if 'audio_data' not in data:
            return jsonify({'error': 'No audio data provided'}), 400
        
        audio_blob = base64.b64decode(data['audio_data'])
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_blob)
            tmp_file.flush()
            audio_data, sr = librosa.load(tmp_file.name, sr=16000)
            os.unlink(tmp_file.name)
        
        result = process_audio_for_phonemes(audio_data, #!/usr/bin/env python3
"""
Unified Phoneme Recognition Web App (Improved)
- Uses Hugging Face AutoProcessor for feature extraction + decoding
- Correct frame-to-time timestamps and per-token confidence
- Streaming with overlap + center-cut emission to avoid duplicates
- Simpler audio preprocessing (no waveform-warping filters)
- No external vocab file (reads labels from model config)
- Optional normalization of symbols for display/eval
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import tempfile
import threading
import time
from collections import deque
from typing import Dict, List, Tuple

import librosa
import numpy as np
import torch
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from transformers import AutoProcessor, Wav2Vec2ForCTC

# ------------------------------
# App & Globals
# ------------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = 'phoneme_secret_key'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

CKPT = "facebook/wav2vec2-lv-60-espeak-cv-ft"
SAMPLE_RATE = 16000

# Device selection (GPU -> MPS -> CPU)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# Limit CPU threads a bit for stability
try:
    torch.set_num_threads(min(4, os.cpu_count() or 4))
except Exception:
    pass

# Model state
model: Wav2Vec2ForCTC | None = None
processor_hf: AutoProcessor | None = None
id2label: Dict[int, str] | None = None
blank_id: int = 0
model_lock = threading.Lock()

# Streaming state flag
is_realtime_active = False

# ------------------------------
# Normalization helpers (display/eval convenience)
# ------------------------------
NORM_MAP = {
    # cosmetic / preference mappings (adjust to taste)
    "r": "ɹ",  # many eSpeak vocabs use plain 'r'
}
STRIP_CHARS = {"ˈ", "ˌ", "ː"}


def normalize_seq(seq: str) -> str:
    toks = [t for t in seq.split() if t not in ("<pad>", "<s>", "</s>", "<unk>")]
    out = []
    for t in toks:
        # drop stress/length
        t = "".join(ch for ch in t if ch not in STRIP_CHARS)
        out.append(NORM_MAP.get(t, t))
    return " ".join(out)


# ------------------------------
# Audio preprocessing (conservative)
# ------------------------------

def preprocess_audio(x: np.ndarray) -> np.ndarray:
    """Rescale to mono float32 in [-1, 1] and peak-normalize to ~0.95."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = np.mean(x, axis=0)
    m = float(np.max(np.abs(x)) or 1.0)
    return 0.95 * (x / m)


# ------------------------------
# Inference utilities
# ------------------------------

def logits_to_tokens(
    logits: torch.Tensor,  # [1, T, V]
    length_seconds: float,
) -> Tuple[List[Dict], float, List[int]]:
    """Collapse framewise logits to token spans with confidence and timestamps.

    Returns: (tokens, sec_per_frame, argmax_ids)
    token dict: {id, phoneme, confidence, start, end}
    """
    assert model is not None and processor_hf is not None and id2label is not None

    with torch.no_grad():
        logp = torch.log_softmax(logits, dim=-1)[0]  # [T, V]
    ids = torch.argmax(logits, dim=-1)[0].tolist()   # [T]

    T = logp.size(0)
    if T == 0:
        return [], 0.0, []
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
        # map id -> label (defensive)
        label = id2label.get(tid, f"<{tid}>")
        tokens.append({
            "id": tid,
            "phoneme": label,
            "confidence": conf,
            "start": i * sec_per_frame,
            "end": j * sec_per_frame,
        })
        i = j

    return tokens, sec_per_frame, ids


def decode_argmax(logits: torch.Tensor) -> str:
    """Best-path CTC decode using the processor's batch_decode (argmax path)."""
    assert processor_hf is not None
    pred_ids = torch.argmax(logits, dim=-1)
    return processor_hf.batch_decode(pred_ids)[0]


def run_model_on_audio(wave: np.ndarray) -> Tuple[str, List[Dict]]:
    """Run full-file inference, return (decoded_str, token_spans)."""
    assert model is not None and processor_hf is not None
    wave = preprocess_audio(wave)

    inputs = processor_hf(wave, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    length_seconds = len(wave) / float(SAMPLE_RATE)

    with model_lock:
        with torch.no_grad():
            logits = model(inputs.input_values.to(DEVICE)).logits.cpu()

    tokens, _, _ = logits_to_tokens(logits, length_seconds)
    decoded = decode_argmax(logits)
    return decoded, tokens


# ------------------------------
# Real-time streaming helper
# ------------------------------
class RealTimePhonemeProcessor:
    """Sliding-window chunker with overlap and center-cut emission."""

    def __init__(self, sample_rate=SAMPLE_RATE, chunk_seconds=0.8, overlap_seconds=0.3,
                 min_confidence=0.35, min_duration_s=0.05):
        self.sr = sample_rate
        self.chunk_s = float(chunk_seconds)
        self.overlap_s = float(overlap_seconds)
        assert 0.0 < self.overlap_s < self.chunk_s
        self.step_s = self.chunk_s - self.overlap_s

        self.chunk_n = int(round(self.chunk_s * self.sr))
        self.step_n = int(round(self.step_s * self.sr))

        self.buf = np.zeros(0, dtype=np.float32)
        self.next_start = 0  # sample index of the next chunk start
        self.global_offset_n = 0  # samples already emitted before buf[0]

        self.emit_start_s = self.overlap_s
        self.emit_end_s = self.chunk_s - self.overlap_s

        self.min_conf = float(min_confidence)
        self.min_dur_s = float(min_duration_s)

    def add_audio(self, audio_data: np.ndarray):
        x = preprocess_audio(audio_data)
        self.buf = np.concatenate([self.buf, x])

    def ready_chunks(self) -> List[Tuple[np.ndarray, float]]:
        """Return list of (chunk_waveform, chunk_global_offset_seconds)."""
        out: List[Tuple[np.ndarray, float]] = []
        while self.next_start + self.chunk_n <= len(self.buf):
            start = self.next_start
            end = start + self.chunk_n
            chunk = self.buf[start:end]
            # global offset in seconds for the beginning of this chunk
            chunk_offset_s = (self.global_offset_n + start) / float(self.sr)
            out.append((chunk, chunk_offset_s))
            # advance by step
            self.next_start += self.step_n
        # compact buffer occasionally to keep memory bounded
        if self.next_start > self.chunk_n * 3:
            # keep a tail that still contains the next chunk start
            keep_from = self.next_start - self.step_n  # keep one step before
            keep_from = max(0, keep_from)
            self.buf = self.buf[keep_from:]
            self.global_offset_n += keep_from
            self.next_start -= keep_from
        return out

    def center_cut(self, token: Dict) -> bool:
        mid = 0.5 * (token["start"] + token["end"])  # seconds relative to chunk
        return self.emit_start_s <= mid <= self.emit_end_s


# Global real-time processor instance
rtp = RealTimePhonemeProcessor()


# ------------------------------
# Flask routes
# ------------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/model_status')
def model_status():
    status = {
        'model_loaded': model is not None,
        'processor_loaded': processor_hf is not None,
        'device': str(DEVICE),
        'vocab_size': (len(id2label) if id2label else 0),
        'blank_id': blank_id,
        'realtime_active': is_realtime_active,
    }
    return jsonify(status)


@app.route('/vocab')
def vocab():
    if id2label is None:
        return jsonify({'error': 'model not loaded'}), 503
    labels = [id2label[i] for i in sorted(id2label.keys())]
    return jsonify({'labels': labels, 'size': len(labels), 'blank_id': blank_id})


@app.route('/analyze', methods=['POST'])
def analyze_audio():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            audio_file.save(tmp.name)
            wave, sr = librosa.load(tmp.name, sr=SAMPLE_RATE)
        os.unlink(tmp.name)

        decoded, tokens = run_model_on_audio(wave)
        norm = normalize_seq(decoded)
        avg_conf = float(np.mean([t['confidence'] for t in tokens]) if tokens else 0.0)
        return jsonify({
            'phonemes': tokens,
            'sequence_raw': decoded,
            'sequence_normalized': norm,
            'ipa_transcription': f"/{norm}/",
            'avg_confidence': avg_conf,
            'total_phonemes': len(tokens),
            'audio_duration': float(len(wave) / SAMPLE_RATE),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/analyze_blob', methods=['POST'])
def analyze_blob():
    try:
        data = request.get_json(force=True)
        if 'audio_data' not in data:
            return jsonify({'error': 'No audio data provided'}), 400
        audio_blob = base64.b64decode(data['audio_data'])
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            tmp.write(audio_blob)
            tmp.flush()
            wave, sr = librosa.load(tmp.name, sr=SAMPLE_RATE)
        os.unlink(tmp.name)

        decoded, tokens = run_model_on_audio(wave)
        norm = normalize_seq(decoded)
        avg_conf = float(np.mean([t['confidence'] for t in tokens]) if tokens else 0.0)
        return jsonify({
            'phonemes': tokens,
            'sequence_raw': decoded,
            'sequence_normalized': norm,
            'ipa_transcription': f"/{norm}/",
            'avg_confidence': avg_conf,
            'total_phonemes': len(tokens),
            'audio_duration': float(len(wave) / SAMPLE_RATE),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ------------------------------
# WebSocket handlers (streaming)
# ------------------------------
@socketio.on('connect')
def on_connect():
    emit('status', {'message': 'Connected to phoneme server'})


@socketio.on('disconnect')
def on_disconnect():
    global is_realtime_active
    is_realtime_active = False


@socketio.on('start_realtime')
def handle_start_realtime():
    global is_realtime_active, rtp
    is_realtime_active = True
    rtp = RealTimePhonemeProcessor()  # reset state
    emit('realtime_status', {'active': True, 'message': 'Real-time processing started'})


@socketio.on('stop_realtime')
def handle_stop_realtime():
    global is_realtime_active
    is_realtime_active = False
    emit('realtime_status', {'active': False, 'message': 'Real-time processing stopped'})


def process_chunk_and_emit(chunk: np.ndarray, chunk_offset_s: float):
    """Infer on one chunk, center-cut, and emit results with global timestamps."""
    assert model is not None and processor_hf is not None

    wave = preprocess_audio(chunk)
    inputs = processor_hf(wave, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    length_seconds = len(wave) / float(SAMPLE_RATE)

    with model_lock:
        with torch.no_grad():
            logits = model(inputs.input_values.to(DEVICE)).logits.cpu()

    # token spans relative to chunk
    tokens, sec_per_frame, _ = logits_to_tokens(logits, length_seconds)

    # center-cut filtering + min-duration/confidence
    min_frames = max(1, math.ceil(rtp.min_dur_s / (sec_per_frame or 1e-6)))
    kept: List[Dict] = []
    for tok in tokens:
        dur_frames = max(1, int(round((tok['end'] - tok['start']) / (sec_per_frame or 1e-6))))
        if tok['confidence'] < rtp.min_conf:
            continue
        if dur_frames < min_frames:
            continue
        if not rtp.center_cut(tok):
            continue
        # convert to global time
        start_g = tok['start'] + chunk_offset_s
        end_g = tok['end'] + chunk_offset_s
        kept.append({
            'id': tok['id'],
            'phoneme': tok['phoneme'],
            'confidence': tok['confidence'],
            'start': start_g,
            'end': end_g,
        })

    # Also produce a string for the whole chunk (raw + normalized)
    decoded = decode_argmax(logits)
    norm = normalize_seq(decoded)
    avg_conf = float(np.mean([t['confidence'] for t in kept]) if kept else 0.0)

    if kept:
        emit('phoneme_result', {
            'phonemes': kept,
            'sequence_raw': decoded,
            'sequence_normalized': norm,
            'ipa_transcription': f"/{norm}/",
            'avg_confidence': avg_conf,
            'timestamp': time.time(),
        })


@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    global is_realtime_active, rtp
    if not is_realtime_active:
        return
    try:
        audio = np.array(data.get('audio', []), dtype=np.float32)
        if audio.size == 0:
            return
        rtp.add_audio(audio)
        for chunk, offset_s in rtp.ready_chunks():
            process_chunk_and_emit(chunk, offset_s)
    except Exception as e:
        emit('error', {'message': f'Error processing audio: {str(e)}'})


# ------------------------------
# Model loading
# ------------------------------

def load_phoneme_model():
    global model, processor_hf, id2label, blank_id
    print("Loading phoneme model…")
    processor = AutoProcessor.from_pretrained(CKPT)
    m = Wav2Vec2ForCTC.from_pretrained(CKPT)
    m.to(DEVICE).eval()

    # id2label mapping
    cfg_map = getattr(m.config, 'id2label', None)
    if isinstance(cfg_map, dict) and cfg_map:
        # keys are strings in HF configs
        id2 = {int(k): v for k, v in cfg_map.items()}
    else:
        # build from processor tokenizer if needed
        labels = processor.tokenizer.get_vocab()
        inv = {v: k for k, v in labels.items()}
        id2 = {int(i): inv[i] for i in inv}

    # CTC blank
    b_id = getattr(processor.tokenizer, 'pad_token_id', None)
    if b_id is None:
        b_id = getattr(m.config, 'pad_token_id', 0)

    # commit
    globals()['processor_hf'] = processor
    globals()['model'] = m
    globals()['id2label'] = id2
    globals()['blank_id'] = int(b_id)

    print(f"✓ Loaded {CKPT} on {DEVICE} | labels: {len(id2)} | blank_id={blank_id}")


# ------------------------------
# Entrypoint
# ------------------------------
if __name__ == '__main__':
    load_phoneme_model()

    print("\n🎤 Phoneme Recognition App (Improved)")
    print("=" * 60)
    print("✓ Model loaded successfully")
    print("✓ Server starting…")
    print("\n📱 Open your browser and go to: http://localhost:5001")
    print("\n🎯 Features:")
    print("   - Post-recording analysis (/analyze, /analyze_blob)")
    print("   - Real-time phoneme streaming (Socket.IO)")
    print("   - Raw & normalized sequences + timestamps")
    print("   - /vocab route to inspect labels")

    socketio.run(app, debug=True, host='0.0.0.0', port=5001, allow_unsafe_werkzeug=True)
sr)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model_status')
def model_status():
    """Check if model is loaded"""
    global model, config, id_to_phoneme, is_realtime_active
    
    status = {
        'model_loaded': model is not None,
        'config_loaded': config is not None,
        'vocab_loaded': id_to_phoneme is not None and len(id_to_phoneme) > 0,
        'vocab_size': len(id_to_phoneme) if id_to_phoneme else 0,
        'realtime_active': is_realtime_active
    }
    
    return jsonify(status)

# WebSocket handlers for real-time mode
@socketio.on('connect')
def on_connect():
    """Handle client connection"""
    print('Client connected')
    emit('status', {'message': 'Connected to phoneme server'})

@socketio.on('disconnect')
def on_disconnect():
    """Handle client disconnection"""
    global is_realtime_active
    is_realtime_active = False
    print('Client disconnected')

@socketio.on('start_realtime')
def handle_start_realtime():
    """Start real-time processing"""
    global is_realtime_active, processor
    
    is_realtime_active = True
    processor = RealTimePhonemeProcessor()
    
    print("Real-time phoneme processing started")
    emit('realtime_status', {'active': True, 'message': 'Real-time processing started'})

@socketio.on('stop_realtime')
def handle_stop_realtime():
    """Stop real-time processing"""
    global is_realtime_active
    
    is_realtime_active = False
    print("Real-time phoneme processing stopped")
    emit('realtime_status', {'active': False, 'message': 'Real-time processing stopped'})

@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    """Handle incoming audio chunk for real-time processing"""
    global is_realtime_active, processor
    
    if not is_realtime_active:
        return
    
    try:
        audio_data = np.array(data.get('audio', []), dtype=np.float32)
        
        if len(audio_data) > 0:
            processor.add_audio(audio_data)
            chunk = processor.get_chunk_for_analysis()
            
            if chunk is not None:
                phonemes = processor.process_chunk(chunk, model, config, id_to_phoneme)
                
                if phonemes:
                    phoneme_sequence = ' '.join([p['phoneme'] for p in phonemes])
                    ipa_transcription = '/' + phoneme_sequence + '/'
                    avg_confidence = np.mean([p['confidence'] for p in phonemes])
                    
                    emit('phoneme_result', {
                        'phonemes': phonemes,
                        'sequence': phoneme_sequence,
                        'ipa_transcription': ipa_transcription,
                        'avg_confidence': float(avg_confidence),
                        'timestamp': time.time()
                    })
    
    except Exception as e:
        print(f"Error processing audio chunk: {e}")
        emit('error', {'message': f'Error processing audio: {str(e)}'})

if __name__ == '__main__':
    # Load model on startup
    load_phoneme_model()
    
    print("\n🎤 Phoneme Recognition App")
    print("=" * 60)
    print("✓ Model loaded successfully")
    print("✓ Server starting...")
    print("\n📱 Open your browser and go to: http://localhost:5001")
    print("\n🎯 Features:")
    print("   - Post-recording analysis")
    print("   - Real-time phoneme streaming")
    print("   - File upload support")
    print("   - Live IPA transcription")
    print("   - WebSocket + HTTP in one app")
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5001, allow_unsafe_werkzeug=True)