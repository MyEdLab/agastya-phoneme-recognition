#!/usr/bin/env python3
"""
Extract the actual phoneme vocabulary from the model
"""

import json
import requests
from transformers import AutoConfig
import os

def get_vocab_from_huggingface():
    """Download the vocab.json directly from HuggingFace"""
    model_name = "facebook/wav2vec2-lv-60-espeak-cv-ft"
    print(f"Fetching vocabulary from HuggingFace for: {model_name}")
    
    # Try to download the vocab.json file
    vocab_url = f"https://huggingface.co/{model_name}/resolve/main/vocab.json"
    
    try:
        response = requests.get(vocab_url)
        if response.status_code == 200:
            vocab = response.json()
            print(f"✓ Downloaded vocab.json successfully")
            print(f"✓ Vocabulary size: {len(vocab)}")
            
            # Save locally
            with open('phoneme_vocab.json', 'w') as f:
                json.dump(vocab, f, indent=2)
            print(f"✓ Saved to phoneme_vocab.json")
            
            return vocab
        else:
            print(f"✗ Failed to download vocab.json: {response.status_code}")
            return None
    except Exception as e:
        print(f"✗ Error downloading vocab: {e}")
        return None

def analyze_phoneme_vocab(vocab):
    """Analyze the phoneme vocabulary"""
    print(f"\n=== PHONEME VOCABULARY ANALYSIS ===")
    
    # Create reverse mapping (ID to phoneme)
    id_to_phoneme = {v: k for k, v in vocab.items()}
    
    # Show some examples
    print(f"\nFirst 20 phonemes:")
    for i in range(20):
        if i in id_to_phoneme:
            print(f"  ID {i:3d}: '{id_to_phoneme[i]}'")
    
    # Show specific IDs from our sample
    sample_ids = [38, 10, 5, 9, 8, 5, 30, 38, 9, 4, 5]
    print(f"\nPhonemes from our 'SHE SELLS SEA SHELLS' sample:")
    print(f"IDs: {sample_ids}")
    
    decoded_phonemes = []
    for pid in sample_ids:
        if pid in id_to_phoneme:
            phoneme = id_to_phoneme[pid]
            decoded_phonemes.append(phoneme)
            print(f"  ID {pid:3d}: '{phoneme}'")
    
    # Join the phonemes
    phoneme_string = ' '.join(decoded_phonemes)
    print(f"\nDecoded phoneme sequence: {phoneme_string}")
    
    # Show special tokens
    print(f"\nSpecial tokens in vocabulary:")
    for token, token_id in vocab.items():
        if token.startswith('<') or token == '|':
            print(f"  '{token}': ID {token_id}")
    
    return id_to_phoneme

def map_phonemes_to_ipa():
    """Map eSpeak phonemes to IPA (International Phonetic Alphabet)"""
    print(f"\n=== ESPEAK TO IPA MAPPING ===")
    
    # Common eSpeak to IPA mappings
    espeak_to_ipa = {
        'ʃ': 'ʃ',     # 'sh' as in 'she'
        'i': 'i',      # 'ee' as in 'see'
        's': 's',      # 's' as in 'see'
        'ɛ': 'ɛ',      # 'e' as in 'sell'
        'l': 'l',      # 'l' as in 'let'
        'z': 'z',      # 'z' as in 'zoo'
        'iː': 'iː',    # long 'ee'
        'ə': 'ə',      # schwa
        'eɪ': 'eɪ',    # 'ay' as in 'say'
        'ɪ': 'ɪ',      # 'i' as in 'sit'
        'æ': 'æ',      # 'a' as in 'cat'
        'ʌ': 'ʌ',      # 'u' as in 'but'
        'ɑ': 'ɑ',      # 'a' as in 'father'
        'ɔ': 'ɔ',      # 'o' as in 'thought'
        'ʊ': 'ʊ',      # 'u' as in 'put'
        'uː': 'uː',    # long 'oo'
        'aɪ': 'aɪ',    # 'i' as in 'eye'
        'aʊ': 'aʊ',    # 'ou' as in 'out'
        'ɔɪ': 'ɔɪ',    # 'oy' as in 'boy'
        'p': 'p',      # 'p' as in 'pat'
        'b': 'b',      # 'b' as in 'bat'
        't': 't',      # 't' as in 'top'
        'd': 'd',      # 'd' as in 'dog'
        'k': 'k',      # 'k' as in 'cat'
        'g': 'g',      # 'g' as in 'go'
        'f': 'f',      # 'f' as in 'fat'
        'v': 'v',      # 'v' as in 'van'
        'θ': 'θ',      # 'th' as in 'think'
        'ð': 'ð',      # 'th' as in 'this'
        'h': 'h',      # 'h' as in 'hat'
        'm': 'm',      # 'm' as in 'mat'
        'n': 'n',      # 'n' as in 'not'
        'ŋ': 'ŋ',      # 'ng' as in 'sing'
        'r': 'r',      # 'r' as in 'run'
        'j': 'j',      # 'y' as in 'yes'
        'w': 'w',      # 'w' as in 'wet'
    }
    
    print("Common phoneme mappings:")
    for espeak, ipa in list(espeak_to_ipa.items())[:10]:
        print(f"  eSpeak '{espeak}' → IPA /{ipa}/")
    
    return espeak_to_ipa

def decode_sample_with_vocab():
    """Decode our sample with the actual vocabulary"""
    # Load vocab if it exists
    if os.path.exists('phoneme_vocab.json'):
        with open('phoneme_vocab.json', 'r') as f:
            vocab = json.load(f)
        print("✓ Loaded existing phoneme_vocab.json")
    else:
        vocab = get_vocab_from_huggingface()
        if not vocab:
            print("Could not get vocabulary")
            return
    
    # Analyze
    id_to_phoneme = analyze_phoneme_vocab(vocab)
    
    # Map to IPA
    espeak_to_ipa = map_phonemes_to_ipa()
    
    # Decode our sample
    sample_ids = [38, 10, 5, 9, 8, 5, 30, 38, 9, 4, 5]
    
    print(f"\n=== FINAL DECODING ===")
    print(f"'SHE SELLS SEA SHELLS' phoneme analysis:")
    print(f"Phoneme IDs: {sample_ids}")
    
    decoded = []
    for pid in sample_ids:
        if pid in id_to_phoneme:
            phoneme = id_to_phoneme[pid]
            decoded.append(phoneme)
    
    print(f"eSpeak phonemes: {' '.join(decoded)}")
    
    # Convert to IPA if possible
    ipa_decoded = []
    for phoneme in decoded:
        ipa = espeak_to_ipa.get(phoneme, phoneme)
        ipa_decoded.append(ipa)
    
    print(f"IPA transcription: /{' '.join(ipa_decoded)}/")
    
    return vocab, id_to_phoneme

if __name__ == "__main__":
    decode_sample_with_vocab()