#!/usr/bin/env python3
"""
Filter to keep only English phonemes from model output
"""

import json

# Load English phoneme set
with open('english_phonemes.json', 'r') as f:
    eng_data = json.load(f)

# Combine all English phonemes
ENGLISH_PHONEMES = set()
for category in ['consonants', 'vowels_monophthongs', 'vowels_diphthongs']:
    ENGLISH_PHONEMES.update(eng_data[category])

# Also include variants with length markers
ENGLISH_PHONEMES_WITH_VARIANTS = ENGLISH_PHONEMES.copy()
for p in list(ENGLISH_PHONEMES):
    ENGLISH_PHONEMES_WITH_VARIANTS.add(p + 'ː')  # long version
    ENGLISH_PHONEMES_WITH_VARIANTS.add(p + '̩')   # syllabic

def is_english_phoneme(phoneme):
    """Check if a phoneme belongs to English"""
    # Remove stress markers to check base phoneme
    base = phoneme.replace('ˈ', '').replace('ˌ', '')
    
    # Check if it's an English phoneme
    return (base in ENGLISH_PHONEMES_WITH_VARIANTS or 
            phoneme in ['<pad>', '<s>', '</s>', '<unk>'])

def filter_english_only(phonemes_list):
    """Filter a list of phoneme dictionaries to keep only English ones"""
    return [p for p in phonemes_list if is_english_phoneme(p['phoneme'])]

def get_english_phoneme_stats():
    """Get statistics about English vs total phonemes"""
    with open('phoneme_vocab.json', 'r') as f:
        full_vocab = json.load(f)
    
    english_in_vocab = []
    non_english = []
    
    for phoneme, id in full_vocab.items():
        if is_english_phoneme(phoneme):
            english_in_vocab.append(phoneme)
        else:
            non_english.append(phoneme)
    
    return {
        'english_count': len(english_in_vocab),
        'non_english_count': len(non_english),
        'total': len(full_vocab),
        'english_phonemes': sorted(english_in_vocab),
        'sample_non_english': non_english[:20]
    }

if __name__ == '__main__':
    stats = get_english_phoneme_stats()
    print(f"English phonemes in vocab: {stats['english_count']}/{stats['total']}")
    print(f"\nEnglish phonemes found:")
    for p in stats['english_phonemes'][:50]:  # First 50
        print(f"  {p}")
    print(f"\nSample non-English phonemes:")
    for p in stats['sample_non_english']:
        print(f"  {p}")