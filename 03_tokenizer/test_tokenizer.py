#!/usr/bin/env python3
"""
Test script for the historical tokenizer
"""

import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config import config

def test_tokenizer():
    """Test the trained tokenizer"""
    print("🧪 Testing Historical Tokenizer")
    print("=" * 50)
    
    # Load tokenizer
    tokenizer_path = config.london_tokenizer_dir
    tokenizer_file = tokenizer_path / "tokenizer.json"
    if not tokenizer_file.exists():
        print(f"❌ Tokenizer not found at {tokenizer_file}")
        return
    
    try:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(tokenizer_file))
        print(f"✅ Tokenizer loaded from {tokenizer_path}")
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        return
    
    # Test sentences
    test_sentences = [
        "The sun was setting over the Thames as Elizabeth walked through the bustling market.",
        "In the year of our Lord 1665, the plague swept through London with great fury.",
        "The court found the defendant guilty of theft and sentenced him to transportation.",
        "She was a woman of great beauty and intelligence, known throughout the parish.",
        "The merchant's warehouse contained goods from the far reaches of the known world.",
        "Parliament debated the new tax measures with great passion and eloquence.",
        "The church bells rang out across the city, calling the faithful to prayer.",
        "London's streets were filled with the sounds of horses, carts, and street vendors.",
        "The young apprentice learned his trade with diligence and determination.",
        "In those days, the river was the lifeblood of the great city."
    ]
    
    print(f"\n📝 Testing {len(test_sentences)} sample sentences:")
    print("-" * 50)
    
    for i, sentence in enumerate(test_sentences, 1):
        # Tokenize
        tokens = tokenizer.encode(sentence)
        token_ids = tokens.ids
        token_texts = tokens.tokens
        
        # Decode back
        decoded = tokenizer.decode(token_ids)
        
        print(f"\n{i}. Original: {sentence}")
        print(f"   Tokens: {len(token_ids)} tokens")
        print(f"   Token IDs: {token_ids[:10]}{'...' if len(token_ids) > 10 else ''}")
        print(f"   Token Texts: {token_texts[:10]}{'...' if len(token_texts) > 10 else ''}")
        print(f"   Decoded: {decoded}")
        # Check if perfect reconstruction (normalize whitespace for comparison)
        import re
        normalized_original = re.sub(r'\s+', ' ', sentence.strip())
        normalized_decoded = re.sub(r'\s+', ' ', decoded.strip())
        
        if normalized_original == normalized_decoded:
            print(f"   Match: ✅")
        else:
            print(f"   Match: ❌ (spacing differences)")
            print(f"   Original: '{normalized_original}'")
            print(f"   Decoded:  '{normalized_decoded}'")
    
    # Test special tokens
    print(f"\n🔤 Testing Special Tokens:")
    print("-" * 30)
    
    special_tokens = ["<|startoftext|>", "<|endoftext|>", "<|pad|>", "<|unk|>", "<|mask|>"]
    for token in special_tokens:
        try:
            # For special tokens, we need to check if they're in the vocabulary
            vocab = tokenizer.get_vocab()
            if token in vocab:
                token_id = vocab[token]
                decoded = tokenizer.decode([token_id])
                print(f"   {token}: ID {token_id} -> '{decoded}' {'✅' if token in decoded else '❌'}")
            else:
                print(f"   {token}: Not in vocabulary")
        except Exception as e:
            print(f"   {token}: Error - {e}")
    
    # Test vocabulary size
    vocab_size = tokenizer.get_vocab_size()
    print(f"\n📊 Vocabulary Statistics:")
    print(f"   Vocabulary size: {vocab_size:,}")
    
    # Test with a longer text
    print(f"\n📖 Testing with longer text:")
    print("-" * 30)
    
    long_text = """
    In the year of our Lord 1665, the plague swept through London with great fury. 
    The streets were empty, the shops closed, and the people fled to the countryside. 
    But some remained, those who could not leave, and they faced the terrible disease 
    with courage and determination. The bells of St. Paul's rang out across the empty 
    city, calling the faithful to prayer, though few dared to answer the call.
    """
    
    tokens = tokenizer.encode(long_text)
    print(f"   Text length: {len(long_text)} characters")
    print(f"   Token count: {len(tokens.ids)} tokens")
    print(f"   Compression ratio: {len(tokens.ids) / len(long_text):.2f} tokens/char")
    
    print(f"\n✅ Tokenizer test completed!")

if __name__ == "__main__":
    test_tokenizer()
