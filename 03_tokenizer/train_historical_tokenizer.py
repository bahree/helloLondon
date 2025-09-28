#!/usr/bin/env python3
"""
Train a custom tokenizer optimized for historical English (1500-1850)
Replaces the basic tokenizer with a historical-specific one
"""

import os
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config import config

try:
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors
    from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence
    from transformers import PreTrainedTokenizerFast
    import tqdm
except ImportError as e:
    print(f"Missing required dependencies: {e}")
    print("Please install: pip install tokenizers transformers tqdm")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HistoricalTokenizerTrainer:
    def __init__(self, corpus_path=None, output_dir=None, vocab_size=30000):
        # Use global config with fallbacks - check for comprehensive corpus first
        if corpus_path:
            self.corpus_path = Path(corpus_path)
        else:
            # Check for comprehensive corpus first, then fall back to standard
            comprehensive_corpus = config.london_historical_data / "london_historical_corpus_comprehensive.txt"
            if comprehensive_corpus.exists():
                self.corpus_path = comprehensive_corpus
            else:
                self.corpus_path = config.corpus_file
        self.output_dir = Path(output_dir) if output_dir else config.london_tokenizer_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.vocab_size = vocab_size
        
        # Historical-specific special tokens
        self.special_tokens = [
            # Basic control tokens
            "<|endoftext|>",
            "<|startoftext|>", 
            "<|pad|>",
            "<|unk|>",
            "<|mask|>",
            
            # Essential historical tokens (high frequency)
            "<|thou|>", "<|thee|>", "<|thy|>", "<|thine|>",
            "<|hast|>", "<|hath|>", "<|doth|>", "<|dost|>",
            "<|quoth|>", "<|verily|>", "<|indeed|>", "<|forsooth|>",
            
            # London landmarks (from your corpus)
            "<|london|>", "<|thames|>", "<|westminster|>", "<|tower|>",
            "<|newgate|>", "<|southwark|>", "<|cheapside|>",
            
            # Historical periods
            "<|tudor|>", "<|stuart|>", "<|georgian|>", "<|regency|>",
            
            # Common archaic words
            "<|perchance|>", "<|anon|>", "<|ere|>", "<|whilst|>",
            "<|betwixt|>", "<|amongst|>", "<|prithee|>", "<|pray|>",
            
            # Additional useful tokens
            "<|year|>", "<|date|>", "<|name|>", "<|place|>",
            "<|chapter|>", "<|verse|>", "<|quote|>", "<|speech|>",
        ]
        
    def train_tokenizer(self):
        """Train a custom tokenizer for historical English"""
        logger.info("Training custom historical tokenizer...")
        logger.info(f"Corpus: {self.corpus_path}")
        logger.info(f"Target vocabulary: {self.vocab_size:,} tokens")
        logger.info(f"Output directory: {self.output_dir}")
        
        # Check if corpus exists
        if not self.corpus_path.exists():
            logger.error(f"Corpus not found: {self.corpus_path}")
            return None
        
        # Initialize tokenizer
        tokenizer = Tokenizer(models.BPE())
        
        # Normalizers for historical text - preserve case for better text reconstruction
        tokenizer.normalizer = Sequence([
            NFD(),           # Unicode normalization
            StripAccents()   # Remove accents
        ])
        
        # Pre-tokenizer for historical English - use simple whitespace splitting
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.WhitespaceSplit(),
            pre_tokenizers.Punctuation()
        ])
        
        # Configure trainer with historical English focus - optimized for segmented data
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens,
            min_frequency=2,     # Minimum frequency for tokens
            show_progress=True,
            # Remove the WordPiece-style prefix that creates ## artifacts
            # continuing_subword_prefix="##",  # ❌ Removed to eliminate ## artifacts
            initial_alphabet=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", 
                            "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
                            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
                            "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
        )
        
        # Train on corpus - now with properly segmented data
        logger.info("Training tokenizer on segmented data...")
        logger.info(f"Corpus size: {self.corpus_path.stat().st_size / (1024*1024):.1f} MB")
        
        # Count lines to show progress
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            line_count = sum(1 for _ in f)
        logger.info(f"Segments to process: {line_count:,}")
        
        tokenizer.train([str(self.corpus_path)], trainer)
        
        # Add post-processor - use template processing for better special token handling
        tokenizer.post_processor = processors.TemplateProcessing(
            single="<|startoftext|> $A <|endoftext|>",
            special_tokens=[
                ("<|startoftext|>", 1),
                ("<|endoftext|>", 0),
            ]
        )
        
        # Add normalizer to fix spacing around punctuation
        tokenizer.normalizer = Sequence([
            NFD(),           # Unicode normalization
            StripAccents(),  # Remove accents
            # Add custom normalizer to fix spacing
        ])
        
        # Save tokenizer
        tokenizer.save(str(self.output_dir / "tokenizer.json"))
        
        # Create Hugging Face tokenizer - use SLM config for max_length
        max_length = config.slm_config["max_length"]
        logger.info(f"Setting tokenizer max_length to: {max_length}")
        
        hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token="<|startoftext|>",
            eos_token="<|endoftext|>",
            pad_token="<|pad|>",
            unk_token="<|unk|>",
            mask_token="<|mask|>",
            model_max_length=max_length
        )
        
        # Ensure max_length is set correctly
        hf_tokenizer.model_max_length = max_length
        logger.info(f"Tokenizer max_length set to: {hf_tokenizer.model_max_length}")
        
        # Save Hugging Face tokenizer
        hf_tokenizer.save_pretrained(str(self.output_dir))
        
        # Create proper config.json for Hugging Face compatibility
        hf_config = {
            "vocab_size": len(tokenizer.get_vocab()),
            "model_max_length": max_length,
            "tokenizer_class": "PreTrainedTokenizerFast",
            "tokenizer_type": "BPE",
            "special_tokens": {
                "bos_token": "<|startoftext|>",
                "eos_token": "<|endoftext|>",
                "pad_token": "<|pad|>",
                "unk_token": "<|unk|>",
                "mask_token": "<|mask|>"
            }
        }
        
        with open(self.output_dir / "config.json", 'w') as f:
            json.dump(hf_config, f, indent=2)
        
        # Save vocab and merges separately
        vocab = tokenizer.get_vocab()
        with open(self.output_dir / "vocab.json", 'w') as f:
            json.dump(vocab, f, indent=2)
        
        logger.info(f"Custom historical tokenizer saved to {self.output_dir}")
        logger.info(f"Vocabulary size: {len(vocab):,} tokens")
        logger.info(f"Special tokens: {len(self.special_tokens)}")
        logger.info(f"Max length: {max_length} tokens")
        logger.info(f"Trained on {line_count:,} text segments")
        
        return tokenizer
    
    def test_tokenizer(self, tokenizer):
        """Test the trained tokenizer"""
        logger.info("🧪 Testing custom historical tokenizer...")
        
        test_texts = [
            "In the year of our Lord 1834, the streets of London were filled with the sounds of horse-drawn carriages.",
            "The gentleman from the country said, 'I have never seen such a sight in all my days.'",
            "Chapter I: The Beginning of the End",
            "Mr. Darcy walked through the ballroom with his usual air of superiority.",
            "The Thames flowed dark and mysterious through the heart of the city.",
            "It was the best of times, it was the worst of times.",
            "The year was 1812, and war had come to England once more.",
            "Lady Catherine de Bourgh was not pleased with the news.",
            "The old man sat by the fire, reading his Bible.",
            "The coach rattled down the cobblestone streets of London."
        ]
        
        for i, text in enumerate(test_texts, 1):
            logger.info(f"\n--- Test {i} ---")
            logger.info(f"Text: {text}")
            
            # Encode
            tokens = tokenizer.encode(text)
            logger.info(f"Tokens: {tokens.ids[:20]}...")  # Show first 20 tokens
            logger.info(f"Token count: {len(tokens.ids)}")
            
            # Decode
            decoded = tokenizer.decode(tokens.ids)
            logger.info(f"Decoded: {decoded}")
            
            # Check if perfect reconstruction
            if text.lower() == decoded.lower():
                logger.info("Perfect reconstruction")
            else:
                logger.info("Reconstruction differs")

def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train custom historical tokenizer")
    parser.add_argument("--corpus_path", type=str, default=None,
                       help="Path to corpus file (uses global config if not specified)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (uses global config if not specified)")
    parser.add_argument("--vocab_size", type=int, default=30000,
                       help="Vocabulary size (default: 30000)")
    
    args = parser.parse_args()
    
    print("Historical English Tokenizer Training")
    print("=" * 50)
    print("Optimized for historical English (1500-1850)")
    print("Special tokens for dates, names, places, dialogue")
    print("BPE tokenization with historical focus")
    print("=" * 50)
    
    # Initialize trainer
    trainer = HistoricalTokenizerTrainer(
        corpus_path=args.corpus_path,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size
    )
    
    # Train tokenizer
    tokenizer = trainer.train_tokenizer()
    
    if tokenizer:
        # Test tokenizer
        trainer.test_tokenizer(tokenizer)
        
        print("\nCustom historical tokenizer training complete!")
        print("This tokenizer is optimized for historical English text")
        print("Better handling of dates, names, places, dialogue")
        print("More efficient vocabulary (30k vs 50k tokens)")
        print("Ready for training your London Historical LLM!")
    else:
        print("\nTokenizer training failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
