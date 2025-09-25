#!/usr/bin/env python3
"""
Token Counter for London Historical Data
========================================
Counts tokens in cleaned data files to provide dataset size metrics
"""

import os
import json
from pathlib import Path
from transformers import AutoTokenizer
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
import sys
sys.path.append(str(project_root))

# Import global configuration
from config import config

def count_tokens_in_file(file_path, tokenizer):
    """Count tokens in a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Tokenize the text
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0

def count_tokens_in_directory(data_dir, tokenizer):
    """Count tokens in all text files in a directory"""
    total_tokens = 0
    total_files = 0
    file_stats = []
    
    # Find all text files
    text_files = list(Path(data_dir).glob("**/*.txt"))
    
    if not text_files:
        print(f"No text files found in {data_dir}")
        return 0, 0, []
    
    print(f"Found {len(text_files)} text files")
    print("Processing files...")
    
    for i, file_path in enumerate(text_files, 1):
        if i % 10 == 0:
            print(f"   Processed {i}/{len(text_files)} files...")
        
        tokens = count_tokens_in_file(file_path, tokenizer)
        if tokens > 0:
            total_tokens += tokens
            total_files += 1
            file_stats.append({
                'file': str(file_path),
                'tokens': tokens,
                'size_mb': file_path.stat().st_size / (1024 * 1024)
            })
    
    return total_tokens, total_files, file_stats

def main():
    parser = argparse.ArgumentParser(description="Count tokens in cleaned data")
    parser.add_argument("--data_dir", type=str, default=None,
                       help="Directory containing cleaned text files (defaults to config)")
    parser.add_argument("--tokenizer_dir", type=str, default=None,
                       help="Path to tokenizer directory (defaults to config)")
    parser.add_argument("--output_file", type=str, default="token_count_report.json",
                       help="Output file for detailed report")
    
    args = parser.parse_args()
    
    # Use config defaults if not provided
    data_dir = args.data_dir or str(config.london_historical_data / "cleaned")
    tokenizer_dir = args.tokenizer_dir or str(config.london_tokenizer_dir)
    
    print("London Historical Data - Token Counter")
    print("=" * 50)
    print(f"Data directory: {data_dir}")
    print(f"Tokenizer: {tokenizer_dir}")
    print()
    
    # Load tokenizer
    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        print(f"Tokenizer loaded (vocab size: {tokenizer.vocab_size:,})")
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        return 1
    
    # Count tokens
    print("\nCounting tokens...")
    total_tokens, total_files, file_stats = count_tokens_in_directory(data_dir, tokenizer)
    
    if total_tokens == 0:
        print("No tokens found!")
        return 1
    
    # Calculate statistics
    avg_tokens_per_file = total_tokens / total_files if total_files > 0 else 0
    total_size_mb = sum(stat['size_mb'] for stat in file_stats)
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 TOKEN COUNT SUMMARY")
    print("=" * 50)
    print(f"📁 Total files processed: {total_files:,}")
    print(f"🔢 Total tokens: {total_tokens:,}")
    print(f"📏 Average tokens per file: {avg_tokens_per_file:,.0f}")
    print(f"💾 Total data size: {total_size_mb:.2f} MB")
    print(f"📈 Tokens per MB: {total_tokens / total_size_mb:,.0f}")
    
    # Estimate training data
    train_tokens = int(total_tokens * 0.9)  # 90% for training
    eval_tokens = int(total_tokens * 0.1)   # 10% for evaluation
    
    print(f"\n🎓 Training Data Estimates:")
    print(f"   Training tokens: {train_tokens:,}")
    print(f"   Evaluation tokens: {eval_tokens:,}")
    
    # Save detailed report
    report = {
        'summary': {
            'total_files': total_files,
            'total_tokens': total_tokens,
            'avg_tokens_per_file': avg_tokens_per_file,
            'total_size_mb': total_size_mb,
            'tokens_per_mb': total_tokens / total_size_mb,
            'train_tokens': train_tokens,
            'eval_tokens': eval_tokens
        },
        'file_details': file_stats
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Detailed report saved to: {args.output_file}")
    print("\n✅ Token counting completed!")
    
    return 0

if __name__ == "__main__":
    exit(main())
