#!/usr/bin/env python3
"""
Improved Checkpoint Testing Script
Better generation parameters for testing SLM checkpoints
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config import config

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import warnings
    warnings.filterwarnings("ignore")
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("Please install: pip install torch transformers")
    sys.exit(1)

def find_latest_checkpoint(checkpoints_dir):
    """Find the latest checkpoint directory"""
    if not checkpoints_dir.exists():
        return None
    
    checkpoint_dirs = [d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
    if not checkpoint_dirs:
        return None
    
    # Sort by checkpoint number
    latest = max(checkpoint_dirs, key=lambda x: int(x.name.split("-")[1]))
    return latest

def load_model_and_tokenizer(checkpoint_dir, tokenizer_dir):
    """Load model and tokenizer from checkpoint"""
    print(f"🔤 Loading tokenizer from: {tokenizer_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    
    print(f"🤖 Loading model from: {checkpoint_dir}")
    try:
        # Try loading from local checkpoint first
        model = AutoModelForCausalLM.from_pretrained(str(checkpoint_dir), local_files_only=True)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None, None
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded on: {device}")
    return model, tokenizer, device

def generate_text_improved(model, tokenizer, device, prompt, max_new_tokens=50, temperature=0.8, top_p=0.9, repetition_penalty=1.1):
    """Generate text with improved parameters for SLM"""
    # Tokenize input
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    input_length = inputs.shape[1]
    
    # Generate text with better parameters for SLM
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,  # Prevent 3-gram repetition
            early_stopping=True,
            use_cache=True
        )
    
    # Decode output (only the new tokens)
    generated_tokens = outputs[0][input_length:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return generated_text

def test_checkpoint_improved(checkpoint_dir=None, tokenizer_dir=None):
    """Test a checkpoint with improved generation"""
    
    # Use global config if not specified
    if checkpoint_dir is None:
        checkpoint_dir = find_latest_checkpoint(config.checkpoints_dir / "slm")
        if checkpoint_dir is None:
            print("❌ No checkpoints found")
            return False
    
    if tokenizer_dir is None:
        tokenizer_dir = config.london_tokenizer_dir
    
    print(f"🧪 Testing Checkpoint: {checkpoint_dir.name}")
    print("=" * 60)
    
    # Load model and tokenizer
    model, tokenizer, device = load_model_and_tokenizer(checkpoint_dir, tokenizer_dir)
    if model is None:
        return False
    
    # Test prompts
    test_prompts = [
        "In the year of our Lord 1750, London was",
        "The Thames flowed through the heart of",
        "Merchants and tradesmen plied their wares",
        "The Great Fire of 1666 had changed",
        "Parliament sat in Westminster, making laws",
        "In the bustling streets of Cheapside,",
        "The Tower of London stood as a symbol",
        "Samuel Pepys wrote in his diary that",
        "The plague had ravaged the city, leaving",
        "Covent Garden was filled with the sounds of"
    ]
    
    print("🔬 Running improved test prompts...")
    print("=" * 60)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n📝 Test {i}: {prompt}")
        print("🤖 Generating...")
        
        try:
            # Try different generation strategies
            generated = generate_text_improved(
                model, tokenizer, device, prompt,
                max_new_tokens=30,  # Shorter for better coherence
                temperature=0.7,    # Lower temperature for more focused output
                top_p=0.9,         # Nucleus sampling
                repetition_penalty=1.2  # Higher penalty to reduce repetition
            )
            print(f"📖 Result: {generated}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 40)
    
    print(f"\n✅ Testing completed for {checkpoint_dir.name}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Test London Historical SLM Checkpoint (Improved)")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to checkpoint directory")
    parser.add_argument("--tokenizer", type=str, default=None,
                       help="Path to tokenizer directory")
    
    args = parser.parse_args()
    
    # Convert to Path objects
    checkpoint_dir = Path(args.checkpoint) if args.checkpoint else None
    tokenizer_dir = Path(args.tokenizer) if args.tokenizer else None
    
    # Run test
    success = test_checkpoint_improved(checkpoint_dir, tokenizer_dir)
    
    if success:
        print("\n🎉 Checkpoint testing completed successfully!")
    else:
        print("\n❌ Checkpoint testing failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
