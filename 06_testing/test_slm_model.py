#!/usr/bin/env python3
"""
Test script for the simple SLM model
Loads the latest checkpoint and generates text
"""

import os
import sys
import torch
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config import config
from transformers import AutoTokenizer

# Import the model classes from the training script
sys.path.append(str(Path(__file__).parent.parent / "04_training"))
from train_model_slm import SimpleGPT, SimpleGPTConfig

def load_latest_checkpoint(checkpoint_dir):
    """Load the latest checkpoint from the directory"""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    # Find all checkpoint files
    checkpoint_files = list(checkpoint_dir.glob("checkpoint-*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
    
    # Sort by iteration number and get the latest
    checkpoint_files.sort(key=lambda x: int(x.stem.split('-')[1]))
    latest_checkpoint = checkpoint_files[-1]
    
    print(f"Loading checkpoint: {latest_checkpoint}")
    checkpoint = torch.load(latest_checkpoint, map_location='cpu')
    
    return checkpoint, latest_checkpoint

def load_model_from_checkpoint(checkpoint, tokenizer):
    """Load the model from checkpoint"""
    # Get model config from checkpoint or use defaults
    # The training used block_size=256 (half of max_length=512)
    model_config = SimpleGPTConfig(
        block_size=256,  # This matches what was used in training
        vocab_size=tokenizer.vocab_size,
        n_layer=8,
        n_head=8,
        n_embd=512,
        dropout=0.1,
        bias=False
    )
    
    # Create model
    model = SimpleGPT(model_config)
    
    # Load state dict - handle compiled model keys
    state_dict = checkpoint['model']
    
    # Check if the state dict has compiled model keys (with _orig_mod. prefix)
    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        print("Detected compiled model checkpoint, removing _orig_mod. prefix...")
        # Remove _orig_mod. prefix from all keys
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('_orig_mod.'):
                new_key = key[10:]  # Remove '_orig_mod.' prefix
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict
    
    # Load the cleaned state dict
    model.load_state_dict(state_dict)
    model.eval()
    
    return model

def generate_text(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8, top_k=50):
    """Generate text from the model"""
    print(f"Prompt: '{prompt}'")
    print("=" * 50)
    
    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    # Generate text token by token
    generated_ids = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get the last block_size tokens (or all if shorter)
            block_size = model.config.block_size
            if generated_ids.size(1) > block_size:
                input_block = generated_ids[:, -block_size:]
            else:
                input_block = generated_ids
            
            # Get logits from the model
            logits, _ = model(input_block)
            
            # Get the logits for the last token
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, top_k)
                logits = torch.full_like(logits, float('-inf'))
                logits.scatter_(1, top_k_indices, top_k_logits)
            
            # Sample from the distribution
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append the new token
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Stop if we hit the end token
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode the generated text
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    # Remove the original prompt from the output
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):].strip()
    
    return generated_text

def main():
    parser = argparse.ArgumentParser(description='Test Simple SLM Model')
    parser.add_argument("--checkpoint", type=str, default="09_models/checkpoints/slm",
                       help="Checkpoint file or directory containing checkpoints")
    parser.add_argument("--tokenizer_dir", type=str, default="09_models/tokenizers/london_historical_tokenizer",
                       help="Directory containing tokenizer")
    parser.add_argument("--prompt", type=str, default="In the year 1834, London was",
                       help="Prompt to start generation")
    parser.add_argument("--max_tokens", type=int, default=100,
                       help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50,
                       help="Top-k sampling")
    
    args = parser.parse_args()
    
    print("🧪 Testing Simple SLM Model")
    print("=" * 50)
    
    try:
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
        print(f"✅ Tokenizer loaded (vocab size: {tokenizer.vocab_size:,})")
        
        # Load checkpoint (handle both file and directory)
        print("Loading checkpoint...")
        checkpoint_path = Path(args.checkpoint)
        
        if checkpoint_path.is_file():
            # Direct file path
            print(f"Loading checkpoint file: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        else:
            # Directory - find latest checkpoint
            checkpoint, checkpoint_path = load_latest_checkpoint(args.checkpoint)
        
        print(f"✅ Checkpoint loaded from {checkpoint_path}")
        print(f"   Iteration: {checkpoint.get('iter_num', 'unknown')}")
        print(f"   Best val loss: {checkpoint.get('best_val_loss', 'unknown'):.4f}")
        
        # Load model
        print("Loading model...")
        model = load_model_from_checkpoint(checkpoint, tokenizer)
        print(f"✅ Model loaded ({model.get_num_params():,} parameters)")
        
        # Test generation
        print("\n🎯 Testing text generation...")
        print("=" * 50)
        
        # Test with the provided prompt
        print(f"\n--- Custom Test ---")
        try:
            generated = generate_text(
                model, tokenizer, args.prompt, 
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k
            )
            print(f"Generated: {generated}")
        except Exception as e:
            print(f"❌ Generation failed: {e}")
        
        # Also test with a few default prompts for comparison
        default_prompts = [
            "In the year 1834, London was",
            "The streets of London were filled with",
            "A gentleman walked through"
        ]
        
        for i, prompt in enumerate(default_prompts, 1):
            print(f"\n--- Default Test {i} ---")
            try:
                generated = generate_text(
                    model, tokenizer, prompt, 
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k
                )
                print(f"Generated: {generated}")
            except Exception as e:
                print(f"❌ Generation failed: {e}")
        
        print("\n✅ Testing completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
