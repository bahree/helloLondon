#!/usr/bin/env python3
"""
Comprehensive test script for the simple SLM model
Tests different parameters and prompts to evaluate quality
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config import config
from transformers import AutoTokenizer
# Import the model classes from the training script
sys.path.append(str(Path(__file__).parent.parent / "04_training"))
from train_model_slm import SimpleGPT, SimpleGPTConfig

def load_model_and_tokenizer(checkpoint_dir, tokenizer_dir):
    """Load model and tokenizer"""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    
    # Load latest checkpoint
    checkpoint_files = list(Path(checkpoint_dir).glob("checkpoint-*.pt"))
    checkpoint_files.sort(key=lambda x: int(x.stem.split('-')[1]))
    latest_checkpoint = checkpoint_files[-1]
    
    checkpoint = torch.load(latest_checkpoint, map_location='cpu')
    
    # Create model with correct architecture
    model_config = SimpleGPTConfig(
        block_size=256,
        vocab_size=tokenizer.vocab_size,
        n_layer=8,
        n_head=8,
        n_embd=512,
        dropout=0.1,
        bias=False
    )
    
    model = SimpleGPT(model_config)
    
    # Handle compiled model keys
    state_dict = checkpoint['model']
    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('_orig_mod.'):
                new_key = key[10:]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8, top_k=50):
    """Generate text from the model"""
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    generated_ids = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            block_size = model.config.block_size
            if generated_ids.size(1) > block_size:
                input_block = generated_ids[:, -block_size:]
            else:
                input_block = generated_ids
            
            logits, _ = model(input_block)
            logits = logits[:, -1, :] / temperature
            
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, top_k)
                logits = torch.full_like(logits, float('-inf'))
                logits.scatter_(1, top_k_indices, top_k_logits)
            
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):].strip()
    
    return generated_text

def test_different_temperatures(model, tokenizer, prompt):
    """Test with different temperature settings"""
    print(f"\nTesting different temperatures for: '{prompt}'")
    print("=" * 80)
    
    temperatures = [0.3, 0.5, 0.7, 0.9, 1.1]
    for temp in temperatures:
        print(f"\nTemperature {temp}:")
        generated = generate_text(model, tokenizer, prompt, max_new_tokens=80, temperature=temp, top_k=50)
        print(f"  {generated[:200]}{'...' if len(generated) > 200 else ''}")

def test_different_prompts(model, tokenizer):
    """Test with different types of prompts"""
    print(f"\nTesting different prompt types")
    print("=" * 80)
    
    prompts = [
        # Historical narrative
        "In the year 1800, a young woman named Elizabeth",
        "The Great Fire of London began when",
        "During the reign of Queen Victoria,",
        
        # Dialogue
        "The magistrate looked at the prisoner and said,",
        "A gentleman approached the young lady and asked,",
        
        # Descriptive
        "The streets of London were bustling with",
        "The old church stood majestically",
        
        # Legal/court
        "The judge declared that the prisoner",
        "In the court of the Old Bailey,",
        
        # Simple continuation
        "Once upon a time in London,",
        "The merchant walked through the market"
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Prompt {i}: {prompt} ---")
        generated = generate_text(model, tokenizer, prompt, max_new_tokens=100, temperature=0.7, top_k=50)
        print(f"Generated: {generated[:300]}{'...' if len(generated) > 300 else ''}")

def test_longer_generation(model, tokenizer):
    """Test longer text generation"""
    print(f"\n📖 Testing longer generation")
    print("=" * 80)
    
    prompt = "In the year 1834, London was a city of great contrasts."
    print(f"Prompt: {prompt}")
    
    # Test with different lengths
    lengths = [50, 100, 200]
    for length in lengths:
        print(f"\n--- Length {length} tokens ---")
        generated = generate_text(model, tokenizer, prompt, max_new_tokens=length, temperature=0.7, top_k=50)
        print(f"Generated: {generated}")

def main():
    parser = argparse.ArgumentParser(description='Comprehensive SLM Model Test')
    parser.add_argument("--checkpoint_dir", type=str, default="09_models/checkpoints/slm")
    parser.add_argument("--tokenizer_dir", type=str, default="09_models/tokenizers/london_historical_tokenizer")
    
    args = parser.parse_args()
    
    print("🧪 Comprehensive SLM Model Testing")
    print("=" * 80)
    
    try:
        # Load model and tokenizer
        print("Loading model and tokenizer...")
        model, tokenizer = load_model_and_tokenizer(args.checkpoint_dir, args.tokenizer_dir)
        print(f"✅ Model loaded ({model.get_num_params():,} parameters)")
        print(f"✅ Tokenizer loaded (vocab size: {tokenizer.vocab_size:,})")
        
        # Test different temperatures
        test_different_temperatures(model, tokenizer, "In the year 1834, London was")
        
        # Test different prompts
        test_different_prompts(model, tokenizer)
        
        # Test longer generation
        test_longer_generation(model, tokenizer)
        
        print(f"\n✅ Comprehensive testing completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
