#!/usr/bin/env python3
"""
Test Checkpoint Inference Script
Loads a trained checkpoint and generates text samples
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
    """Find the latest PyTorch checkpoint file"""
    if not checkpoints_dir.exists():
        return None
    
    checkpoint_files = [f for f in checkpoints_dir.iterdir() if f.is_file() and f.name.startswith("checkpoint-") and f.name.endswith(".pt")]
    if not checkpoint_files:
        return None
    
    # Sort by checkpoint number
    latest = max(checkpoint_files, key=lambda x: int(x.name.split("-")[1].split(".")[0]))
    return latest

def load_model_and_tokenizer(checkpoint_path, tokenizer_dir):
    """Load PyTorch checkpoint and tokenizer"""
    print(f"🔤 Loading tokenizer from: {tokenizer_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    
    print(f"🤖 Loading PyTorch checkpoint from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract model state dict
    if 'model' in checkpoint:
        model_state = checkpoint['model']
        print(f"✅ Found model state in checkpoint")
        
        # Handle torch.compile prefixes (_orig_mod.) if present
        if any(key.startswith('_orig_mod.') for key in model_state.keys()):
            print("🔧 Detected torch.compile checkpoint, stripping _orig_mod. prefixes...")
            # Create new state dict without _orig_mod. prefixes
            clean_state = {}
            for key, value in model_state.items():
                if key.startswith('_orig_mod.'):
                    clean_key = key[10:]  # Remove '_orig_mod.' prefix
                    clean_state[clean_key] = value
                else:
                    clean_state[key] = value
            model_state = clean_state
            print(f"✅ Cleaned {len(clean_state)} parameters")
    else:
        print(f"❌ No 'model' key found in checkpoint")
        return None, None, None
    
    # Create model architecture (matching training config)
    from train_model import GPT, GPTConfig
    
    # Get model config from checkpoint or use defaults
    model_config = GPTConfig(
        n_layer=config.training_config.get("n_layer", 24),  # Regular model: 24 layers
        n_head=config.training_config.get("n_head", 16),    # Regular model: 16 heads
        n_embd=config.training_config.get("n_embd", 1024),  # Regular model: 1024 embeddings
        block_size=config.training_config.get("max_length", 1024),
        bias=False,
        vocab_size=config.training_config.get("vocab_size", 30000),
        dropout=0.1
    )
    
    # Create model
    model = GPT(model_config)
    model.load_state_dict(model_state)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    print(f"✅ PyTorch model loaded on: {device}")
    print(f"   Model parameters: {model.get_num_params():,}")
    return model, tokenizer, device

def generate_text(model, tokenizer, device, prompt, max_new_tokens=50, temperature=0.8):
    """Generate text from a prompt using the PyTorch model"""
    # Tokenize input
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    input_length = inputs.shape[1]
    
    # Generate text
    with torch.no_grad():
        # Use the model's forward method for generation
        generated = inputs.clone()
        
        for _ in range(max_new_tokens):
            # Get logits for next token
            logits, _ = model(generated)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if we hit EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode only the new tokens
    new_tokens = generated[0][input_length:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return generated_text

def test_checkpoint(checkpoint_path=None, tokenizer_dir=None, interactive=False):
    """Test a PyTorch checkpoint with various prompts"""
    
    # Use global config if not specified
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint(config.checkpoints_dir)
        if checkpoint_path is None:
            print("❌ No PyTorch checkpoints found")
            return False
    else:
        # Handle checkpoint name (e.g., "checkpoint-9500.pt")
        if not Path(checkpoint_path).exists():
            # Try as a checkpoint name in the checkpoints directory
            checkpoint_file = config.checkpoints_dir / checkpoint_path
            if checkpoint_file.exists():
                checkpoint_path = checkpoint_file
            else:
                print(f"❌ PyTorch checkpoint not found: {checkpoint_path}")
                return False
        else:
            checkpoint_path = Path(checkpoint_path)
    
    if tokenizer_dir is None:
        tokenizer_dir = config.london_tokenizer_dir
    else:
        tokenizer_dir = Path(tokenizer_dir)
    
    # Load model and tokenizer
    try:
        model, tokenizer, device = load_model_and_tokenizer(checkpoint_path, tokenizer_dir)
        if model is None:
            return False
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return False
    
    # Test prompts for London historical context
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
    
    print(f"\n🧪 Testing PyTorch Checkpoint: {checkpoint_path.name}")
    print("=" * 60)
    
    if interactive:
        print("💬 Interactive mode - enter your own prompts (type 'quit' to exit)")
        print("=" * 60)
        
        while True:
            try:
                user_prompt = input("\n📝 Enter prompt: ").strip()
                if user_prompt.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not user_prompt:
                    continue
                
                print("🤖 Generating...")
                generated = generate_text(model, tokenizer, device, user_prompt)
                print(f"📖 Generated: {generated}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
    else:
        print("🔬 Running test prompts...")
        print("=" * 60)
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n📝 Test {i}: {prompt}")
            print("🤖 Generating...")
            
            try:
                generated = generate_text(model, tokenizer, device, prompt)
                print(f"📖 Result: {generated}")
            except Exception as e:
                print(f"❌ Error: {e}")
            
            print("-" * 40)
    
    print(f"\n✅ PyTorch testing completed for {checkpoint_path.name}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Test London Historical LLM Checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="PyTorch checkpoint file (uses latest if not specified)")
    parser.add_argument("--tokenizer", type=str, default=None,
                       help="Tokenizer directory (uses global config if not specified)")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Interactive mode for custom prompts")
    parser.add_argument("--list-checkpoints", action="store_true",
                       help="List available PyTorch checkpoints")
    
    args = parser.parse_args()
    
    if args.list_checkpoints:
        checkpoints_dir = Path(args.checkpoint) if args.checkpoint else config.checkpoints_dir
        print(f"🔍 Available PyTorch checkpoints in: {checkpoints_dir}")
        
        if checkpoints_dir.exists():
            checkpoints = [f for f in checkpoints_dir.iterdir() if f.is_file() and f.name.startswith("checkpoint-") and f.name.endswith(".pt")]
            if checkpoints:
                for cp in sorted(checkpoints, key=lambda x: int(x.name.split("-")[1].split(".")[0])):
                    print(f"   📄 {cp.name}")
            else:
                print("   ❌ No PyTorch checkpoints found")
        else:
            print("   ❌ Checkpoints directory does not exist")
        return
    
    # Run testing
    success = test_checkpoint(
        checkpoint_path=args.checkpoint,
        tokenizer_dir=args.tokenizer,
        interactive=args.interactive
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()



