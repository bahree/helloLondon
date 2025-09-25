#!/usr/bin/env python3
"""
Sample script for using London Historical SLM from Hugging Face Hub
This script demonstrates how to use the model after it's published
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model():
    """Load the model from Hugging Face Hub"""
    print("🏛️ Loading London Historical SLM from Hugging Face Hub...")
    
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained("bahree/london-historical-slm")
        model = AutoModelForCausalLM.from_pretrained("bahree/london-historical-slm")
        
        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Get device info
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        print("✅ Model loaded successfully!")
        print(f"   Vocabulary size: {tokenizer.vocab_size:,}")
        print(f"   Model parameters: ~117M")
        print(f"   Max length: {tokenizer.model_max_length}")
        print(f"   Device: {device}")
        print(f"   Model type: GPT-2 Small (trained from scratch)")
        
        return tokenizer, model, device
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("💡 Make sure you have an internet connection and the model is published")
        return None, None, None

def generate_historical_text(tokenizer, model, device, prompt, max_length=50, **kwargs):
    """Generate historical text with optimized parameters"""
    
    # Default parameters optimized for historical text (matching the model card)
    default_params = {
        "do_sample": True,
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 40,
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 3,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "early_stopping": True
    }
    
    # Update with any custom parameters
    default_params.update(kwargs)
    default_params["max_new_tokens"] = max_length
    
    # Tokenize input and move to device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            **default_params
        )
    
    # Decode and return
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def interactive_mode(tokenizer, model, device):
    """Interactive mode for testing the model"""
    
    print("\n🎭 Interactive Mode - Type 'quit' to exit")
    print("=" * 50)
    
    while True:
        try:
            prompt = input("\n📝 Enter your prompt: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not prompt:
                print("⚠️  Please enter a prompt")
                continue
            
            print("🤖 Generating...")
            result = generate_historical_text(tokenizer, model, device, prompt)
            print(f"📖 Generated: {result}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def demo_prompts(tokenizer, model, device):
    """Run demo prompts to showcase the model"""
    
    print("\n🎯 Demo Prompts - London Historical SLM")
    print("=" * 60)
    
    demo_prompts = [
        "In the year 1834, I walked through the streets of London and witnessed",
        "The gentleman from the country said, 'I have never seen such a sight",
        "The Thames flowed dark and mysterious through the heart",
        "Merchants plied their wares in the bustling market",
        "The Great Fire of 1666 had destroyed",
        "Parliament sat in Westminster Hall",
        "The Tower of London stood",
        "Samuel Pepys wrote that",
        "The plague had ravaged",
        "Covent Garden was filled with"
    ]
    
    for i, prompt in enumerate(demo_prompts, 1):
        print(f"\n--- Demo {i} ---")
        print(f"Prompt: {prompt}")
        
        result = generate_historical_text(tokenizer, model, device, prompt, max_length=40)
        print(f"Generated: {result}")
        print("-" * 60)

def test_with_actual_output():
    """Test with the actual output format from test_published_models.py"""
    print("\n🧪 Testing with Actual Output Format")
    print("=" * 60)
    
    # This simulates the actual testing output format
    print("🧪 Testing SLM Model: bahree/london-historical-slm")
    print("=" * 60)
    print("📂 Loading model...")
    print("✅ Model loaded in 8.91 seconds")
    print("📊 Model Info:")
    print("   Type: SLM")
    print("   Description: Small Language Model (117M parameters)")
    print("   Device: cuda")
    print("   Vocabulary size: 30,000")
    print("   Max length: 512")
    print()
    print("🎯 Testing generation with 10 prompts...")
    print("[10 automated tests with historical text generation]")

def main():
    """Main function"""
    print("🏛️ London Historical SLM - Sample Usage Script")
    print("=" * 60)
    print("📚 This script demonstrates how to use the model from Hugging Face Hub")
    print("🔗 Model: https://huggingface.co/bahree/london-historical-slm")
    print("🏗️  Architecture: GPT-2 Small (117M params, trained from scratch)")
    print("📅 Period: Historical London texts (1500-1850)")
    print("=" * 60)
    
    # Load model
    tokenizer, model, device = load_model()
    
    if tokenizer is None or model is None:
        print("❌ Failed to load model. Please check your internet connection and try again.")
        return
    
    # Run demo
    demo_prompts(tokenizer, model, device)
    
    # Show actual output format
    test_with_actual_output()
    
    # Ask for interactive mode
    print("\n🎭 Would you like to try interactive mode? (y/n): ", end="")
    try:
        choice = input().strip().lower()
        if choice in ['y', 'yes']:
            interactive_mode(tokenizer, model, device)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    
    print("\n✅ Demo completed!")
    print("\n📚 Usage in your own code:")
    print("""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model
tokenizer = AutoTokenizer.from_pretrained("bahree/london-historical-slm")
model = AutoModelForCausalLM.from_pretrained("bahree/london-historical-slm")

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Generate text with optimized parameters
prompt = "Your prompt here"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(
    inputs['input_ids'],
    max_new_tokens=50,
    do_sample=True,
    temperature=0.8,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    early_stopping=True
)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
""")

if __name__ == "__main__":
    main()
