#!/usr/bin/env python3
"""
Setup script for testing London Historical SLM on any machine
This script installs dependencies and downloads the model
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def install_dependencies():
    """Install required dependencies"""
    dependencies = [
        "torch",
        "transformers",
        "safetensors",
        "accelerate"
    ]
    
    for dep in dependencies:
        if not run_command(f"pip install {dep}", f"Installing {dep}"):
            return False
    return True

def test_model_loading():
    """Test if the model can be loaded"""
    print("🧪 Testing model loading...")
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        print("   Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("bahree/london-historical-slm")
        
        print("   Loading model...")
        model = AutoModelForCausalLM.from_pretrained("bahree/london-historical-slm")
        
        print("   Testing device detection...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        print(f"✅ Model loaded successfully on {device}")
        print(f"   Vocabulary size: {tokenizer.vocab_size:,}")
        print(f"   Model parameters: ~117M")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def create_test_script():
    """Create a simple test script"""
    test_script = """#!/usr/bin/env python3
# Quick test script for London Historical SLM

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("bahree/london-historical-slm")
model = AutoModelForCausalLM.from_pretrained("bahree/london-historical-slm")

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"Model loaded on {device}")

# Test generation
prompt = "In the year 1834, I walked through the streets of London and witnessed"
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
print(f"Generated: {result}")
"""
    
    with open("test_london_slm.py", "w") as f:
        f.write(test_script)
    
    print("✅ Created test_london_slm.py")

def main():
    """Main setup function"""
    print("🏛️ London Historical SLM - Test Environment Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        return False
    
    # Test model loading
    if not test_model_loading():
        print("❌ Model loading test failed")
        return False
    
    # Create test script
    create_test_script()
    
    print("\n🎉 Setup completed successfully!")
    print("\n📝 Next steps:")
    print("1. Run: python test_london_slm.py")
    print("2. Or run: python 07_utilities/test_huggingface_slm.py")
    print("3. Check the model at: https://huggingface.co/bahree/london-historical-slm")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
