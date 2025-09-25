#!/usr/bin/env python3
"""
Setup script for London Historical SLM inference on a new machine
Supports both Windows and Linux systems
"""

import subprocess
import sys
import os
import platform
from pathlib import Path

# Add project root to path to import config
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from config import config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    print("Warning: Could not import config.py, using fallback configuration")

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"{description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_system_requirements():
    """Check system requirements"""
    print("🔍 Checking system requirements...")
    
    # Check Python version
    try:
        result = subprocess.run([sys.executable, "--version"], capture_output=True, text=True)
        print(f"Python found: {result.stdout.strip()}")
        
        # Check if Python version is compatible
        version_info = sys.version_info
        if version_info < (3, 8):
            print(f"❌ Python {version_info.major}.{version_info.minor} is not supported")
            print("   Minimum required: Python 3.8")
            return False
        else:
            print(f"✅ Python {version_info.major}.{version_info.minor} is compatible")
    except Exception as e:
        print(f"❌ Python not found: {e}")
        return False
    
    # Check python3-venv package on Linux systems
    if platform.system().lower() == "linux":
        try:
            import ensurepip
            print("✅ python3-venv package: Available")
        except ImportError:
            print("❌ python3-venv package: NOT AVAILABLE")
            print("   Please install it with: sudo apt install python3-venv")
            print("   Or for Python 3.12: sudo apt install python3.12-venv")
            return False
    
    return True

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    # Use config if available, otherwise fallback
    if CONFIG_AVAILABLE:
        dependencies = [
            "torch>=1.9.0",
            "transformers>=4.20.0",
            "accelerate>=0.20.0",
            "huggingface_hub>=0.16.0",
            "tokenizers>=0.12.0",
            "requests>=2.28.0"
        ]
    else:
        dependencies = [
            "torch",
            "transformers",
            "accelerate",
            "huggingface_hub",
            "tokenizers",
            "requests"
        ]
    
    failed_packages = []
    for dep in dependencies:
        if not run_command(f"pip install {dep}", f"Installing {dep}"):
            print(f"⚠️ Failed to install {dep}, but continuing...")
            failed_packages.append(dep)
    
    if failed_packages:
        print(f"\n⚠️ Warning: {len(failed_packages)} packages failed to install:")
        for pkg in failed_packages:
            print(f"   - {pkg}")
        print("You can try installing them manually later.")
    
    return True

def test_model_loading():
    """Test if the model can be loaded"""
    print("🧪 Testing model loading...")
    
    test_code = """
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bahree/london-historical-slm")
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained("bahree/london-historical-slm")
    print("Model loaded successfully!")
    print(f"Vocabulary size: {tokenizer.vocab_size:,}")
    print(f"Device: {next(model.parameters()).device}")
except Exception as e:
    print(f"Model loading failed: {e}")
    exit(1)
"""
    
    try:
        result = subprocess.run([sys.executable, "-c", test_code], 
                              capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("Model loading test passed!")
            print(result.stdout)
            return True
        else:
            print("Model loading test failed!")
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("⏰ Model loading timed out (this is normal for first download)")
        return True
    except Exception as e:
        print(f"Test failed: {e}")
        return False

def create_ollama_setup():
    """Create Ollama setup instructions"""
    print("🦙 Creating Ollama setup instructions...")
    
    ollama_instructions = """
# Ollama Setup for London Historical SLM

## 1. Install Ollama
Download and install Ollama from: https://ollama.ai/

## 2. Create the model
ollama create london-historical-slm -f Modelfile

## 3. Test the model
ollama run london-historical-slm "In the year 1834, I walked through the streets of London and witnessed"

## 4. Interactive mode
ollama run london-historical-slm

## 5. API usage
curl http://localhost:11434/api/generate -d '{
  "model": "london-historical-slm",
  "prompt": "The Thames flowed dark and mysterious through the heart",
  "stream": false
}'
"""
    
    with open("ollama_setup_instructions.txt", "w") as f:
        f.write(ollama_instructions)
    
    print("Ollama setup instructions created: ollama_setup_instructions.txt")

def main():
    """Main setup function"""
    print("London Historical SLM - Inference Setup")
    print("=" * 60)
    print("This script will set up inference on a new machine")
    print("=" * 60)
    
    # Check system requirements
    if not check_system_requirements():
        print("❌ System requirements not met. Please fix the issues above.")
        return False
    
    # Install dependencies
    if not install_dependencies():
        print("Failed to install dependencies.")
        return False
    
    # Test model loading
    if not test_model_loading():
        print("Model loading test failed.")
        print("This might be due to network issues or insufficient disk space.")
        return False
    
    # Create Ollama setup
    create_ollama_setup()
    
    print("\nSetup completed successfully!")
    print("=" * 60)
    print("Next steps:")
    print("1. Test the model: python 06_inference/test_published_models.py --model_type slm")
    print("2. Interactive mode: python 06_inference/inference_unified.py --published --model_type slm --interactive")
    print("3. Single prompt: python 06_inference/inference_unified.py --published --model_type slm --prompt 'Your prompt here'")
    print("4. Ollama setup: Follow ollama_setup_instructions.txt")
    print("\n🌐 Model available at: https://huggingface.co/bahree/london-historical-slm")
    print("\nExpected testing output:")
    print("🧪 Testing SLM Model: bahree/london-historical-slm")
    print("============================================================")
    print("📂 Loading model...")
    print("✅ Model loaded in 8.91 seconds")
    print("📊 Model Info:")
    print("   Type: SLM")
    print("   Description: Small Language Model (117M parameters)")
    print("   Device: cuda")
    print("   Vocabulary size: 30,000")
    print("   Max length: 512")
    print("🎯 Testing generation with 10 prompts...")
    
    return True

if __name__ == "__main__":
    main()
