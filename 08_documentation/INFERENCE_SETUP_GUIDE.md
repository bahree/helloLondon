# London Historical SLM - Inference Setup Guide

This guide shows you how to set up the London Historical SLM for inference on a new Windows machine with a single GPU.

> **Published Model**: Use the pre-trained [London Historical SLM](https://huggingface.co/bahree/london-historical-slm) on Hugging Face  
> **Status**: Both PyTorch checkpoint and Hugging Face inference are working perfectly!

## **Current Status**

| **Inference Type** | **Status** | **Notes** |
|-------------------|------------|-----------|
| **PyTorch Checkpoints** | Working | Both SLM and Regular models tested |
| **Hugging Face Models** | Working | SLM published, Regular ready to publish |
| **Local Testing** | Complete | Tested on remote Ubuntu machine |
| **Warning Suppression** | Fixed | Clean output without verbose warnings |

## Prerequisites

- **Windows 10/11** with single GPU
- **Python 3.8+** installed
- **Internet connection** for downloading the model
- **At least 4GB RAM** (8GB+ recommended)
- **At least 2GB free disk space** for the model

> **Ubuntu/Debian Users**: You also need `python3-venv` package:
> ```bash
> sudo apt install python3-venv  # For Python 3.8-3.11
> sudo apt install python3.12-venv  # For Python 3.12+
> ```

---

## Set up a virtual environment (Recommended)

> Virtual environments isolate project dependencies and prevent conflicts.

**Check Python & pip**
```powershell
# Windows (PowerShell)
python --version; python -m pip --version
```

**Create the env**
```bash
# Linux/macOS
python3 -m venv helloLondon
```

```powershell
# Windows (PowerShell)
python -m venv helloLondon
```

```cmd
:: Windows (Command Prompt)
python -m venv helloLondon
```

> **Note**: You can name your virtual environment anything you like, e.g., `.venv`, `my_env`, `london_env`.

**Activate**
```bash
# Linux/macOS
source helloLondon/bin/activate
```

```powershell
# Windows (PowerShell)
.\helloLondon\Scripts\Activate.ps1
```

```cmd
:: Windows (CMD)
.\helloLondon\Scripts\activate.bat
```

> If PowerShell blocks activation (*"running scripts is disabled"*), set the policy then retry activation:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
# or just for this session:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

---

## Install libraries

Upgrade basics, then install Hugging Face libs:

```bash
python -m pip install -U pip setuptools wheel
python -m pip install "transformers" "accelerate" "safetensors"
```

**Note**: The installation process includes progress indicators and may take several minutes on slow connections.

---

## Install PyTorch (Choose your variant)

### **A) CPU-only (if no GPU or for testing)**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### **B) NVIDIA GPU (CUDA) - Recommended for performance**
Pick the CUDA series that matches your system:

```bash
# CUDA 12.6 (most recent)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# CUDA 11.8 (older systems)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Quick sanity check**
```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
```

---

## Option 1: Direct Hugging Face (Recommended for Quick Start)

### **Step 2: Test the Model**

#### **Test Published Model (Hugging Face)**
```bash
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model (will download automatically)
tokenizer = AutoTokenizer.from_pretrained('bahree/london-historical-slm')
model = AutoModelForCausalLM.from_pretrained('bahree/london-historical-slm')

# Generate text
prompt = 'In the year 1834, I walked through the streets of London and witnessed'
inputs = tokenizer(prompt, return_tensors='pt')
outputs = model.generate(inputs['input_ids'], max_new_tokens=50, do_sample=True)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
"
```

#### **Test PyTorch Checkpoints (If You Have Local Models)**
```bash
# Test SLM checkpoint (117M parameters)
python 06_inference/inference_pytorch.py \
  --checkpoint 09_models/checkpoints/slm/checkpoint-4000.pt \
  --prompt "In the year 1834, I walked through the streets of London and witnessed"

# Test Regular checkpoint (354M parameters)
python 06_inference/inference_pytorch.py \
  --checkpoint 09_models/checkpoints/checkpoint-60001.pt \
  --prompt "In the year 1834, I walked through the streets of London and witnessed"
```

**Expected Output:**
- Clean logging without verbose warnings
- Model type detection (SLM vs Regular)
- Parameter counts (~80M for SLM, ~233M for Regular)
- Generated historical text with London references

### **Step 3: Use the Inference Script**
```bash
# Download the inference script
curl -O https://raw.githubusercontent.com/bahree/helloLondon/main/07_utilities/inference_slm.py

# Run demo prompts
python inference_slm.py --demo

# Interactive mode
python inference_slm.py --interactive

# Single prompt
python inference_slm.py --prompt "The Thames flowed dark and mysterious through the heart"
```

### **Step 4: Custom Usage in Your Code**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model
tokenizer = AutoTokenizer.from_pretrained("bahree/london-historical-slm")
model = AutoModelForCausalLM.from_pretrained("bahree/london-historical-slm")

# Set pad token if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Generate historical text
def generate_historical_text(prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_new_tokens=max_length,
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
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
prompt = "The gentleman from the country said, 'I have never seen such a sight"
result = generate_historical_text(prompt)
print(result)
```


## Option 3: Automated Setup (Easiest)

### **Step 1: Download Setup Script**
```bash
curl -O https://raw.githubusercontent.com/bahree/helloLondon/main/07_utilities/setup_inference.py
```

### **Step 2: Run Automated Setup**
```bash
python setup_inference.py
```

This will:
- Check Python installation
- Install all dependencies
- Test model loading
- Validate everything works

## **Performance Comparison**

| Method | Setup Time | Memory Usage | API Available | Best For |
|--------|------------|--------------|---------------|----------|
| **Hugging Face Direct** | 2-3 minutes | 2-4GB | No | Development, Testing |
| **Automated Setup** | 3-5 minutes | 2-4GB | Depends | New users, Quick start |

## **Example Prompts to Try**

### **Basic Historical Prompts:**
- "In the year 1834, I walked through the streets of London and witnessed"
- "The gentleman from the country said, 'I have never seen such a sight"
- "The Thames flowed dark and mysterious through the heart"
- "Merchants plied their wares in the bustling market"
- "The Great Fire of 1666 had destroyed"

### **London-Specific Prompts:**
- "Parliament sat in Westminster Hall"
- "The Tower of London stood"
- "In the streets of Cheapside,"
- "Covent Garden was filled with"
- "Samuel Pepys wrote that"

### **Historical Language Prompts:**
- "Chapter I: The Beginning of the End"
- "Mr. Darcy walked through the ballroom with his usual air"
- "The plague had ravaged"
- "The old man sat by the fire, reading his Bible"

## Troubleshooting

### **Common Issues:**

#### **1. `ImportError: AutoModelForCausalLM requires the PyTorch library`**
```bash
# Install PyTorch with the correct accelerator variant
# For CPU:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# For NVIDIA GPU:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

#### **2. Model Download Fails**
```bash
# Check internet connection
ping huggingface.co

# Try with different mirror
export HF_ENDPOINT=https://hf-mirror.com

# Check Hugging Face status page
```

#### **3. Out of Memory**
```bash
# Use CPU instead of GPU
python inference_slm.py --device cpu

# Reduce max length
python inference_slm.py --prompt "Your prompt" --max-length 25

# Use device mapping for large models
python -c "
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('bahree/london-historical-slm', device_map='auto')
"
```

#### **4. PowerShell Script Execution Disabled**
```powershell
# Set execution policy for current user
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

# Or just for this session
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Then retry activation
.\helloLondon\Scripts\Activate.ps1
```


#### **6. Slow Generation**
```bash
# Use GPU if available
python inference_slm.py --device cuda

# Reduce max length
python inference_slm.py --prompt "Your prompt" --max-length 30

# Check GPU utilization
nvidia-smi
```

#### **7. Virtual Environment Issues**
```bash
# If activation fails, try:
python -m venv helloLondon --clear

# Or recreate the environment
rmdir /s helloLondon
python -m venv helloLondon
.\helloLondon\Scripts\activate.bat
```

## **Advanced Usage**

### **Custom Generation Parameters**
```python
# Fine-tune generation for your use case
result = generate_historical_text(
    prompt="Your prompt here",
    max_length=100,
    temperature=0.9,  # More creative
    top_p=0.9,        # More focused
    repetition_penalty=1.1  # Less repetitive
)
```

### **Batch Processing**
```python
prompts = [
    "In the year 1834, I walked through the streets of London and witnessed",
    "The gentleman from the country said, 'I have never seen such a sight",
    "The Thames flowed dark and mysterious through the heart"
]

results = []
for prompt in prompts:
    result = generate_historical_text(prompt)
    results.append(result)
    print(f"Prompt: {prompt}")
    print(f"Generated: {result}")
    print("-" * 50)
```

### **Integration with Other Tools**
```python
# Flask API
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.json['prompt']
    result = generate_historical_text(prompt)
    return jsonify({'generated_text': result})

if __name__ == '__main__':
    app.run(debug=True)
```

## **Success!**

You now have the London Historical SLM running on your Windows machine! 

### **Next Steps:**
1. **Test with different prompts** to see the model's capabilities
2. **Integrate with your applications** using the API
3. **Share your results** with the community
4. **Contribute improvements** to the model

### **Resources:**
- **Model Page**: https://huggingface.co/bahree/london-historical-slm
- **Full Model**: https://huggingface.co/bahree/london-historical-llm
- **GitHub Repository**: https://github.com/bahree/helloLondon
- **Documentation**: https://github.com/bahree/helloLondon/tree/main/08_documentation

## 📞 **Support**

If you encounter issues:
1. **Check the troubleshooting section** above
2. **Open an issue** on GitHub
3. **Join the community** discussions
4. **Read the full documentation**

---

**Happy historical text generation!** 🏛️
