#!/usr/bin/env python3
"""
Publish London Historical SLM to Hugging Face Hub
Adapted from the main model publishing script
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config import config

try:
    from huggingface_hub import HfApi, Repository
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Install with: pip install huggingface_hub")
    sys.exit(1)

def prepare_model_for_upload():
    """Prepare the trained SLM model for Hugging Face upload"""
    print(f"Preparing SLM model for Hugging Face upload...")
    
    # Check if SLM checkpoint exists
    checkpoint_dir = Path(config.checkpoints_dir) / "slm"
    if not checkpoint_dir.exists():
        print(f"SLM checkpoint directory not found: {checkpoint_dir}")
        print(f"Available directories:")
        for d in config.checkpoints_dir.iterdir():
            if d.is_dir():
                print(f"  - {d}")
        return False
    
    # Find the latest checkpoint
    checkpoint_files = list(checkpoint_dir.glob("checkpoint-*.pt"))
    if not checkpoint_files:
        print(f"No checkpoint files found in {checkpoint_dir}")
        return False
    
    # Get the latest checkpoint
    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.stem.split('-')[1]))
    print(f"Found latest checkpoint: {latest_checkpoint}")
    
    # Check if tokenizer exists
    tokenizer_dir = Path(config.london_tokenizer_dir)
    if not tokenizer_dir.exists():
        print(f"Tokenizer directory not found: {tokenizer_dir}")
        return False
    
    # Create a temporary directory for Hugging Face format
    hf_model_dir = Path(config.checkpoints_dir) / "slm_hf_format"
    hf_model_dir.mkdir(exist_ok=True)
    
    try:
        # Load checkpoint and convert to Hugging Face format
        print("Converting checkpoint to Hugging Face format...")
        checkpoint = torch.load(latest_checkpoint, map_location='cpu')
        
        # Infer context length (n_positions) from positional embedding shape
        state = checkpoint.get('model', {})
        pos_keys = [k for k in state.keys() if k.endswith('transformer.wpe.weight') or k.endswith('.wpe.weight')]
        inferred_n_positions = state[pos_keys[0]].shape[0] if pos_keys else 512
        print(f"Inferred n_positions from checkpoint: {inferred_n_positions}")
        
        # Create model config
        model_config = {
            "architectures": ["GPT2LMHeadModel"],
            "model_type": "gpt2",
            "n_layer": 12,
            "n_head": 12,
            "n_embd": 768,
            "n_positions": inferred_n_positions,
            "n_ctx": inferred_n_positions,
            "vocab_size": 30000,
            "activation_function": "gelu_new",
            "attn_pdrop": 0.1,
            "embd_pdrop": 0.1,
            "resid_pdrop": 0.1,
            "layer_norm_epsilon": 1e-5,
            "initializer_range": 0.02,
            "use_cache": True,
            "bos_token_id": 0,
            "eos_token_id": 2,
            "pad_token_id": 1,
            "unk_token_id": 3,
            "mask_token_id": 4,
            "max_length": 512,
            "do_sample": True,
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 40,
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 3,
            "transformers_version": "4.21.0"
        }
        
        # Save model config
        with open(hf_model_dir / "config.json", "w") as f:
            json.dump(model_config, f, indent=2)
        
        # Save generation config
        generation_config = {
            "max_length": inferred_n_positions,
            "do_sample": True,
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 40,
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 3,
            "pad_token_id": 1,
            "eos_token_id": 2,
            "bos_token_id": 0
        }
        
        with open(hf_model_dir / "generation_config.json", "w") as f:
            json.dump(generation_config, f, indent=2)
        
        # Copy tokenizer files (but NOT config.json - we already created the model config)
        import shutil
        for file in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
            src = tokenizer_dir / file
            if src.exists():
                shutil.copy2(src, hf_model_dir / file)
        
        # Convert model weights to safetensors format
        print("Converting model weights to safetensors...")
        from safetensors.torch import save_file
        
        # Load the model state dict
        model_state = checkpoint['model']
        
        # Handle shared tensors and rename keys to match Hugging Face convention
        print("Handling shared tensors and renaming keys...")
        processed_state = {}
        for key, tensor in model_state.items():
            # Create a copy to avoid shared memory issues
            processed_state[key] = tensor.clone()
        
        # Rename keys and fix tensor shapes to match Hugging Face convention
        print("Renaming model weights and fixing shapes for Hugging Face convention...")
        renamed_state = {}
        renamed_count = 0
        transposed_count = 0
        
        for key, tensor in processed_state.items():
            # Remove the '_orig_mod.' prefix
            if key.startswith('_orig_mod.'):
                new_key = key.replace('_orig_mod.', '')
                
                # Fix tensor shapes for specific layers that need transposition
                if 'c_attn.weight' in new_key or 'c_proj.weight' in new_key or 'c_fc.weight' in new_key or 'mlp.c_proj.weight' in new_key:
                    # These layers need to be transposed for Hugging Face format
                    if len(tensor.shape) == 2:
                        tensor = tensor.t().contiguous()
                        transposed_count += 1
                        if transposed_count <= 3:  # Show first 3 transpositions
                            print(f"  Transposed {new_key}: {tensor.shape}")
                
                renamed_state[new_key] = tensor
                renamed_count += 1
                if renamed_count <= 5:  # Show first 5 renames
                    print(f"  {key} -> {new_key}")
            else:
                renamed_state[key] = tensor
        
        print(f"Renamed {renamed_count} weights and transposed {transposed_count} tensors for Hugging Face convention")
        
        # Convert to safetensors format
        save_file(renamed_state, hf_model_dir / "model.safetensors")
        
        print(f"Model prepared for Hugging Face upload")
        print(f"   Model directory: {hf_model_dir}")
        return True
        
    except Exception as e:
        print(f"Error preparing model: {e}")
        return False

def create_model_card():
    """Create a model card for the SLM"""
    model_card = r"""---
license: mit
library_name: transformers
pipeline_tag: text-generation
language:
- en
tags:
- gpt2
- historical
- london
- slm
- small-language-model
- text-generation
- history
- english
- safetensors
---

# London Historical LLM – Small Language Model (SLM)

A compact GPT-2 Small model (~117M params) **trained from scratch** on historical London texts (1500–1850). Fast to run on CPU, and supports NVIDIA (CUDA) and AMD (ROCm) GPUs.

> **Note**: This model was **trained from scratch** - not fine-tuned from existing models.

> This page includes simple **virtual-env setup**, **install choices for CPU/CUDA/ROCm**, and an **auto-device inference** example so anyone can get going quickly.

---

## 🔎 Model Description

This is a **Small Language Model (SLM)** version of the London Historical LLM, **trained from scratch** using GPT-2 Small architecture on historical London texts with a custom historical tokenizer. The model was built from the ground up, not fine-tuned from existing models.

### Key Features
- ~117M parameters (vs ~354M in the full model)  
- Custom historical tokenizer (≈30k vocab)  
- London-specific context awareness and historical language patterns (e.g., *thou, thee, hath*)  
- Lower memory footprint and faster inference on commodity hardware  
- **Trained from scratch** - not fine-tuned from existing models  

---

## 🧪 Intended Use & Limitations

**Use cases:** historical-style narrative generation, prompt-based exploration of London themes (1500–1850), creative writing aids.  
**Limitations:** may produce anachronisms or historically inaccurate statements; smaller models have less complex reasoning than larger LLMs. Validate outputs before downstream use.

---

## 🐍 Set up a virtual environment (Linux/macOS/Windows)

> Virtual environments isolate project dependencies. Official Python docs: `venv`.

**Check Python & pip**
```bash
# Linux/macOS
python3 --version && python3 -m pip --version
```

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

## 📦 Install libraries

Upgrade basics, then install Hugging Face libs:

```bash
python -m pip install -U pip setuptools wheel
python -m pip install "transformers" "accelerate" "safetensors"
```

---

## Install **one** PyTorch variant (CPU / NVIDIA / AMD)

Use **one** of the commands below. For the most accurate command per OS/accelerator and version, prefer PyTorch's **Get Started** selector.

### A) CPU-only (Linux/Windows/macOS)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### B) NVIDIA GPU (CUDA)

Pick the CUDA series that matches your system (examples below):

```bash
# CUDA 12.6
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### C) AMD GPU (ROCm, **Linux-only**)

Install the ROCm build matching your ROCm runtime (examples):

```bash
# ROCm 6.3
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3

# ROCm 6.2 (incl. 6.2.x)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2.4

# ROCm 6.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1
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

## 🚀 Inference (auto-detect device)

This snippet picks the best device (CUDA/ROCm if available, else CPU) and uses sensible generation defaults for this SLM.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "bahree/london-historical-slm"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

prompt = "In the year 1834, I walked through the streets of London and witnessed"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

outputs = model.generate(
    inputs["input_ids"],
    max_new_tokens=50,
    do_sample=True,
    temperature=0.8,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    early_stopping=True,
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## 🧪 **Testing Your Model**

### **Quick Testing (10 Automated Prompts)**
```bash
# Test with 10 automated historical prompts
python 06_inference/test_published_models.py --model_type slm
```

**Expected Output:**
```
🧪 Testing SLM Model: bahree/london-historical-slm
============================================================
📂 Loading model...
✅ Model loaded in 8.91 seconds
📊 Model Info:
   Type: SLM
   Description: Small Language Model (117M parameters)
   Device: cuda
   Vocabulary size: 30,000
   Max length: 512

🎯 Testing generation with 10 prompts...
[10 automated tests with historical text generation]
```

### **Interactive Testing**
```bash
# Interactive mode for custom prompts
python 06_inference/inference_unified.py --published --model_type slm --interactive

# Single prompt test
python 06_inference/inference_unified.py --published --model_type slm --prompt "In the year 1834, I walked through the streets of London and witnessed"
```

**Need more headroom later?** Load with 🤗 Accelerate and `device_map="auto"` to spread layers across available devices/CPU automatically.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
```

---

## 🪟 Windows Terminal one-liners

**PowerShell**

```powershell
python -c "from transformers import AutoTokenizer,AutoModelForCausalLM; m='bahree/london-historical-slm'; t=AutoTokenizer.from_pretrained(m); model=AutoModelForCausalLM.from_pretrained(m); p='In the year 1834, I walked through the streets of London and witnessed'; i=t(p,return_tensors='pt'); print(t.decode(model.generate(i['input_ids'],max_new_tokens=50,do_sample=True)[0],skip_special_tokens=True))"
```

**Command Prompt (CMD)**

```cmd
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM ^&^& import torch ^&^& m='bahree/london-historical-slm' ^&^& t=AutoTokenizer.from_pretrained(m) ^&^& model=AutoModelForCausalLM.from_pretrained(m) ^&^& p='In the year 1834, I walked through the streets of London and witnessed' ^&^& i=t(p, return_tensors='pt') ^&^& print(t.decode(model.generate(i['input_ids'], max_new_tokens=50, do_sample=True)[0], skip_special_tokens=True))"
```

---

## 💡 Basic Usage (Python)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("bahree/london-historical-slm")
model = AutoModelForCausalLM.from_pretrained("bahree/london-historical-slm")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

prompt = "In the year 1834, I walked through the streets of London and witnessed"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    inputs["input_ids"],
    max_new_tokens=50,
    do_sample=True,
    temperature=0.8,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    early_stopping=True,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 🧰 Example Prompts

* **Tudor (1558):** "On this day in 1558, Queen Mary has died and …"
* **Stuart (1666):** "The Great Fire of London has consumed much of the city, and …"
* **Georgian/Victorian:** "As I journeyed through the streets of London, I observed …"
* **London specifics:** "Parliament sat in Westminster Hall …", "The Thames flowed dark and mysterious …"

---

## 🛠️ Training Details

* **Architecture:** GPT-2 Small (12 layers, hidden size 768)
* **Params:** ~117M
* **Tokenizer:** custom historical tokenizer (~30k vocab) with London-specific and historical tokens
* **Data:** historical London corpus (1500–1850)
* **Steps/Epochs:** 30,000 steps (extended training for better convergence)
* **Batch/LR:** 32, 3e-4 (optimized for segmented data)
* **Hardware:** 2× GPU training with Distributed Data Parallel
* **Final Training Loss:** 1.395 (43% improvement from 20K steps)
* **Model Flops Utilization:** 3.5% (excellent efficiency)
* **Training Method:** **Trained from scratch** - not fine-tuned
* **Context Length:** 256 tokens (optimized for historical text segments)
* **Status:** ✅ **Successfully published and tested** - ready for production use

---

## 🔤 Historical Tokenizer

* Compact 30k vocab targeting 1500–1850 English
* Tokens for **year/date/name/place/title**, plus **thames**, **westminster**, etc.; includes **thou/thee/hath/doth** style markers

---

## ⚠️ Troubleshooting

* **`ImportError: AutoModelForCausalLM requires the PyTorch library`**
  → Install PyTorch with the correct accelerator variant (see CPU/CUDA/ROCm above or use the official selector).

* **AMD GPU not used**
  → Ensure you installed a ROCm build and you're on Linux (`pip install ... --index-url https://download.pytorch.org/whl/rocmX.Y`). Verify with `torch.cuda.is_available()` and check the device name. ROCm wheels are Linux-only.

* **Running out of VRAM**
  → Try smaller batch/sequence lengths, or load with `device_map="auto"` via 🤗 Accelerate to offload layers to CPU/disk.

---

## 📚 Citation

If you use this model, please cite:

```bibtex
@misc{london-historical-slm,
  title   = {London Historical LLM - Small Language Model: A Compact GPT-2 for Historical Text Generation},
  author  = {Amit Bahree},
  year    = {2025},
  url     = {https://huggingface.co/bahree/london-historical-slm}
}
```

---

## 🧾 License

MIT (see [LICENSE](https://huggingface.co/bahree/london-historical-slm/blob/main/LICENSE) in repo).
"""
    
    return model_card

def upload_to_huggingface(repo_name: str, username: str, token: str):
    """Upload the SLM model to Hugging Face Hub"""
    print(f"Uploading SLM model to Hugging Face Hub...")
    print(f"   Repository: {username}/{repo_name}")
    
    try:
        # Initialize HF API
        api = HfApi()
        
        # Create repository
        repo_url = api.create_repo(
            repo_id=f"{username}/{repo_name}",
            token=token,
            exist_ok=True,
            private=False
        )
        print(f"Repository created: {repo_url}")
        
        # Upload model files from HF format directory
        hf_model_path = Path(config.checkpoints_dir) / "slm_hf_format"
        api.upload_folder(
            folder_path=str(hf_model_path),
            repo_id=f"{username}/{repo_name}",
            token=token,
            commit_message="Upload London Historical SLM (30K steps, improved training)"
        )
        
        # Upload model card
        model_card = create_model_card()
        api.upload_file(
            path_or_fileobj=model_card.encode(),
            path_in_repo="README.md",
            repo_id=f"{username}/{repo_name}",
            token=token,
            commit_message="Add model card"
        )
        
        # Upload MIT license
        mit_license = """MIT License

Copyright (c) 2025 Amit Bahree

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
        
        api.upload_file(
            path_or_fileobj=mit_license.encode(),
            path_in_repo="LICENSE",
            repo_id=f"{username}/{repo_name}",
            token=token,
            commit_message="Add MIT license"
        )
        
        print(f"SLM model uploaded successfully!")
        print(f"   View at: https://huggingface.co/{username}/{repo_name}")
        return True
        
    except Exception as e:
        print(f"Upload failed: {e}")
        return False

def main():
    """Main function"""
    print(f"London Historical SLM - Hugging Face Upload")
    print("=" * 50)
    
    # Check if model is ready
    if not prepare_model_for_upload():
        return False
    
    # Get user input
    username = input("Enter your Hugging Face username (default: bahree): ").strip()
    if not username:
        username = "bahree"
    
    repo_name = input("Enter repository name (default: london-historical-slm): ").strip()
    if not repo_name:
        repo_name = "london-historical-slm"
    
    # Get token
    token = input("Enter your Hugging Face token (or set HF_TOKEN env var): ").strip()
    if not token:
        token = os.getenv("HF_TOKEN")
        if not token:
            print(f"Hugging Face token is required")
            print("   Get one at: https://huggingface.co/settings/tokens")
            return False
    
    # Upload model
    success = upload_to_huggingface(repo_name, username, token)
    
    if success:
        print("\n🎉 SLM model published successfully!")
        print(f"   Repository: https://huggingface.co/{username}/{repo_name}")
        print("\n📝 Next steps:")
        print("   1. Test your model on Hugging Face")
        print("   2. Share the model with others")
        print("   3. Compare with the full model")
    else:
        print("\n❌ Upload failed. Please check the error messages above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
