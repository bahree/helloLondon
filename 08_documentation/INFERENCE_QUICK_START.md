# Inference Quick Start Guide

**Use the published London Historical SLM in 2 minutes!**

> **Status**: Both PyTorch checkpoint and Hugging Face inference are working perfectly!

## **Current Status**

| **Model** | **PyTorch Checkpoint** | **Hugging Face** | **Tested** |
|-----------|------------------------|------------------|------------|
| **SLM (117M)** | Working | Published | Remote Ubuntu |
| **Regular (354M)** | Working | Ready to publish | Remote Ubuntu |

> **Main Project**: For the complete project overview, see the [London Historical LLM README](README.md)  
> **Want to train your own model?** See [Training Quick Start](TRAINING_QUICK_START.md)

## **Two Setup Paths**

This project supports two different setup approaches:

### **Path 1: Inference-Only Setup (This Guide)**
- **Purpose**: Use the published model for text generation
- **Setup**: `python3 07_utilities/setup_inference.py`
- **Includes**: Lightweight environment, just inference dependencies
- **Time**: 5-10 minutes setup
- **Use when**: You just want to use the model for text generation

### **Path 2: Full Training Setup**
- **Purpose**: Train your own model from scratch
- **Setup**: `python3 01_environment/setup_environment.py`
- **Includes**: Virtual environment, all ML dependencies, data collection, tokenizer training
- **Time**: 15-30 minutes setup + 7-24 hours training
- **Use when**: You want to understand the full pipeline or train custom models

> **This guide covers Path 1 (Inference-Only Setup)**

## **Quick Setup (Recommended)**

### **Prerequisites**
- **Python 3.8+** installed
- **Internet connection** (to download the model)

> **Ubuntu/Debian Users**: You also need `python3-venv` package:
> ```bash
> sudo apt install python3-venv  # For Python 3.8-3.11
> sudo apt install python3.12-venv  # For Python 3.12+
> ```

### **Step 1: Run Automated Setup**
```bash
# Clone the repository
git clone https://github.com/bahree/helloLondon.git
cd helloLondon

# Run the inference setup script
python3 07_utilities/setup_inference.py
```

This will:
- Check Python installation
- Install all dependencies (with progress indicators)
- Test model loading

**Installation Time**: The setup process includes progress indicators and may take 5-10 minutes on slow connections.

### **Step 2: Load and Use the Model**

#### **Option A: Hugging Face Model (Published)**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the pre-trained model (automatically downloads from Hugging Face)
tokenizer = AutoTokenizer.from_pretrained("bahree/london-historical-slm")
model = AutoModelForCausalLM.from_pretrained("bahree/london-historical-slm")

# Generate historical text
prompt = "In the year 1834, I walked through the streets of London and witnessed"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    inputs['input_ids'], 
    max_new_tokens=50, 
    do_sample=True,
    temperature=0.3,
    top_p=0.9,
    top_k=20,
    repetition_penalty=1.2
)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

#### **Option B: PyTorch Checkpoints (Local Training)**
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

### **Step 3: Try Different Prompts**
```python
prompts = [
    "The gentleman from the country said, 'I have never seen such a sight",
    "The Thames flowed dark and mysterious through the heart",
    "Parliament sat in Westminster Hall",
    "The Great Fire of 1666 had destroyed"
]

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs['input_ids'], max_new_tokens=30, do_sample=True)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Generated: {result}\n")
```

## **Option 2: Test with Scripts**

### **Clone and Test**
```bash
# Clone the repository
git clone https://github.com/bahree/helloLondon.git
cd helloLondon

# Test the published model
python 06_inference/test_published_models.py

# Interactive testing
python 06_inference/inference_unified.py --interactive
```


## **What You Get**

### **Historical Language Capabilities:**
- **Tudor English** (1500-1600): "thou", "thee", "hath", "doth"
- **Stuart Period** (1600-1700): Restoration language, court speech
- **Georgian Era** (1700-1800): Austen-style prose, social commentary
- **Victorian Times** (1800-1850): Dickens-style narrative, industrial language

### **London-Specific Knowledge:**
- **Landmarks**: Thames, Westminster, Tower, Fleet Street, Cheapside
- **Historical Events**: Great Fire, Plague, Civil War, Restoration
- **Social Classes**: Nobles, merchants, apprentices, beggars
- **Professions**: Apothecaries, coachmen, watermen, chimneysweeps

## **Troubleshooting**

### **Out of Memory**
```python
# Use CPU instead of GPU
model = AutoModelForCausalLM.from_pretrained("bahree/london-historical-slm", device_map="cpu")
```

### **Import Errors**
```bash
# Install missing dependencies
pip install transformers torch accelerate
```

### **Model Not Found**
```bash
# Check internet connection and try again
# The model will download automatically on first use
```

## **Next Steps**

- **Want to train your own model?** → [Training Quick Start](TRAINING_QUICK_START.md)
- **Need detailed training instructions?** → [Training Guide](TRAINING_GUIDE.md)
- **Want to understand the project?** → [London Historical LLM README](README.md)
- **Need help with evaluation?** → [Evaluation Quick Reference](EVALUATION_QUICK_REFERENCE.md)

