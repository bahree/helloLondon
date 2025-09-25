# 07_utilities

This directory contains **lightweight utility scripts** for the London Historical LLM project, specifically designed for **inference-only** use cases.

> **💡 Note**: This is a **lightweight setup** for inference only. For full development (training, evaluation, data collection), use the complete setup in `01_environment/setup_environment.py`.

## **When to Use This Setup**

### **✅ Use 07_utilities (Lightweight) When:**
- You only want to **run inference** on published models
- You're deploying to **production machines** that don't need training
- You want **minimal dependencies** (just torch, transformers, accelerate)
- You're **testing published models** on new machines

### **✅ Use 01_environment (Full Setup) When:**
- You want to **train models** from scratch
- You need **data collection** and processing tools
- You want **evaluation** and testing capabilities
- You're doing **full development** work

## **Quick Start**

### **Test the Model (2 minutes)**
```bash
python test_huggingface_slm.py
```

### **Interactive Mode**
```bash
# Use the consolidated inference script
python 06_inference/inference_unified.py --published --interactive
```

### **Single Prompt**
```bash
# Use the consolidated inference script
python 06_inference/inference_unified.py --published --prompt "In the year 1834, I walked through the streets of London and witnessed"
```

### **Automated Setup**
```bash
python setup_inference.py
```

## **Available Scripts**

### **Core Inference Scripts**
- **`test_huggingface_slm.py`** - Test the SLM model from Hugging Face Hub
- **`setup_inference.py`** - Automated setup for inference on new machines
- **`setup_test_environment.py`** - Quick test environment setup

### **Consolidated Inference (Recommended)**
- **`06_inference/inference_unified.py`** - Main unified inference script (supports both SLM and regular models)
- **`06_inference/inference_pytorch.py`** - PyTorch checkpoint inference
- **`06_inference/test_published_models.py`** - Comprehensive model testing

### **Ollama Integration**
- **`Modelfile`** - Ollama configuration for the SLM model
- **`ollama_setup_instructions.txt`** - Generated setup instructions

### **Publishing Scripts**
- **`publish_to_huggingface.py`** - Publish the full model to Hugging Face (if exists)
- **`publish_slm_to_huggingface.py`** - Publish the SLM model to Hugging Face (if exists)

## 🎯 **Usage Examples**

### **Basic Usage**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model
tokenizer = AutoTokenizer.from_pretrained("bahree/london-historical-slm")
model = AutoModelForCausalLM.from_pretrained("bahree/london-historical-slm")

# Generate text
prompt = "The Thames flowed dark and mysterious through the heart"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(inputs['input_ids'], max_new_tokens=50, do_sample=True)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

### **Command Line Usage**
```bash
# Demo mode (using consolidated script)
python 06_inference/inference_unified.py --published --demo

# Interactive mode (using consolidated script)
python 06_inference/inference_unified.py --published --interactive

# Single prompt (using consolidated script)
python 06_inference/inference_unified.py --published --prompt "Your prompt here"

# Custom parameters (using consolidated script)
python 06_inference/inference_unified.py --published --prompt "Your prompt" --max_length 100 --device cuda

# Test all published models
python 06_inference/test_published_models.py

# PyTorch checkpoint inference
python 06_inference/inference_pytorch.py --checkpoint path/to/checkpoint.pt --interactive
```

### **Ollama Usage**
```bash
# Create model
ollama create london-historical-slm -f Modelfile

# Test model
ollama run london-historical-slm "Your prompt here"

# API usage
curl http://localhost:11434/api/generate -d '{
  "model": "london-historical-slm",
  "prompt": "Your prompt here",
  "stream": false
}'
```

## 🧪 **Example Prompts**

### **Historical London Prompts**
- "In the year 1834, I walked through the streets of London and witnessed"
- "The gentleman from the country said, 'I have never seen such a sight"
- "The Thames flowed dark and mysterious through the heart"
- "Merchants plied their wares in the bustling market"
- "The Great Fire of 1666 had destroyed"

### **London Landmarks**
- "Parliament sat in Westminster Hall"
- "The Tower of London stood"
- "In the streets of Cheapside,"
- "Covent Garden was filled with"

### **Historical Language**
- "Chapter I: The Beginning of the End"
- "Mr. Darcy walked through the ballroom with his usual air"
- "The plague had ravaged"
- "Samuel Pepys wrote that"

## 🔧 **Troubleshooting**

### **Common Issues**
1. **Model download fails** - Check internet connection
2. **Out of memory** - Use `--device cpu` or reduce `--max-length`
3. **Slow generation** - Use `--device cuda` if GPU available
4. **Ollama fails** - Check if Ollama service is running

### **Performance Tips**
- **Use GPU** if available: `--device cuda`
- **Reduce max length** for faster generation: `--max-length 30`
- **Use Ollama** for production workloads
- **Batch process** multiple prompts together

## 📚 **Documentation**

- **Complete Setup Guide**: [08_documentation/INFERENCE_SETUP_GUIDE.md](08_documentation/INFERENCE_SETUP_GUIDE.md)
- **Inference Quick Start**: [08_documentation/INFERENCE_QUICK_START.md](08_documentation/INFERENCE_QUICK_START.md)
- **Historical Tokenizer**: [08_documentation/HISTORICAL_TOKENIZER.md](08_documentation/HISTORICAL_TOKENIZER.md)
- **Training Guide**: [08_documentation/TRAINING_GUIDE.md](08_documentation/TRAINING_GUIDE.md)

## 🌐 **Model Links**

- **SLM Model**: https://huggingface.co/bahree/london-historical-slm
- **Full Model**: https://huggingface.co/bahree/london-historical-llm
- **GitHub Repository**: https://github.com/bahree/helloLondon

---

**Ready to generate historical London text!** 🏛️
