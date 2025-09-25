# Deployment Guide - London Historical LLM

Complete guide for deploying your trained model to different platforms.

## **Deployment Options**

### **1. Hugging Face Hub (Recommended)**
- **Best for**: Sharing with the community, easy integration
- **Pros**: Free, easy to use, integrates with many tools
- **Cons**: Requires internet connection

### **2. Custom API Server**
- **Best for**: Production applications, custom integrations
- **Pros**: Full control, custom features
- **Cons**: More complex setup

## 🤗 **Option 1: Hugging Face Hub**

### **Prerequisites:**
```bash
pip install huggingface_hub
```

### **Get Hugging Face Token:**
1. Go to https://huggingface.co/settings/tokens
2. Create a new token with "Write" permissions
3. Set environment variable: `export HF_TOKEN=your_token_here`

### **Publish Your Model:**
```bash
# Run the publishing script
python3 10_scripts/publish_to_huggingface.py

# Follow the prompts:
# - Enter your Hugging Face username
# - Enter repository name (default: london-historical-llm)
# - Enter your token
```

### **After Publishing:**
```python
# Use your model anywhere
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("bahree/london-historical-llm")
model = AutoModelForCausalLM.from_pretrained("bahree/london-historical-llm")

# Generate text
prompt = "In the year 1800, London was"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```


## **Option 3: Custom API Server**

### **Create FastAPI Server:**
```python
# api_server.py
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI(title="London Historical LLM API")

# Load model
tokenizer = AutoTokenizer.from_pretrained("./09_models/checkpoints")
model = AutoModelForCausalLM.from_pretrained("./09_models/checkpoints")

@app.post("/generate")
async def generate_text(prompt: str, max_length: int = 100):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_length)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": generated_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### **Run Server:**
```bash
pip install fastapi uvicorn
python3 api_server.py
```

### **Use API:**
```bash
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "In the year 1800, London was", "max_length": 100}'
```

## ⚙️ **Model Configuration**

### **Customize Model Name:**
Edit `config.py` line 54:
```python
"model_name": "london-historical-llm",  # Your custom model name
```

### **Model Architecture:**
This is a **custom model built from scratch** using GPT-2 architecture:
- **Architecture**: GPT-2 (custom implementation)
- **Parameters**: ~354M (medium size)
- **Vocabulary**: 50,000 tokens (custom tokenizer)
- **Context Length**: 1,024 tokens
- **Layers**: 24
- **Heads**: 16

### **Custom Model Names:**
You can customize your model name:
```python
"model_name": "helloLondon",  # or whatever you prefer
"model_name": "london-historical-llm-v1",
"model_name": "time-capsule-llm",
```

## 📊 **Performance Comparison**

| Option | Setup Time | Inference Speed | Offline | Sharing |
|--------|------------|-----------------|---------|---------|
| Hugging Face | 5 min | Medium | No | Easy |
| Custom API | 30 min | Fast | Yes | Custom |

## 🎯 **Recommendations**

### **For Development/Testing:**
- Use **Custom API** for local testing
- Fast setup, easy to iterate

### **For Sharing/Community:**
- Use **Hugging Face Hub**
- Easy to share, integrates with many tools

### **For Production:**
- Use **Custom API Server**
- Full control, can add custom features

## 🔍 **Troubleshooting**

### **Hugging Face Issues:**
```bash
# Check token
echo $HF_TOKEN

# Test connection
huggingface-cli whoami
```


### **Custom API Issues:**
```bash
# Check dependencies
pip install fastapi uvicorn transformers torch

# Check model files
ls -la 09_models/checkpoints/
```

## 📝 **Next Steps**

1. **Choose your deployment option**
2. **Follow the setup instructions**
3. **Test your model**
4. **Share with others!**

Your London Historical LLM is ready to be deployed! 🎉
