# Hugging Face Publishing Guide

This guide walks you through publishing your London Historical LLM models to Hugging Face Hub for sharing and deployment.

> **Success Example**: The [London Historical SLM](https://huggingface.co/bahree/london-historical-slm) has been successfully published using this guide!

## **Overview**

Publish your **SLM** (Small Language Model) to Hugging Face with:

- **Automatic model card generation** with comprehensive documentation
- **Complete installation instructions** for users
- **Dataset information** and training details
- **Usage examples** and troubleshooting guides
- **Professional presentation** for the community
- **Updated model** with 30K training steps and improved performance

## **Checkpoint Format Clarification**

**During Training:** All checkpoints are saved in PyTorch format (`.pt` files)
- Use `python 04_training/test_slm_checkpoint.py` to test during training
- Use `python 04_training/test_checkpoint.py` for full model testing

**After Publishing:** Models are available in Hugging Face format
- Use `python 06_inference/test_published_model.py` to test published models
- Hugging Face format is only created when publishing to the Hub

## **Dataset Information**

### **Training Data Overview**
The London Historical LLM was trained on a comprehensive collection of historical English texts spanning **1500-1850**, making it one of the most extensive historical language datasets available. The dataset includes **218+ sources** with **500M+ characters** of authentic historical content.

### **Dataset Composition**

| **Period** | **Sources** | **Key Authors** | **Content Types** |
|------------|-------------|-----------------|-------------------|
| **Early Modern (1500-1600)** | 18 | Thomas Harman, Hugh Latimer, Sir Thomas More | Street literature, civic docs, religious texts, trade guides |
| **Georgian (1700-1800)** | 50+ | Jane Austen, Mary Shelley, Walter Scott | Novels, poetry, political works, scientific texts |
| **Victorian (1800-1850)** | 50+ | Charles Dickens, Brontë Sisters, Lord Byron | Social novels, Romantic poetry, political treatises |

### **Source Categories**
- **Literature (80+ sources)**: Complete Austen collection, major Dickens works, Brontë sisters, Romantic poetry
- **Non-Fiction (60+ sources)**: Political treatises, economic texts, scientific works, religious sermons
- **Periodicals (25+ sources)**: The Times, Edinburgh Review, Punch, specialized magazines
- **Legal Documents (15+ sources)**: Acts of Parliament, city charters, legal treatises
- **Personal Accounts (20+ sources)**: Diaries, letters, memoirs from historical figures

## **Prerequisites**

- Training completed (final model ready)
- Model evaluation completed
- Hugging Face account created
- Model repository created on Hugging Face

## **Setup (First Time Only)**

### 1. Create Hugging Face Account
1. Go to [huggingface.co](https://huggingface.co)
2. Click "Sign Up" and create your account
3. Verify your email address
4. Complete your profile (optional but recommended)

### 2. Install Hugging Face Hub
```bash
# On your remote machine
pip install huggingface_hub
```

### 3. Login to Hugging Face
```bash
# This will open a browser for authentication
huggingface-cli login
```
- Follow the prompts to authenticate
- This creates a token stored locally for future use

### 4. Create Model Repository
1. Go to [huggingface.co/new](https://huggingface.co/new)
2. Choose **"Model"** (not Dataset)
3. **Repository name**: `london-historical-llm` (or your choice)
4. **Visibility**: **Public** (recommended for visibility and sharing)
5. **Description**: "A custom GPT-2 model trained on London historical texts from 1500-1850"
6. Click **"Create repository"**

## **What Gets Published**

The publishing script automatically uploads:

### Model Files
- **`model.safetensors`** - Model weights (optimized format)
- **`config.json`** - Model configuration
- **`generation_config.json`** - Text generation settings

### Tokenizer Files
- **`tokenizer.json`** - Custom tokenizer with 50k vocabulary
- **`tokenizer_config.json`** - Tokenizer configuration
- **`special_tokens_map.json`** - Special tokens mapping
- **`vocab.txt`** - Vocabulary file

### Training Files
- **`training_args.bin`** - Training arguments and settings
- **`training_statistics.json`** - Training metrics and statistics

### Documentation
- **`README.md`** - Auto-generated model card
- **Model description** - Comprehensive model information
- **Usage examples** - Code examples for loading and using the model

## **Publishing Steps**

### 1. Ensure Training is Complete
```bash
# Check if training is finished
ls -la 09_models/checkpoints/
# Should see the final checkpoint

# Test the final model (PyTorch format during training)
python3 04_training/test_checkpoint.py --checkpoint checkpoint-131384.pt

# After publishing, test the Hugging Face model
python3 06_inference/test_published_model.py --model_name "bahree/london-historical-llm"
```

### 2. Run the Publishing Script
```bash
# Navigate to project root
cd ~/src/helloLondon

# Publish the SLM model (recommended)
python3 10_scripts/publish_slm_to_huggingface.py

# Or publish the full model (if available)
python3 10_scripts/publish_to_huggingface.py
```

### 3. Verify Upload
1. Go to your model repository: `https://huggingface.co/bahree/london-historical-llm`
2. Verify all files are uploaded
3. Check the README.md is properly formatted
4. Test the model card display

## **Using Your Published Model**

### Loading the Model
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load your published model
model_name = "bahree/london-historical-llm"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Generate text
input_text = "In 18th century London,"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100, do_sample=True)
print(tokenizer.decode(outputs[0]))
```


## **Updating Your Model**

### Re-publishing After Changes
```bash
# Make changes to your model
# ... edit files ...

# Re-run the publishing script
python3 10_scripts/publish_to_huggingface.py
```

### Version Control
- Each upload creates a new version
- Previous versions remain available
- Users can specify which version to use

## **Model Card Information**

The auto-generated model card includes:

### Model Card Template for Future Publishing

When publishing future models, ensure the model card includes:

```markdown
---
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
# No base_model field - indicates trained from scratch
---

# Model Name

**Trained from scratch** using [architecture] on [dataset description].

## Key Features
- **Trained from scratch** - not fine-tuned from existing models
- [Other features...]

## Repository

The complete source code, training scripts, and documentation for this model are available on GitHub:

**🔗 [https://github.com/bahree/helloLondon](https://github.com/bahree/helloLondon)**

This repository includes:
- Complete data collection pipeline for 1500-1850 historical English
- Custom tokenizer optimized for historical text
- Training infrastructure with GPU optimization
- Evaluation and deployment tools
- Comprehensive documentation and examples

### Quick Start with Repository
```bash
git clone https://github.com/bahree/helloLondon.git
cd helloLondon
python 06_inference/test_published_models.py --model_type [slm/regular]
```

## Training Details
- **Training Method**: **Trained from scratch** - not fine-tuned
- [Other details...]
```

### Model Description
- **Architecture**: GPT-2 Small (117M parameters) - SLM version
- **Training Method**: **Trained from scratch** - not fine-tuned from existing models
- **Training Data**: London historical texts (1500-1850) with proper segmentation
- **Vocabulary**: 30,000 tokens with historical special tokens
- **Training Steps**: 30,000 steps with multi-GPU training
- **Final Training Loss**: 1.395 (43% improvement from 20K steps)
- **Model Flops Utilization**: 3.5% (excellent efficiency)

### Usage Examples
- Basic text generation
- Historical text completion
- London-specific queries

### Training Details
- **Training Method**: **Trained from scratch** - not fine-tuned
- **Data Sources**: 99+ historical sources with proper segmentation
- **Training Hardware**: Multi-GPU setup with Distributed Data Parallel
- **Evaluation Metrics**: Loss, perplexity, and custom tests
- **Final Performance**: 1.395 training loss, 3.5% MFU efficiency

### Limitations
- **Time Period**: Limited to 1500-1850 historical texts
- **Geographic Focus**: Primarily London-centric content
- **Language**: Historical English variants

## **Troubleshooting**

### Common Issues

#### "Finetuned" Label Issue
**Problem**: Hugging Face automatically shows "Finetuned" even for models trained from scratch.

**Solution**: 
1. **Update Model Card**: Add "**Trained from scratch**" in multiple places:
   - Model Description section
   - Key Features section  
   - Training Details section
2. **YAML Metadata**: Do NOT include `base_model` field for models trained from scratch
3. **Clarify in README**: Make it clear the model is not fine-tuned

#### Weight Naming Convention Issues
**Problem**: Model weights have `_orig_mod.` prefix that Hugging Face doesn't recognize.

**Solution**: 
1. **Rename weights** in publishing script: `_orig_mod.transformer.wte.weight` → `transformer.wte.weight`
2. **Transpose specific layers** for Hugging Face format (c_attn, c_proj, c_fc, mlp.c_proj)
3. **Make tensors contiguous** after transposition to avoid safetensors errors

#### Context Length Mismatch
**Problem**: Model has 256 context length but config shows 512, causing shape mismatch.

**Solution**:
1. **Auto-detect context length** from checkpoint's positional embedding shape
2. **Set n_positions and n_ctx** to match actual training configuration
3. **Update generation config** to use correct max_length

#### "No files have been modified" Error
**Problem**: Hugging Face skips upload because files appear unchanged.

**Solutions**:
1. **Create new repository** with version number (e.g., `london-historical-slm-v2`)
2. **Manual upload** via Hugging Face website
3. **Force upload** by adding timestamp file (advanced)

#### Authentication Error
```bash
# Re-login to Hugging Face
huggingface-cli login
```

#### Upload Permission Error
- Ensure you're logged in with the correct account
- Check repository permissions
- Verify repository name matches your account

#### Model Loading Error
- Check all required files are uploaded
- Verify model configuration is correct
- Test with a simple example first

### Getting Help
1. Check the [Hugging Face documentation](https://huggingface.co/docs)
2. Review the [Transformers library docs](https://huggingface.co/docs/transformers)
3. Check the model repository for issues
4. Contact Hugging Face support if needed

## **Best Practices**

### Before Publishing
- Test the model thoroughly
- Complete all evaluations
- Document any limitations
- Prepare usage examples

### After Publishing
- Monitor model usage
- Respond to user feedback
- Update documentation as needed
- Consider model improvements

### Repository Management
- Keep README.md updated
- Add tags for discoverability
- Respond to community questions
- Share updates and improvements

## **Sharing Your Model**

### Social Media
- Share on Twitter, LinkedIn, Reddit
- Use hashtags: #HuggingFace #LLM #HistoricalAI #London
- Include model link and examples

### Academic Sharing
- Cite in research papers
- Share with academic colleagues
- Present at conferences

### Community Engagement
- Answer questions on Hugging Face
- Share improvements and updates
- Collaborate with other researchers

## **Useful Links**

- [Hugging Face Hub](https://huggingface.co)
- [Model Repository](https://huggingface.co/bahree/london-historical-llm)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Model Card Guide](https://huggingface.co/docs/hub/model-cards)

---

## **Success Story - London Historical SLM**

**Status**: Successfully published and tested!

### **What We Accomplished:**
- **Trained from scratch** GPT-2 Small model (108M parameters)
- **60,000 training steps** with 2.74 final training loss, 3.44 validation loss
- **Custom historical tokenizer** (30k vocab) for 1500-1850 English
- **Multi-GPU training** with 8.25% MFU efficiency on A30s
- **Successfully published** to [Hugging Face Hub](https://huggingface.co/bahree/london-historical-slm)
- **Tested and verified** working on multiple machines

### **Final Training Results:**
- **Final Training Loss**: 2.7437 (excellent convergence)
- **Final Validation Loss**: 3.4409 (good generalization)
- **Validation Perplexity**: 31.21 (reasonable for 108M model)
- **Training Progress**: 60,000 steps completed successfully
- **Model Parameters**: 108,882,432 (108M parameters)

### **Key Technical Solutions:**
- **Weight naming**: Fixed `_orig_mod.` prefix issues from torch.compile
- **Tensor shapes**: Transposed layers for Hugging Face compatibility
- **Context length**: Auto-detected 1024 tokens from checkpoint
- **Model config**: Proper GPT-2 configuration with transformers version
- **Warning suppression**: Clean user experience without confusing warnings

### **Model Performance:**
- **Generates coherent historical text** with period-appropriate language
- **Loads cleanly** without weight mismatch warnings
- **Ready for production use** worldwide
- **Historical accuracy**: Captures 1500-1850 English patterns effectively

## **Testing Your Published Model**

### **Quick Testing (Recommended First)**
```bash
# Test the published model with 10 automated prompts
python 06_inference/test_published_models.py --model_type slm
```

**What this does:**
- Loads model from `bahree/london-historical-slm`
- Tests 10 historical prompts automatically
- Shows model info (vocab size, parameters, etc.)
- Uses SLM-optimized generation parameters
- **No user interaction** - just runs and reports results

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

### **Interactive Testing (For Exploration)**
```bash
# Interactive mode for custom prompts
python 06_inference/inference_unified.py --published --model_type slm --interactive

# Single prompt test
python 06_inference/inference_unified.py --published --model_type slm --prompt "In the year 1834, I walked through the streets of London and witnessed"
```

**What this does:**
- **Interactive mode** - you type prompts and get responses
- **Customizable parameters** (temperature, max_length, etc.)
- **Command-line interface** for single prompts
- **Flexible generation settings**
- Good for **exploration** and **custom testing**

### **Python Code Testing**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load your published model
model_name = "bahree/london-historical-slm"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Generate text
prompt = "In the year 1834, I walked through the streets of London and witnessed"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    inputs["input_ids"],
    max_new_tokens=50,
    do_sample=True,
    temperature=0.3,
    top_p=0.9,
    top_k=20
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

> **💡 Generation Parameters Note**: The example above uses conservative parameters (`temperature=0.3`, `top_k=20`, `top_p=0.9`) that produce high-quality, coherent historical text. These are the recommended settings for best results.

### **Expected Output Quality**
Your model generates text like:
> "In the year 1834, I walked through the streets of London and witnessed the most extraordinary sight. The Thames flowed dark beneath London Bridge, whilst carriages rattled upon the cobblestones with great urgency. Merchants called their wares from Cheapside to Billingsgate, and the smoke from countless chimneys did obscure the morning sun."

**Key Features:**
- **Period-appropriate language** ("whilst", "did obscure")
- **London-specific geography** (Thames, London Bridge, Cheapside, Billingsgate)
- **Historical context** (carriages, cobblestones, chimney smoke)
- **Coherent narrative flow**

**Congratulations!** Your London Historical LLM is now available to the world!

**Next Steps:**
1. Share your model with the community
2. Monitor usage and feedback
3. Consider improvements and updates
4. Explore deployment options (custom API, etc.)
