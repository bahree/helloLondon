# Token Counting Guide

> **📚 Note: Want to understand the core concepts?** This project focuses on implementation and hands-on building. For deeper understanding of foundational concepts like tokenizers, prompt engineering, RAG, responsible AI, fine-tuning, and more, check out [**Generative AI in Action**](https://a.co/d/ffzkJ7T) by Amit Bahree. [Learn more about the book →](https://blog.desigeek.com/post/2024/10/book-release-genai-in-action/)

> **📚 Related Guide**: To understand the custom tokenizer design and special tokens used for counting, see [**TOKENIZER_VOCABULARY.md**](TOKENIZER_VOCABULARY.md) - the companion guide explaining the 30,000 vocabulary and 150+ historical special tokens.

## 🔢 **Overview**

The token counting system helps you analyze your dataset size and understand the scope of your training data. This is crucial for:
- **Training planning** - Estimating training time and memory requirements
- **Data quality assessment** - Understanding the impact of text cleaning
- **Model performance prediction** - Larger datasets generally lead to better models
- **Resource allocation** - Planning GPU memory and storage needs

## 🚀 **Quick Start**

### **Remote Machine Usage (Most Common)**
```bash
# On your remote Ubuntu machine with GPU
cd ~/src/helloLondon

# Count tokens in processed data
python 02_data_collection/count_tokens.py --data_dir ~/src/helloLondon/data/london_historical/processed

# Count tokens in the main corpus file
mkdir -p ~/temp_corpus_counting
cp ~/src/helloLondon/data/london_historical/london_historical_corpus_comprehensive.txt ~/temp_corpus_counting/
python 02_data_collection/count_tokens.py --data_dir ~/temp_corpus_counting
```

### **Basic Token Counting**
```bash
cd 02_data_collection

# Count tokens in cleaned data (uses config paths automatically)
python count_tokens.py

# Count tokens in processed data (most common use case)
python count_tokens.py --data_dir ~/src/helloLondon/data/london_historical/processed

# Or specify other custom paths
python count_tokens.py --data_dir data/london_historical/cleaned
```

### **Compare Raw vs Cleaned Data**

**Option 1: Automatic Comparison (Recommended)**
```bash
# Compare raw vs cleaned data automatically
python compare_token_counts.py
```

**Option 2: Manual Comparison**
```bash
# Count raw data (before cleaning)
python count_tokens.py --data_dir data/london_historical --output_file raw_tokens.json

# Count cleaned data (after cleaning)
python count_tokens.py --data_dir data/london_historical/cleaned --output_file cleaned_tokens.json
```

## 📊 **Understanding the Output**

### **Single Dataset Statistics**
```
📊 TOKEN COUNT SUMMARY
==================================================
📁 Total files processed: 1,247
🔢 Total tokens: 54,356,360
📏 Average tokens per file: 43,580
💾 Total data size: 89.2 MB
📈 Tokens per MB: 609,377

🎓 Training Data Estimates:
   Training tokens: 48,920,724
   Evaluation tokens: 5,435,636
```

### **Comparison Statistics (Raw vs Cleaned)**
```
📊 TOKEN COUNT COMPARISON SUMMARY
============================================================

📁 FILES:
   Raw files: 1,247
   Cleaned files: 1,247
   Difference: +0

🔢 TOKENS:
   Raw tokens: 52,891,234
   Cleaned tokens: 54,356,360
   Difference: +1,465,126
   Improvement: +2.8%

📏 AVERAGE TOKENS PER FILE:
   Raw: 42,415
   Cleaned: 43,580
   Difference: +1,165

💾 FILE SIZE:
   Raw size: 91.3 MB
   Cleaned size: 89.2 MB
   Difference: -2.1 MB
   Size change: -2.3%

📈 EFFICIENCY:
   Raw tokens/MB: 579,312
   Cleaned tokens/MB: 609,377
   Efficiency improvement: +5.2%

✅ QUALITY ASSESSMENT:
   ✅ Token count increased by 2.8% - cleaning improved data quality
   ✅ File size reduced by 2.3% - removed unnecessary content
```

### **Key Metrics Explained**
- **Total tokens**: Your dataset size in tokens (most important metric)
- **Average tokens per file**: Helps identify very large or small files
- **Tokens per MB**: Efficiency of your tokenizer
- **Training/Evaluation split**: 90/10 split for model training

## 🔧 **Advanced Usage**

### **Custom Tokenizer**
```bash
# Use a different tokenizer
python count_tokens.py \
    --data_dir data/london_historical/cleaned \
    --tokenizer_dir 09_models/tokenizers/custom_tokenizer
```

### **Detailed Analysis**
```bash
# Generate detailed per-file report
python count_tokens.py \
    --data_dir data/london_historical/cleaned \
    --output_file detailed_analysis.json
```

## 📈 **Data Quality Analysis**

### **Before vs After Cleaning**
```bash
# Run on both datasets and compare
echo "=== RAW DATA ==="
python count_tokens.py --data_dir data/london_historical --output_file raw_analysis.json

echo "=== CLEANED DATA ==="
python count_tokens.py --data_dir data/london_historical/cleaned --output_file cleaned_analysis.json

# Compare the results
echo "Token count improvement:"
python -c "
import json
with open('raw_analysis.json') as f: raw = json.load(f)
with open('cleaned_analysis.json') as f: clean = json.load(f)
raw_tokens = raw['summary']['total_tokens']
clean_tokens = clean['summary']['total_tokens']
print(f'Raw: {raw_tokens:,} tokens')
print(f'Cleaned: {clean_tokens:,} tokens')
print(f'Difference: {clean_tokens - raw_tokens:,} tokens')
print(f'Improvement: {((clean_tokens - raw_tokens) / raw_tokens * 100):.1f}%')
"
```

## 🎯 **Training Planning**

### **Model Size Recommendations**
Based on your token count:

| Token Count | Recommended Model | Training Time* | Memory |
|-------------|------------------|---------------|---------|
| < 10M | GPT-2 Small (117M) | 3-4 hours | 8GB |
| 10M - 50M | GPT-2 Medium (355M) | 7-8 hours | 16GB |
| 50M - 100M | GPT-2 Large (774M) | 10-12 hours | 24GB |
| > 100M | GPT-2 XL (1.5B) | 16+ hours | 32GB+ |

*Times based on dual GPU training (2x A30 GPUs). Single GPU will take ~2x longer.

### **Training Time Estimation**
```bash
# Estimate training time based on tokens
python -c "
tokens = 54356360  # Your token count
model_params = 355000000  # GPT-2 Medium
gpu_memory = 24  # GB
batch_size = 4

# Rough estimation (varies by hardware)
steps_per_epoch = tokens // (batch_size * 512)  # 512 = sequence length
total_steps = steps_per_epoch * 3  # 3 epochs
hours = total_steps * 0.2  # ~0.2 seconds per step (more realistic)

print(f'Estimated training time: {hours:.1f} hours (dual GPU)')
print(f'Single GPU estimate: {hours * 2:.1f} hours')
print(f'Steps per epoch: {steps_per_epoch:,}')
print(f'Total steps: {total_steps:,}')
"
```

## 📋 **Best Practices**

### **1. Count After Each Step**
```bash
# After data download
python count_tokens.py --data_dir data/london_historical --output_file step1_raw.json

# After text cleaning
python count_tokens.py --data_dir data/london_historical/cleaned --output_file step2_cleaned.json

# After tokenizer training
python count_tokens.py --data_dir data/london_historical/cleaned --output_file step3_final.json
```

### **2. Monitor Data Quality**
- **Token count should increase** after cleaning (removing artifacts)
- **Average tokens per file** should be consistent
- **Very large files** might need splitting
- **Very small files** might be low quality

### **3. Save Reports**
```bash
# Create timestamped reports
timestamp=$(date +%Y%m%d_%H%M%S)
python count_tokens.py \
    --data_dir data/london_historical/cleaned \
    --output_file "token_analysis_${timestamp}.json"
```

## 🔄 **How This Guide Works with TOKENIZER_VOCABULARY.md**

### **Workflow Integration**
```
Data Collection → Text Cleaning → [TOKEN_COUNTING_GUIDE] → Tokenizer Training → [TOKENIZER_VOCABULARY] → Model Training
     ↓              ↓                    ↓                        ↓                    ↓
  Raw texts    Cleaned texts      Token counts           Custom tokenizer      Trained model
```

### **When to Use Each Guide**

| **Use TOKEN_COUNTING_GUIDE.md when:** | **Use TOKENIZER_VOCABULARY.md when:** |
|----------------------------------------|----------------------------------------|
| 📊 Measuring dataset size | 📚 Understanding tokenizer design |
| ⏱️ Planning training time | 🔍 Debugging tokenization issues |
| 💾 Estimating memory needs | 🎯 Understanding special tokens |
| ✅ Validating data quality | 🏗️ Designing custom vocabulary |
| 📈 Comparing before/after cleaning | 🔧 Optimizing tokenization efficiency |

### **Key Integration Points**
- **Vocabulary Size**: Both guides reference the same 30,000 token vocabulary
- **Special Tokens**: The 150+ special tokens (documented in TOKENIZER_VOCABULARY.md) directly impact counting efficiency
- **Training Planning**: Token counts (from this guide) + vocabulary complexity (from TOKENIZER_VOCABULARY.md) = training resource estimates

## 🚨 **Troubleshooting**

### **Common Issues**

**No files found:**
```bash
# Check if data directory exists
ls -la data/london_historical/cleaned/

# Check file extensions
find data/london_historical/cleaned -name "*.txt" | head -5
```

**Tokenizer not found:**
```bash
# Check tokenizer directory
ls -la 09_models/tokenizers/london_historical_tokenizer/

# Train tokenizer first
cd 03_tokenizer
python train_historical_tokenizer.py
```

**Memory issues:**
```bash
# Process files in batches (if implemented)
python count_tokens.py --data_dir data/london_historical/cleaned
```

## 📊 **Integration with Training**

### **Update Training Scripts**
The token count can be used to automatically configure training parameters:

```python
# In your training script
import json

# Load token count
with open('token_analysis.json') as f:
    analysis = json.load(f)

total_tokens = analysis['summary']['total_tokens']

# Adjust training parameters based on dataset size
if total_tokens < 10000000:
    batch_size = 8
    learning_rate = 5e-4
elif total_tokens < 50000000:
    batch_size = 4
    learning_rate = 3e-4
else:
    batch_size = 2
    learning_rate = 1e-4
```

## **Next Steps**

After counting tokens:
1. **Review the statistics** - Ensure data quality looks good
2. **Plan training parameters** - Adjust based on dataset size
3. **Train tokenizer** - Use the cleaned data
4. **Start training** - With appropriate parameters
5. **Monitor progress** - Use WandB to track training

---
