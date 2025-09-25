# 🤖 Synthetic Data Generation Guide

This guide explains how to generate additional historical London data using local or cloud LLMs to enhance your training dataset.

## 🎯 Overview

Synthetic data generation can significantly improve your model by:
- **Expanding Dataset Size**: Generate thousands of additional historical documents
- **Filling Gaps**: Create content for underrepresented periods or document types
- **Improving Diversity**: Add varied writing styles and perspectives
- **Cost-Effective**: Use local models to avoid API costs
- **Controlled Quality**: Generate exactly what you need

## 🚀 Quick Start

### **Option 1: Local Generation (Recommended)**
```bash
# Setup Ollama (one-time)
python setup_ollama.py

# Generate synthetic data
python synthetic_data_generator.py --model_type ollama --num_documents 100
```

### **Option 2: Cloud Generation**
```bash
# OpenAI (requires API key)
export OPENAI_API_KEY="your-api-key"
python synthetic_data_generator.py --model_type openai --num_documents 100

# Anthropic (requires API key)
export ANTHROPIC_API_KEY="your-api-key"
python synthetic_data_generator.py --model_type anthropic --num_documents 100
```

## 📋 Prerequisites

### **For Local Generation (Ollama)**
- **RAM**: 8GB+ recommended (16GB+ for larger models)
- **Storage**: 10GB+ for models
- **Internet**: For initial model download

### **For Cloud Generation**
- **API Keys**: OpenAI or Anthropic account
- **Internet**: Stable connection required
- **Costs**: Pay per token generated

## 🔧 Setup Instructions

### **1. Install Ollama (Local)**

#### **Linux/macOS**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve
```

#### **Windows**
1. Download from [ollama.ai/download](https://ollama.ai/download)
2. Install and run Ollama
3. Open terminal and run: `ollama serve`

### **2. Setup Models**
```bash
# Run the setup script
python setup_ollama.py

# Or manually pull models
ollama pull phi3:latest
ollama pull llama3.1:latest
ollama pull mistral:latest
```

### **3. Test Installation**
```bash
# Test a model
ollama run phi3:latest "Write a short diary entry from 1750 about London"

# Should return generated text
```

## 📊 Generated Content Types

The system generates diverse historical documents:

### **Document Types**
- **Diary Entries**: Personal daily observations
- **Letters**: Correspondence between Londoners
- **Newspaper Articles**: Local news and events
- **Court Records**: Legal proceedings and trials
- **Merchant Accounts**: Business transactions and trade
- **Parish Registers**: Births, marriages, deaths
- **Travel Journals**: Visitors' observations
- **Political Pamphlets**: Political commentary
- **Sermons**: Religious texts
- **Play Scripts**: Theatrical works
- **Poems**: Literary works
- **Recipe Books**: Culinary instructions
- **Medical Treatises**: Health and medicine

### **Historical Periods**
- **1500-1600**: Tudor Period (Renaissance, Reformation)
- **1600-1700**: Stuart Period (Civil War, Restoration, Plague & Fire)
- **1700-1800**: Georgian Period (Enlightenment, Industrial Revolution)
- **1800-1850**: Regency/Victorian Early (Industrial Revolution, Social Reform)

### **London Locations**
- **Commercial**: Cheapside, Fleet Street, Covent Garden
- **Political**: Westminster, Whitehall
- **Religious**: St. Paul's, Westminster Abbey
- **Legal**: Old Bailey, Newgate, Tower Hill
- **Residential**: Southwark, Whitechapel, Spitalfields
- **Trade**: Billingsgate, Smithfield, London Bridge

## 🎛️ Configuration Options

### **Command Line Arguments**
```bash
python synthetic_data_generator.py \
    --num_documents 100 \          # Number of documents to generate
    --model_type ollama \          # Model type (ollama/openai/anthropic)
    --batch_size 5 \               # Concurrent generation batch size
    --output_dir custom/path       # Custom output directory
```

### **Model Settings**
```json
{
  "default_model": "phi3:latest",
  "generation_settings": {
    "temperature": 0.8,           # Creativity (0.0-1.0)
    "top_p": 0.9,                 # Diversity (0.0-1.0)
    "max_tokens": 1000,           # Maximum length
    "batch_size": 5,              # Concurrent requests
    "timeout_seconds": 60         # Request timeout
  }
}
```

## 📁 Output Structure

```
data/london_historical/synthetic/
├── synthetic_1750_diary_entry_001.txt
├── synthetic_1820_letter_002.txt
├── synthetic_1680_court_record_003.txt
├── ...
├── synthetic_metadata.json
└── synthetic_corpus.txt
```

### **Individual Files**
Each document includes:
- **Header**: Generation metadata, model info, year, type
- **Content**: Generated historical text
- **Format**: Clean, readable text ready for training

### **Metadata File**
```json
{
  "generation_info": {
    "model_type": "ollama",
    "model_name": "phi3:latest",
    "total_documents": 100,
    "total_words": 45000,
    "periods_covered": ["Tudor", "Stuart", "Georgian"],
    "document_types": ["diary_entry", "letter", "newspaper_article"],
    "generation_time": 120.5
  },
  "documents": [...]
}
```

### **Corpus File**
- **Combined**: All generated documents in one file
- **Formatted**: Ready for tokenizer training
- **Structured**: Clear document separators

## 🔄 Integration with Training

### **1. Generate Synthetic Data**
```bash
# Generate 500 documents
python synthetic_data_generator.py --num_documents 500 --model_type ollama
```

### **2. Combine with Real Data**
```bash
# The system automatically creates a corpus file
# You can also manually combine:
cat data/london_historical/london_historical_corpus.txt \
    data/london_historical/synthetic/synthetic_corpus.txt > \
    data/london_historical/combined_corpus.txt
```

### **3. Retrain Tokenizer**
```bash
# Retrain tokenizer with combined data
python 03_tokenizer/train_tokenizer.py --corpus_file data/london_historical/combined_corpus.txt
```

### **4. Continue Training**
```bash
# Continue training with enhanced dataset
python 04_training/train_model.py
```

## 📈 Quality Control

### **Automatic Quality Checks**
- **Length Validation**: Ensures appropriate document length
- **Content Filtering**: Removes empty or low-quality generations
- **Metadata Validation**: Verifies historical accuracy
- **Format Consistency**: Ensures proper document structure

### **Manual Quality Review**
```bash
# Review generated documents
ls -la data/london_historical/synthetic/
head -20 data/london_historical/synthetic/synthetic_1750_diary_entry_001.txt

# Check metadata
cat data/london_historical/synthetic/synthetic_metadata.json
```

### **Quality Metrics**
- **Word Count**: Average words per document
- **Diversity**: Number of unique document types
- **Coverage**: Historical periods represented
- **Consistency**: Format and style uniformity

## ⚡ Performance Optimization

### **Local Generation (Ollama)**
- **Batch Size**: 3-5 concurrent requests
- **Model Size**: Use smaller models for faster generation
- **Memory**: Monitor RAM usage with larger models
- **Storage**: Ensure sufficient disk space

### **Cloud Generation**
- **Rate Limits**: Respect API rate limits
- **Costs**: Monitor token usage and costs
- **Batch Size**: Smaller batches to avoid timeouts
- **Retry Logic**: Built-in retry for failed requests

## 🔧 Troubleshooting

### **Common Issues**

#### **Ollama Not Starting**
```bash
# Check if Ollama is running
ps aux | grep ollama

# Start Ollama service
ollama serve

# Check logs
journalctl -u ollama
```

#### **Model Download Fails**
```bash
# Check available models
ollama list

# Pull specific model
ollama pull phi3:latest

# Check disk space
df -h
```

#### **Empty Responses**
```bash
# Test model manually
ollama run phi3:latest "Write a short story"

# Check model status
ollama ps
```

#### **API Errors (Cloud)**
```bash
# Check API key
echo $OPENAI_API_KEY

# Test API connection
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/models
```

### **Getting Help**

1. **Check Logs**: `tail -f data/london_historical/synthetic/synthetic_generation.log`
2. **Test Models**: Use `ollama run model_name "test prompt"`
3. **Verify Setup**: Run `python setup_ollama.py`
4. **Check Resources**: Monitor RAM, disk space, network

## 📊 Expected Results

### **Generation Speed**
- **Local (Ollama)**: 10-50 documents per minute
- **Cloud (OpenAI)**: 20-100 documents per minute
- **Quality**: High-quality, historically accurate text

### **Dataset Enhancement**
- **Size Increase**: 2-10x original dataset size
- **Diversity**: More document types and periods
- **Quality**: Consistent, well-formatted text
- **Coverage**: Better representation of historical periods

## 🎯 Best Practices

### **1. Start Small**
```bash
# Test with small batch first
python synthetic_data_generator.py --num_documents 10
```

### **2. Monitor Quality**
- Review generated samples regularly
- Adjust prompts if needed
- Check for historical accuracy

### **3. Balance Real vs Synthetic**
- Use synthetic data to supplement, not replace
- Maintain high quality standards
- Monitor training performance

### **4. Iterative Improvement**
- Generate in batches
- Evaluate results
- Adjust parameters
- Generate more as needed

## 🔗 Related Documentation

- [Data Collection Guide](DATA_COLLECTION.md)
- [Text Cleaning Guide](TEXT_CLEANING_GUIDE.md)
- [Training Guide](TRAINING_GUIDE.md)

---

**Ready to enhance your historical dataset with synthetic data!** 🤖📚✨

**Next Steps:**
1. Setup Ollama or configure cloud APIs
2. Generate initial batch of synthetic data
3. Review quality and adjust parameters
4. Integrate with existing dataset
5. Continue training with enhanced data
