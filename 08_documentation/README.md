# 📚 helloLondon Documentation Index

Welcome to the **helloLondon** documentation! This folder contains comprehensive guides for building, training, and deploying historical language models.

> **🏠 Main Project**: For the complete project overview and quick start, see the [London Historical LLM README](README.md)

## 📝 **Blog Series**
- [Building LLMs from Scratch - Part 1](https://blog.desigeek.com/post/2025/09/building-llm-from-scratch-part1/) - Quick start and overview
- [Building LLMs from Scratch - Part 2](https://blog.desigeek.com/post/2025/10/building-llm-from-scratch-part2-data-tokenizers/) - Data Collection & Custom Tokenizers
- [Building LLMs from Scratch - Part 3](https://blog.desigeek.com/post/2025/11/building-llm-from-scratch-part3-model-architecture-gpu-training/) - Training Architecture & GPU Optimization

## 🚀 **Quick Navigation**

| **I Want To...** | **Start Here** | **Time** |
|------------------|----------------|----------|
| **Use the published model** | [Inference Quick Start](INFERENCE_QUICK_START.md) | 2 minutes |
| **Train my own model** | [Training Quick Start](TRAINING_QUICK_START.md) | 15 minutes |
| **Understand the system** | [Complete Technical Guide](COMPLETE_TECHNICAL_GUIDE.md) | 30 minutes |
| **Fix any issues** | [Troubleshooting Guide](TROUBLESHOOTING.md) | 5 minutes |
| **Publish a model** | [Hugging Face Publishing](HUGGINGFACE_PUBLISHING.md) | 10 minutes |

## 📖 **Getting Started**

### **Essential Guides (Start Here)**
- **[Inference Quick Start](INFERENCE_QUICK_START.md)** - **Start here!** Use the published model in 2 minutes
- **[Training Quick Start](TRAINING_QUICK_START.md)** - **Want to train?** Get training up and running in 15 minutes
- **[Training Guide](TRAINING_GUIDE.md)** - Complete training for both model variants
- **[Inference Setup](INFERENCE_SETUP_GUIDE.md)** - Deploy and use your trained models

### **Model Variants**
- **SLM (117M parameters)**: Fast, efficient model for quick inference and experimentation
- **Regular Model (354M parameters)**: Larger model with deeper understanding for complex historical analysis
- **Training Time**: 7-8 hours (SLM) / 10-12 hours (Regular) on dual GPU (60,000 iterations each)
- **MFU Targets**: 8-9% (SLM) / 15-20% (Regular) on A30 GPUs

## 📊 **Data & Processing**

### **Data Collection & Management**
- **[Data Collection](DATA_COLLECTION.md)** - Download and process historical data from 218+ sources
- **[London Lives Processing](LONDON_LIVES_PROCESSING.md)** - Specific guide for London Lives data
- **[Old Bailey Processing](OLD_BAILEY_PROCESSING.md)** - Old Bailey court records processing (disabled by default)
- **[Synthetic Data](SYNTHETIC_DATA_GUIDE.md)** - Generate additional training data
- **[Text Cleaning Process](CLEANING_PROCESS_FLOW.md)** - Complete cleaning pipeline implementation

### **Tokenization & Analysis**
- **[Token Counting](TOKEN_COUNTING_GUIDE.md)** - Analyze dataset size and statistics
- **[Tokenizer Vocabulary](TOKENIZER_VOCABULARY.md)** - Custom tokenizer details and special tokens

### **Dataset Overview**
- **Total Sources**: 218+ historical texts (1500-1850)
- **Data Volume**: 500M+ characters of authentic historical English
- **Geographic Focus**: London and England
- **Time Period**: Tudor, Stuart, Georgian, and Victorian eras
- **Text Types**: Literature, legal, scientific, personal, and religious texts

## 🏗️ **Model Architecture & Training**

### **Configuration & Setup**
- **[Global Configuration](GLOBAL_CONFIGURATION.md)** - Centralized configuration system
- **[WandB Setup](WANDB_SETUP.md)** - Experiment tracking and monitoring

### **Training Monitoring**
- 📊 **Real-time metrics** - loss, learning rate, GPU utilization, training speed
- 🔍 **5 key panels** - comprehensive training progress visualization in WandB
- ⚠️ **Red flag detection** - automatic identification of training issues
- 📈 **Performance tracking** - MFU, step times, and hardware efficiency

### **GPU Optimization**
- **[GPU Tuning Guide](GPU_TUNING.md)** - Precision, TF32, batch/sequence sizing per GPU
- **[GPU Troubleshooting](GPU_TROUBLESHOOTING.md)** - Fix GPU issues and driver problems

### **Model Architecture Details**
- **SLM Architecture**: 12 layers, 12 heads, 768 embeddings, 512 context
- **Regular Model Architecture**: 24 layers, 16 heads, 1024 embeddings, 1024 context
- **Custom GPT Implementation**: nanoGPT-style architecture optimized for historical text
- **Training Infrastructure**: Modern training code with DDP, checkpointing, and WandB integration

## 🧪 **Evaluation & Testing**

**📚 Evaluation Documentation Hierarchy** (start with #1):

1. **[Evaluation Quick Reference](EVALUATION_QUICK_REFERENCE.md)** - **Start here!** Quick commands and metrics
2. **[Evaluation Guide](EVALUATION_GUIDE.md)** - **Complete manual** - How to implement it?

### **Evaluation Capabilities**
- **Historical Accuracy**: Period-appropriate language and factual consistency
- **Linguistic Quality**: Grammatical correctness and coherence
- **Performance Metrics**: ROUGE, BLEU, perplexity, and custom historical tests
- **Model Comparison**: SLM vs Regular Model performance analysis

## 🚀 **Deployment & Publishing**

### **Model Publishing**
- **[Hugging Face Publishing](HUGGINGFACE_PUBLISHING.md)** - Publish models to Hugging Face
- **[Deployment Guide](DEPLOYMENT_GUIDE.md)** - Inference deployment options

### **Deployment Options**
- **Hugging Face Hub**: Published models ready for download
- **Local Inference**: Command-line and API deployment
- **Docker Support**: Containerized deployment options

## 🔧 **Troubleshooting & Support**

### **Problem Solving**
- **[Troubleshooting Guide](TROUBLESHOOTING.md)** - **Complete troubleshooting solutions** for all common issues
- **[GPU Troubleshooting](GPU_TROUBLESHOOTING.md)** - GPU-specific issues and optimization
- **[GPU Tuning](GPU_TUNING.md)** - Performance optimization and hardware tuning

### **Common Issue Categories**
- **Environment & Setup** - Virtual environment, path problems, activation failures
- **Data Collection** - Failed downloads, data quality, cleaning issues
- **Tokenizer Issues** - Vocabulary mismatches, WordPiece artifacts, consistency
- **Model Training** - OOM errors, training interruptions, performance problems
- **Model Loading** - Hugging Face errors, checkpoint loading, generation issues
- **Evaluation** - Script problems, missing dependencies, performance issues
- **Multi-Machine** - Git sync, environment differences, consistency issues

## 🔬 **Technical Deep Dives**

### **Advanced Topics**
- **[Complete Technical Guide](COMPLETE_TECHNICAL_GUIDE.md)** - Comprehensive technical overview
- **[GitHub README Section](GITHUB_README_SECTION.md)** - GitHub-specific documentation snippets

### **System Architecture**
- **Custom GPT Implementation**: nanoGPT-style architecture with historical optimizations
- **Distributed Training**: Multi-GPU support with DDP and automatic GPU detection
- **Memory Management**: Optimized for different GPU types and memory constraints
- **Checkpoint System**: Robust checkpointing with automatic cleanup and resume functionality

## ⚡ **Quick Reference**

### **Most Common Tasks**
```bash
# Quick setup
python 01_environment/setup_environment.py

# Download data
python 02_data_collection/historical_data_collector.py

# Train tokenizer
python 03_tokenizer/train_historical_tokenizer.py

# Train SLM model
python 04_training/train_model_slm.py

# Train Regular model
python 04_training/train_model.py

# Quick evaluation
python 05_evaluation/run_evaluation.py --mode quick
```

### **Key Configuration**
- **Vocabulary Size**: 30,000 tokens with 150+ historical special tokens
- **Model Variants**: SLM (117M) and Regular (354M) parameters
- **Training Time**: 7-8 hours (SLM) / 10-12 hours (Regular) on dual GPU (60,000 iterations each)
- **Data Sources**: 218+ historical sources (1500-1850)
- **MFU Targets**: 8-9% (SLM) / 15-20% (Regular) on A30 GPUs

## 📋 **Documentation Structure**

This documentation is organized by workflow:

1. **🚀 Getting Started** → Quick setup and first steps
2. **📊 Data & Processing** → Data collection, cleaning, and preparation
3. **🏗️ Model Architecture** → Training configuration and optimization
4. **🧪 Evaluation** → Testing and validation
5. **🚀 Deployment** → Publishing and educational use
6. **🔬 Technical Deep Dives** → Advanced topics and implementation details

## 🤝 **Contributing to Documentation**

When adding new documentation:
1. **Choose the right category** above
2. **Follow the naming convention**: `TOPIC_GUIDE.md` or `TOPIC_OVERVIEW.md`
3. **Add cross-references** to related guides
4. **Update this index** when adding new files
5. **Test all code examples** before committing

## 📈 **Recent Updates**

- **✅ Both Model Variants**: Complete support for SLM (117M) and Regular (354M) models
- **✅ Unified Inference**: Both PyTorch checkpoint and Hugging Face inference working perfectly
- **✅ MFU Optimization**: Updated GPU tuning with model-specific MFU targets
- **✅ Modern Training Code**: Ported all SLM features to regular model training
- **✅ Comprehensive Documentation**: Updated all guides to reflect recent changes

## 🔗 **Related Resources**

- **Book Reference**: [Generative AI in Action](https://a.co/d/ffzkJ7T) by Amit Bahree

---
