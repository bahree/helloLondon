# 🔧 Troubleshooting Guide

*Comprehensive solutions for common issues encountered during London Historical LLM development*

## 📋 Table of Contents

- [Environment & Setup Issues](#environment--setup-issues)
- [Data Collection & Processing](#data-collection--processing)
- [Tokenizer Issues](#tokenizer-issues)
- [Model Training Issues](#model-training-issues)
- [Model Loading & Inference](#model-loading--inference)
- [Evaluation Issues](#evaluation-issues)
- [Performance Issues](#performance-issues)
- [Multi-Machine Setup](#multi-machine-setup)
- [Quick Fixes Reference](#quick-fixes-reference)
- [Still Need Help?](#still-need-help)

---

## Environment & Setup Issues

### Virtual Environment Path Problems

**Error**: `ModuleNotFoundError: No module named 'torch'` after activation

**Symptoms**:
- Virtual environment activates but Python still points to system Python
- `which python3` shows `/usr/bin/python3` instead of virtual environment path
- `echo $VIRTUAL_ENV` shows old project path

**Cause**: Virtual environment paths not updated after moving project directory

**Solution**:
```bash
# Remove old virtual environment
rm -rf helloLondon/

# Recreate with correct paths
python3 01_environment/setup_environment.py
source activate_env.sh

# Verify activation
which python3  # Should show project path
python3 -c "import torch; print('PyTorch version:', torch.__version__)"
```

### Environment Activation Failures

**Error**: `-bash: /home/user/src/helloLondon/helloLondon/bin/activate: No such file or directory`

**Cause**: Hardcoded paths in `activate_env.sh` script

**Solution**: The `activate_env.sh` script has been updated to use dynamic paths. If you still see this error:
```bash
# Check if script uses dynamic paths
head -5 activate_env.sh
# Should show: SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# If not, update the script or recreate environment
python3 01_environment/setup_environment.py
```

---

## Data Collection & Processing

### Failed Downloads

**Error**: Multiple 404, 403, or timeout errors during data collection

**Symptoms**:
- `historical_data_collector.py` shows many failed downloads
- Incomplete dataset after collection
- Network-related error messages

**Solutions**:
```bash
# Check failed downloads
python3 02_data_collection/check_failed_downloads.py

# Generate detailed report
python3 02_data_collection/generate_report.py --format csv

# Retry failed downloads
python3 02_data_collection/historical_data_collector.py --retry-failed

# Reduce sources for testing
python3 02_data_collection/historical_data_collector.py --max_sources 50
```

### Data Quality Issues

**Error**: Poor quality text in final corpus

**Symptoms**:
- High rejection rates during cleaning
- OCR artifacts in generated text
- Inconsistent text formatting

**Solutions**:
```bash
# Check cleaning statistics
python3 02_data_collection/generate_report.py

# Review rejected files
cat logs/data_collection.log | grep "REJECTED"

# Adjust quality thresholds in cleaning pipeline
# Edit 02_data_collection/historical_data_collector.py
```

---

## Tokenizer Issues

### Vocabulary Size Mismatch

**Error**: `CUDA error: device-side assert triggered` or `index out of range in self`

**Symptoms**:
- Model loads but generates gibberish
- CUDA errors during text generation
- Vocabulary size mismatch between model and tokenizer

**Cause**: Tokenizer vocabulary size doesn't match model's expected vocabulary size

**Solution**:
```bash
# Check tokenizer vocabulary size
python3 -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('09_models/tokenizers/london_historical_tokenizer')
print(f'Tokenizer vocab size: {tokenizer.vocab_size}')
"

# Retrain tokenizer with correct vocabulary size
python3 03_tokenizer/train_historical_tokenizer.py --vocab_size 30000

# Verify both model and tokenizer use 30,000 tokens
```

### WordPiece Artifacts in Generated Text

**Error**: `##` symbols appearing in generated text

**Symptoms**:
- Generated text contains `##` symbols
- Text looks fragmented or unnatural
- Poor text reconstruction quality

**Cause**: WordPiece-style tokenization artifacts from incorrect tokenizer configuration

**Solution**:
```bash
# The tokenizer training script has been updated to remove ## artifacts
python3 03_tokenizer/train_historical_tokenizer.py

# Test tokenizer reconstruction
python3 03_tokenizer/test_tokenizer.py

# Verify clean text generation
python3 04_training/test_slm_checkpoint.py --checkpoint checkpoint-60001.pt
```

### Tokenizer Consistency Across Machines

**Error**: Different tokenizers on different machines causing generation issues

**Symptoms**:
- Same model generates different text on different machines
- Inconsistent tokenization results
- Model works on one machine but not another

**Cause**: Tokenizers trained separately with different configurations or vocabulary sizes

**Solution**:
```bash
# On the machine with the correct tokenizer
git add 09_models/tokenizers/london_historical_tokenizer/
git commit -m "Sync corrected 30k tokenizer"
git push origin main

# On other machines
git pull origin main

# Verify tokenizer consistency
python3 -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('09_models/tokenizers/london_historical_tokenizer')
print(f'Vocab size: {tokenizer.vocab_size}')
print(f'Special tokens: {len(tokenizer.special_tokens_map)}')
"
```

---

## Model Training Issues

### Out of Memory Errors

**Error**: `CUDA out of memory` during training

**Symptoms**:
- Training crashes with OOM errors
- GPU memory usage at 100%
- Batch size too large for available memory

**Solutions**:
```bash
# Reduce batch size in config.py
# For SLM: batch_size: 12 (instead of 20)
# For Regular: batch_size: 8 (instead of 12)

# Or use gradient accumulation
# Set gradient_accumulation_steps: 2

# Check GPU memory
nvidia-smi
nvtop  # If available
```

### Training Interruption Recovery

**Error**: Training stopped unexpectedly

**Symptoms**:
- Training logs show incomplete runs
- Missing final checkpoints
- Need to resume from last checkpoint

**Solution**:
```bash
# Resume from last checkpoint
cd 04_training
./launch_slm_training.sh  # Will automatically resume

# Or manually resume
torchrun --nproc_per_node=2 04_training/train_model_slm.py --resume

# Check available checkpoints
ls -la 09_models/checkpoints/slm/
```

### Poor Training Performance

**Error**: Low MFU, slow training, or poor convergence

**Symptoms**:
- MFU below 15% for SLM models
- Training taking much longer than expected
- Loss not decreasing properly

**Solutions**:
```bash
# Check GPU utilization
nvidia-smi -l 1

# Verify configuration
python3 -c "
from config import config
print(f'Batch size: {config.slm_config[\"batch_size\"]}')
print(f'Enable compile: {config.slm_config[\"enable_compile\"]}')
print(f'AMP dtype: {config.slm_config[\"amp_dtype\"]}')
"

# Enable optimizations in config.py
# enable_compile: True
# amp_dtype: "bf16"
# enable_tf32: True
```

---

## Model Loading & Inference

### Hugging Face Model Loading Errors

**Error**: `ValueError: Unrecognized model. Should have a model_type key in its config.json`

**Symptoms**:
- Cannot load model using `AutoModelForCausalLM.from_pretrained()`
- Missing `model_type` in config.json
- Model directory structure issues

**Cause**: Trying to load PyTorch checkpoints as Hugging Face models

**Solution**:
```bash
# For PyTorch checkpoints (direct loading)
python3 04_training/test_checkpoint.py --checkpoint checkpoint-60001.pt

# For Hugging Face models (published models)
python3 06_inference/inference_unified.py --published --model_type slm

# Convert PyTorch checkpoint to Hugging Face format
python3 10_scripts/publish_slm_to_huggingface.py
```

### Checkpoint Loading Issues

**Error**: `RuntimeError: Error(s) in loading state_dict for GPT: Missing key(s)`

**Symptoms**:
- Cannot load PyTorch checkpoints
- Missing or unexpected keys in state dictionary
- `_orig_mod.` prefixes in checkpoint keys

**Cause**: Checkpoint keys have `_orig_mod.` prefixes from `torch.compile`

**Solution**:
```bash
# Use the updated test scripts that handle _orig_mod. prefixes
python3 04_training/test_slm_checkpoint.py --checkpoint checkpoint-60001.pt
python3 04_training/test_checkpoint.py --checkpoint checkpoint-60001.pt

# These scripts automatically strip _orig_mod. prefixes
```

### Model Generation Issues

**Error**: `AttributeError: 'GPT' object has no attribute 'generate'`

**Symptoms**:
- Cannot call `model.generate()` on custom GPT model
- Missing generation method
- Need to use manual generation loop

**Cause**: Custom GPT model doesn't have Hugging Face-style `generate()` method

**Solution**:
```bash
# Use the test scripts that implement manual generation
python3 04_training/test_slm_checkpoint.py --checkpoint checkpoint-60001.pt

# Or use the evaluation scripts
python3 05_evaluation/quick_eval.py --model_dir 09_models/checkpoints/slm --device gpu
```

---

## Evaluation Issues

### Hardcoded Model References

**Error**: "London Historical SLM - Quick Evaluation" displayed for regular model

**Symptoms**:
- Evaluation output shows wrong model type
- Hardcoded "SLM" references in output
- Inconsistent evaluation reporting

**Cause**: Hardcoded "SLM" references in evaluation script

**Solution**:
```bash
# The evaluation script has been updated to be generic for both models
python3 05_evaluation/quick_eval.py --model_dir 09_models/checkpoints --device gpu

# Both SLM and regular models now show "London Historical Model" in output
```

### Missing Dependencies

**Error**: `Missing required dependencies: No module named 'rouge_score'`

**Symptoms**:
- Evaluation fails due to missing packages
- Import errors during evaluation
- Incomplete evaluation results

**Solution**:
```bash
# Install missing dependencies
pip install rouge-score>=0.1.2

# Or reinstall all requirements
pip install -r requirements.txt

# Verify installation
python3 -c "import rouge_score; print('ROUGE installed successfully')"
```

### Evaluation Performance Issues

**Error**: Evaluation taking too long or consuming too much memory

**Symptoms**:
- Evaluation runs very slowly
- High memory usage during evaluation
- Timeout errors

**Solutions**:
```bash
# Use CPU for evaluation (safer)
python3 05_evaluation/quick_eval.py --model_dir 09_models/checkpoints --device cpu

# Reduce evaluation samples
# Edit 05_evaluation/quick_eval.py to reduce num_samples

# Use smaller model for testing
python3 05_evaluation/quick_eval.py --model_dir 09_models/checkpoints/slm --device gpu
```

---

## Performance Issues

### Low MFU (Model FLOPs Utilization)

**Error**: MFU below expected thresholds

**Symptoms**:
- MFU below 15% for SLM models
- GPU not fully utilized
- Slow training progress

**Solutions**:
```bash
# Check current configuration
python3 -c "
from config import config
print(f'Batch size: {config.slm_config[\"batch_size\"]}')
print(f'Enable compile: {config.slm_config[\"enable_compile\"]}')
print(f'AMP dtype: {config.slm_config[\"amp_dtype\"]}')
"

# Optimize configuration in config.py
# batch_size: 20 (for SLM)
# enable_compile: True
# amp_dtype: "bf16"
# enable_tf32: True

# Check GPU utilization
nvidia-smi -l 1
```

### Memory Leaks

**Error**: GPU memory usage increasing over time

**Symptoms**:
- Memory usage grows during training
- Eventually runs out of memory
- Training becomes slower over time

**Solutions**:
```bash
# Monitor memory usage
nvidia-smi -l 1

# Check for memory leaks in training loop
# Look for unreleased tensors or gradients

# Restart training if necessary
# The checkpoint system allows resuming from last saved state
```

---

## Multi-Machine Setup

### Git Sync Issues

**Error**: `git push origin main` rejected due to divergent branches

**Symptoms**:
- Cannot push changes to remote repository
- Divergent branch errors
- Merge conflicts

**Solution**:
```bash
# Configure git to use merge strategy
git config pull.rebase false

# Pull and merge remote changes
git pull origin main

# Push local changes
git push origin main

# If conflicts occur, resolve them manually
git status
git add <resolved-files>
git commit -m "Resolve merge conflicts"
git push origin main
```

### Environment Differences

**Error**: Code works on one machine but not another

**Symptoms**:
- Different Python versions
- Different package versions
- Different CUDA versions

**Solution**:
```bash
# Check environment consistency
python3 --version
pip list | grep torch
nvidia-smi

# Sync requirements
pip install -r requirements.txt

# Recreate virtual environment if needed
rm -rf helloLondon/
python3 01_environment/setup_environment.py
```

---

## Quick Fixes Reference

For the most common issues, here are the essential commands:

```bash
# Virtual environment issues
rm -rf helloLondon/ && python3 01_environment/setup_environment.py

# Tokenizer vocabulary mismatch  
python3 03_tokenizer/train_historical_tokenizer.py --vocab_size 30000

# Missing dependencies
pip install -r requirements.txt

# Check system status
python3 --version && nvidia-smi && python3 -c "import torch; print('PyTorch OK')"

# Test basic functionality
python3 03_tokenizer/test_tokenizer.py
python3 04_training/test_slm_checkpoint.py --checkpoint checkpoint-60001.pt
```

---

## Still Need Help?

If you're still experiencing issues after trying these solutions:

1. **Check the logs**: Look in the `logs/` directory for detailed error messages
2. **Review the documentation**: Check other files in `08_documentation/` for specific guides
3. **Search existing issues**: Look for similar problems in the GitHub repository
4. **Create a new issue**: Provide detailed information about your setup and error messages

### When Reporting Issues

Please include:
- **Error messages**: Complete error output with stack traces
- **Environment details**: OS, Python version, CUDA version, GPU model
- **Steps to reproduce**: Exact commands that led to the error
- **Log files**: Relevant log files from the `logs/` directory
- **Configuration**: Your current `config.py` settings

### Quick Diagnostic Commands

```bash
# System information
python3 --version
nvidia-smi
pip list | grep -E "(torch|transformers|accelerate)"

# Project status
ls -la 09_models/checkpoints/
ls -la 09_models/tokenizers/
python3 -c "from config import config; print('Config loaded successfully')"

# Test basic functionality
python3 03_tokenizer/test_tokenizer.py
python3 04_training/test_slm_checkpoint.py --checkpoint checkpoint-60001.pt
```

---

*This troubleshooting guide is regularly updated as new issues are discovered and resolved. Check back for the latest solutions!*
