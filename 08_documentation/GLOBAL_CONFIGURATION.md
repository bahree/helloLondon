# Global Configuration System

This document explains the global configuration system used throughout the Hello London Historical LLM project.

## Overview

The project uses a centralized configuration system (`config.py`) that manages all paths, settings, and parameters. This approach provides:

- **Single Source of Truth**: All paths and settings defined in one place
- **Easy Maintenance**: Change settings once, affects entire project
- **Flexible Overrides**: Command-line arguments can override any setting
- **Professional Structure**: Follows industry best practices

## Configuration Structure

### Core Paths
```python
# Data directories
data_dir = "data"
london_historical_data = "data/london_historical"
raw_data = "data/raw"
processed_data = "data/processed"

# Model directories
models_dir = "09_models"
tokenizers_dir = "09_models/tokenizers"
checkpoints_dir = "09_models/checkpoints"
london_tokenizer_dir = "09_models/tokenizers/london_historical_tokenizer"

# Output directories
outputs_dir = "outputs"
logs_dir = "logs"
temp_dir = "temp"
```

### Key Files
```python
# Data files
corpus_file = "data/london_historical/london_historical_corpus.txt"
data_sources_file = "02_data_collection/data_sources.json"

# Model files
tokenizer_file = "09_models/tokenizers/london_historical_tokenizer/tokenizer.json"
model_config_file = "09_models/checkpoints/config.json"
evaluation_results_file = "05_evaluation/results/evaluation_results.json"
```

## Configuration Sections

### 1. Training Configuration
```python
# Regular Model Configuration
training_config = {
    "model_name": "london-historical-llm",
    "architecture": "gpt2",
    "max_length": 1024,
    "batch_size": 2,
    "learning_rate": 3e-5,
    "num_epochs": 15,
    "warmup_steps": 1000,
    "save_steps": 250,
    "eval_steps": 250,
    "logging_steps": 50,
    "gradient_accumulation_steps": 4,
    "weight_decay": 0.01,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "max_grad_norm": 1.0,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.1,
    "group_by_length": True,
    "dataloader_drop_last": True,
    "eval_accumulation_steps": 1,
    "save_safetensors": True,
    "dataloader_prefetch_factor": 2,
    "half_precision_backend": "auto",
    "log_level": "info",
    "ddp_find_unused_parameters": False,
    "save_only_model": True,
    "save_total_limit": 5
}

# SLM (Small Language Model) Configuration
slm_config = {
    "model_name": "london-historical-slm",
    "architecture": "gpt2",
    "max_length": 512,
    "batch_size": 18,
    "learning_rate": 3e-4,
    "num_epochs": 3,
    "max_steps": 60000,
    "warmup_steps": 500,
    "save_steps": 500,
    "eval_steps": 4000,
    "logging_steps": 100,
    "gradient_accumulation_steps": 1,
    "weight_decay": 0.01,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_epsilon": 1e-8,
    "max_grad_norm": 1.0,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.1,
    "group_by_length": True,
    "dataloader_drop_last": True,
    "eval_accumulation_steps": 1,
    "save_safetensors": True,
    "dataloader_prefetch_factor": 2,
    "half_precision_backend": "auto",
    "log_level": "info",
    "ddp_find_unused_parameters": False,
    "save_only_model": True,
    "save_total_limit": 5,
    "use_wandb": True
}
```

### 2. Tokenizer Configuration
```python
tokenizer_config = {
    "vocab_size": 30000,  # Optimized for historical text
    "min_frequency": 2,
    "special_tokens": [
        # Core tokens
        "<|endoftext|>", "<|startoftext|>", "<|pad|>", "<|unk|>", "<|mask|>",
        
        # Historical structure tokens
        "<|year|>", "<|date|>", "<|name|>", "<|place|>", "<|title|>",
        "<|chapter|>", "<|verse|>", "<|quote|>", "<|speech|>", "<|narrator|>",
        "<|author|>", "<|book|>", "<|newline|>", "<|paragraph|>",
        
        # Historical language tokens (150+ tokens)
        "<|thou|>", "<|thee|>", "<|thy|>", "<|thine|>", "<|hast|>", "<|hath|>",
        "<|doth|>", "<|dost|>", "<|art|>", "<|wilt|>", "<|shalt|>", "<|canst|>",
        "<|verily|>", "<|indeed|>", "<|forsooth|>", "<|methinks|>", "<|perchance|>",
        
        # London-specific tokens
        "<|london|>", "<|thames|>", "<|westminster|>", "<|city|>", "<|borough|>",
        "<|parish|>", "<|ward|>", "<|street|>", "<|lane|>", "<|court|>",
        "<|tavern|>", "<|inn|>", "<|coffeehouse|>", "<|market|>", "<|fair|>",
        "<|tower|>", "<|stpauls|>", "<|fleet|>", "<|cheapside|>", "<|smithfield|>",
        
        # Historical period tokens
        "<|tudor|>", "<|stuart|>", "<|georgian|>", "<|regency|>", "<|victorian|>",
        "<|plague|>", "<|fire|>", "<|great|>", "<|civil|>", "<|war|>",
        "<|elizabethan|>", "<|restoration|>", "<|hanoverian|>", "<|enlightenment|>",
        
        # Social class tokens
        "<|noble|>", "<|gentleman|>", "<|lady|>", "<|commoner|>", "<|apprentice|>",
        "<|servant|>", "<|merchant|>", "<|artisan|>", "<|labourer|>", "<|beggar|>",
        "<|yeoman|>", "<|squire|>", "<|knight|>", "<|duke|>", "<|earl|>",
        
        # Legal and court tokens
        "<|trial|>", "<|judge|>", "<|jury|>", "<|witness|>", "<|accused|>",
        "<|sentence|>", "<|punishment|>", "<|gaol|>", "<|transport|>", "<|hanging|>",
        "<|magistrate|>", "<|constable|>", "<|watchman|>", "<|pillory|>", "<|stocks|>",
        
        # Religious tokens
        "<|church|>", "<|parish|>", "<|clergy|>", "<|bishop|>", "<|archbishop|>",
        "<|prayer|>", "<|sermon|>", "<|blessing|>", "<|curse|>", "<|sin|>",
        "<|puritan|>", "<|dissenter|>", "<|catholic|>", "<|protestant|>", "<|chapel|>",
        
        # Economic tokens
        "<|shilling|>", "<|pound|>", "<|penny|>", "<|guinea|>", "<|crown|>",
        "<|trade|>", "<|commerce|>", "<|merchant|>", "<|shop|>", "<|warehouse|>",
        "<|farthing|>", "<|halfpenny|>", "<|groat|>", "<|sovereign|>", "<|noble|>",
        
        # Time and date tokens
        "<|morn|>", "<|noon|>", "<|eve|>", "<|night|>", "<|dawn|>", "<|dusk|>",
        "<|monday|>", "<|tuesday|>", "<|wednesday|>", "<|thursday|>", "<|friday|>",
        "<|saturday|>", "<|sunday|>", "<|january|>", "<|february|>", "<|march|>",
        "<|april|>", "<|may|>", "<|june|>", "<|july|>", "<|august|>",
        "<|september|>", "<|october|>", "<|november|>", "<|december|>",
        "<|fortnight|>", "<|sennight|>", "<|michaelmas|>", "<|ladyday|>", "<|candlemas|>",
        "<|midsummer|>", "<|christmas|>", "<|easter|>", "<|whitsun|>", "<|lent|>",
        
        # Profession tokens
        "<|apothecary|>", "<|barbersurgeon|>", "<|coachman|>", "<|linkboy|>", "<|waterman|>",
        "<|chimneysweep|>", "<|costermonger|>", "<|nightsoilman|>", "<|beadle|>", "<|crier|>",
        
        # Slang and street tokens
        "<|doss|>", "<|ken|>", "<|fawney|>", "<|rig|>", "<|sup|>",
        "<|phiz|>", "<|visage|>", "<|countenance|>", "<|mauther|>", "<|brabble|>",
        "<|chuffed|>", "<|bauchle|>", "<|clomph|>", "<|cramboclink|>", "<|abroad|>"
    ]
}
```

### 3. Data Collection Configuration
```python
data_collection_config = {
    "max_retries": 3,
    "timeout": 30,
    "delay_between_requests": 1.0,
    "user_agent": "HelloLondonBot/1.0 (Historical Research)",
    "chunk_size": 8192,
    "max_file_size_mb": 100
}
```

## Usage Examples

### Basic Usage
```python
from config import config

# Access configuration values
data_dir = config.london_historical_data
vocab_size = config.tokenizer_config["vocab_size"]
max_length = config.training_config["max_length"]
```

### Relative Paths
```python
from config import get_data_path, get_model_path, get_tokenizer_path

# Get paths relative to current component
data_path = get_data_path("tokenizer")  # Returns "data/london_historical"
model_path = get_model_path("training")  # Returns "09_models/checkpoints"
tokenizer_path = get_tokenizer_path("evaluation")  # Returns "09_models/tokenizers/london_historical_tokenizer"
```

### Component-Specific Usage
```python
# In tokenizer component
from config import config, get_data_path

class TokenizerTrainer:
    def __init__(self):
        self.data_dir = Path(get_data_path("tokenizer"))
        self.vocab_size = config.tokenizer_config["vocab_size"]
        self.special_tokens = config.tokenizer_config["special_tokens"]
```

## Command Line Overrides

All components support command-line arguments that override global configuration:

```bash
# Override data directory
python3 train_tokenizer.py --data_dir /custom/path/to/data

# Override tokenizer settings
python3 train_tokenizer.py --vocab_size 60000 --min_frequency 3

# Override training settings
python3 train_model.py --batch_size 4 --learning_rate 5e-5
```

## Benefits

### 1. Maintainability
- Change paths in one place
- Update settings globally
- Easy to add new components

### 2. Flexibility
- Command-line overrides
- Environment-specific configurations
- Easy testing with different settings

### 3. Professional Structure
- Follows industry best practices
- Clear separation of concerns
- Easy to understand and modify

### 4. Scalability
- Easy to add new components
- Consistent path management
- Centralized parameter tuning

## Testing the Configuration

Test the global configuration system:

```bash
# Test configuration
python3 config.py

# Expected output:
# Hello London Configuration
# ==================================================
# Project Root: /path/to/helloLondon
# Data Directory: /path/to/helloLondon/data/london_historical
# Models Directory: /path/to/helloLondon/09_models
# Tokenizer Directory: /path/to/helloLondon/09_models/tokenizers/london_historical_tokenizer
# Checkpoints Directory: /path/to/helloLondon/09_models/checkpoints
```

## WandB Configuration

The project includes integrated WandB (Weights & Biases) support for experiment tracking:

```python
wandb_config = {
    "project": "helloLondon",
    "entity": "amitbahree",  # Your WandB username
    "tags": ["london", "historical", "llm", "gpt2", "1500-1850"],
    "mode": "online",  # or "offline" for local-only logging
    "log_model": True,
    "save_code": True,
    "group": "hello-london-experiments",
    "job_type": "training",
    "notes": "Hello London Historical LLM training with custom tokenizer"
}
```

**Setup Required**: See [WandB Setup Guide](WANDB_SETUP.md) for complete setup instructions.

## Best Practices

1. **Always use global config** for new components
2. **Override via command line** for testing
3. **Document custom settings** in component READMEs
4. **Test configuration** before committing changes
5. **Keep configuration simple** and well-organized

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure `config.py` is in the project root
2. **Path Not Found**: Check that the data directory exists
3. **Permission Error**: Ensure write permissions for output directories
4. **Configuration Override**: Check command-line arguments

### Debugging

```python
# Print current configuration
print(config.to_dict())

# Check specific paths
print(f"Data dir exists: {config.london_historical_data.exists()}")
print(f"Models dir exists: {config.models_dir.exists()}")
```

## GPU Configuration

The project includes automatic GPU detection and configuration:

```python
gpu_config = {
    "auto_detect": True,  # Automatically detect available GPUs
    "max_gpus": 0,  # Maximum number of GPUs to use (0 = no limit, use all available)
    "min_gpu_memory_gb": 8,  # Minimum GPU memory required (GB)
    "preferred_gpu_types": ["A30", "A40", "A100", "V100", "RTX4090", "RTX4080"],
    "fallback_to_cpu": True,  # Fall back to CPU if no suitable GPUs found
    "force_single_gpu": False,  # Force single GPU even if multiple available
    "force_multi_gpu": False,  # Force multi-GPU even if only one available
    "gpu_memory_fraction": 0.9,  # Fraction of GPU memory to use (0.0-1.0)
    "allow_growth": True,  # Allow GPU memory growth
    "log_device_placement": False  # Log device placement for debugging
}
```
