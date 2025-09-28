#!/usr/bin/env python3
"""
Global Configuration for Hello London Historical LLM
Centralized configuration for all paths and settings
"""

import os
from pathlib import Path
from typing import Dict, Any

class Config:
    """Global configuration class"""
    
    def __init__(self):
        # Get project root directory
        self.project_root = Path(__file__).parent.absolute()
        
        # Data paths
        self.data_dir = self.project_root / "data"
        self.london_historical_data = self.data_dir / "london_historical"
        self.raw_data = self.data_dir / "raw"
        self.processed_data = self.data_dir / "processed"
        
        # Model paths
        self.models_dir = self.project_root / "09_models"
        self.tokenizers_dir = self.models_dir / "tokenizers"
        self.checkpoints_dir = self.models_dir / "checkpoints"
        self.london_tokenizer_dir = self.tokenizers_dir / "london_historical_tokenizer"
        
        # Output paths
        self.outputs_dir = self.project_root / "outputs"
        self.logs_dir = self.project_root / "logs"
        self.temp_dir = self.project_root / "temp"
        
        # Component-specific paths
        self.data_collection_dir = self.project_root / "02_data_collection"
        self.tokenizer_dir = self.project_root / "03_tokenizer"
        self.training_dir = self.project_root / "04_training"
        self.evaluation_dir = self.project_root / "05_evaluation"
        self.testing_dir = self.project_root / "06_testing"
        self.utilities_dir = self.project_root / "07_utilities"
        self.documentation_dir = self.project_root / "08_documentation"
        self.scripts_dir = self.project_root / "10_scripts"
        
        # Key files - check for comprehensive corpus first
        comprehensive_corpus = self.london_historical_data / "london_historical_corpus_comprehensive.txt"
        if comprehensive_corpus.exists():
            self.corpus_file = comprehensive_corpus
        else:
            self.corpus_file = self.london_historical_data / "london_historical_corpus.txt"
        self.data_sources_file = self.data_collection_dir / "data_sources.json"
        self.tokenizer_file = self.london_tokenizer_dir / "tokenizer.json"
        self.model_config_file = self.checkpoints_dir / "config.json"
        self.evaluation_results_file = self.evaluation_dir / "results" / "evaluation_results.json"
        
        # Training configuration - Optimized for 2x A30 GPUs (24GB each)
        self.training_config = {
            "model_name": "london-historical-llm",  # Your custom model name
            "architecture": "gpt2",  # Model architecture (not a pre-trained model)
            "max_length": 1024,
            "batch_size": 12,  # Optimized for A30 GPUs (24 effective with 2 GPUs)
            "learning_rate": 3e-5,
            "max_steps": 60000,  # Match SLM training length for proper convergence
            "warmup_steps": 1000,  # More warmup steps for longer training
            "save_steps": 250,
            "eval_steps": 1000,
            "logging_steps": 50,
            "gradient_accumulation_steps": 1,  # Reduced since we increased batch_size
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
            "save_total_limit": 5,  # Keep 5 most recent checkpoints
            "save_strategy": "steps",
            "eval_strategy": "steps",
            "load_best_model_at_end": False,  # No early stopping
            "metric_for_best_model": None,  # No early stopping
            "greater_is_better": None,  # No early stopping
            # Model architecture
            "n_layer": 24,  # Regular model has more layers
            "n_head": 16,   # Regular model has more heads
            "n_embd": 1024, # Regular model has larger embeddings
            # Training parameters
            "eval_iters": 100,
            "use_wandb": True,
            "enable_tf32": True,
            "enable_amp": True,
            "amp_dtype": "bf16",
            "enable_compile": True
        }
        
        # SLM (Small Language Model) configuration - optimized for 2x A30 GPUs (24GB each)
        self.slm_config = {
            "model_name": "london-historical-slm",  # SLM model name
            "architecture": "gpt2",  # Model architecture
            "max_length": 512,  # Full context window for better learning
            "batch_size": 18,  # Optimized for 2x A30s (36 effective batch size)
            "learning_rate": 3e-4,  # Higher LR to unstick training
            "num_epochs": 3,  # Much fewer epochs for SLM - should be enough
            "warmup_steps": 500,  # Longer warmup for stability in fresh training
            "save_steps": 500,  # Less frequent saves for efficiency
            "eval_steps": 4000,  # Less frequent evals to reduce overhead
            "logging_steps": 100,  # Reasonable logging cadence
            "gradient_accumulation_steps": 1,  # Reduced accumulation
            "max_steps": 60000,  # Long overnight training for optimal convergence
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
            "save_total_limit": 5,  # Keep 5 checkpoints for SLM
            "save_strategy": "steps",
            "eval_strategy": "steps",
            "load_best_model_at_end": False,  # No early stopping
            "metric_for_best_model": None,  # No early stopping
            "greater_is_better": None,  # No early stopping
            "dataloader_num_workers": 4,  # Add missing parameter
            "seed": 42,  # Random seed for reproducibility
            "disable_tqdm": False,  # Show progress bars
            "dataloader_persistent_workers": True,  # Keep workers alive between epochs
            "prediction_loss_only": True,  # Only compute loss during training for efficiency
            "remove_unused_columns": False,  # Keep all columns for compatibility
            "dataloader_pin_memory": True,  # Pin memory for faster GPU transfer
            # Runtime/precision knobs (A30 optimized)
            "enable_tf32": True,
            "enable_amp": True,
            "amp_dtype": "bf16",  # bf16 on Ampere; fallback to fp16 if unsupported
            "enable_compile": True,  # torch.compile; set False to reduce memory usage
            # Conservative baseline (for broad hardware) — uncomment to use:
            # "enable_tf32": False,
            # "enable_amp": True,
            # "amp_dtype": "fp16",
            # Sequence/batch control
            "max_length": 1024,  # increase tokens per step when VRAM allows
            "batch_size": 20,    # per-GPU batch; raise if VRAM allows
            # Conservative sequence/batch — uncomment to use:
            # "max_length": 768,
            # "batch_size": 8,
            # Data split configuration
            "eval_split_ratio": 0.05,  # 5% for evaluation (better validation with clean data)
            "max_eval_samples": 3000,  # Reduced eval samples for faster training
            "eval_batch_size": 24,  # Increased eval batch size for A30s
            "eval_iters": 50,  # Reduce eval batches to lower memory/time
            # Pre-tokenization configuration
            "use_pretokenized": False,  # Disabled for now - will enable after first run
            # Data cleaning configuration
            "aggressive_cleaning": False,  # Set to True to enable aggressive structured data filtering
            # Data source configuration
            "enable_old_bailey": False,  # Disabled - causes generation issues with structured legal data
            "enable_london_lives": False,  # Disabled - causes generation issues with semantic markup
            "enable_literature": True,  # Enable literature sources for diverse text
            "enable_newspapers": True,  # Enable newspaper sources
            "enable_diaries": True,  # Enable diary sources
            "enable_archive_org": True,  # Enable archive.org sources (back online)
            # SLM-specific model architecture - optimized for segmented data
            "n_layer": 12,  # More layers for better learning capacity
            "n_head": 12,  # More attention heads for better attention
            "n_embd": 768,  # Larger embedding dimension for better representations
            "n_positions": 512,  # Full context window
            "vocab_size": 30000,  # Match tokenizer vocabulary size
            # WandB configuration
            "use_wandb": True  # Set to True to enable WandB logging
        }
        
        # Historical Tokenizer configuration (optimized for 1500-1850 English)
        self.tokenizer_config = {
            "vocab_size": 30000,  # More focused vocabulary for historical text
            "min_frequency": 2,
            "special_tokens": [
                # Core tokens
                "<|endoftext|>", "<|startoftext|>", "<|pad|>", "<|unk|>", "<|mask|>",
                
                # Historical structure tokens (better than language-specific)
                "<|year|>",      # For years like 1834
                "<|date|>",      # For dates
                "<|name|>",      # For proper names
                "<|place|>",     # For places like London
                "<|title|>",     # For titles like Mr., Mrs.
                "<|chapter|>",   # For chapter markers
                "<|verse|>",     # For verse markers
                "<|quote|>",     # For quotations
                "<|speech|>",    # For dialogue
                "<|narrator|>",  # For narrative text
                "<|author|>",    # For author names
                "<|book|>",      # For book titles
                "<|newline|>",   # For line breaks
                "<|paragraph|>", # For paragraph breaks
                
                # Historical language tokens (essential only)
                "<|thou|>", "<|thee|>", "<|thy|>", "<|thine|>", "<|hast|>", "<|hath|>",
                "<|doth|>", "<|dost|>", "<|art|>", "<|wilt|>", "<|shalt|>", "<|canst|>",
                "<|verily|>", "<|indeed|>", "<|forsooth|>", "<|methinks|>", "<|perchance|>",
                "<|anon|>", "<|ere|>", "<|whilst|>", "<|betwixt|>", "<|amongst|>",
                "<|prithee|>", "<|pray|>", "<|beseech|>", "<|ye|>", "<|yon|>", "<|fain|>",
                "<|quoth|>", "<|afeard|>", "<|affright|>", "<|albeit|>", "<|howbeit|>",
                "<|hither|>", "<|thither|>", "<|whence|>", "<|whither|>", "<|wherefore|>",
                "<|hitherto|>", "<|thereto|>", "<|whereto|>", "<|whereby|>",
                "<|peradventure|>", "<|truly|>", "<|marry|>", "<|goodmorrow|>", "<|farewell|>",
                
                # London-specific tokens (expanded)
                "<|london|>", "<|thames|>", "<|westminster|>", "<|city|>", "<|borough|>",
                "<|parish|>", "<|ward|>", "<|street|>", "<|lane|>", "<|court|>",
                "<|tavern|>", "<|inn|>", "<|coffeehouse|>", "<|market|>", "<|fair|>",
                "<|tower|>", "<|stpauls|>", "<|fleet|>", "<|cheapside|>", "<|smithfield|>",
                "<|tyburn|>", "<|newgate|>", "<|southwark|>", "<|coventgarden|>", "<|billingsgate|>",
                "<|leadenhall|>", "<|guildhall|>", "<|exchange|>", "<|bridge|>", "<|wharf|>",
                
                # Historical period tokens (expanded)
                "<|tudor|>", "<|stuart|>", "<|georgian|>", "<|regency|>", "<|victorian|>",
                "<|plague|>", "<|fire|>", "<|great|>", "<|civil|>", "<|war|>",
                "<|elizabethan|>", "<|restoration|>", "<|hanoverian|>", "<|enlightenment|>",
                "<|gunpowder|>", "<|popish|>", "<|southsea|>", "<|bubble|>", "<|revolution|>", "<|glorious|>",
                
                # Social class tokens (expanded)
                "<|noble|>", "<|gentleman|>", "<|lady|>", "<|commoner|>", "<|apprentice|>",
                "<|servant|>", "<|merchant|>", "<|artisan|>", "<|labourer|>", "<|beggar|>",
                "<|yeoman|>", "<|squire|>", "<|knight|>", "<|duke|>", "<|earl|>",
                "<|vagabond|>", "<|pauper|>", "<|alderman|>", "<|burgess|>", "<|freeman|>",
                
                # Legal and court tokens (expanded)
                "<|trial|>", "<|judge|>", "<|jury|>", "<|witness|>", "<|accused|>",
                "<|sentence|>", "<|punishment|>", "<|gaol|>", "<|transport|>", "<|hanging|>",
                "<|magistrate|>", "<|constable|>", "<|watchman|>", "<|pillory|>", "<|stocks|>",
                "<|indictment|>", "<|verdict|>", "<|execution|>", "<|tyburntree|>", "<|newgatebird|>",
                
                # Religious tokens (expanded)
                "<|church|>", "<|parish|>", "<|clergy|>", "<|bishop|>", "<|archbishop|>",
                "<|prayer|>", "<|sermon|>", "<|blessing|>", "<|curse|>", "<|sin|>",
                "<|puritan|>", "<|dissenter|>", "<|catholic|>", "<|protestant|>", "<|chapel|>",
                "<|tithes|>", "<|communion|>", "<|heresy|>", "<|papist|>", "<|atheist|>",
                
                # Economic tokens (expanded)
                "<|shilling|>", "<|pound|>", "<|penny|>", "<|guinea|>", "<|crown|>",
                "<|trade|>", "<|commerce|>", "<|merchant|>", "<|shop|>", "<|warehouse|>",
                "<|farthing|>", "<|halfpenny|>", "<|groat|>", "<|sovereign|>", "<|noble|>",
                "<|guild|>", "<|livery|>", "<|apprenticeship|>", "<|bargain|>", "<|usury|>",
                
                # Time and date tokens (expanded)
                "<|morn|>", "<|noon|>", "<|eve|>", "<|night|>", "<|dawn|>", "<|dusk|>",
                "<|monday|>", "<|tuesday|>", "<|wednesday|>", "<|thursday|>", "<|friday|>",
                "<|saturday|>", "<|sunday|>", "<|january|>", "<|february|>", "<|march|>",
                "<|april|>", "<|may|>", "<|june|>", "<|july|>", "<|august|>",
                "<|september|>", "<|october|>", "<|november|>", "<|december|>",
                "<|fortnight|>", "<|sennight|>", "<|michaelmas|>", "<|ladyday|>", "<|candlemas|>",
                "<|midsummer|>", "<|christmas|>", "<|easter|>", "<|whitsun|>", "<|lent|>",
                
                # Profession tokens (new category)
                "<|apothecary|>", "<|barbersurgeon|>", "<|coachman|>", "<|linkboy|>", "<|waterman|>",
                "<|chimneysweep|>", "<|costermonger|>", "<|nightsoilman|>", "<|beadle|>", "<|crier|>",
                
                # Slang and street tokens (new category)
                "<|doss|>", "<|ken|>", "<|fawney|>", "<|rig|>", "<|sup|>",
                "<|phiz|>", "<|visage|>", "<|countenance|>", "<|mauther|>", "<|brabble|>",
                "<|chuffed|>", "<|bauchle|>", "<|clomph|>", "<|cramboclink|>", "<|abroad|>"
            ]
        }
        
        # Data collection configuration
        self.data_collection_config = {
            "max_retries": 3,
            "timeout": 30,
            "delay_between_requests": 1.0,
            "user_agent": "HelloLondonBot/1.0 (Historical Research)",
            "chunk_size": 8192,
            "max_file_size_mb": 100
        }
        
        # WandB configuration
        self.wandb_config = {
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
        
        # GPU configuration
        self.gpu_config = {
            "auto_detect": True,  # Automatically detect available GPUs
            "max_gpus": 0,  # Maximum number of GPUs to use (0 = no limit, use all available)
            "min_gpu_memory_gb": 8,  # Minimum GPU memory required (GB)
            "preferred_gpu_types": ["A30", "A40", "A100", "V100", "RTX4090", "RTX4080"],  # Preferred GPU types
            "fallback_to_cpu": True,  # Fall back to CPU if no suitable GPUs found
            "force_single_gpu": False,  # Force single GPU even if multiple available
            "force_multi_gpu": False,  # Force multi-GPU even if only one available
            "gpu_memory_fraction": 0.9,  # Fraction of GPU memory to use (0.0-1.0)
            "allow_growth": True,  # Allow GPU memory growth
            "log_device_placement": False  # Log device placement for debugging
        }
    
    def get_relative_path(self, from_dir: Path, to_path: Path) -> str:
        """Get relative path from one directory to another"""
        try:
            return str(to_path.relative_to(from_dir))
        except ValueError:
            # If paths are not relative, return absolute path
            return str(to_path)
    
    def get_tokenizer_data_path(self) -> str:
        """Get data path relative to tokenizer directory"""
        return self.get_relative_path(self.tokenizer_dir, self.london_historical_data)
    
    def get_training_data_path(self) -> str:
        """Get data path relative to training directory"""
        return self.get_relative_path(self.training_dir, self.london_historical_data)
    
    def get_evaluation_data_path(self) -> str:
        """Get data path relative to evaluation directory"""
        return self.get_relative_path(self.evaluation_dir, self.london_historical_data)
    
    def get_script_data_path(self) -> str:
        """Get data path relative to scripts directory"""
        return self.get_relative_path(self.scripts_dir, self.london_historical_data)
    
    def ensure_directories(self):
        """Ensure all required directories exist"""
        directories = [
            self.data_dir,
            self.london_historical_data,
            self.raw_data,
            self.processed_data,
            self.models_dir,
            self.tokenizers_dir,
            self.checkpoints_dir,
            self.london_tokenizer_dir,
            self.outputs_dir,
            self.logs_dir,
            self.temp_dir,
            self.evaluation_dir / "results"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "project_root": str(self.project_root),
            "data_dir": str(self.data_dir),
            "london_historical_data": str(self.london_historical_data),
            "models_dir": str(self.models_dir),
            "tokenizers_dir": str(self.tokenizers_dir),
            "checkpoints_dir": str(self.checkpoints_dir),
            "training_config": self.training_config,
            "tokenizer_config": self.tokenizer_config,
            "data_collection_config": self.data_collection_config
        }

# Global configuration instance
config = Config()

# Convenience functions
def get_data_path(component: str = "root") -> str:
    """Get data path relative to component directory"""
    if component == "tokenizer":
        return config.get_tokenizer_data_path()
    elif component == "training":
        return config.get_training_data_path()
    elif component == "evaluation":
        return config.get_evaluation_data_path()
    elif component == "scripts":
        return config.get_script_data_path()
    else:
        return str(config.london_historical_data)

def get_model_path(component: str = "root") -> str:
    """Get model path relative to component directory"""
    if component == "tokenizer":
        return config.get_relative_path(config.tokenizer_dir, config.london_tokenizer_dir)
    elif component == "training":
        return config.get_relative_path(config.training_dir, config.checkpoints_dir)
    elif component == "evaluation":
        return config.get_relative_path(config.evaluation_dir, config.checkpoints_dir)
    else:
        return str(config.checkpoints_dir)

def get_tokenizer_path(component: str = "root") -> str:
    """Get tokenizer path relative to component directory"""
    if component == "training":
        return config.get_relative_path(config.training_dir, config.london_tokenizer_dir)
    elif component == "evaluation":
        return config.get_relative_path(config.evaluation_dir, config.london_tokenizer_dir)
    else:
        return str(config.london_tokenizer_dir)

def get_gpu_config() -> dict:
    """Get GPU configuration"""
    return config.gpu_config

if __name__ == "__main__":
    # Test the configuration
    print("Hello London Configuration")
    print("=" * 50)
    print(f"Project Root: {config.project_root}")
    print(f"Data Directory: {config.london_historical_data}")
    print(f"Models Directory: {config.models_dir}")
    print(f"Tokenizer Directory: {config.london_tokenizer_dir}")
    print(f"Checkpoints Directory: {config.checkpoints_dir}")
    print("\nRelative Paths:")
    print(f"From tokenizer: {get_data_path('tokenizer')}")
    print(f"From training: {get_data_path('training')}")
    print(f"From evaluation: {get_data_path('evaluation')}")
    print(f"From scripts: {get_data_path('scripts')}")
