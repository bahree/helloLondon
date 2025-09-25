#!/usr/bin/env python3
"""
Simple SLM Training Script - Based on nanoGPT approach
"""

import os
import sys
import json
import argparse
import logging
import time
import math
import pickle
import inspect
from pathlib import Path
from contextlib import nullcontext
from typing import Dict, Any, Optional

# Suppress tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import global configuration
from config import config

try:
    import torch
    import numpy as np
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.distributed import init_process_group, destroy_process_group
    from transformers import AutoTokenizer
    import wandb
except ImportError as e:
    print(f"Missing required dependencies: {e}")
    print("Please install requirements: pip install -r requirements.txt")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training_simple.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SimpleGPTConfig:
    """Simple GPT configuration"""
    def __init__(self, n_layer=8, n_head=8, n_embd=512, block_size=256, 
                 bias=False, vocab_size=30000, dropout=0.1):
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = block_size
        self.bias = bias
        self.vocab_size = vocab_size
        self.dropout = dropout

class SimpleGPT(torch.nn.Module):
    """Simple GPT model based on nanoGPT"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.transformer = torch.nn.ModuleDict(dict(
            wte = torch.nn.Embedding(config.vocab_size, config.n_embd),
            wpe = torch.nn.Embedding(config.block_size, config.n_embd),
            drop = torch.nn.Dropout(config.dropout),
            h = torch.nn.ModuleList([SimpleBlock(config) for _ in range(config.n_layer)]),
            ln_f = torch.nn.LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Tie weights
        self.transformer.wte.weight = self.lm_head.weight
        
    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, torch.nn.LayerNorm):
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            if module.weight is not None:
                torch.nn.init.ones_(module.weight)
    
    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        
        # Forward the GPT model
        tok_emb = self.transformer.wte(idx)  # Token embeddings
        pos_emb = self.transformer.wpe(pos)  # Position embeddings
        x = self.transformer.drop(tok_emb + pos_emb)
        
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        
        if targets is not None:
            # If we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # Note: using list [-1] to preserve the time dim
            loss = None
        
        return logits, loss
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # Start with all of the network parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # Filter out those that do not require gradients
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # Create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        return optimizer
    
    def get_num_params(self, non_embedding=True):
        """Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params
    
    def estimate_mfu(self, batch_size, dt):
        """Estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        # First estimate the number of flops we do per iteration. See PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        # First we get the number of tokens processed per iteration
        N = batch_size * self.config.block_size
        # Then we get the number of flops per token
        # Forward pass: 2 * n_params * n_tokens (attention + MLP)
        # Backward pass: 2 * n_params * n_tokens (attention + MLP)
        # Total: 4 * n_params * n_tokens
        flops_per_token = 4 * self.get_num_params()
        flops_per_fwdbwd = flops_per_token * N
        flops_per_iter = flops_per_fwdbwd * 2  # Forward + backward pass
        # Express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # Per second
        flops_promised = 312e12  # A100 bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

class SimpleBlock(torch.nn.Module):
    """Simple transformer block"""
    
    def __init__(self, config):
        super().__init__()
        self.ln_1 = torch.nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = SimpleCausalSelfAttention(config)
        self.ln_2 = torch.nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = SimpleMLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class SimpleCausalSelfAttention(torch.nn.Module):
    """Simple causal self-attention"""
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # Key, query, value projections for all heads, but in a batch
        self.c_attn = torch.nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = torch.nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # Regularization
        self.attn_dropout = torch.nn.Dropout(config.dropout)
        self.resid_dropout = torch.nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
    
    def forward(self, x):
        B, T, C = x.size()  # Batch size, sequence length, embedding dimensionality (n_embd)
        
        # Calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        
        # Causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # Re-assemble all head outputs side by side
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class SimpleMLP(torch.nn.Module):
    """Simple MLP"""
    
    def __init__(self, config):
        super().__init__()
        self.c_fc = torch.nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = torch.nn.GELU()
        self.c_proj = torch.nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = torch.nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class SimpleLondonHistoricalTrainer:
    """Simple trainer based on nanoGPT approach"""
    
    def __init__(self, data_dir: str, tokenizer_dir: str, output_dir: str, resume_from_checkpoint: str = None):
        self.data_dir = Path(data_dir)
        self.tokenizer_dir = Path(tokenizer_dir)
        self.output_dir = Path(output_dir)
        self.slm_config = config.slm_config
        self.resume_from_checkpoint = resume_from_checkpoint
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training parameters
        self.batch_size = self.slm_config["batch_size"]
        # Use full max_length to increase tokens per step
        self.block_size = int(self.slm_config["max_length"])  
        self.learning_rate = self.slm_config["learning_rate"]
        self.max_iters = self.slm_config["max_steps"]
        self.eval_interval = self.slm_config.get("eval_steps", 500)
        self.log_interval = self.slm_config.get("logging_steps", 10)
        self.eval_iters = self.slm_config.get("eval_iters", 100)
        
        # Model architecture - from config
        self.n_layer = self.slm_config.get("n_layer", 8)
        self.n_head = self.slm_config.get("n_head", 8)
        self.n_embd = self.slm_config.get("n_embd", 512)
        self.dropout = 0.1
        self.bias = False
        
        # System
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Precision / TF32 knobs from config
        tf32 = self.slm_config.get("enable_tf32", True)
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.backends.cudnn.allow_tf32 = bool(tf32)
        try:
            torch.set_float32_matmul_precision('high' if tf32 else 'medium')
        except Exception:
            pass
        use_amp = self.slm_config.get("enable_amp", True)
        amp_dtype_cfg = self.slm_config.get("amp_dtype", "bf16").lower()
        bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        if use_amp:
            if amp_dtype_cfg == 'bf16' and bf16_ok:
                self.dtype = 'bfloat16'
            else:
                self.dtype = 'float16'
        else:
            self.dtype = 'float32'
        
        # DDP setup (process group already initialized in main())
        self.ddp = int(os.environ.get('RANK', -1)) != -1
        if self.ddp:
            self.ddp_rank = int(os.environ['RANK'])
            self.ddp_local_rank = int(os.environ['LOCAL_RANK'])
            self.ddp_world_size = int(os.environ['WORLD_SIZE'])
            self.device = f'cuda:{self.ddp_local_rank}'
            torch.cuda.set_device(self.device)
            self.master_process = self.ddp_rank == 0
            self.seed_offset = self.ddp_rank
        else:
            self.master_process = True
            self.seed_offset = 0
            self.ddp_world_size = 1
        
        # WandB setup
        self.use_wandb = self.slm_config.get("use_wandb", False) and self.master_process
        if self.use_wandb:
            try:
                from datetime import datetime
                wandb.init(
                    project=config.wandb_config["project"],
                    entity=config.wandb_config["entity"],
                    name=f"london-slm-simple-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                    config=self.slm_config,
                    tags=config.wandb_config["tags"] + ["simple"],
                    group=config.wandb_config["group"],
                    job_type=config.wandb_config["job_type"],
                    notes="Simple London Historical SLM training with clean data"
                )
                logger.info(f"WandB initialized: {wandb.run.url}")
                
                # Initialize metrics order for better mobile UI (loss first)
                wandb.log({
                    "train/loss": 0.0,
                    "eval/train_loss": 0.0,
                    "eval/val_loss": 0.0,
                    "eval/val_ppl": 1.0,
                    "train/lr": 0.0,
                    "train/iter": 0,
                    "train/mfu": 0.0,
                    "train/dt_ms": 0.0,
                    "eval/iter": 0,
                })
            except Exception as e:
                logger.warning(f"WandB initialization failed: {e}")
                self.use_wandb = False
        
        # Load tokenizer
        self.load_tokenizer()
        
        # Prepare data
        self.prepare_data()
        
        # Initialize model
        self.init_model()
        
        # Resume from checkpoint if specified
        if self.resume_from_checkpoint:
            self.resume_from_checkpoint_file()
        
    def load_tokenizer(self):
        """Load the historical tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.tokenizer_dir))
            logger.info("Historical tokenizer loaded successfully")
            logger.info(f"   Vocabulary size: {self.tokenizer.vocab_size:,}")
            logger.info(f"   Model max length: {self.tokenizer.model_max_length}")
            return True
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            return False
    
    def prepare_data(self):
        """Prepare tokenized data in binary format"""
        logger.info("Preparing tokenized data...")
        
        # Check if binary files already exist
        data_dir = self.data_dir / "tokenized_data"
        train_file = data_dir / "train.bin"
        val_file = data_dir / "val.bin"
        meta_file = data_dir / "meta.pkl"
        
        if train_file.exists() and val_file.exists() and meta_file.exists():
            logger.info("Found existing tokenized data, loading...")
            
            # Load metadata
            with open(meta_file, 'rb') as f:
                meta = pickle.load(f)
            
            logger.info(f"Loaded existing data:")
            logger.info(f"  Train tokens: {meta['train_tokens']:,}")
            logger.info(f"  Val tokens: {meta['val_tokens']:,}")
            logger.info(f"  Block size: {meta['block_size']}")
            logger.info(f"  Vocab size: {meta['vocab_size']:,}")
            
            self.data_dir = data_dir
            return
        
        # Load corpus and tokenize
        logger.info("Tokenizing corpus from scratch...")
        corpus_file = self.data_dir / "london_historical_corpus_comprehensive.txt"
        if not corpus_file.exists():
            raise FileNotFoundError(f"Corpus file not found: {corpus_file}")
        
        with open(corpus_file, 'r', encoding='utf-8', errors='ignore') as f:
            corpus_text = f.read()
        
        # Tokenize the entire corpus
        logger.info("Tokenizing corpus...")
        tokens = self.tokenizer.encode(corpus_text, add_special_tokens=False)
        logger.info(f"Tokenized {len(tokens):,} tokens")
        
        # Split into train/val
        split_idx = int(0.9 * len(tokens))
        train_tokens = tokens[:split_idx]
        val_tokens = tokens[split_idx:]
        
        # Save as binary files
        data_dir.mkdir(exist_ok=True)
        
        # Save tokenized data
        np.array(train_tokens, dtype=np.uint16).tofile(train_file)
        np.array(val_tokens, dtype=np.uint16).tofile(val_file)
        
        # Save metadata
        meta = {
            'vocab_size': self.tokenizer.vocab_size,
            'block_size': self.block_size,
            'train_tokens': len(train_tokens),
            'val_tokens': len(val_tokens)
        }
        
        with open(meta_file, 'wb') as f:
            pickle.dump(meta, f)
        
        logger.info(f"Saved {len(train_tokens):,} train tokens to {train_file}")
        logger.info(f"Saved {len(val_tokens):,} val tokens to {val_file}")
        
        self.data_dir = data_dir
    
    def get_batch(self, split):
        """Get a batch of data for training/validation"""
        data_file = self.data_dir / f"{split}.bin"
        
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        # Load data
        data = np.memmap(data_file, dtype=np.uint16, mode='r')
        
        # Sample random sequences
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+self.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+self.block_size]).astype(np.int64)) for i in ix])
        
        if self.device == 'cuda':
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        
        return x, y
    
    def init_model(self):
        """Initialize the model"""
        logger.info("Initializing model...")
        
        # Load metadata
        meta_path = self.data_dir / "meta.pkl"
        meta_vocab_size = None
        if meta_path.exists():
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            meta_vocab_size = meta['vocab_size']
            logger.info(f"Found vocab_size = {meta_vocab_size}")
        
        # Model config
        model_args = dict(
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_embd=self.n_embd,
            block_size=self.block_size,
            bias=self.bias,
            vocab_size=meta_vocab_size if meta_vocab_size is not None else 30000,
            dropout=self.dropout
        )
        
        # Create model
        gptconf = SimpleGPTConfig(**model_args)
        self.model = SimpleGPT(gptconf)
        self.model.to(self.device)
        
        # Initialize optimizer
        if self.dtype == 'float16':
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None
        self.optimizer = self.model.configure_optimizers(
            weight_decay=0.1,
            learning_rate=self.learning_rate,
            betas=(0.9, 0.95),
            device_type='cuda' if 'cuda' in self.device else 'cpu'
        )
        
        # Compile model
        if torch.cuda.is_available() and self.slm_config.get("enable_compile", True):
            logger.info("Compiling model...")
            try:
                self.model = torch.compile(self.model, mode='reduce-overhead')
            except Exception:
                self.model = torch.compile(self.model)
        
        # DDP wrapper
        if self.ddp:
            self.model = DDP(self.model, device_ids=[self.ddp_local_rank])
            # Get parameter count from the underlying model
            param_count = self.model.module.get_num_params()
        else:
            param_count = self.model.get_num_params()
        
        logger.info(f"Model initialized with {param_count:,} parameters")
    
    def resume_from_checkpoint_file(self):
        """Resume training from a checkpoint file"""
        if not self.resume_from_checkpoint:
            return
            
        checkpoint_path = Path(self.resume_from_checkpoint)
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint file not found: {checkpoint_path}")
            return
            
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model state
            raw_model = self.model.module if self.ddp else self.model
            raw_model.load_state_dict(checkpoint['model'])
            logger.info("Model state loaded successfully")
            
            # Load optimizer state
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("Optimizer state loaded successfully")
            
            # Get iteration number and best validation loss
            self.start_iter = checkpoint.get('iter_num', 0)
            self.best_val_loss = checkpoint.get('best_val_loss', 1e9)
            
            logger.info(f"Resuming from iteration: {self.start_iter}")
            logger.info(f"Best validation loss so far: {self.best_val_loss:.4f}")
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            logger.info("Starting training from scratch...")
            self.start_iter = 0
            self.best_val_loss = 1e9
        else:
            self.start_iter = 0
            self.best_val_loss = 1e9
    
    def estimate_loss(self):
        """Estimate loss on train/val sets"""
        out = {}
        raw_model = self.model.module if self.ddp else self.model
        raw_model.eval()
        with torch.no_grad():
            for split in ['train', 'val']:
                losses = torch.zeros(self.eval_iters)
                for k in range(self.eval_iters):
                    X, Y = self.get_batch(split)
                    with torch.amp.autocast(device_type='cuda' if 'cuda' in self.device else 'cpu', dtype=torch.bfloat16 if self.dtype == 'bfloat16' else torch.float16):
                        logits, loss = self.model(X, Y)
                    losses[k] = loss.item()
                out[split] = losses.mean()
        raw_model.train()
        # try to free any transient eval memory
        try:
            if 'cuda' in self.device:
                import gc
                gc.collect()
                torch.cuda.empty_cache()
        except Exception:
            pass
        return out
    
    def get_lr(self, it):
        """Learning rate schedule"""
        warmup_iters = 500
        lr_decay_iters = self.max_iters
        min_lr = self.learning_rate * 0.1
        
        if it < warmup_iters:
            return self.learning_rate * (it + 1) / (warmup_iters + 1)
        if it > lr_decay_iters:
            return min_lr
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (self.learning_rate - min_lr)
    
    def cleanup_old_checkpoints(self, keep_last=3):
        """Clean up old checkpoints, keeping only the last N"""
        if not self.master_process:
            return  # Only master process should clean up
            
        try:
            # Find all checkpoint files
            checkpoint_files = list(self.output_dir.glob("checkpoint-*.pt"))
            
            if len(checkpoint_files) <= keep_last:
                return  # Not enough checkpoints to clean up
            
            # Sort by modification time (newest first)
            checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Keep the newest ones, delete the rest
            files_to_delete = checkpoint_files[keep_last:]
            
            for file_path in files_to_delete:
                try:
                    file_path.unlink()
                    logger.info(f"Deleted old checkpoint: {file_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to delete checkpoint {file_path.name}: {e}")
                    
        except Exception as e:
            logger.warning(f"Checkpoint cleanup failed: {e}")
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        # Training loop
        X, Y = self.get_batch('train')
        t0 = time.time()
        local_iter_num = 0
        raw_model = self.model.module if self.ddp else self.model
        running_mfu = -1.0
        
        iter_num = getattr(self, 'start_iter', 0)
        best_val_loss = getattr(self, 'best_val_loss', 1e9)
        
        logger.info(f"Starting training on {self.device}...")
        logger.info(f"Model parameters: {raw_model.get_num_params():,}")
        
        while True:
            # Learning rate schedule
            lr = self.get_lr(iter_num)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            # Evaluation and checkpointing
            if self.master_process and self.eval_interval > 0 and iter_num > 0 and (iter_num % self.eval_interval == 0):
                losses = self.estimate_loss()
                logger.info(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                
                # Log to WandB
                if self.use_wandb:
                    wandb.log({
                        "eval/train_loss": losses['train'],
                        "eval/val_loss": losses['val'],
                        "eval/val_ppl": math.exp(losses['val']),
                        "eval/iter": iter_num,
                    })
                
                if losses['val'] < best_val_loss:
                    best_val_loss = losses['val']
                    if iter_num > 0:
                        checkpoint = {
                            'model': raw_model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'iter_num': iter_num,
                            'best_val_loss': best_val_loss,
                        }
                        checkpoint_path = self.output_dir / f'checkpoint-{iter_num}.pt'
                        logger.info(f"Saving checkpoint to {checkpoint_path}")
                        torch.save(checkpoint, checkpoint_path)
                        
                        # Clean up old checkpoints - keep only last 3
                        self.cleanup_old_checkpoints()
                # free memory after eval+save
                try:
                    if 'cuda' in self.device:
                        import gc
                        gc.collect()
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            
            # Training step
            with torch.amp.autocast(
                device_type='cuda' if 'cuda' in self.device else 'cpu',
                dtype=(torch.bfloat16 if self.dtype == 'bfloat16' else (torch.float16 if self.dtype == 'float16' else torch.float32))
            ):
                logits, loss = self.model(X, Y)
            
            X, Y = self.get_batch('train')
            
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
                self.optimizer.step()
            
            self.optimizer.zero_grad(set_to_none=True)
            
            # Timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if iter_num % self.log_interval == 0 and self.master_process:
                lossf = loss.item()
                if local_iter_num >= 5:
                    mfu = raw_model.estimate_mfu(self.batch_size, dt)
                    running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
                logger.info(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
                
                # Log to WandB - loss first for better mobile UI
                if self.use_wandb:
                    wandb.log({
                        "train/loss": lossf,
                        "train/lr": lr,
                        "train/iter": iter_num,
                        "train/mfu": running_mfu * 100 if running_mfu > 0 else 0,
                        "train/dt_ms": dt * 1000,
                    })
            
            iter_num += 1
            local_iter_num += 1
            
            # Termination
            if iter_num > self.max_iters:
                break
        
        # Save final checkpoint
        if self.master_process:
            final_checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
            }
            final_checkpoint_path = self.output_dir / f'checkpoint-{iter_num}.pt'
            logger.info(f"Saving final checkpoint to {final_checkpoint_path}")
            torch.save(final_checkpoint, final_checkpoint_path)
            
            # Final cleanup to ensure only last 3 checkpoints remain
            self.cleanup_old_checkpoints()
        
        if self.ddp:
            destroy_process_group()
        
        # Finish WandB run
        if self.use_wandb:
            wandb.finish()
        
        logger.info("Training completed!")

def main():
    parser = argparse.ArgumentParser(description='Simple SLM Training')
    parser.add_argument("--data_dir", type=str, default="data/london_historical",
                       help="Directory containing training data")
    parser.add_argument("--tokenizer_dir", type=str, default="09_models/tokenizers/london_historical_tokenizer",
                       help="Directory containing tokenizer")
    parser.add_argument("--output_dir", type=str, default="09_models/checkpoints/slm",
                       help="Directory to save trained model")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="Path to checkpoint file to resume from")
    
    args = parser.parse_args()
    
    # DDP setup
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
    
    # Set random seed
    torch.manual_seed(1337 + seed_offset)
    
    # Create trainer
    trainer = SimpleLondonHistoricalTrainer(
        data_dir=args.data_dir,
        tokenizer_dir=args.tokenizer_dir,
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume_from_checkpoint
    )
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()
