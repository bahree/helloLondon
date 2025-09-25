#!/usr/bin/env python3
"""
Multi-GPU Training Launcher for London Historical LLM
Supports single GPU, multi-GPU, and CPU-only training
"""

import os
import sys
import subprocess
import argparse
import torch
from pathlib import Path

def check_gpu_availability():
    """Check GPU availability and configuration"""
    print(f"Checking GPU availability...")
    
    if not torch.cuda.is_available():
        print(f"CUDA not available - will use CPU-only training")
        return 0, "cpu"
    
    gpu_count = torch.cuda.device_count()
    print(f"Found {gpu_count} GPU(s)")
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    return gpu_count, "cuda"

def setup_wandb():
    """Setup WandB configuration"""
    print(f"Setting up WandB...")
    
    # Check for WandB API key
    wandb_key = os.getenv('WANDB_API_KEY')
    if not wandb_key:
        print(f" WandB API key not found. Set WANDB_API_KEY environment variable.")
        print("   You can get your API key from: https://wandb.ai/authorize")
        print("   Or run: export WANDB_API_KEY=your_key_here")
        return False
    
    print(f"WandB API key found")
    return True

def launch_training(gpu_count, device_type, args):
    """Launch training with appropriate configuration"""
    print(f"\n🚀 Launching training on {device_type}...")
    
    # Base command
    cmd = [
        sys.executable, "04_training/train_model.py",
        "--data_dir", args.data_dir,
        "--tokenizer_dir", args.tokenizer_dir,
        "--output_dir", args.output_dir,
        "--model_name", args.model_name,
        "--max_length", str(args.max_length),
        "--batch_size", str(args.batch_size),
        "--learning_rate", str(args.learning_rate),
        "--num_epochs", str(args.num_epochs),
        "--warmup_steps", str(args.warmup_steps)
    ]
    
    if gpu_count > 1:
        # Multi-GPU training with torchrun
        print(f"Starting multi-GPU training on {gpu_count} GPUs...")
        torchrun_cmd = [
            "torchrun",
            "--nproc_per_node", str(gpu_count),
            "--nnodes", "1",
            "--node_rank", "0",
            "--master_addr", "localhost",
            "--master_port", "12355"
        ] + cmd
        
        print(f"Command: {' '.join(torchrun_cmd)}")
        subprocess.run(torchrun_cmd, check=True)
        
    elif gpu_count == 1:
        # Single GPU training
        print(f"Starting single GPU training...")
        print(f"Command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
    else:
        # CPU-only training
        print(f"Starting CPU-only training...")
        print(f"Command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(description="Multi-GPU Training Launcher")
    parser.add_argument("--data_dir", type=str, default="data/london_historical",
                       help="Directory containing training data")
    parser.add_argument("--tokenizer_dir", type=str, 
                       default="09_models/tokenizers/london_historical_tokenizer",
                       help="Directory containing tokenizer")
    parser.add_argument("--output_dir", type=str, default="09_models/checkpoints",
                       help="Directory to save trained model")
    parser.add_argument("--model_name", type=str, default="gpt2-medium",
                       help="Base model name")
    parser.add_argument("--max_length", type=int, default=1024,
                       help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-5,
                       help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=5,
                       help="Number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=500,
                       help="Number of warmup steps")
    parser.add_argument("--skip_wandb", action="store_true",
                       help="Skip WandB setup")
    
    args = parser.parse_args()
    
    print("🏛️ London Historical LLM - Multi-GPU Training Launcher")
    print("=" * 60)
    
    # Check GPU availability
    gpu_count, device_type = check_gpu_availability()
    
    # Setup WandB if not skipped
    if not args.skip_wandb:
        wandb_available = setup_wandb()
        if not wandb_available:
            print(f" Continuing without WandB logging...")
    else:
        print("⏭️  Skipping WandB setup...")
    
    # Adjust batch size based on GPU count
    if gpu_count > 1:
        # For multi-GPU, reduce per-device batch size
        original_batch_size = args.batch_size
        args.batch_size = max(1, args.batch_size // gpu_count)
        print(f"Adjusted batch size: {original_batch_size} -> {args.batch_size} per GPU")
        print(f"Effective batch size: {args.batch_size * gpu_count}")
    
    # Launch training
    try:
        launch_training(gpu_count, device_type, args)
        print("\n🎉 Training completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code: {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
