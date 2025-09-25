#!/usr/bin/env python3
"""
Check available checkpoints and their details
"""

import os
import sys
import torch
from pathlib import Path

def check_checkpoints(checkpoint_dir):
    """Check available checkpoints"""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return
    
    print(f"📁 Checkpoint directory: {checkpoint_dir}")
    print("=" * 60)
    
    # Find all checkpoint files
    checkpoint_files = list(checkpoint_dir.glob("checkpoint-*.pt"))
    if not checkpoint_files:
        print("❌ No checkpoint files found")
        return
    
    # Sort by iteration number
    checkpoint_files.sort(key=lambda x: int(x.stem.split('-')[1]))
    
    print(f"Found {len(checkpoint_files)} checkpoint(s):")
    print()
    
    for i, checkpoint_file in enumerate(checkpoint_files, 1):
        try:
            # Load checkpoint to get details
            checkpoint = torch.load(checkpoint_file, map_location='cpu')
            
            # Get file size
            file_size_mb = checkpoint_file.stat().st_size / (1024 * 1024)
            
            print(f"{i:2d}. {checkpoint_file.name}")
            print(f"    Size: {file_size_mb:.1f} MB")
            print(f"    Iteration: {checkpoint.get('iter_num', 'unknown')}")
            print(f"    Val Loss: {checkpoint.get('best_val_loss', 'unknown'):.4f}")
            print(f"    Modified: {datetime.fromtimestamp(checkpoint_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
            print()
            
        except Exception as e:
            print(f"{i:2d}. {checkpoint_file.name} - ❌ Error loading: {e}")
            print()
    
    # Show latest checkpoint info
    latest = checkpoint_files[-1]
    print(f"🎯 Latest checkpoint: {latest.name}")
    print(f"   Use this for testing: python test_slm_model.py --checkpoint_dir {checkpoint_dir}")

if __name__ == "__main__":
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description='Check available checkpoints')
    parser.add_argument("--checkpoint_dir", type=str, default="09_models/checkpoints/slm",
                       help="Directory containing checkpoints")
    
    args = parser.parse_args()
    check_checkpoints(args.checkpoint_dir)
