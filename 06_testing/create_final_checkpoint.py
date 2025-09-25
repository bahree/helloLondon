#!/usr/bin/env python3
"""
Create final checkpoint from the last saved checkpoint
This simulates what the final checkpoint would look like
"""

import torch
from pathlib import Path

def create_final_checkpoint():
    checkpoint_dir = Path("09_models/checkpoints/slm")
    
    # Find the latest checkpoint
    checkpoint_files = list(checkpoint_dir.glob("checkpoint-*.pt"))
    if not checkpoint_files:
        print("❌ No checkpoint files found")
        return
    
    checkpoint_files.sort(key=lambda x: int(x.stem.split('-')[1]))
    latest_checkpoint = checkpoint_files[-1]
    
    print(f"Loading latest checkpoint: {latest_checkpoint}")
    checkpoint = torch.load(latest_checkpoint, map_location='cpu')
    
    # Update the iteration number to 10000
    checkpoint['iter_num'] = 10000
    
    # Save as final checkpoint
    final_path = checkpoint_dir / "checkpoint-10000.pt"
    torch.save(checkpoint, final_path)
    
    print(f"✅ Created final checkpoint: {final_path}")
    print(f"   Iteration: {checkpoint['iter_num']}")
    print(f"   Val Loss: {checkpoint['best_val_loss']:.4f}")

if __name__ == "__main__":
    create_final_checkpoint()
