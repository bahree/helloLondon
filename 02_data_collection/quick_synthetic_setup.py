#!/usr/bin/env python3
"""
Quick Setup for Synthetic Data Generation
One-command setup and generation of synthetic historical data
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, description, check=True):
    """Run a command and return success status"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed")
            return True
        else:
            print(f"❌ {description} failed: {result.stderr}")
            if check:
                return False
            else:
                print("⚠️ Continuing despite error...")
                return True
    except Exception as e:
        print(f"❌ Error during {description}: {e}")
        if check:
            return False
        else:
            print("⚠️ Continuing despite error...")
            return True

def check_ollama_running():
    """Check if Ollama is running"""
    try:
        result = subprocess.run(["curl", "-s", "http://localhost:11434/api/tags"], 
                              capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except:
        return False

def main():
    """Main function for quick synthetic data setup"""
    parser = argparse.ArgumentParser(description="Quick setup and generation of synthetic historical data")
    parser.add_argument("--num_documents", type=int, default=100,
                       help="Number of documents to generate")
    parser.add_argument("--skip_setup", action="store_true",
                       help="Skip Ollama setup (assume already configured)")
    parser.add_argument("--model_type", choices=["ollama", "openai"], default="ollama",
                       help="Model type to use")
    
    args = parser.parse_args()
    
    print("🚀 Quick Synthetic Data Setup")
    print("=" * 50)
    print(f"Target documents: {args.num_documents}")
    print(f"Model type: {args.model_type}")
    print("=" * 50)
    
    # Step 1: Setup Ollama (if not skipped)
    if not args.skip_setup and args.model_type == "ollama":
        print("\n📥 Setting up Ollama...")
        if not run_command("python setup_ollama.py", "Ollama setup"):
            print("❌ Ollama setup failed. Please run manually: python setup_ollama.py")
            return False
    
    # Step 2: Check if Ollama is running (for local models)
    if args.model_type == "ollama":
        if not check_ollama_running():
            print("❌ Ollama is not running. Please start it with: ollama serve")
            return False
        print("✅ Ollama is running")
    
    # Step 3: Generate synthetic data
    print(f"\n🤖 Generating {args.num_documents} synthetic documents...")
    cmd = f"python synthetic_data_generator.py --model_type {args.model_type} --num_documents {args.num_documents}"
    
    if not run_command(cmd, "Synthetic data generation"):
        print("❌ Synthetic data generation failed")
        return False
    
    # Step 4: Clean the generated data
    print("\n🧹 Cleaning synthetic data...")
    synthetic_dir = Path("data/london_historical/synthetic")
    if synthetic_dir.exists():
        cmd = f"python run_text_cleaning.py --input_dir {synthetic_dir}"
        run_command(cmd, "Text cleaning", check=False)  # Don't fail if cleaning has issues
    else:
        print("⚠️ Synthetic data directory not found, skipping cleaning")
    
    # Step 5: Show results
    print("\n📊 Results Summary:")
    print("=" * 30)
    
    # Check generated files
    synthetic_dir = Path("data/london_historical/synthetic")
    if synthetic_dir.exists():
        txt_files = list(synthetic_dir.glob("synthetic_*.txt"))
        print(f"Generated files: {len(txt_files)}")
        
        # Check corpus file
        corpus_file = synthetic_dir / "synthetic_corpus.txt"
        if corpus_file.exists():
            size_mb = corpus_file.stat().st_size / (1024 * 1024)
            print(f"Corpus file: {corpus_file.name} ({size_mb:.2f} MB)")
        
        # Check metadata
        metadata_file = synthetic_dir / "synthetic_metadata.json"
        if metadata_file.exists():
            print(f"Metadata: {metadata_file.name}")
    
    print("\n✅ Quick synthetic data setup completed!")
    print("\n🚀 Next steps:")
    print("1. Review generated data: ls -la data/london_historical/synthetic/")
    print("2. Combine with real data if desired")
    print("3. Continue with training")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
