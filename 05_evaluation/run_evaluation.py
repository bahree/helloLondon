#!/usr/bin/env python3
"""
Evaluation Launcher for London Historical SLM
Easy-to-use script for running different types of evaluations
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    print(f"Command: {command}")
    
    # Create log file for this command
    log_file = f"logs/{description.lower().replace(' ', '_')}.log"
    os.makedirs("logs", exist_ok=True)
    
    try:
        with open(log_file, 'w') as f:
            f.write(f"Command: {command}\n")
            f.write(f"Description: {description}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n")
        
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        
        # Write success output to log
        with open(log_file, 'a') as f:
            f.write("SUCCESS OUTPUT:\n")
            f.write(result.stdout)
            if result.stderr:
                f.write("\nSTDERR:\n")
                f.write(result.stderr)
        
        print(f"{description} completed successfully")
        print(f"Log saved to: {log_file}")
        return True
    except subprocess.CalledProcessError as e:
        # Write error output to log
        with open(log_file, 'a') as f:
            f.write("ERROR OUTPUT:\n")
            f.write(f"Return code: {e.returncode}\n")
            f.write(f"STDOUT:\n{e.stdout}\n")
            f.write(f"STDERR:\n{e.stderr}\n")
        
        print(f"{description} failed: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        print(f"Full error log saved to: {log_file}")
        return False

def check_dependencies():
    """Check if required dependencies are installed"""
    print("Checking dependencies...")
    
    required_packages = [
        'torch', 'transformers', 'datasets', 'nltk', 'rouge_score', 
        'textstat', 'matplotlib', 'seaborn', 'pandas', 'numpy',
        'spacy', 'bert_score', 'openai', 'aiohttp'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Please install with: pip install -r requirements.txt")
        return False
    
    print("All required packages are installed")
    return True

def setup_evaluation():
    """Setup evaluation environment"""
    print("Setting up evaluation environment...")
    
    # Check if we're in the right directory
    if not Path("comprehensive_evaluator.py").exists():
        print("Please run this script from the 05_evaluation directory")
        return False
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        return False
    
    # Download NLTK data
    if not run_command("python -c \"import nltk; nltk.download('punkt'); nltk.download('stopwords')\"", "Downloading NLTK data"):
        print("NLTK data download failed. Some evaluations may be limited.")
    
    # Download spaCy model
    if not run_command("python -m spacy download en_core_web_sm", "Downloading spaCy model"):
        print("spaCy model download failed. Some evaluations may be limited.")
    
    # Create directories
    Path("results").mkdir(exist_ok=True)
    Path("quick_results").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    print("Setup completed successfully!")
    return True

def run_quick_evaluation(model_dir, tokenizer_dir, output_dir, device="cpu"):
    """Run quick evaluation"""
    print("🏃 Running quick evaluation...")
    
    command = f"python quick_eval.py --model_dir {model_dir} --tokenizer_dir {tokenizer_dir} --output_dir {output_dir} --device {device}"
    
    if device == "cpu":
        print("Using CPU for evaluation (safe default)")
    else:
        print("Using GPU for evaluation")
    
    if run_command(command, "Quick evaluation"):
        print("Quick evaluation completed!")
        print(f"Results saved to: {output_dir}")
        return True
    else:
        print("Quick evaluation failed!")
        return False

def run_comprehensive_evaluation(model_dir, tokenizer_dir, output_dir, openai_api_key=None, device="cpu"):
    """Run comprehensive evaluation"""
    print("Running comprehensive evaluation...")
    
    command = f"python comprehensive_evaluator.py --model_dir {model_dir} --tokenizer_dir {tokenizer_dir} --output_dir {output_dir} --device {device}"
    
    if openai_api_key:
        command += f" --openai_api_key {openai_api_key}"
        print("Using OpenAI API for G-Eval")
    else:
        print("No OpenAI API key provided. G-Eval will be disabled.")
    
    if device == "cpu":
        print("Using CPU for evaluation (safe default)")
    else:
        print("Using GPU for evaluation")
    
    if run_command(command, "Comprehensive evaluation"):
        print("Comprehensive evaluation completed!")
        print(f"Results saved to: {output_dir}")
        return True
    else:
        print("Comprehensive evaluation failed!")
        return False

def generate_historical_dataset():
    """Generate historical evaluation dataset"""
    print("Generating historical evaluation dataset...")
    
    if run_command("python historical_evaluation_dataset.py", "Historical dataset generation"):
        print("Historical dataset generated!")
        return True
    else:
        print("Historical dataset generation failed!")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Evaluation Launcher for London Historical SLM")
    parser.add_argument("--mode", type=str, choices=["setup", "quick", "comprehensive", "dataset", "all"], 
                       default="quick", help="Evaluation mode to run")
    parser.add_argument("--model_dir", type=str, default="09_models/checkpoints",
                       help="Directory containing trained model")
    parser.add_argument("--tokenizer_dir", type=str, 
                       default="09_models/tokenizers/london_historical_tokenizer",
                       help="Directory containing tokenizer")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Directory to save evaluation results")
    parser.add_argument("--openai_api_key", type=str, default=None,
                       help="OpenAI API key for G-Eval (optional)")
    parser.add_argument("--skip_dependency_check", action="store_true",
                       help="Skip dependency check")
    parser.add_argument("--device", type=str, choices=["cpu", "gpu"], default="cpu",
                       help="Device to use for evaluation (default: cpu for safety)")
    
    args = parser.parse_args()
    
    print("London Historical SLM Evaluation Launcher")
    print("=" * 50)
    
    # Check dependencies unless skipped
    if not args.skip_dependency_check:
        if not check_dependencies():
            print("Dependency check failed. Please install missing packages.")
            return False
    
    # Run based on mode
    if args.mode == "setup":
        return setup_evaluation()
    
    elif args.mode == "quick":
        return run_quick_evaluation(args.model_dir, args.tokenizer_dir, "quick_results", args.device)
    
    elif args.mode == "comprehensive":
        return run_comprehensive_evaluation(args.model_dir, args.tokenizer_dir, args.output_dir, args.openai_api_key, args.device)
    
    elif args.mode == "dataset":
        return generate_historical_dataset()
    
    elif args.mode == "all":
        print("Running all evaluation modes...")
        
        # Generate dataset
        if not generate_historical_dataset():
            print("Dataset generation failed, continuing...")
        
        # Run quick evaluation
        if not run_quick_evaluation(args.model_dir, args.tokenizer_dir, "quick_results", args.device):
            print("Quick evaluation failed, continuing...")
        
        # Run comprehensive evaluation
        if not run_comprehensive_evaluation(args.model_dir, args.tokenizer_dir, args.output_dir, args.openai_api_key, args.device):
            print("Comprehensive evaluation failed, continuing...")
        
        print("All evaluations completed!")
        return True
    
    else:
        print(f"Unknown mode: {args.mode}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
