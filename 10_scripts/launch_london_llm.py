#!/usr/bin/env python3
"""
London Historical LLM - Main Launch Script
Interactive launcher for the complete system
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import Dict, Any, List
import time
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

class LondonHistoricalLLMLauncher:
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.system_info = self.get_system_info()
        
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        import platform
        import torch
        
        return {
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'platform': platform.system(),
            'architecture': platform.architecture()[0],
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'project_root': str(self.project_root)
        }
    
    def print_banner(self):
        """Print welcome banner"""
        print("\n" + "="*80)
        print("🏛️ LONDON HISTORICAL LLM (1500-1850)")
        print("="*80)
        print("A comprehensive system for training a Small Language Model")
        print("on historical London texts from 1500-1850")
        print("="*80)
        print(f"📁 Project Root: {self.project_root}")
        print(f"🐍 Python Version: {self.system_info['python_version']}")
        print(f"💻 Platform: {self.system_info['platform']} {self.system_info['architecture']}")
        print(f"🚀 CUDA Available: {self.system_info['cuda_available']}")
        if self.system_info['cuda_available']:
            print(f"🎮 CUDA Devices: {self.system_info['cuda_device_count']}")
        print("="*80)
    
    def print_menu(self):
        """Print main menu"""
        print("\n📋 MAIN MENU")
        print("-" * 40)
        print("1. 🏗️  Setup Environment")
        print("2. 📚 Download Historical Data")
        print("3. 🔤 Train Custom Tokenizer")
        print("4. 🤖 Train Language Model")
        print("5. 📊 Evaluate Model")
        print("6. 🧪 Run System Tests")
        print("7. 🚀 Complete Pipeline (All Steps)")
        print("8. 📊 View System Status")
        print("9. ❓ Help & Documentation")
        print("0. 🚪 Exit")
        print("-" * 40)
    
    def run_command(self, command: str, description: str, cwd: str = None) -> bool:
        """Run a command and return success status"""
        print(f"\n🔄 {description}")
        print(f"Command: {command}")
        print("-" * 50)
        
        try:
            if cwd:
                cwd_path = self.project_root / cwd
                cwd_path.mkdir(parents=True, exist_ok=True)
            else:
                cwd_path = self.project_root
            
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd_path,
                capture_output=False,
                text=True
            )
            
            if result.returncode == 0:
                print(f"✅ {description} completed successfully!")
                return True
            else:
                print(f"❌ {description} failed with return code: {result.returncode}")
                return False
                
        except Exception as e:
            print(f"❌ {description} failed with error: {e}")
            return False
    
    def setup_environment(self) -> bool:
        """Setup the environment"""
        print("\n🏗️ SETTING UP ENVIRONMENT")
        print("=" * 50)
        
        command = "python setup_environment.py"
        return self.run_command(command, "Environment Setup", "01_environment")
    
    def download_data(self) -> bool:
        """Download historical data"""
        print("\n📚 DOWNLOADING HISTORICAL DATA")
        print("=" * 50)
        
        command = "python download_historical_data.py"
        return self.run_command(command, "Historical Data Download", "02_data_collection")
    
    def train_tokenizer(self) -> bool:
        """Train custom tokenizer"""
        print("\n🔤 TRAINING CUSTOM TOKENIZER")
        print("=" * 50)
        
        command = "python train_tokenizer.py"
        return self.run_command(command, "Tokenizer Training", "03_tokenizer")
    
    def train_model(self) -> bool:
        """Train language model"""
        print("\n🤖 TRAINING LANGUAGE MODEL")
        print("=" * 50)
        
        command = "python train_model.py"
        return self.run_command(command, "Model Training", "04_training")
    
    def evaluate_model(self) -> bool:
        """Evaluate model"""
        print("\n📊 EVALUATING MODEL")
        print("=" * 50)
        
        command = "python evaluate_model.py"
        return self.run_command(command, "Model Evaluation", "05_evaluation")
    
    def run_tests(self) -> bool:
        """Run system tests"""
        print("\n🧪 RUNNING SYSTEM TESTS")
        print("=" * 50)
        
        command = "python test_system.py"
        return self.run_command(command, "System Testing", "06_testing")
    
    def run_complete_pipeline(self) -> bool:
        """Run the complete pipeline"""
        print("\n🚀 RUNNING COMPLETE PIPELINE")
        print("=" * 50)
        print("This will run all steps in sequence:")
        print("1. Environment Setup")
        print("2. Data Download")
        print("3. Tokenizer Training")
        print("4. Model Training")
        print("5. Model Evaluation")
        print("6. System Testing")
        print("=" * 50)
        
        # Ask for confirmation
        response = input("\n⚠️  This will take several hours. Continue? (y/N): ")
        if response.lower() != 'y':
            print("❌ Pipeline cancelled by user")
            return False
        
        steps = [
            (self.setup_environment, "Environment Setup"),
            (self.download_data, "Data Download"),
            (self.train_tokenizer, "Tokenizer Training"),
            (self.train_model, "Model Training"),
            (self.evaluate_model, "Model Evaluation"),
            (self.run_tests, "System Testing")
        ]
        
        start_time = time.time()
        successful_steps = 0
        
        for step_func, step_name in steps:
            print(f"\n🔄 Starting: {step_name}")
            if step_func():
                successful_steps += 1
                print(f"✅ {step_name} completed successfully!")
            else:
                print(f"❌ {step_name} failed!")
                print("⚠️  Pipeline stopped due to failure")
                break
        
        end_time = time.time()
        duration = (end_time - start_time) / 60
        
        print(f"\n📊 PIPELINE SUMMARY")
        print("=" * 50)
        print(f"Successful steps: {successful_steps}/{len(steps)}")
        print(f"Total time: {duration:.1f} minutes")
        
        if successful_steps == len(steps):
            print("🎉 Complete pipeline finished successfully!")
            return True
        else:
            print("⚠️  Pipeline completed with some failures")
            return False
    
    def view_system_status(self):
        """View system status"""
        print("\n📊 SYSTEM STATUS")
        print("=" * 50)
        
        # Check directories
        directories = [
            "01_environment",
            "02_data_collection",
            "03_tokenizer", 
            "04_training",
            "05_evaluation",
            "06_testing",
            "07_utilities",
            "08_documentation",
            "09_models",
            "10_scripts"
        ]
        
        print("📁 Directory Status:")
        for dir_name in directories:
            dir_path = self.project_root / dir_name
            status = "✅" if dir_path.exists() else "❌"
            print(f"  {status} {dir_name}")
        
        # Check key files
        key_files = [
            "data/london_historical/london_historical_corpus.txt",
            "09_models/tokenizers/london_historical_tokenizer/tokenizer.json",
            "09_models/checkpoints/config.json",
            "05_evaluation/results/evaluation_results.json"
        ]
        
        print("\n📄 Key Files Status:")
        for file_name in key_files:
            file_path = self.project_root / file_name
            status = "✅" if file_path.exists() else "❌"
            size = f" ({file_path.stat().st_size / (1024*1024):.1f} MB)" if file_path.exists() else ""
            print(f"  {status} {file_name}{size}")
        
        # Check data size
        data_dir = self.project_root / "data" / "london_historical"
        if data_dir.exists():
            total_size = sum(f.stat().st_size for f in data_dir.rglob('*') if f.is_file())
            print(f"\n📊 Data Directory Size: {total_size / (1024*1024):.1f} MB")
        
        print("\n💡 Run 'python test_system.py' for detailed system testing")
    
    def show_help(self):
        """Show help and documentation"""
        print("\n❓ HELP & DOCUMENTATION")
        print("=" * 50)
        print("📚 Available Documentation:")
        print("  - README.md: Main project documentation")
        print("  - 08_documentation/: Detailed guides for each component")
        print("  - 06_testing/test_summary.txt: Latest test results")
        print("  - 05_evaluation/results/: Model evaluation results")
        
        print("\n🔧 Quick Commands:")
        print("  - Setup: cd 01_environment && python setup_environment.py")
        print("  - Download: cd 02_data_collection && python download_historical_data.py")
        print("  - Tokenizer: cd 03_tokenizer && python train_tokenizer.py")
        print("  - Training: cd 04_training && python train_model.py")
        print("  - Evaluation: cd 05_evaluation && python evaluate_model.py")
        print("  - Testing: cd 06_testing && python test_system.py")
        
        print("\n📊 Expected Timeline:")
        print("  - Environment Setup: 5-10 minutes")
        print("  - Data Download: 30-60 minutes")
        print("  - Tokenizer Training: 10-30 minutes")
        print("  - Model Training: 2-7 days (depending on hardware)")
        print("  - Model Evaluation: 10-30 minutes")
        print("  - System Testing: 5-10 minutes")
        
        print("\n💡 Tips:")
        print("  - Start with environment setup")
        print("  - Check system status before training")
        print("  - Monitor logs for progress")
        print("  - Use GPU for faster training")
        print("  - Check failed downloads for manual retry")
    
    def run_interactive(self):
        """Run interactive launcher"""
        while True:
            self.print_banner()
            self.print_menu()
            
            try:
                choice = input("\n🎯 Select an option (0-9): ").strip()
                
                if choice == "0":
                    print("\n👋 Goodbye! Thanks for using London Historical LLM!")
                    break
                elif choice == "1":
                    self.setup_environment()
                elif choice == "2":
                    self.download_data()
                elif choice == "3":
                    self.train_tokenizer()
                elif choice == "4":
                    self.train_model()
                elif choice == "5":
                    self.evaluate_model()
                elif choice == "6":
                    self.run_tests()
                elif choice == "7":
                    self.run_complete_pipeline()
                elif choice == "8":
                    self.view_system_status()
                elif choice == "9":
                    self.show_help()
                else:
                    print("❌ Invalid option. Please select 0-9.")
                
                if choice != "0":
                    input("\n⏸️  Press Enter to continue...")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Thanks for using London Historical LLM!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                input("\n⏸️  Press Enter to continue...")
    
    def run_non_interactive(self, step: str):
        """Run non-interactive mode"""
        self.print_banner()
        
        if step == "setup":
            return self.setup_environment()
        elif step == "download":
            return self.download_data()
        elif step == "tokenizer":
            return self.train_tokenizer()
        elif step == "train":
            return self.train_model()
        elif step == "evaluate":
            return self.evaluate_model()
        elif step == "test":
            return self.run_tests()
        elif step == "pipeline":
            return self.run_complete_pipeline()
        elif step == "status":
            self.view_system_status()
            return True
        else:
            print(f"❌ Unknown step: {step}")
            return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="London Historical LLM Launcher")
    parser.add_argument("--step", type=str, default="interactive",
                       choices=["interactive", "setup", "download", "tokenizer", "train", "evaluate", "test", "pipeline", "status"],
                       help="Step to run (default: interactive)")
    parser.add_argument("--project_root", type=str, default=".",
                       help="Project root directory")
    
    args = parser.parse_args()
    
    # Initialize launcher
    launcher = LondonHistoricalLLMLauncher(args.project_root)
    
    try:
        if args.step == "interactive":
            launcher.run_interactive()
        else:
            success = launcher.run_non_interactive(args.step)
            sys.exit(0 if success else 1)
            
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye! Thanks for using London Historical LLM!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
