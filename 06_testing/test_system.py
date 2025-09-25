#!/usr/bin/env python3
"""
System Testing for London Historical LLM
Comprehensive testing suite for all components
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import time
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    import torch
    import requests
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import numpy as np
    from tqdm import tqdm
    import pandas as pd
except ImportError as e:
    print(f"Missing required dependencies: {e}")
    print("Please install requirements: pip install -r requirements.txt")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('system_testing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LondonHistoricalSystemTester:
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.test_results = {
            'test_time': datetime.now().isoformat(),
            'project_root': str(self.project_root),
            'tests_passed': 0,
            'tests_failed': 0,
            'total_tests': 0,
            'test_details': {},
            'system_info': {},
            'recommendations': []
        }
    
    def test_environment(self) -> bool:
        """Test Python environment and dependencies"""
        logger.info("🐍 Testing Python environment...")
        
        test_name = "environment"
        self.test_results['total_tests'] += 1
        
        try:
            # Test Python version
            python_version = sys.version_info
            if python_version < (3, 8):
                raise ValueError(f"Python {python_version.major}.{python_version.minor} is not supported")
            
            # Test PyTorch
            if not torch.cuda.is_available():
                logger.warning(f"CUDA not available - will use CPU")
            
            # Test basic imports
            import transformers
            import tokenizers
            import datasets
            import accelerate
            
            self.test_results['test_details'][test_name] = {
                'status': 'passed',
                'python_version': f"{python_version.major}.{python_version.minor}.{python_version.micro}",
                'pytorch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'transformers_version': transformers.__version__,
                'tokenizers_version': tokenizers.__version__,
                'datasets_version': datasets.__version__,
                'accelerate_version': accelerate.__version__
            }
            
            self.test_results['tests_passed'] += 1
            logger.info(f"Environment test passed")
            return True
            
        except Exception as e:
            self.test_results['test_details'][test_name] = {
                'status': 'failed',
                'error': str(e)
            }
            self.test_results['tests_failed'] += 1
            logger.error(f"Environment test failed: {e}")
            return False
    
    def test_data_collection(self) -> bool:
        """Test data collection system"""
        logger.info(f"Testing data collection system...")
        
        test_name = "data_collection"
        self.test_results['total_tests'] += 1
        
        try:
            # Check if data collection script exists
            data_script = self.project_root / "02_data_collection" / "download_historical_data.py"
            if not data_script.exists():
                raise FileNotFoundError(f"Data collection script not found: {data_script}")
            
            # Check if data directory exists
            data_dir = self.project_root / "data" / "london_historical"
            if not data_dir.exists():
                logger.warning(f"Data directory does not exist - run data collection first")
            
            # Check for corpus file
            corpus_file = data_dir / "london_historical_corpus.txt"
            corpus_exists = corpus_file.exists()
            corpus_size = corpus_file.stat().st_size if corpus_exists else 0
            
            # Check for individual text files
            text_files = list(data_dir.glob("*.txt")) if data_dir.exists() else []
            
            self.test_results['test_details'][test_name] = {
                'status': 'passed',
                'data_script_exists': True,
                'data_directory_exists': data_dir.exists(),
                'corpus_file_exists': corpus_exists,
                'corpus_size_mb': corpus_size / (1024 * 1024) if corpus_exists else 0,
                'text_files_count': len(text_files),
                'text_files': [f.name for f in text_files[:10]]  # First 10 files
            }
            
            self.test_results['tests_passed'] += 1
            logger.info(f"Data collection test passed")
            return True
            
        except Exception as e:
            self.test_results['test_details'][test_name] = {
                'status': 'failed',
                'error': str(e)
            }
            self.test_results['tests_failed'] += 1
            logger.error(f"Data collection test failed: {e}")
            return False
    
    def test_tokenizer(self) -> bool:
        """Test tokenizer system"""
        logger.info(f"Testing tokenizer system...")
        
        test_name = "tokenizer"
        self.test_results['total_tests'] += 1
        
        try:
            # Check if tokenizer script exists
            tokenizer_script = self.project_root / "03_tokenizer" / "train_tokenizer.py"
            if not tokenizer_script.exists():
                raise FileNotFoundError(f"Tokenizer script not found: {tokenizer_script}")
            
            # Check if tokenizer directory exists
            tokenizer_dir = self.project_root / "09_models" / "tokenizers" / "london_historical_tokenizer"
            tokenizer_exists = tokenizer_dir.exists()
            
            if tokenizer_exists:
                # Test loading tokenizer
                try:
                    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
                    
                    # Test basic tokenization
                    test_text = "In the year of our Lord 1665, the Great Plague swept through London."
                    tokens = tokenizer.encode(test_text)
                    decoded = tokenizer.decode(tokens)
                    
                    self.test_results['test_details'][test_name] = {
                        'status': 'passed',
                        'tokenizer_script_exists': True,
                        'tokenizer_directory_exists': True,
                        'tokenizer_loads': True,
                        'vocab_size': tokenizer.vocab_size,
                        'test_tokenization_works': True,
                        'test_text': test_text,
                        'test_tokens_count': len(tokens),
                        'test_decoded': decoded
                    }
                    
                except Exception as e:
                    self.test_results['test_details'][test_name] = {
                        'status': 'failed',
                        'tokenizer_script_exists': True,
                        'tokenizer_directory_exists': True,
                        'tokenizer_loads': False,
                        'error': str(e)
                    }
                    self.test_results['tests_failed'] += 1
                    logger.error(f"Tokenizer test failed: {e}")
                    return False
            else:
                self.test_results['test_details'][test_name] = {
                    'status': 'passed',
                    'tokenizer_script_exists': True,
                    'tokenizer_directory_exists': False,
                    'note': 'Tokenizer not trained yet - run tokenizer training first'
                }
            
            self.test_results['tests_passed'] += 1
            logger.info(f"Tokenizer test passed")
            return True
            
        except Exception as e:
            self.test_results['test_details'][test_name] = {
                'status': 'failed',
                'error': str(e)
            }
            self.test_results['tests_failed'] += 1
            logger.error(f"Tokenizer test failed: {e}")
            return False
    
    def test_model_training(self) -> bool:
        """Test model training system"""
        logger.info(f"Testing model training system...")
        
        test_name = "model_training"
        self.test_results['total_tests'] += 1
        
        try:
            # Check if training script exists
            training_script = self.project_root / "04_training" / "train_model.py"
            if not training_script.exists():
                raise FileNotFoundError(f"Training script not found: {training_script}")
            
            # Check if model directory exists
            model_dir = self.project_root / "09_models" / "checkpoints"
            model_exists = model_dir.exists()
            
            if model_exists:
                # Check for model files
                model_files = list(model_dir.glob("*.bin")) + list(model_dir.glob("*.safetensors"))
                config_file = model_dir / "config.json"
                
                # Test loading model if files exist
                if model_files and config_file.exists():
                    try:
                        model = AutoModelForCausalLM.from_pretrained(str(model_dir))
                        
                        self.test_results['test_details'][test_name] = {
                            'status': 'passed',
                            'training_script_exists': True,
                            'model_directory_exists': True,
                            'model_files_exist': True,
                            'model_loads': True,
                            'model_parameters': sum(p.numel() for p in model.parameters()),
                            'model_files_count': len(model_files)
                        }
                        
                    except Exception as e:
                        self.test_results['test_details'][test_name] = {
                            'status': 'failed',
                            'training_script_exists': True,
                            'model_directory_exists': True,
                            'model_files_exist': True,
                            'model_loads': False,
                            'error': str(e)
                        }
                        self.test_results['tests_failed'] += 1
                        logger.error(f"Model training test failed: {e}")
                        return False
                else:
                    self.test_results['test_details'][test_name] = {
                        'status': 'passed',
                        'training_script_exists': True,
                        'model_directory_exists': True,
                        'model_files_exist': False,
                        'note': 'Model not trained yet - run model training first'
                    }
            else:
                self.test_results['test_details'][test_name] = {
                    'status': 'passed',
                    'training_script_exists': True,
                    'model_directory_exists': False,
                    'note': 'Model directory not created yet - run model training first'
                }
            
            self.test_results['tests_passed'] += 1
            logger.info(f"Model training test passed")
            return True
            
        except Exception as e:
            self.test_results['test_details'][test_name] = {
                'status': 'failed',
                'error': str(e)
            }
            self.test_results['tests_failed'] += 1
            logger.error(f"Model training test failed: {e}")
            return False
    
    def test_evaluation(self) -> bool:
        """Test evaluation system"""
        logger.info(f"Testing evaluation system...")
        
        test_name = "evaluation"
        self.test_results['total_tests'] += 1
        
        try:
            # Check if evaluation script exists
            eval_script = self.project_root / "05_evaluation" / "evaluate_model.py"
            if not eval_script.exists():
                raise FileNotFoundError(f"Evaluation script not found: {eval_script}")
            
            # Check if evaluation results exist
            eval_dir = self.project_root / "05_evaluation" / "results"
            eval_results_exist = eval_dir.exists() and any(eval_dir.iterdir())
            
            self.test_results['test_details'][test_name] = {
                'status': 'passed',
                'evaluation_script_exists': True,
                'evaluation_results_exist': eval_results_exist,
                'note': 'Run evaluation after training model' if not eval_results_exist else 'Evaluation results available'
            }
            
            self.test_results['tests_passed'] += 1
            logger.info(f"Evaluation test passed")
            return True
            
        except Exception as e:
            self.test_results['test_details'][test_name] = {
                'status': 'failed',
                'error': str(e)
            }
            self.test_results['tests_failed'] += 1
            logger.error(f"Evaluation test failed: {e}")
            return False
    
    def test_network_connectivity(self) -> bool:
        """Test network connectivity for data sources"""
        logger.info("🌐 Testing network connectivity...")
        
        test_name = "network_connectivity"
        self.test_results['total_tests'] += 1
        
        try:
            test_urls = [
                'https://www.gutenberg.org',
                'https://archive.org',
                'https://www.google.com'
            ]
            
            connectivity_results = {}
            
            for url in test_urls:
                try:
                    response = requests.get(url, timeout=10)
                    connectivity_results[url] = {
                        'status': 'success',
                        'status_code': response.status_code,
                        'response_time': response.elapsed.total_seconds()
                    }
                except Exception as e:
                    connectivity_results[url] = {
                        'status': 'failed',
                        'error': str(e)
                    }
            
            successful_connections = sum(1 for result in connectivity_results.values() if result['status'] == 'success')
            
            self.test_results['test_details'][test_name] = {
                'status': 'passed' if successful_connections > 0 else 'failed',
                'connectivity_results': connectivity_results,
                'successful_connections': successful_connections,
                'total_tested': len(test_urls)
            }
            
            if successful_connections > 0:
                self.test_results['tests_passed'] += 1
                logger.info(f"Network connectivity test passed")
                return True
            else:
                self.test_results['tests_failed'] += 1
                logger.error(f"Network connectivity test failed")
                return False
                
        except Exception as e:
            self.test_results['test_details'][test_name] = {
                'status': 'failed',
                'error': str(e)
            }
            self.test_results['tests_failed'] += 1
            logger.error(f"Network connectivity test failed: {e}")
            return False
    
    def test_file_structure(self) -> bool:
        """Test project file structure"""
        logger.info(f"Testing project file structure...")
        
        test_name = "file_structure"
        self.test_results['total_tests'] += 1
        
        try:
            required_dirs = [
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
            
            required_files = [
                "README.md",
                "requirements.txt",
                "01_environment/setup_environment.py",
                "02_data_collection/download_historical_data.py",
                "03_tokenizer/train_tokenizer.py",
                "04_training/train_model.py",
                "05_evaluation/evaluate_model.py",
                "06_testing/test_system.py"
            ]
            
            dir_results = {}
            for dir_name in required_dirs:
                dir_path = self.project_root / dir_name
                dir_results[dir_name] = dir_path.exists()
            
            file_results = {}
            for file_name in required_files:
                file_path = self.project_root / file_name
                file_results[file_name] = file_path.exists()
            
            missing_dirs = [name for name, exists in dir_results.items() if not exists]
            missing_files = [name for name, exists in file_results.items() if not exists]
            
            self.test_results['test_details'][test_name] = {
                'status': 'passed' if not missing_dirs and not missing_files else 'failed',
                'directory_results': dir_results,
                'file_results': file_results,
                'missing_dirs': missing_dirs,
                'missing_files': missing_files
            }
            
            if not missing_dirs and not missing_files:
                self.test_results['tests_passed'] += 1
                logger.info(f"File structure test passed")
                return True
            else:
                self.test_results['tests_failed'] += 1
                logger.error(f"File structure test failed - missing: {missing_dirs + missing_files}")
                return False
                
        except Exception as e:
            self.test_results['test_details'][test_name] = {
                'status': 'failed',
                'error': str(e)
            }
            self.test_results['tests_failed'] += 1
            logger.error(f"File structure test failed: {e}")
            return False
    
    def generate_recommendations(self):
        """Generate recommendations based on test results"""
        logger.info("💡 Generating recommendations...")
        
        recommendations = []
        
        # Check data collection
        if 'data_collection' in self.test_results['test_details']:
            data_test = self.test_results['test_details']['data_collection']
            if not data_test.get('corpus_file_exists', False):
                recommendations.append("Run data collection: cd 02_data_collection && python download_historical_data.py")
        
        # Check tokenizer
        if 'tokenizer' in self.test_results['test_details']:
            tokenizer_test = self.test_results['test_details']['tokenizer']
            if not tokenizer_test.get('tokenizer_directory_exists', False):
                recommendations.append("Train tokenizer: cd 03_tokenizer && python train_tokenizer.py")
        
        # Check model training
        if 'model_training' in self.test_results['test_details']:
            model_test = self.test_results['test_details']['model_training']
            if not model_test.get('model_files_exist', False):
                recommendations.append("Train model: cd 04_training && python train_model.py")
        
        # Check evaluation
        if 'evaluation' in self.test_results['test_details']:
            eval_test = self.test_results['test_details']['evaluation']
            if not eval_test.get('evaluation_results_exist', False):
                recommendations.append("Run evaluation: cd 05_evaluation && python evaluate_model.py")
        
        # Check network connectivity
        if 'network_connectivity' in self.test_results['test_details']:
            network_test = self.test_results['test_details']['network_connectivity']
            if network_test.get('successful_connections', 0) == 0:
                recommendations.append("Check internet connection for data collection")
        
        self.test_results['recommendations'] = recommendations
    
    def save_results(self):
        """Save test results"""
        logger.info(f"Saving test results...")
        
        try:
            # Save JSON results
            results_file = self.project_root / "06_testing" / "test_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(self.test_results, f, indent=2, ensure_ascii=False)
            
            # Save summary report
            summary_file = self.project_root / "06_testing" / "test_summary.txt"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("London Historical LLM - System Test Summary\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"Test Time: {self.test_results['test_time']}\n")
                f.write(f"Tests Passed: {self.test_results['tests_passed']}\n")
                f.write(f"Tests Failed: {self.test_results['tests_failed']}\n")
                f.write(f"Total Tests: {self.test_results['total_tests']}\n")
                f.write(f"Success Rate: {(self.test_results['tests_passed']/max(1, self.test_results['total_tests'])*100):.1f}%\n\n")
                
                f.write("Test Details:\n")
                for test_name, details in self.test_results['test_details'].items():
                    f.write(f"  {test_name}: {details['status']}\n")
                    if 'error' in details:
                        f.write(f"    Error: {details['error']}\n")
                f.write("\n")
                
                if self.test_results['recommendations']:
                    f.write("Recommendations:\n")
                    for i, rec in enumerate(self.test_results['recommendations'], 1):
                        f.write(f"  {i}. {rec}\n")
            
            logger.info(f"Test results saved to: {self.project_root / '06_testing'}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save test results: {e}")
            return False
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*70)
        print("🧪 SYSTEM TEST SUMMARY")
        print("="*70)
        print(f"Tests Passed: {self.test_results['tests_passed']}")
        print(f"Tests Failed: {self.test_results['tests_failed']}")
        print(f"Total Tests: {self.test_results['total_tests']}")
        print(f"Success Rate: {(self.test_results['tests_passed']/max(1, self.test_results['total_tests'])*100):.1f}%")
        
        print(f"\n📊 Test Details:")
        for test_name, details in self.test_results['test_details'].items():
            status_emoji = "✅" if details['status'] == 'passed' else "❌"
            print(f"  {status_emoji} {test_name}: {details['status']}")
            if 'error' in details:
                print(f"    Error: {details['error']}")
        
        if self.test_results['recommendations']:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(self.test_results['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        print(f"\n📁 Test results saved to: {self.project_root / '06_testing'}")
        print("="*70)

def main():
    """Main testing function"""
    parser = argparse.ArgumentParser(description="Test London Historical LLM system")
    parser.add_argument("--project_root", type=str, default=".",
                       help="Project root directory")
    parser.add_argument("--test_type", type=str, default="all",
                       choices=["all", "environment", "data", "tokenizer", "model", "evaluation", "network", "structure"],
                       help="Type of test to run")
    
    args = parser.parse_args()
    
    print("🧪 London Historical LLM - System Testing")
    print("=" * 50)
    
    # Initialize tester
    tester = LondonHistoricalSystemTester(args.project_root)
    
    try:
        # Run tests based on type
        if args.test_type == "all":
            tests = [
                tester.test_environment,
                tester.test_file_structure,
                tester.test_network_connectivity,
                tester.test_data_collection,
                tester.test_tokenizer,
                tester.test_model_training,
                tester.test_evaluation
            ]
        elif args.test_type == "environment":
            tests = [tester.test_environment]
        elif args.test_type == "data":
            tests = [tester.test_data_collection]
        elif args.test_type == "tokenizer":
            tests = [tester.test_tokenizer]
        elif args.test_type == "model":
            tests = [tester.test_model_training]
        elif args.test_type == "evaluation":
            tests = [tester.test_evaluation]
        elif args.test_type == "network":
            tests = [tester.test_network_connectivity]
        elif args.test_type == "structure":
            tests = [tester.test_file_structure]
        else:
            print(f"Unknown test type: {args.test_type}")
            return False
        
        # Run tests
        for test_func in tests:
            test_func()
        
        # Generate recommendations
        tester.generate_recommendations()
        
        # Save results
        tester.save_results()
        
        # Print summary
        tester.print_summary()
        
        logger.info(f"System testing completed!")
        return True
        
    except Exception as e:
        logger.error(f"System testing failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
