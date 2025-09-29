#!/usr/bin/env python3
"""
London Historical LLM - Test Published Models
Test both SLM and Regular models from Hugging Face Hub
"""

import os
import sys
import argparse
import logging
import warnings
from pathlib import Path
from typing import List, Dict, Any
import time

# Suppress warnings and logging
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore')
logging.getLogger("transformers").setLevel(logging.ERROR)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
except ImportError as e:
    print(f"Missing required dependencies: {e}")
    print("Please install requirements: pip install -r requirements.txt")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PublishedModelTester:
    def __init__(self, device: str = "auto"):
        """
        Initialize published model tester
        
        Args:
            device: Device to run on ('cuda', 'cpu', or 'auto')
        """
        self.device = self._get_device(device)
        
        # Published models
        self.published_models = {
            "slm": {
                "name": "bahree/london-historical-slm",
                "description": "Small Language Model (117M parameters)",
                "max_length": 100,
                "temperature": 0.3,
                "top_p": 0.9,
                "top_k": 20
            },
            "regular": {
                "name": "bahree/london-historical-llm",  # Placeholder - update when available
                "description": "Regular Language Model (354M parameters)",
                "max_length": 150,
                "temperature": 0.3,
                "top_p": 0.9,
                "top_k": 20
            }
        }
        
        # Test prompts
        self.test_prompts = [
            "In the year 1834, I walked through the streets of London and witnessed",
            "The gentleman from the country said, 'I have never seen such a sight",
            "The Thames flowed dark and mysterious through the heart",
            "Merchants plied their wares in the bustling market",
            "The Great Fire of 1666 had destroyed",
            "Parliament sat in Westminster Hall",
            "The Tower of London stood",
            "Samuel Pepys wrote that",
            "The plague had ravaged",
            "Covent Garden was filled with"
        ]
        
    def _get_device(self, device: str) -> str:
        """Auto-detect device if needed"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def test_model(self, model_type: str, model_name: str) -> Dict[str, Any]:
        """
        Test a specific published model
        
        Args:
            model_type: "slm" or "regular"
            model_name: Hugging Face model name
            
        Returns:
            Dictionary with test results
        """
        print(f"\n🧪 Testing {model_type.upper()} Model: {model_name}")
        print("=" * 60)
        
        results = {
            "model_type": model_type,
            "model_name": model_name,
            "success": False,
            "load_time": 0,
            "generation_times": [],
            "generated_texts": [],
            "errors": []
        }
        
        try:
            # Load model
            print("📂 Loading model...")
            start_time = time.time()
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device == "cpu":
                model = model.to(self.device)
            
            model.eval()
            
            load_time = time.time() - start_time
            results["load_time"] = load_time
            print(f"✅ Model loaded in {load_time:.2f} seconds")
            
            # Model info
            config = self.published_models[model_type]
            print(f"📊 Model Info:")
            print(f"   Type: {model_type.upper()}")
            print(f"   Description: {config['description']}")
            print(f"   Device: {self.device}")
            print(f"   Vocabulary size: {tokenizer.vocab_size:,}")
            print(f"   Max length: {tokenizer.model_max_length}")
            
            # Test generation
            print(f"\n🎯 Testing generation with {len(self.test_prompts)} prompts...")
            
            for i, prompt in enumerate(self.test_prompts, 1):
                print(f"\n--- Test {i}/{len(self.test_prompts)} ---")
                print(f"Prompt: {prompt}")
                
                try:
                    start_time = time.time()
                    
                    # Generate text
                    inputs = tokenizer.encode(prompt, return_tensors="pt").to(self.device)
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            inputs,
                            max_length=config["max_length"],
                            temperature=config["temperature"],
                            top_p=config["top_p"],
                            top_k=config["top_k"],
                            do_sample=True,
                            pad_token_id=tokenizer.eos_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            early_stopping=True
                        )
                    
                    # Decode output
                    generated_tokens = outputs[0][inputs.shape[1]:]
                    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    
                    generation_time = time.time() - start_time
                    results["generation_times"].append(generation_time)
                    results["generated_texts"].append(generated_text.strip())
                    
                    print(f"Generated: {generated_text.strip()}")
                    print(f"Time: {generation_time:.2f}s")
                    
                except Exception as e:
                    error_msg = f"Generation failed for prompt {i}: {e}"
                    print(f"❌ {error_msg}")
                    results["errors"].append(error_msg)
            
            # Calculate statistics
            if results["generation_times"]:
                avg_generation_time = sum(results["generation_times"]) / len(results["generation_times"])
                results["avg_generation_time"] = avg_generation_time
                results["success"] = True
                
                print(f"\n📊 Test Results:")
                print(f"   Load time: {load_time:.2f}s")
                print(f"   Average generation time: {avg_generation_time:.2f}s")
                print(f"   Successful generations: {len(results['generated_texts'])}/{len(self.test_prompts)}")
                print(f"   Errors: {len(results['errors'])}")
            else:
                print("❌ No successful generations")
                
        except Exception as e:
            error_msg = f"Model loading failed: {e}"
            print(f"❌ {error_msg}")
            results["errors"].append(error_msg)
        
        return results
    
    def test_all_models(self) -> Dict[str, Any]:
        """Test all available published models"""
        print("🏛️ London Historical LLM - Published Models Test")
        print("=" * 70)
        
        all_results = {}
        
        for model_type, config in self.published_models.items():
            try:
                results = self.test_model(model_type, config["name"])
                all_results[model_type] = results
            except Exception as e:
                print(f"❌ Failed to test {model_type}: {e}")
                all_results[model_type] = {
                    "model_type": model_type,
                    "success": False,
                    "errors": [str(e)]
                }
        
        return all_results
    
    def print_summary(self, all_results: Dict[str, Any]):
        """Print test summary"""
        print(f"\n📋 Test Summary")
        print("=" * 50)
        
        for model_type, results in all_results.items():
            print(f"\n{model_type.upper()} Model:")
            if results["success"]:
                print(f"   ✅ Status: SUCCESS")
                print(f"   ⏱️  Load time: {results['load_time']:.2f}s")
                print(f"   ⏱️  Avg generation: {results['avg_generation_time']:.2f}s")
                print(f"   📝 Successful generations: {len(results['generated_texts'])}/{len(self.test_prompts)}")
            else:
                print(f"   ❌ Status: FAILED")
                print(f"   🐛 Errors: {len(results['errors'])}")
                for error in results['errors']:
                    print(f"      - {error}")
        
        # Overall status
        successful_models = sum(1 for r in all_results.values() if r["success"])
        total_models = len(all_results)
        
        print(f"\n🎯 Overall: {successful_models}/{total_models} models successful")
        
        if successful_models == total_models:
            print("🎉 All models working correctly!")
        elif successful_models > 0:
            print("⚠️  Some models working, some failed")
        else:
            print("❌ All models failed")

def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description="Test Published London Historical LLM Models")
    
    parser.add_argument("--model_type", type=str, default="all",
                       choices=["all", "slm", "regular"],
                       help="Model type to test")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"],
                       help="Device to run on")
    parser.add_argument("--model_name", type=str, default=None,
                       help="Custom model name to test")
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = PublishedModelTester(device=args.device)
    
    if args.model_name:
        # Test custom model
        model_type = "custom"
        results = tester.test_model(model_type, args.model_name)
        tester.print_summary({model_type: results})
    elif args.model_type == "all":
        # Test all models
        all_results = tester.test_all_models()
        tester.print_summary(all_results)
    else:
        # Test specific model type
        config = tester.published_models[args.model_type]
        results = tester.test_model(args.model_type, config["name"])
        tester.print_summary({args.model_type: results})
    
    print("\n✅ Testing completed!")

if __name__ == "__main__":
    main()
