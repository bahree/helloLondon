#!/usr/bin/env python3
"""
London Historical LLM - Unified Inference Script
Supports both SLM (117M) and Regular (354M) models in all formats
"""

import os
import sys
import argparse
import logging
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Any
import torch

# Suppress warnings and logging
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore')
logging.getLogger("transformers").setLevel(logging.ERROR)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import global configuration
from config import config

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
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

class LondonHistoricalInference:
    def __init__(self, 
                 model_type: str = "auto",
                 device: str = "auto"):
        """
        Initialize unified inference engine
        
        Args:
            model_type: "slm", "regular", or "auto" (auto-detect)
            device: Device to run on ('cuda', 'cpu', or 'auto')
        """
        self.model_type = model_type
        self.device = self._get_device(device)
        
        # Model configurations
        self.model_configs = {
            "slm": {
                "published_name": "bahree/london-historical-slm",
                "description": "Small Language Model (117M parameters)",
                "max_length": 512,
                "temperature": 0.8,
                "top_p": 0.9,
                "top_k": 50
            },
            "regular": {
                "published_name": "bahree/london-historical-llm",  # Note: This model may not be published yet
                "description": "Regular Language Model (354M parameters)", 
                "max_length": 1024,
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40
            }
        }
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.current_config = None
        
    def _get_device(self, device: str) -> str:
        """Auto-detect device if needed"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _detect_model_type(self, model_path: str) -> str:
        """Auto-detect model type from path or name"""
        model_path_lower = model_path.lower()
        
        if "slm" in model_path_lower or "small" in model_path_lower:
            return "slm"
        elif "regular" in model_path_lower or "llm" in model_path_lower:
            return "regular"
        elif "117" in model_path_lower or "117m" in model_path_lower:
            return "slm"
        elif "354" in model_path_lower or "354m" in model_path_lower:
            return "regular"
        else:
            # Default to SLM for now
            logger.warning("Could not auto-detect model type, defaulting to SLM")
            return "slm"
    
    def load_published_model(self, model_name: str = None):
        """
        Load published model from Hugging Face Hub
        
        Args:
            model_name: Hugging Face model name (if None, uses model_type default)
        """
        if model_name is None:
            model_name = self.model_configs[self.model_type]["published_name"]
        
        logger.info(f"📂 Loading published model: {model_name}")
        
        try:
            # Auto-detect model type if not set
            if self.model_type == "auto":
                self.model_type = self._detect_model_type(model_name)
            
            self.current_config = self.model_configs[self.model_type]
            logger.info(f"🎯 Model type: {self.model_type} - {self.current_config['description']}")
            
            # Load tokenizer
            logger.info("🔤 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            logger.info("🤖 Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            logger.info("✅ Model loaded successfully!")
            self._print_model_info()
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load published model: {e}")
            logger.error(f"   Model: {model_name}")
            logger.error(f"   Device: {self.device}")
            logger.error(f"   Model type: {self.model_type}")
            return False
    
    def load_local_model(self, model_path: str, tokenizer_path: str = None):
        """
        Load local model from directory
        
        Args:
            model_path: Path to model directory
            tokenizer_path: Path to tokenizer directory (if None, uses model_path)
        """
        model_path = Path(model_path)
        tokenizer_path = Path(tokenizer_path) if tokenizer_path else model_path
        
        logger.info(f"📂 Loading local model from: {model_path}")
        
        try:
            # Auto-detect model type if not set
            if self.model_type == "auto":
                self.model_type = self._detect_model_type(str(model_path))
            
            self.current_config = self.model_configs[self.model_type]
            logger.info(f"🎯 Model type: {self.model_type} - {self.current_config['description']}")
            
            # Load tokenizer
            if not tokenizer_path.exists():
                logger.error(f"❌ Tokenizer directory not found: {tokenizer_path}")
                return False
            
            logger.info("🔤 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_path), 
                local_files_only=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            logger.info("🤖 Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            logger.info("✅ Model loaded successfully!")
            self._print_model_info()
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load local model: {e}")
            return False
    
    def _print_model_info(self):
        """Print model information"""
        if self.model and self.tokenizer:
            print(f"\n📊 Model Information:")
            print(f"   Type: {self.model_type.upper()}")
            print(f"   Description: {self.current_config['description']}")
            print(f"   Device: {self.device}")
            print(f"   Vocabulary size: {self.tokenizer.vocab_size:,}")
            print(f"   Max length: {self.tokenizer.model_max_length}")
            print(f"   Model parameters: ~{self._estimate_parameters():,}")
    
    def _estimate_parameters(self) -> int:
        """Estimate model parameters"""
        if self.model_type == "slm":
            return 117_000_000
        elif self.model_type == "regular":
            return 354_000_000
        else:
            return 0
    
    def generate_text(self, 
                     prompt: str,
                     max_length: int = None,
                     temperature: float = None,
                     top_p: float = None,
                     top_k: int = None,
                     repetition_penalty: float = 1.1,
                     num_return_sequences: int = 1) -> List[str]:
        """
        Generate text from a prompt
        
        Args:
            prompt: Input text prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repetition
            num_return_sequences: Number of sequences to generate
            
        Returns:
            List of generated text sequences
        """
        if self.model is None or self.tokenizer is None:
            logger.error("❌ Model or tokenizer not loaded")
            return []
        
        # Use model-specific defaults if not provided
        if max_length is None:
            max_length = self.current_config["max_length"]
        if temperature is None:
            temperature = self.current_config["temperature"]
        if top_p is None:
            top_p = self.current_config["top_p"]
        if top_k is None:
            top_k = self.current_config["top_k"]
        
        try:
            # Encode input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # Generation config
            generation_config = GenerationConfig(
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                early_stopping=True
            )
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    generation_config=generation_config,
                    attention_mask=torch.ones_like(inputs)
                )
            
            # Decode outputs
            generated_texts = []
            for output in outputs:
                # Remove input tokens from output
                generated_tokens = output[inputs.shape[1]:]
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                generated_texts.append(generated_text.strip())
            
            return generated_texts
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            return []
    
    def interactive_mode(self):
        """Run interactive text generation"""
        print(f"\n🎭 London Historical {self.model_type.upper()} - Interactive Mode")
        print("=" * 60)
        print("Type 'quit' to exit, 'help' for commands, 'config' for settings")
        print()
        
        while True:
            try:
                prompt = input("📝 Enter prompt: ").strip()
                
                if prompt.lower() == 'quit':
                    break
                elif prompt.lower() == 'help':
                    print("\nCommands:")
                    print("  quit - Exit the program")
                    print("  help - Show this help")
                    print("  config - Show current generation settings")
                    print("  Any other text - Generate continuation")
                    print()
                    continue
                elif prompt.lower() == 'config':
                    print(f"\nCurrent Settings:")
                    print(f"  Model Type: {self.model_type}")
                    print(f"  Max Length: {self.current_config['max_length']}")
                    print(f"  Temperature: {self.current_config['temperature']}")
                    print(f"  Top-p: {self.current_config['top_p']}")
                    print(f"  Top-k: {self.current_config['top_k']}")
                    print()
                    continue
                elif not prompt:
                    continue
                
                print("\n🤖 Generating...")
                generated = self.generate_text(prompt, max_length=150)
                
                if generated:
                    print(f"\n📖 Generated text:")
                    print("-" * 50)
                    print(generated[0])
                    print("-" * 50)
                else:
                    print("❌ Generation failed")
                
                print()
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def demo_prompts(self):
        """Run demo prompts to showcase the model"""
        print(f"\n🎯 Demo Prompts - London Historical {self.model_type.upper()}")
        print("=" * 70)
        
        demo_prompts = [
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
        
        for i, prompt in enumerate(demo_prompts, 1):
            print(f"\n--- Demo {i} ---")
            print(f"Prompt: {prompt}")
            
            result = self.generate_text(prompt, max_length=40)
            if result:
                print(f"Generated: {result[0]}")
            else:
                print("Generation failed")
            print("-" * 70)

def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description="London Historical LLM - Unified Inference")
    
    # Model selection
    parser.add_argument("--model_type", type=str, default="auto",
                       choices=["auto", "slm", "regular"],
                       help="Model type: auto, slm, or regular")
    parser.add_argument("--published", action="store_true",
                       help="Load published model from Hugging Face Hub")
    parser.add_argument("--model_name", type=str, default=None,
                       help="Hugging Face model name (when using --published)")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to local model directory")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                       help="Path to tokenizer directory (if different from model_path)")
    
    # Device and generation settings
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"],
                       help="Device to run on")
    parser.add_argument("--max_length", type=int, default=None,
                       help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=None,
                       help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=None,
                       help="Nucleus sampling parameter")
    parser.add_argument("--top_k", type=int, default=None,
                       help="Top-k sampling parameter")
    
    # Usage modes
    parser.add_argument("--prompt", type=str, default=None,
                       help="Single prompt to generate from")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
    parser.add_argument("--demo", action="store_true",
                       help="Run demo prompts")
    
    args = parser.parse_args()
    
    print("🏛️ London Historical LLM - Unified Inference")
    print("=" * 60)
    
    # Initialize inference engine
    inference = LondonHistoricalInference(
        model_type=args.model_type,
        device=args.device
    )
    
    # Load model
    if args.published:
        model_name = args.model_name
        if not inference.load_published_model(model_name):
            print("❌ Failed to load published model")
            return False
    elif args.model_path:
        if not inference.load_local_model(args.model_path, args.tokenizer_path):
            print("❌ Failed to load local model")
            return False
    else:
        # Default: try to load published SLM
        print("No model specified, loading published SLM...")
        if not inference.load_published_model():
            print("❌ Failed to load published model")
            return False
    
    # Run inference
    if args.interactive:
        inference.interactive_mode()
    elif args.demo:
        inference.demo_prompts()
    elif args.prompt:
        print(f"\n📝 Prompt: {args.prompt}")
        print("🤖 Generating...")
        
        generated = inference.generate_text(
            args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k
        )
        
        if generated:
            print(f"\n📖 Generated text:")
            print("-" * 50)
            print(generated[0])
            print("-" * 50)
        else:
            print("❌ Generation failed")
    else:
        # Default: run demo then ask for interactive
        inference.demo_prompts()
        
        print("\n🎭 Would you like to try interactive mode? (y/n): ", end="")
        try:
            choice = input().strip().lower()
            if choice in ['y', 'yes']:
                inference.interactive_mode()
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
    
    print("\n✅ Inference completed!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
