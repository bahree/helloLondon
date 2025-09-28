#!/usr/bin/env python3
"""
London Historical LLM - PyTorch Checkpoint Inference
Direct inference from PyTorch .pt checkpoint files (during training)
"""

import os
import sys
import argparse
import logging
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Any
import torch

# Suppress Hugging Face warnings
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

class PyTorchCheckpointInference:
    def __init__(self, 
                 model_type: str = "auto",
                 device: str = "auto"):
        """
        Initialize PyTorch checkpoint inference engine
        
        Args:
            model_type: "slm", "regular", or "auto" (auto-detect)
            device: Device to run on ('cuda', 'cpu', or 'auto')
        """
        self.model_type = model_type
        self.device = self._get_device(device)
        
        # Model configurations - will be updated after loading config
        self.model_configs = {
            "slm": {
                "description": "Small Language Model (117M parameters)",
                "max_length": 512,  # Will be updated from config
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40
            },
            "regular": {
                "description": "Regular Language Model (354M parameters)", 
                "max_length": 1024,  # Will be updated from config
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40
            }
        }
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.current_config = None
        
        # Update model configs from global config
        self._update_model_configs()
        
    def _update_model_configs(self):
        """Update model configurations from global config"""
        # Update SLM config
        slm_config = config.slm_config
        self.model_configs["slm"]["max_length"] = slm_config["max_length"]
        
        # Update regular model config
        reg_config = config.training_config
        self.model_configs["regular"]["max_length"] = reg_config["max_length"]
    
    def _get_device(self, device: str) -> str:
        """Auto-detect device if needed"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _detect_model_type(self, checkpoint_path: str) -> str:
        """Auto-detect model type from checkpoint path"""
        checkpoint_lower = checkpoint_path.lower()
        
        if "slm" in checkpoint_lower or "small" in checkpoint_lower:
            return "slm"
        elif "regular" in checkpoint_lower or "llm" in checkpoint_lower:
            return "regular"
        elif "117" in checkpoint_lower or "117m" in checkpoint_lower:
            return "slm"
        elif "354" in checkpoint_lower or "354m" in checkpoint_lower:
            return "regular"
        elif "/checkpoints/slm/" in checkpoint_lower:
            return "slm"
        elif "/checkpoints/" in checkpoint_lower and "/slm/" not in checkpoint_lower:
            # Regular model checkpoints are in /checkpoints/ (not /checkpoints/slm/)
            return "regular"
        else:
            # Default to regular model for checkpoints in main checkpoints directory
            logger.debug("Could not auto-detect model type from path, defaulting to regular model")
            return "regular"
    
    def _validate_checkpoint(self, checkpoint_path: str) -> bool:
        """Validate that checkpoint can be loaded"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Check if it has the expected structure
            if isinstance(checkpoint, dict):
                # Check for common checkpoint keys
                if any(key in checkpoint for key in ['model', 'model_state_dict', 'state_dict']):
                    return True
                # Check if it's a direct state dict (has model layer names)
                elif any(key.startswith(('transformer.', 'lm_head.')) for key in checkpoint.keys()):
                    return True
                else:
                    logger.warning(f"Unknown checkpoint structure. Keys: {list(checkpoint.keys())}")
                    return True  # Still try to load it
            else:
                logger.warning("Checkpoint is not a dictionary")
                return False
                
        except Exception as e:
            logger.error(f"Failed to validate checkpoint: {e}")
            return False
    
    def load_checkpoint(self, checkpoint_path: str, tokenizer_path: str = None):
        """
        Load PyTorch checkpoint and tokenizer
        
        Args:
            checkpoint_path: Path to .pt checkpoint file
            tokenizer_path: Path to tokenizer directory (if None, uses config default)
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            logger.error(f"❌ Checkpoint file not found: {checkpoint_path}")
            return False
        
        logger.info(f"📂 Loading PyTorch checkpoint: {checkpoint_path}")
        
        try:
            # Auto-detect model type if not set
            if self.model_type == "auto":
                self.model_type = self._detect_model_type(str(checkpoint_path))
            
            self.current_config = self.model_configs[self.model_type]
            logger.info(f"🎯 Model type: {self.model_type} - {self.current_config['description']}")
            
            # Load tokenizer
            if tokenizer_path is None:
                # Use default tokenizer path from config
                tokenizer_path = config.london_tokenizer_dir
            else:
                tokenizer_path = Path(tokenizer_path)
            
            if not tokenizer_path.exists():
                logger.error(f"❌ Tokenizer directory not found: {tokenizer_path}")
                return False
            
            logger.info(f"🔤 Loading tokenizer from: {tokenizer_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_path), 
                local_files_only=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model from checkpoint
            logger.info("🤖 Loading model from PyTorch checkpoint...")
            
            # Load the checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Extract model state dict - handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                logger.info("Using 'model_state_dict' from checkpoint")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                logger.info("Using 'state_dict' from checkpoint")
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
                logger.info("Using 'model' from checkpoint")
            else:
                # Check if this is a direct model state dict
                # Look for typical model layer names
                if any(key.startswith(('transformer.', 'lm_head.')) for key in checkpoint.keys()):
                    state_dict = checkpoint
                    logger.info("Using checkpoint directly as state_dict (detected model layers)")
                else:
                    state_dict = checkpoint
                    logger.info("Using checkpoint directly as state_dict (fallback)")
            
            # Try to load as Hugging Face model first
            try:
                logger.info("🤖 Attempting to load as Hugging Face model...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(checkpoint_path.parent),  # Use directory containing the .pt file
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
                
                if self.device == "cpu":
                    self.model = self.model.to(self.device)
                
                self.model.eval()
                logger.info("✅ Loaded as Hugging Face model")
                
            except Exception as e:
                logger.debug(f"Failed to load as HF model: {e}")
                logger.info("🔄 Attempting to load as raw PyTorch checkpoint...")
                
                # Both models use the same nanoGPT approach, just different configs
                # Add training directory to path for imports
                training_dir = Path(__file__).parent.parent / "04_training"
                sys.path.insert(0, str(training_dir))
                
                if self.model_type == "slm":
                    logger.info("🤖 Using SimpleGPT architecture for SLM...")
                    from train_model_slm import SimpleGPT, SimpleGPTConfig
                    model_class = SimpleGPT
                    config_class = SimpleGPTConfig
                    model_config_dict = config.slm_config
                else:  # regular model
                    logger.info("🤖 Using GPT architecture for regular model...")
                    from train_model import GPT, GPTConfig
                    model_class = GPT
                    config_class = GPTConfig
                    model_config_dict = config.training_config
                
                # Create model config using the appropriate config values
                model_config = config_class(
                    n_layer=model_config_dict["n_layer"],
                    n_head=model_config_dict["n_head"],
                    n_embd=model_config_dict["n_embd"],
                    block_size=model_config_dict["max_length"],
                    bias=False,
                    vocab_size=self.tokenizer.vocab_size,
                    dropout=0.1
                )
                
                # Create model instance
                self.model = model_class(model_config)
                
                # Handle torch.compile prefixes (_orig_mod.) if present
                if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
                    logger.info("🔧 Detected torch.compile checkpoint, stripping _orig_mod. prefixes...")
                    clean_state = {}
                    for key, value in state_dict.items():
                        if key.startswith('_orig_mod.'):
                            clean_key = key[10:]  # Remove '_orig_mod.' prefix
                            clean_state[clean_key] = value
                        else:
                            clean_state[key] = value
                    state_dict = clean_state
                    logger.info(f"✅ Cleaned {len(clean_state)} parameters")
                
                # Load state dict
                try:
                    self.model.load_state_dict(state_dict, strict=False)
                    self.model.to(self.device)
                    self.model.eval()
                    logger.info("✅ Loaded as raw PyTorch checkpoint")
                except Exception as e:
                    logger.error(f"❌ Failed to load state dict: {e}")
                    logger.info("🔄 Trying to load with strict=False and ignore missing keys...")
                    # Try loading with more lenient settings
                    missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                    if missing_keys:
                        logger.debug(f"Missing keys: {missing_keys[:5]}...")  # Show first 5
                    if unexpected_keys:
                        logger.debug(f"Unexpected keys: {unexpected_keys[:5]}...")  # Show first 5
                    self.model.to(self.device)
                    self.model.eval()
                    logger.info("✅ Loaded as raw PyTorch checkpoint (with warnings)")
            
            logger.info("✅ Model loaded successfully!")
            self._print_model_info()
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint: {e}")
            return False
    
    def _print_model_info(self):
        """Print model information"""
        if self.model and self.tokenizer:
            print(f"\n📊 Model Information:")
            print(f"   Type: {self.model_type.upper()}")
            print(f"   Description: {self.current_config['description']}")
            print(f"   Device: {self.device}")
            print(f"   Vocabulary size: {self.tokenizer.vocab_size:,}")
            print(f"   Model parameters: ~{self._estimate_parameters():,}")
    
    def _estimate_parameters(self) -> int:
        """Estimate model parameters based on config - both models use same nanoGPT architecture"""
        # Get config based on model type
        if self.model_type == "slm":
            model_config_dict = config.slm_config
        elif self.model_type == "regular":
            model_config_dict = config.training_config
        else:
            # Try to estimate from actual model if available
            if self.model:
                total_params = sum(p.numel() for p in self.model.parameters())
                return total_params
            return 0
        
        # Both models use the same nanoGPT parameter calculation
        vocab_size = self.tokenizer.vocab_size if self.tokenizer else 30000
        n_embd = model_config_dict["n_embd"]
        n_layer = model_config_dict["n_layer"]
        block_size = model_config_dict["max_length"]
        
        # nanoGPT parameter calculation (same for both SimpleGPT and GPT)
        # Embedding: vocab_size * n_embd + block_size * n_embd
        # Transformer: n_layer * (4 * n_embd^2 + 2 * n_embd^2 + 2 * n_embd^2)
        # LM Head: n_embd * vocab_size (but tied to embedding, so no extra params)
        embedding_params = vocab_size * n_embd + block_size * n_embd
        transformer_params = n_layer * (4 * n_embd**2 + 2 * n_embd**2 + 2 * n_embd**2)
        # LM head is tied to embedding, so no additional parameters
        
        return embedding_params + transformer_params
    
    def generate_text(self, 
                     prompt: str,
                     max_length: int = None,
                     temperature: float = None,
                     top_p: float = None,
                     top_k: int = None,
                     repetition_penalty: float = 1.2,
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
            input_length = inputs.shape[1]
            
            # Both SLM and regular models use custom architectures with same forward method
            # Use custom generation for both SimpleGPT and GPT
            generated = inputs.clone()
            
            # Track recent tokens to detect repetitive patterns
            recent_tokens = []
            repetition_threshold = 5  # Stop if we see the same token repeated 5 times in a row
            
            for step in range(max_length - input_length):
                # Get logits for next token
                logits, _ = self.model(generated)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    # Get unique tokens in the generated sequence so far
                    unique_tokens = torch.unique(generated[0])
                    for token_id in unique_tokens:
                        if next_token_logits[0, token_id] > 0:
                            next_token_logits[0, token_id] /= repetition_penalty
                        else:
                            next_token_logits[0, token_id] *= repetition_penalty
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Check for repetitive patterns
                recent_tokens.append(next_token.item())
                if len(recent_tokens) > repetition_threshold:
                    recent_tokens.pop(0)
                
                # If we see the same token repeated too many times, break
                if len(recent_tokens) == repetition_threshold and len(set(recent_tokens)) == 1:
                    logger.warning(f"Detected repetitive pattern, stopping generation at step {step}")
                    break
                
                # Append to sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if we hit EOS token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
            
            # Decode only the new tokens
            new_tokens = generated[0][input_length:]
            generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            generated_texts = [generated_text.strip()]
            
            return generated_texts
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            return []
    
    def interactive_mode(self):
        """Run interactive text generation"""
        print(f"\n🎭 London Historical {self.model_type.upper()} - PyTorch Checkpoint Mode")
        print("=" * 70)
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
    parser = argparse.ArgumentParser(description="London Historical LLM - PyTorch Checkpoint Inference")
    
    # Required arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to PyTorch .pt checkpoint file")
    parser.add_argument("--tokenizer", type=str, default=None,
                       help="Path to tokenizer directory (if different from config default)")
    
    # Model selection
    parser.add_argument("--model_type", type=str, default="auto",
                       choices=["auto", "slm", "regular"],
                       help="Model type: auto, slm, or regular")
    
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
    
    print("🏛️ London Historical LLM - PyTorch Checkpoint Inference")
    print("=" * 70)
    
    # Initialize inference engine
    inference = PyTorchCheckpointInference(
        model_type=args.model_type,
        device=args.device
    )
    
    # Load checkpoint
    if not inference.load_checkpoint(args.checkpoint, args.tokenizer):
        print("❌ Failed to load checkpoint")
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
