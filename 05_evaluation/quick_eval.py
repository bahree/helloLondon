#!/usr/bin/env python3
"""
Quick Evaluation Script for London Historical Models
Runs essential evaluations without requiring OpenAI API
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import numpy as np
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from rouge_score import rouge_scorer
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize, sent_tokenize
    from textstat import flesch_reading_ease, flesch_kincaid_grade
    import warnings
    warnings.filterwarnings("ignore")
except ImportError as e:
    print(f"Missing required dependencies: {e}")
    print("Please install requirements: pip install -r requirements.txt")
    sys.exit(1)

# Download required NLTK data (robust across NLTK versions)
try:
    for resource in ['punkt', 'punkt_tab', 'stopwords']:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass
except Exception:
    pass

class QuickEvaluator:
    """Quick evaluator for London Historical Models"""
    
    def __init__(self, model_dir: str, tokenizer_dir: str, output_dir: str = "quick_results", device: str = "cpu"):
        self.model_dir = Path(model_dir)
        self.tokenizer_dir = Path(tokenizer_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        
        # Device selection - defaults to CPU for safety
        if device.lower() == "gpu" and torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("🚀 Using GPU for evaluation")
        else:
            self.device = torch.device("cpu")
            if device.lower() == "gpu" and not torch.cuda.is_available():
                print("⚠️  GPU requested but not available, falling back to CPU")
            else:
                print("🖥️  Using CPU for evaluation (safe default)")
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Results storage
        self.results = {
            'model_dir': str(self.model_dir),
            'tokenizer_dir': str(self.tokenizer_dir),
            'device': str(self.device),
            'evaluation_type': 'quick',
            'metrics': {},
            'generation_samples': []
        }
    
    def load_model_and_tokenizer(self):
        """Load the trained model and tokenizer"""
        print(f"🔄 Loading model and tokenizer...")
        print(f"   Model dir: {self.model_dir}")
        print(f"   Tokenizer dir: {self.tokenizer_dir}")
        print(f"   Device: {self.device}")
        
        try:
            # Check if directories exist
            if not self.model_dir.exists():
                print(f"❌ Model directory does not exist: {self.model_dir}")
                return False
            
            if not self.tokenizer_dir.exists():
                print(f"❌ Tokenizer directory does not exist: {self.tokenizer_dir}")
                return False
            
            # List files in model directory
            model_files = list(self.model_dir.glob("*"))
            print(f"   Model files found: {[f.name for f in model_files]}")
            
            # Load tokenizer
            print(f"   Loading tokenizer from: {self.tokenizer_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.tokenizer_dir))
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model - try different approaches
            print(f"   Loading model from: {self.model_dir}")
            
            # First try loading as a directory
            try:
                self.model = AutoModelForCausalLM.from_pretrained(str(self.model_dir))
            except Exception as e1:
                print(f"   Failed to load as directory: {e1}")
                
                # Try loading the latest checkpoint file
                checkpoint_files = list(self.model_dir.glob("checkpoint-*.pt"))
                if checkpoint_files:
                    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.stem.split('-')[1]))
                    print(f"   Trying to load checkpoint: {latest_checkpoint}")
                    
                    # Load checkpoint (this will be a state dict)
                    checkpoint = torch.load(latest_checkpoint, map_location=self.device)
                    print(f"   Checkpoint keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'Not a dict'}")
                    
                    # If it's a state dict, we need to create the model first
                    if isinstance(checkpoint, dict):
                        # Extract the actual model state dict
                        if 'model' in checkpoint:
                            model_state_dict = checkpoint['model']
                            print(f"   Found model state dict with {len(model_state_dict)} keys")
                        else:
                            model_state_dict = checkpoint
                            print(f"   Using checkpoint as model state dict with {len(model_state_dict)} keys")
                        
                        # Debug: Print first 10 keys to understand structure
                        print(f"   First 10 keys: {list(model_state_dict.keys())[:10]}")
                        
                        # Import the custom SimpleGPT model from training script
                        import sys
                        import os
                        sys.path.append(str(Path(__file__).parent.parent / "04_training"))
                        from train_model_slm import SimpleGPT, SimpleGPTConfig
                        
                        # Create model config based on the state dict structure
                        # Infer architecture from the state dict keys (robust to _orig_mod. prefix)
                        # Normalize keys for parsing
                        parsed_keys = [k.replace('_orig_mod.', '') for k in model_state_dict.keys()]
                        import re
                        layer_nums = []
                        for k in parsed_keys:
                            m = re.search(r"transformer\.h\.(\d+)\.", k)
                            if m:
                                layer_nums.append(int(m.group(1)))
                        n_layer = max(layer_nums) + 1 if layer_nums else 8
                        
                        # Helper to fetch tensors regardless of _orig_mod. prefix
                        def get_tensor(primary_key: str, alt_key: str):
                            if primary_key in model_state_dict:
                                return model_state_dict[primary_key]
                            if alt_key in model_state_dict:
                                return model_state_dict[alt_key]
                            return None

                        # Try to infer architecture from available keys
                        wte = get_tensor('transformer.wte.weight', '_orig_mod.transformer.wte.weight')
                        if wte is not None:
                            n_embd = wte.shape[1]
                            vocab_size = wte.shape[0]
                        else:
                            # Fallback: look for any embedding weight
                            embedding_keys = [k for k in model_state_dict.keys() if ('wte' in k or 'embed' in k) and k.endswith('.weight')]
                            if embedding_keys:
                                t = model_state_dict[embedding_keys[0]]
                                n_embd = t.shape[1] if t.ndim == 2 else 512
                                vocab_size = t.shape[0] if t.ndim >= 1 else self.tokenizer.vocab_size
                            else:
                                n_embd = 512  # Default fallback
                                vocab_size = self.tokenizer.vocab_size
                        
                        wpe = get_tensor('transformer.wpe.weight', '_orig_mod.transformer.wpe.weight')
                        if wpe is not None:
                            block_size = wpe.shape[0]
                        else:
                            block_size = 256  # Default fallback
                        
                        n_head = 8  # Default from training script
                        
                        print(f"   Inferred architecture: n_layer={n_layer}, n_embd={n_embd}, n_head={n_head}, block_size={block_size}, vocab_size={vocab_size}")
                        
                        # Create model config
                        config = SimpleGPTConfig(
                            n_layer=n_layer,
                            n_head=n_head,
                            n_embd=n_embd,
                            block_size=block_size,
                            bias=False,
                            vocab_size=vocab_size,
                            dropout=0.1
                        )
                        
                        # Create model with config
                        self.model = SimpleGPT(config)
                        
                        # Strip _orig_mod. prefix from keys (added by torch.compile)
                        cleaned_state_dict = {}
                        for key, value in model_state_dict.items():
                            if key.startswith('_orig_mod.'):
                                cleaned_key = key[10:]  # Remove '_orig_mod.' prefix
                            else:
                                cleaned_key = key
                            cleaned_state_dict[cleaned_key] = value
        
                        # Load state dict
                        self.model.load_state_dict(cleaned_state_dict)
                        print(f"   Loaded state dict into custom SimpleGPT model")
                    else:
                        # If it's already a model object
                        self.model = checkpoint
                else:
                    raise e1
            
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ Model and tokenizer loaded successfully")
            print(f"   Device: {self.device}")
            print(f"   Vocabulary size: {self.tokenizer.vocab_size:,}")
            print(f"   Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model/tokenizer: {e}")
            print(f"   Error type: {type(e).__name__}")
            import traceback
            print(f"   Traceback: {traceback.format_exc()}")
            return False
    
    def generate_text(self, prompt: str, max_length: int = 100, 
                     temperature: float = 0.8, top_p: float = 0.9) -> str:
        """Generate text from a prompt"""
        try:
            # Tokenize input properly
            enc = self.tokenizer(
                prompt,
                return_tensors='pt',
                add_special_tokens=False
            )
            input_ids = enc['input_ids'].to(self.device)

            with torch.no_grad():
                if hasattr(self.model, 'generate'):
                    # Use HF-style generation if available
                    outputs = self.model.generate(
                        input_ids=input_ids,
                        max_new_tokens=max_length,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=1.1
                    )
                    gen_ids = outputs[0]
                else:
                    # Custom autoregressive generation for SimpleGPT
                    gen_ids = self._generate_with_simple_model(
                        input_ids=input_ids,
                        max_new_tokens=max_length,
                        temperature=temperature,
                        top_p=top_p,
                    )[0]

            # Decode only the newly generated tokens
            prompt_len = input_ids.shape[1]
            new_tokens = gen_ids[prompt_len:]
            generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            
            return generated_text
            
        except Exception as e:
            print(f"❌ Text generation failed: {e}")
            return ""

    def _generate_with_simple_model(self, input_ids: torch.Tensor, max_new_tokens: int = 100,
                                    temperature: float = 0.8, top_p: float = 0.9) -> torch.Tensor:
        """Greedy/nucleus sampling generation loop for SimpleGPT model."""
        generated = input_ids.clone()
        eos_id = self.tokenizer.eos_token_id

        for _ in range(max_new_tokens):
            # Respect block size
            block_size = getattr(self.model.config, 'block_size', 256)
            if generated.size(1) > block_size:
                generated = generated[:, -block_size:]

            logits, _ = self.model(generated)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = torch.softmax(logits, dim=-1)

            if top_p < 1.0:
                # Nucleus sampling
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative = torch.cumsum(sorted_probs, dim=-1)
                cutoff = cumulative > top_p
                # Ensure at least one token
                cutoff[..., 0] = False
                sorted_probs[cutoff] = 0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                next_token_sorted = torch.multinomial(sorted_probs, num_samples=1)
                next_token = sorted_indices.gather(-1, next_token_sorted)
            else:
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            if eos_id is not None and next_token.item() == eos_id:
                break

        return generated
    
    def evaluate_historical_accuracy(self) -> Dict[str, float]:
        """Evaluate historical accuracy on key test cases"""
        print(f"🏛️ Evaluating historical accuracy...")
        
        test_cases = [
            {
                'prompt': 'In 1665, London was struck by',
                'expected_keywords': ['plague', 'pestilence', 'disease', 'death'],
                'expected_phrases': ['Great Plague', 'bubonic plague']
            },
            {
                'prompt': 'The King of England in 1665 was',
                'expected_keywords': ['Charles', 'King', 'monarch', 'throne'],
                'expected_phrases': ['Charles II', 'King Charles']
            },
            {
                'prompt': 'In 1666, London experienced',
                'expected_keywords': ['fire', 'burning', 'destruction', 'flames'],
                'expected_phrases': ['Great Fire', 'London Fire']
            },
            {
                'prompt': 'The Church of England was',
                'expected_keywords': ['Anglican', 'Protestant', 'Church', 'religion'],
                'expected_phrases': ['established church', 'Anglican Church']
            },
            {
                'prompt': 'The poor people of London',
                'expected_keywords': ['poverty', 'poor', 'beggars', 'workhouse'],
                'expected_phrases': ['poor relief', 'workhouse', 'alms']
            }
        ]
        
        correct_predictions = 0
        total_predictions = 0
        category_scores = {}
        
        for test_case in tqdm(test_cases, desc="Testing historical accuracy"):
            prompt = test_case['prompt']
            expected_keywords = test_case.get('expected_keywords', [])
            expected_phrases = test_case.get('expected_phrases', [])
            
            # Generate response
            generated_text = self.generate_text(prompt, max_length=100)
            
            if generated_text:
                generated_text_lower = generated_text.lower()
                
                # Check for expected keywords
                keyword_matches = sum(1 for keyword in expected_keywords if keyword.lower() in generated_text_lower)
                keyword_score = keyword_matches / len(expected_keywords) if expected_keywords else 0
                
                # Check for expected phrases
                phrase_matches = sum(1 for phrase in expected_phrases if phrase.lower() in generated_text_lower)
                phrase_score = phrase_matches / len(expected_phrases) if expected_phrases else 0
                
                # Overall score
                overall_score = (keyword_score + phrase_score) / 2 if (expected_keywords or expected_phrases) else 0
                
                if overall_score > 0.5:  # Threshold for correctness
                    correct_predictions += 1
                
                total_predictions += 1
                
                # Store sample
                self.results['generation_samples'].append({
                    'prompt': prompt,
                    'generated': generated_text,
                    'keyword_score': keyword_score,
                    'phrase_score': phrase_score,
                    'overall_score': overall_score
                })
        
        # Calculate overall accuracy
        overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        results = {
            'overall_accuracy': overall_accuracy,
            'correct_predictions': correct_predictions,
            'total_predictions': total_predictions
        }
        
        print(f"   Overall accuracy: {overall_accuracy:.2%}")
        print(f"   Correct predictions: {correct_predictions}/{total_predictions}")
        
        return results
    
    def evaluate_language_quality(self) -> Dict[str, float]:
        """Evaluate language quality metrics"""
        print(f"📝 Evaluating language quality...")
        
        # Generate texts for evaluation
        prompts = [
            "In the year of our Lord 1665, the Great Plague swept through London.",
            "The streets were empty, and the bells tolled for the dead.",
            "Methinks the city hath never seen such sorrow.",
            "Verily, I say unto you, this is a time of great tribulation.",
            "The King's men rode through the streets, seeking those who had fled."
        ]
        
        generated_texts = []
        for prompt in tqdm(prompts, desc="Generating texts for quality evaluation"):
            generated = self.generate_text(prompt, max_length=100)
            if generated:
                generated_texts.append(generated)
        
        if not generated_texts:
            print("❌ No generated texts for quality evaluation")
            return {}
        
        # Calculate quality metrics
        total_chars = sum(len(text) for text in generated_texts)
        total_words = sum(len(word_tokenize(text)) for text in generated_texts)
        total_sentences = sum(len(sent_tokenize(text)) for text in generated_texts)
        
        # Average metrics
        avg_chars_per_text = total_chars / len(generated_texts)
        avg_words_per_text = total_words / len(generated_texts)
        avg_sentences_per_text = total_sentences / len(generated_texts)
        avg_words_per_sentence = total_words / total_sentences if total_sentences > 0 else 0
        
        # Vocabulary diversity (unique words / total words)
        all_words = []
        for text in generated_texts:
            all_words.extend(word_tokenize(text.lower()))
        
        unique_words = len(set(all_words))
        vocabulary_diversity = unique_words / len(all_words) if all_words else 0
        
        # Check for historical language patterns
        historical_patterns = [
            r'\b(?:thou|thee|thy|thine|hast|hath|doth|dost|art|wilt|shalt)\b',
            r'\b(?:verily|indeed|forsooth|methinks|perchance|albeit)\b',
            r'\b(?:anon|ere|whilst|betwixt|amongst|amidst)\b',
            r'\b(?:prithee|pray\s+thee|I\s+pray\s+you)\b'
        ]
        
        historical_pattern_count = 0
        for text in generated_texts:
            for pattern in historical_patterns:
                import re
                if re.search(pattern, text, re.IGNORECASE):
                    historical_pattern_count += 1
                    break
        
        historical_pattern_ratio = historical_pattern_count / len(generated_texts)
        
        # Readability scores
        readability_scores = []
        for text in generated_texts:
            try:
                flesch_score = flesch_reading_ease(text)
                readability_scores.append(flesch_score)
            except:
                pass
        
        avg_readability = np.mean(readability_scores) if readability_scores else 0
        
        results = {
            'avg_chars_per_text': avg_chars_per_text,
            'avg_words_per_text': avg_words_per_text,
            'avg_sentences_per_text': avg_sentences_per_text,
            'avg_words_per_sentence': avg_words_per_sentence,
            'vocabulary_diversity': vocabulary_diversity,
            'historical_pattern_ratio': historical_pattern_ratio,
            'avg_readability': avg_readability,
            'total_generated_texts': len(generated_texts)
        }
        
        print(f"   Average words per text: {avg_words_per_text:.1f}")
        print(f"   Vocabulary diversity: {vocabulary_diversity:.3f}")
        print(f"   Historical pattern ratio: {historical_pattern_ratio:.3f}")
        print(f"   Average readability: {avg_readability:.1f}")
        
        return results
    
    def evaluate_coherence(self) -> Dict[str, float]:
        """Evaluate coherence using ROUGE scores"""
        print(f"🔗 Evaluating coherence...")
        
        test_prompts = [
            "In the year of our Lord 1665,",
            "The streets of London were",
            "Methinks the city",
            "Verily, I say unto you,",
            "The King's men"
        ]
        
        rouge_scores = []
        
        for prompt in tqdm(test_prompts, desc="Evaluating coherence"):
            generated_text = self.generate_text(prompt, max_length=100)
            
            if generated_text:
                # Calculate ROUGE scores
                rouge_score = self.rouge_scorer.score(prompt, generated_text)
                rouge_scores.append({
                    'rouge1': rouge_score['rouge1'].fmeasure,
                    'rouge2': rouge_score['rouge2'].fmeasure,
                    'rougeL': rouge_score['rougeL'].fmeasure
                })
        
        if rouge_scores:
            avg_rouge1 = np.mean([s['rouge1'] for s in rouge_scores])
            avg_rouge2 = np.mean([s['rouge2'] for s in rouge_scores])
            avg_rougeL = np.mean([s['rougeL'] for s in rouge_scores])
            
            results = {
                'avg_rouge1': avg_rouge1,
                'avg_rouge2': avg_rouge2,
                'avg_rougeL': avg_rougeL,
                'count': len(rouge_scores)
            }
            
            print(f"   Average ROUGE-1: {avg_rouge1:.3f}")
            print(f"   Average ROUGE-2: {avg_rouge2:.3f}")
            print(f"   Average ROUGE-L: {avg_rougeL:.3f}")
            
            return results
        else:
            print("❌ No coherence scores computed")
            return {}
    
    def run_quick_evaluation(self):
        """Run quick evaluation suite"""
        print(f"🚀 Running quick evaluation...")
        
        try:
            # 1. Historical accuracy evaluation
            historical_results = self.evaluate_historical_accuracy()
            self.results['metrics']['historical_accuracy'] = historical_results
            
            # 2. Language quality evaluation
            quality_results = self.evaluate_language_quality()
            self.results['metrics']['language_quality'] = quality_results
            
            # 3. Coherence evaluation
            coherence_results = self.evaluate_coherence()
            self.results['metrics']['coherence'] = coherence_results
            
            print("✅ Quick evaluation completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Quick evaluation failed: {e}")
            return False
    
    def save_results(self):
        """Save evaluation results"""
        print(f"💾 Saving results...")
        
        try:
            # Save JSON results
            results_file = self.output_dir / "quick_evaluation_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            
            # Save summary report
            summary_file = self.output_dir / "quick_evaluation_summary.txt"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("London Historical Model - Quick Evaluation Summary\n")
                f.write("=" * 50 + "\n\n")
                
                # Historical accuracy results
                if 'historical_accuracy' in self.results['metrics']:
                    f.write("Historical Accuracy:\n")
                    for key, value in self.results['metrics']['historical_accuracy'].items():
                        f.write(f"  {key}: {value:.3f}\n")
                    f.write("\n")
                
                # Language quality results
                if 'language_quality' in self.results['metrics']:
                    f.write("Language Quality:\n")
                    for key, value in self.results['metrics']['language_quality'].items():
                        f.write(f"  {key}: {value:.3f}\n")
                    f.write("\n")
                
                # Coherence results
                if 'coherence' in self.results['metrics']:
                    f.write("Coherence:\n")
                    for key, value in self.results['metrics']['coherence'].items():
                        f.write(f"  {key}: {value:.3f}\n")
                    f.write("\n")
                
                # Generation samples
                if 'generation_samples' in self.results:
                    f.write("Generation Samples:\n")
                    for i, sample in enumerate(self.results['generation_samples']):
                        f.write(f"  Sample {i+1}:\n")
                        f.write(f"    Prompt: {sample['prompt']}\n")
                        f.write(f"    Generated: {sample['generated']}\n")
                        f.write(f"    Overall Score: {sample['overall_score']:.3f}\n")
                        f.write("\n")
            
            print(f"✅ Results saved to: {self.output_dir}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save results: {e}")
            return False
    
    def print_summary(self):
        """Print evaluation summary"""
        print("\n" + "="*60)
        print(f"LONDON HISTORICAL MODEL - QUICK EVALUATION SUMMARY")
        print("="*60)
        
        # Historical accuracy results
        if 'historical_accuracy' in self.results['metrics']:
            print(f"🏛️ Historical Accuracy:")
            for key, value in self.results['metrics']['historical_accuracy'].items():
                print(f"  {key}: {value:.3f}")
            print()
        
        # Language quality results
        if 'language_quality' in self.results['metrics']:
            print(f"📝 Language Quality:")
            for key, value in self.results['metrics']['language_quality'].items():
                print(f"  {key}: {value:.3f}")
            print()
        
        # Coherence results
        if 'coherence' in self.results['metrics']:
            print(f"🔗 Coherence:")
            for key, value in self.results['metrics']['coherence'].items():
                print(f"  {key}: {value:.3f}")
            print()
        
        # Generation samples
        if 'generation_samples' in self.results:
            print(f"📝 Generation Samples:")
            for i, sample in enumerate(self.results['generation_samples'][:3]):  # Show first 3
                print(f"  Sample {i+1}:")
                print(f"    Prompt: {sample['prompt']}")
                print(f"    Generated: {sample['generated'][:100]}{'...' if len(sample['generated']) > 100 else ''}")
                print(f"    Score: {sample['overall_score']:.3f}")
                print()
        
        print(f"Results saved to: {self.output_dir}")
        print("="*60)

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Quick Evaluation for London Historical Models")
    parser.add_argument("--model_dir", type=str, default="09_models/checkpoints",
                       help="Directory containing trained model")
    parser.add_argument("--tokenizer_dir", type=str, 
                       default="09_models/tokenizers/london_historical_tokenizer",
                       help="Directory containing tokenizer")
    parser.add_argument("--output_dir", type=str, default="quick_results",
                       help="Directory to save evaluation results")
    parser.add_argument("--device", type=str, choices=["cpu", "gpu"], default="cpu",
                       help="Device to use for evaluation (default: cpu for safety)")
    
    args = parser.parse_args()
    
    print(f"London Historical Model - Quick Evaluation")
    print("=" * 50)
    
    # Initialize evaluator
    evaluator = QuickEvaluator(
        model_dir=args.model_dir,
        tokenizer_dir=args.tokenizer_dir,
        output_dir=args.output_dir,
        device=args.device
    )
    
    try:
        # Load model and tokenizer
        if not evaluator.load_model_and_tokenizer():
            return False
        
        # Run quick evaluation
        if not evaluator.run_quick_evaluation():
            return False
        
        # Save results
        if not evaluator.save_results():
            return False
        
        # Print summary
        evaluator.print_summary()
        
        print("✅ Quick evaluation completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
