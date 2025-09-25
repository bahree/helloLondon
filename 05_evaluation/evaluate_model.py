#!/usr/bin/env python3
"""
Model Evaluation for London Historical LLM
Evaluates the trained model on various metrics and tasks
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
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import Dataset as HFDataset
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import seaborn as sns
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
        logging.FileHandler('model_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LondonHistoricalEvaluator:
    def __init__(self, 
                 model_dir: str = "09_models/checkpoints",
                 tokenizer_dir: str = "09_models/tokenizers/london_historical_tokenizer",
                 output_dir: str = "results"):
        
        self.model_dir = Path(model_dir)
        self.tokenizer_dir = Path(tokenizer_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Evaluation results
        self.evaluation_results = {
            'model_dir': str(self.model_dir),
            'tokenizer_dir': str(self.tokenizer_dir),
            'evaluation_time': datetime.now().isoformat(),
            'device': str(self.device),
            'metrics': {},
            'generation_samples': [],
            'perplexity_scores': {},
            'historical_accuracy': {},
            'language_quality': {}
        }
    
    def load_model_and_tokenizer(self):
        """Load the trained model and tokenizer"""
        logger.info(f"Loading model and tokenizer...")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.tokenizer_dir))
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(str(self.model_dir))
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model and tokenizer loaded successfully")
            logger.info(f"   Device: {self.device}")
            logger.info(f"   Vocabulary size: {self.tokenizer.vocab_size:,}")
            logger.info(f"   Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model/tokenizer: {e}")
            return False
    
    def generate_text(self, prompt: str, max_length: int = 100, 
                     temperature: float = 0.8, top_p: float = 0.9,
                     num_return_sequences: int = 1) -> List[str]:
        """Generate text from a prompt"""
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    num_return_sequences=num_return_sequences,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode outputs
            generated_texts = []
            for output in outputs:
                generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
                # Remove the original prompt
                generated_text = generated_text[len(prompt):].strip()
                generated_texts.append(generated_text)
            
            return generated_texts
            
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            return []
    
    def evaluate_perplexity(self, test_texts: List[str]) -> Dict[str, float]:
        """Evaluate perplexity on test texts"""
        logger.info(f"Evaluating perplexity...")
        
        perplexities = []
        
        for text in tqdm(test_texts, desc="Computing perplexity"):
            try:
                # Tokenize text
                inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Compute loss
                with torch.no_grad():
                    outputs = self.model(**inputs, labels=inputs['input_ids'])
                    loss = outputs.loss
                    perplexity = torch.exp(loss).item()
                    perplexities.append(perplexity)
                    
            except Exception as e:
                logger.warning(f"Error computing perplexity for text: {e}")
                continue
        
        if perplexities:
            avg_perplexity = np.mean(perplexities)
            std_perplexity = np.std(perplexities)
            min_perplexity = np.min(perplexities)
            max_perplexity = np.max(perplexities)
            
            results = {
                'average': avg_perplexity,
                'std': std_perplexity,
                'min': min_perplexity,
                'max': max_perplexity,
                'count': len(perplexities)
            }
            
            logger.info(f"Perplexity evaluation completed")
            logger.info(f"   Average: {avg_perplexity:.2f}")
            logger.info(f"   Std: {std_perplexity:.2f}")
            logger.info(f"   Min: {min_perplexity:.2f}")
            logger.info(f"   Max: {max_perplexity:.2f}")
            
            return results
        else:
            logger.error(f"No valid perplexity scores computed")
            return {}
    
    def evaluate_historical_accuracy(self, test_cases: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate historical accuracy on specific test cases"""
        logger.info("🏛️ Evaluating historical accuracy...")
        
        correct_predictions = 0
        total_predictions = 0
        category_scores = {}
        
        for test_case in tqdm(test_cases, desc="Testing historical accuracy"):
            category = test_case.get('category', 'general')
            prompt = test_case['prompt']
            expected_keywords = test_case.get('expected_keywords', [])
            expected_phrases = test_case.get('expected_phrases', [])
            
            # Generate response
            generated_texts = self.generate_text(prompt, max_length=100, num_return_sequences=1)
            
            if generated_texts:
                generated_text = generated_texts[0].lower()
                
                # Check for expected keywords
                keyword_matches = sum(1 for keyword in expected_keywords if keyword.lower() in generated_text)
                keyword_score = keyword_matches / len(expected_keywords) if expected_keywords else 0
                
                # Check for expected phrases
                phrase_matches = sum(1 for phrase in expected_phrases if phrase.lower() in generated_text)
                phrase_score = phrase_matches / len(expected_phrases) if expected_phrases else 0
                
                # Overall score
                overall_score = (keyword_score + phrase_score) / 2 if (expected_keywords or expected_phrases) else 0
                
                if overall_score > 0.5:  # Threshold for correctness
                    correct_predictions += 1
                
                total_predictions += 1
                
                # Track by category
                if category not in category_scores:
                    category_scores[category] = []
                category_scores[category].append(overall_score)
        
        # Calculate overall accuracy
        overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        # Calculate category accuracies
        category_accuracies = {}
        for category, scores in category_scores.items():
            category_accuracies[category] = np.mean(scores)
        
        results = {
            'overall_accuracy': overall_accuracy,
            'correct_predictions': correct_predictions,
            'total_predictions': total_predictions,
            'category_accuracies': category_accuracies
        }
        
        logger.info(f"Historical accuracy evaluation completed")
        logger.info(f"   Overall accuracy: {overall_accuracy:.2%}")
        logger.info(f"   Correct predictions: {correct_predictions}/{total_predictions}")
        
        return results
    
    def evaluate_language_quality(self, test_texts: List[str]) -> Dict[str, float]:
        """Evaluate language quality metrics"""
        logger.info("📝 Evaluating language quality...")
        
        # Generate texts for evaluation
        generated_texts = []
        for text in tqdm(test_texts, desc="Generating texts for quality evaluation"):
            generated = self.generate_text(text, max_length=100, num_return_sequences=1)
            if generated:
                generated_texts.extend(generated)
        
        if not generated_texts:
            logger.error(f"No generated texts for quality evaluation")
            return {}
        
        # Calculate quality metrics
        total_chars = sum(len(text) for text in generated_texts)
        total_words = sum(len(text.split()) for text in generated_texts)
        total_sentences = sum(len([s for s in text.split('.') if s.strip()]) for text in generated_texts)
        
        # Average metrics
        avg_chars_per_text = total_chars / len(generated_texts)
        avg_words_per_text = total_words / len(generated_texts)
        avg_sentences_per_text = total_sentences / len(generated_texts)
        avg_words_per_sentence = total_words / total_sentences if total_sentences > 0 else 0
        
        # Vocabulary diversity (unique words / total words)
        all_words = []
        for text in generated_texts:
            all_words.extend(text.lower().split())
        
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
        
        results = {
            'avg_chars_per_text': avg_chars_per_text,
            'avg_words_per_text': avg_words_per_text,
            'avg_sentences_per_text': avg_sentences_per_text,
            'avg_words_per_sentence': avg_words_per_sentence,
            'vocabulary_diversity': vocabulary_diversity,
            'historical_pattern_ratio': historical_pattern_ratio,
            'total_generated_texts': len(generated_texts)
        }
        
        logger.info(f"Language quality evaluation completed")
        logger.info(f"   Average words per text: {avg_words_per_text:.1f}")
        logger.info(f"   Vocabulary diversity: {vocabulary_diversity:.3f}")
        logger.info(f"   Historical pattern ratio: {historical_pattern_ratio:.3f}")
        
        return results
    
    def run_comprehensive_evaluation(self):
        """Run comprehensive evaluation suite"""
        logger.info(f"Running comprehensive evaluation...")
        
        # Test texts for perplexity
        test_texts = [
            "In the year of our Lord 1665, the Great Plague swept through London.",
            "The streets were empty, and the bells tolled for the dead.",
            "Methinks the city hath never seen such sorrow.",
            "Verily, I say unto you, this is a time of great tribulation.",
            "The King's men rode through the streets, seeking those who had fled."
        ]
        
        # Historical accuracy test cases
        historical_test_cases = [
            {
                'category': 'plague',
                'prompt': 'In 1665, London was struck by',
                'expected_keywords': ['plague', 'pestilence', 'disease', 'death'],
                'expected_phrases': ['Great Plague', 'bubonic plague']
            },
            {
                'category': 'royalty',
                'prompt': 'The King of England in 1665 was',
                'expected_keywords': ['Charles', 'King', 'monarch', 'throne'],
                'expected_phrases': ['Charles II', 'King Charles']
            },
            {
                'category': 'religion',
                'prompt': 'The Church of England was',
                'expected_keywords': ['Anglican', 'Protestant', 'Church', 'religion'],
                'expected_phrases': ['established church', 'Anglican Church']
            },
            {
                'category': 'social',
                'prompt': 'The poor people of London',
                'expected_keywords': ['poverty', 'poor', 'beggars', 'workhouse'],
                'expected_phrases': ['poor relief', 'workhouse', 'alms']
            }
        ]
        
        # Run evaluations
        try:
            # Perplexity evaluation
            perplexity_results = self.evaluate_perplexity(test_texts)
            self.evaluation_results['perplexity_scores'] = perplexity_results
            
            # Historical accuracy evaluation
            historical_results = self.evaluate_historical_accuracy(historical_test_cases)
            self.evaluation_results['historical_accuracy'] = historical_results
            
            # Language quality evaluation
            quality_results = self.evaluate_language_quality(test_texts)
            self.evaluation_results['language_quality'] = quality_results
            
            # Generate sample texts
            sample_prompts = [
                "In the year of our Lord 1665,",
                "The streets of London were",
                "Methinks the city",
                "Verily, I say unto you,",
                "The King's men"
            ]
            
            logger.info("📝 Generating sample texts...")
            for prompt in sample_prompts:
                generated = self.generate_text(prompt, max_length=100, num_return_sequences=1)
                if generated:
                    self.evaluation_results['generation_samples'].append({
                        'prompt': prompt,
                        'generated': generated[0],
                        'timestamp': datetime.now().isoformat()
                    })
            
            logger.info(f"Comprehensive evaluation completed")
            return True
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return False
    
    def save_results(self):
        """Save evaluation results"""
        logger.info(f"Saving evaluation results...")
        
        try:
            # Save JSON results
            results_file = self.output_dir / "evaluation_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(self.evaluation_results, f, indent=2, ensure_ascii=False)
            
            # Save summary report
            summary_file = self.output_dir / "evaluation_summary.txt"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("London Historical LLM - Evaluation Summary\n")
                f.write("=" * 50 + "\n\n")
                
                # Perplexity results
                if 'perplexity_scores' in self.evaluation_results:
                    f.write("Perplexity Scores:\n")
                    for key, value in self.evaluation_results['perplexity_scores'].items():
                        f.write(f"  {key}: {value:.2f}\n")
                    f.write("\n")
                
                # Historical accuracy results
                if 'historical_accuracy' in self.evaluation_results:
                    f.write("Historical Accuracy:\n")
                    f.write(f"  Overall accuracy: {self.evaluation_results['historical_accuracy'].get('overall_accuracy', 0):.2%}\n")
                    f.write(f"  Correct predictions: {self.evaluation_results['historical_accuracy'].get('correct_predictions', 0)}/{self.evaluation_results['historical_accuracy'].get('total_predictions', 0)}\n")
                    f.write("\n")
                
                # Language quality results
                if 'language_quality' in self.evaluation_results:
                    f.write("Language Quality:\n")
                    for key, value in self.evaluation_results['language_quality'].items():
                        if isinstance(value, float):
                            f.write(f"  {key}: {value:.3f}\n")
                        else:
                            f.write(f"  {key}: {value}\n")
                    f.write("\n")
                
                # Generation samples
                if 'generation_samples' in self.evaluation_results:
                    f.write("Generation Samples:\n")
                    for i, sample in enumerate(self.evaluation_results['generation_samples']):
                        f.write(f"  Sample {i+1}:\n")
                        f.write(f"    Prompt: {sample['prompt']}\n")
                        f.write(f"    Generated: {sample['generated']}\n")
                        f.write("\n")
            
            logger.info(f"Results saved to: {self.output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return False
    
    def print_summary(self):
        """Print evaluation summary"""
        print("\n" + "="*70)
        print(f"MODEL EVALUATION SUMMARY")
        print("="*70)
        
        # Perplexity results
        if 'perplexity_scores' in self.evaluation_results:
            print(f"Perplexity Scores:")
            for key, value in self.evaluation_results['perplexity_scores'].items():
                print(f"  {key}: {value:.2f}")
            print()
        
        # Historical accuracy results
        if 'historical_accuracy' in self.evaluation_results:
            print(f"Historical Accuracy:")
            print(f"  Overall accuracy: {self.evaluation_results['historical_accuracy'].get('overall_accuracy', 0):.2%}")
            print(f"  Correct predictions: {self.evaluation_results['historical_accuracy'].get('correct_predictions', 0)}/{self.evaluation_results['historical_accuracy'].get('total_predictions', 0)}")
            print()
        
        # Language quality results
        if 'language_quality' in self.evaluation_results:
            print(f"Language Quality:")
            for key, value in self.evaluation_results['language_quality'].items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")
            print()
        
        # Generation samples
        if 'generation_samples' in self.evaluation_results:
            print(f"Generation Samples:")
            for i, sample in enumerate(self.evaluation_results['generation_samples'][:3]):  # Show first 3
                print(f"  Sample {i+1}:")
                print(f"    Prompt: {sample['prompt']}")
                print(f"    Generated: {sample['generated'][:100]}{'...' if len(sample['generated']) > 100 else ''}")
                print()
        
        print(f"Results saved to: {self.output_dir}")
        print("="*70)

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Evaluate London Historical LLM")
    parser.add_argument("--model_dir", type=str, default="09_models/checkpoints",
                       help="Directory containing trained model")
    parser.add_argument("--tokenizer_dir", type=str, 
                       default="09_models/tokenizers/london_historical_tokenizer",
                       help="Directory containing tokenizer")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Directory to save evaluation results")
    
    args = parser.parse_args()
    
    print(f"London Historical LLM - Model Evaluation")
    print("=" * 50)
    
    # Initialize evaluator
    evaluator = LondonHistoricalEvaluator(
        model_dir=args.model_dir,
        tokenizer_dir=args.tokenizer_dir,
        output_dir=args.output_dir
    )
    
    try:
        # Load model and tokenizer
        if not evaluator.load_model_and_tokenizer():
            return False
        
        # Run comprehensive evaluation
        if not evaluator.run_comprehensive_evaluation():
            return False
        
        # Save results
        if not evaluator.save_results():
            return False
        
        # Print summary
        evaluator.print_summary()
        
        logger.info(f"Model evaluation completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
