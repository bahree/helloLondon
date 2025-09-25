#!/usr/bin/env python3
"""
Comprehensive Evaluation Framework for London Historical SLM (1500-1850)
Implements modern LLM evaluation metrics including G-Eval, HELM, and historical-specific tests
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import time
from datetime import datetime
import asyncio
import aiohttp
import re
from dataclasses import dataclass
from enum import Enum

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from datasets import Dataset as HFDataset, load_dataset
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import openai
    from rouge_score import rouge_scorer
    from bert_score import score as bert_score
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    import spacy
    from textstat import flesch_reading_ease, flesch_kincaid_grade
    import warnings
    warnings.filterwarnings("ignore")
except ImportError as e:
    print(f"Missing required dependencies: {e}")
    print("Please install requirements: pip install -r requirements.txt")
    sys.exit(1)

# Download required NLTK data (robust across NLTK versions)
try:
    for resource in ['punkt', 'punkt_tab', 'stopwords', 'averaged_perceptron_tagger']:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass
except Exception:
    pass

class EvaluationType(Enum):
    """Types of evaluations to run"""
    AUTOMATED = "automated"
    HISTORICAL = "historical"
    BENCHMARK = "benchmark"
    HUMAN = "human"
    COMPREHENSIVE = "comprehensive"

@dataclass
class EvaluationResult:
    """Container for evaluation results"""
    metric_name: str
    score: float
    confidence: float
    details: Dict[str, Any]
    timestamp: str

class HistoricalLondonEvaluator:
    """Comprehensive evaluator for historical London SLM"""
    
    def __init__(self, 
                 model_dir: str = "09_models/checkpoints",
                 tokenizer_dir: str = "09_models/tokenizers/london_historical_tokenizer",
                 output_dir: str = "results",
                 openai_api_key: Optional[str] = None,
                 device: str = "cpu"):
        
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
            print("Using GPU for evaluation")
        else:
            self.device = torch.device("cpu")
            if device.lower() == "gpu" and not torch.cuda.is_available():
                print("GPU requested but not available, falling back to CPU")
            else:
                print("Using CPU for evaluation (safe default)")
        
        # Evaluation results storage
        self.evaluation_results = {
            'model_dir': str(self.model_dir),
            'tokenizer_dir': str(self.tokenizer_dir),
            'evaluation_time': datetime.now().isoformat(),
            'device': str(self.device),
            'evaluation_type': 'comprehensive',
            'metrics': {},
            'detailed_results': [],
            'generation_samples': [],
            'benchmark_scores': {},
            'historical_accuracy': {},
            'language_quality': {},
            'coherence_scores': {},
            'fluency_scores': {},
            'groundedness_scores': {}
        }
        
        # Initialize OpenAI for G-Eval
        if openai_api_key:
            openai.api_key = openai_api_key
            self.openai_available = True
        else:
            self.openai_available = False
            print("⚠️  OpenAI API key not provided. G-Eval will be disabled.")
        
        # Initialize text processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("⚠️  spaCy English model not found. Some evaluations may be limited.")
            self.nlp = None
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'comprehensive_evaluation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_model_and_tokenizer(self):
        """Load the trained model and tokenizer"""
        self.logger.info(f"Loading model and tokenizer...")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.tokenizer_dir))
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model - try different approaches
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
            
            self.logger.info(f"Model and tokenizer loaded successfully")
            self.logger.info(f"   Device: {self.device}")
            self.logger.info(f"   Vocabulary size: {self.tokenizer.vocab_size:,}")
            self.logger.info(f"   Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model/tokenizer: {e}")
            return False
    
    def generate_text(self, prompt: str, max_length: int = 100, 
                     temperature: float = 0.7, top_p: float = 0.85,
                     num_return_sequences: int = 1) -> List[str]:
        """Generate text from a prompt (supports SimpleGPT fallback)."""
        try:
            enc = self.tokenizer(
                prompt,
                return_tensors='pt',
                add_special_tokens=False
            )
            input_ids = enc['input_ids'].to(self.device)

            with torch.no_grad():
                if hasattr(self.model, 'generate'):
                    outputs = self.model.generate(
                        input_ids=input_ids,
                        max_new_tokens=max_length,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=1.1,
                        num_return_sequences=num_return_sequences
                    )
                    gen_ids_list = [o for o in outputs]
                else:
                    gen_ids = self._generate_with_simple_model(
                        input_ids=input_ids,
                        max_new_tokens=max_length,
                        temperature=temperature,
                        top_p=top_p
                    )
                    gen_ids_list = [gen_ids]

            prompt_len = input_ids.shape[1]
            texts = []
            for gen_ids in gen_ids_list:
                new_tokens = gen_ids[prompt_len:]
                text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                texts.append(text)
            return texts
        except Exception as e:
            self.logger.error(f"Text generation failed: {e}")
            return []

    def _generate_with_simple_model(self, input_ids: torch.Tensor, max_new_tokens: int = 100,
                                    temperature: float = 0.7, top_p: float = 0.85) -> torch.Tensor:
        """Autoregressive generation loop for SimpleGPT."""
        generated = input_ids.clone()
        eos_id = self.tokenizer.eos_token_id

        for _ in range(max_new_tokens):
            block_size = getattr(self.model.config, 'block_size', 256)
            if generated.size(1) > block_size:
                generated = generated[:, -block_size:]

            logits, _ = self.model(generated)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = torch.softmax(logits, dim=-1)

            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative = torch.cumsum(sorted_probs, dim=-1)
                cutoff = cumulative > top_p
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

        return generated[0]
    
    def evaluate_groundedness(self, test_cases: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate groundedness using G-Eval methodology"""
        self.logger.info("🔍 Evaluating groundedness...")
        
        if not self.openai_available:
            self.logger.warning("OpenAI API not available. Skipping groundedness evaluation.")
            return {}
        
        groundedness_scores = []
        
        for test_case in tqdm(test_cases, desc="Evaluating groundedness"):
            prompt = test_case['prompt']
            context = test_case.get('context', '')
            expected_facts = test_case.get('expected_facts', [])
            
            # Generate response
            generated_texts = self.generate_text(prompt, max_length=150, num_return_sequences=1)
            
            if generated_texts:
                generated_text = generated_texts[0]
                
                # Use G-Eval approach with GPT-4
                try:
                    groundedness_score = self._evaluate_with_gpt4(
                        prompt=prompt,
                        context=context,
                        generated_text=generated_text,
                        expected_facts=expected_facts,
                        evaluation_type="groundedness"
                    )
                    groundedness_scores.append(groundedness_score)
                except Exception as e:
                    self.logger.warning(f"G-Eval failed for test case: {e}")
                    continue
        
        if groundedness_scores:
            results = {
                'average_groundedness': np.mean(groundedness_scores),
                'std_groundedness': np.std(groundedness_scores),
                'min_groundedness': np.min(groundedness_scores),
                'max_groundedness': np.max(groundedness_scores),
                'count': len(groundedness_scores)
            }
            
            self.logger.info(f"Groundedness evaluation completed")
            self.logger.info(f"   Average: {results['average_groundedness']:.3f}")
            self.logger.info(f"   Std: {results['std_groundedness']:.3f}")
            
            return results
        else:
            self.logger.error("No groundedness scores computed")
            return {}
    
    def evaluate_coherence(self, test_cases: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate coherence using multiple metrics"""
        self.logger.info("🔗 Evaluating coherence...")
        
        coherence_scores = []
        
        for test_case in tqdm(test_cases, desc="Evaluating coherence"):
            prompt = test_case['prompt']
            
            # Generate response
            generated_texts = self.generate_text(prompt, max_length=150, num_return_sequences=1)
            
            if generated_texts:
                generated_text = generated_texts[0]
                
                # Calculate coherence metrics
                coherence_score = self._calculate_coherence_metrics(prompt, generated_text)
                coherence_scores.append(coherence_score)
        
        if coherence_scores:
            results = {
                'average_coherence': np.mean(coherence_scores),
                'std_coherence': np.std(coherence_scores),
                'min_coherence': np.min(coherence_scores),
                'max_coherence': np.max(coherence_scores),
                'count': len(coherence_scores)
            }
            
            self.logger.info(f"Coherence evaluation completed")
            self.logger.info(f"   Average: {results['average_coherence']:.3f}")
            self.logger.info(f"   Std: {results['std_coherence']:.3f}")
            
            return results
        else:
            self.logger.error("No coherence scores computed")
            return {}
    
    def evaluate_fluency(self, test_cases: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate fluency using multiple linguistic metrics"""
        self.logger.info("💬 Evaluating fluency...")
        
        fluency_scores = []
        
        for test_case in tqdm(test_cases, desc="Evaluating fluency"):
            prompt = test_case['prompt']
            
            # Generate response
            generated_texts = self.generate_text(prompt, max_length=150, num_return_sequences=1)
            
            if generated_texts:
                generated_text = generated_texts[0]
                
                # Calculate fluency metrics
                fluency_score = self._calculate_fluency_metrics(generated_text)
                fluency_scores.append(fluency_score)
        
        if fluency_scores:
            results = {
                'average_fluency': np.mean(fluency_scores),
                'std_fluency': np.std(fluency_scores),
                'min_fluency': np.min(fluency_scores),
                'max_fluency': np.max(fluency_scores),
                'count': len(fluency_scores)
            }
            
            self.logger.info(f"Fluency evaluation completed")
            self.logger.info(f"   Average: {results['average_fluency']:.3f}")
            self.logger.info(f"   Std: {results['std_fluency']:.3f}")
            
            return results
        else:
            self.logger.error("No fluency scores computed")
            return {}
    
    def _evaluate_with_gpt4(self, prompt: str, context: str, generated_text: str, 
                           expected_facts: List[str], evaluation_type: str) -> float:
        """Use GPT-4 for evaluation (G-Eval approach)"""
        if not self.openai_available:
            return 0.0
        
        # Create evaluation prompt based on type
        if evaluation_type == "groundedness":
            eval_prompt = f"""
            Evaluate the groundedness of the following generated text on a scale of 1-10.
            Consider how well the text is based on factual information and historical accuracy.
            
            Context: {context}
            Expected facts: {', '.join(expected_facts)}
            Generated text: {generated_text}
            
            Rate the groundedness (1=not grounded, 10=highly grounded):
            """
        elif evaluation_type == "coherence":
            eval_prompt = f"""
            Evaluate the coherence of the following generated text on a scale of 1-10.
            Consider logical flow, consistency, and overall structure.
            
            Prompt: {prompt}
            Generated text: {generated_text}
            
            Rate the coherence (1=incoherent, 10=highly coherent):
            """
        elif evaluation_type == "fluency":
            eval_prompt = f"""
            Evaluate the fluency of the following generated text on a scale of 1-10.
            Consider grammar, readability, and natural language flow.
            
            Generated text: {generated_text}
            
            Rate the fluency (1=not fluent, 10=highly fluent):
            """
        else:
            return 0.0
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": eval_prompt}],
                max_tokens=10,
                temperature=0.0
            )
            
            # Extract score from response
            score_text = response.choices[0].message.content.strip()
            score = float(re.findall(r'\d+', score_text)[0]) if re.findall(r'\d+', score_text) else 5.0
            return min(max(score, 1.0), 10.0) / 10.0  # Normalize to 0-1
            
        except Exception as e:
            self.logger.warning(f"GPT-4 evaluation failed: {e}")
            return 0.5  # Default neutral score
    
    def _calculate_coherence_metrics(self, prompt: str, generated_text: str) -> float:
        """Calculate coherence metrics using multiple approaches"""
        scores = []
        
        # 1. Sentence-level coherence using BERTScore
        try:
            sentences = sent_tokenize(generated_text)
            if len(sentences) > 1:
                # Calculate BERTScore between consecutive sentences
                for i in range(len(sentences) - 1):
                    P, R, F1 = bert_score([sentences[i]], [sentences[i+1]], lang="en")
                    scores.append(F1.item())
        except Exception:
            pass
        
        # 2. ROUGE-L for coherence with prompt
        try:
            rouge_scores = self.rouge_scorer.score(prompt, generated_text)
            scores.append(rouge_scores['rougeL'].fmeasure)
        except Exception:
            pass
        
        # 3. Vocabulary overlap
        try:
            prompt_words = set(word_tokenize(prompt.lower()))
            generated_words = set(word_tokenize(generated_text.lower()))
            if prompt_words and generated_words:
                overlap = len(prompt_words.intersection(generated_words)) / len(prompt_words.union(generated_words))
                scores.append(overlap)
        except Exception:
            pass
        
        return np.mean(scores) if scores else 0.5
    
    def _calculate_fluency_metrics(self, generated_text: str) -> float:
        """Calculate fluency metrics using multiple approaches"""
        scores = []
        
        # 1. Flesch Reading Ease
        try:
            flesch_score = flesch_reading_ease(generated_text)
            # Normalize to 0-1 (higher is better)
            normalized_flesch = min(max(flesch_score / 100.0, 0.0), 1.0)
            scores.append(normalized_flesch)
        except Exception:
            pass
        
        # 2. Sentence length consistency
        try:
            sentences = sent_tokenize(generated_text)
            if len(sentences) > 1:
                sentence_lengths = [len(word_tokenize(s)) for s in sentences]
                # Lower variance in sentence lengths indicates better fluency
                length_variance = np.var(sentence_lengths)
                # Normalize (lower variance = higher fluency)
                fluency_from_length = max(0, 1 - (length_variance / 100))
                scores.append(fluency_from_length)
        except Exception:
            pass
        
        # 3. Word repetition penalty
        try:
            words = word_tokenize(generated_text.lower())
            if words:
                unique_words = len(set(words))
                total_words = len(words)
                repetition_penalty = unique_words / total_words
                scores.append(repetition_penalty)
        except Exception:
            pass
        
        return np.mean(scores) if scores else 0.5
    
    def run_benchmark_evaluation(self) -> Dict[str, float]:
        """Run benchmark evaluations (MMLU subset, HellaSWAG, etc.)"""
        self.logger.info("📊 Running benchmark evaluations...")
        
        benchmark_scores = {}
        
        # 1. MMLU subset evaluation
        try:
            mmlu_score = self._evaluate_mmlu_subset()
            benchmark_scores['mmlu'] = mmlu_score
        except Exception as e:
            self.logger.warning(f"MMLU evaluation failed: {e}")
        
        # 2. HellaSWAG evaluation
        try:
            hellaswag_score = self._evaluate_hellaswag()
            benchmark_scores['hellaswag'] = hellaswag_score
        except Exception as e:
            self.logger.warning(f"HellaSWAG evaluation failed: {e}")
        
        # 3. Historical accuracy benchmark
        try:
            historical_score = self._evaluate_historical_benchmark()
            benchmark_scores['historical_accuracy'] = historical_score
        except Exception as e:
            self.logger.warning(f"Historical benchmark evaluation failed: {e}")
        
        self.logger.info(f"Benchmark evaluation completed")
        for benchmark, score in benchmark_scores.items():
            self.logger.info(f"   {benchmark}: {score:.3f}")
        
        return benchmark_scores
    
    def _evaluate_mmlu_subset(self) -> float:
        """Evaluate on a subset of MMLU questions"""
        # Load MMLU subset
        try:
            mmlu_dataset = load_dataset("lukaemon/mmlu", "all", split="test")
            # Select a subset for evaluation
            subset_size = min(100, len(mmlu_dataset))
            mmlu_subset = mmlu_dataset.select(range(subset_size))
        except Exception as e:
            self.logger.warning(f"Could not load MMLU dataset: {e}")
            return 0.0
        
        correct = 0
        total = 0
        
        for item in tqdm(mmlu_subset, desc="Evaluating MMLU subset"):
            question = item['question']
            choices = item['choices']
            answer = item['answer']
            
            # Create prompt
            prompt = f"Question: {question}\nChoices: {', '.join(choices)}\nAnswer:"
            
            # Generate response
            generated_texts = self.generate_text(prompt, max_length=50, num_return_sequences=1)
            
            if generated_texts:
                generated_text = generated_texts[0].lower()
                
                # Check if any choice is mentioned in the response
                for i, choice in enumerate(choices):
                    if choice.lower() in generated_text:
                        if i == answer:
                            correct += 1
                        break
                
                total += 1
        
        return correct / total if total > 0 else 0.0
    
    def _evaluate_hellaswag(self) -> float:
        """Evaluate on HellaSWAG dataset"""
        try:
            hellaswag_dataset = load_dataset("Rowan/hellaswag", split="test")
            # Select a subset for evaluation
            subset_size = min(100, len(hellaswag_dataset))
            hellaswag_subset = hellaswag_dataset.select(range(subset_size))
        except Exception as e:
            self.logger.warning(f"Could not load HellaSWAG dataset: {e}")
            return 0.0
        
        correct = 0
        total = 0
        
        for item in tqdm(hellaswag_subset, desc="Evaluating HellaSWAG"):
            context = item['ctx']
            endings = item['endings']
            label = item['label']
            
            # Create prompt
            prompt = f"Context: {context}\nEndings: {', '.join(endings)}\nBest ending:"
            
            # Generate response
            generated_texts = self.generate_text(prompt, max_length=50, num_return_sequences=1)
            
            if generated_texts:
                generated_text = generated_texts[0].lower()
                
                # Check which ending is most similar to the response
                best_match = 0
                best_score = 0
                
                for i, ending in enumerate(endings):
                    # Simple similarity check
                    ending_words = set(word_tokenize(ending.lower()))
                    response_words = set(word_tokenize(generated_text))
                    similarity = len(ending_words.intersection(response_words)) / len(ending_words.union(response_words))
                    
                    if similarity > best_score:
                        best_score = similarity
                        best_match = i
                
                if best_match == label:
                    correct += 1
                
                total += 1
        
        return correct / total if total > 0 else 0.0
    
    def _evaluate_historical_benchmark(self) -> float:
        """Evaluate on historical accuracy benchmark"""
        historical_test_cases = [
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
                'prompt': 'The Church of England was',
                'expected_keywords': ['Anglican', 'Protestant', 'Church', 'religion'],
                'expected_phrases': ['established church', 'Anglican Church']
            },
            {
                'prompt': 'The poor people of London',
                'expected_keywords': ['poverty', 'poor', 'beggars', 'workhouse'],
                'expected_phrases': ['poor relief', 'workhouse', 'alms']
            },
            {
                'prompt': 'In 1666, London experienced',
                'expected_keywords': ['fire', 'burning', 'destruction', 'flames'],
                'expected_phrases': ['Great Fire', 'London Fire']
            }
        ]
        
        correct = 0
        total = 0
        
        for test_case in tqdm(historical_test_cases, desc="Evaluating historical benchmark"):
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
                    correct += 1
                
                total += 1
        
        return correct / total if total > 0 else 0.0
    
    def run_comprehensive_evaluation(self):
        """Run comprehensive evaluation suite"""
        self.logger.info("🚀 Running comprehensive evaluation...")
        
        # Test cases for evaluation
        test_cases = [
            {
                'prompt': 'In the year of our Lord 1665, the Great Plague swept through London.',
                'context': 'The Great Plague of London was a bubonic plague epidemic that occurred in 1665-1666.',
                'expected_facts': ['plague', '1665', 'London', 'bubonic', 'epidemic']
            },
            {
                'prompt': 'The streets were empty, and the bells tolled for the dead.',
                'context': 'During the plague, London streets were deserted and church bells rang for the dead.',
                'expected_facts': ['empty streets', 'bells', 'dead', 'plague']
            },
            {
                'prompt': 'Methinks the city hath never seen such sorrow.',
                'context': 'Historical language from the 17th century.',
                'expected_facts': ['historical language', 'sorrow', 'city']
            },
            {
                'prompt': 'Verily, I say unto you, this is a time of great tribulation.',
                'context': 'Biblical and historical language from the period.',
                'expected_facts': ['verily', 'tribulation', 'historical language']
            },
            {
                'prompt': 'The King\'s men rode through the streets, seeking those who had fled.',
                'context': 'Royal officials during the plague period.',
                'expected_facts': ['King\'s men', 'streets', 'fled', 'royal officials']
            }
        ]
        
        try:
            # Run all evaluations
            self.logger.info("Starting comprehensive evaluation...")
            
            # 1. Groundedness evaluation
            groundedness_results = self.evaluate_groundedness(test_cases)
            self.evaluation_results['groundedness_scores'] = groundedness_results
            
            # 2. Coherence evaluation
            coherence_results = self.evaluate_coherence(test_cases)
            self.evaluation_results['coherence_scores'] = coherence_results
            
            # 3. Fluency evaluation
            fluency_results = self.evaluate_fluency(test_cases)
            self.evaluation_results['fluency_scores'] = fluency_results
            
            # 4. Benchmark evaluation
            benchmark_results = self.run_benchmark_evaluation()
            self.evaluation_results['benchmark_scores'] = benchmark_results
            
            # 5. Generate sample texts
            self._generate_sample_texts()
            
            self.logger.info("✅ Comprehensive evaluation completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Comprehensive evaluation failed: {e}")
            return False
    
    def _generate_sample_texts(self):
        """Generate sample texts for evaluation"""
        sample_prompts = [
            "In the year of our Lord 1665,",
            "The streets of London were",
            "Methinks the city",
            "Verily, I say unto you,",
            "The King's men",
            "In the time of the plague,",
            "The bells of London tolled",
            "Forsooth, this is a time",
            "The poor people of London",
            "In the year 1666, London"
        ]
        
        self.logger.info("📝 Generating sample texts...")
        for prompt in sample_prompts:
            generated = self.generate_text(prompt, max_length=100, num_return_sequences=1)
            if generated:
                self.evaluation_results['generation_samples'].append({
                    'prompt': prompt,
                    'generated': generated[0],
                    'timestamp': datetime.now().isoformat()
                })
    
    def save_results(self):
        """Save evaluation results"""
        self.logger.info("💾 Saving evaluation results...")
        
        try:
            # Save JSON results
            results_file = self.output_dir / "comprehensive_evaluation_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(self.evaluation_results, f, indent=2, ensure_ascii=False)
            
            # Save summary report
            summary_file = self.output_dir / "evaluation_summary.txt"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("London Historical SLM - Comprehensive Evaluation Summary\n")
                f.write("=" * 60 + "\n\n")
                
                # Groundedness results
                if 'groundedness_scores' in self.evaluation_results:
                    f.write("Groundedness Scores:\n")
                    for key, value in self.evaluation_results['groundedness_scores'].items():
                        f.write(f"  {key}: {value:.3f}\n")
                    f.write("\n")
                
                # Coherence results
                if 'coherence_scores' in self.evaluation_results:
                    f.write("Coherence Scores:\n")
                    for key, value in self.evaluation_results['coherence_scores'].items():
                        f.write(f"  {key}: {value:.3f}\n")
                    f.write("\n")
                
                # Fluency results
                if 'fluency_scores' in self.evaluation_results:
                    f.write("Fluency Scores:\n")
                    for key, value in self.evaluation_results['fluency_scores'].items():
                        f.write(f"  {key}: {value:.3f}\n")
                    f.write("\n")
                
                # Benchmark results
                if 'benchmark_scores' in self.evaluation_results:
                    f.write("Benchmark Scores:\n")
                    for key, value in self.evaluation_results['benchmark_scores'].items():
                        f.write(f"  {key}: {value:.3f}\n")
                    f.write("\n")
                
                # Generation samples
                if 'generation_samples' in self.evaluation_results:
                    f.write("Generation Samples:\n")
                    for i, sample in enumerate(self.evaluation_results['generation_samples']):
                        f.write(f"  Sample {i+1}:\n")
                        f.write(f"    Prompt: {sample['prompt']}\n")
                        f.write(f"    Generated: {sample['generated']}\n")
                        f.write("\n")
            
            self.logger.info(f"Results saved to: {self.output_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            return False
    
    def print_summary(self):
        """Print evaluation summary"""
        print("\n" + "="*80)
        print(f"LONDON HISTORICAL SLM - COMPREHENSIVE EVALUATION SUMMARY")
        print("="*80)
        
        # Groundedness results
        if 'groundedness_scores' in self.evaluation_results:
            print(f"🔍 Groundedness Scores:")
            for key, value in self.evaluation_results['groundedness_scores'].items():
                print(f"  {key}: {value:.3f}")
            print()
        
        # Coherence results
        if 'coherence_scores' in self.evaluation_results:
            print(f"🔗 Coherence Scores:")
            for key, value in self.evaluation_results['coherence_scores'].items():
                print(f"  {key}: {value:.3f}")
            print()
        
        # Fluency results
        if 'fluency_scores' in self.evaluation_results:
            print(f"💬 Fluency Scores:")
            for key, value in self.evaluation_results['fluency_scores'].items():
                print(f"  {key}: {value:.3f}")
            print()
        
        # Benchmark results
        if 'benchmark_scores' in self.evaluation_results:
            print(f"📊 Benchmark Scores:")
            for key, value in self.evaluation_results['benchmark_scores'].items():
                print(f"  {key}: {value:.3f}")
            print()
        
        # Generation samples
        if 'generation_samples' in self.evaluation_results:
            print(f"📝 Generation Samples:")
            for i, sample in enumerate(self.evaluation_results['generation_samples'][:5]):  # Show first 5
                print(f"  Sample {i+1}:")
                print(f"    Prompt: {sample['prompt']}")
                print(f"    Generated: {sample['generated'][:100]}{'...' if len(sample['generated']) > 100 else ''}")
                print()
        
        print(f"Results saved to: {self.output_dir}")
        print("="*80)

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Comprehensive Evaluation for London Historical SLM")
    parser.add_argument("--model_dir", type=str, default="09_models/checkpoints",
                       help="Directory containing trained model")
    parser.add_argument("--tokenizer_dir", type=str, 
                       default="09_models/tokenizers/london_historical_tokenizer",
                       help="Directory containing tokenizer")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Directory to save evaluation results")
    parser.add_argument("--openai_api_key", type=str, default=None,
                       help="OpenAI API key for G-Eval (optional)")
    parser.add_argument("--device", type=str, choices=["cpu", "gpu"], default="cpu",
                       help="Device to use for evaluation (default: cpu for safety)")
    
    args = parser.parse_args()
    
    print(f"London Historical SLM - Comprehensive Evaluation")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = HistoricalLondonEvaluator(
        model_dir=args.model_dir,
        tokenizer_dir=args.tokenizer_dir,
        output_dir=args.output_dir,
        openai_api_key=args.openai_api_key,
        device=args.device
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
        
        evaluator.logger.info(f"✅ Model evaluation completed successfully!")
        return True
        
    except Exception as e:
        evaluator.logger.error(f"❌ Evaluation failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
