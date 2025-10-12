# London Historical SLM - Comprehensive Evaluation Guide

**Complete evaluation framework for your historical language model (1500-1850 London)**

## 📚 **Documentation Navigation**
- **⚡ [EVALUATION_QUICK_REFERENCE.md](EVALUATION_QUICK_REFERENCE.md)** - Quick commands and metrics reference
- **👀 You are here:** Complete technical guide

## **Overview**

This guide provides comprehensive evaluation methods for your London Historical SLM, including both automated metrics and historical-specific assessments. The evaluation framework is designed specifically for the 1500-1850 London period and includes modern LLM evaluation techniques.

## **Quick Start**

### **1. Basic Evaluation (5 minutes)**
```bash
cd 05_evaluation
python run_evaluation.py --mode quick
```

### **2. Full Setup & Evaluation**
```bash
cd 05_evaluation
python run_evaluation.py --mode setup
python run_evaluation.py --mode all
```

### **3. Windows Users**
```cmd
cd 05_evaluation
run_evaluation.bat quick
```

### **4. Device Safety (Important!)**
```bash
# CPU evaluation (safe during training) - DEFAULT
python run_evaluation.py --mode quick --device cpu

# GPU evaluation (only when GPU is free)
python run_evaluation.py --mode quick --device gpu
```

> **💡 Testing Checkpoints**: For checkpoint testing during training, see [EVALUATION_QUICK_REFERENCE.md](EVALUATION_QUICK_REFERENCE.md)

## 📋 **Evaluation Types**

### **1. Quick Evaluation** ⚡
**File**: `quick_eval.py`

**What it does**:
- Historical accuracy testing
- Language quality metrics
- Coherence evaluation using ROUGE scores
- No external API dependencies

**Usage**:
```bash
python quick_eval.py --model_dir 09_models/checkpoints --tokenizer_dir 09_models/tokenizers/london_historical_tokenizer
```

**Metrics**:
- Historical accuracy (keyword/phrase matching)
- Vocabulary diversity
- Historical language pattern detection
- Readability scores
- ROUGE-1, ROUGE-2, ROUGE-L scores

### **2. Comprehensive Evaluation** 🔬
**File**: `comprehensive_evaluator.py`

**What it does**:
- G-Eval methodology for groundedness
- Coherence evaluation with multiple metrics
- Fluency assessment
- Benchmark testing (MMLU subset, HellaSWAG)
- Historical accuracy benchmarks

**Usage**:
```bash
python comprehensive_evaluator.py --model_dir 09_models/checkpoints --tokenizer_dir 09_models/tokenizers/london_historical_tokenizer --openai_api_key YOUR_API_KEY
```

**Requirements**:
- OpenAI API key for G-Eval (optional)
- Additional dependencies (see `evaluation_requirements.txt`)

### **3. Historical Dataset Evaluation** 🏛️
**File**: `historical_evaluation_dataset.py`

**What it does**:
- Creates specialized test cases for 1500-1850 London
- Tests historical accuracy across multiple categories
- Language pattern recognition
- London-specific knowledge

**Usage**:
```bash
python historical_evaluation_dataset.py
```

## 🖥️ **Device Safety & Selection**

### **Device Parameter**
The evaluation framework defaults to **CPU for safety** and includes a clear `--device` parameter to choose between CPU and GPU.

### **Default Behavior (Safe)**
- **CPU by default** - Won't interfere with training
- **Automatic fallback** - If GPU requested but not available, falls back to CPU
- **Clear messaging** - Shows which device is being used

### **When to Use Each Device**

#### **Use CPU When:**
- ✅ Training is running on GPU
- ✅ GPU memory is limited
- ✅ You want to be safe
- ✅ Running evaluation in background
- ✅ Testing during development

#### **Use GPU When:**
- ✅ Training is not running
- ✅ You have sufficient GPU memory
- ✅ You want faster evaluation
- ✅ Running comprehensive evaluation
- ✅ You have multiple GPUs

### **Device Selection Logic**
```python
# The evaluation framework automatically handles device selection:

if device.lower() == "gpu" and torch.cuda.is_available():
    # Use GPU if requested and available
    self.device = torch.device("cuda")
    print("🚀 Using GPU for evaluation")
else:
    # Use CPU (default or fallback)
    self.device = torch.device("cpu")
    if device.lower() == "gpu" and not torch.cuda.is_available():
        print("⚠️  GPU requested but not available, falling back to CPU")
    else:
        print("🖥️  Using CPU for evaluation (safe default)")
```

### **Performance Comparison**

| Device | Speed | Memory Usage | Safety | Best For |
|--------|-------|--------------|--------|----------|
| **CPU** | Slower | Low | High | During training, testing |
| **GPU** | Faster | High | Medium | When GPU is free, final evaluation |

### **Recommended Workflow**

#### **During Training**
```bash
# Safe CPU evaluation every hour
python run_evaluation.py --mode quick --device cpu

# Check specific checkpoint
python run_evaluation.py --mode quick --device cpu --model_dir 09_models/checkpoints/checkpoint-51000
```

#### **Between Training Sessions**
```bash
# Comprehensive GPU evaluation
python run_evaluation.py --mode comprehensive --device gpu

# All evaluations on GPU
python run_evaluation.py --mode all --device gpu
```

#### **Background Monitoring**
```bash
# CPU evaluation in background (safe)
nohup python run_evaluation.py --mode quick --device cpu > evaluation.log 2>&1 &
```

## 📊 **Evaluation Metrics Explained**

### **Historical Accuracy** 🏛️
- **Keyword Matching**: Checks if expected historical terms appear
- **Phrase Matching**: Verifies historical phrases and expressions
- **Contextual Accuracy**: Ensures responses match historical context

**Test Categories**:
- Plague 1665 (Great Plague)
- Fire 1666 (Great Fire of London)
- Royalty (Charles II, Tudor/Stuart/Georgian periods)
- Religion (Church of England, Anglican)
- Social classes (nobility, merchants, artisans, labourers)
- Legal system (Old Bailey, punishments, law enforcement)
- Economic context (currency, trade, guilds)

### **Language Quality** 📝
- **Vocabulary Diversity**: Unique words / total words ratio
- **Historical Patterns**: Detection of period-appropriate language
- **Readability**: Flesch Reading Ease scores
- **Text Structure**: Average words per sentence, sentence count

**Historical Language Patterns**:
- Archaic pronouns (thou, thee, thy, thine, hast, hath, doth, dost, art, wilt, shalt)
- Archaic adverbs (verily, indeed, forsooth, methinks, perchance, albeit)
- Archaic prepositions (betwixt, amongst, amidst, anon, ere, whilst)
- Archaic interjections (prithee, pray thee, I pray you, beseech, ye, yon)

### **Coherence** 🔗
- **ROUGE Scores**: Measures overlap with reference text
- **BERTScore**: Semantic similarity between sentences
- **Vocabulary Overlap**: Word overlap with prompts

**ROUGE Metrics**:
- ROUGE-1: Unigram overlap
- ROUGE-2: Bigram overlap
- ROUGE-L: Longest common subsequence

### **Fluency** 💬
- **Readability Metrics**: Flesch scores, sentence length consistency
- **Repetition Penalty**: Unique words / total words
- **Grammar Quality**: Basic grammatical structure assessment

**Readability Scores**:
- Flesch Reading Ease: 0-100 (higher = easier)
- Flesch-Kincaid Grade Level: US grade level
- Sentence length consistency

## 🏆 **Benchmark Tests**

### **MMLU (Measuring Massive Multitask Language Understanding)**
- **Purpose**: General knowledge assessment
- **Implementation**: Subset of 100 questions
- **Scoring**: Accuracy on multiple-choice questions

### **HellaSWAG**
- **Purpose**: Commonsense reasoning
- **Implementation**: Subset of 100 scenarios
- **Scoring**: Accuracy on scenario completion

### **Historical Benchmark**
- **Purpose**: Historical accuracy specific to 1500-1850 London
- **Implementation**: Custom test cases
- **Scoring**: Keyword/phrase matching accuracy

## 🛠️ **Setup Instructions**

### **1. Automatic Setup (Recommended)**
```bash
# One command sets up everything including evaluation dependencies
python 01_environment/setup_environment.py
```

### **2. Manual Installation (if needed)**
```bash
# All dependencies are in the main requirements.txt
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### **3. Set Environment Variables (Optional)**
```bash
export OPENAI_API_KEY="your_api_key_here"  # For G-Eval
```

## 🚀 **Running Evaluations**

### **Local Machine**
```bash
# Basic evaluation
python run_evaluation.py --mode quick

# With custom paths
python run_evaluation.py --mode quick --model_dir /path/to/model --tokenizer_dir /path/to/tokenizer
```

### **Remote Machine (During Training)**
```bash
# SSH to your training machine
ssh your_user@your_machine
cd ~/src/helloLondon

# Run safe CPU evaluation while training on GPU
python 05_evaluation/run_evaluation.py --mode quick --device cpu

# Monitor training progress with periodic evaluation
python 05_evaluation/run_evaluation.py --mode quick --device cpu --model_dir 09_models/checkpoints/checkpoint-51000
```

### **Comprehensive Evaluation**
```bash
# With OpenAI API
python run_evaluation.py --mode comprehensive --openai_api_key YOUR_KEY

# Without OpenAI API (limited functionality)
python run_evaluation.py --mode comprehensive
```

### **All Evaluations**
```bash
# Run everything
python run_evaluation.py --mode all
```

### **Generate Historical Dataset**
```bash
# Create historical test cases
python run_evaluation.py --mode dataset
```

## 📈 **Interpreting Results**

### **Good Scores** ✅
- **Historical Accuracy**: >0.7 (70% correct)
- **Vocabulary Diversity**: >0.3 (30% unique words)
- **Historical Patterns**: >0.2 (20% contain historical language)
- **ROUGE-L**: >0.3 (30% overlap with reference)
- **Readability**: 30-70 (moderate difficulty)

### **Areas for Improvement** ⚠️
- **Low Historical Accuracy**: Model needs more historical training data
- **Low Vocabulary Diversity**: Model may be repetitive
- **Low Historical Patterns**: Model not using period-appropriate language
- **Low Coherence**: Model may have structural issues

### **Score Interpretation Guide**

| Metric | Excellent | Good | Fair | Poor |
|--------|-----------|------|------|------|
| Historical Accuracy | >0.8 | 0.6-0.8 | 0.4-0.6 | <0.4 |
| Vocabulary Diversity | >0.4 | 0.3-0.4 | 0.2-0.3 | <0.2 |
| Historical Patterns | >0.3 | 0.2-0.3 | 0.1-0.2 | <0.1 |
| ROUGE-L | >0.4 | 0.3-0.4 | 0.2-0.3 | <0.2 |
| Readability | 40-60 | 30-70 | 20-80 | <20 or >80 |

## 🔧 **Customizing Evaluations**

### **Adding New Test Cases**
Edit `historical_evaluation_dataset.py` to add new test cases:

```python
new_test_case = {
    'category': 'your_category',
    'prompt': 'Your test prompt',
    'expected_keywords': ['keyword1', 'keyword2'],
    'expected_phrases': ['phrase1', 'phrase2'],
    'context': 'Historical context',
    'expected_facts': ['fact1', 'fact2']
}
```

### **Modifying Metrics**
Edit the evaluation functions in `comprehensive_evaluator.py` to add new metrics or modify existing ones.

### **Custom Historical Tests**
Create period-specific test cases for:
- Tudor period (1485-1603)
- Stuart period (1603-1714)
- Georgian period (1714-1830)
- Specific events (Great Plague, Great Fire, etc.)

## 🐛 **Troubleshooting**

### **Common Issues**

1. **Model Loading Errors**
   - Check model directory path
   - Ensure model is fully trained
   - Verify tokenizer compatibility

2. **Memory Issues**
   - Reduce batch size in evaluation
   - Use `--device cpu` instead of GPU
   - Reduce max_length parameter
   - Force CPU usage: `CUDA_VISIBLE_DEVICES="" python run_evaluation.py --mode quick`

3. **API Errors**
   - Check OpenAI API key
   - Verify internet connection
   - Check API rate limits

4. **Dependency Issues**
   - Run `pip install -r evaluation_requirements.txt`
   - Check Python version compatibility
   - Update packages if needed

5. **Device Issues**
   - **GPU Memory Issues**: Use `--device cpu` or set `CUDA_VISIBLE_DEVICES=""`
   - **Device Not Available**: Framework automatically falls back to CPU
   - **Check Available Devices**: `python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}')"`

### **Performance Tips**

1. **Faster Evaluation**
   - Use `quick_eval.py` for basic assessment
   - Reduce number of test cases
   - Use smaller max_length values
   - Use `--device gpu` when GPU is available

2. **Better Accuracy**
   - Use `comprehensive_evaluator.py` with OpenAI API
   - Increase number of test cases
   - Use historical-specific datasets

## 📁 **Results Analysis**

### **Understanding Output Files**

1. **JSON Results**: Detailed metrics and scores
2. **Summary Text**: Human-readable summary
3. **Log Files**: Debugging and progress information

### **File Structure**
```
05_evaluation/
├── results/                    # Comprehensive evaluation results
│   ├── comprehensive_evaluation_results.json
│   ├── evaluation_summary.txt
│   └── comprehensive_evaluation.log
├── quick_results/              # Quick evaluation results
│   ├── quick_evaluation_results.json
│   ├── quick_evaluation_summary.txt
│   └── quick_evaluation.log
└── historical_evaluation_dataset.json  # Generated test cases
```

### **Comparing Models**

1. **Baseline Comparison**: Compare with previous model versions
2. **Benchmark Comparison**: Compare with other SLMs
3. **Historical Comparison**: Compare with period-appropriate texts

## 🔄 **Continuous Evaluation**

### **Automated Evaluation Pipeline**
```bash
# Set up automated evaluation
python run_evaluation.py --mode setup

# Run evaluation after each training iteration
python run_evaluation.py --mode quick

# Run comprehensive evaluation weekly
python run_evaluation.py --mode comprehensive
```

### **Monitoring Model Performance**
- Track metrics over time
- Compare with baseline models
- Monitor historical accuracy trends
- Alert on performance degradation

## 📚 **Advanced Usage**

### **Custom Evaluation Scripts**
```python
from comprehensive_evaluator import HistoricalLondonEvaluator

# Initialize evaluator
evaluator = HistoricalLondonEvaluator(
    model_dir="path/to/model",
    tokenizer_dir="path/to/tokenizer"
)

# Load model
evaluator.load_model_and_tokenizer()

# Run specific evaluation
results = evaluator.evaluate_historical_accuracy(test_cases)
```

### **Integration with Training**
```python
# Add evaluation to training loop
if step % eval_steps == 0:
    evaluator = HistoricalLondonEvaluator()
    results = evaluator.run_quick_evaluation()
    log_evaluation_results(results)
```

## 🎯 **Best Practices**

### **Evaluation Strategy**
1. **Regular Evaluation**: Run evaluations after each training iteration
2. **Multiple Metrics**: Don't rely on single metric
3. **Historical Context**: Always consider historical accuracy
4. **Human Review**: Supplement automated metrics with human evaluation
5. **Documentation**: Keep detailed records of evaluation results

### **Model Improvement**
1. **Identify Weaknesses**: Focus on lowest-scoring metrics
2. **Data Augmentation**: Add more historical data for weak areas
3. **Fine-tuning**: Adjust training parameters based on results
4. **Iterative Improvement**: Regular evaluation and adjustment cycle

## 🆘 **Support**

### **Getting Help**
1. **Check Documentation**: Review relevant guides above
2. **Check Logs**: Look in respective folder logs
3. **GitHub Issues**: Create issue on repository
4. **Environment Issues**: Check `environment_config.json`

### **Common Solutions**
- **Import Errors**: Install missing dependencies
- **Memory Issues**: Reduce batch size or use CPU
- **API Errors**: Check API keys and rate limits
- **Model Errors**: Verify model and tokenizer compatibility

## 📊 **Example Results**

### **Sample Output**
```
LONDON HISTORICAL SLM - QUICK EVALUATION SUMMARY
============================================================
🏛️ Historical Accuracy:
  overall_accuracy: 0.750
  correct_predictions: 15
  total_predictions: 20

📝 Language Quality:
  avg_words_per_text: 45.2
  vocabulary_diversity: 0.342
  historical_pattern_ratio: 0.250
  avg_readability: 52.1

🔗 Coherence:
  avg_rouge1: 0.456
  avg_rouge2: 0.234
  avg_rougeL: 0.378
  count: 5

📝 Generation Samples:
  Sample 1:
    Prompt: In 1665, London was struck by
    Generated: the Great Plague, a devastating bubonic plague that swept through the city, claiming thousands of lives and leaving the streets empty and desolate.
    Score: 0.850
```

## 🚀 **Next Steps**

1. **Run Quick Evaluation**: Start with basic assessment
2. **Analyze Results**: Identify strengths and weaknesses
3. **Run Comprehensive Evaluation**: Get detailed metrics
4. **Iterate on Model**: Improve based on results
5. **Set Up Continuous Evaluation**: Monitor model performance over time

---

**Ready to evaluate your historical London SLM!** 🏛️✨
