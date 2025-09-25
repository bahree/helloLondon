# 📊 London Historical LLM - Evaluation Quick Reference

**Quick reference for evaluating your historical language model (1500-1850 London)**

> **📖 Complete Guide**: For detailed implementation, see [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md)

## 🚀 **Quick Commands**

### **Test Checkpoints During Training**
```bash
# Test regular model checkpoint (replace XXXX with step number)
cd 04_training
python test_checkpoint.py --checkpoint_path 09_models/checkpoints/checkpoint-XXXX.pt

# Test SLM checkpoint (replace XXXX with step number)
python test_checkpoint_slm.py --checkpoint_path 09_models/checkpoints/slm/checkpoint-XXXX.pt

# Quick test - auto-detects latest checkpoint
python test_checkpoint.py        # Regular model
python test_checkpoint_slm.py    # SLM model
```

### **Basic Evaluation (Start Here)**
```bash
cd 05_evaluation
python run_evaluation.py --mode quick
```

### **Device Safety (Important!)**
```bash
# CPU evaluation (safe during training) - DEFAULT
python run_evaluation.py --mode quick --device cpu

# GPU evaluation (only when GPU is free)
python run_evaluation.py --mode quick --device gpu
```

### **Full Evaluation Suite**
```bash
cd 05_evaluation
python run_evaluation.py --mode all
```

### **Windows Users**
```cmd
cd 05_evaluation
run_evaluation.bat quick
```

## 📋 **Evaluation Types**

| Type | Command | Time | Dependencies | Best For |
|------|---------|------|--------------|----------|
| **Checkpoint Testing** | `python test_checkpoint.py` | 1-2 min | None | During training validation |
| **Quick** | `--mode quick` | 2-5 min | Basic | Daily testing |
| **Comprehensive** | `--mode comprehensive` | 10-15 min | OpenAI API | Weekly assessment |
| **Dataset** | `--mode dataset` | 1 min | None | Generate test cases |
| **All** | `--mode all` | 15-20 min | OpenAI API | Complete evaluation |

## 📊 **Key Metrics**

### **Historical Accuracy** 🏛️
- **Target**: >70% correct
- **Tests**: Plague 1665, Fire 1666, Royalty, Religion, Social classes
- **Method**: Keyword/phrase matching

### **Language Quality** 📝
- **Vocabulary Diversity**: >30% unique words
- **Historical Patterns**: >20% contain period language
- **Readability**: 30-70 (moderate difficulty)

### **Coherence** 🔗
- **ROUGE-L**: >30% overlap with reference
- **BERTScore**: Semantic similarity
- **Vocabulary Overlap**: Word overlap with prompts

### **Fluency** 💬
- **Flesch Reading Ease**: 30-70
- **Sentence Consistency**: Low variance in length
- **Repetition Penalty**: High unique word ratio

## 🎯 **Score Interpretation**

| Metric | Excellent | Good | Fair | Poor |
|--------|-----------|------|------|------|
| Historical Accuracy | >80% | 60-80% | 40-60% | <40% |
| Vocabulary Diversity | >40% | 30-40% | 20-30% | <20% |
| Historical Patterns | >30% | 20-30% | 10-20% | <10% |
| ROUGE-L | >40% | 30-40% | 20-30% | <20% |
| Readability | 40-60 | 30-70 | 20-80 | <20 or >80 |

## 🏛️ **Historical Test Categories**

1. **Plague 1665** - Great Plague knowledge
2. **Fire 1666** - Great Fire of London
3. **Royalty** - Charles II, Tudor/Stuart/Georgian periods
4. **Religion** - Church of England, Anglican
5. **Social Classes** - Nobility, merchants, artisans, labourers
6. **Legal System** - Old Bailey, punishments, law enforcement
7. **Economic Context** - Currency, trade, guilds
8. **Language Patterns** - Archaic pronouns, adverbs, prepositions

## 🔧 **Troubleshooting**

### **Common Issues**
- **Model Loading**: Check paths in `--model_dir` and `--tokenizer_dir`
- **Memory Issues**: Use `--device cpu` or reduce batch size
- **API Errors**: Check OpenAI API key for comprehensive evaluation
- **Dependencies**: Run `python setup_evaluation.py`
- **GPU Conflicts**: Use `--device cpu` during training to avoid interference

### **Performance Tips**
- **Faster**: Use `--mode quick` for basic assessment
- **Better**: Use `--mode comprehensive` with OpenAI API
- **Custom**: Edit `historical_evaluation_dataset.py` for new test cases
- **Safe**: Use `--device cpu` during training (won't interfere with GPU training)
- **Fast**: Use `--device gpu` when GPU is free (faster evaluation)

## 📁 **Output Files**

### **Quick Evaluation**
```
quick_results/
├── quick_evaluation_results.json
├── quick_evaluation_summary.txt
└── quick_evaluation.log
```

### **Comprehensive Evaluation**
```
results/
├── comprehensive_evaluation_results.json
├── evaluation_summary.txt
└── comprehensive_evaluation.log
```

### **Historical Dataset**
```
historical_evaluation_dataset.json
```

## 📈 **Continuous Evaluation**

### **Daily**
```bash
python run_evaluation.py --mode quick
```

### **Weekly**
```bash
python run_evaluation.py --mode comprehensive
```

### **Monthly**
```bash
python run_evaluation.py --mode all
```

## 🎯 **Model Improvement**

### **Low Historical Accuracy**
- Add more historical training data
- Focus on specific time periods
- Include more primary sources

### **Low Vocabulary Diversity**
- Increase training data variety
- Adjust repetition penalty
- Check tokenizer vocabulary

### **Low Historical Patterns**
- Add period-specific language examples
- Include more historical literature
- Focus on language pattern training

### **Low Coherence**
- Check model architecture
- Adjust training parameters
- Verify data quality

## 🎯 **What You Get**

### **Complete Evaluation Coverage**
- ✅ Historical accuracy for 1500-1850 London
- ✅ Modern LLM evaluation metrics
- ✅ Period-specific language patterns
- ✅ London geography and context
- ✅ Social, legal, and economic knowledge
- ✅ Easy-to-use launcher scripts

### **Professional-Grade Assessment**
- ✅ G-Eval methodology for groundedness
- ✅ MMLU and HellaSWAG benchmarks
- ✅ ROUGE and BERTScore metrics
- ✅ Historical-specific test cases
- ✅ Continuous evaluation capabilities

## 📚 **Documentation**

- **Complete Guide**: [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md)
- **Inference Quick Start**: [INFERENCE_QUICK_START.md](INFERENCE_QUICK_START.md)
- **Training Guide**: [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
- **Inference Setup**: [INFERENCE_SETUP_GUIDE.md](INFERENCE_SETUP_GUIDE.md)

## 🆘 **Support**

1. **Check Logs**: Look in evaluation folder logs
2. **GitHub Issues**: Create issue on repository
3. **Documentation**: Review relevant guides
4. **Environment**: Check `environment_config.json`

---

**Ready to evaluate your historical London LLM!** 🏛️✨
