# WandB Setup Guide

## **WandB Configuration for London Historical LLM**

This guide explains how to set up Weights & Biases (WandB) for experiment tracking and monitoring during model training.

## **Prerequisites**

1. **WandB Account**: Sign up at [https://wandb.ai](https://wandb.ai)
2. **API Key**: Get your API key from [https://wandb.ai/authorize](https://wandb.ai/authorize)
3. **WandB Package**: Install with `pip install wandb`

## 🚀 **Quick Setup**

### **1. Install WandB**
```bash
pip install wandb
```

### **2. Set API Key**
```bash
# Option 1: Environment variable (recommended)
export WANDB_API_KEY=your_api_key_here

# Option 2: Login via CLI
wandb login
```

### **3. Verify Setup**
```bash
python -c "import wandb; print('WandB version:', wandb.__version__)"
```

## 🔧 **Configuration Options**

### **Environment Variables**
```bash
# Required
export WANDB_API_KEY=your_api_key_here

# Optional
export WANDB_PROJECT=helloLondon
export WANDB_ENTITY=your_username
export WANDB_MODE=online  # or offline
```

### **Project Settings**
- **Project Name**: `helloLondon`
- **Entity**: Your WandB username
- **Tags**: `london`, `historical`, `llm`, `gpt2`, `1500-1850`

## 📊 **What Gets Tracked**

### **Training Metrics**
- **Loss**: Training and validation loss
- **Learning Rate**: Current learning rate
- **Epoch**: Current epoch
- **Step**: Current training step
- **GPU Usage**: GPU memory and utilization
- **Training Time**: Time per step and epoch

## 📈 **Understanding WandB Dashboard Panels**

Your WandB dashboard shows 5 key panels that track different aspects of training as shown below.

<img width="1827" height="688" alt="train16-5" src="https://github.com/user-attachments/assets/02bda017-3dfa-46c3-9b34-12796b63185d" />

Here's what each panel means and how to interpret them:

### **1. `train/loss` (Top Left Panel)**

The training loss panel is the most critical metric for monitoring your model's learning progress. This graph displays the cross-entropy loss value over training steps, where the y-axis typically ranges from 0 to 10 and the x-axis shows the progression of training iterations. The loss quantifies how well your model is predicting the next token in the sequence - lower values indicate better performance.

At the beginning of training, you'll typically see the loss start around 8-10, which represents essentially random predictions (since the model hasn't learned anything yet). As training progresses, a healthy model will show a steady downward trend, with the loss gradually decreasing as the model learns patterns in the data. By mid-training, you should expect to see values in the 4-6 range, indicating the model is generating somewhat coherent text. For a well-trained model, the loss should eventually plateau around 2-4, representing high-quality text generation capabilities.

The key things to watch for are a consistent downward trend without sudden spikes or plateaus that last too long. If you see the loss increasing over time, extreme fluctuations, or the loss not decreasing after many steps, these are red flags that might indicate issues with your learning rate, data quality, or model architecture. The loss curve should be smooth and show clear evidence of learning progress.

### **2. `train/mfu` (Top Middle Panel)**

The Model FLOPs Utilization (MFU) panel measures how efficiently your GPU hardware is being utilized during training. This metric compares the actual computational throughput of your model against the theoretical maximum performance of your GPU, expressed as a percentage. Higher MFU values indicate that you're getting better value from your hardware investment and faster training times.

In your screenshot, you can see the MFU starts at 0 and then rises to around 9% by step 5. While this might seem low, it's actually quite normal for the early stages of training, especially with smaller models or when using certain optimizations like gradient accumulation. For larger models and more mature training setups, you should expect to see MFU values in the 30-80% range, with 50%+ being considered excellent utilization.

The MFU metric is particularly valuable for identifying performance bottlenecks. If you consistently see low MFU values (under 20%), it might indicate that your training is being limited by data loading speed, CPU processing, memory bandwidth, or other system constraints rather than GPU compute power. A stable, high MFU throughout training indicates that your setup is well-optimized and making efficient use of your hardware resources.

### **3. `train/lr` (Top Right Panel)**

The learning rate panel tracks the evolution of your learning rate schedule throughout training. This is one of the most important hyperparameters in deep learning, as it directly controls how large steps the model takes when updating its weights to minimize the loss function. The learning rate schedule you see in your screenshot shows a classic warmup pattern, starting at 0 and gradually increasing to around 0.00024 by step 5.

This warmup phase is crucial for training stability, especially with large models. Starting with a very low learning rate and gradually increasing it helps prevent the model from making drastic changes to its weights early in training, which can lead to instability or poor convergence. After the warmup period, the learning rate typically remains at its peak value for most of the training, then gradually decreases toward the end to allow for fine-tuning convergence.

The specific pattern you see (linear warmup) is common and effective, but other schedules like cosine decay or step decay can also work well depending on your specific use case. The key is to ensure the learning rate is neither too high (which can cause instability) nor too low (which can lead to extremely slow learning). The smooth, predictable curve in your screenshot indicates a well-designed learning rate schedule.

### **4. `train/iter` (Bottom Left Panel)**

The training iterations panel provides a simple but essential view of training progress by showing the cumulative number of training steps completed over time. This should appear as a steady, linear progression from 0 to your total planned training steps. In your screenshot, you can see it starts at 0, remains there until step 1, then increases linearly to 400 iterations by step 5.

This linear progression is exactly what you want to see, as it confirms that your training is running smoothly without interruptions, crashes, or unexpected pauses. Any deviations from this straight line - such as sudden stops, jumps, or plateaus - would indicate potential issues with your training setup, system resources, or data pipeline.

For context, a typical SLM training run might complete 60,000 iterations, while a full model training might run for 100,000+ iterations depending on your configuration. The steady increase in this panel gives you confidence that your training will complete successfully and helps you estimate how much time remains in your training run.

### **5. `train/dt_ms` (Bottom Right Panel)**

The delta time panel measures the time taken for each individual training step in milliseconds, providing crucial insights into your training performance and potential bottlenecks. In your screenshot, you can see a dramatic pattern that's actually quite normal: an initial spike to over 15,000ms (15 seconds) at step 1, followed by a sharp drop to around 600-700ms for subsequent steps.

This initial spike is completely expected and represents the overhead of model compilation, GPU kernel optimization, and the first-time setup costs that PyTorch incurs. Modern PyTorch with `torch.compile` does significant optimization work on the first few iterations, which explains the high initial time. After this compilation phase, the training time should stabilize at a much lower, consistent value.

The stable values you see (around 623ms in your screenshot) indicate good performance for most hardware setups. These times can vary significantly based on your model size, batch size, GPU type, and other factors. For reference, smaller models might achieve 200-500ms per step, while larger models might take 1-3 seconds per step. The key is consistency - if you see the step time gradually increasing over the course of training, it might indicate memory leaks or other performance degradation issues that need attention.

## 🔍 **Quick Tip: SLM vs Regular Model Identification**

When training starts, you can quickly identify which model you're running by looking at these key indicators in the console output:

### **Regular Model (24 layers, 16 heads, 1024 embeddings)**
```
Model parameters: 332,760,064
num decayed parameter tensors: 98, with 333,758,464 parameters
wandb: Syncing run london-historical-llm-YYYYMMDD-HHMMSS
```

### **SLM Model (12 layers, 12 heads, 768 embeddings)**
```
Model parameters: 108,761,088
num decayed parameter tensors: 50, with 108,761,088 parameters
wandb: Syncing run london-slm-simple-YYYYMMDD-HHMMSS
```

### **Key Differences:**
- **Parameter Count**: Regular model has ~333M parameters vs SLM's ~109M
- **WandB Run Name**: `london-historical-llm` vs `london-slm-simple`
- **Decayed Tensors**: 98 vs 50 parameter tensors
- **Memory Usage**: Regular model uses significantly more GPU memory

## 🎯 **How to Track Training Progress**

### **Early Training (Steps 0-2000)**
- **Loss**: Should drop rapidly from 8-10 to 6-7
- **MFU**: May start low, should increase to 50%+
- **Learning Rate**: Should be in warmup phase (increasing)
- **Time per step**: Should stabilize after initial compilation

### **Mid Training (Steps 2000-20000)**
- **Loss**: Steady decrease from 6-7 to 4-5
- **MFU**: Should be stable at 50-80%
- **Learning Rate**: Should be at peak value
- **Time per step**: Should be consistent

### **Late Training (Steps 20000-60000)**
- **Loss**: Slow decrease from 4-5 to 2-4
- **MFU**: Should remain high
- **Learning Rate**: Should be in decay phase
- **Time per step**: Should remain consistent

### **Red Flags to Watch For**
- **Loss plateauing**: Model might need more data or different learning rate
- **MFU dropping**: Hardware issues or data loading problems
- **Time per step increasing**: Memory leaks or system issues
- **Loss increasing**: Learning rate too high or model instability

### **Model Configuration**
- **Model Name**: GPT-2 Medium
- **Vocabulary Size**: 50,000 tokens
- **Context Length**: 1,024 tokens
- **Batch Size**: Per-device batch size
- **Learning Rate**: 3e-5
- **Epochs**: 5
- **Special Tokens**: 100+ historical tokens

### **Hardware Information**
- **GPU Count**: Number of available GPUs
- **GPU Names**: GPU model names
- **GPU Memory**: Available GPU memory
- **Mixed Precision**: FP16/BF16 status
- **Distributed Training**: Multi-GPU status

## 🎯 **Usage Examples**

### **Single GPU Training**
```bash
# Set API key
export WANDB_API_KEY=your_key_here

# Run training
python 04_training/train_model.py
```

### **Multi-GPU Training**
```bash
# Set API key
export WANDB_API_KEY=your_key_here

# Run multi-GPU training
python 10_scripts/launch_multi_gpu_training.py
```

### **Interactive Launcher**
```bash
# Set API key
export WANDB_API_KEY=your_key_here

# Use interactive launcher
python 10_scripts/launch_london_llm.py
```

## 📈 **Monitoring Training**

### **Real-time Monitoring**
1. **Open WandB Dashboard**: [https://wandb.ai](https://wandb.ai)
2. **Navigate to Project**: `london-historical-llm`
3. **Select Run**: Choose your current training run
4. **View Metrics**: Monitor loss, learning rate, GPU usage

### **Key Metrics to Watch**
- **Training Loss**: Should decrease over time
- **Validation Loss**: Should track training loss
- **Learning Rate**: Should follow cosine schedule
- **GPU Memory**: Should be stable
- **Training Speed**: Steps per second

### **Alerts and Notifications**
- **Loss Plateau**: Set up alerts for loss stagnation
- **GPU Memory**: Monitor for out-of-memory errors
- **Training Time**: Track total training duration

## 🔧 **Advanced Configuration**

### **Custom WandB Settings**
```python
# In train_model.py, you can customize:
wandb.init(
    project="london-historical-llm",
    name="custom-run-name",
    config={
        "custom_param": "value",
        "data_sources": 12,
        "historical_period": "1500-1850"
    },
    tags=["custom", "experiment"],
    notes="Custom experiment notes"
)
```

### **Offline Mode**
```bash
# For offline training
export WANDB_MODE=offline
python 04_training/train_model.py

# Sync later
wandb sync wandb/offline-run-*
```

### **Team Collaboration**
```bash
# Set team entity
export WANDB_ENTITY=your-team-name
python 04_training/train_model.py
```

## 🐛 **Troubleshooting**

### **Common Issues**

1. **API Key Not Found**
   ```
   Error: WandB API key not found
   Solution: Set WANDB_API_KEY environment variable
   ```

2. **Network Issues**
   ```
   Error: Failed to connect to WandB
   Solution: Check internet connection or use offline mode
   ```

3. **Permission Denied**
   ```
   Error: Permission denied
   Solution: Check API key permissions
   ```

4. **Project Not Found**
   ```
   Error: Project not found
   Solution: Create project in WandB dashboard
   ```

### **Debug Mode**
```bash
# Enable debug logging
export WANDB_DEBUG=true
python 04_training/train_model.py
```

## 📊 **Expected Dashboard Views**

### **Training Overview**
- **Loss Curves**: Training and validation loss over time
- **Learning Rate**: Learning rate schedule
- **GPU Usage**: Memory and utilization graphs
- **Training Speed**: Steps per second

### **Model Information**
- **Architecture**: GPT-2 Medium configuration
- **Parameters**: 355M parameters
- **Vocabulary**: 50,000 tokens
- **Context Length**: 1,024 tokens

### **Hardware Metrics**
- **GPU Memory**: Peak and average usage
- **GPU Utilization**: Percentage usage over time
- **Training Time**: Time per step and epoch
- **Data Throughput**: Samples per second

## 🎉 **Benefits of WandB Integration**

### **Experiment Tracking**
- **Compare Runs**: Compare different training configurations
- **Hyperparameter Tuning**: Track parameter effects
- **Model Versioning**: Version control for models
- **Reproducibility**: Reproduce successful runs

### **Collaboration**
- **Team Sharing**: Share results with team members
- **Real-time Monitoring**: Monitor training progress
- **Alerts**: Get notified of issues
- **Documentation**: Document experiments and findings

### **Analysis**
- **Performance Metrics**: Analyze training performance
- **Resource Usage**: Monitor hardware utilization
- **Model Quality**: Track model quality metrics
- **Historical Data**: Compare with previous experiments

## 🚀 **Ready to Track!**

With WandB properly configured, you'll get comprehensive monitoring of your London Historical LLM training, including:

- ✅ **Real-time Metrics**: Loss, learning rate, GPU usage
- ✅ **Model Configuration**: All hyperparameters tracked
- ✅ **Hardware Monitoring**: GPU memory and utilization
- ✅ **Experiment Comparison**: Compare different runs
- ✅ **Team Collaboration**: Share results with others
- ✅ **Reproducibility**: Reproduce successful experiments

**Start training with full monitoring!** 📊✨
