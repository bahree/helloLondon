#!/bin/bash
# Launch script for London Historical LLM Training
# Automatically detects and uses all available GPUs

set -e  # Exit on any error

echo "🎓 London Historical LLM - Training Launcher"
echo "=============================================="

# Function to detect Python command
detect_python() {
    if command -v python3 &> /dev/null; then
        echo "python3"
    elif command -v python &> /dev/null; then
        echo "python"
    else
        echo "python3"  # Default fallback
    fi
}

# Function to check GPU memory
check_gpu_memory() {
    if command -v nvidia-smi &> /dev/null; then
        echo "🔍 GPU Memory Status:"
        nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits | while read line; do
            echo "   $line"
        done
        echo ""
    fi
}

# Detect Python command
PYTHON_CMD=$(detect_python)
echo "🐍 Using Python: $PYTHON_CMD"

# Check if GPUs are available
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "🔍 Detected $GPU_COUNT GPU(s)"
    
    # Show GPU memory status
    check_gpu_memory
    
    if [ $GPU_COUNT -gt 1 ]; then
        echo "🚀 Launching multi-GPU training with torchrun..."
        echo "   Using $GPU_COUNT GPUs for distributed training"
        echo "   Expected speed: ~${GPU_COUNT}x faster than single GPU"
        echo "   Memory: Distributed across all GPUs"
        echo ""
        
        # Set environment variables for multi-GPU
        export TOKENIZERS_PARALLELISM=false
        export OMP_NUM_THREADS=1
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPU_COUNT-1)))
        
        # Launch with torchrun
        torchrun --nproc_per_node=$GPU_COUNT train_model.py
    else
        echo "🚀 Launching single-GPU training..."
        echo "   Using 1 GPU"
        echo "   Memory: Single GPU utilization"
        echo ""
        
        # Set environment variables for single GPU
        export TOKENIZERS_PARALLELISM=false
        export CUDA_VISIBLE_DEVICES=0
        
        # Launch single process
        $PYTHON_CMD train_model.py
    fi
else
    echo "⚠️  NVIDIA drivers not detected. Launching CPU-only training..."
    echo "   Performance: Slower but will work on any machine"
    echo ""
    
    # Set environment variables for CPU
    export TOKENIZERS_PARALLELISM=false
    export CUDA_VISIBLE_DEVICES=""
    
    # Launch single process
    $PYTHON_CMD train_model.py
fi

echo ""
echo "✅ Training completed!"
