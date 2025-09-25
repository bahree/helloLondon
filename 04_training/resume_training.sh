#!/bin/bash
# Resume Training Script for London Historical LLM (SLM)

set -e  # Exit on any error

echo "🔄 London Historical LLM - Resume Training (SLM)"
echo "==============================================="

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

# Get paths from config.py
CKPT_DIR=$($PYTHON_CMD -c "from config import config; print(config.checkpoints_dir / 'slm')")
DATA_DIR=$($PYTHON_CMD -c "from config import config; print(config.london_historical_data)")
TOKENIZER_DIR=$($PYTHON_CMD -c "from config import config; print(config.london_tokenizer_dir)")

echo "📁 Checkpoint directory: $CKPT_DIR"
echo "📁 Data directory: $DATA_DIR"
echo "📁 Tokenizer directory: $TOKENIZER_DIR"

# Check if GPUs are available
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "🔍 Detected $GPU_COUNT GPU(s)"
    
    # Show GPU memory status
    check_gpu_memory
else
    GPU_COUNT=0
    echo "⚠️ No GPUs detected, using CPU"
fi

# Handle checkpoint selection
if [ $# -eq 0 ]; then
    echo "ℹ️  No checkpoint specified, using latest in $CKPT_DIR"
    LATEST=$(ls -1 "$CKPT_DIR"/checkpoint-*.pt 2>/dev/null | sort -V | tail -1)
    if [ -z "$LATEST" ]; then
        echo "❌ No checkpoints found in $CKPT_DIR"
        exit 1
    fi
    CHECKPOINT=$LATEST
else
    CHECKPOINT="$CKPT_DIR/$1"
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Checkpoint file not found: $CHECKPOINT"
    exit 1
fi

echo "✅ Using checkpoint: $(basename "$CHECKPOINT")"

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

# Launch training based on GPU count
if [ $GPU_COUNT -gt 1 ]; then
    echo "🚀 Resuming SLM training with torchrun (multi-GPU)..."
    echo "   Using $GPU_COUNT GPUs for distributed training"
    torchrun --nproc_per_node=$GPU_COUNT train_model_slm.py \
        --data_dir "$DATA_DIR" \
        --tokenizer_dir "$TOKENIZER_DIR" \
        --output_dir "$CKPT_DIR" \
        --resume_from_checkpoint "$CHECKPOINT"
elif [ $GPU_COUNT -eq 1 ]; then
    echo "🚀 Resuming SLM training (single-GPU)..."
    export CUDA_VISIBLE_DEVICES=0
    $PYTHON_CMD train_model_slm.py \
        --data_dir "$DATA_DIR" \
        --tokenizer_dir "$TOKENIZER_DIR" \
        --output_dir "$CKPT_DIR" \
        --resume_from_checkpoint "$CHECKPOINT"
else
    echo "🚀 Resuming SLM training (CPU-only)..."
    export CUDA_VISIBLE_DEVICES=""
    $PYTHON_CMD train_model_slm.py \
        --data_dir "$DATA_DIR" \
        --tokenizer_dir "$TOKENIZER_DIR" \
        --output_dir "$CKPT_DIR" \
        --resume_from_checkpoint "$CHECKPOINT"
fi

echo ""
echo "✅ SLM training resumed successfully!"
