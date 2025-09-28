#!/bin/bash
# Hello London Environment Activation
echo "🏛️ Activating Hello London Environment..."

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate virtual environment
source "$SCRIPT_DIR/helloLondon/bin/activate"

# Set environment variables
export HELLO_LONDON_ROOT="$SCRIPT_DIR"
export HELLO_LONDON_DATA="$SCRIPT_DIR/data"
export HELLO_LONDON_MODELS="$SCRIPT_DIR/09_models"

echo "✅ Environment activated!"
echo "📁 Project root: $HELLO_LONDON_ROOT"
echo "📊 Data directory: $HELLO_LONDON_DATA"
echo "🤖 Models directory: $HELLO_LONDON_MODELS"
echo ""
echo "Ready to start training your Hello London LLM! 🏛️✨"
