#!/bin/bash
# Hello London Environment Activation
echo "🏛️ Activating Hello London Environment..."

# Activate virtual environment
source "/home/amit/src/helloLondon/helloLondon/bin/activate"

# Set environment variables
export HELLO_LONDON_ROOT="/home/amit/src/helloLondon"
export HELLO_LONDON_DATA="/home/amit/src/helloLondon/data"
export HELLO_LONDON_MODELS="/home/amit/src/helloLondon/09_models"

echo "✅ Environment activated!"
echo "📁 Project root: $HELLO_LONDON_ROOT"
echo "📊 Data directory: $HELLO_LONDON_DATA"
echo "🤖 Models directory: $HELLO_LONDON_MODELS"
echo ""
echo "Ready to start training your Hello London LLM! 🏛️✨"
