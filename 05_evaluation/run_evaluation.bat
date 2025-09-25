@echo off
REM London Historical SLM Evaluation Launcher for Windows
REM Usage: run_evaluation.bat [mode] [model_dir] [tokenizer_dir] [output_dir] [device] [openai_api_key]

echo 🏛️ London Historical SLM Evaluation Launcher
echo ================================================

REM Set default values
set MODE=%1
if "%MODE%"=="" set MODE=quick

set MODEL_DIR=%2
if "%MODEL_DIR%"=="" set MODEL_DIR=09_models/checkpoints

set TOKENIZER_DIR=%3
if "%TOKENIZER_DIR%"=="" set TOKENIZER_DIR=09_models/tokenizers/london_historical_tokenizer

set OUTPUT_DIR=%4
if "%OUTPUT_DIR%"=="" set OUTPUT_DIR=results

set DEVICE=%5
if "%DEVICE%"=="" set DEVICE=cpu

set OPENAI_API_KEY=%6

REM Run the evaluation
python run_evaluation.py --mode %MODE% --model_dir "%MODEL_DIR%" --tokenizer_dir "%TOKENIZER_DIR%" --output_dir "%OUTPUT_DIR%" --device %DEVICE% --openai_api_key "%OPENAI_API_KEY%"

if %ERRORLEVEL% EQU 0 (
    echo ✅ Evaluation completed successfully!
) else (
    echo ❌ Evaluation failed!
    exit /b 1
)
