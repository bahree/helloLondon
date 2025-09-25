@echo off
REM Hello London Environment Activation
echo Activating Hello London Environment...

REM Activate virtual environment
call "helloLondon\Scripts\activate.bat"

REM Set environment variables
set HELLO_LONDON_ROOT=%CD%
set HELLO_LONDON_DATA=%CD%\data\london_historical
set HELLO_LONDON_MODELS=%CD%\09_models

echo Environment activated successfully!
echo Project root: %HELLO_LONDON_ROOT%
echo Data directory: %HELLO_LONDON_DATA%
echo Models directory: %HELLO_LONDON_MODELS%
echo.
echo Ready to start training your Hello London LLM!
