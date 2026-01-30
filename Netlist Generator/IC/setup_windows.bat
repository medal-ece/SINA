@echo off
echo Creating virtual environment...
py -m venv env_netlist

echo Checking for NVIDIA GPU...
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo NVIDIA GPU detected. Installing PyTorch with CUDA support...
    env_netlist\Scripts\pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
) else (
    echo No NVIDIA GPU detected. Installing PyTorch for CPU...
    env_netlist\Scripts\pip install torch torchvision torchaudio
)
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to install PyTorch. Please check the output above.
    pause
    exit /b 1
)

echo Installing remaining dependencies...
env_netlist\Scripts\pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to install dependencies. Please check the output above.
    pause
    exit /b 1
)

if not "%1"=="" (
    echo Creating .env file with API key...
    echo OPENAI_API_KEY=%1 > .env
    echo .env file created successfully!
)

echo.
echo Setup complete! Activating environment...
cmd /k env_netlist\Scripts\activate
