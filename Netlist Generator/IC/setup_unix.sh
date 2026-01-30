#!/bin/bash
echo "Creating virtual environment..."
python3 -m venv env_netlist

echo "Checking for NVIDIA GPU..."
if nvidia-smi > /dev/null 2>&1; then
    echo "NVIDIA GPU detected. Installing PyTorch with CUDA support..."
    env_netlist/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
else
    echo "No NVIDIA GPU detected. Installing PyTorch for CPU..."
    env_netlist/bin/pip install torch torchvision torchaudio
fi
if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Failed to install PyTorch. Please check the output above."
    exit 1
fi

echo "Installing remaining dependencies..."
env_netlist/bin/pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Failed to install dependencies. Please check the output above."
    exit 1
fi

if [ -n "$1" ]; then
    echo "Creating .env file with API key..."
    echo "OPENAI_API_KEY=$1" > .env
    echo ".env file created successfully!"
fi

echo ""
echo "Setup complete! Activating environment..."
source env_netlist/bin/activate
