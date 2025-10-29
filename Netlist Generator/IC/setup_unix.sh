#!/bin/bash
echo "Creating virtual environment..."
python3 -m venv env_netlist

echo "Installing dependencies..."
env_netlist/bin/pip install -r requirements.txt

if [ -n "$1" ]; then
    echo "Creating .env file with API key..."
    echo "OPENAI_API_KEY=$1" > .env
    echo ".env file created successfully!"
fi

echo ""
echo "Setup complete! Activating environment..."
source env_netlist/bin/activate
