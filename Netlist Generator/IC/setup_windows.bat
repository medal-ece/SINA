@echo off
echo Creating virtual environment...
py -m venv env_netlist

echo Installing dependencies...
env_netlist\Scripts\pip install -r requirements.txt

if not "%1"=="" (
    echo Creating .env file with API key...
    echo OPENAI_API_KEY=%1 > .env
    echo .env file created successfully!
)

echo.
echo Setup complete! Activating environment...
cmd /k env_netlist\Scripts\activate
