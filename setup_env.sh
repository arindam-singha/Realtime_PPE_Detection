#!/bin/bash
# Script to install uv and set up Python environment

# Exit on error
set -e

# Check for Python installation
if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed. Please install Python3 first."
    exit 1
fi

# Install uv (universal virtualenv manager)
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    pip install uv || pip3 install uv
else
    echo "uv is already installed."
fi

# Create a virtual environment using uv
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment with uv..."
    uv venv .venv
else
    echo ".venv already exists."
fi

# Activate the virtual environment
source .venv/bin/activate


    # Install dependencies from pyproject.toml if it exists
    if [ -f "pyproject.toml" ]; then
        echo "Installing dependencies from pyproject.toml..."
        uv pip install . || pip install .
    else
        echo "pyproject.toml not found, skipping."
    fi

    # Install dependencies from requirements.txt if it exists
    if [ -f "requirements.txt" ]; then
        echo "Installing dependencies from requirements.txt..."
        uv pip install -r requirements.txt || pip install -r requirements.txt
    else
        echo "requirements.txt not found, skipping."
    fi

    echo "Environment setup complete. To activate later, run: source .venv/bin/activate"
