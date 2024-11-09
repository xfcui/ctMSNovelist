#!/bin/bash

# Environment name (you can change "myenv" to your preferred name)
ENV_NAME="myenv"

# Create a new Conda environment with Python 3.7
conda create -y -n $ENV_NAME python=3.7

# Activate the environment
source activate $ENV_NAME

# Install packages from requirements.txt
conda install --yes --file requirements.txt

# Alternatively, if there are packages in requirements.txt that need pip
# pip install -r requirements.txt

echo "Environment $ENV_NAME created successfully with Python 3.7 and packages from requirements.txt."
