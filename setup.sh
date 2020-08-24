#!/bin/bash

echo "----- Setup everything you need for Skyhawk in current folder ----"

# install virtual environment package 
echo "Setting up virtual environment..."
apt-get install python3-venv

# create a virtual environment
python3 -m venv env

# activate virtual environment
source env/bin/activate

# install and upgrade pip
echo "Installing and upgrading pip..."
apt install python3-pip
pip install --upgrade pip

# upgrade setuptools
echo "Upgrading setuptools..."
pip install --upgrade setuptools

# install dependencies using pip
echo "Installing dependencies..."
pip install -r requirements.txt

# deactivate virtual environment
deactivate

echo "----- Done! You're ready to have fun with Skyhawk! -----"
