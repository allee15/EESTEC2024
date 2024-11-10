#!/bin/bash

# Activate the virtual environment
source /usr/src/app/venv/bin/activate

cd /usr/src/app/source/source_files_balmainparis
python model_predict.py ../../InputData/test

# Deactivate the virtual environment after installation
deactivate