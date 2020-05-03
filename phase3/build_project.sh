#!/bin/bash
virtualenv --python=python3 env
bash -c "source env/bin/activate; pip install -r requirements.txt;echo Starting Project; python index.py"