#!/bin/bash

source ../.venv/bin/activate

python3 -m pytest --cov-report html --cov=topollm .