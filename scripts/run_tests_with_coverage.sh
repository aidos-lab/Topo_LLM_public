#!/bin/bash

python3 -m pytest \
    -m "not slow" \
    tests/ \
    --cov=topollm/ \
    --cov-report=html:tests/temp_files/coverage_report