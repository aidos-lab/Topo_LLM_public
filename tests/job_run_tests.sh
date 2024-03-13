#!/bin/bash

# This script will run all the tests in the tests directory,
# and create an html coverage report in the htmlcov directory.

source ../.venv/bin/activate

keep_test_data=true
# keep_test_data=false

# Add the flag --keep-test-data to keep the test data,
# otherwise the test data will be placed into temporary directories
# which are deleted after the tests are run.
if [ "$keep_test_data" = true ]; then
    python3 -m pytest --keep-test-data --cov-report html --cov=topollm .
else
    python3 -m pytest --cov-report html --cov=topollm .
fi