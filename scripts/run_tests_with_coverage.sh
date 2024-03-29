#!/bin/bash

# This script will run all the tests in the tests directory,
# and create an html coverage report in the htmlcov directory.

source ./.venv/bin/activate

# Add the flag --keep-test-data to keep the test data,
# otherwise the test data will be placed into temporary directories
# which are deleted after the tests are run.

keep_test_data=true
# keep_test_data=false

if [ "$keep_test_data" = true ]; then
    KEEP_TEST_DATA_FLAG="--keep-test-data"
else
    KEEP_TEST_DATA_FLAG=""
fi

python3 -m pytest $KEEP_TEST_DATA_FLAG \
    -m "not slow" \
    tests/ \
    --cov=topollm/ \
    --cov-report=html:tests/temp_files/coverage_report \
    --hypothesis-show-statistics