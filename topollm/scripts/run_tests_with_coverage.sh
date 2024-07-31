#!/bin/bash

# This script will run tests in the tests directory, and create an html coverage report in the htmlcov directory.
# Add the flag --keep-test-data to keep the test data,
# otherwise the test data will be placed into temporary directories
# which are deleted after the tests are run.
# Add the flag --run-slow-tests to include slow tests.
# Add the flag --capture-output to capture the output of the print statements.

keep_test_data=false
run_slow_tests=false
capture_output=false

# Function to display usage
usage() {
    echo "Usage: $0 [--keep-test-data] [--run-slow-tests] [--capture-output]"
    exit 1
}

# Parse command line arguments using a while loop and case statement
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --keep-test-data)
            keep_test_data=true
            shift
            ;;
        --run-slow-tests)
            run_slow_tests=true
            shift
            ;;
        --capture-output)
            capture_output=true
            shift
            ;;
        -*|--*)
            echo "Unknown option $1"
            usage
            ;;
        *)
            break
            ;;
    esac
done

if [ "$keep_test_data" = true ]; then
    KEEP_TEST_DATA_FLAG="--keep-test-data"
else
    KEEP_TEST_DATA_FLAG=""
fi

# Use the following array to select specific test cases to run.
# Initialize the array:
SELECTED_TEST_CASES=()

if [ "$run_slow_tests" = true ]; then
    SELECTED_TEST_CASES+=()
else
    SELECTED_TEST_CASES+=(
        -m "not slow"
    )
fi

if [ "$capture_output" = true ]; then
    ADDITIONAL_PYTEST_OPTIONS="--capture=no"
else
    ADDITIONAL_PYTEST_OPTIONS=""
fi

export WANDB_MODE=disabled
# export WANDB_DISABLED=true # TODO: Uncommenting this leads to the following error:
# `FAILED tests/model_finetuning/test_do_finetuning_process.py::test_do_finetuning_process[language_model_config0-standard] - RuntimeError: WandbCallback requires wandb to be installed. Run `pip install wandb`.`

poetry run python3 -m pytest \
    $KEEP_TEST_DATA_FLAG \
    "${SELECTED_TEST_CASES[@]}" \
    ${TOPO_LLM_REPOSITORY_BASE_PATH}/tests/ \
    --cov=topollm/ \
    --cov-report=html:tests/temp_files/coverage_report \
    --hypothesis-show-statistics \
    $ADDITIONAL_PYTEST_OPTIONS
