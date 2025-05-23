#!/usr/bin/env python3

# Copyright 2024
# [ANONYMIZED_INSTITUTION],
# [ANONYMIZED_FACULTY],
# [ANONYMIZED_DEPARTMENT]
#
# Authors:
# AUTHOR_1 (author1@example.com)
#
# Code generation tools and workflows:
# First versions of this code were potentially generated
# with the help of AI writing assistants including
# GitHub Copilot, ChatGPT, Microsoft Copilot, Google Gemini.
# Afterwards, the generated segments were manually reviewed and edited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run tests in the tests directory, and create an html coverage report in the htmlcov directory."""

import argparse
import os
import subprocess
import sys

from topollm.config_classes.constants import TOPO_LLM_REPOSITORY_BASE_PATH


def parse_arguments() -> argparse.Namespace:
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run tests and create an HTML coverage report.",
    )
    parser.add_argument(
        "--keep-test-data",
        action="store_true",
        help="Keep the test data. If false, the test data will be placed into temporary directories "
        "which are deleted after the tests are run.",
    )
    parser.add_argument(
        "--run-slow-tests",
        action="store_true",
        help="Include slow tests.",
    )
    parser.add_argument(
        "--capture-output",
        action="store_true",
        help="Capture the output of the print statements.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the tests."""
    args = parse_arguments()

    keep_test_data_flag = "--keep-test-data" if args.keep_test_data else ""
    selected_test_cases = [] if args.run_slow_tests else ["-m", "not slow"]
    additional_pytest_options = "--capture=no" if args.capture_output else ""

    os.environ["WANDB_MODE"] = "disabled"
    # Warning: Using `export WANDB_DISABLED=true` leads to the following error:
    # `FAILED tests/model_finetuning/test_do_finetuning_process.py::test_do_finetuning_process
    #   [language_model_config0-standard] - RuntimeError:
    #   WandbCallback requires wandb to be installed. Run `pip install wandb`.`

    # Construct the command
    command: list[str] = [
        "uv",
        "run",
        "python3",
        "-m",
        "pytest",
        keep_test_data_flag,
        *selected_test_cases,
        f"{TOPO_LLM_REPOSITORY_BASE_PATH}/tests/",
        "--cov=topollm/",
        "--cov-report=html:tests/temp_files/coverage_report",
        "--hypothesis-show-statistics",
        additional_pytest_options,
    ]

    # Remove empty strings from the command list
    command = [arg for arg in command if arg]

    # Run the command
    result = subprocess.run(  # noqa: S603 - we trust the command
        command,
        text=True,
        check=True,
    )

    if result.returncode != 0:
        print(  # noqa: T201 - script should print to stdout
            "Tests failed.",
            file=sys.stderr,
        )

    if args.capture_output:
        print(  # noqa: T201 - script should print to stdout
            result,
        )

    sys.exit(
        result.returncode,
    )


if __name__ == "__main__":
    main()
