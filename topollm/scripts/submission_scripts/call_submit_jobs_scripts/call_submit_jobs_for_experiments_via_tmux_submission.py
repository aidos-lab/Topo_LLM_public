#!/usr/bin/env python3

# Copyright 2024-2025
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Ruppik (mail@ruppik.net)
# Julius von Rohrscheidt (julius.rohrscheidt@helmholtz-muenchen.de)
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

"""Submit jobs in tmux sessions with logging and resource management."""

import datetime
import pathlib
import pprint
import subprocess
import time
from itertools import product

import click


@click.command()
@click.option(
    "--do_submission",
    is_flag=True,
    help="Skip the dry-run option and run the actual submission.",
)
@click.option(
    "--run_configs_option",
    type=click.Choice(
        choices=[
            "run_all",
            "run_single_random",
            "run_only_first",
        ],
        case_sensitive=False,
    ),
    required=True,
    help="Run configuration option.",
)
def submit_jobs(
    run_configs_option: str,
    *,
    do_submission: bool,
) -> None:
    """Submit jobs in tmux sessions with logging and resource management."""
    # Define job-specific configurations
    data_list_options: list[str] = [
        # "reddit_only",
        "multiwoz21_only",
        # "wikitext_only",
    ]

    # data_subsampling_sampling_mode_option = "random"
    data_subsampling_sampling_mode_option = "take_first"

    experiment_selector_options: list[str] = [
        # "coarse_checkpoint_resolution",
        # "regular_token_embeddings",
        "masked_token_embeddings",
        # "tiny_dropout_variations_coarse_checkpoint_resolution",
    ]

    # We do not make the experiment_stage into a list option, because the embedding computation jobs
    # need to be run before the additional pipeline runs (they depend on the embeddings).
    #
    # experiment_stage = "compute_embeddings_plus_single_pipeline_run"
    experiment_stage = "skip_compute_embeddings_and_multiple_pipeline_runs"

    model_selection_option_list: list[str] = [
        # "--use-roberta-base",
        "--use-finetuned-model",
    ]

    log_dir: pathlib.Path = create_log_directory()

    # Convert the do_submission flag to a dry_run_option.
    # Note:
    # - By default, the script will run in dry-run mode.
    #   To actually submit the jobs, the --do_submission flag must be set.
    if not do_submission:
        print(  # noqa: T201 - we want this script to print"
            70 * "@",
        )
        print(  # noqa: T201 - we want this script to print"
            "@@@ Dry-run mode enabled. No actual submission will take place.",
        )
        print(  # noqa: T201 - we want this script to print"
            70 * "@",
        )
    dry_run_option: str = "" if do_submission else "--dry-run"

    # Generator combinations to call submissions for
    combinations_to_call = product(
        data_list_options,
        experiment_selector_options,
        model_selection_option_list,
    )

    # Track tmux session names and logs
    session_names: list = []

    for session_counter, (
        data_option,
        experiment_selector,
        model_selection_option,
    ) in enumerate(
        iterable=combinations_to_call,
    ):
        session_name: str = (
            f"job_session_{session_counter=}_{data_option=}_{experiment_selector=}_{model_selection_option=}"
        )
        log_file = pathlib.Path(
            log_dir,
            f"output_{session_name}.log",
        )
        session_names.append(
            (
                session_name,
                log_file,
            ),
        )

        # Submit the job in a tmux session
        run_tmux_session(
            session_name=session_name,
            log_file=str(object=log_file),
            data_option=data_option,
            data_subsampling_sampling_mode_option=data_subsampling_sampling_mode_option,
            experiment_selector=experiment_selector,
            experiment_stage=experiment_stage,
            model_selection_option=model_selection_option,
            run_configs_option=run_configs_option,
            dry_run_option=dry_run_option,
        )

    # Automatically attach to the first session
    #
    if session_names:
        print(  # noqa: T201 - we want this script to print
            f">>> Attaching to the first session: {session_names[0][0]}",
        )
        attach_tmux_session(
            session_name=session_names[0][0],
        )

    # Wait for all jobs to finish
    while session_names:
        for session_name, log_file in session_names:
            if not pathlib.Path(log_file).exists():
                continue

            with pathlib.Path(
                log_file,
            ).open() as file:
                log_content: str = file.read()

            if ">>> All jobs in this tmux session submitted." in log_content:
                print(  # noqa: T201 - we want this script to print
                    f">>> {session_name} finished. Checking the next session.",
                )
                session_names.remove(
                    (session_name, log_file),
                )
                continue

            print(  # noqa: T201 - we want this script to print
                f">>> {session_name} still running.",
            )

            time.sleep(1)

    if not do_submission:
        print(  # noqa: T201 - we want this script to print"
            70 * "@",
        )
        print(  # noqa: T201 - we want this script to print"
            "@@@ Dry-run mode enabled. No actual submission took place.",
        )
        print(  # noqa: T201 - we want this script to print"
            70 * "@",
        )

    print(  # noqa: T201 - we want this script to print
        ">>> Call submit jobs script finished.",
    )


def create_log_directory() -> pathlib.Path:
    """Create a directory for log files based on timestamp."""
    log_dir = pathlib.Path(
        "logs",
        datetime.datetime.now(tz=datetime.UTC).strftime(format="%Y%m%d_%H%M%S"),
    )
    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    return log_dir


def run_tmux_session(
    session_name: str,
    log_file: str,
    data_option: str,
    data_subsampling_sampling_mode_option: str,
    experiment_selector: str,
    experiment_stage: str,
    model_selection_option: str,
    run_configs_option: str,
    dry_run_option: str,
    session_timeout: int = 6,
) -> None:
    """Start a tmux session to run the job and log the output."""
    timeout_message = (
        f">>> Session {session_name} will remain open for {session_timeout} seconds after job completion."
        if session_timeout > 0
        else ">>> Session will terminate immediately after job completion."
    )

    command_for_tmux_session: str = (
        f"poetry run submit_jobs "
        f"--data-list-option {data_option} "
        f"--data-subsampling-sampling-mode {data_subsampling_sampling_mode_option} "
        f"--experiment-selector {experiment_selector} "
        f"--experiment-stage {experiment_stage} "
        f"{model_selection_option} "
        f"--task=pipeline "
        f"{dry_run_option} "
        f"--run-only-selected-configs-option {run_configs_option} "
        f"2>&1 | tee -a {log_file}; "
        f'echo ">>> All jobs in this tmux session submitted." | tee -a {log_file}; '  # Note: The "..." quotes are necessary for the echo command.
        f'echo "{timeout_message}" | tee -a {log_file}; '  # Note: The "..." quotes are necessary for the echo command.
        f"sleep {session_timeout}; "
        f"tmux kill-session -t {session_name}"
    )

    command_to_run = [
        "tmux",
        "new-session",
        "-d",
        "-s",
        session_name,
        f"bash -c '{command_for_tmux_session}'",
    ]

    print(  # noqa: T201 - we want this script to print
        f"command_for_tmux_session:\n{command_for_tmux_session}",
    )
    print(  # noqa: T201 - we want this script to print
        f"command_to_run:\n{pprint.pformat(command_to_run)}",
    )

    # Start tmux session
    subprocess.run(
        args=command_to_run,
        check=False,
    )
    print(  # noqa: T201 - we want this script to print
        f">>> Started session {session_name=} for {data_option=} (logs: {log_file=})",
    )


def attach_tmux_session(
    session_name: str,
) -> None:
    """Attach to a tmux session."""
    subprocess.run(
        args=[
            "tmux",
            "attach-session",
            "-t",
            session_name,
        ],
        check=True,
    )


if __name__ == "__main__":
    submit_jobs()
