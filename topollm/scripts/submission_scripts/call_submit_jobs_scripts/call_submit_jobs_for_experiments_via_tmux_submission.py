#!/usr/bin/env python3

# Copyright 2024-2025
# [ANONYMIZED_INSTITUTION],
# [ANONYMIZED_FACULTY],
# [ANONYMIZED_DEPARTMENT]
#
# Authors:
# AUTHOR_1 (author1@example.com)
# AUTHOR_2 (author2@example.com)
#
# Code generation tools and workflows:
# First versions of this code were potentially generated
# with the help of AI writing assistants including
# GitHub Copilot, ChatGPT, Microsoft Copilot, Google Gemini.
# Afterwards, the generated segments were manually reviewed and edited.
#


"""Submit jobs in tmux sessions with logging and resource management."""

import datetime
import pathlib
import pprint
import subprocess
import time
from itertools import product

import click

from topollm.scripts.submission_scripts.submission_config import Template
from topollm.scripts.submission_scripts.types import (
    DataListOption,
    ExperimentSelector,
    ExperimentStage,
    ModelGroupOption,
    RunOnlySelectedConfigsOption,
    RunOption,
)
from topollm.typing.enums import DataSamplingMode, SubmissionMode


@click.command()
@click.option(
    "--run-option",
    type=RunOption,
    # This is a required option, so that you have to actively set
    # `--run-option do_submission` to submit the jobs.
    required=True,
    help="Whether to do the submission or start a dry run.",
)
@click.option(
    "--run-only-selected-configs-option",
    type=RunOnlySelectedConfigsOption,
    # We want to make this a mandatory argument,
    # so that when submitting you consciously decide which configurations to run.
    required=True,
    help="Run only a selected set of configurations.",
)
@click.option(
    "--submission-mode",
    type=SubmissionMode,
    default=SubmissionMode.HPC_SUBMISSION,
    help="Whether to run the job on the HPC or locally.",
)
# Note: To use multiple options in click, you need to repeat the option multiple times:
# `--data-list-options "reddit_only" --data-list-options "multiwoz21_only" --data-list-options "wikitext_only"`
# will result in `data_list_options = ["reddit_only", "multiwoz21_only", "wikitext_only"]`
# https://click.palletsprojects.com/en/stable/options/#multiple-options
@click.option(
    "--data-list-options",
    type=DataListOption,  # The type has to be a string, otherwise the list will be interpreted as a list of characters.
    multiple=True,
    default=[
        DataListOption.MULTIWOZ21_ONLY,
    ],
    help="List of data options to use.",
)
@click.option(
    "--experiment-selector-options",
    type=ExperimentSelector,
    multiple=True,
    default=[
        ExperimentSelector.REGULAR_TOKEN_EMBEDDINGS,
    ],
    help="List of experiment selector options to use.",
)
@click.option(
    "--model-group-options",
    type=ModelGroupOption,
    multiple=True,
    default=[
        ModelGroupOption.ROBERTA_BASE_WITHOUT_MODIFICATIONS,
        # ModelGroupOption.ROBERTA_BASE_FINETUNED_FOR_FEW_EPOCHS_OLD_AND_NEW_DATA_SINGLE_SEED_LAST_CHECKPOINT,
    ],
    help="List of model group options to use.",
)
@click.option(
    "--data-subsampling-sampling-mode",
    type=DataSamplingMode,
    default=DataSamplingMode.RANDOM,
    help="Data subsampling sampling mode to use.",
)
# We do not make the experiment_stage into a list option, because the embedding computation jobs
# need to be run before the additional pipeline runs (they depend on the embeddings).
@click.option(
    "--experiment-stage",
    type=ExperimentStage,
    default=ExperimentStage.COMPUTE_EMBEDDINGS_PLUS_SINGLE_PIPELINE_RUN,
    help="Specify the experiment stage to run.",
)
@click.option(
    "--template-to-use-for-compute-embeddings",
    type=Template,
    default=Template.RTX6000,
    help="Template to use for the compute embeddings job submission.",
)
def submit_jobs_in_separate_tmux_sessions(
    *,
    run_option: RunOption,
    run_only_selected_configs_option: RunOnlySelectedConfigsOption,
    submission_mode: SubmissionMode,
    data_list_options: list[DataListOption],
    experiment_selector_options: list[ExperimentSelector],
    model_group_options: list[ModelGroupOption],
    data_subsampling_sampling_mode: DataSamplingMode,
    experiment_stage: ExperimentStage,
    template_to_use_for_compute_embeddings: Template,
) -> None:
    """Submit jobs in tmux sessions with logging and resource management."""
    # Cast arguments to list to convert the tuple type returned by click to a list.
    data_list_options = list(data_list_options)
    experiment_selector_options = list(experiment_selector_options)
    model_group_options = list(model_group_options)

    log_dir: pathlib.Path = create_log_directory()

    if run_option == RunOption.DRY_RUN:
        print(  # noqa: T201 - we want this script to print"
            70 * "@",
        )
        print(  # noqa: T201 - we want this script to print"
            "@@@ Dry-run mode enabled. No actual submission will take place.",
        )
        print(  # noqa: T201 - we want this script to print"
            70 * "@",
        )

    # Generator combinations to call submissions for
    combinations_to_call = product(
        data_list_options,
        experiment_selector_options,
        model_group_options,
    )

    # Track tmux session names and logs
    session_names: list = []

    for session_counter, (
        data_list_option,
        experiment_selector,
        model_group_option,
    ) in enumerate(
        iterable=combinations_to_call,
    ):
        print(  # noqa: T201 - we want this script to print
            f">>> Current choices:\n\t{data_list_option = }\n\t{experiment_selector = }\n\t{model_group_option= }",
        )

        # Note: We shorten the segments of the session name to avoid a too long file name.
        session_name: str = (
            f"job_{session_counter}_{str(object=data_list_option)[:30]}"
            f"_{str(object=experiment_selector)[:30]}_{str(object=model_group_option)[:30]}"
        )
        # Truncate the session name to 200 characters
        session_name = session_name[:200]

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
            data_list_option=data_list_option,
            data_subsampling_sampling_mode_option=data_subsampling_sampling_mode,
            experiment_selector=experiment_selector,
            experiment_stage=experiment_stage,
            model_group_option=model_group_option,
            run_only_selected_configs_option=run_only_selected_configs_option,
            submission_mode=submission_mode,
            run_option=run_option,
            template_to_use_for_compute_embeddings=template_to_use_for_compute_embeddings,
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

    if run_option == RunOption.DRY_RUN:
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
    data_list_option: DataListOption,
    data_subsampling_sampling_mode_option: DataSamplingMode,
    experiment_selector: ExperimentSelector,
    experiment_stage: ExperimentStage,
    model_group_option: ModelGroupOption,
    run_only_selected_configs_option: RunOnlySelectedConfigsOption,
    run_option: RunOption,
    submission_mode: SubmissionMode,
    template_to_use_for_compute_embeddings: Template,
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
        f"--data-list-option {str(object=data_list_option)} "
        f"--data-subsampling-sampling-mode {str(object=data_subsampling_sampling_mode_option)} "
        f"--experiment-selector {str(object=experiment_selector)} "
        f"--experiment-stage {str(object=experiment_stage)} "
        f"--model-group-option {str(object=model_group_option)} "
        f"--task=pipeline "
        f"--run-option {str(object=run_option)} "
        f"--run-only-selected-configs-option {str(object=run_only_selected_configs_option)} "
        f"--submission-mode {str(object=submission_mode)} "
        f"--template-to-use-for-compute-embeddings {str(object=template_to_use_for_compute_embeddings)} "
        f"2>&1 | tee -a {log_file}; "
        f'echo ">>> All jobs in this tmux session submitted." | tee -a {log_file}; '  # Note: The "..." quotes are necessary for the echo command.
        f'echo "{timeout_message}" | tee -a {log_file}; '  # Note: The "..." quotes are necessary for the echo command.
        f"sleep {session_timeout}; "
        f"tmux kill-session -t {session_name}"
    )

    command_to_run: list[str] = [
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
        f">>> Started session {session_name=} for {data_list_option=} (logs: {log_file=})",
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
    submit_jobs_in_separate_tmux_sessions()
