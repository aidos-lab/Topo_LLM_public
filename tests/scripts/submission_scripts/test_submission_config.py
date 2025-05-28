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


"""Test the submission config."""

import logging
import pprint

import pytest

from topollm.scripts.submission_scripts.submission_config import SubmissionConfig
from topollm.typing.enums import SubmissionMode, Task


@pytest.fixture
def default_submission_config() -> SubmissionConfig:
    """Fixture to create a default SubmissionConfig instance."""
    return SubmissionConfig()


def test_get_command_for_pipeline_task(
    default_submission_config: SubmissionConfig,
    logger_fixture: logging.Logger,
) -> None:
    """Test that the command for the pipeline task is correctly generated."""
    command: list[str] = default_submission_config.get_command(
        task=Task.PIPELINE,
    )
    logger_fixture.info(
        msg=f"command:\n{pprint.pformat(object=command)}",  # noqa: G004 - low overhead
    )

    assert "hydra/launcher=hpc_submission" in command  # noqa: S101 - pytest assertion
    assert "language_model=roberta-base" in command  # noqa: S101 - pytest assertion
    assert "data=iclr_2024_submissions_validation,multiwoz21_validation" in command  # noqa: S101 - pytest assertion


def test_generate_feature_flags_command(
    default_submission_config: SubmissionConfig,
    logger_fixture: logging.Logger,
) -> None:
    """Test that the feature flags are generated correctly."""
    feature_flags_command: list[str] = default_submission_config.generate_feature_flags_command()

    logger_fixture.info(
        msg=f"feature_flags_command:\n{pprint.pformat(object=feature_flags_command)}",  # noqa: G004 - low overhead
    )

    assert "feature_flags.wandb.use_wandb=true" in feature_flags_command  # noqa: S101 - pytest assertion


def test_hydra_launcher_command_local_mode(
    default_submission_config: SubmissionConfig,
    logger_fixture: logging.Logger,
) -> None:
    """Test the hydra launcher command in LOCAL mode."""
    default_submission_config.submission_mode = SubmissionMode.LOCAL
    hydra_command: list[str] = default_submission_config.generate_hydra_launcher_command()

    logger_fixture.info(
        msg=f"hydra_command:\n{pprint.pformat(object=hydra_command)}",  # noqa: G004 - low overhead
    )

    assert hydra_command == ["hydra/launcher=basic"]  # noqa: S101 - pytest assertion


def test_hydra_launcher_command_hpc_mode(
    default_submission_config: SubmissionConfig,
    logger_fixture: logging.Logger,
) -> None:
    """Test the hydra launcher command in HPC mode."""
    default_submission_config.submission_mode = SubmissionMode.HPC_SUBMISSION
    hydra_command: list[str] = default_submission_config.generate_hydra_launcher_command()

    logger_fixture.info(
        msg=f"hydra_command:\n{pprint.pformat(object=hydra_command)}",  # noqa: G004 - low overhead
    )

    assert "hydra/launcher=hpc_submission" in hydra_command  # noqa: S101 - pytest assertion
    assert f"hydra.launcher.memory={default_submission_config.machine_config.memory}" in hydra_command  # noqa: S101 - pytest assertion
    assert f"hydra.launcher.walltime={default_submission_config.machine_config.walltime}" in hydra_command  # noqa: S101 - pytest assertion


def test_generate_local_estimates_command(
    default_submission_config: SubmissionConfig,
    logger_fixture: logging.Logger,
) -> None:
    """Test that the local estimates command is generated correctly."""
    local_estimates_command = default_submission_config.generate_local_estimates_command()

    logger_fixture.info(
        msg=f"local_estimates_command:\n{pprint.pformat(object=local_estimates_command)}",  # noqa: G004 - low overhead
    )

    assert "local_estimates.filtering.deduplication_mode=array_deduplicator" in local_estimates_command  # noqa: S101 - pytest assertion
    assert "local_estimates.pointwise.n_neighbors_mode=absolute_size" in local_estimates_command  # noqa: S101 - pytest assertion
    assert "local_estimates.filtering.num_samples=60000" in local_estimates_command  # noqa: S101 - pytest assertion


def test_finetuning_specific_command(
    default_submission_config: SubmissionConfig,
    logger_fixture: logging.Logger,
) -> None:
    """Test that finetuning-specific commands are correctly generated."""
    finetuning_command: list[str] = default_submission_config.generate_task_specific_command_finetuning()

    logger_fixture.info(
        msg=f"finetuning_command:\n{pprint.pformat(object=finetuning_command)}",  # noqa: G004 - low overhead
    )

    assert "finetuning.fp16=true" in finetuning_command  # noqa: S101 - pytest assertion
    assert "finetuning.lr_scheduler_type=linear" in finetuning_command  # noqa: S101 - pytest assertion
    assert "finetuning.num_train_epochs=5" in finetuning_command  # noqa: S101 - pytest assertion
