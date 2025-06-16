# Copyright 2024
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


"""Test the compute_last_save_step function."""

import logging

from topollm.model_finetuning.compute_last_save_step import compute_last_save_step


def test_compute_last_save_step_typical_case(
    logger_fixture: logging.Logger,
) -> None:
    """Test with typical parameters."""
    result = compute_last_save_step(
        total_samples=10_000,
        batch_size=16,
        gradient_accumulation_steps=2,
        num_epochs=5,
        save_steps=400,
    )

    logger_fixture.info(
        f"{result = }",  # noqa: G004 - low overhead
    )

    assert result == 1200  # noqa: S101, PLR2004 - pytest assert, constant value for test


def test_compute_last_save_step_exact_multiple(
    logger_fixture: logging.Logger,
) -> None:
    """Test where total training steps are an exact multiple of save steps."""
    result = compute_last_save_step(
        total_samples=9_600,
        batch_size=16,
        gradient_accumulation_steps=2,
        num_epochs=5,
        save_steps=400,
    )

    logger_fixture.info(
        f"{result = }",  # noqa: G004 - low overhead
    )

    # 9600 // (16 * 2) * 5 = 1200 steps
    assert result == 1200  # noqa: S101, PLR2004 - pytest assert, constant value for test


def test_compute_last_save_step_no_checkpoint(
    logger_fixture: logging.Logger,
) -> None:
    """Test where no checkpoints would be saved."""
    result = compute_last_save_step(
        total_samples=1_000,
        batch_size=32,
        gradient_accumulation_steps=1,
        num_epochs=2,
        save_steps=1_000,
    )

    logger_fixture.info(
        f"{result = }",  # noqa: G004 - low overhead
    )

    # Only 62 steps total, no checkpoints
    assert result == 0  # noqa: S101 - pytest assert, constant value for test


def test_compute_last_save_step_large_save_steps(
    logger_fixture: logging.Logger,
) -> None:
    """Test with a large save step value."""
    result = compute_last_save_step(
        total_samples=10_000,
        batch_size=16,
        gradient_accumulation_steps=2,
        num_epochs=8,
        save_steps=2_000,
    )

    logger_fixture.info(
        f"{result = }",  # noqa: G004 - low overhead
    )

    # Last checkpoint at 2000 steps
    assert result == 2_000  # noqa: S101, PLR2004 - pytest assert, constant value for test


def test_compute_last_save_step_small_batch_size(
    logger_fixture: logging.Logger,
) -> None:
    """Test with a small batch size."""
    result = compute_last_save_step(
        total_samples=10_000,
        batch_size=4,
        gradient_accumulation_steps=1,
        num_epochs=5,
        save_steps=100,
    )

    logger_fixture.info(
        f"{result = }",  # noqa: G004 - low overhead
    )

    # 10000 // 4 * 5 = 12500 steps, last save step 12500
    assert result == 12_500  # noqa: S101, PLR2004 - pytest assert, constant value for test


def test_compute_last_save_step_edge_case(
    logger_fixture: logging.Logger,
) -> None:
    """Test with edge case parameters."""
    result = compute_last_save_step(
        total_samples=1,
        batch_size=1,
        gradient_accumulation_steps=1,
        num_epochs=1,
        save_steps=1,
    )

    logger_fixture.info(
        f"{result = }",  # noqa: G004 - low overhead
    )

    # 1 step total, last save step at 1
    assert result == 1  # noqa: S101 - pytest assert, constant value for test
