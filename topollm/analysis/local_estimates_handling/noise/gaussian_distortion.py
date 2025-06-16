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


from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Generator


def add_gaussian_distortion(
    original_array: np.ndarray,
    distortion_param: float,
    seed: int = 0,
) -> np.ndarray:
    """Distort and array with Gaussian noise.

    Args:
        original_array:
            The original NumPy array.
        distortion_param:
            Standard deviation of the Gaussian noise.
        seed:
            Random seed for reproducibility.

    Returns:
        np.ndarray: The distorted array.

    """
    if distortion_param < 0:
        msg = "Distortion parameter must be non-negative."
        raise ValueError(msg)

    rng: Generator[
        int,
        None,
        None,
    ] = np.random.default_rng(
        seed=seed,
    )  # type: ignore - typing problem with default_rng

    # Generate Gaussian noise
    noise: np.ndarray = rng.normal(  # type: ignore - typing problem with numpy typing
        0,
        distortion_param,
        original_array.shape,
    )
    distorted_array: np.ndarray = original_array + noise

    return distorted_array
