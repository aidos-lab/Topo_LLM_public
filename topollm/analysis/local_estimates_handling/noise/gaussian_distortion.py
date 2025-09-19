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
