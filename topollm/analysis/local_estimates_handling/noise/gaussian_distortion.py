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
