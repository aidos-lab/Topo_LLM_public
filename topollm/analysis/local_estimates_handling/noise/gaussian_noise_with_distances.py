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
from scipy.spatial import KDTree
from scipy.spatial.distance import directed_hausdorff

if TYPE_CHECKING:
    from collections.abc import Generator


def compute_exact_hausdorff(
    array_1: np.ndarray,
    array_2: np.ndarray,
) -> float:
    """Compute the exact Hausdorff distance.

    Note that this might be expensive, depending on the number of points.
    """
    # Compute directed Hausdorff distances
    h1 = directed_hausdorff(
        array_1,
        array_2,
    )[0]
    h2 = directed_hausdorff(
        array_2,
        array_1,
    )[0]

    # Return the Hausdorff distance (maximum of the two) and the distorted points
    return max(h1, h2)


def approximate_hausdorff_via_kdtree(
    array_1: np.ndarray,
    array_2: np.ndarray,
) -> float:
    """Approximate the Hausdorff distance using KDTree for nearest neighbor search.

    Returns:
        float: Approximated Hausdorff distance.

    """
    # Build KDTree for efficient nearest neighbor search
    tree_1 = KDTree(
        data=array_1,
    )
    tree_dist = KDTree(
        data=array_2,
    )

    # Compute nearest neighbor distances
    distances_orig_to_dist, _ = tree_1.query(
        x=array_2,
    )
    distances_dist_to_orig, _ = tree_dist.query(
        array_1,
    )

    # Approximate Hausdorff distance
    hausdorff_distance = max(
        np.max(
            distances_orig_to_dist,
        ),
        np.max(
            distances_dist_to_orig,
        ),
    )

    return hausdorff_distance


def add_gaussian_distortion(
    original_array: np.ndarray,
    distortion_param: np.ndarray,
    seed: int = 0,
) -> np.ndarray:
    """Distort and array with Gaussian noise.

    Args:
        original_array (np.ndarray): The original NumPy array.
        distortion_param (float): Standard deviation of the Gaussian noise.
        seed (int, optional): Random seed for reproducibility.

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
    noise = rng.normal(  # type: ignore - typing problem with numpy typing
        0,
        distortion_param,
        original_array.shape,
    )
    distorted_array: np.ndarray = original_array + noise

    return distorted_array
