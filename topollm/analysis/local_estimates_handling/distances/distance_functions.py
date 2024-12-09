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


import numpy as np
import torch
from geomloss import SamplesLoss
from scipy.spatial import KDTree
from scipy.spatial.distance import directed_hausdorff


def compute_exact_hausdorff(
    array_1: np.ndarray,
    array_2: np.ndarray,
) -> float:
    """Compute the exact Hausdorff distance.

    Note that this might be expensive, depending on the number of points.
    """
    # Compute directed Hausdorff distances
    h1: float = directed_hausdorff(
        array_1,
        array_2,
    )[0]
    h2: float = directed_hausdorff(
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
    distances_array_1_to_array_2, _ = tree_1.query(
        x=array_2,
    )
    distances_array_2_to_array_1, _ = tree_dist.query(
        array_1,
    )

    # Approximate Hausdorff distance
    hausdorff_distance = max(
        np.max(
            distances_array_1_to_array_2,
        ),
        np.max(
            distances_array_2_to_array_1,
        ),
    )

    return hausdorff_distance


def geomloss_sinkhorn_wasserstein(
    P_np: np.ndarray,
    Q_np: np.ndarray,
    weights_P_np: np.ndarray | None = None,
    weights_Q_np: np.ndarray | None = None,
    p: int = 2,
    blur: float = 0.05,
    scaling: float = 0.9,
):
    """SCompute Wasserstein distance W_p using GeomLoss Sinkhorn divergence.

    Parameters
    ----------
    - P (ndarray): Array of shape (n, d), representing n points in d-dimensional space.
    - Q (ndarray): Array of shape (m, d), representing m points in d-dimensional space.
    - weights_P (ndarray): Optional array of shape (n,) for weights of P. Defaults to uniform.
    - weights_Q (ndarray): Optional array of shape (m,) for weights of Q. Defaults to uniform.
    - p: Order of Wasserstein distance.
    - blur: Regularization parameter (similar to reg in Sinkhorn).
    - scaling: Scaling parameter for convergence.

    Returns
    -------
    - wp (float): Approximate Wasserstein distance.

    """
    # Convert to PyTorch tensors
    P = torch.tensor(
        P_np,
        dtype=torch.float32,
    )
    Q = torch.tensor(
        Q_np,
        dtype=torch.float32,
    )

    n, d = P.shape
    m, _ = Q.shape

    # Default weights
    if weights_P_np is None:
        weights_P = torch.ones(n, dtype=torch.float32) / n
    else:
        weights_P = torch.tensor(
            weights_P_np,
            dtype=torch.float32,
        )
    if weights_Q_np is None:
        weights_Q = torch.ones(m, dtype=torch.float32) / m
    else:
        weights_Q = torch.tensor(
            weights_Q_np,
            dtype=torch.float32,
        )

    # Define the Sinkhorn divergence loss
    loss_fn = SamplesLoss(
        "sinkhorn",
        p=p,
        blur=blur,
        scaling=scaling,
    )

    # Compute Wasserstein distance
    #
    # TODO:
    #     Exception has occurred: ValueError
    # Input weights 'α' and 'β' should have the same number of dimensions.
    #   File "/Users/ruppik/git-source/Topo_LLM/topollm/analysis/local_estimates_handling/distances/distance_functions.py", line 161, in geomloss_sinkhorn_wasserstein
    #     wp = loss_fn(
    #          ^^^^^^^^
    #   File "/Users/ruppik/git-source/Topo_LLM/topollm/analysis/local_estimates_computation/global_and_pointwise_local_estimates_worker.py", line 240, in compute_distance_metrics
    #     sinkhorn_wasserstein = geomloss_sinkhorn_wasserstein(
    #                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #   File "/Users/ruppik/git-source/Topo_LLM/topollm/analysis/local_estimates_computation/global_and_pointwise_local_estimates_worker.py", line 128, in global_and_pointwise_local_estimates_worker
    #     additional_distance_computations_results: dict = compute_distance_metrics(
    #                                                      ^^^^^^^^^^^^^^^^^^^^^^^^^
    #   File "/Users/ruppik/git-source/Topo_LLM/topollm/pipeline_scripts/worker_for_pipeline.py", line 118, in worker_for_pipeline
    #     global_and_pointwise_local_estimates_worker(
    #   File "/Users/ruppik/git-source/Topo_LLM/topollm/pipeline_scripts/run_pipeline_compute_embeddings_and_data_prep_and_local_estimate.py", line 90, in main
    #     worker_for_pipeline(
    #   File "/Users/ruppik/git-source/Topo_LLM/topollm/pipeline_scripts/run_pipeline_compute_embeddings_and_data_prep_and_local_estimate.py", line 100, in <module>
    #     main()
    # ValueError: Input weights 'α' and 'β' should have the same number of dimensions.
    wp = loss_fn(
        P,
        Q,
        weights_P,
        weights_Q,
    )

    return wp.item()
