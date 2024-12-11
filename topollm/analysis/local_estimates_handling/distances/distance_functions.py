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
import ot
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


def pot_sinkhorn_wasserstein(
    P: np.ndarray,
    Q: np.ndarray,
    weights_P: np.ndarray | None = None,
    weights_Q: np.ndarray | None = None,
    p: int = 2,
    reg: float = 0.5,
) -> float:
    """Compute Wasserstein distance W_p using POT Sinkhorn algorithm.

    Parameters
    ----------
    - P (ndarray): Array of shape (n, d), representing n points in d-dimensional space.
    - Q (ndarray): Array of shape (m, d), representing m points in d-dimensional space.
    - weights_P (ndarray): Optional array of shape (n,) for weights of P. Defaults to uniform.
    - weights_Q (ndarray): Optional array of shape (m,) for weights of Q. Defaults to uniform.
    - p (float): Order of Wasserstein distance.
    - reg (float): Regularization parameter for Sinkhorn algorithm.

    Returns
    -------
    - wp (float): Approximate Wasserstein distance.

    """
    # Default weights
    n = P.shape[0]
    m = Q.shape[0]

    if weights_P is None:
        weights_P = np.ones(shape=n) / n
    if weights_Q is None:
        weights_Q = np.ones(shape=m) / m

    # Compute the cost matrix (pairwise distances to the power of p)
    cost_matrix = (
        ot.dist(
            x1=P,
            x2=Q,
            metric="euclidean",
        )
        ** p
    )

    # Compute the Sinkhorn divergence
    transport_matrix = ot.sinkhorn(
        a=weights_P,
        b=weights_Q,
        M=cost_matrix,
        reg=reg,
    )

    # Compute the Wasserstein distance (Sinkhorn output is the transport cost)
    wp = np.sum(a=transport_matrix * cost_matrix) ** (1 / p)

    return wp


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
    P: torch.Tensor = torch.tensor(
        data=P_np,
        dtype=torch.float32,
    )
    Q: torch.Tensor = torch.tensor(
        data=Q_np,
        dtype=torch.float32,
    )

    n, d = P.shape
    m, _ = Q.shape

    # Default weights
    if weights_P_np is None:
        weights_P: torch.Tensor = torch.ones(n, dtype=torch.float32) / n
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
        loss="sinkhorn",
        p=p,
        blur=blur,
        scaling=scaling,
    )

    # Compute the Wasserstein distance
    wp = loss_fn(
        weights_P,
        P,
        weights_Q,
        Q,
    ).item()

    return wp
