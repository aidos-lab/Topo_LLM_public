import numpy as np

# # # # # # # # # # # #
# Neighborhood ranks


def pairwise_distances(
    X: np.ndarray,
) -> np.ndarray:
    """Calculate pairwise distance matrix of a given data matrix and return said matrix."""
    D = np.sum((X[None, :] - X[:, None]) ** 2, -1) ** 0.5
    return D


def get_neighbours_and_ranks(
    X: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the neighbourhoods and the ranks of a given space `X`, and return the corresponding tuple.

    An additional parameter $k$,
    the size of the neighbourhood, is required.
    """
    X = pairwise_distances(X)

    # Warning: this is only the ordering of neighbours that we need to
    # extract neighbourhoods below. The ranking comes later!
    X_ranks = np.argsort(
        X,
        axis=-1,
        kind="stable",
    )

    # Extract neighbourhoods.
    X_neighbourhood = X_ranks[:, 1 : k + 1]

    # Convert this into ranks (finally)
    X_ranks = X_ranks.argsort(
        axis=-1,
        kind="stable",
    )

    return X_neighbourhood, X_ranks


def MRRE_pointwise(
    X: np.ndarray,
    Z: np.ndarray,
    k: int,
) -> np.ndarray:
    """Calculate the pointwise mean rank distortion for each data point in the data space `X` with respect to the latent space `Z`.

    Inputs:
        - X: array of shape (m, n) (data space)
        - Z: array of shape (m, l) (latent space)
        - k: number of nearest neighbors to consider
    Output:
        - mean_rank_distortions: array of length m
    """
    (
        X_neighbourhood,
        X_ranks,
    ) = get_neighbours_and_ranks(
        X,
        k,
    )
    (
        Z_neighbourhood,
        Z_ranks,
    ) = get_neighbours_and_ranks(
        Z,
        k,
    )

    n = X.shape[0]
    mean_rank_distortions = np.zeros(n)

    for row in range(n):
        rank_differences = []
        for neighbour in Z_neighbourhood[row]:
            rx = X_ranks[row, neighbour]
            rz = Z_ranks[row, neighbour]
            rank_differences.append(abs(rx - rz) / rz)
        mean_rank_distortions[row] = np.mean(rank_differences)

    return mean_rank_distortions
