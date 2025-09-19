import logging

import numpy as np
import sklearn.decomposition
import sklearn.manifold

from topollm.typing.enums import Verbosity

default_logger = logging.getLogger(__name__)


def create_projected_data(
    array: np.ndarray,
    pca_n_components: int | None = 50,
    tsne_n_components: int = 2,
    tsne_random_state: int = 42,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> np.ndarray:
    """Create a projection of the input data using PCA and t-SNE."""
    # Apply PCA if requested
    if pca_n_components:
        if verbosity >= Verbosity.VERBOSE:
            logger.info(
                "Applying PCA to reduce the number of dimensions to %d ...",
                pca_n_components,
            )
        pca = sklearn.decomposition.PCA(
            n_components=pca_n_components,
        )
        array = pca.fit_transform(
            array,
        )
        if verbosity >= Verbosity.VERBOSE:
            logger.info(
                "Applying PCA to reduce the number of dimensions to %d DONE",
                pca_n_components,
            )

    # Apply t-SNE
    if verbosity >= Verbosity.VERBOSE:
        logger.info(
            "Applying t-SNE to reduce the number of dimensions to %d ...",
            tsne_n_components,
        )
    tsne = sklearn.manifold.TSNE(
        n_components=tsne_n_components,
        random_state=tsne_random_state,
    )
    tsne_array = tsne.fit_transform(
        array,
    )
    if verbosity >= Verbosity.VERBOSE:
        logger.info(
            "Applying t-SNE to reduce the number of dimensions to %d DONE",
            tsne_n_components,
        )

    return tsne_array
