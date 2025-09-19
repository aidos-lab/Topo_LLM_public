import logging

from topollm.config_classes.local_estimates.pointwise_config import LocalEstimatesPointwiseConfig
from topollm.typing.enums import NNeighborsMode, Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def get_n_neighbors_from_array_len_and_pointwise_config(
    array_len: int,
    pointwise_config: LocalEstimatesPointwiseConfig,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> int:
    """Get the number of neighbors from the array length and the pointwise config."""
    if pointwise_config.n_neighbors_mode == NNeighborsMode.ABSOLUTE_SIZE:
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg="Using absolute number of neighbors.",
            )
        n_neighbors: int = pointwise_config.absolute_n_neighbors
        if n_neighbors > array_len:
            if verbosity >= Verbosity.NORMAL:
                logger.warning(
                    msg="The number of neighbors is larger than the array length. "
                    "Setting the number of neighbors to the array length. "
                    "Note that these are not 'local' estimates anymore.",
                )
            n_neighbors = array_len
    elif pointwise_config.n_neighbors_mode == NNeighborsMode.RELATIVE_SIZE:
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg="Using relative number of neighbors.",
            )
        n_neighbors = round(
            number=array_len * pointwise_config.relative_n_neighbors,
        )
    else:
        msg = f"Unsupported {pointwise_config.n_neighbors_mode = }"
        raise ValueError(
            msg,
        )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"{n_neighbors = }",  # noqa: G004 - low overhead
        )

    return n_neighbors
