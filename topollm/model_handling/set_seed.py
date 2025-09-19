import logging
import random

import numpy as np
import torch
import torch.backends.cudnn

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def set_seed(
    seed: int,
    logger: logging.Logger = default_logger,
) -> None:
    """Set the seed for generating random numbers in PyTorch and numpy.

    Args:
    ----
        seed:
            The seed for the random number generator.
        logger:
            The logger to use for logging messages.

    Notes:
    -----
        1. The RNG state for the CUDA is set, which makes CUDA operations deterministic.
        2. A seed for the Python built-in random module is also set.
        3. PyTorch's cuDNN uses nondeterministic algorithms which can be
           disabled setting `torch.backends.cudnn.deterministic = True`.
           However, this can slow down the computations.
        4. PyTorch's cuDNN has a benchmark mode which allows hardware
           optimizations for the operations. This can be enabled or disabled
           using `torch.backends.cudnn.benchmark`. Disabling it helps in making
           the computations deterministic.
        5. For operations performed on CPU and CUDA, setting the seed ensures
           reproducibility across multiple runs.

    """
    # Set the seed for Python's built-in random module
    random.seed(
        a=seed,
    )

    # Set the seed for numpy random number generator
    np.random.seed(  # noqa: NPY002 - we set the seed here for reproducibility
        seed=seed,
    )

    # Set the seed for CPU operations
    torch.manual_seed(
        seed=seed,
    )

    # Set the seed for all GPU devices and accelarators if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed=seed,
        )
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(
            seed=seed,
        )

    torch.backends.cudnn.deterministic = True  # Disable nondeterministic algorithms
    torch.backends.cudnn.benchmark = False  # Disable hardware optimizations

    logger.info(
        msg=f"seed set to {seed = }.",  # noqa: G004 - low overhead
    )
