import logging
import pathlib
import pickle

from tqdm import tqdm

from topollm.typing.enums import Verbosity
from topollm.typing.types import PerplexityResultsList

default_logger = logging.getLogger(__name__)


def load_perplexity_containers_from_pickle_files(
    path_list: list[pathlib.Path],
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> list[PerplexityResultsList]:
    """Load perplexity containers from pickle files."""
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            f"Loading perplexity containers from {path_list = } ...",  # noqa: G004 - low overhead
        )

    loaded_data_list: list[PerplexityResultsList] = []
    for path in tqdm(
        path_list,
        desc="Iterating over path_list",
    ):
        with pathlib.Path(path).open(
            mode="rb",
        ) as file:
            loaded_data = pickle.load(  # noqa: S301 - trusted source
                file,
            )
            loaded_data_list.append(
                loaded_data,
            )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            f"Loading perplexity containers from {path_list = } DONE",  # noqa: G004 - low overhead
        )

    return loaded_data_list
