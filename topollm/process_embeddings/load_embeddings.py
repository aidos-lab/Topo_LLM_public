"""Load embeddings and metadata."""

import logging
import pathlib
import pickle
from typing import TYPE_CHECKING

import hydra
import hydra.core.hydra_config
import omegaconf
import zarr

from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.setup_exception_logging import setup_exception_logging

if TYPE_CHECKING:
    from topollm.config_classes.main_config import MainConfig


global_logger: logging.Logger = logging.getLogger(
    name=__name__,
)

setup_exception_logging(
    logger=global_logger,
)


@hydra.main(
    config_path="../../configs",
    config_name="main_config",
    version_base="1.2",
)
def main(
    config: omegaconf.DictConfig,
) -> None:
    """Run the script."""
    global_logger.info("Running script ...")

    main_config: MainConfig = initialize_configuration(
        config=config,
        logger=global_logger,
    )

    # # # #
    # Load the embeddings

    array_path = pathlib.Path(
        pathlib.Path.home(),
        "git-source",
        "Topo_LLM",
        "data",
        "embeddings",
        "arrays",
        "data-xsum_split-train_ctxt-dataset_entry/lvl-token/add-prefix-space-False_max-len-512/model-roberta-base_mask-no_masking/layer-[-1]_agg-mean/norm-None/",
        "array_dir",
        "test_array_dir",
    )

    if not array_path.exists():
        msg = f"{array_path = } does not exist."
        raise FileNotFoundError(msg)

    array = zarr.open(
        store=array_path,  # type: ignore - zarr typing problem
        mode="r",
    )

    print(f"{array.shape = }")
    print(f"{array = }")
    print(f"{array[0] = }")

    # # # #
    # Load the metadata

    metadata_root_storage_path = pathlib.Path(
        pathlib.Path.home(),
        "git-source",
        "Topo_LLM",
        "data",
        "embeddings",
        "metadata",
        "data-xsum_split-train_ctxt-dataset_entry/lvl-token/add-prefix-space-False_max-len-512/model-roberta-base_mask-no_masking/layer-[-1]_agg-mean/norm-None/",
        "metadata_dir",
    )

    if not metadata_root_storage_path.exists():
        msg = f"{metadata_root_storage_path = } does not exist."
        raise FileNotFoundError(msg)

    # Load pickled metadata
    metadata_chunk_path = metadata_root_storage_path / "chunk_00156.pkl"

    with open(
        file=metadata_chunk_path,
        mode="rb",
    ) as file:
        metadata_chunk = pickle.load(
            file=file,
        )

    # "pickle_chunked_metadata_storage/chunk_00002.pkl"


if __name__ == "__main__":
    main()
