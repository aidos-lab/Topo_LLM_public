"""Load the local estimates metadata from pickle files and save as CSV."""

import logging
import pathlib
from itertools import product

import pandas as pd
from tqdm import tqdm

from topollm.config_classes.constants import TOPO_LLM_REPOSITORY_BASE_PATH
from topollm.typing.enums import Verbosity

_POINTWISE_METADATA_PICKLE_FILE_NAME = "local_estimates_pointwise_meta.pkl"
_POINTWISE_METADATA_CSV_FILE_NAME = "local_estimates_pointwise_meta.csv"

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def main() -> None:
    """Call the metadata conversion for multiple files."""
    logger: logging.Logger = default_logger
    verbosity: Verbosity = Verbosity.NORMAL

    relative_path_list_debug_data: list[pathlib.Path] = [
        pathlib.Path(
            "data/analysis/local_estimates/data=multiwoz21_with_bio_tags_rm-empty=True_spl-mode=do_nothing_ctxt=dataset_entry_feat-col=ner_tags/",
            "split=validation_samples=512_sampling=random_sampling-seed=778/edh-mode=regular_lvl=token/add-prefix-space=False_max-len=512/model=roberta-base_task=masked_lm_dr=defaults/layer=-1_agg=mean/norm=None/sampling=random_seed=42_samples=3000/desc=twonn_samples=500_zerovec=keep_dedup=array_deduplicator_noise=do_nothing/n-neighbors-mode=absolute_size_n-neighbors=128",
        ),
        pathlib.Path(
            "data/analysis/local_estimates/data=sgd_with_bio_tags_rm-empty=True_spl-mode=do_nothing_ctxt=dataset_entry_feat-col=ner_tags/",
            "split=validation_samples=512_sampling=random_sampling-seed=778/edh-mode=regular_lvl=token/add-prefix-space=False_max-len=512/model=roberta-base_task=masked_lm_dr=defaults/layer=-1_agg=mean/norm=None/sampling=random_seed=42_samples=3000/desc=twonn_samples=500_zerovec=keep_dedup=array_deduplicator_noise=do_nothing/n-neighbors-mode=absolute_size_n-neighbors=128",
        ),
    ]

    relative_path_list_full_data: list[pathlib.Path] = [
        pathlib.Path(
            "data/analysis/local_estimates/",
            f"data={dataset_description_string}_rm-empty=True_spl-mode=do_nothing_ctxt=dataset_entry_feat-col=ner_tags/",
            f"split={split}_samples=10000_sampling=random_sampling-seed=778/",
            "edh-mode=regular_lvl=token/add-prefix-space=False_max-len=512/model=roberta-base_task=masked_lm_dr=defaults/layer=-1_agg=mean/norm=None/sampling=random_seed=42_samples=150000/desc=twonn_samples=60000_zerovec=keep_dedup=array_deduplicator_noise=do_nothing/n-neighbors-mode=absolute_size_n-neighbors=128",
        )
        for dataset_description_string, split in product(
            [
                "multiwoz21_with_bio_tags",
                "sgd_with_bio_tags",
            ],
            [
                "train",
                "validation",
                "test",
            ],
        )
    ]

    relative_path_list_to_convert: list[pathlib.Path] = relative_path_list_debug_data + relative_path_list_full_data

    for relative_path in tqdm(
        iterable=relative_path_list_to_convert,
        desc="Iterating over relative paths and converting metadata",
    ):
        local_estimates_pointwise_dir_absolute_path = pathlib.Path(
            TOPO_LLM_REPOSITORY_BASE_PATH,
            relative_path,
        )

        load_metadata_and_save_as_csv(
            local_estimates_pointwise_dir_absolute_path=local_estimates_pointwise_dir_absolute_path,
            verbosity=verbosity,
            logger=logger,
        )


def load_metadata_and_save_as_csv(
    local_estimates_pointwise_dir_absolute_path: pathlib.Path,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Load the metadata from a pickle file and save it as a CSV file."""
    meta_pickle_file_path = pathlib.Path(
        local_estimates_pointwise_dir_absolute_path,
        _POINTWISE_METADATA_PICKLE_FILE_NAME,
    )

    meta_csv_file_path = pathlib.Path(
        local_estimates_pointwise_dir_absolute_path,
        _POINTWISE_METADATA_CSV_FILE_NAME,
    )

    # Load the metadata from the pickle file
    meta_df: pd.DataFrame = pd.read_pickle(  # noqa: S301 - we trust this input
        filepath_or_buffer=meta_pickle_file_path,
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"{type(meta_df)=} loaded from {meta_pickle_file_path=}",  # noqa: G004 - low overhead
        )

    # Save the metadata as a CSV file
    meta_df.to_csv(
        path_or_buf=meta_csv_file_path,
        index=False,
    )


if __name__ == "__main__":
    main()
