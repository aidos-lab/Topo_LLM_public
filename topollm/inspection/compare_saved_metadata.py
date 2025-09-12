"""Compare saved metadata files from different models."""

import pathlib

import pandas as pd

from topollm.config_classes.constants import TOPO_LLM_REPOSITORY_BASE_PATH


def main() -> None:
    """Compare saved metadata files from different models."""
    common_prefix: pathlib.Path = pathlib.Path(
        "data/embeddings/metadata/data=luster_column=source_target_rm-empty=True_spl-mode=do_nothing_ctxt=dataset_entry_feat-col=ner_tags/split=validation_samples=7000_sampling=random_sampling-seed=778/edh-mode=regular_lvl=token/add-prefix-space=False_max-len=512/",
    )
    common_suffix: pathlib.Path = pathlib.Path(
        "layer=-1_agg=mean/norm=None/metadata_dir/pickle_chunked_metadata_storage",
    )
    file_name: str = "chunk_00310.pkl"

    meta_file_paths: list[pathlib.Path] = [
        pathlib.Path(
            TOPO_LLM_REPOSITORY_BASE_PATH,
            common_prefix,
            model_path,
            common_suffix,
            file_name,
        )
        for model_path in [
            "model=Phi-3.5-mini-instruct_task=causal_lm_dr=defaults",
            "model=luster-full_task=causal_lm_dr=defaults",
        ]
    ]

    loaded_meta_containers: list[dict] = []

    for meta_file_path in meta_file_paths:
        loaded_meta_container = pd.read_pickle(  # noqa: S301 - we trust these files
            filepath_or_buffer=meta_file_path,
        )
        loaded_meta_containers.append(
            loaded_meta_container,
        )

    print(  # noqa: T201 - we want this script to print to console
        f"{len(loaded_meta_containers)=}",
    )
    for loaded_meta_container in loaded_meta_containers:
        print(  # noqa: T201 - we want this script to print to console
            f"{type(loaded_meta_container)=}",
        )

    pass  # noqa: PIE790 - This `pass` statement can be used for setting breakpoints.


if __name__ == "__main__":
    main()
