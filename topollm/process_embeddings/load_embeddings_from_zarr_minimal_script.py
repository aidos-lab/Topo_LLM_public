import pathlib

import zarr

from topollm.config_classes.constants import TOPO_LLM_REPOSITORY_BASE_PATH


def main():
    # array_path = pathlib.Path(
    #     pathlib.Path.home(),
    #     "git-source",
    #     "Topo_LLM",
    #     "data",
    #     "embeddings",
    #     "test_array_dir",
    # )

    array_path = pathlib.Path(
        TOPO_LLM_REPOSITORY_BASE_PATH,
        "data/embeddings/test_array_dir",
    )

    print(f"{array_path = }")

    if not array_path.exists():
        raise FileNotFoundError(f"{array_path = } does not exist.")

    array = zarr.open(
        store=array_path,  # type: ignore
        mode="r",
    )

    print(f"{array.shape = }")
    print(f"{array = }")
    print(f"{array[0] = }")


if __name__ == "__main__":
    main()
