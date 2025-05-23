# Copyright 2024
# [ANONYMIZED_INSTITUTION],
# [ANONYMIZED_FACULTY],
# [ANONYMIZED_DEPARTMENT]
#
# Authors:
# AUTHOR_1 (author1@example.com)
# AUTHOR_2 (author2@example.com)
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
