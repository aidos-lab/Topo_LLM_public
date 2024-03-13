# coding=utf-8
#
# Copyright 2024
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Ruppik (ruppik@hhu.de)
# Julius von Rohrscheidt (julius.rohrscheidt@helmholtz-muenchen.de)
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

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Imports

# System imports
import pathlib

# Third-party imports
import pytest
import pickle

# Local imports

# END Imports
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


@pytest.fixture(scope="session")
def example_batch() -> dict:
    example_data_pickle_path = pathlib.Path(
        pathlib.Path(__file__).parent,
        "example_data",
        "example_data_batch.pkl",
    )

    with open(
        file=example_data_pickle_path,
        mode="rb",
    ) as file:
        example_data = pickle.load(
            file=file,
        )

        return example_data


@pytest.fixture(scope="session")
def chunk_idx() -> int:
    return 7
