# Copyright 2024
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Ruppik (ruppik@hhu.de)
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

"""Protocol for managing paths to embeddings."""

import pathlib
from typing import Protocol


class EmbeddingsPathManager(Protocol):
    """Protocol for managing paths to embeddings."""

    @property
    def data_dir(
        self,
    ) -> pathlib.Path: ...  # pragma: no cover

    @property
    def array_dir_absolute_path(
        self,
    ) -> pathlib.Path: ...  # pragma: no cover

    @property
    def metadata_dir_absolute_path(
        self,
    ) -> pathlib.Path: ...  # pragma: no cover

    @property
    def perplexity_dir_absolute_path(
        self,
    ) -> pathlib.Path: ...  # pragma: no cover

    @property
    def prepared_data_dir_absolute_path(
        self,
    ) -> pathlib.Path: ...  # pragma: no cover

    def get_prepared_data_array_save_path(
        self,
    ) -> pathlib.Path: ...  # pragma: no cover

    def get_prepared_data_meta_save_path(
        self,
    ) -> pathlib.Path: ...  # pragma: no cover

    def get_local_estimates_dir_absolute_path(
        self,
    ) -> pathlib.Path: ...  # pragma: no cover

    def get_local_estimates_array_save_path(
        self,
    ) -> pathlib.Path: ...  # pragma: no cover

    def get_local_estimates_meta_save_path(
        self,
    ) -> pathlib.Path: ...  # pragma: no cover

    def get_analyzed_data_dir_absolute_path(
        self,
    ) -> pathlib.Path: ...  # pragma: no cover
