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

"""Path manager for embeddings with separate directories for arrays and metadata."""

import logging
import pathlib

from topollm.config_classes.data.data_config import DataConfig
from topollm.config_classes.embeddings.embeddings_config import EmbeddingsConfig
from topollm.config_classes.embeddings_data_prep.embeddings_data_prep_config import EmbeddingsDataPrepConfig
from topollm.config_classes.language_model.language_model_config import (
    LanguageModelConfig,
)
from topollm.config_classes.local_estimates.local_estimates_config import LocalEstimatesConfig
from topollm.config_classes.paths.paths_config import PathsConfig
from topollm.config_classes.tokenizer.tokenizer_config import TokenizerConfig
from topollm.config_classes.transformations.transformations_config import TransformationsConfig
from topollm.typing.enums import Verbosity

default_logger = logging.getLogger(__name__)

default_embeddings_data_prep_config = EmbeddingsDataPrepConfig()
default_local_estimates_config = LocalEstimatesConfig()


class EmbeddingsPathManagerSeparateDirectories:
    """Path manager for embeddings with separate directories for arrays and metadata."""

    def __init__(
        self,
        data_config: DataConfig,
        embeddings_config: EmbeddingsConfig,
        language_model_config: LanguageModelConfig,
        paths_config: PathsConfig,
        transformations_config: TransformationsConfig,
        tokenizer_config: TokenizerConfig,
        embeddings_data_prep_config: EmbeddingsDataPrepConfig = default_embeddings_data_prep_config,
        local_estimates_config: LocalEstimatesConfig = default_local_estimates_config,
        verbosity: Verbosity = Verbosity.NORMAL,
        logger: logging.Logger = default_logger,
    ) -> None:
        """Initialize the path manager."""
        self.data_config: DataConfig = data_config
        self.embeddings_config: EmbeddingsConfig = embeddings_config
        self.language_model_config: LanguageModelConfig = language_model_config
        self.paths_config: PathsConfig = paths_config
        self.transformations_config: TransformationsConfig = transformations_config
        self.tokenizer_config: TokenizerConfig = tokenizer_config
        self.embeddings_data_prep_config: EmbeddingsDataPrepConfig = embeddings_data_prep_config
        self.local_estimates_config: LocalEstimatesConfig = local_estimates_config

        self.verbosity = verbosity
        self.logger = logger

    @property
    def data_dir(
        self,
    ) -> pathlib.Path:
        return self.paths_config.data_dir

    def get_nested_subfolder_path(
        self,
    ) -> pathlib.Path:
        """Construct a nested subfolder path based on specific attributes.

        Returns
        -------
            pathlib.Path: The constructed nested subfolder path.

        """
        path = pathlib.Path(
            self.data_config.config_description,
            self.embeddings_config.config_description,
            self.tokenizer_config.config_description,
            self.language_model_config.config_description,
            self.embeddings_config.embedding_extraction.config_description,
            self.transformations_config.config_description,
        )

        return path

    # # # #
    # array_dir

    @property
    def array_dir_absolute_path(
        self,
    ) -> pathlib.Path:
        path = pathlib.Path(
            self.data_dir,
            self.array_dir_relative_path,
        )

        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                "array_dir_absolute_path:\n%s",
                path,
            )

        return path

    @property
    def array_dir_relative_path(
        self,
    ) -> pathlib.Path:
        path = pathlib.Path(
            "embeddings",
            "arrays",
            self.get_nested_subfolder_path(),
            self.array_dir_name,
        )

        return path

    @property
    def array_dir_name(
        self,
    ) -> str:
        return "array_dir"

    # # # #
    # metadata_dir

    @property
    def metadata_dir_absolute_path(
        self,
    ) -> pathlib.Path:
        path = pathlib.Path(
            self.data_dir,
            self.metadata_dir_relative_path,
        )

        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                "metadata_dir_absolute_path:\n%s",
                path,
            )

        return path

    @property
    def metadata_dir_relative_path(
        self,
    ) -> pathlib.Path:
        path = pathlib.Path(
            "embeddings",
            "metadata",
            self.get_nested_subfolder_path(),
            self.metadata_dir_name,
        )

        return path

    @property
    def metadata_dir_name(
        self,
    ) -> str:
        return "metadata_dir"

    # # # #
    # perplexity directory

    @property
    def perplexity_dir_absolute_path(
        self,
    ) -> pathlib.Path:
        path = pathlib.Path(
            self.data_dir,
            self.perplexity_dir_relative_path,
        )

        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                "perplexity_dir_absolute_path:\n%s",
                path,
            )

        return path

    @property
    def perplexity_dir_relative_path(
        self,
    ) -> pathlib.Path:
        path = pathlib.Path(
            "embeddings",
            "perplexity",
            self.get_nested_subfolder_path(),
            self.perplexity_dir_name,
        )

        return path

    @property
    def perplexity_dir_name(
        self,
    ) -> str:
        return "perplexity_dir"

    # # # #
    # prepared data directory

    @property
    def prepared_data_dir_absolute_path(
        self,
    ) -> pathlib.Path:
        path = pathlib.Path(
            self.data_dir,
            "analysis",
            "prepared",
            self.get_nested_subfolder_path(),
            self.embeddings_data_prep_config.config_description,
        )

        return path

    def get_prepared_data_array_save_path(
        self,
        prepared_data_array_file_name: str = "embeddings_samples_paddings_removed.npy",
    ) -> pathlib.Path:
        """Get the path to save the prepared data array.

        Note: If this does not have the '.npy' extension,
        the numpy save function will add it automatically.
        In particular, do not use '.np' here.
        """
        path = pathlib.Path(
            self.prepared_data_dir_absolute_path,
            prepared_data_array_file_name,
        )

        return path

    def get_prepared_data_meta_save_path(
        self,
        prepared_data_meta_file_name: str = "embeddings_samples_paddings_removed_meta.pkl",
    ) -> pathlib.Path:
        path = pathlib.Path(
            self.prepared_data_dir_absolute_path,
            prepared_data_meta_file_name,
        )

        return path

    # # # #
    # local estimates directories

    def get_local_estimates_dir_absolute_path(
        self,
    ) -> pathlib.Path:
        path = pathlib.Path(
            self.data_dir,
            "analysis",
            self.local_estimates_config.description,
            self.get_nested_subfolder_path(),
            self.embeddings_data_prep_config.config_description,  # We include this because the local estimates are computed on the prepared data
            self.local_estimates_config.config_description,
        )

        return path

    def get_local_estimates_array_save_path(
        self,
        local_estimates_file_name: str = "local_estimates_paddings_removed.npy",
    ) -> pathlib.Path:
        path = pathlib.Path(
            self.get_local_estimates_dir_absolute_path(),
            local_estimates_file_name,
        )

        return path

    def get_local_estimates_meta_save_path(
        self,
        local_estimates_meta_file_name: str = "local_estimates_paddings_removed_meta.pkl",
    ) -> pathlib.Path:
        path = pathlib.Path(
            self.get_local_estimates_dir_absolute_path(),
            local_estimates_meta_file_name,
        )

        return path

    # # # #
    # analysed data directories

    def get_analyzed_data_dir_absolute_path(
        self,
    ) -> pathlib.Path:
        path: pathlib.Path = pathlib.Path(
            self.perplexity_dir_absolute_path,
            f"layer-{self.embeddings_config.embedding_extraction.layer_indices}",
            self.local_estimates_config.config_description,
        )

        return path
