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

from topollm.config_classes.main_config import MainConfig
from topollm.typing.enums import PerplexityContainerSaveFormat, Verbosity

default_logger = logging.getLogger(__name__)


class EmbeddingsPathManagerSeparateDirectories:
    """Path manager for embeddings with separate directories for arrays and metadata."""

    def __init__(
        self,
        main_config: MainConfig,
        verbosity: Verbosity = Verbosity.NORMAL,
        logger: logging.Logger = default_logger,
    ) -> None:
        """Initialize the path manager."""
        self.main_config = main_config

        self.verbosity = verbosity
        self.logger = logger

    @property
    def data_dir(
        self,
    ) -> pathlib.Path:
        return self.main_config.paths.data_dir

    def get_nested_subfolder_path(
        self,
    ) -> pathlib.Path:
        """Construct a nested subfolder path based on specific attributes.

        Returns
        -------
            pathlib.Path: The constructed nested subfolder path.

        """
        path = pathlib.Path(
            self.main_config.data.config_description,
            self.main_config.embeddings.config_description,
            self.main_config.tokenizer.config_description,
            self.main_config.language_model.config_description,
            self.main_config.embeddings.embedding_extraction.config_description,
            self.main_config.transformations.config_description,
        )

        return path

    # # # #
    # array directory

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
    # metadata directory

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

    def get_perplexity_container_save_file_absolute_path(
        self,
        perplexity_container_save_format: PerplexityContainerSaveFormat = PerplexityContainerSaveFormat.LIST_AS_JSONL,
    ) -> pathlib.Path:
        file_name = get_perplexity_container_save_file_name(
            perplexity_container_save_format=perplexity_container_save_format,
        )

        path = pathlib.Path(
            self.perplexity_dir_absolute_path,
            file_name,
        )

        return path

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
            self.main_config.embeddings_data_prep.config_description,
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

    def get_local_estimates_subfolder_path(
        self,
    ) -> pathlib.Path:
        """Construct a nested subfolder path to describe the local estimates."""
        # We include the
        # `embeddings_data_prep.config_description`
        # because the local estimates are computed on the prepared data.
        path = pathlib.Path(
            self.main_config.local_estimates.description,
            self.get_nested_subfolder_path(),
            self.main_config.embeddings_data_prep.config_description,
            self.main_config.local_estimates.config_description,
        )

        return path

    def get_local_estimates_dir_absolute_path(
        self,
    ) -> pathlib.Path:
        path = pathlib.Path(
            self.data_dir,
            "analysis",
            self.get_local_estimates_subfolder_path(),
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
    # saved plots directories

    @property
    def saved_plots_dir_absolute_path(
        self,
    ) -> pathlib.Path:
        path = pathlib.Path(
            self.data_dir,
            "saved_plots",
        )

        return path

    def get_saved_plots_local_estimates_projection_dir_absolute_path(
        self,
    ) -> pathlib.Path:
        path = pathlib.Path(
            self.saved_plots_dir_absolute_path,
            "local_estimates_projection",
            self.get_local_estimates_subfolder_path(),
        )

        return path

    def get_local_estimates_projection_plot_save_path(
        self,
        file_name: str = "tsne_plot.html",
    ) -> pathlib.Path:
        path = pathlib.Path(
            self.get_saved_plots_local_estimates_projection_dir_absolute_path(),
            file_name,
        )

        return path

    # # # #
    # analyzed data directories

    def get_analyzed_data_dir_absolute_path(
        self,
    ) -> pathlib.Path:
        path: pathlib.Path = pathlib.Path(
            self.data_dir,
            "analysis",
            "aligned_and_analyzed",
            self.get_local_estimates_subfolder_path(),
        )

        return path

    def get_aligned_df_save_path(
        self,
        file_name: str = "aligned_df.csv",
    ) -> pathlib.Path:
        path = pathlib.Path(
            self.get_analyzed_data_dir_absolute_path(),
            file_name,
        )

        return path

    def get_correlation_results_dir_absolute_path(
        self,
    ) -> pathlib.Path:
        path = pathlib.Path(
            self.get_analyzed_data_dir_absolute_path(),
            "correlation_results",
        )

        return path

    def get_correlation_results_df_save_path(
        self,
        method: str = "kendall",
    ) -> pathlib.Path:
        path = pathlib.Path(
            self.get_correlation_results_dir_absolute_path(),
            f"correlation_results_{method}.csv",
        )

        return path

    def get_aligned_histograms_dir_absolute_path(
        self,
    ) -> pathlib.Path:
        path = pathlib.Path(
            self.get_analyzed_data_dir_absolute_path(),
            "histograms",
        )

        return path

    def get_aligned_histograms_plot_save_path(
        self,
        file_name: str = "histograms.pdf",
    ) -> pathlib.Path:
        path = pathlib.Path(
            self.get_aligned_histograms_dir_absolute_path(),
            file_name,
        )

        return path

    def get_aligned_scatter_plots_dir_absolute_path(
        self,
    ) -> pathlib.Path:
        path = pathlib.Path(
            self.get_analyzed_data_dir_absolute_path(),
            "scatter_plots",
        )

        return path

    def get_aligned_scatter_plot_save_path(
        self,
        file_name: str = "scatter_plot.pdf",
    ) -> pathlib.Path:
        path = pathlib.Path(
            self.get_aligned_scatter_plots_dir_absolute_path(),
            file_name,
        )

        return path


def get_perplexity_container_save_file_name(
    perplexity_container_save_format: PerplexityContainerSaveFormat = PerplexityContainerSaveFormat.LIST_AS_JSONL,
) -> str:
    """Get the file name for saving the perplexity container."""
    match perplexity_container_save_format:
        case PerplexityContainerSaveFormat.LIST_AS_JSONL:
            file_name = "perplexity_results_list.jsonl"
        case PerplexityContainerSaveFormat.LIST_AS_PICKLE:
            file_name = "perplexity_results_list.pkl"
        case PerplexityContainerSaveFormat.CONCATENATED_DATAFRAME_AS_CSV:
            file_name = "token_perplexities_df.csv"
        case PerplexityContainerSaveFormat.CONCATENATED_ARRAY_AS_ZARR:
            file_name = "token_perplexities_array.zarr"
        case _:
            msg = f"Unsupported {perplexity_container_save_format = }."
            raise ValueError(msg)

    return file_name
