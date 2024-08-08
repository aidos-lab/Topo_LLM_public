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

"""Run script to compute twoNN estimates from prepared embeddings."""

import logging
import os
import pathlib
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import skdim
import sklearn.decomposition
import sklearn.manifold
import torch

from topollm.analysis.local_estimates.filter.get_local_estimates_filter import get_local_estimates_filter
from topollm.analysis.local_estimates.saving.local_estimates_containers import LocalEstimatesContainer
from topollm.analysis.local_estimates.saving.save_local_estimates import save_local_estimates
from topollm.analysis.twonn.truncate_prepared_data import truncate_prepared_data
from topollm.config_classes.main_config import MainConfig
from topollm.embeddings_data_prep.prepared_data_containers import PreparedData
from topollm.embeddings_data_prep.save_prepared_data import load_prepared_data
from topollm.logging.log_array_info import log_array_info
from topollm.logging.log_dataframe_info import log_dataframe_info
from topollm.path_management.embeddings.factory import get_embeddings_path_manager
from topollm.typing.enums import Verbosity

if TYPE_CHECKING:
    from topollm.analysis.local_estimates.filter.protocol import LocalEstimatesFilter

default_device = torch.device("cpu")
default_logger = logging.getLogger(__name__)


def twonn_worker(
    main_config: MainConfig,
    device: torch.device = default_device,  # noqa: ARG001 - placeholder for future use
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Prepare the embedding data of a model and its metadata for further analysis."""
    embeddings_path_manager = get_embeddings_path_manager(
        main_config=main_config,
        logger=logger,
    )

    # # # #
    # Load the prepared data.
    # Logging of the loaded data is handled in the loading function.
    prepared_data: PreparedData = load_prepared_data(
        embeddings_path_manager=embeddings_path_manager,
        verbosity=verbosity,
        logger=logger,
    )

    # # # #
    if verbosity >= Verbosity.NORMAL:
        logger.info("Filtering local estimates and truncating to first vectors ...")

    local_estimates_filter: LocalEstimatesFilter = get_local_estimates_filter(
        local_estimates_filtering_config=main_config.local_estimates.filtering,
        verbosity=verbosity,
        logger=logger,
    )

    # Filter the array, for example, by potentially removing zero vectors
    prepared_data_filtered: PreparedData = local_estimates_filter.filter_data(
        prepared_data=prepared_data,
    )

    # Restrict to the first `local_estimates_sample_size` samples
    local_estimates_sample_size = main_config.local_estimates.filtering.num_samples
    prepared_data_filtered_truncated: PreparedData = truncate_prepared_data(
        prepared_data=prepared_data_filtered,
        local_estimates_sample_size=local_estimates_sample_size,
    )

    array_for_estimator = prepared_data_filtered_truncated.array

    if verbosity >= Verbosity.NORMAL:
        log_array_info(
            array_for_estimator,
            array_name="array_for_estimator",
            log_array_size=True,
            log_row_l2_norms=True,
            logger=logger,
        )

    if verbosity >= Verbosity.NORMAL:
        logger.info("Filtering local estimates and truncating to first vectors DONE")

    # # # #
    # Local estimates computation

    # provide number of jobs for the computation
    results_array_np = run_local_estimates_computation(
        array_for_estimator=array_for_estimator,
        verbosity=verbosity,
        logger=logger,
    )

    # # # #
    # Save the results
    local_estimates_container = LocalEstimatesContainer(
        results_array_np=results_array_np,
        results_meta_frame=prepared_data_filtered_truncated.meta_df,
    )

    save_local_estimates(
        embeddings_path_manager=embeddings_path_manager,
        local_estimates_container=local_estimates_container,
        verbosity=verbosity,
        logger=logger,
    )

    # # # #
    # Create plots
    if main_config.feature_flags.analysis.create_plots_in_local_estimates_worker:
        tsne_array = create_projected_data(
            array=prepared_data_filtered_truncated.array,
        )

        # TODO: Get a proper path from the embeddings path manager
        tsne_df = create_projection_plot(
            tsne_result=tsne_array,
            meta_df=prepared_data_filtered_truncated.meta_df,
            output_folder=pathlib.Path("/Users/ruppik/git-source/Topo_LLM/data/saved_plots/local_estimates_projection"),
            verbosity=verbosity,
            logger=logger,
        )

        pass  # for setting a breakpoint here


def create_projection_plot(
    tsne_result: np.ndarray,
    meta_df: pd.DataFrame,
    output_folder: os.PathLike,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> pd.DataFrame:
    """Create a plot annotated with the metadata and save it.

    Args:
    ----
        tsne_result:
            The t-SNE result array, its coordinates are used for the plot.
        meta_df:
            The metadata DataFrame, used for annotating the points in the plot.
        output_folder:
            The output folder for the plot files.
            Will be created if it does not exist, and the plot will be saved as HTML and PDF.

    Returns:
    -------
        tsne_df:
            The DataFrame used for the plot, including the t-SNE results and the metadata
            (with truncated elements for better display).

    """
    tsne_df = pd.DataFrame(
        tsne_result,
        columns=[
            "TSNE-1",
            "TSNE-2",
        ],
    )
    tsne_df = pd.concat(
        [
            tsne_df,
            meta_df.reset_index(),
        ],
        axis=1,
    )

    # # # #
    # For better display in the plot, we truncate certain elements:
    # in the 'tokens_list' column to a maximum of 10 elements
    tsne_df["tokens_list"] = tsne_df["tokens_list"].apply(
        lambda x: x[:20],
    )
    # in the concatenated_tokens column to a maximum of 100 characters
    tsne_df["concatenated_tokens"] = tsne_df["concatenated_tokens"].apply(
        lambda x: x[:100],
    )

    fig = px.scatter(
        tsne_df,
        x="TSNE-1",
        y="TSNE-2",
        text="token_name",
        hover_data={
            "token_id": True,
            "sentence_idx": True,
            "subsample_idx": True,
            "token_name": True,
            "tokens_list": True,
            "concatenated_tokens": True,
        },
    )

    pathlib.Path(output_folder).mkdir(
        parents=True,
        exist_ok=True,
    )

    html_file = pathlib.Path(
        output_folder,
        "tsne_plot.html",
    )
    pdf_file = pathlib.Path(
        output_folder,
        "tsne_plot.pdf",
    )

    pio.write_html(
        fig=fig,
        file=html_file,
        auto_open=False,
    )
    pio.write_image(
        fig=fig,
        file=pdf_file,
        format="pdf",
        width=2400,
        height=1600,
    )

    if verbosity >= Verbosity.NORMAL:
        log_dataframe_info(
            tsne_df,
            df_name="tsne_df",
            logger=logger,
        )
        logger.info(
            f"Plot saved as HTML to {html_file = }",  # noqa: G004 - low overhead
        )
        logger.info(
            f"Plot saved as PDF to {pdf_file = }",  # noqa: G004 - low overhead
        )

    return tsne_df


def create_projected_data(
    array: np.ndarray,
    pca_n_components: int | None = 50,
    tsne_n_components: int = 2,
    tsne_random_state: int = 42,
) -> np.ndarray:
    """Create a projection of the input data using PCA and t-SNE."""
    # Apply PCA if requested
    if pca_n_components:
        pca = sklearn.decomposition.PCA(
            n_components=pca_n_components,
        )
        array = pca.fit_transform(
            array,
        )

    # Apply t-SNE
    tsne = sklearn.manifold.TSNE(
        n_components=tsne_n_components,
        random_state=tsne_random_state,
    )
    tsne_array = tsne.fit_transform(
        array,
    )

    return tsne_array


def run_local_estimates_computation(
    array_for_estimator: np.ndarray,
    discard_fraction: float = 0.1,
    n_jobs: int = 1,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> np.ndarray:
    """Run the local estimates computation."""
    # Number of neighbors which are used for the computation
    n_neighbors = round(len(array_for_estimator) * 0.8)
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            f"{n_neighbors = }",  # noqa: G004 - low overhead
        )

    estimator = skdim.id.TwoNN(
        discard_fraction=discard_fraction,
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info("Calling estimator.fit_pw() ...")

    fitted_estimator = estimator.fit_pw(
        X=array_for_estimator,
        precomputed_knn=None,
        smooth=False,
        n_neighbors=n_neighbors,
        n_jobs=n_jobs,
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info("Calling estimator.fit_pw() DONE")

    results_array = list(fitted_estimator.dimension_pw_)

    results_array_np = np.array(
        results_array,
    )

    if verbosity >= Verbosity.NORMAL:
        log_array_info(
            results_array_np,
            array_name="results_array_np",
            log_array_size=True,
            log_row_l2_norms=False,  # Note: This is a one-dimensional array, so the l2-norms are not meaningful
            logger=logger,
        )

        # Log the mean and standard deviation of the local estimates
        logger.info(
            f"Mean of local estimates: {results_array_np.mean() = }",  # noqa: G004 - low overhead
        )
        logger.info(
            f"Standard deviation of local estimates: {results_array_np.std() = }",  # noqa: G004 - low overhead
        )

    return results_array_np
