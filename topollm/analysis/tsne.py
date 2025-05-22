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

########################################################

# This is a script to calculate and store lPCA dimension
# estimates for two given arrays for a comparison
# of embeddings of a base model and a corresponding
# fine-tuned variant. To obtain these arrays, the
# `data_prep.py` may be used.

# third party imports
import pathlib

import hydra
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import skdim
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from sklearn.neighbors import KernelDensity


@hydra.main(
    config_path="../../configs/analysis",
    config_name="comparison",
    version_base="1.2",
)
def main(cfg):
    array_name_1 = (
        "embeddings_" + str(cfg.embedding_level_1) + "_" + str(cfg.samples_1) + "_samples_paddings_removed.npy"
    )
    array_name_2 = (
        "embeddings_" + str(cfg.embedding_level_2) + "_" + str(cfg.samples_2) + "_samples_paddings_removed.npy"
    )

    layer_1 = "layer-[" + str(cfg.layer) + "]_agg-mean"
    layer_2 = layer_1
    path_1 = pathlib.Path(
        "..",
        "..",
        "data",
        "analysis",
        "prepared",
        cfg.data_name,
        cfg.level_1,
        cfg.prefix_1,
        cfg.model_1,
        layer_1,
        cfg.norm_1,
        cfg.array_dir_1,
        array_name_1,
    )

    path_2 = pathlib.Path(
        "..",
        "..",
        "data",
        "analysis",
        "prepared",
        cfg.data_name,
        cfg.level_2,
        cfg.prefix_2,
        cfg.model_2,
        layer_2,
        cfg.norm_2,
        cfg.array_dir_2,
        array_name_2,
    )
    meta_path_1 = str(path_1)[:-4] + "_meta.pkl"
    meta_path_2 = str(path_2)[:-4] + "_meta.pkl"

    tokens_1 = list(pd.read_pickle(meta_path_1).token_name)
    tokens_2 = list(pd.read_pickle(meta_path_2).token_name)

    tokens_1 = [x[1:] for x in tokens_1]
    tokens_2 = [x[1:] for x in tokens_2]

    arr_no_pad = np.load(path_1)
    arr_no_pad_finetuned = np.load(path_2)

    # provide number of components of the projection
    n_components = 2

    # choose between vis_type "comparison", "twonn", "density", "iso", "lpca"
    vis_type = ""

    dataset = pd.DataFrame({f"Column{i + 1}": arr_no_pad[:, i] for i in range(arr_no_pad.shape[1])})
    dataset["class"] = "base"

    dataset_finetuned = pd.DataFrame(
        {f"Column{i + 1}": arr_no_pad_finetuned[:, i] for i in range(arr_no_pad_finetuned.shape[1])}
    )
    dataset_finetuned["class"] = "finetuned"

    df = pd.concat((dataset, dataset_finetuned))

    df.reset_index(inplace=True)
    df.drop(columns="index", inplace=True)

    tsne = TSNE(n_components=n_components, random_state=0)
    embedding_concat = tsne.fit_transform(df.iloc[:, :-1])

    idx = len(dataset)

    embedding = embedding_concat[:idx]
    embedding_finetuned = embedding_concat[idx:]

    embedding = embedding[:2000]
    embedding_finetuned = embedding_finetuned[:2000]
    if n_components == 3:
        fig = px.scatter_3d(x=embedding[:, 0], y=embedding[:, 1], z=embedding[:, 2], title="t-SNE embedding")
        fig.add_trace(
            px.scatter_3d(x=embedding_finetuned[:, 0], y=embedding_finetuned[:, 1], z=embedding_finetuned[:, 2]).data[0]
        )
        fig.show()
    elif n_components == 2:
        title_name = ""
        if cfg.model_1 != cfg.model_2:
            title_name += str(cfg.model_1) + " vs. " + str(cfg.model_2) + "_"
        elif cfg.model_1 == cfg.model_2:
            title_name += str(cfg.model_1) + "_"
        title_name += str(cfg.data_name) + " "

        if vis_type == "comparison":
            # Create a Plotly scatter plot with text annotations for tokens
            trace1 = go.Scatter(
                x=embedding[:, 0],
                y=embedding[:, 1],
                mode="markers+text",
                marker=dict(size=5, color="rgba(255, 127, 14, 0.7)"),
                text=tokens_1,
                textposition="bottom center",
                name="base",
            )

            trace2 = go.Scatter(
                x=embedding_finetuned[:, 0],
                y=embedding_finetuned[:, 1],
                mode="markers+text",
                marker=dict(size=5, color="rgba(255, 127, 14, 0.7)"),
                text=tokens_2,
                textposition="bottom center",
                name="finetuned",
            )

            layout = go.Layout(
                title="t-SNE Projection of Embeddings with Tokens, " + title_name,
                xaxis=dict(title="t-SNE Dimension 1"),
                yaxis=dict(title="t-SNE Dimension 2"),
                hovermode="closest",
                plot_bgcolor="rgba(0,0,0,0)",  # Set plot background color to transparent
            )

            fig = go.Figure(data=[trace1, trace2], layout=layout)
            fig.update_layout(template="plotly_white")  # Set plot template to white background
            fig.update_traces(
                marker=dict(line=dict(width=0.5, color="rgba(255, 255, 255, 0.7)"))
            )  # Set marker line color and width
            fig.update_layout(
                xaxis=dict(showgrid=False, zeroline=False), yaxis=dict(showgrid=False, zeroline=False)
            )  # Hide gridlines and zero lines
            fig.show()
        elif vis_type == "iso":
            clf = IsolationForest(random_state=0).fit(arr_no_pad)
            scores = clf.score_samples(arr_no_pad)
            trace1 = go.Scatter(
                x=embedding[:, 0],
                y=embedding[:, 1],
                mode="markers+text",
                marker=dict(
                    size=5, color=list(scores), colorscale="Viridis", showscale=True, colorbar=dict(title="Iso scores")
                ),
                text=tokens_1,
                textposition="bottom center",
                name="base",
            )

            layout = go.Layout(
                title="t-SNE Projection of Embeddings with Tokens, " + title_name,
                xaxis=dict(title="t-SNE Dimension 1"),
                yaxis=dict(title="t-SNE Dimension 2"),
                hovermode="closest",
                plot_bgcolor="rgba(0,0,0,0)",  # Set plot background color to transparent
            )

            fig = go.Figure(data=[trace1], layout=layout)
            fig.update_layout(template="plotly_white")  # Set plot template to white background
            fig.update_traces(
                marker=dict(line=dict(width=0.5, color="rgba(255, 255, 255, 0.7)"))
            )  # Set marker line color and width
            fig.update_layout(
                xaxis=dict(showgrid=False, zeroline=False), yaxis=dict(showgrid=False, zeroline=False)
            )  # Hide gridlines and zero lines
            fig.show()
        elif vis_type == "density":
            kde = KernelDensity(kernel="gaussian", bandwidth=0.2).fit(arr_no_pad)
            density = kde.score_samples(arr_no_pad)
            trace1 = go.Scatter(
                x=embedding[:, 0],
                y=embedding[:, 1],
                mode="markers+text",
                marker=dict(
                    size=5, color=list(density), colorscale="Viridis", showscale=True, colorbar=dict(title="density")
                ),
                text=tokens_1,
                textposition="bottom center",
                name="base",
            )

            layout = go.Layout(
                title="t-SNE Projection of Embeddings with Tokens, " + title_name,
                xaxis=dict(title="t-SNE Dimension 1"),
                yaxis=dict(title="t-SNE Dimension 2"),
                hovermode="closest",
                plot_bgcolor="rgba(0,0,0,0)",  # Set plot background color to transparent
            )

            fig = go.Figure(data=[trace1], layout=layout)
            fig.update_layout(template="plotly_white")  # Set plot template to white background
            fig.update_traces(
                marker=dict(line=dict(width=0.5, color="rgba(255, 255, 255, 0.7)"))
            )  # Set marker line color and width
            fig.update_layout(
                xaxis=dict(showgrid=False, zeroline=False), yaxis=dict(showgrid=False, zeroline=False)
            )  # Hide gridlines and zero lines
            fig.show()

        elif vis_type == "twonn":
            n_jobs = 1

            # provide number of neighbors which are used for the computation
            n_neighbors = 300

            lPCA = skdim.id.TwoNN().fit_pw(arr_no_pad, n_neighbors=n_neighbors, n_jobs=n_jobs)

            trace1 = go.Scatter(
                x=embedding[:, 0],
                y=embedding[:, 1],
                mode="markers+text",
                marker=dict(
                    size=5,
                    color=list(lPCA.dimension_pw_),
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="TwoNN"),
                ),
                text=tokens_1,
                textposition="bottom center",
                name="base",
            )

            layout = go.Layout(
                title="t-SNE Projection of Embeddings with Tokens, " + title_name,
                xaxis=dict(title="t-SNE Dimension 1"),
                yaxis=dict(title="t-SNE Dimension 2"),
                hovermode="closest",
                plot_bgcolor="rgba(0,0,0,0)",  # Set plot background color to transparent
            )

            fig = go.Figure(data=[trace1], layout=layout)
            fig.update_layout(template="plotly_white")  # Set plot template to white background
            fig.update_traces(
                marker=dict(line=dict(width=0.5, color="rgba(255, 255, 255, 0.7)"))
            )  # Set marker line color and width
            fig.update_layout(
                xaxis=dict(showgrid=False, zeroline=False), yaxis=dict(showgrid=False, zeroline=False)
            )  # Hide gridlines and zero lines
            fig.show()
        else:
            n_jobs = 1

            # provide number of neighbors which are used for the computation
            n_neighbors = 100

            lPCA = skdim.id.lPCA().fit_pw(arr_no_pad, n_neighbors=n_neighbors, n_jobs=n_jobs)

            trace1 = go.Scatter(
                x=embedding[:, 0],
                y=embedding[:, 1],
                mode="markers+text",
                marker=dict(
                    size=5,
                    color=list(lPCA.dimension_pw_),
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Euclidicity"),
                ),
                text=tokens_1,
                textposition="bottom center",
                name="base",
            )

            layout = go.Layout(
                title="t-SNE Projection of Embeddings with Tokens, " + title_name,
                xaxis=dict(title="t-SNE Dimension 1"),
                yaxis=dict(title="t-SNE Dimension 2"),
                hovermode="closest",
                plot_bgcolor="rgba(0,0,0,0)",  # Set plot background color to transparent
            )

            fig = go.Figure(data=[trace1], layout=layout)
            fig.update_layout(template="plotly_white")  # Set plot template to white background
            fig.update_traces(
                marker=dict(line=dict(width=0.5, color="rgba(255, 255, 255, 0.7)"))
            )  # Set marker line color and width
            fig.update_layout(
                xaxis=dict(showgrid=False, zeroline=False), yaxis=dict(showgrid=False, zeroline=False)
            )  # Hide gridlines and zero lines
            fig.show()


if __name__ == "__main__":
    main()  # type: ignore
