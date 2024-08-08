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

########################################################
import pathlib

import hydra
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA


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

    path_1 = pathlib.Path(
        "..",
        "..",
        "data",
        "analysis",
        "prepared",
        cfg.data_name_1,
        cfg.level_1,
        cfg.prefix_1,
        cfg.model_1,
        cfg.layer_1,
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
        cfg.data_name_2,
        cfg.level_2,
        cfg.prefix_2,
        cfg.model_2,
        cfg.layer_2,
        cfg.norm_2,
        cfg.array_dir_2,
        array_name_2,
    )

    arr_no_pad = np.load(path_1)
    arr_no_pad_finetuned = np.load(path_2)

    dataset = pd.DataFrame({f"Column{i+1}": arr_no_pad[:, i] for i in range(arr_no_pad.shape[1])})
    dataset["class"] = "base"

    dataset_finetuned = pd.DataFrame(
        {f"Column{i+1}": arr_no_pad_finetuned[:, i] for i in range(arr_no_pad_finetuned.shape[1])}
    )
    dataset_finetuned["class"] = "finetuned"

    df = pd.concat((dataset, dataset_finetuned))

    df.reset_index(inplace=True)
    df.drop(columns="index", inplace=True)

    features = list(df.columns)[:-1]

    pca = PCA(n_components=10)
    components = pca.fit_transform(df[features])
    labels = {str(i): f"PC{i+1}" for i, var in enumerate(pca.explained_variance_ratio_ * 100)}

    fig = px.scatter_matrix(components, labels=labels, dimensions=range(10), color=df["class"])
    fig.update_traces(diagonal_visible=False)
    fig.show()

    return None


if __name__ == "__main__":
    main()  # type: ignore
