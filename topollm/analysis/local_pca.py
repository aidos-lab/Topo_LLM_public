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

# This is a script to calculate and store lPCA dimension
# estimates for two given arrays for a comparison
# of embeddings of a base model and a corresponding
# fine-tuned variant. To obtain these arrays, the
# `data_prep.py` may be used.

# third party imports
import os
import pathlib

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import skdim


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

    layer_1 = "layer-" + str(cfg.layer) + "_agg-mean"
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

    arr_no_pad = np.load(path_1)
    arr_no_pad_finetuned = np.load(path_2)

    np.random.seed(2)
    sample_size = 1500
    sample_size = min(sample_size, arr_no_pad.shape[0], arr_no_pad_finetuned.shape[0])

    arr_no_pad = arr_no_pad[:sample_size]
    arr_no_pad_finetuned = arr_no_pad_finetuned[:sample_size]

    # provide number of jobs for the computation
    n_jobs = 1

    # provide number of neighbors which are used for the computation
    n_neighbors = 100

    lPCA = skdim.id.lPCA().fit_pw(arr_no_pad, n_neighbors=n_neighbors, n_jobs=n_jobs)

    lPCA_finetuned = skdim.id.lPCA().fit_pw(arr_no_pad_finetuned, n_neighbors=n_neighbors, n_jobs=n_jobs)

    dim_frame = pd.DataFrame({"lpca_finetuned": list(lPCA_finetuned.dimension_pw_), "lpca": list(lPCA.dimension_pw_)})

    print(dim_frame.corr())

    plt.ioff()
    scatter_plot = sns.scatterplot(x=list(lPCA.dimension_pw_), y=list(lPCA_finetuned.dimension_pw_))
    scatter_fig = scatter_plot.get_figure()

    # use savefig function to save the plot and give
    # a desired name to the plot.

    file_name = ""
    file_name += str(cfg.data_name) + "_"
    file_name += str(cfg.model_1) + "_"
    file_name += str(cfg.model_2)

    save_path = "../../data/analysis/lpca/" + str(cfg.embedding_level_1) + "/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_name = save_path + file_name + str(len(arr_no_pad)) + "_samples_" + str(cfg.layer) + ".pkl"
    scatter_fig.savefig(save_name + ".png")
    dim_frame.to_pickle(save_name)

    # plt.show()
    plt.close()
    return None


if __name__ == "__main__":
    main()  # type: ignore
