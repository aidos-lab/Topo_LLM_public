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

    similarities = []
    for A, B in zip(arr_no_pad, arr_no_pad_finetuned):
        dot_product = np.dot(A, B)
        magnitude_A = np.linalg.norm(A)
        magnitude_B = np.linalg.norm(B)
        cosine_similarity = dot_product / (magnitude_A * magnitude_B)
        print(f"Cosine Similarity using NumPy: {cosine_similarity}")
        similarities.append(cosine_similarity)

    similarities = np.array(similarities)
    similarities = similarities[~np.isnan(similarities)]

    print(sum(similarities) / len(similarities))

    file_name = ""
    if cfg.data_name_1 == cfg.data_name_2:
        file_name += str(cfg.data_name_1) + "_"
    if cfg.model_1 == cfg.model_2:
        file_name += str(cfg.model_1) + "_"
    names = set([name[:-1] for name in cfg.keys()])
    for name in names:
        if cfg[name + str(1)] != cfg[name + str(2)]:
            additional_string = str(cfg[name + str(1)]) + "_vs_" + str(cfg[name + str(2)] + "_")
            file_name += additional_string

    save_path = "../../data/analysis/cosine/" + str(cfg.embedding_level_1) + "/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_name = save_path + file_name + str(len(arr_no_pad)) + "_samples"
    np.save(save_name, similarities)
    return None


if __name__ == "__main__":
    main()  # type: ignore
