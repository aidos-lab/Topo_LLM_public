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

########################################################
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA

# provide names of numpy array to be used for dimension estimation
data_name = "sample_embeddings_data-multiwoz21_split-test_ctxt-dataset_entry_model-roberta-base_mask-no_masking_no_paddings.npy"
data_name_finetuned = "sample_embeddings_data-multiwoz21_split-test_ctxt-dataset_entry_model-roberta-base_finetuned-on-multiwoz21-train_mask-no_masking_no_paddings.npy"

arr_no_pad = np.load(data_name)
arr_no_pad_finetuned = np.load(data_name_finetuned)

dataset = pd.DataFrame({f'Column{i+1}': arr_no_pad[:,i] for i in range(arr_no_pad.shape[1])})
dataset['class'] = 'base'

dataset_finetuned = pd.DataFrame({f'Column{i+1}': arr_no_pad_finetuned[:,i] for i in
                                  range(arr_no_pad_finetuned.shape[1])})
dataset_finetuned['class'] = 'finetuned'

df = pd.concat((dataset,dataset_finetuned))

df.reset_index(inplace=True)
df.drop(columns='index',inplace=True)

features = list(df.columns)[:-1]

pca = PCA(n_components=10)
components = pca.fit_transform(df[features])
labels = {
    str(i): f"PC{i+1}"
    for i, var in enumerate(pca.explained_variance_ratio_ * 100)
}

fig = px.scatter_matrix(
    components,
    labels=labels,
    dimensions=range(10),
    color=df["class"]
)
fig.update_traces(diagonal_visible=False)
fig.show()