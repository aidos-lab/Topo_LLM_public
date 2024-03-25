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

# This is a script to calculate and store lPCA dimension
# estimates for two given arrays for a comparison
# of embeddings of a base model and a corresponding
# fine-tuned variant. To obtain these arrays, the
# `data_prep.py` may be used.

# third party imports
import numpy as np
import pandas as pd
import skdim
import matplotlib.pyplot as plt
import seaborn as sns

# provide names of numpy array to be used for dimension estimation
data_name = "sample_embeddings_data-iclr_2024_submissions_split-train_ctxt-dataset_entry_model-roberta-base_mask-no_masking_no_paddings.npy"
data_name_finetuned = "sample_embeddings_data-iclr_2024_submissions_split-train_ctxt-dataset_entry_model-roberta-base_finetuned-on-multiwoz21-train_context-dialogue_mask-no_masking_no_paddings.npy"

# provide number of jobs for the computation
n_jobs = 1

# provide number of neighbors which are used for the computation
n_neighbors = 100

arr_no_pad = np.load(data_name)
arr_no_pad_finetuned = np.load(data_name_finetuned)

lPCA = skdim.id.lPCA().fit_pw(arr_no_pad,
                              n_neighbors = n_neighbors,
                              n_jobs = n_jobs)


lPCA_finetuned = skdim.id.lPCA().fit_pw(arr_no_pad_finetuned,
                              n_neighbors = n_neighbors,
                              n_jobs = n_jobs)

dim_frame = pd.DataFrame({
                         'lpca_finetuned':list(lPCA_finetuned.dimension_pw_),
                         'lpca':list(lPCA.dimension_pw_)
                         })

print(dim_frame.corr())

scatter_plot = sns.scatterplot(x = list(lPCA.dimension_pw_),y = list(lPCA_finetuned.dimension_pw_))
scatter_fig = scatter_plot.get_figure()

# use savefig function to save the plot and give
# a desired name to the plot.
save_name = 'lpca/lpca'+data_name[:-4]+'_roberta_vs_finetuned(multiwoz)'

scatter_fig.savefig(save_name+'.png')
dim_frame.to_pickle(save_name)

plt.show()

