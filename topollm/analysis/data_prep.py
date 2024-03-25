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

# This is a script to prepare the embedding data of a
# model and its corresopnding fine-tuned variant.
# The script outputs two numpy arrays of subsamples
# of the respective arrays that correspond to the
# embeddings of the base model and the fine-tuned model,
# respectively.
# The arrays are stored in the directory where this
# script is executed.
# Since paddings are removed from the embeddings,
# the resulting size of the arrays will usually be
# significantly lower than the specified sample size
# (often ~5% of the specified size).

# third party imports
import pathlib
import zarr
import numpy as np
import os
import pickle
import pandas as pd

# choose dataset name
#dataset_name = "data-multiwoz21_split-test_ctxt-dataset_entry"
#dataset_name = "data-xsum_split-train_ctxt-dataset_entry"
#dataset_name = "data-wikitext_split-train_ctxt-dataset_entry"
#dataset_name = "data-xsum_split-train_ctxt-dataset_entry"
dataset_name = "data-iclr_2024_submissions_split-train_ctxt-dataset_entry"

# choose model name
#model_name = "model-roberta-base_mask-no_masking"
model_name = "model-roberta-base_mask-no_masking"

# choose model name of the finetuned model
#model_name_finetuned = "model-roberta-base_finetuned-on-multiwoz21-train_mask-no_masking"
model_name_finetuned = "model-roberta-base_finetuned-on-multiwoz21-train_context-dialogue_mask-no_masking"

# choose sample size of the arrays
sample_size = 30000

# local path
local_path = "Documents/LLM-Analysis"
#local_path = "git-source"

# potentially adapt paths
array_path = pathlib.Path(
    pathlib.Path.home(),
    local_path,
    "Topo_LLM",
    "data",
    "embeddings",
    "arrays",
    dataset_name,
    "lvl-token",
    "add-prefix-space-False_max-len-512",
    model_name,
    "layer-[-1]_agg-mean",
    "norm-None",
    "array_dir",
)

array_path_finetuned = pathlib.Path(
    pathlib.Path.home(),
    local_path,
    "Topo_LLM",
    "data",
    "embeddings",
    "arrays",
    dataset_name,
    "lvl-token",
    "add-prefix-space-False_max-len-512",
    model_name_finetuned,
    "layer-[-1]_agg-mean",
    "norm-None",
    "array_dir",
)

meta_path = pathlib.Path(
    pathlib.Path.home(),
    local_path,
    "Topo_LLM",
    "data",
    "embeddings",
    "metadata",
    dataset_name,
    "lvl-token",
    "add-prefix-space-False_max-len-512",
    model_name,
    "layer-[-1]_agg-mean",
    "norm-None",
    "metadata_dir",
    "pickle_chunked_metadata_storage",
)

meta_path_finetuned = pathlib.Path(
    pathlib.Path.home(),
    local_path,
    "Topo_LLM",
    "data",
    "embeddings",
    "metadata",
    dataset_name,
    "lvl-token",
    "add-prefix-space-False_max-len-512",
    model_name_finetuned,
    "layer-[-1]_agg-mean",
    "norm-None",
    "metadata_dir",
    "pickle_chunked_metadata_storage",
)

print(array_path)

if not array_path.exists():
    raise FileNotFoundError(f"{array_path = } does not exist.")

print(array_path_finetuned)

if not array_path_finetuned.exists():
    raise FileNotFoundError(f"{array_path_finetuned = } does not exist.")

print(meta_path)

if not meta_path.exists():
    raise FileNotFoundError(f"{meta_path = } does not exist.")

print(meta_path_finetuned)

if not meta_path_finetuned.exists():
    raise FileNotFoundError(f"{meta_path_finetuned = } does not exist.")

array = zarr.open(
    store=array_path,  # type: ignore
    mode="r",
)

array_finetuned = zarr.open(
    store=array_path_finetuned,  # type: ignore
    mode="r",
)

arr = np.array(array)
arr = arr.reshape(arr.shape[0]*arr.shape[1],arr.shape[2])

arr_finetuned = np.array(array_finetuned)
arr_finetuned = arr_finetuned.reshape(arr_finetuned.shape[0]*arr_finetuned.shape[1],arr_finetuned.shape[2])

idx = np.random.choice(range(len(arr)),replace=False,size=sample_size)

arr = arr[idx]
arr_finetuned = arr_finetuned[idx]

# function to load pickle files stored in the respective directory
def load_pickle_files(directory):
    data = []
    chunk_list = []
    for i in range(len(os.listdir(meta_path))):
        if i >=10000:
            chunk_list.append('chunk_'+str(i)+'.pkl')
        elif i >=1000:
            chunk_list.append('chunk_0'+str(i)+'.pkl')
        elif i >=100:
            chunk_list.append('chunk_00'+str(i)+'.pkl')
        elif i >=10:
            chunk_list.append('chunk_000'+str(i)+'.pkl')
        else:
            chunk_list.append('chunk_0000'+str(i)+'.pkl')
    for filename in chunk_list:
        if filename.endswith(".pkl"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "rb") as f:
                chunk = pickle.load(f)
                data.append(chunk)
    return data


loaded_data = load_pickle_files(meta_path)
loaded_data_finetuned = load_pickle_files(meta_path_finetuned)
print("Loaded pickle files:", loaded_data)

input_ids = []
for i in range(len(loaded_data)):
    input_ids.append(loaded_data[i]['input_ids'].tolist())

stacked_meta = np.vstack(input_ids)
stacked_meta = stacked_meta.reshape(stacked_meta.shape[0] *
                                    stacked_meta.shape[1])

input_ids_finetuned = []
for i in range(len(loaded_data_finetuned)):
    input_ids_finetuned.append(loaded_data_finetuned[i]['input_ids'].numpy())

stacked_meta_finetuned = np.vstack(input_ids_finetuned)
stacked_meta_finetuned = stacked_meta_finetuned.reshape(stacked_meta_finetuned.shape[0] *
                                                        stacked_meta_finetuned.shape[1])

stacked_meta_sub = stacked_meta[idx]
stacked_meta_finetuned_sub = stacked_meta_finetuned[idx]

df = pd.DataFrame({'arr': list(arr), 'meta': list(stacked_meta_sub)})
arr_no_pad = np.array(list(df[(df['meta'] != 2) & (df['meta'] != 1)].arr))

df_finetuned = pd.DataFrame({'arr': list(arr_finetuned), 'meta': list(stacked_meta_finetuned_sub)})
arr_no_pad_finetuned = np.array(list(df_finetuned[(df['meta'] != 2) & (df['meta'] != 1)].arr))

print(arr_no_pad[0])
print(arr_no_pad_finetuned[0])

np.save('sample_embeddings_'+dataset_name+'_'+model_name+'_no_paddings',arr_no_pad)
np.save('sample_embeddings_'+dataset_name+'_'+model_name_finetuned+'_no_paddings',arr_no_pad_finetuned)