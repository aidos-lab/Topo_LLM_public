# Copyright 2024
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Ruppik (ruppik@hhu.de)
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

import pathlib

from pydantic import Field

from topollm.config_classes.config_base_model import ConfigBaseModel


class SubmitPipelineJobsConfig(ConfigBaseModel):
    """Config for submitting finetuning jobs."""

    data_list: list[str] = Field(
        default_factory=list,
    )
    language_model_list: list[str] = Field(
        default_factory=list,
    )
    layer_indices_list: list[str] = Field(
        default_factory=list,
    )
    checkpoint_no_list: list[int] = Field(
        default_factory=list,
    )
    data_number_of_samples_list: list[int] = Field(
        default_factory=list,
    )
    embeddings_data_prep_num_samples_list: list[int] = Field(
        default_factory=list,
    )

    pipeline_python_script_relative_path: pathlib.Path = pathlib.Path(
        "topollm",
        "pipeline_scripts",
        "run_pipeline_embeddings_data_prep_local_estimate.py",
    )

    wandb_project: str = "Topo_LLM_submit_jobs_via_hydra_debug"
