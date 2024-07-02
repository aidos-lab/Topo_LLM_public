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

"""Configuration class for feature flags."""

from pydantic import Field

from topollm.config_classes.config_base_model import ConfigBaseModel


class EmbeddingsDataPrepFeatureFlagsConfig(ConfigBaseModel):
    """Feature flags for the embeddings data preparation process."""

    write_sentences_to_meta: bool = Field(
        default=True,
    )


class FinetuningFeatureFlagsConfig(ConfigBaseModel):
    """Feature flags for the finetuning process."""

    do_create_finetuned_language_model_config: bool = Field(
        default=True,
        title="Create finetuned language model config.",
        description="Whether to create the finetuned language model config after the finetuning process.",
    )

    skip_finetuning: bool = Field(
        default=False,
        title="Skip finetuning.",
        description="Whether to skip the finetuning process.",
    )

    use_wandb: bool = Field(
        default=True,
        title="Use wandb.",
        description="Whether to use wandb for logging.",
    )


class FeatureFlagsConfig(ConfigBaseModel):
    """Configurations for specifying feature flags."""

    embeddings_data_prep: EmbeddingsDataPrepFeatureFlagsConfig = Field(
        default_factory=EmbeddingsDataPrepFeatureFlagsConfig,
        title="Embeddings data preparation feature flags.",
        description="Feature flags for the embeddings data preparation process.",
    )

    finetuning: FinetuningFeatureFlagsConfig = Field(
        default_factory=FinetuningFeatureFlagsConfig,
        title="Finetuning feature flags.",
        description="Feature flags for the finetuning process.",
    )
