# Copyright 2024-2025
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Ruppik (mail@ruppik.net)
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

"""Parameters for the analysis of results."""

from pydantic import Field

from topollm.config_classes.config_base_model import ConfigBaseModel
from topollm.config_classes.constants import KV_SEP, NAME_PREFIXES


class InvestigateDistancesConfig(ConfigBaseModel):
    """Parameters for the analysis of distances."""

    array_truncation_size: int = Field(
        default=5_000,
        title="Array truncation size.",
        description="The size to which arrays are truncated for the analysis functions.",
    )

    def get_config_description(
        self,
    ) -> str:
        """Return a description of the configuration."""
        description: str = f"{NAME_PREFIXES['array_truncation_size']}"
        description += KV_SEP
        description += f"{str(object=self.array_truncation_size)}"

        return description


class PlottingConfig(ConfigBaseModel):
    """Parameters for plotting configuration."""

    patterns_to_iterate_over: list[str] = Field(
        default=[],
        title="Patterns to iterate over.",
        description="The patterns to iterate over for plotting.",
    )


class TaskPerformanceAnalysisConfig(ConfigBaseModel):
    """Parameters for the analysis of task performance."""

    plotting: PlottingConfig = Field(
        default_factory=PlottingConfig,
        title="Plotting configuration.",
        description="The configuration for specifying parameters in the plotting.",
    )


class WandbExportConfig(ConfigBaseModel):
    """Config to specify what needs to be modified to pick out the comparison data."""

    wandb_id: str = Field(
        default="debug_id",
        title="WandB ID.",
        description="The WandB ID for the project.",
    )

    project_name: str = Field(
        default="debug_name",
        title="Project name.",
        description="The name of the project in WandB.",
    )

    samples: int = Field(
        default=5_000,
        title="Samples.",
        description="The number of samples to use for the analysis.",
    )

    use_saved_concatenated_df: bool = Field(
        default=False,
        title="Use saved concatenated DataFrame.",
        description="Whether to use the saved concatenated DataFrame or not.",
    )


class AnalysisConfig(ConfigBaseModel):
    """Parameters for the analysis of results."""

    investigate_distances: InvestigateDistancesConfig = Field(
        default_factory=InvestigateDistancesConfig,
        title="Distances investigations configuration.",
        description="The configuration for specifying parameters in the distances investigations.",
    )

    task_performance_analysis: TaskPerformanceAnalysisConfig = Field(
        default_factory=TaskPerformanceAnalysisConfig,
        title="Task performance configuration.",
        description="The configuration for specifying parameters in the task performance analysis.",
    )

    wandb_export: WandbExportConfig = Field(
        default_factory=WandbExportConfig,
        title="WandB export configuration.",
        description="The configuration for specifying parameters in the WandB export.",
    )
