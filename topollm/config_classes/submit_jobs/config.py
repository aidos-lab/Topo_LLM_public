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

from dataclasses import dataclass

from hydra.core.config_store import ConfigStore

from topollm.config_classes.submit_jobs.machine_configuration_config import MachineConfigurationConfig
from topollm.config_classes.submit_jobs.submit_finetuning_jobs_config import (
    SubmitFinetuningJobsConfig,
)
from topollm.config_classes.submit_jobs.submit_pipeline_jobs_config import SubmitPipelineJobsConfig


@dataclass
class Config:
    """Config for the main function."""

    machine_configuration: MachineConfigurationConfig
    submit_finetuning_jobs: SubmitFinetuningJobsConfig
    submit_pipeline_jobs: SubmitPipelineJobsConfig


# # # # # # # # # # # # # # # # # # # # # # # #
# Register the config classes with Hydra

cs = ConfigStore.instance()
cs.store(
    group="configs/submit_jobs/machine_configuration",
    name="base_machine_configuration_config",
    node=MachineConfigurationConfig,
)
cs.store(
    group="configs/submit_jobs/submit_finetuning_jobs",
    name="base_submit_finetuning_jobs_config",
    node=SubmitFinetuningJobsConfig,
)
cs.store(
    group="configs/submit_jobs/submit_pipeline_jobs",
    name="base_submit_pipeline_jobs_config",
    node=SubmitPipelineJobsConfig,
)

# Note: Do not register the TrainingScheduleConfig class as a separate config class,
# because we use an additional key for nesting so that they do not overwrite each other.
# https://hydra.cc/docs/patterns/select_multiple_configs_from_config_group/
cs.store(
    name="base_config",
    node=Config,
)
