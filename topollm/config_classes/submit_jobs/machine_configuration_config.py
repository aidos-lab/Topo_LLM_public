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

import os

from pydantic import Field

from topollm.config_classes.config_base_model import ConfigBaseModel
from topollm.typing.enums import JobRunMode, Verbosity

USER = os.getenv(
    "USER",
)


class MachineConfigurationConfig(ConfigBaseModel):
    """Config for machine configuration.

    Note: Since we register this with the hydra config,
    we do not want to add methods to this class.
    """

    accelerator_model: str = "rtx6000"
    queue: str = "CUDA"
    ncpus: int = 2
    ngpus: int = 1
    memory_gb: int = 32
    walltime: str = "08:00:00"
    submit_job_hilbert_command: list[str] = Field(
        default_factory=lambda: [
            "python3",
            f"/gpfs/project/{USER}/.usr_tls/tools/submit_job.py",
        ],
    )
    run_job_locally_command: list[str] = Field(
        default_factory=lambda: [
            "python3",
        ],
    )

    job_run_mode: JobRunMode = JobRunMode.HHU_HILBERT
    dry_run: bool = False


def get_machine_configuration_args_list(
    machine_configuration_config: MachineConfigurationConfig,
) -> list[str]:
    """Get machine configuration args list.

    Note that we have this as a separate function and not as a method of the MachineConfigurationConfig dataclass,
    since we register this dataclass with the hydra config.
    """
    machine_configuration_args_list: list[str] = [
        "--ncpus",
        str(machine_configuration_config.ncpus),
        "--memory",
        str(machine_configuration_config.memory_gb),
        "--ngpus",
        str(machine_configuration_config.ngpus),
        "--accelerator_model",
        machine_configuration_config.accelerator_model,
        "--queue",
        machine_configuration_config.queue,
        "--walltime",
        machine_configuration_config.walltime,
    ]

    return machine_configuration_args_list
