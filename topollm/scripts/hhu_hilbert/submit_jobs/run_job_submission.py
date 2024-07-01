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

import logging
import pathlib

from topollm.config_classes.submit_jobs.machine_configuration_config import (
    MachineConfigurationConfig,
    get_machine_configuration_args_list,
)
from topollm.scripts.hhu_hilbert.submit_jobs.call_command import call_command
from topollm.typing.enums import JobRunMode, Verbosity

default_logger = logging.getLogger(__name__)


def run_job_submission(
    python_script_absolute_path: pathlib.Path,
    job_script_args: list[str],
    machine_configuration: MachineConfigurationConfig,
    job_name: str = "default_wandb_project_name_0",
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Run the job submission by assembling the command and calling it."""
    job_script_args_str = " ".join(
        job_script_args,
    )
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            f"JOB_SCRIPT_ARGS={job_script_args_str}",  # noqa: G004 - low overhead
        )

    machine_configuration_args_list = get_machine_configuration_args_list(
        machine_configuration_config=machine_configuration,
    )

    if machine_configuration.job_run_mode == JobRunMode.HHU_HILBERT:
        command: list[str] = [
            *machine_configuration.submit_job_hilbert_command,
            "--job_name",
            str(job_name),
            "--job_script",
            str(python_script_absolute_path),
            *machine_configuration_args_list,
            "--job_script_args",
            job_script_args_str,
        ]
    elif machine_configuration.job_run_mode == JobRunMode.LOCAL:
        command = [
            *machine_configuration.run_job_locally_command,
            str(python_script_absolute_path),
            *job_script_args,
        ]
    else:
        msg = f"Invalid: {machine_configuration.job_run_mode = }"
        raise ValueError(msg)

    # Logging of the command is done in the `call_command` function
    call_command(
        command=command,
        dry_run=machine_configuration.dry_run,
        verbosity=verbosity,
        logger=logger,
    )
