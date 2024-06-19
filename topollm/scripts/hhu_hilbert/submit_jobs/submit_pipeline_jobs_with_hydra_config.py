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

"""Submit jobs for finetuning language models on huggingface datasets."""

import logging
import pathlib
import pprint
from itertools import product
from typing import TYPE_CHECKING

import hydra
from tqdm import tqdm

from topollm.config_classes.constants import HYDRA_CONFIGS_BASE_PATH
from topollm.config_classes.submit_jobs.submit_jobs_config import SubmitJobsConfig
from topollm.config_classes.submit_jobs.submit_pipeline_jobs_config import SubmitPipelineJobsConfig
from topollm.scripts.hhu_hilbert.submit_jobs.call_command import call_command

if TYPE_CHECKING:
    from topollm.config_classes.submit_jobs.machine_configuration_config import MachineConfigurationConfig
    from topollm.config_classes.submit_jobs.submit_finetuning_jobs_config import TrainingScheduleConfig

logger = logging.getLogger(__name__)


@hydra.main(
    config_path=f"{HYDRA_CONFIGS_BASE_PATH}/submit_jobs",
    config_name="config",
    version_base="1.2",
)
def main(
    submit_jobs_config: SubmitJobsConfig,
) -> None:
    """Run the main function."""
    logger.info("Running main ...")

    logger.info(
        "cfg:\n%s",
        pprint.pformat(submit_jobs_config),
    )

    machine_configuration: MachineConfigurationConfig = submit_jobs_config.machine_configuration
    submit_pipeline_jobs_config: SubmitPipelineJobsConfig = submit_jobs_config.submit_pipeline_jobs

    pipeline_python_script_absolute_path = pathlib.Path(
        submit_jobs_config.topo_llm_repository_base_path,
        submit_pipeline_jobs_config.pipeline_python_script_relative_path,
    )

    # # # #
    # Argument combinations
    combinations = product(
        submit_pipeline_jobs_config.data_list,
        submit_pipeline_jobs_config.language_model_list,
    )

    for job_id, combination in enumerate(
        tqdm(
            combinations,
            desc="Submitting jobs",
        ),
    ):
        logger.info(
            "combination:\n%s",
            combination,
        )
        data, language_model = combination

        # TODO: Continue here

        logger.info(
            f"{data = }",  # noqa: G004 - low overhead
        )
        logger.info(
            f"{language_model = }",  # noqa: G004 - low overhead
        )

        job_script_args = [
            "--multirun",
            f"data={data}",
            f"language_model={language_model}",
            f"wandb.project={submit_pipeline_jobs_config.wandb_project}",
        ]

        job_script_args_str = " ".join(job_script_args)
        logger.info(
            f"JOB_SCRIPT_ARGS={job_script_args_str}",  # noqa: G004 - low overhead
        )

        command: list[str] = (
            [
                *machine_configuration.submit_job_command,
                "--job_name",
                f"{submit_pipeline_jobs_config.wandb_project}_{job_id}",
                "--job_script",
                str(pipeline_python_script_absolute_path),
            ]
            + [
                "--ncpus",
                str(machine_configuration.ncpus),
                "--memory",
                str(machine_configuration.memory_gb),
                "--ngpus",
                str(machine_configuration.ngpus),
                "--accelerator_model",
                machine_configuration.accelerator_model,
                "--queue",
                machine_configuration.queue,
                "--walltime",
                machine_configuration.walltime,
            ]
            + [
                "--job_script_args",
                job_script_args_str,
            ]
        )

        call_command(
            command=command,
            dry_run=machine_configuration.dry_run,
            logger=logger,
        )

    logger.info(
        "Running main DONE",
    )


if __name__ == "__main__":
    main()
