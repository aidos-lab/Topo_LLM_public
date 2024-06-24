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
import subprocess
from itertools import product
from typing import TYPE_CHECKING

import hydra
from tqdm import tqdm

from topollm.config_classes.constants import HYDRA_CONFIGS_BASE_PATH
from topollm.config_classes.submit_jobs.machine_configuration_config import get_machine_configuration_args_list
from topollm.config_classes.submit_jobs.submit_jobs_config import SubmitJobsConfig
from topollm.scripts.hhu_hilbert.submit_jobs.call_command import call_command

if TYPE_CHECKING:
    from topollm.config_classes.submit_jobs.machine_configuration_config import MachineConfigurationConfig
    from topollm.config_classes.submit_jobs.submit_finetuning_jobs_config import (
        SubmitFinetuningJobsConfig,
        TrainingScheduleConfig,
    )

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

    submit_finetuning_jobs_config: SubmitFinetuningJobsConfig = submit_jobs_config.submit_finetuning_jobs
    machine_configuration: MachineConfigurationConfig = submit_jobs_config.machine_configuration

    python_script_absolute_path = pathlib.Path(
        submit_jobs_config.topo_llm_repository_base_path,
        submit_finetuning_jobs_config.finetuning_python_script_relative_path,
    )
    wandb_project: str = submit_finetuning_jobs_config.wandb_project

    combinations = product(
        submit_finetuning_jobs_config.base_model_list,
        submit_finetuning_jobs_config.finetuning_dataset_list,
        submit_finetuning_jobs_config.peft_list,
        submit_finetuning_jobs_config.gradient_modifier_list,
        submit_finetuning_jobs_config.lora_parameters.values(),
        submit_finetuning_jobs_config.training_schedule.values(),
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
        base_model, finetuning_dataset, peft, gradient_modifier, lora_parameters, training_schedule = combination
        training_schedule: TrainingScheduleConfig

        logger.info(
            f"{base_model = }",  # noqa: G004 - low overhead
        )
        logger.info(
            f"{finetuning_dataset = }",  # noqa: G004 - low overhead
        )
        logger.info(
            f"{peft = }",  # noqa: G004 - low overhead
        )
        logger.info(
            f"{gradient_modifier = }",  # noqa: G004 - low overhead
        )
        logger.info(
            f"{lora_parameters = }",  # noqa: G004 - low overhead
        )
        logger.info(
            f"{training_schedule = }",  # noqa: G004 - low overhead
        )

        job_script_args = [
            "--multirun",
            f"finetuning/base_model@finetuning={base_model}",
            f"finetuning.num_train_epochs={training_schedule.num_train_epochs}",
            f"finetuning.lr_scheduler_type={training_schedule.lr_scheduler_type}",
            f"finetuning.batch_sizes.train={submit_finetuning_jobs_config.common_batch_size}",
            f"finetuning.batch_sizes.eval={submit_finetuning_jobs_config.common_batch_size}",
            f"finetuning.save_steps={submit_finetuning_jobs_config.save_steps}",
            f"finetuning.eval_steps={submit_finetuning_jobs_config.eval_steps}",
            f"finetuning.fp16={submit_finetuning_jobs_config.fp16}",
            f"finetuning/finetuning_datasets={finetuning_dataset}",
            f"finetuning/peft={peft}",
            f"finetuning/gradient_modifier={gradient_modifier}",
            f"wandb.project={wandb_project}",
            f"++finetuning.peft.r={lora_parameters.lora_r}",
            f"++finetuning.peft.lora_alpha={lora_parameters.lora_alpha}",
            f"++finetuning.peft.use_rslora={lora_parameters.use_rslora}",
        ]

        job_script_args_str = " ".join(job_script_args)
        logger.info(
            f"JOB_SCRIPT_ARGS={job_script_args_str}",  # noqa: G004 - low overhead
        )

        machine_configuration_args_list = get_machine_configuration_args_list(
            machine_configuration_config=machine_configuration,
        )

        # TODO(Ben): Add an option to run the jobs locally in a sequence instead of submitting them to the cluster
        # (this can be useful to run the config file creation scripts locally)

        command: list[str] = [
            *machine_configuration.submit_job_command,
            "--job_name",
            f"{wandb_project}_{job_id}",
            "--job_script",
            str(python_script_absolute_path),
            *machine_configuration_args_list,
            "--job_script_args",
            job_script_args_str,
        ]

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
