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

from scripts.hhu_hilbert.submit_jobs.config_classes.config import Config

if TYPE_CHECKING:
    from scripts.hhu_hilbert.submit_jobs.config_classes.submit_finetuning_jobs_config import SubmitFinetuningJobsConfig

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="config",
)
def main(
    cfg: Config,
) -> None:
    """Run the main function."""
    logger.info("Running main ...")

    logger.info(
        "cfg:\n%s",
        pprint.pformat(cfg),
    )

    submit_finetuning_jobs_config: SubmitFinetuningJobsConfig = cfg.submit_finetuning_jobs

    finetuning_python_script_absolute_path = pathlib.Path(
        submit_finetuning_jobs_config.topo_llm_repository_base_path,
        submit_finetuning_jobs_config.finetuning_python_script_relative_path,
    )

    combinations = product(
        submit_finetuning_jobs_config.base_model,
        submit_finetuning_jobs_config.finetuning_dataset,
        submit_finetuning_jobs_config.peft,
        submit_finetuning_jobs_config.gradient_modifier,
        submit_finetuning_jobs_config.lora_r,
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
        base_model, finetuning_dataset, peft, gradient_modifier, lora_r = combination

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
            f"{lora_r = }",  # noqa: G004 - low overhead
        )

        job_script_args = [
            "--multirun",
            f"finetuning/base_model@finetuning={base_model}",
            f"finetuning.num_train_epochs={submit_finetuning_jobs_config.num_train_epochs}",
            f"finetuning.lr_scheduler_type={submit_finetuning_jobs_config.lr_scheduler_type}",
            f"finetuning.batch_sizes.train={submit_finetuning_jobs_config.common_batch_size}",
            f"finetuning.batch_sizes.eval={submit_finetuning_jobs_config.common_batch_size}",
            f"finetuning.save_steps={submit_finetuning_jobs_config.save_steps}",
            f"finetuning.eval_steps={submit_finetuning_jobs_config.eval_steps}",
            "finetuning.fp16=true",
            f"finetuning/finetuning_datasets={finetuning_dataset}",
            f"finetuning/peft={peft}",
            f"finetuning/gradient_modifier={gradient_modifier}",
            f"wandb.project={submit_finetuning_jobs_config.wandb_project}",
            f"++finetuning.peft.r={lora_r}",
        ]

        job_script_args_str = " ".join(job_script_args)
        logger.info(
            f"JOB_SCRIPT_ARGS={job_script_args_str}",  # noqa: G004 - low overhead
        )

        command: list[str] = [
            *submit_finetuning_jobs_config.submit_job_command,
            "--job_name",
            f"{submit_finetuning_jobs_config.wandb_project}_{job_id}",
            "--job_script",
            str(finetuning_python_script_absolute_path),
            "--ncpus",
            str(submit_finetuning_jobs_config.ncpus),
            "--memory",
            str(submit_finetuning_jobs_config.memory_gb),
            "--ngpus",
            str(submit_finetuning_jobs_config.ngpus),
            "--accelerator_model",
            submit_finetuning_jobs_config.accelerator_model,
            "--queue",
            submit_finetuning_jobs_config.queue,
            "--walltime",
            submit_finetuning_jobs_config.walltime,
            "--job_script_args",
            job_script_args_str,
        ]

        # Add separator line to log
        logger.info(
            30 * "=",
        )

        if submit_finetuning_jobs_config.dry_run:
            logger.info(
                "Dry run enabled. Command not executed.",
            )
            logger.info(
                "** Dry run ** command:\n%s",
                command,
            )
        else:
            # Calling submit_job
            logger.info(
                "Calling submit_job ...",
            )
            logger.info(
                "command:\n%s",
                command,
            )
            subprocess.run(
                args=command,  # noqa: S603 , S607 - we trust the input; we need to use the "submit_job" here
                shell=False,
                check=True,
            )
            logger.info(
                "Calling submit_job DONE",
            )

        # Add separator line to log
        logger.info(
            30 * "=",
        )

    logger.info(
        "Running main DONE",
    )


if __name__ == "__main__":
    main()
