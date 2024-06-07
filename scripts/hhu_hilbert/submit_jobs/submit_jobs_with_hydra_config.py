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

"""Submit jobs for finetuning language models on huggingface datasets."""

import logging
import os
import pathlib
import pprint
import subprocess
from dataclasses import dataclass, field
from itertools import product

import hydra
from hydra.core.config_store import ConfigStore
from tqdm import tqdm

TOPO_LLM_REPOSITORY_BASE_PATH = os.getenv(
    "TOPO_LLM_REPOSITORY_BASE_PATH",
    "$HOME/git-source/Topo_LLM",
)


@dataclass
class SubmitJobsConfig:
    """Config for submitting jobs."""

    base_model: list[str]
    finetuning_dataset: list[str]
    peft: list[str]
    gradient_modifier: list[str]
    lora_r: list[str]
    num_train_epochs: int = 5
    common_batch_size: int = 16
    save_steps: int = 400
    eval_steps: int = 100
    lr_scheduler_type: str = "linear"
    accelerator_model: str = "rtx6000"
    queue: str = "CUDA"
    finetuning_python_script_name: str = "run_finetune_language_model_on_huggingface_dataset.py"
    topo_llm_repository_base_path: str = TOPO_LLM_REPOSITORY_BASE_PATH

    ncpus: int = 2
    walltime: str = "08:00:00"
    ngpus: int = 1
    memory_gb: int = 32

    submit_job_command: list[str] = field(
        default_factory=lambda: [
            "python3",
            "/gpfs/project/ruppik/.usr_tls/tools/submit_job.py",
        ],
    )

    wandb_project: str = "Topo_LLM_submit_jobs_via_hydra_debug"
    dry_run: bool = False


@dataclass
class Config:
    """Config for the main function."""

    submit_jobs: SubmitJobsConfig


cs = ConfigStore.instance()
cs.store(
    group="submit_jobs",
    name="base_submit_jobs_config",
    node=SubmitJobsConfig,
)
cs.store(
    name="base_config",
    node=Config,
)

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

    submit_jobs_config = cfg.submit_jobs

    finetuning_python_script_path = pathlib.Path(
        submit_jobs_config.topo_llm_repository_base_path,
        "topollm",
        "model_finetuning",
        submit_jobs_config.finetuning_python_script_name,
    )

    combinations = product(
        submit_jobs_config.base_model,
        submit_jobs_config.finetuning_dataset,
        submit_jobs_config.peft,
        submit_jobs_config.gradient_modifier,
        submit_jobs_config.lora_r,
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
            f"BASE_MODEL={base_model}",  # noqa: G004 - low overhead
        )
        logger.info(
            f"FINETUNING_DATASET={finetuning_dataset}",  # noqa: G004 - low overhead
        )
        logger.info(
            f"PEFT={peft}",  # noqa: G004 - low overhead
        )
        logger.info(
            f"GRADIENT_MODIFIER={gradient_modifier}",  # noqa: G004 - low overhead
        )
        logger.info(
            f"LORA_R={lora_r}",  # noqa: G004 - low overhead
        )

        job_script_args = [
            "--multirun",
            f"finetuning/base_model@finetuning={base_model}",
            f"finetuning.num_train_epochs={submit_jobs_config.num_train_epochs}",
            f"finetuning.lr_scheduler_type={submit_jobs_config.lr_scheduler_type}",
            f"finetuning.batch_sizes.train={submit_jobs_config.common_batch_size}",
            f"finetuning.batch_sizes.eval={submit_jobs_config.common_batch_size}",
            f"finetuning.save_steps={submit_jobs_config.save_steps}",
            f"finetuning.eval_steps={submit_jobs_config.eval_steps}",
            "finetuning.fp16=true",
            f"finetuning/finetuning_datasets={finetuning_dataset}",
            f"finetuning/peft={peft}",
            f"finetuning/gradient_modifier={gradient_modifier}",
            f"wandb.project={submit_jobs_config.wandb_project}",
            f"++finetuning.peft.r={lora_r}",
        ]

        job_script_args_str = " ".join(job_script_args)
        logger.info(
            f"JOB_SCRIPT_ARGS={job_script_args_str}",  # noqa: G004 - low overhead
        )

        command: list[str] = [
            *submit_jobs_config.submit_job_command,
            "--job_name",
            f"my_python_job_{job_id}",
            "--job_script",
            str(finetuning_python_script_path),
            "--ncpus",
            str(submit_jobs_config.ncpus),
            "--memory",
            str(submit_jobs_config.memory_gb),
            "--ngpus",
            str(submit_jobs_config.ngpus),
            "--accelerator_model",
            submit_jobs_config.accelerator_model,
            "--queue",
            submit_jobs_config.queue,
            "--walltime",
            submit_jobs_config.walltime,
            "--job_script_args",
            job_script_args_str,
        ]

        if submit_jobs_config.dry_run:
            logger.info(
                "Dry run enabled. Command not executed.",
            )
            logger.info(
                "Dry run command:\n%s",
                command,
            )
        else:
            # Calling submit_job
            logger.info(
                "Calling submit_job ...",
            )
            subprocess.run(
                args=command,  # noqa: S603 , S607 - we trust the input; we need to use the "submit_job" here
                shell=False,
                check=True,
            )
            logger.info(
                "Calling submit_job DONE",
            )

    logger.info(
        "Running main DONE",
    )


if __name__ == "__main__":
    main()
