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
from topollm.config_classes.submit_jobs.machine_configuration_config import get_machine_configuration_args_list
from topollm.config_classes.submit_jobs.submit_jobs_config import SubmitJobsConfig
from topollm.config_classes.submit_jobs.submit_pipeline_jobs_config import SubmitPipelineJobsConfig
from topollm.scripts.hhu_hilbert.submit_jobs.call_command import call_command
from topollm.scripts.hhu_hilbert.submit_jobs.run_job_submission import run_job_submission
from topollm.typing.enums import Verbosity

if TYPE_CHECKING:
    from topollm.config_classes.submit_jobs.machine_configuration_config import MachineConfigurationConfig
    from topollm.config_classes.submit_jobs.submit_finetuning_jobs_config import TrainingScheduleConfig

global_logger = logging.getLogger(__name__)


@hydra.main(
    config_path=f"{HYDRA_CONFIGS_BASE_PATH}/submit_jobs",
    config_name="config",
    version_base="1.2",
)
def main(
    submit_jobs_config: SubmitJobsConfig,
) -> None:
    """Run the main function."""
    logger = global_logger
    verbosity: Verbosity = submit_jobs_config.machine_configuration.verbosity

    logger.info("Running main ...")
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            "submit_jobs_config:\n%s",
            pprint.pformat(submit_jobs_config),
        )

    submit_pipeline_jobs_config: SubmitPipelineJobsConfig = submit_jobs_config.submit_pipeline_jobs
    machine_configuration: MachineConfigurationConfig = submit_jobs_config.machine_configuration

    python_script_absolute_path = pathlib.Path(
        submit_jobs_config.topo_llm_repository_base_path,
        submit_pipeline_jobs_config.pipeline_python_script_relative_path,
    )
    wandb_project: str = submit_pipeline_jobs_config.wandb_project

    # # # #
    # Argument combinations
    combinations = product(
        submit_pipeline_jobs_config.data_list,
        submit_pipeline_jobs_config.language_model_list,
        submit_pipeline_jobs_config.checkpoint_no_list,
        submit_pipeline_jobs_config.layer_indices_list,
        submit_pipeline_jobs_config.data_number_of_samples_list,
        submit_pipeline_jobs_config.embeddings_data_prep_num_samples_list,
    )

    for job_id, combination in enumerate(
        tqdm(
            combinations,
            desc="Submitting jobs",
        ),
    ):
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                60 * "=",
            )
            logger.info(
                "combination:\n%s",
                combination,
            )

        (
            data,
            language_model,
            checkpoint_no,
            layer_indices,
            data_number_of_samples,
            embeddings_data_prep_num_samples,
        ) = combination

        job_script_args = [
            "--multirun",
            f"data={data}",
            f"language_model={language_model}",
            f"+language_model.checkpoint_no={checkpoint_no}",
            f"embeddings.embedding_extraction.layer_indices={layer_indices}",
            f"data.number_of_samples={data_number_of_samples}",
            f"embeddings_data_prep.num_samples={embeddings_data_prep_num_samples}",
            f"wandb.project={wandb_project}",
        ]

        job_name: str = f"{wandb_project}_{job_id}"

        run_job_submission(
            python_script_absolute_path=python_script_absolute_path,
            job_script_args=job_script_args,
            machine_configuration=machine_configuration,
            job_name=job_name,
            verbosity=machine_configuration.verbosity,
        )

    logger.info(
        "Running main DONE",
    )


if __name__ == "__main__":
    main()
