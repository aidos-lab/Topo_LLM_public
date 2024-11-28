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

"""Configuration for submitting a certain job."""

import pathlib
import random
from enum import StrEnum

from pydantic import BaseModel, Field

from topollm.config_classes.constants import TOPO_LLM_REPOSITORY_BASE_PATH
from topollm.scripts.submission_scripts.types import RunOnlySelectedConfigsOption
from topollm.typing.enums import EmbeddingsDataPrepSamplingMode, SubmissionMode, Task


class Template(StrEnum):
    """Enumeration of HPC templates."""

    DSML_SHORT = "DSML_SHORT"
    DSML = "DSML"
    CPU = "CPU"
    GTX1080 = "GTX1080"
    TESLAT4 = "TESLAT4"
    RTX6000 = "RTX6000"
    RTX8000 = "RTX8000"
    A100_40GB = "A100_40GB"
    A100_80GB = "A100_80GB"


class MachineConfig(BaseModel):
    """Configuration for the machine on which the job is run."""

    queue: str | None = ""  # Empty string means the default queue
    template: Template | None = Template.DSML

    memory: str | None = "64"
    ncpus: str | None = "2"
    ngpus: str | None = "1"
    walltime: str | None = "16:00:00"


class SubmissionConfig(BaseModel):
    """Configuration for submitting a certain job."""

    # Submission parameters
    submission_mode: SubmissionMode = SubmissionMode.HPC_SUBMISSION
    machine_config: MachineConfig = MachineConfig()

    # Common parameters
    add_prefix_space: bool = False
    data_list: list[str] = Field(
        default_factory=lambda: [
            "iclr_2024_submissions_validation",
            "multiwoz21_validation",
        ],
    )
    data_subsampling_sampling_mode: str | None = "random"
    data_subsampling_number_of_samples_list: list[str] | None = [
        "10000",
    ]
    data_subsampling_sampling_seed_list: list[str] | None = [
        "777",
    ]

    # The additional data options are just used for extending the data command as they are
    additional_data_options: list[str] | None = None

    language_model_list: list[str] = Field(
        default_factory=lambda: [
            "roberta-base",
        ],
    )
    language_model_seed_list: list[str] | None = None
    checkpoint_no_list: list[str] | None = None

    layer_indices: str | None = "[-1]"
    embeddings_data_prep_num_samples_list: list[str] | None = [
        "60000",
    ]
    embeddings_data_prep_sampling_mode: EmbeddingsDataPrepSamplingMode | None = (
        EmbeddingsDataPrepSamplingMode.TAKE_FIRST  # type: ignore - problem with StrEnum type
    )
    embeddings_data_prep_sampling_seed_list: list[str] | None = [
        "42",
    ]
    additional_overrides: list[str] | None = None

    # # # #
    # Local estimates parameters
    local_estimates_filtering_num_samples_list: list[str] | None = [
        "60000",
    ]
    local_estimates_filtering_deduplication_mode: str = "array_deduplicator"
    local_estimates_pointwise_n_neighbors_mode: str = "absolute_size"
    local_estimates_pointwise_absolute_n_neighbors_list: list[str] | None = None

    # Finetuning-specific parameters
    base_model_list: list[str] = Field(
        default_factory=lambda: [
            "roberta-base_for_masked_lm",
        ],
    )
    fp16: str = Field(
        default="true",
        examples=[
            "true",
            "false",
        ],
    )
    num_train_epochs: str | None = "5"
    save_steps: str | None = "400"
    eval_steps: str | None = "100"
    finetuning_datasets_list: list[str] = Field(
        default_factory=lambda: [
            "train_and_eval_on_multiwoz21_train-samples-small",
        ],
    )
    finetuning_seed_list: list[str] | None = None
    lr_scheduler_type: str | None = Field(
        default="linear",
        examples=[
            "linear",
            "constant",
        ],
    )
    peft_list: str | None = Field(
        default="standard",
        examples=[
            "standard",
            "lora",
        ],
    )
    gradient_modifier_list: str | None = Field(
        default="do_nothing",
        examples=[
            "do_nothing",
            "freeze_first_layers_bert-style-models",
        ],
    )
    batch_size_train: int | None = 8
    batch_size_eval: int | None = 8

    wandb_project: str | None = "Topo_LLM_finetuning_from_submission_script"

    # # # #
    # Feature flags
    skip_compute_and_store_embeddings: bool = False
    feature_flags_wandb_use_wandb: str | None = "true"

    python_script_name: str = Field(
        default="run_pipeline_compute_embeddings_and_data_prep_and_local_estimate.py",
    )

    relative_python_script_folder: pathlib.Path = Field(
        default=pathlib.Path(
            "topollm",
            "pipeline_scripts",
        ),
    )

    @property
    def relative_python_script_path(
        self,
    ) -> pathlib.Path:
        return pathlib.Path(
            self.relative_python_script_folder,
            self.python_script_name,
        )

    @property
    def absolute_python_script_path(
        self,
    ) -> pathlib.Path:
        return pathlib.Path(
            TOPO_LLM_REPOSITORY_BASE_PATH,
            self.relative_python_script_path,
        )

    @property
    def poetry_run_command(
        self,
    ) -> list[str]:
        return [
            "poetry",
            "run",
            "python3",
            str(object=self.relative_python_script_path),
            "--multirun",
            "hydra/sweeper=basic",
        ]

    def get_command(
        self,
        task: Task,
    ) -> list[str]:
        """Get the command which runs the task."""
        command: list[str] = (
            self.poetry_run_command
            + self.generate_task_specific_command(
                task=task,
            )
            + self.generate_feature_flags_command()
            + self.generate_hydra_launcher_command()
        )

        # We need to check that the additional overrides are not None and not empty
        if self.additional_overrides and len(self.additional_overrides) > 0:
            command.extend(
                self.additional_overrides,
            )

        return command

    def generate_feature_flags_command(
        self,
    ) -> list[str]:
        feature_flags_command: list[str] = []

        if self.skip_compute_and_store_embeddings:
            feature_flags_command.append(
                "feature_flags.compute_and_store_embeddings.skip_compute_and_store_embeddings=true",
            )
        if self.feature_flags_wandb_use_wandb:
            feature_flags_command.append(
                "feature_flags.wandb.use_wandb=" + self.feature_flags_wandb_use_wandb,
            )

        return feature_flags_command

    def generate_task_specific_command(
        self,
        task: Task,
    ) -> list[str]:
        """Generate the task-specific command."""
        match task:
            case Task.LOCAL_ESTIMATES_COMPUTATION:
                # We can use the same command for the local estimates computation
                # as for the pipeline, because the configuration is the same.
                # The only difference here is that for this task, only the last script in the pipeline is run.
                task_specific_command: list[str] = self.generate_task_specific_command_pipeline()
            case Task.PIPELINE:
                task_specific_command: list[str] = self.generate_task_specific_command_pipeline()
            case Task.PERPLEXITY:
                task_specific_command: list[str] = self.generate_task_specific_command_perplexity()
            case Task.FINETUNING:
                task_specific_command: list[str] = self.generate_task_specific_command_finetuning()
            case _:
                msg: str = f"Unknown {task = }"
                raise ValueError(
                    msg,
                )

        return task_specific_command

    def generate_hydra_launcher_command(
        self,
    ) -> list[str]:
        """Generate the hydra launcher command."""
        hydra_launcher_command: list[str] = []

        match self.submission_mode:
            case SubmissionMode.HPC_SUBMISSION:
                hydra_launcher_command.append(
                    "hydra/launcher=hpc_submission",
                )
                if self.machine_config.queue:
                    hydra_launcher_command.append(
                        f"hydra.launcher.queue={self.machine_config.queue}",
                    )
                if self.machine_config.template:
                    hydra_launcher_command.append(
                        f"hydra.launcher.template={self.machine_config.template}",
                    )
                if self.machine_config.memory:
                    hydra_launcher_command.append(
                        f"hydra.launcher.memory={self.machine_config.memory}",
                    )
                if self.machine_config.ncpus:
                    hydra_launcher_command.append(
                        f"hydra.launcher.ncpus={self.machine_config.ncpus}",
                    )
                if self.machine_config.ngpus:
                    hydra_launcher_command.append(
                        f"hydra.launcher.ngpus={self.machine_config.ngpus}",
                    )
                if self.machine_config.walltime:
                    hydra_launcher_command.append(
                        f"hydra.launcher.walltime={self.machine_config.walltime}",
                    )
            case SubmissionMode.LOCAL:
                hydra_launcher_command.extend(
                    [
                        "hydra/launcher=basic",
                    ],
                )
            case _:
                msg: str = f"Unknown {self.submission_mode = }"
                raise ValueError(
                    msg,
                )

        return hydra_launcher_command

    def generate_task_specific_command_finetuning(
        self,
    ) -> list[str]:
        task_specific_command: list[str] = [
            f"finetuning/base_model={','.join(self.base_model_list)}",
            f"finetuning.num_train_epochs={self.num_train_epochs}",
            f"finetuning.lr_scheduler_type={self.lr_scheduler_type}",
            f"finetuning.save_steps={self.save_steps}",
            f"finetuning.eval_steps={self.eval_steps}",
            f"finetuning.fp16={self.fp16}",
            f"finetuning.batch_sizes.train={self.batch_size_train}",
            f"finetuning.batch_sizes.eval={self.batch_size_eval}",
            f"finetuning/finetuning_datasets={','.join(self.finetuning_datasets_list)}",
            f"finetuning/peft={self.peft_list}",
            f"finetuning/gradient_modifier={self.gradient_modifier_list}",
        ]

        if self.finetuning_seed_list:
            task_specific_command.append(
                "finetuning.seed=" + ",".join(self.finetuning_seed_list),
            )

        if self.wandb_project:
            task_specific_command.append(
                f"wandb.project={self.wandb_project}",
            )

        return task_specific_command

    def generate_task_specific_command_perplexity(
        self,
    ) -> list[str]:
        task_specific_command: list[str] = []

        data_command: list[str] = self.generate_data_command()
        task_specific_command.extend(
            data_command,
        )

        task_specific_command.append(
            f"tokenizer.add_prefix_space={self.add_prefix_space}",
        )

        language_model_command: list[str] = self.generate_language_model_command()
        task_specific_command.extend(
            language_model_command,
        )

        return task_specific_command

    def generate_task_specific_command_pipeline(
        self,
    ) -> list[str]:
        task_specific_command: list[str] = []

        data_command: list[str] = self.generate_data_command()
        task_specific_command.extend(
            data_command,
        )

        task_specific_command.append(
            f"tokenizer.add_prefix_space={self.add_prefix_space}",
        )

        # Add the language model command
        language_model_command: list[str] = self.generate_language_model_command()
        task_specific_command.extend(
            language_model_command,
        )

        if self.layer_indices:
            task_specific_command.append(
                f"embeddings.embedding_extraction.layer_indices={self.layer_indices}",
            )

        if self.embeddings_data_prep_num_samples_list:
            task_specific_command.append(
                "embeddings_data_prep.sampling.num_samples=" + ",".join(self.embeddings_data_prep_num_samples_list),
            )

        if self.embeddings_data_prep_sampling_mode:
            task_specific_command.append(
                f"embeddings_data_prep.sampling.sampling_mode={str(object=self.embeddings_data_prep_sampling_mode)}",
            )
        if self.embeddings_data_prep_sampling_seed_list:
            task_specific_command.append(
                "embeddings_data_prep.sampling.seed=" + ",".join(self.embeddings_data_prep_sampling_seed_list),
            )

        local_estimates_command: list[str] = self.generate_local_estimates_command()
        task_specific_command.extend(
            local_estimates_command,
        )

        return task_specific_command

    def generate_data_command(
        self,
    ) -> list[str]:
        data_command: list[str] = []

        data_command.append(
            "data=" + ",".join(self.data_list),
        )

        if self.data_subsampling_sampling_mode:
            data_command.append(
                f"data.data_subsampling.sampling_mode={self.data_subsampling_sampling_mode}",
            )

        if self.data_subsampling_number_of_samples_list:
            data_command.append(
                "data.data_subsampling.number_of_samples=" + ",".join(self.data_subsampling_number_of_samples_list),
            )

        if self.data_subsampling_sampling_seed_list:
            data_command.append(
                "data.data_subsampling.sampling_seed=" + ",".join(self.data_subsampling_sampling_seed_list),
            )

        # The additional data options are just used for extending the data command as they are
        if self.additional_data_options:
            data_command.extend(
                self.additional_data_options,
            )

        return data_command

    def generate_language_model_command(
        self,
    ) -> list[str]:
        language_model_command: list[str] = []

        language_model_command.append(
            "language_model=" + ",".join(self.language_model_list),
        )
        if self.checkpoint_no_list:
            language_model_command.append(
                "language_model.checkpoint_no=" + ",".join(self.checkpoint_no_list),
            )
        if self.language_model_seed_list:
            language_model_command.append(
                "++language_model.seed=" + ",".join(self.language_model_seed_list),
            )

        return language_model_command

    def generate_local_estimates_command(
        self,
    ) -> list[str]:
        local_estimates_command: list[str] = []

        local_estimates_command.append(
            f"local_estimates.pointwise.n_neighbors_mode={self.local_estimates_pointwise_n_neighbors_mode}",
        )
        local_estimates_command.append(
            "local_estimates.filtering.deduplication_mode=" + self.local_estimates_filtering_deduplication_mode,
        )

        if self.local_estimates_filtering_num_samples_list:
            local_estimates_command.append(
                "local_estimates.filtering.num_samples=" + ",".join(self.local_estimates_filtering_num_samples_list),
            )
        if self.local_estimates_pointwise_absolute_n_neighbors_list:
            local_estimates_command.append(
                "local_estimates.pointwise.absolute_n_neighbors="
                + ",".join(self.local_estimates_pointwise_absolute_n_neighbors_list),
            )

        return local_estimates_command


def pick_selected_options_in_each_list(
    submission_config: SubmissionConfig,
    run_only_selected_configs_option: RunOnlySelectedConfigsOption,
) -> SubmissionConfig:
    """Make new submission config which picks out selected options in each list field of the submission config."""
    submission_config_copy: SubmissionConfig = submission_config.model_copy(
        deep=True,
    )

    for field_name, field_value in submission_config_copy:
        if (
            isinstance(
                field_value,
                list,
            )
            and field_value  # Check that field_value is not None
            and len(field_value) > 0  # Check that field_value is not an empty list
        ):
            # Skip the field for additional overrides
            if field_name == "additional_overrides":
                continue

            match run_only_selected_configs_option:
                case RunOnlySelectedConfigsOption.RUN_ONLY_FIRST:
                    selection_index = 0
                case RunOnlySelectedConfigsOption.RUN_ONLY_LAST:
                    selection_index = -1
                case RunOnlySelectedConfigsOption.RUN_SINGLE_RANDOM:
                    length_of_field_value = len(field_value)
                    selection_index = random.randint(  # noqa: S311 - this random is not used for security purposes
                        0,
                        length_of_field_value - 1,
                    )
                case _:
                    msg: str = f"Unknown {run_only_selected_configs_option = }"
                    raise ValueError(
                        msg,
                    )
            setattr(
                submission_config_copy,
                field_name,
                [
                    field_value[selection_index],
                ],
            )

    return submission_config_copy
