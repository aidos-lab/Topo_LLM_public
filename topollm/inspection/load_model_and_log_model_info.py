# Copyright 2024-2025
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Matthias Ruppik (mail@ruppik.net)
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

"""Scrip to load model and log model info."""

import json
import logging
import pathlib
from enum import StrEnum, auto, unique
from typing import TYPE_CHECKING

import hydra
import omegaconf
import torch
import transformers

from topollm.config_classes.constants import HYDRA_CONFIGS_BASE_PATH
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.log_model_info import log_model_info
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.model_handling.prepare_loaded_model_container import (
    prepare_device_and_tokenizer_and_model_from_main_config,
)
from topollm.path_management.embeddings.factory import get_embeddings_path_manager
from topollm.path_management.embeddings.protocol import EmbeddingsPathManager
from topollm.typing.enums import Verbosity

if TYPE_CHECKING:
    from topollm.config_classes.main_config import MainConfig
    from topollm.model_handling.loaded_model_container import LoadedModelContainer

# Logger for this file
default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)
global_logger: logging.Logger = logging.getLogger(
    name=__name__,
)

setup_exception_logging(
    logger=global_logger,
)


@unique
class ModelLoadingMode(StrEnum):
    """How to load the model."""

    MANUALLY_FROM_IDENTIFIER = auto()
    FROM_MAIN_CONFIG = auto()


@hydra.main(
    config_path=f"{HYDRA_CONFIGS_BASE_PATH}",
    config_name="main_config",
    version_base="1.3",
)
def main(
    config: omegaconf.DictConfig,
) -> None:
    """Run the script."""
    logger: logging.Logger = global_logger
    logger.info(
        msg="Running script ...",
    )

    main_config: MainConfig = initialize_configuration(
        config=config,
        logger=logger,
    )
    verbosity: Verbosity = main_config.verbosity

    # The embeddings path manager will be used to get the data directory
    embeddings_path_manager: EmbeddingsPathManager = get_embeddings_path_manager(
        main_config=main_config,
        logger=logger,
    )

    # ==================================================== #
    # Load model
    # ==================================================== #
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"{transformers.__version__ = }",  # noqa: G004 - low overhead
        )

    # Select how the model should be loaded
    model_loading_mode: ModelLoadingMode = ModelLoadingMode.FROM_MAIN_CONFIG

    match model_loading_mode:
        case ModelLoadingMode.MANUALLY_FROM_IDENTIFIER:
            model: transformers.PreTrainedModel = load_model_manually_from_identifier(
                embeddings_path_manager=embeddings_path_manager,
                verbosity=verbosity,
                logger=logger,
            )
        case ModelLoadingMode.FROM_MAIN_CONFIG:
            loaded_model_container: LoadedModelContainer = prepare_device_and_tokenizer_and_model_from_main_config(
                main_config=main_config,
                verbosity=verbosity,
                logger=logger,
            )
            model: transformers.PreTrainedModel = loaded_model_container.model
        case _:
            msg: str = f"Invalid {model_loading_mode = }."
            raise ValueError(
                msg,
            )

    # ==================================================== #
    # Log model information
    # ==================================================== #

    log_model_info_for_notes_text_files(
        model=model,
        logger=logger,
    )

    # Example for accessing specific model parameters
    compare_different_ways_to_access_model_parameters(
        model=model,
        logger=logger,
    )

    model_information_to_save: dict = {}

    # # # # # # # #
    # Note:
    # The L2 norm of the model parameters is only well-defined for a model
    # if all weights have been initialized from a fixed distribution.
    # When loading a model from disk, make sure that the correct model loading class is used,
    # so that you do not accidentally load only a partial model
    # where certain weights are missing and thus not initialized.
    average_l2_norm_of_model_parameters: float = compute_average_l2_norm_of_model_parameters(
        model=model,
        verbosity=verbosity,
        logger=logger,
    )

    model_information_to_save["average_l2_norm_of_model_parameters"] = average_l2_norm_of_model_parameters

    # ==================================================== #
    # Save model information
    # ==================================================== #

    output_folder: pathlib.Path = embeddings_path_manager.get_language_model_information_dir_absolute_path()
    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file_path: pathlib.Path = pathlib.Path(
        output_folder,
        "model_information.json",
    )

    # Save the model information
    with output_file_path.open(
        mode="w",
    ) as file:
        json.dump(
            obj=model_information_to_save,
            fp=file,
            indent=4,
        )

    logger.info(
        msg="Running script DONE",
    )


def compute_average_l2_norm_of_model_parameters(
    model: transformers.PreTrainedModel,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> float:
    """Compute the average L2 norm of the model parameters."""
    average_l2_norm: float = 0.0
    num_parameter_components: int = 0
    skipped_parameter_components: int = 0

    for parameter_value in model.state_dict().values():
        if hasattr(
            parameter_value,
            "shape",
        ):
            num_parameter_components += 1
            average_l2_norm += (
                torch.linalg.norm(
                    input=parameter_value,
                )
                .detach()
                .cpu()
                .numpy()
            )
        else:
            skipped_parameter_components += 1

    average_l2_norm /= num_parameter_components

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"{num_parameter_components = }",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"{skipped_parameter_components = }",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"{average_l2_norm = }",  # noqa: G004 - low overhead
        )

    return average_l2_norm


def load_model_manually_from_identifier(
    embeddings_path_manager: EmbeddingsPathManager,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> transformers.PreTrainedModel:
    """Load model manually from identifier."""
    example_base_model_identifier = "roberta-base"
    example_1_finetuned_model_identifier = pathlib.Path(
        embeddings_path_manager.data_dir,
        "models/finetuned_models/data=one-year-of-tsla-on-reddit_rm-empty=True_spl-mode=proportions_spl-shuf=True_spl-seed=0_tr=0.8_va=0.1_te=0.1_ctxt=dataset_entry_feat-col=ner_tags/split=train_samples=80_sampling=take_first/model=roberta-base_task=masked_lm_dr=defaults/ftm=standard/lora-None/",
        "gradmod=do_nothing_target-freeze=/lr=5e-05_lr-scheduler-type=linear_wd=0.01/bs-train=16/ep=2/",
        "seed=1235/model_files/checkpoint-4",
    )
    example_2_finetuned_model_identifier = pathlib.Path(
        embeddings_path_manager.data_dir,
        "models/finetuned_models/data=one-year-of-tsla-on-reddit_rm-empty=True_spl-mode=proportions_spl-shuf=True_spl-seed=0_tr=0.8_va=0.1_te=0.1_ctxt=dataset_entry_feat-col=ner_tags/split=train_samples=80_sampling=take_first/model=roberta-base_task=masked_lm_dr=defaults/ftm=standard/lora-None/",
        "gradmod=freeze_layers_target-freeze=lm_head/lr=5e-05_lr-scheduler-type=linear_wd=0.01/bs-train=16/ep=2/",
        "seed=1235/model_files/checkpoint-4",
    )
    example_3_finetuned_model_identifier = pathlib.Path(
        embeddings_path_manager.data_dir,
        "models/finetuned_models/data=one-year-of-tsla-on-reddit_rm-empty=True_spl-mode=proportions_spl-shuf=True_spl-seed=0_tr=0.8_va=0.1_te=0.1_ctxt=dataset_entry_feat-col=ner_tags/split=train_samples=80_sampling=take_first/model=roberta-base_task=masked_lm_dr=defaults/ftm=standard/lora-None/",
        "gradmod=freeze_layers_target-freeze=lm_head_embeddings.word_embeddings/lr=5e-05_lr-scheduler-type=linear_wd=0.01/bs-train=16/ep=2/",
        "seed=1235/model_files/checkpoint-4",
    )

    # ==================================================== #
    # > Select the model identifier.
    # > Note: You can change the model identifier here.
    model_identifier: str = str(
        object=example_base_model_identifier,
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"{model_identifier = }",  # noqa: G004 - low overhead
        )

    # Load the model
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Loading model from {model_identifier = } ...",  # noqa: G004 - low overhead
        )
    model = transformers.AutoModelForMaskedLM.from_pretrained(
        pretrained_model_name_or_path=model_identifier,
    )
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Loading model from {model_identifier = } DONE",  # noqa: G004 - low overhead
        )

    return model


def log_model_info_for_notes_text_files(
    model: transformers.PreTrainedModel,
    logger: logging.Logger = default_logger,
) -> None:
    """Log model information for notes text files."""
    log_model_info(
        model=model,
        model_name="model_name",
        logger=logger,
    )

    # # # #
    # Print the information for easier copy-pasting into notes files

    # Log the shapes of the parameters in the state dict
    for key, value in model.state_dict().items():
        if hasattr(
            value,
            "shape",
        ):
            print(  # noqa: T201 - we want this script to print
                f"{key = }:\n{value.shape = }.",
            )


def compare_different_ways_to_access_model_parameters(
    model: transformers.PreTrainedModel,
    logger: logging.Logger = default_logger,
) -> None:
    """Compare different ways to access model parameters."""
    # Access the model parameters for RoBERTa models
    try:
        # Two different ways to access the same parameter:
        # 1. Via the model object
        lm_head_decoder_weight_tensor_via_model_object = model.lm_head.decoder.weight
        # 2. Via the state_dict
        lm_head_decoder_weight_tensor_via_state_dict = model.state_dict()["lm_head.decoder.weight"]

        # Check if the two tensors are equal
        comparison_result = torch.allclose(
            input=lm_head_decoder_weight_tensor_via_model_object,
            other=lm_head_decoder_weight_tensor_via_state_dict,
        )
        logger.info(
            msg="Comparing the tensors via the model object and the state dict:",
        )
        logger.info(
            msg=f"{comparison_result = }",  # noqa: G004 - low overhead
        )

        # Since the above comparison should return True,
        # we can use either of the two tensors to access the parameter.
        lm_head_decoder_weight_tensor = lm_head_decoder_weight_tensor_via_state_dict

        # # # # # # # #
        # For the RoBERTa model,
        # the output embedding weights are tied to the input embedding weights by default.
        #
        # See:
        # https://github.com/huggingface/transformers/issues/9753
        embeddings_word_embeddings_weight_tensor = model.state_dict()["roberta.embeddings.word_embeddings.weight"]

        # Check if the two tensors are equal
        comparison_result = torch.allclose(
            input=lm_head_decoder_weight_tensor,
            other=embeddings_word_embeddings_weight_tensor,
        )
        logger.info(
            msg="Comparing the output embedding weights with the input embedding weights:",
        )
        logger.info(
            msg=f"{comparison_result = }",  # noqa: G004 - low overhead
        )
    except KeyError:
        logger.exception(
            msg="KeyError when trying to access model parameters.",
        )
    except AttributeError:
        logger.exception(
            msg="AttributeError when trying to access model parameters.",
        )


if __name__ == "__main__":
    main()
