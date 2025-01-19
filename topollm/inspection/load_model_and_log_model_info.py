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

"""Scrip to load model and log model info."""

import logging
import pathlib
from typing import TYPE_CHECKING

import hydra
import hydra.core.hydra_config
import omegaconf
import torch
import transformers

from topollm.config_classes.constants import HYDRA_CONFIGS_BASE_PATH
from topollm.config_classes.setup_OmegaConf import setup_omega_conf
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.log_model_info import log_model_info
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.model_handling.get_torch_device import get_torch_device
from topollm.path_management.embeddings.factory import get_embeddings_path_manager
from topollm.path_management.embeddings.protocol import EmbeddingsPathManager
from topollm.pipeline_scripts.worker_for_pipeline import worker_for_pipeline
from topollm.typing.enums import Verbosity

if TYPE_CHECKING:
    from topollm.config_classes.main_config import MainConfig

try:
    from hydra_plugins import hpc_submission_launcher

    hpc_submission_launcher.register_plugin()
except ImportError:
    pass

# Logger for this file
global_logger: logging.Logger = logging.getLogger(
    name=__name__,
)

setup_exception_logging(
    logger=global_logger,
)

setup_omega_conf()


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

    # # # #
    # Load model and call logging function
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"{transformers.__version__ = }",  # noqa: G004 - low overhead
        )

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

    # Select the model identifier
    model_identifier = example_2_finetuned_model_identifier

    # Load the model
    model = transformers.AutoModelForMaskedLM.from_pretrained(
        pretrained_model_name_or_path=model_identifier,
    )

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

    # # # #
    # Accessing specific model parameters

    # Try to access the model parameters for RoBERTa models
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

    logger.info(
        msg="Running script DONE",
    )


if __name__ == "__main__":
    main()
