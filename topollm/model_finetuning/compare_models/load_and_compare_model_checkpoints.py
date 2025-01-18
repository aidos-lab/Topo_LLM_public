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

# TODO This script is under development and not yet finished.

"""Load and compare model checkpoints."""

import itertools
import logging
import pathlib
from typing import TYPE_CHECKING

import hydra
import hydra.core.hydra_config
import omegaconf
import transformers

from topollm.config_classes.get_data_dir import get_data_dir
from topollm.config_classes.setup_OmegaConf import setup_omega_conf
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.path_management.embeddings.factory import get_embeddings_path_manager
from topollm.path_management.embeddings.protocol import EmbeddingsPathManager
from topollm.typing.enums import Verbosity

if TYPE_CHECKING:
    from topollm.config_classes.main_config import MainConfig


global_logger: logging.Logger = logging.getLogger(
    name=__name__,
)

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)

setup_exception_logging(
    logger=global_logger,
)
setup_omega_conf()


@hydra.main(
    config_path="../../../configs",
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
        logger=global_logger,
    )
    verbosity: Verbosity = main_config.verbosity

    embeddings_path_manager: EmbeddingsPathManager = get_embeddings_path_manager(
        main_config=main_config,
        logger=logger,
    )

    # # # #
    # Load the models
    model_loading_class = transformers.AutoModelForMaskedLM

    model_files_dir_relative_to_data_dir_list: list = [
        pathlib.Path(
            "models/finetuned_models/data=one-year-of-tsla-on-reddit_rm-empty=True_spl-mode=proportions_spl-shuf=True_spl-seed=0_tr=0.8_va=0.1_te=0.1_ctxt=dataset_entry_feat-col=ner_tags/split=train_samples=80_sampling=take_first/model=roberta-base_task=masked_lm_dr=defaults/ftm=standard/lora-None/gradmod=freeze_layers_target-freeze=lm_head/lr=5e-05_lr-scheduler-type=linear_wd=0.01/bs-train=16/ep=2/",
            "seed=1235/model_files/checkpoint-2",
        ),
        pathlib.Path(
            "models/finetuned_models/data=one-year-of-tsla-on-reddit_rm-empty=True_spl-mode=proportions_spl-shuf=True_spl-seed=0_tr=0.8_va=0.1_te=0.1_ctxt=dataset_entry_feat-col=ner_tags/split=train_samples=80_sampling=take_first/model=roberta-base_task=masked_lm_dr=defaults/ftm=standard/lora-None/gradmod=freeze_layers_target-freeze=lm_head/lr=5e-05_lr-scheduler-type=linear_wd=0.01/bs-train=16/ep=2/",
            "seed=1235/model_files/checkpoint-4",
        ),
    ]

    # This holds the list of model files to compare
    models_identifier_or_paths_list: list[str] = [
        str(
            object=pathlib.Path(
                embeddings_path_manager.data_dir,
                relative_dir,
            ),
        )
        for relative_dir in model_files_dir_relative_to_data_dir_list
    ]

    # Load the models
    models_list: list[transformers.PreTrainedModel] = [
        model_loading_class.from_pretrained(
            pretrained_model_name_or_path=model_identifier_or_path,
        )
        for model_identifier_or_path in models_identifier_or_paths_list
    ]

    # Example lists of parameters to compare
    parameter_names_to_compare_debug_list: list[str] = [
        "lm_head.dense.weight",
    ]

    parameter_names_to_compare_extended_list: list[str] = [
        "roberta.embeddings.word_embeddings.weight",
        "roberta.embeddings.position_embeddings.weight",
        "roberta.embeddings.token_type_embeddings.weight",
        "roberta.embeddings.LayerNorm.weight",
        "roberta.embeddings.LayerNorm.bias",
        "lm_head.bias",
        "lm_head.dense.weight",
        "lm_head.dense.bias",
        "lm_head.layer_norm.weight",
        "lm_head.layer_norm.bias",
        "lm_head.decoder",
        "roberta.encoder.layer.1.attention.self.query.weight",
        "roberta.encoder.layer.11.attention.self.query.weight",
    ]

    # This holds the list of parameters to compare
    parameter_names_to_compare = parameter_names_to_compare_debug_list

    for parameter_name in parameter_names_to_compare:
        compare_model_components(
            models_list=models_list,
            parameters_to_compare=parameter_name,
            verbosity=verbosity,
            logger=logger,
        )

    logger.info(
        msg="Running script DONE",
    )


def compare_model_components(
    models_list: list[transformers.PreTrainedModel],
    parameters_to_compare: str,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
):
    """Compare the model components."""
    # Iterate over pairs of models
    pairs_of_models = itertools.combinations(
        iterable=models_list,
        r=2,
    )

    for model_1, model_2 in pairs_of_models:
        # Compare the parameters
        model_1_parameter = model_1.state_dict()[parameters_to_compare]
        model_2_parameter = model_2.state_dict()[parameters_to_compare]

        # Compare the parameters

    # TODO: Implement this


if __name__ == "__main__":
    main()
