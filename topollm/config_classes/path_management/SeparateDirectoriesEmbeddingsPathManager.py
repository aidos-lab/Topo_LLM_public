# coding=utf-8
#
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

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Imports

# System imports
import logging
import os
import pathlib

# Local imports
from convlab.tda.config_classes.ExperimentConfig import (
    ExperimentConfig,
    labels_column_name_for_file_path_dict,
)
from convlab.tda.config_classes.path_management.truncate_length_of_desc import (
    truncate_length_of_desc,
)
from convlab.tda.config_classes.utils_validate_data_types import (
    list_of_ints_to_filename_str,
)
from convlab.util.handle_tda_environment_variables import (
    get_paths_from_environment_variables,
)

# Third-party imports


# END Imports
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Globals

tda_base_path, term_extraction_base_path = get_paths_from_environment_variables(
    set_variables_in_this_script=True,
)

# END Globals
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class SeparateDirectoriesEmbeddingsPathManager:
    def __init__(
        self,
        config: ExperimentConfig,
        logger: logging.Logger = logging.getLogger(__name__),
    ):
        self.config = config
        self.logger = logger

    def get_output_dir_absolute_path(
        self,
        base_path: os.PathLike = term_extraction_base_path,
    ) -> pathlib.Path:
        path = pathlib.Path(
            base_path,
            "data",
            "experiments",
            self.get_output_dir_partial_path(),
            self.config.timestamp,  # timestamp to make sure we do not overwrite previous runs
        )

        return path

    def get_output_dir_partial_path(
        self,
    ) -> pathlib.Path:
        path = pathlib.Path(
            "BioTagModels_CFPV",
            self.config.run_name,
            self.config.fusion_features_config_for_file_paths.embeddings_config.parent_directory,
            self.get_prediction_description_partial_path(),
            self.get_validation_description_partial_path(),
            self.get_training_description_partial_path(),
        )

        return path

    def get_prediction_description_partial_path(
        self,
    ) -> pathlib.Path:
        """
        This will be used to construct the 'output_dir' parameter
        of the Trainer config class.
        """
        fusion_features_config_predict = (
            self.config.fusion_features_config_collection.prediction
        )

        if fusion_features_config_predict is None:
            raise ValueError(
                f"{fusion_features_config_predict = }" f" is not allowed to be None."
            )

        path = pathlib.Path(
            f"pdata"  # This denotes for which dataset the predictions are made
            f"-{fusion_features_config_predict.embeddings_config.dataset_desc}",
            f"pfeatures"  # This denotes which features are used for prediction
            f"-{fusion_features_config_predict.final_features_descriptor_string}",
        )

        return path

    def get_validation_description_partial_path(
        self,
    ) -> pathlib.Path:
        """
        This will be used to construct the 'output_dir' parameter
        of the Trainer config class.
        """
        fusion_features_config_validation = (
            self.config.fusion_features_config_collection.validation
        )

        path = pathlib.Path(
            f"vdata"  # This denotes for which dataset the predictions are made
            f"-{fusion_features_config_validation.embeddings_config.dataset_desc}",
            f"vfeatures"  # This denotes which features are used for prediction
            f"-{fusion_features_config_validation.final_features_descriptor_string}",
        )

        return path

    def get_training_description_partial_path(
        self,
    ) -> pathlib.Path:
        fusion_features_config_train = (
            self.config.fusion_features_config_collection.training
        )

        path = pathlib.Path(
            f"tdata-"  # This denotes which data is used for training
            f"{fusion_features_config_train.embeddings_config.dataset_desc}",
            f"tfeatures-"  # This denotes which features are used for training
            f"{fusion_features_config_train.final_features_descriptor_string}",
            f"labels-{labels_column_name_for_file_path_dict[self.config.labels_column_name]}",
            f"lvl-{fusion_features_config_train.embeddings_config.level}",
            f"ctxt-{fusion_features_config_train.embeddings_config.context}",
            f"model-{fusion_features_config_train.embeddings_config.embedding_model_identifier}",
            self.construct_partialdropout_config_desc(),
            self.construct_inproj_config_desc(),
            self.construct_tagger_config_desc(),
            self.construct_training_parameters_desc(),
            f"seed-{self.config.seed}",
        )

        return path

    def get_model_files_dir_absolute_path(
        self,
        base_path: os.PathLike = term_extraction_base_path,
    ) -> pathlib.Path:
        """
        Construct the 'model_files' folder path.
        """

        model_files_root_folder_relative_path = pathlib.Path(
            self.get_output_dir_absolute_path(
                base_path=base_path,
            ),
            "model_files",
        )

        return model_files_root_folder_relative_path

    def get_dataloaders_dir_absolute_path(
        self,
        base_path: os.PathLike = term_extraction_base_path,
    ) -> pathlib.Path:
        """
        Construct the 'dataloaders' folder path.
        """

        dataloaders_root_folder_relative_path = pathlib.Path(
            self.get_output_dir_absolute_path(
                base_path=base_path,
            ),
            "dataloaders",
        )

        return dataloaders_root_folder_relative_path

    def get_tensorboard_dir_absolute_path(
        self,
        base_path: os.PathLike = term_extraction_base_path,
    ) -> pathlib.Path:
        path = pathlib.Path(
            self.get_output_dir_absolute_path(
                base_path=base_path,
            ),
            "tb_logs",
        )

        return path

    def get_logging_file_absolute_path(
        self,
        base_path: os.PathLike = term_extraction_base_path,
    ) -> pathlib.Path:
        path = pathlib.Path(
            self.get_output_dir_absolute_path(
                base_path=base_path,
            ),
            "run.log",
        )

        return path

    def get_metrics_dir_absolute_path(
        self,
        base_path: os.PathLike = term_extraction_base_path,
    ) -> pathlib.Path:
        path = pathlib.Path(
            self.get_output_dir_absolute_path(
                base_path=base_path,
            ),
            "metrics",
        )

        return path

    def get_best_model_scores_dir_absolute_path(
        self,
        base_path: os.PathLike = term_extraction_base_path,
    ) -> pathlib.Path:
        path = pathlib.Path(
            self.get_metrics_dir_absolute_path(
                base_path=base_path,
            ),
            "best_model_scores",
        )

        return path

    def get_model_predictions_dir_absolute_path(
        self,
        base_path: os.PathLike = term_extraction_base_path,
    ) -> pathlib.Path:
        path = pathlib.Path(
            self.get_output_dir_absolute_path(
                base_path=base_path,
            ),
            "model_predictions",
        )

        return path

    def construct_model_description_string(
        self,
    ) -> str:
        """
        Construct the 'model_description_string'.
        """

        model_description_string = (
            f"tdata"
            f"-{self.config.fusion_features_config_for_file_paths.embeddings_config.dataset_desc}"
            f"_model-{self.config.fusion_features_config_for_file_paths.embeddings_config.embedding_model_identifier}"
            f"_tfeatures"  # This denotes which features are used for training
            f"-{self.config.fusion_features_config_for_file_paths.final_features_descriptor_string}"
        )

        model_description_string = truncate_length_of_desc(
            desc=model_description_string,
            logger=self.logger,
        )

        return model_description_string

    def construct_partialdropout_config_desc(
        self,
    ) -> str:
        """
        Construct the 'partialdropout_config_desc'.
        """

        partialdropout_config_desc = (
            f"partialdrop"
            f"_coordrop-{str(self.config.coordinate_dropout_p)}"
            f"_coordropK-{str(self.config.coordinate_dropout_K)}"
            f"_coordrops-{str(self.config.apply_scaling_coordinate_dropout)}"
            f"_tokdrop-{str(self.config.token_dropout_p)}"
            f"_tokdropK-{str(self.config.token_dropout_K)}"
            f"_tokdrops-{str(self.config.apply_scaling_token_dropout)}"
        )

        partialdropout_config_desc = truncate_length_of_desc(
            desc=partialdropout_config_desc,
            logger=self.logger,
        )

        return partialdropout_config_desc

    def construct_inproj_config_desc(
        self,
    ) -> str:
        """
        Construct the 'inproj_config_desc'.
        """

        inproj_config_desc = (
            f"inproj-{self.config.in_projection_type}"
            f"_extra-{self.config.num_extra_in_projection_layers}"
            f"_lnorm-{self.config.additional_layer_norm_for_in_projection}"
            f"_indrop-{self.config.in_projection_dropout}"
            f"_sdims-{list_of_ints_to_filename_str(self.config.slice_dims)}"
            f"_sed-{self.config.slice_encoder_depth}"
            f"_snt-{self.config.slice_norm_type}"
            f"_escomb-{self.config.encoded_slice_combination_mode}"
        )

        inproj_config_desc = truncate_length_of_desc(
            desc=inproj_config_desc,
            logger=self.logger,
        )

        return inproj_config_desc

    def construct_tagger_config_desc(
        self,
    ) -> str:
        """
        Construct the 'tagger_config_desc'.
        """

        tagger_config_desc = (
            f"tagger-{self.config.tagger_type}"
            f"_hsize-{self.config.roberta_hidden_size}"
            f"_hheads-{self.config.roberta_num_attention_heads}"
            f"_hlayers-{self.config.roberta_num_hidden_layers}"
            f"_clsdrop-{self.config.classifier_dropout}"
        )

        tagger_config_desc = truncate_length_of_desc(
            desc=tagger_config_desc,
            logger=self.logger,
        )

        return tagger_config_desc

    def construct_training_parameters_desc(
        self,
    ) -> str:
        """
        Constructs a string description of the training parameters
        based on the current configuration.

        Returns:
            A string description of the training parameters.
        """
        training_parameters_desc = (
            f"tparam"
            f"_ep-{self.config.num_train_epochs}"
            f"_lr-{self.config.learning_rate}"
            f"_lrsch-{str(self.config.lr_scheduler_type)}"
            f"_warmup-{self.config.warmup_proportion}"
            f"_wd-{self.config.weight_decay}"
            f"_tbs-{self.config.train_batch_size}"
        )

        training_parameters_desc = truncate_length_of_desc(
            desc=training_parameters_desc,
            logger=self.logger,
        )

        return training_parameters_desc
