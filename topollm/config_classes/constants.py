"""Script for setting global variables for the config files."""

import logging
import os

from dotenv import load_dotenv

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Globals

load_dotenv()

# # # #
# The following options are inspired by the hydra framework
# for customizing the output directory.
# https://hydra.cc/docs/configure_hydra/workdir/
#
# key-value separator for paths
KV_SEP: str = "="
# item separator for paths
ITEM_SEP: str = "_"

# # # #
# This dictionary of prefixes allows us to
# easily change the prefixes for file paths and names
# in one place without modifying the functions itself,
# enhancing the maintainability of the code.
#
# Note: This should NOT come with key-value separators,
# which we keep separate to make them configurable.
NAME_PREFIXES: dict[
    str,
    str,
] = {
    "array_truncation_size": "a-tr-s",
    "add_prefix_space": "add-prefix-space",
    "add_prefix_space_short": "aps",
    "aggregation": "agg",
    "attention_probs_dropout_prob": "attn-dr",
    "batch_size": "bs",
    "batch_size_eval": "bs-eval",
    "batch_size_test": "bs-test",
    "batch_size_train": "bs-train",
    "center": "center",
    "classifier_dropout": "clf-dr",
    "context": "ctxt",
    "data": "data",
    "dataloader_desc": "dataloader",
    "data_splitting_mode": "spl-mode",
    "data_filtering_remove_empty_sequences": "rm-empty",
    "deduplication_mode": "dedup",
    "description": "desc",
    "dropout_mode": "dr",
    "embedding_data_handler_mode": "edh-mode",
    "epoch": "ep",
    "feature_column_name": "feat-col",
    "FinetuningMode": "ftm",
    "global_step": "gs",
    "GradientModifierMode": "gradmod",
    "hidden_dropout_prob": "h-dr",
    "label_map_description": "labelmap",
    "layer": "layer",
    "learning_rate": "lr",
    "lr_scheduler_type": "lr-scheduler-type",
    "level": "lvl",
    "local_estimates_noise_artificial_noise_mode": "noise",
    "local_estimates_noise_distortion_parameter": "distor",
    "local_estimates_noise_seed": "seed",
    "lora_alpha": "alpha",
    "lora_dropout": "lora-dr",
    "lora_r": "r",
    "lora_target_modules": "lora-target",
    "metric": "metric",
    "model": "model",
    "model_parameters": "mparam",
    "masking_mode": "mask",
    "max_length": "max-len",
    "max_length_short": "mx",
    "normalization": "norm",
    "number_of_samples": "samples",
    "num_samples": "samples",
    "n_neighbors": "n-neighbors",
    "n_neighbors_mode": "n-neighbors-mode",
    "query": "query",
    "sampling_mode": "sampling",
    "sampling_seed": "sampling-seed",
    "seed": "seed",
    "split": "split",
    "split_shuffle": "spl-shuf",
    "split_seed": "spl-seed",
    "target_modules_to_freeze": "target-freeze",
    "target_modules_to_freeze_short": "freeze",
    "task_type": "task",
    "test_short": "te",
    "transformation": "trans",
    "train_short": "tr",
    "use_canonical_values_from_dataset": "use-canonical-val",
    "use_rslora": "rslora",
    "validation_short": "va",
    "weight_decay": "wd",
    "zero_vector_handling_mode": "zerovec",
}

# The values in this dictionary can be used to uniquely identify
# the full descriptions of the augmented descriptions,
# for example, for the checkpoints they also contain the prefix
# "model_" to make them unique.
# They will usually be used as column headers in the analysis dataframes.
NAME_PREFIXES_TO_FULL_AUGMENTED_DESCRIPTIONS: dict[
    str,
    str,
] = {
    # Data parameters
    "data": "data_dataset_name",
    "dataset_seed": "data_dataset_seed",
    "data_debug": "data_debug",
    "use_context": "data_use_context",
    "rm-empty": "data_filtering_remove_empty_sequences",
    "spl-mode": "data_splitting_mode",
    "data_ctxt": "data_context",
    "feat-col": "data_feature_column",
    # Model parameters
    "model_ckpt": "model_checkpoint",
    "model_seed": "model_seed",
    # Dropout parameters
    "dr": "model_dropout_mode",
    "h-dr": "model_hidden_dropout_prob",
    "attn-dr": "model_attention_probs_dropout_prob",
    "clf-dr": "model_classifier_dropout",
    # Noise parameters
    "local_estimates_dedup": "local_estimates_deduplication",
    "local_estimates_desc": "local_estimates_description",
    "local_estimates_distor": "local_estimates_noise_distortion",
    "local_estimates_noise": "local_estimates_noise_artificial_noise_mode",
    "local_estimates_seed": "local_estimates_noise_seed",
    "local_estimates_samples": "local_estimates_samples",
    "local_estimates_zerovec": "local_estimates_zero_vector_handling_mode",
}

# # # #
# Limit for length of file names
FILE_NAME_TRUNCATION_LENGTH: int = 200

EXTERNAL_DRIVE_TOPO_LLM_REPOSITORY_BASE_PATH: str = os.getenv(
    key="EXTERNAL_DRIVE_TOPO_LLM_REPOSITORY_BASE_PATH",
    default="/Volumes/${USER}_external/research_data/Topo_LLM",
)
TOPO_LLM_REPOSITORY_BASE_PATH: str = os.path.expandvars(
    path=os.getenv(
        key="TOPO_LLM_REPOSITORY_BASE_PATH",
        default="${HOME}/git-source/Topo_LLM",
    ),
)
# Note that the 'data/' directory can be set in the hydra config
ZIM_TOPO_LLM_REPOSITORY_BASE_PATH: str = os.path.expandvars(
    path=os.getenv(
        key="ZIM_TOPO_LLM_REPOSITORY_BASE_PATH",
        default="/gpfs/project/${ZIM_USERNAME}/git-source/Topo_LLM",
    ),
)

HYDRA_CONFIGS_BASE_PATH: str = f"{TOPO_LLM_REPOSITORY_BASE_PATH}/configs"

logger_section_separation_line = 30 * "="
