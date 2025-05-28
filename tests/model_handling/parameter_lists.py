# Copyright 2024
# [ANONYMIZED_INSTITUTION],
# [ANONYMIZED_FACULTY],
# [ANONYMIZED_DEPARTMENT]
#
# Authors:
# AUTHOR_1 (author1@example.com)
# AUTHOR_2 (author2@example.com)
#
# Code generation tools and workflows:
# First versions of this code were potentially generated
# with the help of AI writing assistants including
# GitHub Copilot, ChatGPT, Microsoft Copilot, Google Gemini.
# Afterwards, the generated segments were manually reviewed and edited.
#


import os
import pathlib

example_pretrained_model_name_or_path_list: list[str | os.PathLike] = [
    "roberta-base",
    "gpt2-medium",
    pathlib.Path(
        pathlib.Path(__file__).parent,
        "example_model_files",
        "example_lora_finetuning_data-multiwoz21-10000_checkpoint-1200",
    ),
]
