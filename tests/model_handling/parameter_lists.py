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
