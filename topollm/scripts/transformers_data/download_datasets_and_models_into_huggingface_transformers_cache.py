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

"""Download datasets and models into huggingface transformers cache.

This is convenient for working with HHU ZIM, because we do not have internet access on the cluster,
so we need to download the datasets and models to our local machines and then copy them to the cluster.
"""

import argparse
import logging
import pathlib

import datasets
import transformers
from tqdm import tqdm

from topollm.logging.setup_exception_logging import setup_exception_logging

global_logger: logging.Logger = logging.getLogger(
    name=__name__,
)
global_logger.setLevel(
    level=logging.INFO,
)
logging_formatter = logging.Formatter(
    fmt="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
)

logging_file_path = pathlib.Path(
    pathlib.Path(__file__).parent,
    "logs",
    f"{pathlib.Path(__file__).stem}.log",
)
pathlib.Path.mkdir(
    self=logging_file_path.parent,
    parents=True,
    exist_ok=True,
)

logging_file_handler = logging.FileHandler(
    filename=logging_file_path,
)
logging_file_handler.setFormatter(
    fmt=logging_formatter,
)
global_logger.addHandler(
    hdlr=logging_file_handler,
)

logging_console_handler = logging.StreamHandler()
logging_console_handler.setFormatter(
    fmt=logging_formatter,
)
global_logger.addHandler(
    hdlr=logging_console_handler,
)


setup_exception_logging(
    logger=global_logger,
)

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def main() -> None:
    """Run through downloads."""
    logger: logging.Logger = global_logger

    args: argparse.Namespace = get_command_line_arguments()

    for dataset_name in tqdm(
        args.dataset_names,
        desc="Iterating over dataset names",
    ):
        logger.info(
            msg=f"Downloading {dataset_name = } into huggingface transformers cache ...",  # noqa: G004 - low overhead
        )

        match dataset_name:
            case "SocialGrep/one-year-of-tsla-on-reddit":
                name = "comments"
            case "wikitext":
                name = "wikitext-103-v1"
            case _:
                name = None
                logger.info(
                    msg=f"No specific name necessary, using {name = } for {dataset_name = }.",  # noqa: G004 - low overhead
                )

        logger.info(
            msg=f"{dataset_name = }; {name = }",  # noqa: G004 - low overhead
        )

        dataset = datasets.load_dataset(
            path=dataset_name,
            name=name,
            download_mode=datasets.DownloadMode.FORCE_REDOWNLOAD,
            trust_remote_code=True,
        )
        logger.info(
            msg=f"dataset:\n{dataset}",  # noqa: G004 - low overhead
        )

        logger.info(
            msg=f"Downloading {dataset_name = } into huggingface transformers cache DONE",  # noqa: G004 - low overhead
        )

    for model_name in tqdm(
        args.model_names,
        desc="Iterating over model names",
    ):
        logger.info(
            msg=f"Downloading {model_name = } into huggingface transformers cache ...",  # noqa: G004 - low overhead
        )

        model = transformers.AutoModel.from_pretrained(
            pretrained_model_name_or_path=model_name,
            force_download=True,
        )
        logger.info(
            msg=f"model:\n{model}",  # noqa: G004 - low overhead
        )

        logger.info(
            msg=f"Downloading {model_name = } into huggingface transformers cache DONE",  # noqa: G004 - low overhead
        )


def get_command_line_arguments() -> argparse.Namespace:
    """Get command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run downloads into huggingface transformers cache.",
    )

    parser.add_argument(
        "--dataset_names",
        nargs="+",
        type=str,
        default=[
            "SocialGrep/one-year-of-tsla-on-reddit",
        ],
        required=False,
        help="Names of the datasets.",
    )
    parser.add_argument(
        "--model_names",
        nargs="+",
        type=str,
        default=[
            "roberta-base",
        ],
        required=False,
        help="Names of the models.",
    )

    args: argparse.Namespace = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
