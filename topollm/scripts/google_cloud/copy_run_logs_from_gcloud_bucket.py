# Copyright 2024
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

"""Synchronize 'run/' logs from Google Cloud Storage to local machine."""

import argparse
import logging
import os
import pathlib
import subprocess

from tqdm import tqdm

from topollm.logging.log_list_info import log_list_info
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.typing.enums import Verbosity

global_logger = logging.getLogger(__name__)
global_logger.setLevel(
    logging.INFO,
)
logging_formatter = logging.Formatter(
    "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
)

logging_file_path = pathlib.Path(
    pathlib.Path(__file__).parent,
    "logs",
    f"{pathlib.Path(__file__).stem}.log",
)
pathlib.Path.mkdir(
    logging_file_path.parent,
    parents=True,
    exist_ok=True,
)

logging_file_handler = logging.FileHandler(
    logging_file_path,
)
logging_file_handler.setFormatter(
    logging_formatter,
)
global_logger.addHandler(
    logging_file_handler,
)

logging_console_handler = logging.StreamHandler()
logging_console_handler.setFormatter(logging_formatter)
global_logger.addHandler(logging_console_handler)


setup_exception_logging(
    logger=global_logger,
)

default_logger = logging.getLogger(__name__)


def copy_run_logs_from_gcloud_bucket(
    bucket_name: str,
    local_dest_dir: str,
    pattern_to_match_in_path: str = "/runs/",
    *,
    dry_run: bool = False,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Synchronize files from a Google Cloud bucket that are contained in directories called 'run/' to a local machine.

    Args:
    ----
        bucket_name:
            The name of the Google Cloud bucket.
        local_dest_dir:
            The local directory to sync the files to.
        pattern_to_match_in_path:
            The pattern to match in the file paths.
        dry_run:
            If True, only list the files to be copied without actually copying them.
        verbosity:
            The verbosity level of the logging.
        logger:
            The logger to use for logging.

    """
    # List all files in the bucket and filter paths that contain "/run/"
    logger.info(
        f"Calling `gsutil ls -r {bucket_name}/**` ...",  # noqa: G004 - low overhead
    )
    result = subprocess.run(
        [  # noqa: S607 , S603 - we trust the PATH and the command
            "gsutil",
            "ls",
            "-r",
            f"{bucket_name}/**",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    logger.info(
        f"Calling `gsutil ls -r {bucket_name}/**` DONE",  # noqa: G004 - low overhead
    )

    all_paths: list[str] = result.stdout.splitlines()
    if verbosity >= Verbosity.NORMAL:
        log_list_info(
            all_paths,
            list_name="all_paths",
            logger=logger,
        )
    logger.info(
        f"{len(all_paths) = }",  # noqa: G004 - low overhead
    )
    filtered_paths = [path for path in all_paths if pattern_to_match_in_path in path]

    if dry_run:
        logger.info("Dry run: The following files would be copied:")
        logger.info(
            f"{len(filtered_paths) = } files in total.",  # noqa: G004 - low overhead
        )
        for file_path in filtered_paths:
            logger.info(file_path)
        return

    if verbosity >= Verbosity.NORMAL:
        log_list_info(
            filtered_paths,
            list_name="filtered_paths",
            logger=logger,
        )

    # Create directories locally to preserve the structure
    for file_path in tqdm(
        filtered_paths,
        desc="Creating directories",
    ):
        # Remove the bucket name prefix from the file path
        relative_path = file_path[len(bucket_name) + 1 :]
        # Get the directory path
        dir_path = pathlib.Path(relative_path).parent
        # Create the directory locally
        pathlib.Path.mkdir(
            pathlib.Path(
                local_dest_dir,
                dir_path,
            ),
            exist_ok=True,
            parents=True,
        )

    # Copy the filtered files to the corresponding local directories
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            "Calling `gsutil cp` commands ...",
        )
    for file_path in tqdm(
        filtered_paths,
        desc="Copying files",
    ):
        # Remove the bucket name prefix from the file path
        relative_path = file_path[len(bucket_name) + 1 :]
        # Copy the file to the local destination, preserving the directory structure
        source_path = file_path
        destination_path = os.path.join(
            local_dest_dir,
            relative_path,
        )
        try:
            subprocess.run(
                [  # noqa: S607 , S603 - we trust the PATH and the command
                    "gsutil",
                    "cp",
                    str(source_path),
                    str(destination_path),
                ],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            # Note: There appears to be a problem with those paths which contain square brackets and quotation marks in the path.
            logger.exception(
                f"Error while copying {source_path} to {destination_path}: {e}",  # noqa: G004 - low overhead
            )
            continue
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            "Calling `gsutil cp` commands DONE",
        )
        logger.info("Copying complete.")


def main() -> None:
    """Parse command line arguments and call the `sync_run_logs` function."""
    logger = global_logger

    parser = argparse.ArgumentParser(
        description="Synchronize 'run/' logs from Google Cloud Storage to local machine.",
    )

    parser.add_argument(
        "--bucket_name",
        type=str,
        default="gs://ruppik-eu/Topo_LLM",
        required=False,
        help="The name of the Google Cloud bucket.",
    )
    parser.add_argument(
        "--local_dest_dir",
        type=str,
        default="/Volumes/ruppik_external/research_data/Topo_LLM/copied_logs",
        required=False,
        help="The local directory to sync the files to.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the files to be copied without actually copying them.",
    )

    args = parser.parse_args()

    copy_run_logs_from_gcloud_bucket(
        bucket_name=args.bucket_name,
        local_dest_dir=args.local_dest_dir,
        dry_run=args.dry_run,
        logger=logger,
    )


if __name__ == "__main__":
    main()
