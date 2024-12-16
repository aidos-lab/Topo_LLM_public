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

"""Syncing parts of the analysis directory from HHU Hilbert to local machine."""

import argparse
import pprint
import subprocess
import sys

from tqdm import tqdm

from topollm.config_classes.constants import (
    EXTERNAL_DRIVE_TOPO_LLM_REPOSITORY_BASE_PATH,
    TOPO_LLM_REPOSITORY_BASE_PATH,
    ZIM_TOPO_LLM_REPOSITORY_BASE_PATH,
)


def sync_directories(
    directories: list[str],
    zim_topo_llm_repository_base_path: str,
    target_base_path: str,
    dry_run_option: str,
    file_type: str,
) -> None:
    """Sync directories from HHU Hilbert to the local machine."""
    # Determine the include pattern based on file type.
    # See the following link for more information about include and exclude patterns in rsync:
    # https://stackoverflow.com/a/51480550/10011325
    #
    # E.g., this command *in the shell* only syncs the .npy files:
    # > rsync \
    # >     -avz --progress --dry-run
    # >     --include="*/" --include="*.npy" --exclude="*"
    # >     Hilbert-Storage:/gpfs/project/$USER/git-source/Topo_LLM/data/analysis/twonn/
    # >     /Users/$USER/git-source/Topo_LLM/data/analysis/twonn/`
    #
    # Note:
    # Do not use quotes in the include/exclude patterns here,
    # since we will run the command directly without using a shell.
    # The quotes are only needed in the shell to prevent the shell from expanding the patterns.
    include_pattern: list[str] = []

    if file_type == "pkl":
        include_pattern: list[str] = [
            "--include=*/",  # do not skip any directories
            "--include=*.pkl",  # do not skip any .pkl files
        ]
    elif file_type == "npy":
        include_pattern = [
            "--include=*/",
            "--include=*.npy",
        ]
    else:
        include_pattern = [
            "--include=*/",
            "--include=*",
        ]

    for directory in tqdm(
        iterable=directories,
        desc="Syncing directories",
    ):
        print(  # noqa: T201 - this script should print to stdout
            "===========================================================",
        )
        print(  # noqa: T201 - this script should print to stdout
            f"{directory = }",
        )

        execute_single_directory_sync(
            directory=directory,
            zim_topo_llm_repository_base_path=zim_topo_llm_repository_base_path,
            target_base_path=target_base_path,
            dry_run_option=dry_run_option,
            include_pattern=include_pattern,
        )

        print(  # noqa: T201 - this script should print to stdout
            "===========================================================",
        )


def sync_only_results_arrays_and_statistics(
    directories: list[str],
    zim_topo_llm_repository_base_path: str,
    target_base_path: str,
    dry_run_option: str,
) -> None:
    """Specific sync which is meant for the twonn directory.

    This sync skips the large array_for_estimator.npy files and the metadata files.
    """
    include_pattern_to_iterate_over: list[list[str]] = [
        # The statistics files are saved as .json files and are very small, so we include all files of this type.
        [
            "--include=*/",
            "--include=*.json",
        ],
        # Note that we cannot restrict to all .npy files,
        # since the very large array_for_estimator.npy files are also .npy files.
        # The local estimates files are recognized by their names.
        [
            "--include=*/",
            "--include=local_estimates_pointwise_array.npy",
        ],
        # The global estimates files are recognized by their names.
        [
            "--include=*/",
            "--include=global_estimate.npy",
        ],
    ]

    for include_pattern in tqdm(
        iterable=include_pattern_to_iterate_over,
        desc="Iterating over include patterns",
    ):
        for directory in tqdm(
            iterable=directories,
            desc="Syncing directories",
        ):
            print(  # noqa: T201 - this script should print to stdout
                "===========================================================",
            )
            print(  # noqa: T201 - this script should print to stdout
                f"{include_pattern = }",
            )
            print(  # noqa: T201 - this script should print to stdout
                f"{directory = }",
            )

            execute_single_directory_sync(
                directory=directory,
                zim_topo_llm_repository_base_path=zim_topo_llm_repository_base_path,
                target_base_path=target_base_path,
                dry_run_option=dry_run_option,
                include_pattern=include_pattern,
            )

            print(  # noqa: T201 - this script should print to stdout
                "===========================================================",
            )


def execute_single_directory_sync(
    directory: str,
    zim_topo_llm_repository_base_path: str,
    target_base_path: str,
    dry_run_option: str = "",
    include_pattern: list[str] | None = None,
) -> None:
    """Execute a single directory sync."""
    if include_pattern is None:
        include_pattern = []

    src: str = f"Hilbert-Storage:{zim_topo_llm_repository_base_path}/{directory}"
    dest: str = f"{target_base_path}/{directory}"

    rsync_command = [
        "rsync",
        "-zarv",
        "--progress",
        dry_run_option,
        *include_pattern,
        "--exclude=*",  # exclude everything that has not been matched by include patterns; Note: no quotes in pattern
        src,
        dest,
    ]

    # Remove empty strings from the command
    rsync_command: list[str] = [arg for arg in rsync_command if arg]

    print(  # noqa: T201 - this script should print to stdout
        f"Running command: {rsync_command = }",
    )

    try:
        result = subprocess.run(
            args=rsync_command,
            capture_output=True,
            shell=False,  # execute the command directly without using a shell, thus we need unquoted arguments
            text=True,
            check=True,
        )
        print(result.stdout)  # noqa: T201 - this script should print to stdout
    except subprocess.CalledProcessError as e:
        print(  # noqa: T201 - this script should print to stdout
            f"Error syncing {directory = }:",
        )
        print(  # noqa: T201 - this script should print to stdout
            e.stderr,
        )
        sys.exit(
            e.returncode,
        )

    if result.returncode != 0:
        print(  # noqa: T201 - this script should print to stdout
            f"Error syncing {directory = }:",
            result.stderr,
            file=sys.stderr,
        )
        sys.exit(
            result.returncode,
        )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Sync experiment results to the local directory.",
    )
    parser.add_argument(
        "--sync_to_external_drive",
        action="store_true",
        help="Sync to external hard drive.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Perform a dry run.",
    )

    parser.add_argument(
        "--directories",
        nargs="+",
        default=[
            "data/analysis/twonn/",
        ],
        help="List of directories to sync.",
    )
    parser.add_argument(
        "--syncing_mode",
        choices=[
            "all",
            "only_results_arrays_and_statistics",
        ],
        default="all",
        help="Syncing mode: Everything or only results arrays and statistics (specifically for twonn directory).",
    )
    parser.add_argument(
        "--file-type",
        choices=[
            "all",
            "pkl",
            "npy",
        ],
        default="npy",
        help="File type to sync: 'all' for all files, 'pkl' for .pkl files, 'npy' for .npy files.",
    )

    args = parser.parse_args()
    return args


def main() -> None:
    """Run the main function."""
    args: argparse.Namespace = parse_arguments()

    target_base_path = (
        EXTERNAL_DRIVE_TOPO_LLM_REPOSITORY_BASE_PATH if args.sync_to_external_drive else TOPO_LLM_REPOSITORY_BASE_PATH
    )
    dry_run_option = "--dry-run" if args.dry_run else ""

    print(  # noqa: T201 - this script should print to stdout
        f"{TOPO_LLM_REPOSITORY_BASE_PATH = }",
    )
    print(  # noqa: T201 - this script should print to stdout
        f"{ZIM_TOPO_LLM_REPOSITORY_BASE_PATH = }",
    )

    directories_to_sync = args.directories
    print(  # noqa: T201 - this script should print to stdout
        f"directories_to_sync:\n{pprint.pformat(object=directories_to_sync)}",
    )

    match args.syncing_mode:
        case "all":
            sync_directories(
                directories=directories_to_sync,
                zim_topo_llm_repository_base_path=ZIM_TOPO_LLM_REPOSITORY_BASE_PATH,
                target_base_path=target_base_path,
                dry_run_option=dry_run_option,
                file_type=args.file_type,
            )
        case "only_results_arrays_and_statistics":
            print(  # noqa: T201 - this script should print to stdout
                "Syncing only results arrays and statistics. "
                "Note that the file type argument is ignored in this mode.",
            )
            sync_only_results_arrays_and_statistics(
                directories=directories_to_sync,
                zim_topo_llm_repository_base_path=ZIM_TOPO_LLM_REPOSITORY_BASE_PATH,
                target_base_path=target_base_path,
                dry_run_option=dry_run_option,
            )


if __name__ == "__main__":
    main()
