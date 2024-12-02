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

import datetime
import pathlib
import shutil
from pathlib import Path

from tqdm import tqdm

from topollm.config_classes.constants import TOPO_LLM_REPOSITORY_BASE_PATH


def backup_and_zip_and_copy_list_of_folders(
    folder_paths: list[pathlib.Path],
    dropbox_base_dir: str,
    *,
    remove_backup_from_original_location: bool = False,
) -> None:
    """Backup and zip a list of folders and copy the zip files to a Dropbox directory.

    Copies a folder and its contents to a new backup directory with a timestamped name,
    zips the backup folder, and copies the zip file to a specified Dropbox directory.

    Args:
        folder_paths:
            List of folder paths to backup and zip.
        dropbox_base_dir:
            Base Dropbox directory where zipped archives are copied.
        remove_backup_from_original_location:
            If True, the backup folder and zip file are deleted from the original location.

    """
    # Create the current time string first, so that it is the same for all folders
    current_time: str = datetime.datetime.now(tz=datetime.UTC).strftime(
        format="%Y-%m-%d.%H%M%S",
    )

    for folder_path in tqdm(iterable=folder_paths):
        source_folder = Path(folder_path)
        if not source_folder.is_dir():
            print(  # noqa: T201 - we want this script to print
                f">>> Skipping: {source_folder = } is not a valid directory.",
            )
            continue

        parent_dir = source_folder.parent
        folder_name = source_folder.name
        backup_folder_name = f"{folder_name}.backup.{current_time}"
        backup_folder = parent_dir / backup_folder_name

        # Step 1: Copy folder to a new backup directory with a timestamped name
        print(  # noqa: T201 - we want this script to print
            f">>> Copying\n{source_folder = }\nto\n{backup_folder = }\n...",
        )
        shutil.copytree(
            src=source_folder,
            dst=backup_folder,
        )
        print(  # noqa: T201 - we want this script to print
            f">>> Copying\n{source_folder = }\nto\n{backup_folder = }\nDONE",
        )

        # Step 2: Create a zip archive from the backup folder
        zip_file_name = f"{backup_folder_name}.zip"
        zip_file_path = parent_dir / zip_file_name

        print(  # noqa: T201 - we want this script to print
            f">>> Creating zip archive at {zip_file_path = } ...",
        )
        shutil.make_archive(
            base_name=str(object=backup_folder),
            format="zip",
            root_dir=backup_folder,
        )
        print(  # noqa: T201 - we want this script to print
            f">>> Creating zip archive at {zip_file_path = } DONE",
        )

        # Step 3: Copy the zip archive to the Dropbox directory
        dropbox_target_dir = Path(dropbox_base_dir) / folder_name
        dropbox_target_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        dropbox_target_path = dropbox_target_dir / zip_file_name

        print(  # noqa: T201 - we want this script to print
            f">>> Copying zip archive to Dropbox directory {dropbox_target_path = } ...",
        )
        shutil.copy(
            src=zip_file_path,
            dst=dropbox_target_path,
        )
        print(  # noqa: T201 - we want this script to print
            f">>> Copying zip archive to Dropbox directory {dropbox_target_path = } DONE",
        )

        # Optional cleanup:
        # Delete the backup directory and zip file from the original location
        if remove_backup_from_original_location:
            shutil.rmtree(
                backup_folder,
            )
            pathlib.Path(zip_file_path).unlink()


def main() -> None:
    """Apply the backup_and_zip_and_copy_list_of_folders function."""
    input_folders: list[pathlib.Path] = [
        pathlib.Path(
            TOPO_LLM_REPOSITORY_BASE_PATH,
            "data/analysis/sample_sizes/run_general_comparisons",
        ),
        pathlib.Path(
            TOPO_LLM_REPOSITORY_BASE_PATH,
            "data/saved_plots/mean_estimates_over_different_checkpoints",
        ),
        pathlib.Path(
            TOPO_LLM_REPOSITORY_BASE_PATH,
            "data/saved_plots/mean_estimates_over_different_data_subsampling_number_of_samples",
        ),
    ]
    dropbox_directory = (
        "/Users/ruppik/Library/CloudStorage/Dropbox/05_Sharing/Project - Topo_LLM - TDA in LLMs - Shared Folder/results"
    )
    backup_and_zip_and_copy_list_of_folders(
        folder_paths=input_folders,
        dropbox_base_dir=dropbox_directory,
        remove_backup_from_original_location=True,
    )


if __name__ == "__main__":
    main()
