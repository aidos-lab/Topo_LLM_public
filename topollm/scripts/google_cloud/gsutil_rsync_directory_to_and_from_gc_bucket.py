# Copyright 2024-2025
# [ANONYMIZED_INSTITUTION],
# [ANONYMIZED_FACULTY],
# [ANONYMIZED_DEPARTMENT]
#
# Authors:
# AUTHOR_1 (author1@example.com)
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

"""Sync data through Google Cloud Storage."""

import argparse
import os
import pprint
import subprocess
import sys
import time
from dataclasses import dataclass
from enum import StrEnum, auto

from tqdm import tqdm

from topollm.scripts.google_cloud.sync_config import SyncConfig


class SyncSource(StrEnum):
    """Enum to define valid sync sources."""

    VM = auto()
    BUCKET = auto()
    LOCAL = auto()


class SyncTarget(StrEnum):
    """Enum to define valid sync targets."""

    BUCKET = auto()
    VM = auto()
    LOCAL = auto()


def run_gsutil_rsync(
    source_path: str,  # Note: These should be strings, since we do not use pathlib on cloud storage URIs
    destination_path: str,  # Note: These should be strings, since we do not use pathlib on cloud storage URIs
    *,
    dry_run: bool = False,
) -> None:
    """Run the gsutil rsync command between source and destination.

    Args:
        source_path:
            Path to source directory.
        destination_path:
            Path to destination directory.
        dry_run:
            Whether to perform a dry run of the operation.

    """
    # Note: We use "gcloud storage rsync" instead of the obsolete "gsutil rsync".
    # The gcloud storage command does not need the '-m' flag for multithreading.
    command: list[str] = [
        "gcloud",
        "storage",
        "rsync",
        "-r",
    ]

    # Add the dry-run flag if requested
    if dry_run:
        # Note that "gcloud storage rsync" uses the "--dry-run" flag instead of "-n" in "gsutil rsync".
        command.append(
            "--dry-run",
        )

    # Add source and destination paths
    command.extend(
        [
            str(source_path),
            str(destination_path),
        ],
    )

    # Print the command for verification
    command_str: str = " ".join(
        command,
    )
    print(  # noqa: T201 - we want this script to print to stdout
        f">>> Command to be executed:\n{command_str}",
    )

    # Countdown for 3 seconds before executing
    for i in range(3, 0, -1):
        print(  # noqa: T201 - we want this script to print to stdout
            f">>> Executing in {i} seconds... Press Ctrl+C to cancel.",
        )
        time.sleep(1)

    # Run the command
    try:
        subprocess.run(
            args=command,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(  # noqa: T201 - we want this script to print to stdout
            f"Error occurred during synchronization: {e}",
        )
        sys.exit(1)


@dataclass
class SyncManager:
    """Manager class to control synchronization between local, GCP VM, and GCP bucket."""

    config: SyncConfig

    def sync(
        self,
        source: SyncSource,
        target: SyncTarget,
        *,
        dry_run: bool = False,
        subdirectory: str = "data/analysis/twonn",
    ) -> None:
        """Perform synchronization between source and target.

        Args:
            source:
                The name of the sync source.
            target:
                The name of the sync target.
            dry_run:
                Whether to perform a dry run.
            subdirectory:
                Relative subdirectory to sync within the source directory.

        """
        # Warning: Using os.path.join here to avoid issues with URI paths such as gs://.
        # pathlib's `Path` is not suitable for cloud storage URIs (gs://) as it strips slashes improperly.

        # Determine source and target paths based on SyncConfig
        if source == SyncSource.VM and target == SyncTarget.BUCKET:
            print(  # noqa: T201 - we want this script to print to stdout
                ">>> Syncing from Google Cloud VM to Google Cloud Storage Bucket ...",
            )
            source_path = os.path.join(  # noqa: PTH118 - do not use pathlib on cloud storage URIs
                self.config.gc_vm_repository_base_path,
                subdirectory,
            )
            destination_path = os.path.join(  # noqa: PTH118 - do not use pathlib on cloud storage URIs
                self.config.gc_bucket_repository_base_path,
                subdirectory,
            )

        elif source == SyncSource.BUCKET and target == SyncTarget.LOCAL:
            print(  # noqa: T201 - we want this script to print to stdout
                ">>> Syncing from Google Cloud Storage Bucket to Local Machine ...",
            )
            source_path = os.path.join(  # noqa: PTH118 - do not use pathlib on cloud storage URIs
                self.config.gc_bucket_repository_base_path,
                subdirectory,
            )
            destination_path = os.path.join(  # noqa: PTH118 - do not use pathlib on cloud storage URIs
                self.config.local_repository_base_path,
                subdirectory,
            )

        elif source == SyncSource.BUCKET and target == SyncTarget.VM:
            print(  # noqa: T201 - we want this script to print to stdout
                ">>> Syncing from Google Cloud Storage Bucket to Google Cloud VM ...",
            )
            source_path = os.path.join(  # noqa: PTH118 - do not use pathlib on cloud storage URIs
                self.config.gc_bucket_repository_base_path,
                subdirectory,
            )
            destination_path = os.path.join(  # noqa: PTH118 - do not use pathlib on cloud storage URIs
                self.config.gc_vm_repository_base_path,
                subdirectory,
            )

        elif source == SyncSource.LOCAL and target == SyncTarget.BUCKET:
            print(  # noqa: T201 - we want this script to print to stdout
                ">>> Syncing from Local Machine to Google Cloud Storage Bucket ...",
            )
            source_path = os.path.join(  # noqa: PTH118 - do not use pathlib on cloud storage URIs
                self.config.local_repository_base_path,
                subdirectory,
            )
            destination_path = os.path.join(  # noqa: PTH118 - do not use pathlib on cloud storage URIs
                self.config.gc_bucket_repository_base_path,
                subdirectory,
            )

        else:
            print(  # noqa: T201 - we want this script to print to stdout
                f"@@@ Invalid combination of source '{source = }' and target '{target = }'.",
            )
            print(  # noqa: T201 - we want this script to print to stdout
                "@@@ Valid combinations are: "
                "'vm' to 'bucket', 'bucket' to 'local', 'bucket' to 'vm', or 'local' to 'bucket'.",
            )
            sys.exit(1)

        # Execute the sync command
        run_gsutil_rsync(
            source_path=source_path,
            destination_path=destination_path,
            dry_run=dry_run,
        )

        print(  # noqa: T201 - we want this script to print to stdout
            ">>> Sync complete.",
        )


def parse_command_line_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Sync files between Google Cloud VM, Bucket, and Local.",
    )
    parser.add_argument(
        "--source",
        type=SyncSource,
        choices=list(SyncSource),
        required=True,
        help="The source of the sync: 'vm', 'bucket', or 'local'.",
    )
    parser.add_argument(
        "--target",
        type=SyncTarget,
        choices=list(SyncTarget),
        required=True,
        help="The target of the sync: 'bucket', 'vm', or 'local'.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Perform a dry run of the sync operation.",
    )
    parser.add_argument(
        "--subdirectory",
        nargs="+",  # Allow multiple arguments to be passed, e.g., '--subdirectory data/analysis/twonn hydra_output_dir'
        type=str,
        default="data/analysis/twonn",
        help="Relative subdirectory to sync within the source directory.",
    )

    args: argparse.Namespace = parser.parse_args()

    return args


def main() -> None:
    """Execute the sync operation."""
    # Load SyncConfig instance from environment variables
    config: SyncConfig = SyncConfig.load_from_env()

    # Parse command-line arguments
    args: argparse.Namespace = parse_command_line_arguments()

    # The subdirectory argument is a list of strings,
    # since we allow multiple subdirectories to be passed.
    subdirectory_list: list[str] = args.subdirectory

    print(  # noqa: T201 - we want this script to print to stdout
        ">>> args.subdirectory:\n",
        pprint.pformat(object=subdirectory_list),
    )

    # Instantiate the SyncManager and execute sync
    sync_manager = SyncManager(
        config=config,
    )

    for subdirectory in tqdm(
        iterable=subdirectory_list,
        desc="Syncing subdirectories",
    ):
        sync_manager.sync(
            source=args.source,
            target=args.target,
            dry_run=args.dry_run,
            subdirectory=subdirectory,
        )


if __name__ == "__main__":
    main()
