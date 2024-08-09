"""Syncing parts of the analysis directory from HHU Hilbert to local machine."""

import argparse
import subprocess
import sys

from topollm.config_classes.constants import (
    EXTERNAL_DRIVE_TOPO_LLM_REPOSITORY_BASE_PATH,
    TOPO_LLM_REPOSITORY_BASE_PATH,
    ZIM_TOPO_LLM_REPOSITORY_BASE_PATH,
)


def sync_directories(
    directories: list[str],
    target_base_path: str,
    dry_run_option: str,
    file_type: str,
    zim_topo_llm_repository_base_path: str,
) -> None:
    """Sync directories from HHU Hilbert to the local machine."""
    # Determine the include pattern based on file type
    include_pattern = []
    if file_type == "pkl":
        include_pattern = ["--include='*.pkl'"]
    elif file_type == "npy":
        include_pattern = ["--include='*.npy'"]
    else:
        include_pattern = ["--include='*'"]

    for directory in directories:
        print("===========================================================")  # noqa: T201 - this script should print to stdout
        print(f"{directory = }")  # noqa: T201 - this script should print to stdout

        src = f"Hilbert-Storage:{zim_topo_llm_repository_base_path}/{directory}"
        dest = f"{target_base_path}/{directory}"

        rsync_command = [
            "rsync",
            "-avz",
            "--progress",
            dry_run_option,
            *include_pattern,
            "--exclude='*'",
            src,
            dest,
        ]

        # Remove empty strings from the command
        rsync_command = [arg for arg in rsync_command if arg]

        print(  # noqa: T201 - this script should print to stdout
            f"Running command: {rsync_command = }",
        )

        try:
            result = subprocess.run(  # noqa: S603 - we trust the command
                rsync_command,
                capture_output=True,
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
            sys.exit(e.returncode)

        if result.returncode != 0:
            print(  # noqa: T201 - this script should print to stdout
                f"Error syncing {directory = }:",
                result.stderr,
                file=sys.stderr,
            )
            sys.exit(result.returncode)

        print("===========================================================")  # noqa: T201 - this script should print to stdout


def main() -> None:
    """Run the main function."""
    args = parse_arguments()

    target_base_path = (
        EXTERNAL_DRIVE_TOPO_LLM_REPOSITORY_BASE_PATH if args.sync_to_external_drive else TOPO_LLM_REPOSITORY_BASE_PATH
    )
    dry_run_option = "--dry-run" if args.dry_run else ""

    directories_to_sync = args.directories
    if args.skip_prepared:
        directories_to_sync = [d for d in directories_to_sync if d != "data/analysis/prepared/"]

    print(f"{TOPO_LLM_REPOSITORY_BASE_PATH = }")  # noqa: T201 - this script should print to stdout
    print(f"{ZIM_TOPO_LLM_REPOSITORY_BASE_PATH = }")  # noqa: T201 - this script should print to stdout

    sync_directories(
        directories=directories_to_sync,
        target_base_path=target_base_path,
        dry_run_option=dry_run_option,
        file_type=args.file_type,
        zim_topo_llm_repository_base_path=ZIM_TOPO_LLM_REPOSITORY_BASE_PATH,
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
        "--dry-run",
        action="store_true",
        help="Perform a dry run.",
    )
    parser.add_argument(
        "--skip-prepared",
        action="store_true",
        help="Skip 'data/analysis/prepared/' directory.",
    )
    parser.add_argument(
        "--directories",
        nargs="+",
        default=[
            "data/analysis/prepared/",
            "data/analysis/twonn/",
            "data/saved_plots/",
        ],
        help="List of directories to sync.",
    )
    parser.add_argument(
        "--file-type",
        choices=[
            "all",
            "pkl",
            "npy",
        ],
        default="all",
        help="File type to sync: 'all' for all files, 'pkl' for .pkl files, 'npy' for .npy files.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
