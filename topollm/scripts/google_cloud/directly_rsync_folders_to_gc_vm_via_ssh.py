# Copyright 2024-2025
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Matthias Ruppik (mail@ruppik.net)
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

"""Submit jobs for pipeline, perplexity, or finetuning."""

import logging
import pathlib
import pprint
import subprocess
from typing import TYPE_CHECKING

import hydra
import hydra.core.hydra_config
import omegaconf

from topollm.config_classes.constants import HYDRA_CONFIGS_BASE_PATH
from topollm.config_classes.get_data_dir import get_data_dir
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.scripts.google_cloud.sync_config import SyncConfig
from topollm.typing.enums import Verbosity

if TYPE_CHECKING:
    from topollm.config_classes.main_config import MainConfig


# Logger for this file
global_logger: logging.Logger = logging.getLogger(
    name=__name__,
)
default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)

setup_exception_logging(
    logger=global_logger,
)


def build_sync_paths(
    sync_config: SyncConfig,
    sub_paths: list[str],
) -> list[tuple[str, str]]:
    """Construct source and destination paths for synchronization.

    Args:
        sync_config:
            The configuration for synchronization paths.
        sub_paths:
            Subdirectory paths to be appended to the base directory.

    Returns:
        List[Tuple[str, str]]: A list of (source, destination) path tuples.

    """
    sync_paths = []
    for sub_path in sub_paths:
        local_path = str(
            object=pathlib.Path(
                sync_config.local_data_dir,
                sub_path,
            ),
        )
        remote_path: str = f"{sync_config.gc_vm_hostname}:{pathlib.Path(sync_config.gc_vm_data_dir, sub_path)}"
        sync_paths.append(
            (local_path, remote_path),
        )

    return sync_paths


def sync_datasets(
    sync_paths: list[tuple[str, str]],
    *,
    dry_run: bool = False,
    logger: logging.Logger = default_logger,
) -> None:
    """Synchronize each pair of local and remote paths, with an optional dry-run mode.

    Args:
        sync_paths:
            List of (source, destination) path tuples.
        dry_run:
            If True, logs commands instead of executing them.
        logger:
            The logger to use for logging.

    """
    for local_path, remote_path in sync_paths:
        rsync_command: list[str] = [
            "rsync",
            "-avz",
            "--progress",
            f"{local_path}",
            f"{remote_path}",
        ]

        # Log or execute the command based on dry_run
        command_str = " ".join(rsync_command)
        if dry_run:
            logger.info(
                msg=f">>> [DRY RUN] Command:\n{command_str}",  # noqa: G004 - low overhead
            )
        else:
            logger.info(
                msg=f">>> Executing Command:\n{command_str}",  # noqa: G004 - low overhead
            )
            try:
                subprocess.run(
                    args=rsync_command,
                    check=True,
                )
                logger.info(
                    msg=f">>> Synchronization completed for {local_path = }.",  # noqa: G004 - low overhead
                )
            except subprocess.CalledProcessError as e:
                logger.exception(
                    msg=f"@@@ Error occurred during synchronization for {local_path = }:\n{e}",  # noqa: G004 - low overhead
                )


@hydra.main(
    config_path=f"{HYDRA_CONFIGS_BASE_PATH}",
    config_name="main_config",
    version_base="1.3",
)
def main(
    config: omegaconf.DictConfig,
) -> None:
    """Run the script."""
    logger: logging.Logger = global_logger
    logger.info(
        msg="Running script ...",
    )

    main_config: MainConfig = initialize_configuration(
        config=config,
        logger=logger,
    )
    verbosity: Verbosity = main_config.verbosity
    dry_run: bool = main_config.feature_flags.scripts.dry_run

    data_dir: pathlib.Path = get_data_dir(
        main_config=main_config,
        verbosity=main_config.verbosity,
        logger=global_logger,
    )

    # Load configuration and define paths
    sync_config: SyncConfig = SyncConfig.load_from_env(
        local_data_dir_overwrite=str(data_dir),
    )
    sub_paths: list[str] = [
        "datasets/dialogue_datasets",
        # Note: Additional folders can be added here
    ]

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            "sync_config:\n%s",
            pprint.pformat(object=sync_config),
        )
        logger.info(
            "sub_paths:\n%s",
            pprint.pformat(object=sub_paths),
        )

    # Build the list of source and destination paths
    sync_paths = build_sync_paths(
        sync_config=sync_config,
        sub_paths=sub_paths,
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            "sync_paths:\n%s",
            pprint.pformat(object=sync_paths),
        )

    # Synchronize each source-destination path with dry-run support
    sync_datasets(
        sync_paths=sync_paths,
        dry_run=dry_run,
    )

    logger.info(
        msg="Running script DONE",
    )


if __name__ == "__main__":
    main()
