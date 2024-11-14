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

"""Configuration for synchronization including VM hostname and base data directory."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass
class SyncConfig:
    """Configuration for synchronization including VM hostname and base data directory."""

    local_data_dir: str
    gc_vm_hostname: str
    gc_vm_data_dir: str

    @staticmethod
    def load_from_env(
        local_data_dir_overwrite: str | None = None,
    ) -> "SyncConfig":
        """Load environment variables and initialize SyncConfig.

        Returns:
            SyncConfig: The synchronization configuration instance.

        """
        load_dotenv()

        # # # #
        # Optional variables
        local_data_dir: str | None = os.getenv(
            key="LOCAL_TOPO_LLM_DATA_DIR",
        )

        if local_data_dir_overwrite:
            local_data_dir = local_data_dir_overwrite

        if not local_data_dir:
            msg = "LOCAL_TOPO_LLM_DATA_DIR environment variable is not set and no overwrite provided."
            raise ValueError(
                msg,
            )

        vm_hostname: str | None = os.getenv(
            key="GC_DEV_VM_HOSTNAME",
        )
        gc_vm_data_dir: str | None = os.getenv(
            key="GC_DEV_VM_DATA_DIR",
        )

        if not vm_hostname:
            msg = "GC_DEV_VM_HOSTNAME environment variable is not set."
            raise ValueError(
                msg,
            )
        if not gc_vm_data_dir:
            msg = "GC_DEV_VM_DATA_DIR environment variable is not set"
            raise ValueError(
                msg,
            )

        return SyncConfig(
            local_data_dir=local_data_dir,
            gc_vm_hostname=vm_hostname,
            gc_vm_data_dir=gc_vm_data_dir,
        )
