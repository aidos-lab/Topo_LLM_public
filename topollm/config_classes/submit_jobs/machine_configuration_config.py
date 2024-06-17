# Copyright 2024
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Ruppik (ruppik@hhu.de)
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

import os
from dataclasses import dataclass, field

USER = os.getenv(
    "USER",
)


@dataclass
class MachineConfigurationConfig:
    """Config for machine configuration."""

    accelerator_model: str = "rtx6000"
    queue: str = "CUDA"
    ncpus: int = 2
    ngpus: int = 1
    memory_gb: int = 32
    walltime: str = "08:00:00"
    submit_job_command: list[str] = field(
        default_factory=lambda: [
            "python3",
            f"/gpfs/project/{USER}/.usr_tls/tools/submit_job.py",
        ],
    )
    dry_run: bool = False
