# Copyright 2024
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

"""Run all perplexity analysis steps in sequence."""

import os
import pathlib
import sys
from enum import StrEnum, auto
from subprocess import CalledProcessError, run

from pydantic import BaseModel, Field, field_validator

from topollm.config_classes.constants import TOPO_LLM_REPOSITORY_BASE_PATH


class ScriptType(StrEnum):
    """Enumeration for supported script types."""

    SHELL = auto()
    PYTHON = auto()


class ScriptConfig(BaseModel):
    """Configuration model for scripts using Pydantic."""

    type: ScriptType = Field(
        default=ScriptType.SHELL,
    )
    path: os.PathLike
    args: list[str] = []

    @field_validator("path")
    def validate_path(
        cls,
        v,
        values,
    ):
        if not v:
            msg = "Script path cannot be empty."
            raise ValueError(msg)
        return v

    def construct_command(self) -> list[str]:
        """Construct the command based on the script type."""
        if self.type == ScriptType.SHELL:
            return [
                str(self.path),
                *self.args,
            ]
        elif self.type == ScriptType.PYTHON:
            return [
                "poetry",
                "run",
                "python",
                str(self.path),
                *self.args,
            ]
        else:
            msg = f"Unsupported script type: {self.type}"
            raise ValueError(msg)


def run_script(command: list[str]) -> None:
    """Run a shell or Python script command using poetry."""
    try:
        print(
            f"Running: {' '.join(command)}",
        )
        run(  # noqa: S603 - we trust the command construction
            command,
            check=True,
        )
    except CalledProcessError as e:
        print(
            f"Error while running command: {' '.join(command)}",
        )
        print(
            f"Exit code: {e.returncode}",
        )
        sys.exit(1)


def main() -> None:
    """Run all perplexity analysis steps in sequence."""
    # Define script configurations using Pydantic models
    scripts = [
        ScriptConfig(
            type=ScriptType.SHELL,
            path=pathlib.Path(
                TOPO_LLM_REPOSITORY_BASE_PATH,
                "topollm",
                "model_inference",
                "perplexity",
                "saved_perplexity_processing",
                "align_and_analyse",
                "run_multiple_load_saved_perplexity_and_concatenate_into_array.sh",
            ),
            args=[],
        ),
        ScriptConfig(
            type=ScriptType.PYTHON,
            path=pathlib.Path(
                TOPO_LLM_REPOSITORY_BASE_PATH,
                "topollm",
                "model_inference",
                "perplexity",
                "saved_perplexity_processing",
                "align_and_analyse",
                "run_combine_histogram_plots.py",
            ),
            args=[],
        ),
        ScriptConfig(
            type=ScriptType.PYTHON,
            path=pathlib.Path(
                TOPO_LLM_REPOSITORY_BASE_PATH,
                "topollm",
                "model_inference",
                "perplexity",
                "saved_perplexity_processing",
                "correlation",
                "run_aggregated_analysis.py",
            ),
            args=[],
        ),
    ]

    # Loop through each script configuration, construct the command, and execute it
    for script in scripts:
        command = script.construct_command()
        run_script(command)

    print(
        "All scripts executed successfully!",
    )


if __name__ == "__main__":
    main()
