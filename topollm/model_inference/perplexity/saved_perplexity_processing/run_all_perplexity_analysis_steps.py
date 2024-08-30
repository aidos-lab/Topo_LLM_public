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

from subprocess import CalledProcessError, run

from pydantic import BaseModel, field_validator


class ScriptConfig(BaseModel):
    """Configuration model for scripts using Pydantic."""

    type: Literal["shell", "python"]
    path: str
    args: list[str] = []

    @field_validator("path")
    def validate_path(cls, v, values):
        if not v:
            raise ValueError("Script path cannot be empty.")
        return v

    def construct_command(self) -> list[str]:
        """Construct the command based on the script type."""
        if self.type == "shell":
            return [self.path] + self.args
        elif self.type == "python":
            return ["poetry", "run", "python", self.path] + self.args
        else:
            raise ValueError(f"Unsupported script type: {self.type}")


def run_script(command: list[str]) -> None:
    """Run a shell or Python script command using poetry."""
    try:
        print(f"Running: {' '.join(command)}")
        run(command, check=True)
    except CalledProcessError as e:
        print(f"Error while running command: {' '.join(command)}")
        print(f"Exit code: {e.returncode}")
        exit(1)


def main() -> None:
    # Define script configurations using Pydantic models
    scripts = [
        ScriptConfig(type="shell", path="./script1.sh", args=["arg1", "arg2"]),
        ScriptConfig(type="python", path="script1.py", args=["--arg1", "value1", "--arg2", "value2"]),
        ScriptConfig(type="shell", path="./script2.sh", args=["arg1", "arg2"]),
        ScriptConfig(type="python", path="script2.py", args=["--arg1", "value1", "--arg2", "value2"]),
    ]

    # Loop through each script configuration, construct the command, and execute it
    for script in scripts:
        command = script.construct_command()
        run_script(command)

    print("All scripts executed successfully!")


if __name__ == "__main__":
    main()


# TODO(Ben): Implement this script which runs the steps one after another
