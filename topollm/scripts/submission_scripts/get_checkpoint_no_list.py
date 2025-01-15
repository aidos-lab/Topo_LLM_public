# Copyright 2024
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Ruppik (ruppik@hhu.de)
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


"""Get the list of checkpoint numbers to use."""

from topollm.scripts.submission_scripts.types import CheckpointNoListOption


def get_checkpoint_no_list(
    checkpoint_no_list_option: CheckpointNoListOption,
    num_train_epochs: int = 5,
) -> list[str]:
    """Get the list of checkpoint numbers to use.

    Note: This function only works for certain values of `num_train_epochs`.
    """
    match checkpoint_no_list_option:
        case CheckpointNoListOption.SELECTED:
            if num_train_epochs == 5:
                checkpoint_no_list: list[str] = [
                    "400",
                    "1200",
                    "2000",
                    "2800",
                ]
            elif num_train_epochs == 50:
                checkpoint_no_list = [
                    "400",
                    "3200",
                    "6000",
                    "8800",
                    "11600",
                    "14400",
                    "17200",
                    "20000",
                    "22800",
                    "25600",
                    "28400",
                    "31200",
                ]
            else:
                msg = f"Unknown {num_train_epochs = }"
                raise ValueError(msg)
        case CheckpointNoListOption.FULL:
            if num_train_epochs == 5:
                # All checkpoints from 400 to 2800
                # (for ep-5 and batch size 8)
                checkpoint_no_list = [
                    "400",
                    "800",
                    "1200",
                    "1600",
                    "2000",
                    "2400",
                    "2800",
                ]
            elif num_train_epochs == 50:
                # All checkpoints from 400 to 31200
                # (for ep-50 and batch size 8)
                checkpoint_no_list = [
                    str(i)
                    for i in range(
                        400,
                        31201,
                        400,
                    )
                ]
            else:
                msg = f"Unknown {num_train_epochs = }"
                raise ValueError(msg)
        case CheckpointNoListOption.ONLY_BEGINNING_AND_MIDDLE_AND_END:
            if num_train_epochs == 5:
                checkpoint_no_list = [
                    "400",
                    "1200",
                    "2800",
                ]
            elif num_train_epochs == 50:
                checkpoint_no_list = [
                    "400",
                    "14400",
                    "31200",
                ]
            else:
                msg: str = f"Unknown {num_train_epochs = }"
                raise ValueError(msg)
        case CheckpointNoListOption.FIXED_2800:
            checkpoint_no_list = [
                "2800",
            ]
        case CheckpointNoListOption.FIXED_1200_1600:
            checkpoint_no_list = [
                "1200",
                "1600",
            ]
        case CheckpointNoListOption.RANGE_START_400_STOP_3200_STEP_400:
            checkpoint_no_list = [
                str(i)
                for i in range(
                    400,
                    3200,
                    400,
                )
            ]
        case _:
            msg = f"Unknown {checkpoint_no_list_option = }"
            raise ValueError(msg)

    return checkpoint_no_list
