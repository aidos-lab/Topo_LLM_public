# coding=utf-8
#
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

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Imports

# System imports
import warnings

# Third-party imports
from tqdm.auto import tqdm

from torch.utils.data import Dataset

# END Imports
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class DialogueUtteranceDataset(Dataset):
    def __init__(
        self,
        dialogues: list[dict],
        split: str,
    ):
        self.dialogues = dialogues
        self.dialogue_turns_utterances = []
        self.split = split

        for dialogue in tqdm(
            self.dialogues,
        ):
            for turn_index, turn in enumerate(
                dialogue["turns"],
            ):
                utterance = turn["utterance"]
                self.dialogue_turns_utterances.append(
                    (
                        utterance,
                        dialogue["dialogue_id"],
                        turn_index,
                    ),
                )

                if not utterance.strip():
                    warnings.warn(
                        f"Encountered an empty utterance "
                        f"(dialogue_id: {dialogue['dialogue_id']}, "
                        f"turn_index: {turn_index})"
                    )

    def __len__(self):
        return len(self.dialogue_turns_utterances)

    def __getitem__(
        self,
        idx,
    ):
        utterance, dialogue_id, turn_index = self.dialogue_turns_utterances[idx]

        return {
            "text": utterance,
            "dialogue_id": dialogue_id,
            "turn_index": turn_index,
            "split": self.split,
        }
