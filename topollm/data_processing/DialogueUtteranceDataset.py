# coding=utf-8
#
# Copyright 2024
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


# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Imports

# System imports
import warnings

# Third-party imports
from torch.utils.data import Dataset
from tqdm.auto import tqdm

# Local imports

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

    def __len__(
        self,
    ) -> int:
        return len(self.dialogue_turns_utterances)

    def __getitem__(
        self,
        idx,
    ) -> dict[str, str]:
        utterance, dialogue_id, turn_index = self.dialogue_turns_utterances[idx]

        entry = {
            "text": utterance,
            "dialogue_id": dialogue_id,
            "turn_index": turn_index,
            "split": self.split,
        }

        return entry
