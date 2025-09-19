import warnings

from torch.utils.data import Dataset
from tqdm.auto import tqdm


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
