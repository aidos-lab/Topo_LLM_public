"""Run inference with a language model."""

import datetime
import hashlib
import json
import logging
import pathlib
import pprint
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from topollm.config_classes.main_config import MainConfig
from topollm.model_handling.prepare_loaded_model_container import (
    prepare_device_and_tokenizer_and_model_from_main_config,
)
from topollm.model_inference.causal_language_modeling.do_text_generation import (
    do_text_generation,
)
from topollm.model_inference.default_prompts import (
    get_default_clm_prompts,
    get_default_mlm_prompts,
)
from topollm.model_inference.masked_language_modeling.do_fill_mask import do_fill_mask
from topollm.path_management.embeddings.protocol import EmbeddingsPathManager
from topollm.typing.enums import LMmode, Verbosity

if TYPE_CHECKING:
    import torch
    import transformers
    from transformers.modeling_utils import PreTrainedModel

    from topollm.model_handling.loaded_model_container import LoadedModelContainer

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


@dataclass(
    slots=True,
    frozen=True,
)
class PromptHasher:
    """Generate deterministic filenames for prompt lists with an optional timestamp.

    Attributes:
        digest_size:
            Number of hex characters from the SHA-256 digest to use for the filename.
        include_timestamp:
            When True, append a timestamp to the filename.
        timestamp_format:
            A datetime.strftime pattern controlling how the timestamp is rendered.
            Defaults to a compact ISO-style format with microseconds and timezone
            (e.g. '20250801T154530_123456_UTC').

    """

    digest_size: int = 16
    include_timestamp: bool = False
    timestamp_format: str = "%Y%m%dT%H%M%S_%f_%Z"

    def compute_hash(
        self,
        prompts: Iterable[str],
    ) -> str:
        """Compute a full SHA-256 hexadecimal digest for the given prompts.

        The prompts are serialised to JSON with compact separators to ensure
        that logically equivalent lists produce identical input to the hash function.

        Args:
            prompts: An iterable of prompt strings.

        Returns:
            The 64-character hexadecimal SHA-256 digest.

        """
        prompt_list: list[str] = list(prompts)  # preserve ordering
        json_bytes: bytes = json.dumps(
            obj=prompt_list,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode(encoding="utf-8")
        # hashlib algorithms operate on bytes and return deterministic digests
        return hashlib.sha256(string=json_bytes).hexdigest()

    def filename_for(
        self,
        prompts: Iterable[str],
    ) -> str:
        """Generate a deterministic filename with optional timestamp.

        If include_timestamp is True, the filename will be of the form
        '<digest>_<timestamp>.json', otherwise '<digest>.json'.  The digest
        is truncated to `digest_size` characters.

        Args:
            prompts: An iterable of prompt strings.

        Returns:
            A filename string that uniquely identifies this list of prompts.

        """
        digest: str = self.compute_hash(prompts=prompts)[: self.digest_size]
        if self.include_timestamp:
            timestamp: str = (
                datetime.datetime.now(
                    tz=datetime.UTC,
                )
                .astimezone()
                .strftime(
                    format=self.timestamp_format,
                )
            )
            return f"{digest}_{timestamp}.json"

        return f"{digest}.json"


def do_inference(
    main_config: MainConfig,
    embeddings_path_manager: EmbeddingsPathManager,
    *,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> list[list]:
    """Run inference with a language model.

    If `prompts` is `None`, default prompts are used.
    Make sure to not accidentally use an empty list as the default argument.
    """
    loaded_model_container: LoadedModelContainer = prepare_device_and_tokenizer_and_model_from_main_config(
        main_config=main_config,
        verbosity=verbosity,
        logger=logger,
    )
    device: torch.device = loaded_model_container.device
    tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast = (
        loaded_model_container.tokenizer
    )
    lm_mode: LMmode = loaded_model_container.lm_mode
    model: PreTrainedModel = loaded_model_container.model

    # Set up the model for evaluation.
    model.eval()

    prompts: list[str] | None = main_config.inference.prompts

    match lm_mode:
        case LMmode.MLM:
            if prompts is None:
                prompts = get_default_mlm_prompts(
                    mask_token=tokenizer.mask_token,  # type: ignore - problem with inferring the correct type of the mask token
                )
            logger.info(
                msg=f"prompts:\n{pprint.pformat(object=prompts)}",  # noqa: G004 - low overhead
            )

            results = do_fill_mask(
                tokenizer=tokenizer,
                model=model,
                prompts=prompts,
                device=device,
                logger=logger,
            )
        case LMmode.CLM:
            if prompts is None:
                prompts = get_default_clm_prompts()
            logger.info(
                msg=f"prompts:\n{pprint.pformat(object=prompts)}",  # noqa: G004 - low overhead
            )

            results = do_text_generation(
                tokenizer=tokenizer,
                model=model,
                prompts=prompts,
                max_length=main_config.inference.max_length,
                num_return_sequences=main_config.inference.num_return_sequences,
                device=device,
                logger=logger,
            )
        case _:
            msg: str = f"Invalid lm_mode: {lm_mode = }"
            raise ValueError(msg)

    # # # #
    # Collect prompts and results in a dictionary.

    if not isinstance(
        results,
        list,
    ):
        msg = f"Expected results to be a list, got {type(results)=}"
        raise TypeError(
            msg,
        )

    if len(prompts) != len(results):
        msg = f"Number of prompts ({len(prompts)=}) does not match number of results ({len(results)=})."
        raise ValueError(msg)

    combined_results: dict[
        str,
        list,
    ] = dict(
        zip(
            prompts,
            results,
            strict=False,
        ),
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Combined results:\n{pprint.pformat(object=combined_results, width=200)}",  # noqa: G004 - low overhead
        )

    save_folder: pathlib.Path = embeddings_path_manager.get_model_inference_dir_absolute_path()

    # To create different files for different prompts, we will hash the prompts.
    hasher = PromptHasher(
        digest_size=16,
        include_timestamp=main_config.inference.include_timestamp_in_filename,
    )
    filename: str = hasher.filename_for(
        prompts=prompts,
    )
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Folder for saving inference results: {save_folder=}",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"Filename for saving inference results: {filename=}",  # noqa: G004 - low overhead
        )

    save_path: pathlib.Path = save_folder / filename
    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Saving inference results to {save_path=} ...",  # noqa: G004 - low overhead
        )

    with save_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            obj=combined_results,
            fp=file,
            ensure_ascii=False,
            indent=4,
        )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Saving inference results to {save_path=} DONE",  # noqa: G004 - low overhead
        )

    return results
