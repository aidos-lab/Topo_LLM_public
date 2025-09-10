"""Tests functions for converting dataset entries to features in embedding_dataloader_preparer module."""

import copy
import logging

import pytest
from transformers.tokenization_utils_base import BatchEncoding

from topollm.compute_embeddings.embedding_dataloader_preparer.convert_dataset_entry_to_features_functions import (
    convert_dataset_entry_to_features,
    convert_dataset_entry_to_features_luster_data,
)
from topollm.config_classes.language_model.language_model_config import LanguageModelConfig
from topollm.config_classes.tokenizer.tokenizer_config import TokenizerConfig
from topollm.model_handling.tokenizer.load_tokenizer import load_modified_tokenizer
from topollm.typing.enums import LMmode, TaskType, Verbosity

# Global dataset entry example for LUSTER dataset type.
# This is a small example with 4 entries.
dataset_entry_luster_data_example: dict[
    str,
    list,
] = {
    "dialogue_id": ["PMUL4488.json", "PMUL2599.json", "MUL2387.json", "MUL2693.json"],
    "turn_id": [0, 4, 2, 2],
    "source": [
        " user : i am looking for a place to dine. the restaurant should serve hungarian food and should be in the south.</s> emotion : neutral</s> domain : restaurant</s> state : food hungarian ; price range unknown ; name unknown ; area south ; book time unknown ; book day unknown ; book people unknown</s> database : no entity found in database</s> action :",
        "user : i'm looking for a 0 star hotel that is expensive.</s> system : i'm sorry there are no hotels that fit that criteria. would you like a different amount of stars?</s> user : what star ratings do you have for hotels in the centre?</s> system : there are 3 and 4 star hotels.</s> user : can you check for one in the moderate price range.</s> emotion : neutral</s> domain : hotel</s> state : name unknown ; area unknown ; parking unknown ; price range moderate ; stars dontcare ; internet unknown ; type unknown ; book stay unknown ; book day unknown ; book people unknown</s> database : 18 found in database - address sleeperz hotel, station road ; area centre ; internet yes ; parking no ; name cityroomz ; phone 01223304050 ; postcode cb12tz ; pricerange moderate ; stars 0 ; type hotel ; ref 29q1x35w</s> action :",
        "user : hi. i'm looking for a restaurant. i think it's called the rice ship or rice boat or something like that.</s> system : the rice boat is located in the west and is in the expensive range. would you like to book a reservation?</s> user : yes please. thanks for your help.</s> emotion : satisfied</s> domain : restaurant</s> state : food unknown ; price range unknown ; name rice boat ; area unknown ; book time unknown ; book day unknown ; book people unknown</s> database : 1 found in database - address 37 newnham road newnham ; area west ; food indian ; name rice boat ; phone 01223302800 ; postcode cb39ey ; pricerange expensive ; ref 24akjdgh</s> action :",
        "user : where is whipple museum of the history of science located? </s> system : the address is free school lane and the postcode is cb23rh. </s> user : what is the type of attraction and area for the whipple museum?</s> emotion : neutral</s> domain : attraction</s> state : type museum ; name whipple museum of the history of science ; area unknown</s> database : 1 found in database - address free school lane ; area centre ; entrance fee free ; name whipple museum of the history of science ; phone 01223330906 ; postcode cb23rh ; type museum</s> action :",
    ],
    "target": [
        " nooffer food south</s> conduct : apologetic</s> system : regretfully, we have nothing like that in the south.</s>",
        " inform price range moderate ; inform choice 18 ; inform type hotel</s> conduct : neutral</s> system : i have 18 different hotels in the moderate price range. is there a certain area you would like?</s>",
        " request book people</s> conduct : neutral</s> system : how many people would you like to reserve a table for?</s>",
        " inform type museum ; inform area centre ; inform name the whipple ; request more</s> conduct : neutral</s> system : the whipple is a museum type attraction located in the centre area. can i do anything else for you?</s>",
    ],
    "source_target": [
        " user : i am looking for a place to dine. the restaurant should serve hungarian food and should be in the south.</s> emotion : neutral</s> domain : restaurant</s> state : food hungarian ; price range unknown ; name unknown ; area south ; book time unknown ; book day unknown ; book people unknown</s> database : no entity found in database</s> action : nooffer food south</s> conduct : apologetic</s> system : regretfully, we have nothing like that in the south.</s>",
        "user : i'm looking for a 0 star hotel that is expensive.</s> system : i'm sorry there are no hotels that fit that criteria. would you like a different amount of stars?</s> user : what star ratings do you have for hotels in the centre?</s> system : there are 3 and 4 star hotels.</s> user : can you check for one in the moderate price range.</s> emotion : neutral</s> domain : hotel</s> state : name unknown ; area unknown ; parking unknown ; price range moderate ; stars dontcare ; internet unknown ; type unknown ; book stay unknown ; book day unknown ; book people unknown</s> database : 18 found in database - address sleeperz hotel, station road ; area centre ; internet yes ; parking no ; name cityroomz ; phone 01223304050 ; postcode cb12tz ; pricerange moderate ; stars 0 ; type hotel ; ref 29q1x35w</s> action : inform price range moderate ; inform choice 18 ; inform type hotel</s> conduct : neutral</s> system : i have 18 different hotels in the moderate price range. is there a certain area you would like?</s>",
        "user : hi. i'm looking for a restaurant. i think it's called the rice ship or rice boat or something like that.</s> system : the rice boat is located in the west and is in the expensive range. would you like to book a reservation?</s> user : yes please. thanks for your help.</s> emotion : satisfied</s> domain : restaurant</s> state : food unknown ; price range unknown ; name rice boat ; area unknown ; book time unknown ; book day unknown ; book people unknown</s> database : 1 found in database - address 37 newnham road newnham ; area west ; food indian ; name rice boat ; phone 01223302800 ; postcode cb39ey ; pricerange expensive ; ref 24akjdgh</s> action : request book people</s> conduct : neutral</s> system : how many people would you like to reserve a table for?</s>",
        "user : where is whipple museum of the history of science located? </s> system : the address is free school lane and the postcode is cb23rh. </s> user : what is the type of attraction and area for the whipple museum?</s> emotion : neutral</s> domain : attraction</s> state : type museum ; name whipple museum of the history of science ; area unknown</s> database : 1 found in database - address free school lane ; area centre ; entrance fee free ; name whipple museum of the history of science ; phone 01223330906 ; postcode cb23rh ; type museum</s> action : inform type museum ; inform area centre ; inform name the whipple ; request more</s> conduct : neutral</s> system : the whipple is a museum type attraction located in the centre area. can i do anything else for you?</s>",
    ],
}


@pytest.fixture
def dataset_entry() -> dict[str, list]:
    """Fixture for dataset entry."""
    # Return a deep copy of the global dataset entry example to avoid mutation issues in tests.
    return copy.deepcopy(dataset_entry_luster_data_example)


def check_features_basic_properties(
    features: BatchEncoding,
    dataset_entry: dict[str, list],
    column_name: str,
) -> None:
    """Check basic properties of the features."""
    assert isinstance(  # noqa: S101 - pytest assertion
        features,
        BatchEncoding,
    )
    assert "input_ids" in features  # noqa: S101 - pytest assertion
    assert "attention_mask" in features  # noqa: S101 - pytest assertion
    assert len(features.input_ids) == len(dataset_entry[column_name])  # noqa: S101 - pytest assertion


def test_convert_dataset_entry_to_features(
    dataset_entry: dict[str, list],
    language_model_config: LanguageModelConfig,
    tokenizer_config: TokenizerConfig,
    verbosity: Verbosity,
    logger_fixture: logging.Logger,
) -> None:
    """Test convert_dataset_entry_to_features function."""
    column_name = "source_target"

    (
        tokenizer,
        _tokenizer_modifier,
    ) = load_modified_tokenizer(
        language_model_config=language_model_config,
        tokenizer_config=tokenizer_config,
        verbosity=verbosity,
        logger=logger_fixture,
    )

    features: BatchEncoding = convert_dataset_entry_to_features(
        dataset_entry=dataset_entry,
        tokenizer=tokenizer,
        column_name=column_name,
        max_length=512,
    )

    check_features_basic_properties(
        features=features,
        dataset_entry=dataset_entry,
        column_name=column_name,
    )


def test_convert_dataset_entry_to_features_luster_data(
    dataset_entry: dict[str, list],
    tokenizer_config: TokenizerConfig,
    verbosity: Verbosity,
    logger_fixture: logging.Logger,
) -> None:
    """Test convert_dataset_entry_to_features_luster_data function."""
    column_name = "source_target"

    language_model_config = LanguageModelConfig(
        lm_mode=LMmode.CLM,
        task_type=TaskType.CAUSAL_LM,
        pretrained_model_name_or_path="microsoft/Phi-3.5-mini-instruct",
        short_model_name="Phi-3.5-mini-instruct",
    )

    (
        tokenizer,
        _tokenizer_modifier,
    ) = load_modified_tokenizer(
        language_model_config=language_model_config,
        tokenizer_config=tokenizer_config,
        verbosity=verbosity,
        logger=logger_fixture,
    )

    features: BatchEncoding = convert_dataset_entry_to_features_luster_data(
        dataset_entry=dataset_entry,
        tokenizer=tokenizer,
        column_name=column_name,
        max_length=512,
    )

    check_features_basic_properties(
        features=features,
        dataset_entry=dataset_entry,
        column_name=column_name,
    )
    # TODO: Implement further tests to check whether the token masks are created correctly
    pass  # TODO: For setting breakpoints, remove if unnecessary
