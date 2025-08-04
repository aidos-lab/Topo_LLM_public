"""Fixtures for the tests."""

import datetime
import logging
import os
import pathlib

import pytest
import torch
import torch.backends
import torch.backends.mps
import transformers
from dotenv import find_dotenv, load_dotenv

from topollm.config_classes.data.data_config import DataConfig
from topollm.config_classes.data.data_subsampling_config import DataSubsamplingConfig
from topollm.config_classes.data.dataset_map_config import DatasetMapConfig
from topollm.config_classes.embeddings.embedding_extraction_config import (
    EmbeddingExtractionConfig,
)
from topollm.config_classes.embeddings.embeddings_config import EmbeddingsConfig
from topollm.config_classes.embeddings_data_prep.embeddings_data_prep_config import EmbeddingsDataPrepConfig
from topollm.config_classes.feature_flags.feature_flags_config import FeatureFlagsConfig, WandbFeatureFlagsConfig
from topollm.config_classes.finetuning.batch_sizes.batch_sizes_config import BatchSizesConfig
from topollm.config_classes.finetuning.finetuning_config import FinetuningConfig
from topollm.config_classes.finetuning.finetuning_datasets.finetuning_datasets_config import (
    FinetuningDatasetsConfig,
)
from topollm.config_classes.finetuning.gradient_modifier.gradient_modifier_config import GradientModifierConfig
from topollm.config_classes.finetuning.peft.peft_config import PEFTConfig
from topollm.config_classes.inference.inference_config import InferenceConfig
from topollm.config_classes.language_model.language_model_config import (
    DropoutConfig,
    DropoutProbabilities,
    LanguageModelConfig,
)
from topollm.config_classes.language_model.tokenizer_modifier.tokenizer_modifier_config import TokenizerModifierConfig
from topollm.config_classes.main_config import MainConfig
from topollm.config_classes.paths.paths_config import PathsConfig
from topollm.config_classes.storage.storage_config import StorageConfig
from topollm.config_classes.tokenizer.tokenizer_config import TokenizerConfig
from topollm.config_classes.transformations.transformations_config import TransformationsConfig
from topollm.config_classes.wandb.wandb_config import WandBConfig
from topollm.model_handling.tokenizer.load_tokenizer import load_modified_tokenizer
from topollm.path_management.embeddings.embeddings_path_manager_separate_directories import (
    EmbeddingsPathManagerSeparateDirectories,
)
from topollm.path_management.finetuning.finetuning_path_manager_basic import (
    FinetuningPathManagerBasic,
)
from topollm.path_management.finetuning.protocol import (
    FinetuningPathManager,
)
from topollm.typing.enums import (
    AggregationType,
    ArrayStorageType,
    DatasetType,
    DropoutMode,
    FinetuningMode,
    GradientModifierMode,
    LMmode,
    MetadataStorageType,
    PreferredTorchBackend,
    Split,
    TaskType,
    TokenizerModifierMode,
    Verbosity,
)

logger: logging.Logger = logging.getLogger(
    name=__name__,
)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Configuration of pytest


@pytest.fixture(
    scope="session",
    autouse=True,
)
def _load_env() -> None:
    """Load the environment variables from the .test.env file.

    https://stackoverflow.com/questions/48211784/best-way-to-use-python-dotenv-with-pytest-or-best-way-to-have-a-pytest-test-dev
    """
    env_file = find_dotenv(
        filename=".test.env",
    )
    result = load_dotenv(
        dotenv_path=env_file,
        verbose=True,
    )

    if result:
        logger.info(
            f"Loaded environment variables from {env_file = }",  # noqa: G004 - low overhead
        )
    else:
        logger.warning(
            f"No environment variables loaded from {env_file = }",  # noqa: G004 - low overhead
        )


def pytest_addoption(
    parser: pytest.Parser,
) -> None:
    """Add a command line option to keep the test data after the tests are done."""
    parser.addoption(
        "--keep-test-data",
        action="store_true",
        help="Keep test data after tests are done",
    )


def pytest_configure(
    config: pytest.Config,
) -> None:
    """Create a custom path to the log file if log_file is not mentioned in pytest.ini file."""
    if not config.option.log_file:
        timestamp: str = datetime.datetime.now(
            tz=datetime.UTC,
        ).strftime(
            format="%Y-%m-%d_%H-%M-%S_%Z",
        )
        # Note: the doubling {{ and }} is necessary to escape the curly braces
        config.option.log_file = f"logs/pytest-logs_{timestamp}.log"


# END Configuration of pytest
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


@pytest.fixture(
    scope="session",
)
def repository_base_path() -> pathlib.Path:
    """Return the base path of the repository."""
    # Get the values from the
    # 'TOPO_LLM_REPOSITORY_BASE_PATH' environment variable
    topo_llm_repository_base_path = os.getenv(
        key="TOPO_LLM_REPOSITORY_BASE_PATH",
    )

    if topo_llm_repository_base_path is None:
        msg = "The 'TOPO_LLM_REPOSITORY_BASE_PATH' environment variable is not set."
        raise ValueError(msg)

    path = pathlib.Path(
        topo_llm_repository_base_path,
    )

    return path


@pytest.fixture(
    scope="session",
)
def temp_files_dir() -> pathlib.Path:
    """Return the directory for temporary files."""
    # Get the values from the 'TEMP_FILES_DIR' environment variable
    temp_files_dir = os.getenv("TEMP_FILES_DIR")

    if temp_files_dir is None:
        msg = "The 'TEMP_FILES_DIR' environment variable is not set."
        raise ValueError(msg)

    path = pathlib.Path(
        temp_files_dir,
    )

    return path


@pytest.fixture(
    scope="session",
)
def test_data_dir(
    repository_base_path: pathlib.Path,
    temp_files_dir: pathlib.Path,
    tmp_path_factory: pytest.TempPathFactory,
    pytestconfig: pytest.Config,
) -> pathlib.Path:
    """Return the directory for the test data."""
    if pytestconfig.getoption(
        name="--keep-test-data",
    ):
        # Create a more permanent directory
        base_dir = pathlib.Path(
            repository_base_path,
            temp_files_dir,
        )
        base_dir.mkdir(
            exist_ok=True,
        )
        return base_dir

    # Use pytest's tmp_path_factory for a truly temporary directory
    return tmp_path_factory.mktemp(
        basename="data-",
        numbered=True,
    )


@pytest.fixture(
    scope="session",
)
def logger_fixture() -> logging.Logger:
    """Return a logger object."""
    return logger


@pytest.fixture(
    scope="session",
)
def verbosity() -> Verbosity:
    """Return a Verbosity object."""
    return Verbosity.NORMAL


@pytest.fixture(
    scope="session",
)
def data_subsampling_config() -> DataSubsamplingConfig:
    """Return a DataSubsamplingConfig object."""
    config = DataSubsamplingConfig(
        number_of_samples=10,
        sampling_seed=42,
        split=Split.TRAIN,
    )

    return config


@pytest.fixture(
    scope="session",
)
def data_config() -> DataConfig:
    """Return a DataConfig object."""
    config = DataConfig(
        column_name="summary",
        context="dataset_entry",
        dataset_description_string="xsum",
        dataset_type=DatasetType.HUGGINGFACE_DATASET,
        data_dir=None,
        dataset_path="xsum",
        dataset_name=None,
        feature_column_name="summary",
    )

    return config


@pytest.fixture(
    scope="session",
)
def tokenizer_config() -> TokenizerConfig:
    """Return a TokenizerConfig object."""
    config = TokenizerConfig(
        add_prefix_space=False,
        max_length=512,
    )

    return config


@pytest.fixture(
    scope="session",
)
def dataset_map_config() -> DatasetMapConfig:
    """Return a DatasetMapConfig object."""
    config = DatasetMapConfig()

    return config


model_config_list_for_testing: list[
    tuple[
        LMmode,
        TaskType,
        str,
        str,
        TokenizerModifierConfig,
        DropoutConfig,
    ],
] = [
    (
        LMmode.MLM,
        TaskType.MASKED_LM,
        "roberta-base",
        "roberta-base",
        TokenizerModifierConfig(
            mode=TokenizerModifierMode.DO_NOTHING,
            padding_token="<pad>",  # noqa: S106 - This is the hardcoded padding token
        ),
        DropoutConfig(),  # Use the default dropout configuration
    ),
    (
        LMmode.MLM,
        TaskType.MASKED_LM,
        "roberta-base",
        "roberta-base",
        TokenizerModifierConfig(
            mode=TokenizerModifierMode.DO_NOTHING,
            padding_token="<pad>",  # noqa: S106 - This is the hardcoded padding token
        ),
        DropoutConfig(
            mode=DropoutMode.MODIFY_ROBERTA_DROPOUT_PARAMETERS,
            probabilities=DropoutProbabilities(
                hidden_dropout_prob=0.2,
                attention_probs_dropout_prob=0.3,
                classifier_dropout=None,
            ),
        ),
    ),
    (
        LMmode.MLM,
        TaskType.MASKED_LM,
        "bert-base-uncased",
        "bert-base-uncased",
        TokenizerModifierConfig(
            mode=TokenizerModifierMode.DO_NOTHING,
            padding_token="[PAD]",  # noqa: S106 - This is the hardcoded padding token
        ),
        DropoutConfig(),  # Use the default dropout configuration
    ),
    (
        LMmode.CLM,
        TaskType.CAUSAL_LM,
        "gpt2-medium",
        "gpt2-medium",
        TokenizerModifierConfig(
            mode=TokenizerModifierMode.ADD_PADDING_TOKEN,
            padding_token="<|pad|>",  # noqa: S106 - This is the hardcoded padding token
        ),
        DropoutConfig(),  # Use the default dropout configuration for the causal language model
    ),
]


@pytest.fixture(
    scope="session",
    params=model_config_list_for_testing,
)
def language_model_config(
    request: pytest.FixtureRequest,
) -> LanguageModelConfig:
    """Return a LanguageModelConfig object."""
    (
        lm_mode,
        task_type,
        pretrained_model_name_or_path,
        short_model_name,
        tokenizer_modifier_config,
        dropout_config,
    ) = request.param

    config = LanguageModelConfig(
        lm_mode=lm_mode,
        task_type=task_type,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        short_model_name=short_model_name,
        dropout=dropout_config,
        tokenizer_modifier=tokenizer_modifier_config,
    )

    return config


@pytest.fixture(
    scope="session",
)
def embedding_extraction_config() -> EmbeddingExtractionConfig:
    """Return an EmbeddingExtractionConfig object."""
    config = EmbeddingExtractionConfig(
        layer_indices=[
            -1,
        ],
        aggregation=AggregationType.MEAN,
    )

    return config


@pytest.fixture(
    scope="session",
)
def embeddings_config(
    dataset_map_config: DatasetMapConfig,
    embedding_extraction_config: EmbeddingExtractionConfig,
) -> EmbeddingsConfig:
    """Return an EmbeddingsConfig object.

    Notes:
    - You should set 'num_workers=0' to avoid the following multiprocessing error
      on the torch.device("mps") backend:
      `RuntimeError: _share_filename_: only available on CPU`
      Setting 'num_workers=1', while only starting a single process,
      does use the multiprocessing module and can lead to the error.

    """
    config = EmbeddingsConfig(
        dataset_map=dataset_map_config,
        embedding_extraction=embedding_extraction_config,
    )

    return config


@pytest.fixture(
    scope="session",
)
def paths_config(
    test_data_dir: pathlib.Path,
    repository_base_path: pathlib.Path,
) -> PathsConfig:
    """Return a PathsConfig object."""
    return PathsConfig(
        data_dir=test_data_dir,
        repository_base_path=repository_base_path,
    )


@pytest.fixture(
    scope="session",
)
def transformations_config() -> TransformationsConfig:
    """Return a TransformationsConfig object."""
    return TransformationsConfig(
        normalization="None",
    )


@pytest.fixture(
    scope="session",
    params=[
        FinetuningMode.STANDARD,
        FinetuningMode.LORA,
    ],
)
def peft_config(
    request: pytest.FixtureRequest,
) -> PEFTConfig:
    """Return a PEFTConfig object."""
    finetuning_mode = request.param

    config = PEFTConfig(
        finetuning_mode=finetuning_mode,
    )

    return config


@pytest.fixture(
    scope="session",
)
def feature_flags_config() -> FeatureFlagsConfig:
    """Return a FeatureFlagsConfig object."""
    wandb_config = WandbFeatureFlagsConfig(
        use_wandb=False,
    )

    config = FeatureFlagsConfig(
        wandb=wandb_config,
    )

    return config


@pytest.fixture(
    scope="session",
)
def finetuning_datasets_config(
    data_config: DataConfig,
) -> FinetuningDatasetsConfig:
    """Return a FinetuningDatasetsConfig object."""
    config = FinetuningDatasetsConfig(
        train_dataset=data_config,
        eval_dataset=data_config,
    )

    return config


@pytest.fixture(
    scope="session",
)
def batch_sizes_config() -> BatchSizesConfig:
    """Return a BatchSizesConfig object."""
    config = BatchSizesConfig()

    return config


@pytest.fixture(
    scope="session",
)
def gradient_modifier_config() -> GradientModifierConfig:
    """Return a GradientModifierConfig object."""
    config = GradientModifierConfig(
        mode=GradientModifierMode.FREEZE_LAYERS,
        target_modules_to_freeze=[
            "encoder.layer.0.",
            "encoder.layer.1.",
        ],
    )

    return config


@pytest.fixture(
    scope="session",
)
def finetuning_config(
    language_model_config: LanguageModelConfig,
    gradient_modifier_config: GradientModifierConfig,
    batch_sizes_config: BatchSizesConfig,
    finetuning_datasets_config: FinetuningDatasetsConfig,
    peft_config: PEFTConfig,
    tokenizer_config: TokenizerConfig,
) -> FinetuningConfig:
    """Return a FinetuningConfig object."""
    config = FinetuningConfig(
        base_model=language_model_config,
        gradient_modifier=gradient_modifier_config,
        peft=peft_config,
        batch_sizes=batch_sizes_config,
        finetuning_datasets=finetuning_datasets_config,
        max_steps=2,
        tokenizer=tokenizer_config,
    )

    return config


@pytest.fixture(
    scope="session",
)
def device_fixture() -> torch.device:
    """Return the device to use for the tests."""
    use_mps_if_available = False

    if use_mps_if_available:
        device = torch.device(
            device="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        )
    else:
        device = torch.device(
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    return device


@pytest.fixture(
    scope="session",
)
def inference_config() -> InferenceConfig:
    """Return an InferenceConfig object."""
    config = InferenceConfig(
        max_length=50,
        num_return_sequences=2,
    )

    return config


@pytest.fixture(
    scope="session",
)
def storage_config() -> StorageConfig:
    """Return a StorageConfig object."""
    config = StorageConfig(
        array_storage_type=ArrayStorageType.ZARR,
        metadata_storage_type=MetadataStorageType.PICKLE,
        chunk_size=512,
    )

    return config


@pytest.fixture(
    scope="session",
)
def transformers_config() -> TransformationsConfig:
    """Return a TransformationsConfig object."""
    config = TransformationsConfig(
        normalization="None",
    )

    return config


@pytest.fixture(
    scope="session",
)
def embeddings_data_prep_config() -> EmbeddingsDataPrepConfig:
    """Return an EmbeddingsDataPrepConfig object."""
    config = EmbeddingsDataPrepConfig()

    return config


@pytest.fixture(
    scope="session",
)
def wandb_config() -> WandBConfig:
    """Return a WandBConfig object."""
    config = WandBConfig(
        tags=[
            "tests",
        ],
    )

    return config


@pytest.fixture(
    scope="session",
)
def main_config(  # noqa: PLR0913 - many arguments here because main config contains many components
    data_config: DataConfig,
    embeddings_config: EmbeddingsConfig,
    embeddings_data_prep_config: EmbeddingsDataPrepConfig,
    feature_flags_config: FeatureFlagsConfig,
    finetuning_config: FinetuningConfig,
    inference_config: InferenceConfig,
    language_model_config: LanguageModelConfig,
    paths_config: PathsConfig,
    storage_config: StorageConfig,
    tokenizer_config: TokenizerConfig,
    transformations_config: TransformationsConfig,
    wandb_config: WandBConfig,
    verbosity: Verbosity,
) -> MainConfig:
    """Return a MainConfig object."""
    config = MainConfig(
        data=data_config,
        embeddings_data_prep=embeddings_data_prep_config,
        embeddings=embeddings_config,
        feature_flags=feature_flags_config,
        finetuning=finetuning_config,
        inference=inference_config,
        language_model=language_model_config,
        paths=paths_config,
        preferred_torch_backend=PreferredTorchBackend.CPU,
        storage=storage_config,
        tokenizer=tokenizer_config,
        transformations=transformations_config,
        wandb=wandb_config,
        verbosity=verbosity,
    )

    return config


@pytest.fixture(
    scope="session",
)
def tokenizer(
    main_config: MainConfig,
    verbosity: Verbosity,
    logger_fixture: logging.Logger,
) -> transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast:
    """Return a tokenizer object."""
    tokenizer, _ = load_modified_tokenizer(
        language_model_config=main_config.language_model,
        tokenizer_config=main_config.tokenizer,
        verbosity=verbosity,
        logger=logger_fixture,
    )

    return tokenizer


@pytest.fixture(
    scope="session",
)
def embeddings_path_manager(
    main_config: MainConfig,
    verbosity: Verbosity,
    logger_fixture: logging.Logger,
) -> EmbeddingsPathManagerSeparateDirectories:
    """Return an EmbeddingsPathManagerSeparateDirectories object."""
    path_manager = EmbeddingsPathManagerSeparateDirectories(
        main_config=main_config,
        verbosity=verbosity,
        logger=logger_fixture,
    )

    return path_manager


@pytest.fixture(
    scope="session",
)
def finetuning_path_manager_basic(
    data_config: DataConfig,
    paths_config: PathsConfig,
    finetuning_config: FinetuningConfig,
    verbosity: Verbosity,
    logger_fixture: logging.Logger,
) -> FinetuningPathManager:
    """Return a FinetuningPathManagerBasic object."""
    path_manager = FinetuningPathManagerBasic(
        data_config=data_config,
        paths_config=paths_config,
        finetuning_config=finetuning_config,
        verbosity=verbosity,
        logger=logger_fixture,
    )

    return path_manager
