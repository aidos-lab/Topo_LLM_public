from enum import StrEnum, auto


class RunOption(StrEnum):
    """Options for selecting a dry run."""

    DO_SUBMISSION = auto()
    DRY_RUN = auto()


class ExperimentStage(StrEnum):
    """Options for the experiment stage."""

    COMPUTE_EMBEDDINGS_PLUS_SINGLE_PIPELINE_RUN = auto()
    SKIP_COMPUTE_EMBEDDINGS_BUT_DO_MULTIPLE_PIPELINE_RUNS = auto()
    SKIP_COMPUTE_EMBEDDINGS_AND_SKIP_EMBEDDINGS_DATA_PREP = auto()


class ExperimentSelector(StrEnum):
    """Options for the experiment selector."""

    # >>> Sensitivity analysis

    # Influence of data subsampling number of samples
    SENSITIVITY_ANALYSIS_MULTIWOZ21_DIFFERENT_DATA_SUBSAMPLING_NUMBER_OF_SAMPLES = auto()
    SENSITIVITY_ANALYSIS_REDDIT_DIFFERENT_DATA_SUBSAMPLING_NUMBER_OF_SAMPLES = auto()
    SENSITIVITY_ANALYSIS_DIFFERENT_LOCAL_ESTIMATES_FILTERING_NUMBER_OF_SAMPLES = auto()
    SENSITIVITY_ANALYSIS_DIFFERENT_LOCAL_ESTIMATES_POINTWISE_ABSOLUTE_N_NEIGHBORS = auto()

    COARSE_CHECKPOINT_RESOLUTION = auto()
    EXPLORATORY_DROPOUT_ANALYSIS_COARSE_CHECKPOINT_RESOLUTION = auto()
    TINY_DROPOUT_VARIATIONS_COARSE_CHECKPOINT_RESOLUTION = auto()
    FIXED_PARAMETERS_HIGH_CHECKPOINT_RESOLUTION = auto()

    REGULAR_TOKEN_EMBEDDINGS = auto()
    MASKED_TOKEN_EMBEDDINGS = auto()

    # >>> Embedding computation
    REGULAR_TOKEN_EMBEDDINGS_MULTIPLE_LAYERS_SINGLE_SAMPLE = auto()
    REGULAR_TOKEN_EMBEDDINGS_LAST_LAYER_SINGLE_SAMPLE = auto()
    MASKED_TOKEN_EMBEDDINGS_LAST_LAYER_SINGLE_SAMPLE = auto()
    MASKED_TOKEN_EMBEDDINGS_LAST_LAYER_TWO_DATA_SUBSAMPLING_SAMPLING_SEEDS = auto()


class CheckpointNoListOption(StrEnum):
    """Options for the checkpoint number list."""

    FULL = auto()
    ONLY_BEGINNING_AND_MIDDLE_AND_END = auto()
    SELECTED = auto()

    # Fixed checkpoints
    FIXED_2800 = auto()

    FIXED_1200_1600 = auto()

    RANGE_START_400_STOP_3200_STEP_400 = auto()
    RANGE_START_100_STOP_3200_STEP_100 = auto()


class DataListOption(StrEnum):
    """Options for the data list."""

    DEBUG = auto()
    FULL = auto()
    MANUAL_IN_PYTHON_SCRIPT = auto()

    # Selecting only one dataset
    ICLR_ONLY = auto()
    ICLR_TEST_ONLY = auto()
    ICLR_TRAIN_ONLY = auto()
    ICLR_VALIDATION_ONLY = auto()
    MULTIWOZ21_ONLY = auto()
    MULTIWOZ21_TEST_ONLY = auto()
    MULTIWOZ21_TRAIN_ONLY = auto()
    MULTIWOZ21_VALIDATION_ONLY = auto()
    REDDIT_ONLY = auto()
    REDDIT_TEST_ONLY = auto()
    REDDIT_TRAIN_ONLY = auto()
    REDDIT_VALIDATION_ONLY = auto()
    SGD_ONLY = auto()
    SGD_TEST_ONLY = auto()
    SGD_TRAIN_ONLY = auto()
    SGD_VALIDATION_ONLY = auto()
    WIKITEXT_ONLY = auto()
    WIKITEXT_TEST_ONLY = auto()
    WIKITEXT_TRAIN_ONLY = auto()
    WIKITEXT_VALIDATION_ONLY = auto()

    # Select certain data splits
    TRAIN_SPLIT_ONLY = auto()
    VALIDATION_SPLIT_ONLY = auto()

    # Mixing two datasets
    MULTIWOZ21_AND_REDDIT = auto()
    MULTIWOZ21_TRAIN_AND_REDDIT_TRAIN = auto()
    MULTIWOZ21_VALIDATION_AND_REDDIT_VALIDATION = auto()

    # Mixing three datasets
    ICLR_VALIDATION_AND_SGD_VALIDATION_AND_WIKITEXT_VALIDATION = auto()


class DataSubsamplingNumberOfSamplesListOption(StrEnum):
    """Options for the data number of samples list."""

    NONE = auto()
    FIXED_3000 = auto()
    FIXED_10000 = auto()
    FIXED_12000 = auto()
    FIXED_16000 = auto()
    FIXED_22000 = auto()
    RANGE_START_2000_STOP_12000_STEP_2000 = auto()
    RANGE_START_2000_STOP_18000_STEP_2000 = auto()  # For exhausting the entire multiwoz21 validation and test sets
    RANGE_START_12000_STOP_18000_STEP_2000 = auto()
    RANGE_START_2000_STOP_24000_STEP_2000 = (
        auto()
    )  # For exhausting the entire one-year-of-tsla-on-reddit validation and test sets
    RANGE_START_12000_STOP_24000_STEP_2000 = auto()


class DataSubsamplingSamplingSeedListOption(StrEnum):
    """Options for the seed lists for the data subsampling."""

    NONE = auto()
    DEFAULT = auto()
    FIXED_777 = auto()
    FIXED_778_779 = auto()
    TWO_SEEDS = auto()
    THREE_SEEDS = auto()
    FIVE_SEEDS = auto()
    TEN_SEEDS = auto()
    TWENTY_SEEDS = auto()


class FinetuningBaseModelListOption(StrEnum):
    """Options for the finetuning base model list."""

    ROBERTA_BASE = auto()
    GPT2_MEDIUM = auto()


class FinetuningDatasetsListOption(StrEnum):
    """Options for the finetuning dataset list."""

    DEBUG = auto()
    MANUAL_IN_PYTHON_SCRIPT = auto()

    # Single datasets
    ICLR_SMALL = auto()
    MULTIWOZ21_SMALL = auto()
    MULTIWOZ21_FULL = auto()
    REDDIT_SMALL = auto()
    REDDIT_FULL = auto()
    SGD_SMALL = auto()
    WIKITEXT_SMALL = auto()

    # Multiple datasets in separate runs
    MULTIWOZ21_AND_REDDIT_FULL = auto()
    MULTIWOZ21_AND_REDDIT_SMALL = auto()


class ModelGroupOption(StrEnum):
    """Options for specifying the model and finetuning regime."""

    # # # # # # # # # # # #
    # RoBERTa-base models
    ROBERTA_BASE_WITHOUT_MODIFICATIONS = auto()

    ROBERTA_BASE_FINETUNED_FOR_FEW_EPOCHS_OLD_AND_NEW_DATA_SINGLE_SEED_LAST_CHECKPOINT = auto()
    ROBERTA_BASE_FINETUNED_FOR_FEW_EPOCHS_MULTIWOZ_DATA_SINGLE_SEED_LAST_CHECKPOINT = auto()
    ROBERTA_BASE_FINETUNED_FOR_FEW_EPOCHS_MULTIWOZ_DATA_SINGLE_SEED_ALL_CHECKPOINTS_STEP_100 = auto()
    ROBERTA_BASE_FINETUNED_FOR_FEW_EPOCHS_MULTIWOZ_DATA_SINGLE_SEED_FROZEN_LM_HEAD_ALL_CHECKPOINTS_STEP_100 = auto()
    ROBERTA_BASE_FINETUNED_FOR_FEW_EPOCHS_MULTIWOZ_AND_REDDIT_AND_WIKITEXT_DATA_SINGLE_SEED_ALL_CHECKPOINTS_STEP_400 = (
        auto()
    )

    ROBERTA_BASE_FINETUNED_FOR_MANY_EPOCHS = auto()

    # # # # # # # # # # # #
    # GPT-2 models

    GPT2_MEDIUM_WITHOUT_MODIFICATIONS = auto()

    GPT2_MEDIUM_FINETUNED_FOR_FEW_EPOCHS_MULTIWOZ_AND_REDDIT_AND_WIKITEXT_DATA_SINGLE_SEED_LAST_CHECKPOINT = auto()
    GPT2_MEDIUM_FINETUNED_FOR_FEW_EPOCHS_WIKITEXT_DATA_SINGLE_SEED_CHECKPOINTS_1200_1600 = auto()
    GPT2_MEDIUM_FINETUNED_FOR_FEW_EPOCHS_MULTIWOZ_AND_REDDIT_AND_WIKITEXT_DATA_SINGLE_SEED_CHECKPOINTS_1200_1600 = (
        auto()
    )


class FinetuningRegimeOption(StrEnum):
    """Options for the finetuning regime."""

    FEW_EPOCHS = auto()
    MANY_EPOCHS_WITH_OVERFITTING_RISK = auto()


class LanguageModelListOption(StrEnum):
    """Options for the language model list."""

    # # # #
    # RoBERTa-base models

    ROBERTA_BASE = auto()

    # Models fine-tuned on "old data" (i.e., data which was part of the pretraining data)
    # and models fine-tuned on "new data" (i.e., data which was created after the model was pre-trained)
    FINETUNED_ON_OLD_AND_NEW_DATA_FEW_EPOCHS_FROM_ROBERTA_BASE = auto()

    FINETUNED_ON_MULTIWOZ_DATA_FEW_EPOCHS_FROM_ROBERTA_BASE = auto()
    FINETUNED_ON_MULTIWOZ_DATA_FEW_EPOCHS_FROZEN_LM_HEAD_FROM_ROBERTA_BASE = auto()
    FINETUNED_ON_MULTIWOZ_AND_REDDIT_AND_WIKITEXT_DATA_FEW_EPOCHS_FROM_ROBERTA_BASE = auto()

    # Selected models
    SELECTED_FINETUNED_FEW_EPOCHS_FROM_ROBERTA_BASE = auto()
    SELECTED_FINETUNED_MANY_EPOCHS_FROM_ROBERTA_BASE = auto()

    FULL_FINETUNED_FEW_EPOCHS_FROM_ROBERTA_BASE = auto()

    # Models finetuned with different dropout rates
    WITH_005_015_02_DROPOUT_FINETUNED_ON_MULTIWOZ_SMALL_MANY_EPOCHS_FROM_ROBERTA_BASE = auto()
    WITH_006_007_DROPOUT_FINETUNED_ON_MULTIWOZ_SMALL_MANY_EPOCHS_FROM_ROBERTA_BASE = auto()

    # # # #
    # GPT-2 models
    GPT2_MEDIUM = auto()

    FINETUNED_ON_OLD_AND_NEW_DATA_FEW_EPOCHS_FROM_GPT2_MEDIUM = auto()
    FINETUNED_ON_MULTIWOZ_DATA_FEW_EPOCHS_FROM_GPT2_MEDIUM = auto()
    FINETUNED_ON_WIKITEXT_DATA_FEW_EPOCHS_FROM_GPT2_MEDIUM = auto()
    FINETUNED_ON_MULTIWOZ_AND_REDDIT_AND_WIKITEXT_DATA_FEW_EPOCHS_FROM_GPT2_MEDIUM = auto()

    # # # #
    # Other models
    SETSUMBT_SELECTED = auto()


class SeedListOption(StrEnum):
    """Options for the seed lists."""

    DO_NOT_SET = auto()
    ONE_SEED = auto()
    TWO_SEEDS = auto()
    FIVE_SEEDS = auto()
    FIXED_SEED_1234 = auto()
    FIXED_SEEDS_1234_1235_1236 = auto()
    FIXED_SEEDS_1235_1236 = auto()


class EmbeddingsDataPrepSamplingSeedListOption(StrEnum):
    """Options for the seed lists for the embeddings data preparation sampling."""

    DEFAULT = auto()
    TWO_SEEDS = auto()
    FIVE_SEEDS = auto()
    TEN_SEEDS = auto()
    TWENTY_SEEDS = auto()


class EmbeddingsDataPrepNumSamplesListOption(StrEnum):
    """Options for the number of samples in the embeddings data preparation sampling."""

    DEFAULT = auto()
    SINGLE_CHOICE_50000 = auto()
    SINGLE_CHOICE_100000 = auto()
    SINGLE_CHOICE_150000 = auto()
    SINGLE_CHOICE_250000 = auto()
    FIVE_CHOICES_10000_STEPS = auto()


class LocalEstimatesFilteringNumSamplesListOption(StrEnum):
    """Options for the number of samples for local estimates filtering."""

    DEFAULT = auto()
    SINGLE_CHOICE_60000 = auto()
    FEW_SMALL_STEPS_NUM_SAMPLES = auto()
    MEDIUM_SMALL_STEPS_NUM_SAMPLES = auto()

    RANGE_START_20000_STOP_110000_STEP_20000 = auto()
    RANGE_START_10000_STOP_110000_STEP_10000 = auto()


class LocalEstimatesPointwiseAbsoluteNNeighborsListOption(StrEnum):
    """Options for the number of neighbors for pointwise absolute local estimates."""

    DEFAULT = auto()
    SINGLE_CHOICE_128 = auto()
    POWERS_OF_TWO_UP_TO_1024 = auto()


class RunOnlySelectedConfigsOption(StrEnum):
    """Options to run only a single or selected config."""

    RUN_ALL = auto()
    RUN_ONLY_FIRST = auto()
    RUN_ONLY_LAST = auto()
    RUN_SINGLE_RANDOM = auto()
