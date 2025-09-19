"""Variables to describe the different data selections."""


def get_data_folder_list() -> list[str]:
    data_folder_list: list[str] = [
        "data-multiwoz21_split-train_ctxt-dataset_entry_samples-10000_feat-col-ner_tags",
        "data-multiwoz21_split-validation_ctxt-dataset_entry_samples-10000_feat-col-ner_tags",
        "data-multiwoz21_split-test_ctxt-dataset_entry_samples-10000_feat-col-ner_tags",
        "data-multiwoz21_split-validation_ctxt-dataset_entry_samples-3000_feat-col-ner_tags",
        "data-multiwoz21_split-test_ctxt-dataset_entry_samples-3000_feat-col-ner_tags",
        "data-one-year-of-tsla-on-reddit_split-train_ctxt-dataset_entry_samples-10000_feat-col-ner_tags",
        "data-one-year-of-tsla-on-reddit_split-validation_ctxt-dataset_entry_samples-3000_feat-col-ner_tags",
        "data-one-year-of-tsla-on-reddit_split-test_ctxt-dataset_entry_samples-3000_feat-col-ner_tags",
    ]

    return data_folder_list


def get_model_folder_list() -> list[str]:
    model_folder_list: list[str] = [
        "model-roberta-base_task-masked_lm",
        "model-model-roberta-base_task-masked_lm_multiwoz21-train-10000-ner_tags_ftm-standard_lora-None_5e-05-constant-0.01-50_seed-1234_ckpt-14400_task-masked_lm",
        "model-model-roberta-base_task-masked_lm_multiwoz21-train-10000-ner_tags_ftm-standard_lora-None_5e-05-constant-0.01-50_seed-1234_ckpt-31200_task-masked_lm",
        "model-model-roberta-base_task-masked_lm_multiwoz21-train-10000-ner_tags_ftm-standard_lora-None_5e-05-constant-0.01-50_seed-1234_ckpt-400_task-masked_lm",
        "model-model-roberta-base_task-masked_lm_one-year-of-tsla-on-reddit-train-10000-ner_tags_ftm-standard_lora-None_5e-05-constant-0.01-50_seed-1234_ckpt-14400_task-masked_lm",
        "model-model-roberta-base_task-masked_lm_one-year-of-tsla-on-reddit-train-10000-ner_tags_ftm-standard_lora-None_5e-05-constant-0.01-50_seed-1234_ckpt-31200_task-masked_lm",
        "model-model-roberta-base_task-masked_lm_one-year-of-tsla-on-reddit-train-10000-ner_tags_ftm-standard_lora-None_5e-05-constant-0.01-50_seed-1234_ckpt-400_task-masked_lm",
    ]

    return model_folder_list
