import yaml
import hydra
import omegaconf
import numpy as np
from topollm.compute_embeddings import run_compute_embeddings
from topollm.analysis import data_prep_all, data_prep_mean, data_prep_cls

@hydra.main(
    config_path="../../configs",
    config_name="main_config",
    version_base="1.2",
)
def main(
    config: omegaconf.DictConfig,
) -> None:
    """
    for i in np.arange(1, 13):
        i = int(i)
        with open('../../configs/embeddings/embedding_extraction/last_hidden.yaml', 'r') as file:
            hidden_state = yaml.safe_load(file)
            hidden_state['layer_indices'] = [-i]
        with open('../../configs/embeddings/embedding_extraction/' + str(i) + '_last_hidden.yaml', 'w') as outfile:
            yaml.dump(hidden_state, outfile)

        config.embeddings.embedding_extraction.layer_indices = [-i]
        config.language_model.pretrained_model_name_or_path = 'roberta-base'
        config.language_model.short_model_name = '${embeddings.language_model.pretrained_model_name_or_path}'

        run_compute_embeddings.main(config)
        data_prep_all.main(config)
    """
    for i in np.arange(1, 13):
        i = int(i)

        with open('../../configs/embeddings/basic_embeddings.yaml', 'r') as file:
            basic = yaml.safe_load(file)
            basic['defaults'][1]['embedding_extraction'] = str(i) + '_last_hidden'
            basic['defaults'][2]['language_model'] = 'roberta-base_finetuned-on-sgd-train-10000_context-dialogue'
        with open('../../configs/embeddings/basic_embeddings.yaml', 'w') as outfile:
            yaml.dump(basic, outfile)

        config.embeddings.embedding_extraction.layer_indices = [-i]
        config.language_model.pretrained_model_name_or_path = '${paths.data_dir}/models/finetuned_models/data-sgd_split-train_ctxt-dataset_entry_samples-10000/ftm-standard/lora-None/ep-5/model_files'
        config.language_model.short_model_name = 'roberta-base_finetuned-on-sgd-train-10000_context-dialogue'

        run_compute_embeddings.main(config)
        data_prep_all.main(config)



    return None



if __name__ == "__main__":
    main()  # type: ignore