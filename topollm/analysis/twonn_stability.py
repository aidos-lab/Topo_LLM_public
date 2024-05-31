import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

analysis_folder = 'twonn'
file_name = 'data-multiwoz21_split-validation_ctxt-dataset_entry_samples-1000_model-roberta-base_mask-no_masking_model-roberta-base_finetuned-on-multiwoz21_ftm-standard_overfitted_mask-no_masking__[-1]_'

min_num_samples = pd.read_pickle('../../data/analysis/'+analysis_folder+
        '/token_lvl/'+ file_name +
                                      str(10000) + '.pkl').shape[0]

last_layer = []
last_layer_finetuned = []

for i in [10000, 20000, 30000, 50000]:
    df = pd.read_pickle('../../data/analysis/'+analysis_folder+
        '/token_lvl/'+ file_name + str(
            i) + '.pkl')

    twonn = np.array(list(df.twonn))[:min_num_samples]
    twonn_finetuned = np.array(list(df.twonn_finetuned))[:min_num_samples]
    last_layer.append(twonn)
    last_layer_finetuned.append(twonn_finetuned)

plt.plot([str(10000), str(20000), str(30000), str(50000)], np.array(last_layer))
plt.xlabel('Sample Size')
plt.ylabel('TwoNN Estimate')
plt.title('TwoNN Estimates of corresponding points Over Sample Sizes (Multiwoz data embedded with RoBERTa base model)')
plt.show()