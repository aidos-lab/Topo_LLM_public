import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

analysis_folder = 'twonn'
min_num_samples = 356

last_layer = []
last_layer_finetuned = []

for i in [10000, 20000, 30000, 50000]:
    df = pd.read_pickle('../../data/analysis/'+analysis_folder+
        '/token_lvl/data-multiwoz21_split-validation_ctxt-dataset_entry_samples-1000_model-gpt2-medium_mask-no_masking_model-gpt2-medium_finetuned-on-multiwoz21_ftm-standard_checkpoint-2000_mask-no_masking__[-1]_' + str(
            i) + '.pkl')

    twonn = np.array(list(df.twonn))[:min_num_samples]
    twonn_finetuned = np.array(list(df.twonn_finetuned))[:min_num_samples]
    last_layer.append(twonn)
    last_layer_finetuned.append(twonn_finetuned)

plt.plot([str(10000), str(20000), str(30000), str(50000)], np.array(last_layer))
plt.xlabel('Sample Size')
plt.ylabel('TwoNN Estimate')
plt.title('TwoNN Estimates of corresponding points Over Sample Sizes (Multiwoz data embedded with GPT-2 base model)')
plt.show()