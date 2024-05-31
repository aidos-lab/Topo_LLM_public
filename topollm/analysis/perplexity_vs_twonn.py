import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

checkpoint_list = list(np.arange(10000,31600,400))
#checkpoint_list = [400,800,1200,1600,2000,2400,2800]
#checkpoint_list = [2800]
mean_perplexity = []
mean_twonn = []
for i in checkpoint_list:
    twonn = pd.read_pickle('../../data/analysis/twonn/token_lvl/data-reddit_split-validation_ctxt-dataset_entry_samples-3000_model-roberta-base_mask-no_masking_heckpoint-'+str(i)+'__[-1]_50000.pkl')
    #meta_frame = pd.read_pickle('../../data/analysis/prepared/data-multiwoz21_split-validation_ctxt-dataset_entry_samples-3000/lvl-token/add-prefix-space-False_max-len-512/model-roberta-base_finetuned-on-reddit_ftm-standard_checkpoint-'+str(i)+'/layer-[-1]_agg-mean/norm-None/array_dir/embeddings_token_lvl_30000_samples_paddings_removed_meta.pkl')
    perplexity = pd.read_pickle('../../data/embeddings/perplexity/data-one-year-of-tsla-on-reddit_split-validation_ctxt-dataset_entry_samples-3000/lvl-token/add-prefix-space-False_max-len-512/model-roberta-base_finetuned-on-one-year-of-tsla-on-reddit_ftm-standard_overfitted_checkpoint-'+str(i)+'_mask-no_masking/layer-[-1]_agg-mean/norm-None/perplexity_dir/perplexity_results_list_new_format.pkl')

    token_perplexities = []
    token_strings = []
    token_ids = []
    for j in range(len(perplexity)):
        token_perplexities += list(perplexity[j][1].token_perplexities)
        token_strings += list(perplexity[j][1].token_strings)
        token_ids += list(perplexity[j][1].token_ids)

    token_perplexities = np.array(token_perplexities)
    token_strings = np.array(token_strings)
    token_ids = np.array(token_ids)
    token_strings = token_strings[token_ids!=2]

    np.random.seed(42)
    sample_size = 30000

    if len(token_strings) >= sample_size:
        idx = np.random.choice(
            range(len(token_strings)),
            replace=False,
            size=sample_size,
        )
    else:
        idx = np.random.choice(
            range(len(token_strings)),
            replace=False,
            size=len(token_strings),
        )

    token_strings = token_strings[idx]
    token_perplexities = token_perplexities[idx]

    mean_twonn.append(np.array(list(twonn.twonn_finetuned)).mean())
    mean_perplexity.append(token_perplexities.mean())

print(np.array(checkpoint_list))
print(mean_perplexity)
print(mean_twonn)

plt.plot([str(x) for x in checkpoint_list],mean_twonn)
plt.title('Mean TwoNN estimates for Reddit data embeddings at different checkpoints')
plt.xlabel('Training checkpoint')
plt.ylabel('TwoNN estimate')
#plt.plot(checkpoint_list,mean_perplexity)
    #plt.scatter(list(twonn.twonn_finetuned),list(token_perplexities[:1500]))
plt.show()
