import numpy as np
import os

# Define the dataset names and masked levels
dataset_names = [
    "data=one-year-of-tsla-on-reddit_rm-empty=True_spl-mode=proportions_spl-shuf=True_spl-seed=0_tr=0.8_va=0.1_te=0.1_ctxt=dataset_entry_feat-col=ner_tags",
    "data=wikitext-103-v1_rm-empty=True_spl-mode=proportions_spl-shuf=True_spl-seed=0_tr=0.8_va=0.1_te=0.1_ctxt=dataset_entry_feat-col=ner_tags",
"data=multiwoz21_rm-empty=True_spl-mode=do_nothing_ctxt=dataset_entry_feat-col=ner_tags"
]

dataset_name = dataset_names[0]
# Define the base directory (unchanged)
base_directory = "data=one-year-of-tsla-on-reddit_rm-empty=True_spl-mode=proportions_spl-shuf=True_spl-seed=0_tr=0.8_va=0.1_te=0.1_ctxt=dataset_entry_feat-col=ner_tags/split=validation_samples=10000_sampling=random_sampling-seed=777/edh-mode=regular_lvl=token/add-prefix-space=True_max-len=512/model=gpt2-medium-causal_lm-defaults_wikitext-103-v1-rm-empty-True-proportions-True-0-0.8-0.1-0.1-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5_seed-1234_ckpt-1200_task=causal_lm_dr=defaults/layer={layer}_agg=mean/norm=None/sampling=random_seed=42_samples=150000/desc=twonn_samples=60000_zerovec=keep_dedup=array_deduplicator_noise=do_nothing/n-neighbors-mode=absolute_size_n-neighbors=128"

# Loop through layers [-23, -21, -19, ..., -1]
means = []
stds = []

for layer_index, layer in enumerate(range(-23, 0, 2), start=1):
    # Replace the layer placeholder in the base directory with the current layer
    layer_directory = base_directory.format(layer=layer)
    file_path = os.path.join(layer_directory, "local_estimates_pointwise_array.npy")

    try:
        # Load the .npy file
        data_array = np.load(file_path)

        # Compute the average and standard deviation
        avg = np.mean(data_array)
        std = np.std(data_array)
        means.append((layer_index, avg))
        stds.append((layer_index, std))
    except FileNotFoundError:
        print(f"File not found for layer {layer}: {file_path}")
    except Exception as e:
        print(f"An error occurred for layer {layer}: {e}")

# Print the means
print("Means:")
for index, mean in means:
    print(f"{index:<4} {mean:.8f}")

for index, std in stds:
    print(f"{index:<4} {std:.8f}")