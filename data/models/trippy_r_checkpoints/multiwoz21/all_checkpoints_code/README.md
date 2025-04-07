## Introduction

Generalising dialogue state tracking (DST) to new data is especially challenging due to the strong reliance on abundant and fine-grained supervision during training. Sample sparsity, distributional shift and the occurrence of new concepts and topics frequently lead to severe performance degradation during inference. TripPy-R (pronounced "Trippier"), robust triple copy strategy DST, can use a training strategy to build extractive DST models without the need for fine-grained manual span labels ("spanless training"). Further, two novel input-level dropout methods mitigate the negative impact of sample sparsity. TripPy-R uses a new model architecture with a unified encoder that supports value as well as slot independence by leveraging the attention mechanism, making it zero-shot capable. The framework combines the strengths of triple copy strategy DST and value matching to benefit from complementary predictions without violating the principle of ontology independence. In our paper we demonstrate that an extractive DST model can be trained without manual span labels. Our architecture and training strategies improve robustness towards sample sparsity, new concepts and topics, leading to state-of-the-art performance on a range of benchmarks.

## Recent updates

- 2023.08.08: Initial commit

## How to run

Two example scripts are provided for how to use TripPy-R.

`DO.example` will train and evaluate a model with recommended settings with the default supervised training strategy.

`DO.example.spanless` will train and evaluate a model with recommended settings with the novel spanless training strategy. The training consists of three steps: 1) Training a proto-DST that learns to tag the positions of queried subsequences in an input sequence. 2) Applying the proto-DST to tag the positions of slot-value occurrences in the training data. 3) Training the DST using the automatic labels produced by the previous step.

See below table for expected performance per dataset and training strategy. Our scripts use the parameters that were used for experiments in our paper "Robust Dialogue State Tracking with Weak Supervision and Sparse Data". Thus, performance will be similar to the reported ones. For more challenging datasets with longer dialogues, better performance may be achieved by using the maximum sequence length of 512.

## Trouble-shooting

When conducting spanless training, the training of the proto-DST (Step 1 of 3, see above) is rather sensitive to the training hyperparameters such as learning rate, warm-up ratio and max. number of epochs, as well as the random model initialization. We recommend the hyperparameters as listed in the example script above. If the proto-DST's tagging performance (Step 2 of 3) remains below expectations for one or more slots, try running the training with a different random initialization, i.e. pick a different random seed, while using the recommended hyperparameters.

## Datasets

Supported datasets are:
- sim-M (https://github.com/google-research-datasets/simulated-dialogue.git)
- sim-R (https://github.com/google-research-datasets/simulated-dialogue.git)
- WOZ 2.0 (see https://gitlab.cs.uni-duesseldorf.de/general/dsml/trippy-public.git)
- MultiWOZ 2.0 (https://github.com/budzianowski/multiwoz.git)
- MultiWOZ 2.1 (https://github.com/budzianowski/multiwoz.git)
- MultiWOZ 2.1 legacy version (see https://gitlab.cs.uni-duesseldorf.de/general/dsml/trippy-public.git)
- MultiWOZ 2.2 (https://github.com/budzianowski/multiwoz.git)
- MultiWOZ 2.3 (https://github.com/lexmen318/MultiWOZ-coref.git)
- MultiWOZ 2.4 (https://github.com/smartyfh/MultiWOZ2.4.git)
- Unified data format (currently supported: MultiWOZ 2.1) (see https://github.com/ConvLab/ConvLab-3)

See the [README file](https://gitlab.cs.uni-duesseldorf.de/general/dsml/trippy-public/-/blob/master/data/README.md) in 'data/' in the original [TripPy repo](https://gitlab.cs.uni-duesseldorf.de/general/dsml/trippy-public) for more details how to obtain and prepare the datasets for use in TripPy-R.

The ```--task_name``` is
- 'sim-m', for sim-M
- 'sim-r', for sim-R
- 'woz2', for WOZ 2.0
- 'multiwoz21', for MultiWOZ 2.0-2.4
- 'multiwoz21_legacy', for MultiWOZ 2.1 legacy version
- 'unified', for ConvLab-3's unified data format

With a sequence length of 180, you should expect the following average JGA:

| Dataset | Normal training | Spanless training |
| ------ | ------ | ------ |
| MultiWOZ 2.0 | 51% | tbd |
| MultiWOZ 2.1 | 56% | 55% |
| MultiWOZ 2.1 legacy | 56% | 55% |
| MultiWOZ 2.2 | 56% | tbd |
| MultiWOZ 2.3 | 62% | tbd |
| MultiWOZ 2.4 | 69% | tbd |
| sim-M | 95% | 95% |
| sim-R | 92% | 92% |
| WOZ 2.0 | 92% | 91% |

## Requirements

- torch (tested: 1.12.1)
- transformers (tested: 4.18.0)
- tensorboardX (tested: 2.5.1)

## Citation

This work is published as [Robust Dialogue State Tracking with Weak Supervision and Sparse Data ](https://doi.org/10.1162/tacl_a_00513)

If you use TripPy-R in your own work, please cite our work as follows:

```
@article{heck-etal-2022-robust,
    title = "Robust Dialogue State Tracking with Weak Supervision and Sparse Data",
    author = "Heck, Michael and Lubis, Nurul and van Niekerk, Carel and
              Feng, Shutong and Geishauser, Christian and Lin, Hsien-Chin and Ga{\v{s}}i{\'c}, Milica",
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "10",
    year = "2022",
    address = "Cambridge, MA",
    publisher = "MIT Press",
    url = "https://aclanthology.org/2022.tacl-1.68",
    doi = "10.1162/tacl_a_00513",
    pages = "1175--1192",
}
```
