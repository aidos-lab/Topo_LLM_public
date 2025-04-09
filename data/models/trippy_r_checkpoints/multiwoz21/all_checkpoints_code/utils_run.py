# coding=utf-8
#
# Copyright 2020-2022 Heinrich Heine University Duesseldorf
#
# Part of this code is based on the source code of BERT-DST
# (arXiv:1907.03040)
# Part of this code is based on the source code of Transformers
# (arXiv:1910.03771)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import random

import numpy as np
import torch
from utils_dst import TrippyDataset

logger = logging.getLogger(__name__)


def print_header():
    logger.info(r" _________  ________  ___  ________  ________  ___    ___        ________     ")
    logger.info(r"|\___   ___\\\   __  \|\  \|\   __  \|\   __  \|\  \  /  /|      |\   __  \    ")
    logger.info(r"\|___ \  \_\ \  \|\  \ \  \ \  \|\  \ \  \|\  \ \  \/  / /______\ \  \|\  \   ")
    logger.info(r"     \ \  \ \ \   _  _\ \  \ \   ____\ \   ____\ \    / /\_______\ \   _  _\  ")
    logger.info(r"      \ \  \ \ \  \\\  \\\ \  \ \  \___|\ \  \___|\/  /  /\|_______|\ \  \\\  \| ")
    logger.info(r"       \ \__\ \ \__\\\ _\\\ \__\ \__\    \ \__\ __/  / /             \ \__\\\ _\ ")
    logger.info(r"        \|__|  \|__|\|__|\|__|\|__|     \|__||\___/ /               \|__|\|__|")
    logger.info(r"          (c) 2022 Heinrich Heine University \|___|/                          ")
    logger.info("")


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_device(batch, device):
    if isinstance(batch, tuple):
        batch_on_device = tuple([to_device(element, device) for element in batch])
    if isinstance(batch, dict):
        batch_on_device = {k: to_device(v, device) for k, v in batch.items()}
    else:
        batch_on_device = batch.to(device) if batch is not None else batch
    return batch_on_device


def from_device(batch):
    if isinstance(batch, tuple):
        batch_on_cpu = tuple([from_device(element) for element in batch])
    elif isinstance(batch, dict):
        batch_on_cpu = {k: from_device(v) for k, v in batch.items()}
    else:
        batch_on_cpu = batch.cpu() if batch is not None else batch
    return batch_on_cpu


def save_checkpoint(args, global_step, model, prefix="", keep_only_last_checkpoint=False):
    if len(prefix) > 0:
        prefix = prefix + "_"
    if keep_only_last_checkpoint:
        output_dir = os.path.join(args.output_dir, prefix + "checkpoint")
    else:
        output_dir = os.path.join(args.output_dir, prefix + "checkpoint-{}".format(global_step))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    torch.save(args, os.path.join(output_dir, prefix + "training_args.bin"))
    logger.info("Saving model checkpoint after global step %d to %s" % (global_step, output_dir))
    with open(os.path.join(args.output_dir, "last_" + prefix + "checkpoint.txt"), "w") as f:
        f.write("{}checkpoint-{}".format(prefix, global_step))


def load_and_cache_examples(args, model, tokenizer, processor, dset="train", evaluate=False, automatic_labels=None):
    assert dset in ["train", "dev", "test"]

    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    dataset = TrippyDataset(
        args,
        examples=None,
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        dset=dset,
        evaluate=evaluate,
        automatic_labels=automatic_labels,
    )

    # Load data features from cache or dataset file
    cached_file = os.path.join(os.path.dirname(args.output_dir), "cached_{}_features{}".format(dset, args.cache_suffix))
    if os.path.exists(cached_file) and not args.overwrite_cache and not args.no_cache:
        dataset.load_features_from_file(cached_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        processor_args = {
            "no_label_value_repetitions": args.no_label_value_repetitions,
            "swap_utterances": args.swap_utterances,
            "delexicalize_sys_utts": args.delexicalize_sys_utts,
            "unk_token": tokenizer.unk_token,
        }
        if dset == "dev":
            examples = processor.get_dev_examples(processor_args)
        elif dset == "test":
            examples = processor.get_test_examples(processor_args)
        elif dset == "train":
            examples = processor.get_train_examples(processor_args)
        else:
            logger.warning('Unknown dataset "%s". Aborting' % (dset))
        dataset.build_features_from_examples(examples)
        if not args.no_cache:
            if args.local_rank in [-1, 0]:
                dataset.save_features_to_file(cached_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    return dataset


def dilate_and_erode(weights, threshold):
    def dilate(seq):
        result = np.clip(np.convolve(seq, [1, 1, 1], mode="same"), 0, 1)
        return result

    def erode(seq):
        result = (~np.clip(np.convolve(~seq.astype(bool), [1, 1, 1], mode="same"), 0, 1).astype(bool)).astype(float)
        return result

    result = []
    for seq in weights:
        d = dilate(seq)
        dt = d > threshold
        e = erode(dt)
        result.append(torch.tensor(e))
    result = torch.stack(result)
    return result
