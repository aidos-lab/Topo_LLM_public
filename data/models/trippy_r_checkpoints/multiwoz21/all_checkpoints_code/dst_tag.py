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

import torch
from torch.utils.data import (DataLoader, SequentialSampler)
from tqdm import tqdm

from utils_run import (set_seed, to_device, from_device,
                       save_checkpoint, load_and_cache_examples,
                       dilate_and_erode)

logger = logging.getLogger(__name__)


def tag_values(args, dataset, model, tokenizer, processor, no_print=False, prefix="", threshold=0.0, dae=False):
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    model.eval() # No dropout

    dataset.tag() # This controls the item return function (__getitem__).

    dataset.encode_slot_values(val_rep_mode="encode", val_rep="v")

    args.eval_batch_size = args.per_gpu_eval_batch_size
    eval_sampler = SequentialSampler(dataset) # Note that DistributedSampler samples randomly
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Tag!
    logger.info("***** Running value tagging {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_labels = []
    all_results = []
    for batch in tqdm(eval_dataloader, desc="Tagging"):
        model.eval()

        with torch.no_grad():
            batch['encoded_slot_values'] = dataset.encoded_slot_values
            batch = to_device(batch, args.device)
            outputs = model(batch, mode="tag")
            outputs = from_device(outputs)
            batch = from_device(batch)

        unique_ids = [dataset.features[i.item()].guid for i in batch['example_id']]
        values = [dataset.features[i.item()].values for i in batch['example_id']]
        input_ids = [dataset.features[i.item()].input_ids for i in batch['example_id']]
        inform = [dataset.features[i.item()].inform for i in batch['example_id']]

        auto_labels, results = label_and_eval_tags(args, model, tokenizer, batch, outputs, values, threshold=threshold, dae=dae)
        all_labels.append(auto_labels)
        all_results.append(results)
        if not no_print:
            predict_and_print_tags(args, model, tokenizer, batch, outputs, unique_ids, input_ids, values, inform)

    # Generate final labels
    final_labels = {slot: {} for slot in model.slot_list}
    for b in all_labels:
        for s in b:
            final_labels[s].update(b[s])

    # Generate final results
    final_results = {}
    for k in all_results[0].keys():
        final_results[k] = (torch.stack([r[k] for r in all_results]).sum() / len(dataset)).item()

    dataset.reset() # This controls the item return function (__getitem__).

    return final_labels, final_results


def label_and_eval_tags(args, model, tokenizer, batch, outputs, values, threshold=0.0, dae=False):
    per_slot_start_logits = outputs[0]

    input_ids = []
    for i in range(len(batch['input_ids'])):
        clipped = batch['input_ids'][i].tolist()
        clipped = clipped[:len(clipped) - clipped[::-1].index(tokenizer.sep_token_id)]
        input_ids.append(clipped)

    auto_labels = {}
    metric_dict = {}
    per_slot_correctness = {}
    for s_itr, slot in enumerate(model.slot_list):
        start_logits = per_slot_start_logits[:, s_itr]
        mean = []
        for i in range(len(batch['input_ids'])):
            mean.append(torch.mean(start_logits[i][:len(input_ids[i])]))
        mean = torch.stack(mean)
        norm_logits = torch.clamp(start_logits - mean.unsqueeze(1), min=0) / start_logits.max(1)[0].unsqueeze(1)

        start_pos = batch['start_pos'][slot]

        # "is pointable" means whether there is a span to be detected.
        token_is_pointable = (start_pos.sum(1) > 0).float()

        if dae:
            token_prediction = dilate_and_erode(norm_logits, threshold)
        else:
            token_prediction = norm_logits > threshold
        token_prediction[:, 0] = False # Ignore [CLS]/<s>
        token_correctness = torch.all(torch.eq(token_prediction, start_pos), 1).float()
        token_accuracy = (token_correctness * token_is_pointable).sum() + (1 - token_is_pointable).sum()
        total_correctness = token_correctness * token_is_pointable + (1 - token_is_pointable)

        metric_dict['eval_accuracy_%s' % slot] = token_accuracy
        per_slot_correctness[slot] = total_correctness

        auto_labels[slot] = {}
        for i in range(len(batch['input_ids'])):
            auto_labels[slot][int(batch['example_id'][i])] = token_prediction[i] * token_is_pointable[i]

    goal_correctness = torch.stack([c for c in per_slot_correctness.values()], 1).prod(1)
    goal_accuracy = goal_correctness.sum()
    metric_dict['eval_accuracy_goal'] = goal_accuracy
    return auto_labels, metric_dict


def predict_and_print_tags(args, model, tokenizer, batch, outputs, ids, input_ids_unmasked, values, inform):
    per_slot_start_logits = outputs[0]

    class_types = model.class_types

    for i in range(len(ids)):
        input_tokens = tokenizer.convert_ids_to_tokens(input_ids_unmasked[i])

        input_ids = batch['input_ids'][i].tolist()
        input_ids = input_ids[:len(input_ids) - input_ids[::-1].index(tokenizer.sep_token_id)]

        pos_i = {}
        clb_i = {}
        token_norm_weights = {}
        for s_itr, slot in enumerate(model.slot_list):
            pos_i[slot] = batch['start_pos'][slot][i].tolist()
            clb_i[slot] = batch['class_label_id'][slot][i]
            token_weights = per_slot_start_logits[i][s_itr][:len(input_ids)]
            token_norm_weights[slot] = torch.clamp(token_weights - torch.mean(token_weights), min=0) / max(token_weights)

        print(ids[i])
        print(" ", end="")
        for slot in model.slot_list:
            if clb_i[slot] == class_types.index('copy_value'):
                print("\033[1m%s\033[0m " % (slot[0]), end="")
            else:
                print(slot[0] + " ", end="")
        print()
        for k in range(len(input_ids)):
            bold = False
            print(" ", end="")
            for slot in model.slot_list:
                t_weight = token_norm_weights[slot][k]
                if t_weight == 0.0:
                    print("  ", end="")
                elif t_weight < 0.25:
                    print("\u2591 ", end="")
                elif t_weight < 0.5:
                    print("\u2592 ", end="")
                elif t_weight < 0.75:
                    print("\u2593 ", end="")
                else:
                    print("\u2588 ", end="")
                if pos_i[slot][k]:
                    bold = True
            if bold:
                print("\033[1m%s\033[0m" % (input_tokens[k]))
            else:
                print("%s" % (input_tokens[k]))
