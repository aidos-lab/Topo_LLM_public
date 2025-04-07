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
import math

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler)
from torch.utils.data.distributed import DistributedSampler
from torch.optim import (AdamW)
from tqdm import tqdm, trange

from tensorboardX import SummaryWriter
from transformers import (get_linear_schedule_with_warmup)
from utils_run import (set_seed, to_device, from_device,
                       save_checkpoint, load_and_cache_examples,
                       dilate_and_erode)

logger = logging.getLogger(__name__)

        
def train_proto(args, train_dataset, dev_dataset, model, tokenizer, processor):
    """ Train the proto model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    # This controls the item return function (__getitem__).
    train_dataset.proto()
    if dev_dataset is not None:
        dev_dataset.proto()

    model.eval() # No dropout

    # If sequences were not tokenized yet, do so now, then save tokenized and encoded sequences.
    if not train_dataset.load_tokenized_sequences(args.output_dir):
        train_dataset.tokenize_sequences(max_len=args.rand_seq_max_len)
        train_dataset.save_tokenized_sequences(args.output_dir, overwrite=False)
    if dev_dataset is not None and not dev_dataset.load_tokenized_sequences(args.output_dir):
        dev_dataset.tokenize_sequences(max_len=args.rand_seq_max_len)
        dev_dataset.save_tokenized_sequences(args.output_dir, overwrite=False)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    if args.save_epochs > 0:
        args.save_steps = t_total // args.num_train_epochs * args.save_epochs
    assert args.save_steps == 0 or args.patience < 0

    num_warmup_steps = int(t_total * args.warmup_proportion)
    if args.patience > 0:
        patience = args.patience
        cur_min_loss = math.inf

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total)
    scaler = torch.cuda.amp.GradScaler()
    if 'cuda' in args.device.type:
        autocast = torch.cuda.amp.autocast(enabled=args.fp16)
    else:
        autocast = torch.cpu.amp.autocast(enabled=args.fp16)

    # multi-gpu training
    model_single_gpu = model
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model_single_gpu)

    # Distributed training
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Pretrain!
    logger.info("***** Running proto training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Warmup steps = %d", num_warmup_steps)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)

    for e_itr, _ in enumerate(train_iterator):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        train_dataset.update_samples_for_proto(max_len=args.rand_seq_max_len)

        all_train_results = []
        for step, batch in enumerate(epoch_iterator):
            model.train()

            batch = to_device(batch, args.device)
            with autocast:
                outputs = model(batch, step=step, mode="proto") # calls the "forward" def.
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            outputs = from_device(outputs)
            batch = from_device(batch)

            train_results = eval_metric_proto(args, model, tokenizer, batch, outputs, threshold=0.5)
            all_train_results.append(train_results)

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            epoch_iterator.set_postfix({'loss': loss.item()})

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log metrics
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                # Save model checkpoint
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    save_checkpoint(args, global_step, model, prefix='proto')

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        # Generate final results
        final_train_results = {'loss': torch.tensor(0), 'accuracy': torch.tensor(0)}
        if len(all_train_results) > 0:
            for k in all_train_results[0].keys():
                final_train_results[k] = torch.stack([r[k] for r in all_train_results]).sum() / len(train_dataset)

        # Only evaluate when single GPU otherwise metrics may not average well
        if args.local_rank == -1 and dev_dataset is not None:
            results = evaluate_proto(args, dev_dataset, model_single_gpu, tokenizer, processor, no_print=True, prefix=global_step)
            for key, value in results.items():
                tb_writer.add_scalar('eval_{}'.format(key), value, global_step)

            # Patience
            if args.patience > 0:
                if args.early_stop_criterion == "loss":
                    criterion = results['proto_loss'].item()
                elif args.early_stop_criterion == "goal":
                    criterion = -1 * results['proto_accuracy'].item()
                else:
                    logger.warn("Early stopping criterion %s not known. Aborting" % (args.early_stop_criterion))
                if criterion > cur_min_loss:
                    patience -= 1
                else:
                    # Save model checkpoint
                    patience = args.patience
                    save_checkpoint(args, global_step, model, prefix='proto', keep_only_last_checkpoint=True)
                    cur_min_loss = criterion
                train_iterator.set_postfix({'patience': patience,
                                            'train loss': final_train_results['proto_loss'].item(),
                                            'eval loss': results['proto_loss'].item(),
                                            'train acc': final_train_results['proto_accuracy'].item(),
                                            'eval acc': results['proto_accuracy'].item()})
                if patience == 0:
                    train_iterator.close()
                    break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    # This controls the item return function (__getitem__).
    train_dataset.reset()
    if dev_dataset is not None:
        dev_dataset.reset()

    return global_step, tr_loss / global_step


def evaluate_proto(args, dataset, model, tokenizer, processor, no_print=False, prefix=""):
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    dataset.proto() # This controls the item return function (__getitem__).

    model.eval() # No dropout

    if not dataset.load_tokenized_sequences(args.output_dir):
        dataset.tokenize_sequences(max_len=args.rand_seq_max_len)

    dataset.update_samples_for_proto(max_len=args.rand_seq_max_len)

    args.eval_batch_size = args.per_gpu_eval_batch_size
    eval_sampler = SequentialSampler(dataset) # Note that DistributedSampler samples randomly
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation of proto training {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_results = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()

        with torch.no_grad():
            batch = to_device(batch, args.device)
            outputs = model(batch, mode="proto")
            outputs = from_device(outputs)
            batch = from_device(batch)

        unique_ids = [dataset.features[i.item()].guid for i in batch['example_id']]
        values = [dataset.features[i.item()].values for i in batch['example_id']]
        input_ids = [dataset.features[i.item()].input_ids for i in batch['example_id']]
        inform = [dataset.features[i.item()].inform for i in batch['example_id']]

        results = eval_metric_proto(args, model, tokenizer, batch, outputs, threshold=0.5)
        all_results.append(results)
        if not no_print:
            predict_and_print_proto(args, model, tokenizer, batch, outputs, unique_ids, input_ids, values, inform)

    # Generate final results
    final_results = {}
    for k in all_results[0].keys():
        final_results[k] = torch.stack([r[k] for r in all_results]).sum() / len(dataset)

    dataset.reset() # This controls the item return function (__getitem__).

    return final_results


def eval_metric_proto(args, model, tokenizer, batch, outputs, threshold=0.0, dae=False):
    loss = outputs[0]
    logits = outputs[1]

    input_ids = []
    for i in range(len(batch['input_ids'])):
        clipped = batch['input_ids'][i].tolist()
        clipped = clipped[:len(clipped) - clipped[::-1].index(tokenizer.sep_token_id)]
        input_ids.append(clipped)

    metric_dict = {}

    mean = []
    for i in range(len(batch['input_ids'])):
        mean.append(torch.mean(logits[i][:len(input_ids[i])]))
    mean = torch.stack(mean)
    norm_logits = torch.clamp(logits - mean.unsqueeze(1), min=0) / logits.max(1)[0].unsqueeze(1)

    start_pos = batch['start_pos']

    if dae:
        token_prediction = dilate_and_erode(norm_logits, threshold)
    else:
        token_prediction = norm_logits > threshold
    token_prediction[:, 0] = False # Ignore <s>
    token_correctness = torch.all(torch.eq(token_prediction, start_pos), 1).float()
    token_accuracy = token_correctness.sum()
    metric_dict['proto_loss'] = loss
    metric_dict['proto_accuracy'] = token_accuracy
    return metric_dict


def predict_and_print_proto(args, model, tokenizer, batch, outputs, ids, input_ids_unmasked, values, inform):
    per_slot_start_logits = outputs[1]

    for i in range(len(ids)):
        input_tokens = tokenizer.convert_ids_to_tokens(input_ids_unmasked[i])

        input_ids = batch['input_ids'][i].tolist()
        input_ids = input_ids[:len(input_ids) - input_ids[::-1].index(tokenizer.sep_token_id)]

        token_norm_weights = {}
        pos_i = batch['start_pos'][i].tolist()
        token_weights = per_slot_start_logits[i][:len(input_ids)]
        token_norm_weights = torch.clamp(token_weights - torch.mean(token_weights), min=0) / torch.max(token_weights)

        print(ids[i])
        for k in range(len(input_ids)):
            bold = False
            print(" ", end="")
            t_weight = token_norm_weights[k]
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
            if pos_i[k]:
                bold = True
            if bold:
                print("\033[1m%s\033[0m" % (input_tokens[k]))
            else:
                print("%s" % (input_tokens[k]))
