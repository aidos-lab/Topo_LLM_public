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

import json
import logging
import math
import os
import re

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import get_linear_schedule_with_warmup
from utils_run import dilate_and_erode, from_device, load_and_cache_examples, save_checkpoint, set_seed, to_device

logger = logging.getLogger(__name__)


def train(
    args,
    train_dataset,
    dev_dataset,
    automatic_labels,
    model,
    tokenizer,
    processor,
):
    """Train the model"""
    if args.local_rank in [-1, 0]:
        # Note:
        # `comment` is added to the logdir to make it unique and avoid a problem with PBS job arrays
        # trying to write to the same dir.
        # We add the PBS_ARRAY_INDEX to the logdir to make it unique.
        pbs_array_index = os.environ.get(
            "PBS_ARRAY_INDEX",
            default=None,
        )
        logger.info(
            msg=f"Using PBS_ARRAY_INDEX={pbs_array_index} to make logdir unique",  # noqa: G004 - low overhead
        )

        tb_writer = SummaryWriter(
            comment=pbs_array_index,
        )

    model.eval()  # No dropout

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
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total
    )
    scaler = torch.cuda.amp.GradScaler()
    if "cuda" in args.device.type:
        autocast = torch.cuda.amp.autocast(enabled=args.fp16)
    else:
        autocast = torch.cpu.amp.autocast(enabled=args.fp16)

    # multi-gpu training
    model_single_gpu = model
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model_single_gpu)

    # Distributed training
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Warmup steps = %d", num_warmup_steps)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)

    for e_itr, _ in enumerate(train_iterator):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        train_dataset.dropout_input()
        train_dataset.encode_slots()
        train_dataset.encode_slot_values()

        for step, batch in enumerate(epoch_iterator):
            model.train()

            # Add tokenized or encoded slot descriptions and encoded values to batch.
            # We do this here instead of in TrippyDataset.__getitem__() because we only
            # need them once, and not once for the entire batch.
            batch["slot_ids"] = []
            batch["slot_mask"] = []
            for slot in model.slot_list:
                batch["slot_ids"].append(train_dataset.encoded_slots_ids[slot][0])
                batch["slot_mask"].append(train_dataset.encoded_slots_ids[slot][1])
            batch["slot_ids"] = torch.stack(batch["slot_ids"])
            batch["slot_mask"] = torch.stack(batch["slot_mask"])
            batch["encoded_slot_values"] = train_dataset.encoded_slot_values

            batch = to_device(batch, args.device)
            with autocast:
                outputs = model(batch, step=step)  # calls the "forward" def.
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            outputs = from_device(outputs)
            batch = from_device(batch)

            cl_loss = outputs[1]
            tk_loss = outputs[2]
            tp_loss = outputs[3]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            epoch_iterator.set_postfix(
                {"loss": loss.item(), "cl": cl_loss.item(), "tk": tk_loss.item(), "tp": tp_loss.item()}
            )

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
                    tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                # Save model checkpoint
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    save_checkpoint(args, global_step, model)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        # Only evaluate when single GPU otherwise metrics may not average well
        if args.local_rank == -1 and dev_dataset is not None:
            results = evaluate(
                args,
                dev_dataset,
                model_single_gpu,
                tokenizer,
                processor,
                no_print=True,
                no_output=True,
                prefix=global_step,
            )
            for key, value in results.items():
                tb_writer.add_scalar("eval_{}".format(key), value, global_step)

            # Patience
            if args.patience > 0:
                if args.early_stop_criterion == "loss":
                    criterion = results["loss"].item()
                elif args.early_stop_criterion == "goal":
                    criterion = -1 * results["eval_accuracy_goal"].item()
                else:
                    logger.warn("Early stopping criterion %s not known. Aborting" % (args.early_stop_criterion))
                if criterion > cur_min_loss:
                    patience -= 1
                else:
                    # Save model checkpoint
                    patience = args.patience
                    save_checkpoint(args, global_step, model, keep_only_last_checkpoint=True)
                    cur_min_loss = criterion
                train_iterator.set_postfix(
                    {
                        "patience": patience,
                        "eval loss": results["loss"].item(),
                        "cl": results["cl_loss"].item(),
                        "tk": results["tk_loss"].item(),
                        "tp": results["tp_loss"].item(),
                        "eval goal": results["eval_accuracy_goal"].item(),
                    }
                )
                if patience == 0:
                    train_iterator.close()
                    break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, dataset, model, tokenizer, processor, no_print=False, no_output=False, prefix=""):
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    model.eval()  # No dropout

    dataset.encode_slots()
    dataset.save_encoded_slots(args.output_dir)

    dataset.encode_slot_values()
    dataset.save_encoded_slot_values(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size
    eval_sampler = SequentialSampler(dataset)  # Note that DistributedSampler samples randomly
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_results = []
    all_preds = []
    ds = {slot: "none" for slot in model.slot_list}
    diag_state = {
        slot: torch.tensor([0 for _ in range(args.eval_batch_size)]).to(args.device) for slot in model.slot_list
    }
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()

        # Reset dialog state if turn is first in the dialog.
        turn_itrs = [dataset.features[i.item()].guid.split("-")[2] for i in batch["example_id"]]
        reset_diag_state = np.where(np.array(turn_itrs) == "0")[0]
        for slot in model.slot_list:
            for i in reset_diag_state:
                diag_state[slot][i] = 0

        with torch.no_grad():
            batch["diag_state"] = diag_state  # Update
            batch["encoded_slots_pooled"] = dataset.encoded_slots_pooled
            batch["encoded_slots_seq"] = dataset.encoded_slots_seq
            batch["encoded_slot_values"] = dataset.encoded_slot_values
            batch = to_device(batch, args.device)
            outputs = model(batch)
            outputs = from_device(outputs)
            batch = from_device(batch)

        unique_ids = [dataset.features[i.item()].guid for i in batch["example_id"]]
        values = [dataset.features[i.item()].values for i in batch["example_id"]]
        input_ids = [dataset.features[i.item()].input_ids for i in batch["example_id"]]
        inform = [dataset.features[i.item()].inform for i in batch["example_id"]]

        # Update dialog state for next turn.
        for slot in model.slot_list:
            updates = outputs[8][slot].max(1)[1]
            for i, u in enumerate(updates):
                if u != 0:
                    diag_state[slot][i] = u

        value_match = dataset.query_values(outputs[13])

        results = eval_metric(args, model, tokenizer, batch, outputs, 0.5, False, values, value_match)
        all_results.append(results)
        if not no_print:
            predict_and_print(
                args, model, tokenizer, batch, outputs, unique_ids, input_ids, values, inform, ds, value_match
            )
        if not no_output:
            preds, ds = predict_and_format(
                args,
                model,
                tokenizer,
                processor,
                batch,
                outputs,
                unique_ids,
                input_ids,
                values,
                inform,
                ds,
                value_match,
            )
            all_preds.append(preds)

    if not no_output:
        all_preds = [item for sublist in all_preds for item in sublist]  # Flatten list

    # Generate final results
    final_results = {}
    for k in all_results[0].keys():
        final_results[k] = torch.stack([r[k] for r in all_results]).sum() / len(dataset)

    # Write final predictions (for evaluation with external tool)
    output_prediction_file = os.path.join(
        args.output_dir, "pred_res%s.%s.%s.json" % (args.cache_suffix, args.predict_type, prefix)
    )
    if not no_output:
        with open(output_prediction_file, "w") as f:
            json.dump(all_preds, f, indent=2)

    return final_results


def eval_metric(args, model, tokenizer, batch, outputs, threshold=0.0, dae=False, values=None, value_match=None):
    total_loss = outputs[0]
    total_cl_loss = outputs[1]
    total_tk_loss = outputs[2]
    total_tp_loss = outputs[3]
    per_slot_per_example_loss = outputs[4]
    per_slot_per_example_cl_loss = outputs[5]
    per_slot_per_example_tk_loss = outputs[6]
    per_slot_per_example_tp_loss = outputs[7]
    per_slot_class_logits = outputs[8]
    per_slot_start_logits = outputs[9]
    # per_slot_value_logits = outputs[10]
    per_slot_refer_logits = outputs[11]

    class_types = model.class_types

    input_ids = []
    for i in range(len(batch["input_ids"])):
        clipped = batch["input_ids"][i].tolist()
        clipped = clipped[: len(clipped) - clipped[::-1].index(tokenizer.sep_token_id)]
        input_ids.append(clipped)

    metric_dict = {}
    per_slot_correctness = {}
    for slot in model.slot_list:
        per_example_loss = per_slot_per_example_loss[slot]
        per_example_cl_loss = per_slot_per_example_cl_loss[slot]
        per_example_tk_loss = per_slot_per_example_tk_loss[slot]
        per_example_tp_loss = per_slot_per_example_tp_loss[slot]
        class_logits = per_slot_class_logits[slot]
        start_logits = per_slot_start_logits[slot]
        # value_logits = per_slot_value_logits[slot]
        refer_logits = per_slot_refer_logits[slot]

        mean = []
        is_value_match = []
        for i in range(len(batch["input_ids"])):
            mean.append(torch.mean(start_logits[i][: len(input_ids[i])]))
            is_value_match.append(value_match[slot][i][0] == values[i][slot])
        mean = torch.stack(mean)
        is_value_match = torch.tensor(is_value_match)
        norm_logits = torch.clamp(start_logits - mean.unsqueeze(1), min=0) / start_logits.max(1)[0].unsqueeze(1)

        class_label_id = batch["class_label_id"][slot]
        start_pos = batch["start_pos"][slot]
        value_label_id = batch["value_labels"][slot] if slot in batch["value_labels"] else None
        refer_id = batch["refer_id"][slot]

        _, class_prediction = class_logits.max(1)
        class_correctness = torch.eq(class_prediction, class_label_id).float()
        class_accuracy = class_correctness.sum()

        # "is pointable" means whether class label is "copy_value",
        # i.e., that there is a span to be detected.
        token_is_pointable = torch.eq(class_label_id, class_types.index("copy_value")).float()  # TODO: which is better?
        # token_is_pointable = (start_pos.sum(1) > 0).float()
        if dae:
            token_prediction = dilate_and_erode(norm_logits, threshold)
        else:
            token_prediction = norm_logits > threshold
        token_correctness = torch.all(torch.eq(token_prediction, start_pos), 1).float()
        token_accuracy = (token_correctness * token_is_pointable).sum() + (1 - token_is_pointable).sum()

        value_correctness = is_value_match
        value_accuracy = (value_correctness * token_is_pointable).sum() + (1 - token_is_pointable).sum()

        token_is_referrable = torch.eq(
            class_label_id, class_types.index("refer") if "refer" in class_types else -1
        ).float()
        _, refer_prediction = refer_logits.max(1)
        refer_correctness = torch.eq(refer_prediction, refer_id).float()
        refer_accuracy = (refer_correctness * token_is_referrable).sum() + (1 - token_is_referrable).sum()
        # NaNs mean that none of the examples in this batch contain referrals. -> division by 0
        # The accuracy therefore is 1 by default. -> replace NaNs
        # if math.isnan(refer_accuracy) or math.isinf(refer_accuracy):
        #    refer_accuracy = torch.tensor(1.0, device=refer_accuracy.device)

        if args.value_matching_weight > 0.0:
            total_correctness = (
                class_correctness
                * (token_is_pointable * token_correctness + (1 - token_is_pointable))
                * (token_is_pointable * value_correctness + (1 - token_is_pointable))
                * (token_is_referrable * refer_correctness + (1 - token_is_referrable))
            )
        else:
            total_correctness = (
                class_correctness
                * (token_is_pointable * token_correctness + (1 - token_is_pointable))
                * (token_is_referrable * refer_correctness + (1 - token_is_referrable))
            )
        total_accuracy = total_correctness.sum()

        loss = per_example_loss.sum()
        cl_loss = per_example_cl_loss.sum()
        tk_loss = per_example_tk_loss.sum()
        tp_loss = per_example_tp_loss.sum()
        metric_dict["eval_accuracy_class_%s" % slot] = class_accuracy
        metric_dict["eval_accuracy_token_%s" % slot] = token_accuracy
        metric_dict["eval_accuracy_value_%s" % slot] = value_accuracy
        metric_dict["eval_accuracy_refer_%s" % slot] = refer_accuracy
        metric_dict["eval_accuracy_%s" % slot] = total_accuracy
        metric_dict["eval_loss_%s" % slot] = loss
        metric_dict["eval_cl_loss_%s" % slot] = cl_loss
        metric_dict["eval_tk_loss_%s" % slot] = tk_loss
        metric_dict["eval_tp_loss_%s" % slot] = tp_loss
        per_slot_correctness[slot] = total_correctness

    goal_correctness = torch.stack([c for c in per_slot_correctness.values()], 1).prod(1)
    goal_accuracy = goal_correctness.sum()
    metric_dict["eval_accuracy_goal"] = goal_accuracy
    metric_dict["loss"] = total_loss
    metric_dict["cl_loss"] = total_cl_loss
    metric_dict["tk_loss"] = total_tk_loss
    metric_dict["tp_loss"] = total_tp_loss
    return metric_dict


def get_spans(pred, norm_logits, input_tokens, usr_utt_spans):
    span_indices = [i for i in range(len(pred)) if pred[i]]
    prev_si = None
    spans = []
    for si in span_indices:
        if prev_si is None or si - prev_si > 1:
            spans.append(([], [], []))
        spans[-1][0].append(si)
        spans[-1][1].append(input_tokens[si])
        spans[-1][2].append(norm_logits[si])
        prev_si = si
    spans = [(min(i), max(i), " ".join(t for t in s), (sum(c) / len(c)).item()) for (i, s, c) in spans]
    final_spans = {}
    for s in spans:
        for us_itr, us in enumerate(usr_utt_spans):
            if s[0] >= us[0] and s[1] <= us[1]:
                if us_itr not in final_spans:
                    final_spans[us_itr] = []
                final_spans[us_itr].append(s[2:])
                break
    final_spans = list(final_spans.values())
    return final_spans


def get_usr_utt_spans(usr_mask):
    span_indices = [i for i in range(len(usr_mask)) if usr_mask[i]]
    prev_si = None
    spans = []
    for si in span_indices:
        if prev_si is None or si - prev_si > 1:
            spans.append([])
        spans[-1].append(si)
        prev_si = si
    spans = [[min(s), max(s)] for s in spans]
    return spans


def smooth_roberta_predictions(pred, input_tokens, tokenizer):
    smoothed_pred = pred.detach().clone()
    # Forward
    span = False
    i = 0
    while i < len(pred):
        if pred[i] > 0:
            span = True

        elif (
            span
            and input_tokens[i][0] != "\u0120"
            and input_tokens[i]
            not in [
                tokenizer.unk_token,
                tokenizer.bos_token,
                tokenizer.eos_token,
                tokenizer.unk_token,
                tokenizer.sep_token,
                tokenizer.pad_token,
                tokenizer.cls_token,
                tokenizer.mask_token,
            ]
        ):
            smoothed_pred[i] = 1  # use label for in-span tokens
        elif span and (
            input_tokens[i][0] == "\u0120"
            or input_tokens[i]
            in [
                tokenizer.unk_token,
                tokenizer.bos_token,
                tokenizer.eos_token,
                tokenizer.unk_token,
                tokenizer.sep_token,
                tokenizer.pad_token,
                tokenizer.cls_token,
                tokenizer.mask_token,
            ]
        ):
            span = False
        i += 1
    # Backward
    span = False
    i = len(pred) - 1
    while i >= 0:
        if pred[i] > 0:
            span = True
        if (
            span
            and input_tokens[i][0] != "\u0120"
            and input_tokens[i]
            not in [
                tokenizer.unk_token,
                tokenizer.bos_token,
                tokenizer.eos_token,
                tokenizer.unk_token,
                tokenizer.sep_token,
                tokenizer.pad_token,
                tokenizer.cls_token,
                tokenizer.mask_token,
            ]
        ):
            smoothed_pred[i] = 1  # use label for in-span tokens
        elif span and input_tokens[i][0] == "\u0120":
            smoothed_pred[i] = 1  # use label for in-span tokens
            span = False
        i -= 1
    return smoothed_pred


def smooth_bert_predictions(pred, input_tokens, tokenizer):
    smoothed_pred = pred.detach().clone()
    # Forward
    span = False
    i = 0
    while i < len(pred):
        if pred[i] > 0:
            span = True
        elif span and input_tokens[i][0:2] == "##":
            smoothed_pred[i] = 1  # use label for in-span tokens
        else:
            span = False
        i += 1
    # Backward
    span = False
    i = len(pred) - 1
    while i >= 0:
        if pred[i] > 0:
            span = True
        if span and input_tokens[i + 1][0:2] == "##":
            smoothed_pred[i] = 1  # use label for in-span tokens
        else:
            span = False
        i -= 1
    return smoothed_pred


def predict_and_format(
    args,
    model,
    tokenizer,
    processor,
    batch,
    outputs,
    ids,
    input_ids_unmasked,
    values,
    inform,
    ds,
    value_match,
    dae=False,
):
    def _tokenize(text):
        if "\u0120" in text:
            text = re.sub(" ", "", text)
            text = re.sub("\u0120", " ", text)
        else:
            text = re.sub(" ##", "", text)
        text = text.strip()
        return " ".join([tok for tok in map(str.strip, re.split("(\W+)", text)) if len(tok) > 0])

    per_slot_class_logits = outputs[8]
    per_slot_start_logits = outputs[9]
    per_slot_value_logits = outputs[10]
    per_slot_refer_logits = outputs[11]

    class_types = model.class_types

    prediction_list = []
    dialog_state = ds
    for i in range(len(ids)):
        if int(ids[i].split("-")[2]) == 0:
            dialog_state = {slot: "none" for slot in model.slot_list}

        input_tokens = tokenizer.convert_ids_to_tokens(input_ids_unmasked[i])

        prediction = {}
        prediction_addendum = {}

        prediction["guid"] = ids[i].split("-")
        input_ids = batch["input_ids"][i].tolist()
        input_ids = input_ids[: len(input_ids) - input_ids[::-1].index(tokenizer.sep_token_id)]
        prediction["input_ids"] = input_ids

        # assign identified spans to their respective usr turns (simply append spans as list of lists)
        usr_utt_spans = get_usr_utt_spans(batch["usr_mask"][i][1:])

        for slot in model.slot_list:
            class_logits = per_slot_class_logits[slot][i]
            start_logits = per_slot_start_logits[slot][i]
            value_logits = per_slot_value_logits[slot][i] if per_slot_value_logits != {} else None
            refer_logits = per_slot_refer_logits[slot][i]

            weights = start_logits[: len(input_ids)]
            norm_logits = torch.clamp(weights - torch.mean(weights), min=0) / torch.max(weights)

            class_label_id = int(batch["class_label_id"][slot][i])
            start_pos = batch["start_pos"][slot][i].tolist()
            refer_id = int(batch["refer_id"][slot][i])

            class_prediction = int(class_logits.argmax())

            if dae:
                start_prediction = dilate_and_erode(norm_logits.unsqueeze(0), 0.0).squeeze(0)
            else:
                start_prediction = norm_logits > 0.0
            if "roberta" in args.model_type:
                start_prediction = smooth_roberta_predictions(start_prediction, input_tokens, tokenizer)
            else:
                start_prediction = smooth_bert_predictions(start_prediction, input_tokens, tokenizer)
            start_prediction[0] = False  # Ignore <s>

            value_label_id = []
            if slot in batch["value_labels"]:
                value_label_id = batch["value_labels"][slot][i] > 0.0
            if value_logits is not None:
                value_logits /= sum(value_logits)  # Scale
                value_prediction = value_logits >= (1.0 / len(value_logits))  # For attention based value matching
                value_logits = value_logits.tolist()

            refer_prediction = int(refer_logits.argmax())

            prediction["class_prediction_%s" % slot] = class_prediction
            prediction["class_label_id_%s" % slot] = class_label_id
            prediction["start_prediction_%s" % slot] = [
                i for i in range(len(start_prediction)) if start_prediction[i] > 0
            ]
            prediction["start_confidence_%s" % slot] = [
                norm_logits[j].item() for j in range(len(start_prediction)) if start_prediction[j] > 0
            ]
            prediction["start_pos_%s" % slot] = [i for i in range(len(start_pos)) if start_pos[i] > 0]
            prediction["value_label_id_%s" % slot] = [i for i in range(len(value_label_id)) if value_label_id[i] > 0]
            prediction["value_prediction_%s" % slot] = []
            prediction["value_confidence_%s" % slot] = []
            if value_logits is not None:
                prediction["value_prediction_%s" % slot] = [
                    i for i in range(len(value_prediction)) if value_prediction[i] > 0
                ]
                prediction["value_confidence_%s" % slot] = [
                    value_logits[i] for i in range(len(value_logits)) if value_prediction[i] > 0
                ]
            prediction["refer_prediction_%s" % slot] = refer_prediction
            prediction["refer_id_%s" % slot] = refer_id

            if class_prediction == class_types.index("dontcare"):
                dialog_state[slot] = "dontcare"
            elif class_prediction == class_types.index("copy_value"):
                spans = get_spans(start_prediction[1:], norm_logits[1:], input_tokens[1:], usr_utt_spans)
                if len(spans) > 0:
                    for e_itr in range(len(spans)):
                        for ee_itr in range(len(spans[e_itr])):
                            tmp = list(spans[e_itr][ee_itr])
                            tmp[0] = _tokenize(tmp[0])
                            spans[e_itr][ee_itr] = tuple(tmp)
                    dialog_state[slot] = spans
                else:
                    dialog_state[slot] = "none"
            elif "true" in model.class_types and class_prediction == class_types.index("true"):
                dialog_state[slot] = "true"
            elif "false" in model.class_types and class_prediction == class_types.index("false"):
                dialog_state[slot] = "false"
            elif class_prediction == class_types.index("inform"):
                dialog_state[slot] = "§§" + inform[i][slot]  # TODO: implement handling of multiple informed values
            elif "request" in model.class_types and class_prediction == model.class_types.index("request"):
                # Don't carry over requested slots, except of type Boolean
                if hasattr(processor, "boolean") and processor.boolean is not None and slot in processor.boolean:
                    dialog_state[slot] = "true"
            # Referral case is handled below

            prediction_addendum["slot_prediction_%s" % slot] = dialog_state[slot]
            prediction_addendum["slot_groundtruth_%s" % slot] = values[i][slot]
            prediction_addendum["slot_dist_prediction_%s" % slot] = value_match[slot][i][0]
            prediction_addendum["slot_dist_confidence_%s" % slot] = value_match[slot][i][2]
            prediction_addendum["slot_dist_similarity_%s" % slot] = value_match[slot][i][1]
            prediction_addendum["slot_value_prediction_%s" % slot] = ""
            prediction_addendum["slot_value_confidence_%s" % slot] = 1.0
            if len(prediction["value_prediction_%s" % slot]) > 0:
                top_conf = np.argmax(prediction["value_confidence_%s" % slot])
                top_pred = prediction["value_prediction_%s" % slot][top_conf]
                top_val = list(batch["encoded_slot_values"][slot].keys())[top_pred]
                prediction_addendum["slot_value_prediction_%s" % slot] = top_val
                prediction_addendum["slot_value_confidence_%s" % slot] = np.max(
                    prediction["value_confidence_%s" % slot]
                )

        # Referral case. All other slot values need to be seen first in order
        # to be able to do this correctly.
        for slot in model.slot_list:
            class_logits = per_slot_class_logits[slot][i]
            refer_logits = per_slot_refer_logits[slot][i]

            class_prediction = int(class_logits.argmax())
            refer_prediction = int(refer_logits.argmax())

            if "refer" in class_types and class_prediction == class_types.index("refer"):
                # Only slots that have been mentioned before can be referred to.
                # One can think of a situation where one slot is referred to in the same utterance.
                # This phenomenon is however currently not properly covered in the training data
                # label generation process.
                dialog_state[slot] = dialog_state[list(model.slot_list.keys())[refer_prediction]]
                prediction_addendum["slot_prediction_%s" % slot] = dialog_state[slot]  # Value update

        # Normalize value predictions
        for slot in model.slot_list:
            if isinstance(dialog_state[slot], list):
                for e_itr in range(len(dialog_state[slot])):
                    for f_itr in range(len(dialog_state[slot][e_itr])):
                        tmp_state = list(dialog_state[slot][e_itr][f_itr])
                        tmp_state[0] = processor.prediction_normalization(slot, tmp_state[0])
                        dialog_state[slot][e_itr][f_itr] = tuple(tmp_state)
            else:
                dialog_state[slot] = processor.prediction_normalization(slot, dialog_state[slot])
            prediction_addendum["slot_prediction_%s" % slot] = dialog_state[slot]  # Value update

        prediction.update(prediction_addendum)
        prediction_list.append(prediction)

    return prediction_list, dialog_state


def predict_and_print(args, model, tokenizer, batch, outputs, ids, input_ids_unmasked, values, inform, ds, value_match):
    per_slot_class_logits = outputs[8]
    per_slot_start_logits = outputs[9]
    per_slot_att_weights = outputs[12]

    class_types = model.class_types

    for i in range(len(ids)):
        input_tokens = tokenizer.convert_ids_to_tokens(input_ids_unmasked[i])

        input_ids = batch["input_ids"][i].tolist()
        input_ids = input_ids[: len(input_ids) - input_ids[::-1].index(tokenizer.sep_token_id)]

        print(ids[i])

        pos_i = {}
        clb_i = {}
        class_norm_weights = {}
        token_norm_weights = {}
        is_value_match = {}
        for s in model.slot_list:
            pos_i[s] = batch["start_pos"][s][i].tolist()
            clb_i[s] = batch["class_label_id"][s][i]
            if per_slot_att_weights[s] is not None:
                class_weights = per_slot_att_weights[s][i][: len(input_ids)]
                class_norm_weights[s] = torch.clamp(class_weights - torch.mean(class_weights), min=0) / torch.max(
                    class_weights
                )
            token_weights = per_slot_start_logits[s][i][: len(input_ids)]
            token_norm_weights[s] = torch.clamp(token_weights - torch.mean(token_weights), min=0) / torch.max(
                token_weights
            )
            is_value_match[s] = value_match[s][i][0] == values[i][s]
            # Print value matching results
            sorted_dists = value_match[s][i][3]
            print("%20s: %s %s ..." % (s, values[i][s], sorted_dists[:3]))

        print(" ", end="")
        if per_slot_att_weights[s] is not None:
            for s in model.slot_list:
                if clb_i[s] != class_types.index("none"):
                    print("\033[1m%s\033[0m" % (s[0]), end="")
                else:
                    print(s[0], end="")
            print(" | ", end="")
        missed = ""
        no_value_match = ""
        for s in model.slot_list:
            class_prediction = int(per_slot_class_logits[s][i].argmax())
            if clb_i[s] != class_types.index("none") and clb_i[s] != class_prediction:
                missed += "%s: %d -> %d " % (s[0], clb_i[s], class_prediction)
            if clb_i[s] == class_types.index("copy_value"):
                print("\033[1m%s\033[0m " % (s[0]), end="")
            else:
                print(s[0] + " ", end="")
            if clb_i[s] == class_types.index("copy_value") and not is_value_match[s]:
                no_value_match += "%s (%s)" % (s[0], value_match[s][i][0])
        if len(missed) > 0:
            print("| missed: " + missed, end="")
        if len(no_value_match) > 0:
            print("| wrong value match: %s" % no_value_match, end="")
        print()
        for k in range(len(input_ids)):
            bold = False
            print(" ", end="")
            if per_slot_att_weights[s] is not None:
                for s in model.slot_list:
                    c_weight = class_norm_weights[s][k]
                    if c_weight == 0.0:
                        print(" ", end="")
                    elif c_weight < 0.25:
                        print("\u2591", end="")
                    elif c_weight < 0.5:
                        print("\u2592", end="")
                    elif c_weight < 0.75:
                        print("\u2593", end="")
                    else:
                        print("\u2588", end="")
                print(" | ", end="")
            for s in model.slot_list:
                t_weight = token_norm_weights[s][k]
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
                if pos_i[s][k]:
                    bold = True
            if bold:
                print("\033[1m%s\033[0m" % (input_tokens[k]))
            else:
                print("%s" % (input_tokens[k]))
