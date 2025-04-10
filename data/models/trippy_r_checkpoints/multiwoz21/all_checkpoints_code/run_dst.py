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

import argparse
import glob
import json
import logging
import os
import pickle

import torch
from data_processors import PROCESSORS
from dst_enums import LrSchedulerType
from dst_proto import evaluate_proto, train_proto
from dst_tag import tag_values
from dst_train import evaluate, train
from modeling_dst import TransformerForDST
from transformers import (
    BertConfig,
    BertTokenizer,
    ElectraConfig,
    ElectraTokenizer,
    RobertaConfig,
    RobertaTokenizer,
)
from utils_run import load_and_cache_examples, print_header, set_seed

# Note:
# - We are removing the import of WEIGHTS_NAME from transformers and are setting it manually here.
# - "pytorch_model.bin" was used in older versions of transformers.
# > WEIGHTS_NAME = "pytorch_model.bin"
# Note: "model.safetensors" is used in our current setup.
WEIGHTS_NAME = "model.safetensors"

logger = logging.getLogger(__name__)


class BertForDST(TransformerForDST("bert")):
    pass


class RobertaForDST(TransformerForDST("roberta")):
    pass


class ElectraForDST(TransformerForDST("electra")):
    pass


MODEL_CLASSES = {
    "bert": (BertConfig, BertForDST, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForDST, RobertaTokenizer),
    "electra": (ElectraConfig, ElectraForDST, ElectraTokenizer),
}


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--task_name", default=None, type=str, required=True, help="Name of the task (e.g., multiwoz21)."
    )
    parser.add_argument("--data_dir", default=None, type=str, required=True, help="Task database.")
    parser.add_argument("--dataset_config", default=None, type=str, required=True, help="Dataset configuration file.")
    parser.add_argument(
        "--predict_type",
        default=None,
        type=str,
        required=True,
        help="Portion of the data to perform prediction on (e.g., dev, test).",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints and predictions will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name", default="", type=str, help="Pretrained tokenizer name or path if not the same as model_name"
    )

    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="Maximum input length after tokenization. Longer sequences will be truncated, shorter ones padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the <predict_type> set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Rul evaluation during training at each logging step."
    )
    parser.add_argument("--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.")

    parser.add_argument("--dropout_rate", default=0.3, type=float, help="Dropout rate for transformer representations.")
    parser.add_argument(
        "--class_loss_ratio",
        default=0.8,
        type=float,
        help="The ratio applied on class loss in total loss calculation. "
        "Should be a value in [0.0, 1.0]. "
        "The ratio applied on token loss is (1-class_loss_ratio)/2. "
        "The ratio applied on refer loss is (1-class_loss_ratio)/2.",
    )
    parser.add_argument(
        "--proto_loss_function",
        type=str,
        default="mse",
        help="Loss function for proto DST training (mse|ce). Default 'mse'",
    )
    parser.add_argument(
        "--token_loss_function",
        type=str,
        default="mse",
        help="Loss function for sequence tagging (mse|ce). Default 'mse'",
    )
    parser.add_argument(
        "--value_loss_function",
        type=str,
        default="mse",
        help="Loss function for value matching (mse|ce). Default 'mse'",
    )
    parser.add_argument("--slot_attention_heads", type=int, default=8, help="Number of heads in multihead attention")

    parser.add_argument(
        "--no_append_history", action="store_true", help="Do not append the dialog history to each turn."
    )
    parser.add_argument("--no_use_history_labels", action="store_true", help="Do not label the history as well.")
    parser.add_argument(
        "--no_label_value_repetitions", action="store_true", help="Do not label values that have been mentioned before."
    )
    parser.add_argument(
        "--swap_utterances", action="store_true", help="Swap the turn utterances (default: usr|sys, swapped: sys|usr)."
    )
    parser.add_argument("--delexicalize_sys_utts", action="store_true", help="Delexicalize the system utterances.")
    parser.add_argument(
        "--none_weight", type=float, default=1.0, help="Weight for the none class of the slot gates (default: 1.0)"
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=LrSchedulerType,
        default=LrSchedulerType.LINEAR_SCHEDULE_WITH_WARMUP,
        help="The scheduler type to use.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Overwrites num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_proportion", default=0.0, type=float, help="Linear warmup over warmup_proportion * steps."
    )
    parser.add_argument(
        "--patience", type=int, default=-1, help="Patience for early stopping. When -1, no patience is used"
    )
    parser.add_argument(
        "--early_stop_criterion", type=str, default="goal", help="Early stopping criterion (goal|loss). Default 'goal'"
    )

    parser.add_argument("--logging_steps", type=int, default=10, help="Log every X updates steps.")
    parser.add_argument(
        "--save_steps", type=int, default=0, help="Save checkpoint every X updates steps. Overwritten by --save_epochs."
    )
    parser.add_argument(
        "--save_epochs", type=int, default=0, help="Save checkpoint every X epochs. Overwrites --save_steps."
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--no_cache", action="store_true", help="Don't use cached training and evaluation sets")
    parser.add_argument(
        "--cache_suffix", default="", type=str, help="Optionally add a suffix to the cache files (use trailing _)."
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--fp16", action="store_true", help="Whether to use 16-bit (mixed) precision instead of 32-bit")
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Whether to only load local model files (useful when working offline).",
    )

    parser.add_argument(
        "--training_phase",
        type=int,
        default=-1,
        help="-1: regular training, 0: proto training, 1: tagging, 2: spanless training",
    )

    # Slot value dropout, token noising, history dropout
    parser.add_argument("--svd", default=0.0, type=float, help="Slot value dropout ratio (default: 0.0)")
    parser.add_argument(
        "--use_td",
        action="store_true",
        help="Do slot value dropout using random tokens, i.e., token noising. Requires --svd",
    )
    parser.add_argument(
        "--td_ratio",
        type=float,
        default=1.0,
        help="Fraction of token vocabulary to draw replacements from for token noising. Requires --use_td",
    )
    parser.add_argument(
        "--svd_for_all_slots",
        action="store_true",
        help="By default, SVD/TD is used for noncategorical slots only. Set to use SVD/TD for categorical slots as well",
    )
    parser.add_argument("--hd", type=float, default=0.0, help="History dropout ratio")

    # Spanless training
    parser.add_argument(
        "--tag_none_target", action="store_true", help="Use <none>/[NONE] as target when tagging negative samples"
    )
    parser.add_argument(
        "--use_none_target_tags", action="store_true", help="Use <none>/[NONE] as target during spanless training"
    )
    parser.add_argument(
        "--rand_seq_max_len", type=int, default=4, help="Maximum length of random sequences for proto DST training"
    )
    parser.add_argument(
        "--proto_neg_sample_ratio",
        type=float,
        default=0.1,
        help="Negative sample ratio for proto DST training. Requires --tag_none_target",
    )

    # Value matching
    parser.add_argument(
        "--value_matching_weight",
        type=float,
        default=0.0,
        help="Value matching weight. When 0.0, value matching is not used",
    )

    # DEBUG
    parser.add_argument(
        "--max_slots", type=int, default=-1, help="Maximum number of slots to add to a batch. -1 means all."
    )

    args = parser.parse_args()

    assert args.warmup_proportion >= 0.0 and args.warmup_proportion <= 1.0
    assert args.svd >= 0.0 and args.svd <= 1.0
    assert args.hd >= 0.0 and args.hd <= 1.0
    assert args.td_ratio >= 0.0 and args.td_ratio <= 1.0
    assert args.proto_neg_sample_ratio >= 0.0 and args.proto_neg_sample_ratio <= 1.0
    assert args.training_phase in [-1, 0, 1, 2]
    assert not args.tag_none_target or args.training_phase in [0, 1]
    assert not args.use_none_target_tags or args.training_phase == 2

    task_name = args.task_name.lower()
    if task_name not in PROCESSORS:
        raise ValueError("Task not found: %s" % (task_name))

    processor = PROCESSORS[task_name](
        args.dataset_config, args.data_dir, "train" if not args.do_eval else args.predict_type
    )
    slot_list = processor.slot_list
    noncategorical = processor.noncategorical
    class_types = processor.class_types
    class_labels = len(class_types)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path, local_files_only=args.local_files_only
    )

    # Add DST specific parameters to config
    config.max_seq_length = args.max_seq_length
    config.dropout_rate = args.dropout_rate
    config.class_loss_ratio = args.class_loss_ratio
    config.slot_list = slot_list
    config.noncategorical = noncategorical
    config.class_types = class_types
    config.class_labels = class_labels
    config.tag_none_target = args.tag_none_target
    config.value_matching_weight = args.value_matching_weight
    config.none_weight = args.none_weight
    config.proto_loss_function = args.proto_loss_function
    config.token_loss_function = args.token_loss_function
    config.value_loss_function = args.value_loss_function
    config.slot_attention_heads = args.slot_attention_heads

    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        local_files_only=args.local_files_only,
    )
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        local_files_only=args.local_files_only,
    )

    if args.tag_none_target:
        if args.model_type == "roberta":
            tokenizer.add_special_tokens({"additional_special_tokens": ["<none>"]})
        else:
            tokenizer.add_special_tokens({"additional_special_tokens": ["[NONE]"]})
        model.resize_token_embeddings(len(tokenizer))
        config.vocab_size = len(tokenizer)

    logger.info("Updated model config: %s" % config)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        global_step = 0
        proto_checkpoints = []
        checkpoints = []
        if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
            proto_checkpoints = list(
                os.path.dirname(c)
                for c in sorted(glob.glob(args.output_dir + "/proto_checkpoint*/" + WEIGHTS_NAME, recursive=True))
            )
            checkpoints = list(
                os.path.dirname(c)
                for c in sorted(glob.glob(args.output_dir + "/checkpoint*/" + WEIGHTS_NAME, recursive=True))
            )

        if args.training_phase in [-1, 0, 1]:
            train_dataset = load_and_cache_examples(args, model, tokenizer, processor, dset="train", evaluate=False)
            dev_dataset = None
            if args.evaluate_during_training:
                dev_dataset = load_and_cache_examples(
                    args, model, tokenizer, processor, dset=args.predict_type, evaluate=True
                )

        # Step 1: Pretrain attention layer for random sequence tagging.
        if args.training_phase == 0:
            if len(proto_checkpoints) == 0:
                global_step, tr_loss = train_proto(args, train_dataset, dev_dataset, model, tokenizer, processor)
                logger.info(" global_step = %s, average loss = %s (proto)", global_step, tr_loss)
            else:
                logger.warning(" Preconditions for proto training not fulfilled! Skipping.")

        # Step 2: Get labels for slot values.
        if args.training_phase == 1:
            if len(checkpoints) == 0 and len(proto_checkpoints) > 0:
                # Load correct proto checkpoint, otherwise last model state is used, which is not desired.
                proto_checkpoint = proto_checkpoints[-1]
                model = model_class.from_pretrained(proto_checkpoint)
                model.to(args.device)
                train_dataset.update_model(model)
                if dev_dataset is not None:
                    dev_dataset.update_model(model)
                max_tag_goal = 0.0
                max_tag_thresh = 0.0
                max_dae = True  # default should be true
                if not os.path.exists(os.path.join(args.output_dir, "tagging_threshold.txt")):
                    for tagging_threshold in [0.2, 0.3, 0.4]:
                        max_dae = True  # default should be true
                        for dae in [True]:
                            file_name = os.path.join(
                                args.output_dir, "automatic_labels_%s_%s.pickle" % (tagging_threshold, dae)
                            )
                            automatic_labels, tag_eval = tag_values(
                                args,
                                train_dataset,
                                model,
                                tokenizer,
                                processor,
                                no_print=(tagging_threshold > 0.0 or dae is True),
                                prefix=global_step,
                                threshold=tagging_threshold,
                                dae=dae,
                            )
                            logger.info("tagging_threshold: %s, dae: %s %s" % (tagging_threshold, dae, tag_eval))
                            tag_goal = tag_eval["eval_accuracy_goal"]
                            pickle.dump(automatic_labels, open(file_name, "wb"))
                            if tag_goal > max_tag_goal:
                                max_tag_goal = tag_goal
                                max_tag_thresh = tagging_threshold
                                max_dae = dae
                    with open(os.path.join(args.output_dir, "tagging_threshold.txt"), "w") as f:
                        f.write("%f %d" % (max_tag_thresh, max_dae))
            else:
                logger.warning(" Preconditions for tagging not fulfilled! Skipping.")

        # Step 3: Train full model.
        if args.training_phase == 2:
            if len(checkpoints) == 0 and os.path.exists(os.path.join(args.output_dir, "tagging_threshold.txt")):
                with open(os.path.join(args.output_dir, "tagging_threshold.txt"), "r") as f:
                    max_tag_thresh, max_dae = f.readline().split()
                max_tag_thresh = float(max_tag_thresh)
                max_dae = bool(int(max_dae))
                al_file_name = os.path.join(
                    args.output_dir, "automatic_labels_%s_%s.pickle" % (max_tag_thresh, max_dae)
                )
                logger.info("Loading automatic labels: %s" % (al_file_name))
                automatic_labels = pickle.load(open(al_file_name, "rb"))
                train_dataset = load_and_cache_examples(
                    args, model, tokenizer, processor, dset="train", evaluate=False, automatic_labels=automatic_labels
                )
                dev_dataset = None
                if args.do_train and args.evaluate_during_training:
                    dev_dataset = load_and_cache_examples(
                        args, model, tokenizer, processor, dset=args.predict_type, evaluate=True
                    )
                train_dataset.compute_vectors()
                if dev_dataset is not None:
                    dev_dataset.compute_vectors()
                global_step, tr_loss = train(
                    args, train_dataset, dev_dataset, automatic_labels, model, tokenizer, processor
                )
                logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
            else:
                logger.warning(" Preconditions for spanless training not fulfilled! Skipping.")

        # Train full model with original training.
        if args.training_phase == -1:
            # If output files already exists, assume to continue training from latest checkpoint (unless overwrite_output_dir is set)
            continue_from_global_step = 0
            if len(checkpoints) > 0:
                with open(os.path.join(args.output_dir, "last_checkpoint.txt"), "r") as f:
                    continue_from_global_step = int((f.readline()).split("-")[-1])
                checkpoint = os.path.join(args.output_dir, "checkpoint-%s" % continue_from_global_step)
                logger.warning(" Resuming training from the latest checkpoint: %s", checkpoint)
                model = model_class.from_pretrained(checkpoint)
                model.to(args.device)
                train_dataset.update_model(model)
                if dev_dataset is not None:
                    dev_dataset.update_model(model)
            train_dataset.compute_vectors()
            if dev_dataset is not None:
                dev_dataset.compute_vectors()
            global_step, tr_loss = train(
                args,
                train_dataset,
                dev_dataset,
                None,
                model,
                tokenizer,
                processor,
                # NOTE: We needed to comment out the `continue_from_global_step` argument from the `train` function call,
                # because the train function is not compatible with it.
                # > continue_from_global_step,
            )
            logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Save the trained model and the tokenizer
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        if args.save_steps == 0 and args.save_epochs == 0 and args.patience < 0:
            logger.info("Saving model checkpoint to %s", args.output_dir)
            # Save a trained model, configuration and tokenizer using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = []
    if args.do_eval and args.local_rank in [-1, 0]:
        dataset = load_and_cache_examples(
            args,
            model,
            tokenizer,
            processor,
            dset=args.predict_type,
            evaluate=True,
        )
        dataset.compute_vectors()

        output_eval_file = os.path.join(args.output_dir, "eval_res%s.%s.json" % (args.cache_suffix, args.predict_type))
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            model_file_pattern_to_look_for = args.output_dir + "/**/" + WEIGHTS_NAME

            logger.info(
                msg=f"Selected to run evaluation on all checkpoints. "
                f"Looking for pattern {model_file_pattern_to_look_for = }",
            )
            checkpoints = [
                os.path.dirname(c) for c in sorted(glob.glob(model_file_pattern_to_look_for, recursive=True))
            ]
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for cItr, checkpoint in enumerate(checkpoints):
            # Reload the model
            checkpoint_name = checkpoint.split("/")[-1]
            model = model_class.from_pretrained(checkpoint)
            model.slot_list = slot_list  # Necessary for slot independent DST, as slots might differ during evaluation
            model.noncategorical = noncategorical
            model.to(args.device)
            dataset.update_model(model)

            # Evaluate
            if "proto" in checkpoint_name:
                result = evaluate_proto(args, dataset, model, tokenizer, processor, prefix=checkpoint_name)
            else:
                result = evaluate(args, dataset, model, tokenizer, processor, no_print=True, prefix=checkpoint_name)

            result_dict = {k: float(v) for k, v in result.items()}
            result_dict["checkpoint_name"] = checkpoint_name
            results.append(result_dict)

            for key in sorted(result_dict.keys()):
                logger.info("%s = %s", key, str(result_dict[key]))

        with open(output_eval_file, "w") as f:
            json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    main()
