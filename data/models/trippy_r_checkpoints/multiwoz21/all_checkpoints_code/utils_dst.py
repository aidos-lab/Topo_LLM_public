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
import six
import numpy as np
import os
import re
import pickle
import random
import copy
from scipy.stats import invgauss
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class DSTExample(object):
    """
    A single training/test example for the DST dataset.
    """

    def __init__(self,
                 guid,
                 text_a,
                 text_b,
                 text_a_label=None,
                 text_b_label=None,
                 values=None,
                 inform_label=None,
                 inform_slot_label=None,
                 refer_label=None,
                 diag_state=None,
                 slot_update=None,
                 class_label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_a_label = text_a_label
        self.text_b_label = text_b_label
        self.values = values
        self.inform_label = inform_label
        self.inform_slot_label = inform_slot_label
        self.refer_label = refer_label
        self.diag_state = diag_state
        self.slot_update = slot_update
        self.class_label = class_label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "guid: %s" % (self.guid)
        s += ", text_a: %s" % (self.text_a)
        s += ", text_b: %s" % (self.text_b)
        if self.text_a_label:
            s += ", text_a_label: %s" % (self.text_a_label)
        if self.text_b_label:
            s += ", text_b_label: %s" % (self.text_b_label)
        if self.values:
            s += ", values: %s" % (self.values)
        if self.inform_label:
            s += ", inform_label: %s" % (self.inform_label)
        if self.inform_slot_label:
            s += ", inform_slot_label: %s" % (self.inform_slot_label)
        if self.refer_label:
            s += ", refer_label: %s" % (self.refer_label)
        if self.diag_state:
            s += ", diag_state: %s" % (self.diag_state)
        if self.slot_update:
            s += ", slot_update: %s" % (self.slot_update)
        if self.class_label:
            s += ", class_label: %s" % (self.class_label)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 usr_mask,
                 start_pos=None,
                 values=None,
                 inform=None,
                 inform_slot=None,
                 refer_id=None,
                 diag_state=None,
                 class_label_id=None,
                 hst_boundaries=None,
                 guid="NONE"):
        self.guid = guid
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.usr_mask = usr_mask
        self.start_pos = start_pos
        self.values = values
        self.inform = inform
        self.inform_slot = inform_slot
        self.refer_id = refer_id
        self.diag_state = diag_state
        self.class_label_id = class_label_id
        self.hst_boundaries = hst_boundaries


class TrippyDataset(Dataset):
    def __init__(self, args, examples, model, tokenizer, processor, dset="train", evaluate=False, automatic_labels=None):
        self.args = args
        self.examples = examples
        self.automatic_labels = automatic_labels
        self.model = model
        self.tokenizer = tokenizer
        self.slot_list = model.slot_list
        self.slot_dict = model.slot_list
        self.noncategorical = model.noncategorical
        self.class_list = model.class_types
        self.class_dict = model.class_types
        self.evaluate = evaluate
        self.encoded_slots_pooled = None
        self.encoded_slots_seq = None
        self.encoded_slot_values = None
        self.negative_samples = None
        self.tokenized_sequences_ids = None
        self.dropout_value_seq = None
        self.dropout_value_list = None
        self.dset = dset
        self.mode = "default" # default, proto, tag

        self.label_maps = copy.deepcopy(processor.label_maps)
        self.value_list = copy.deepcopy(processor.value_list['train'])
        if evaluate:
            for s in processor.value_list[dset]:
                for v in processor.value_list[dset][s]:
                    if v not in self.value_list[s]:
                        self.value_list[s][v] = processor.value_list[dset][s][v]
                    else:
                        self.value_list[s][v] += processor.value_list[dset][s][v]

        if examples is None:
            logger.warn("Creating empty dataset. You should load or build features before use.")
            self.features = None
        else:
            self.features = self._convert_examples_to_features(examples=examples,
                                                               slot_list=self.slot_list,
                                                               class_list=self.class_list,
                                                               model_type=self.args.model_type,
                                                               max_seq_length=self.args.max_seq_length,
                                                               automatic_labels=self.automatic_labels)

    def proto(self):
        self.mode = "proto"

    def tag(self):
        self.mode = "tag"

    def reset(self):
        self.mode = "default"

    def update_model(self, model):
        self.model = model
        self.slot_list = model.slot_list
        self.slot_dict = model.slot_list
        self.class_list = model.class_types
        self.class_dict = model.class_types

    def load_features_from_file(self, cached_file):
        logger.info("Loading features from cached file %s", cached_file)
        self.features, self.examples = torch.load(cached_file)
        self.size = len(self.features)
        self._build_dataset()

    def save_features_to_file(self, cached_file):
        logger.info("Saving features into cached file %s", cached_file)
        torch.save((self.features, self.examples), cached_file)

    def build_features_from_examples(self, examples):
        self.examples = examples
        self.features = self._convert_examples_to_features(examples=self.examples,
                                                           slot_list=self.slot_list,
                                                           class_list=self.class_list,
                                                           model_type=self.args.model_type,
                                                           max_seq_length=self.args.max_seq_length,
                                                           automatic_labels=self.automatic_labels)
        self.size = len(self.features)
        self._build_dataset()

    def _build_dataset(self):
        assert self.features is not None
        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in self.features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in self.features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in self.features], dtype=torch.long)
        all_usr_mask = torch.tensor([f.usr_mask for f in self.features], dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        f_start_pos = [f.start_pos for f in self.features]
        f_inform_slot_ids = [f.inform_slot for f in self.features]
        f_refer_ids = [f.refer_id for f in self.features]
        f_diag_state = [f.diag_state for f in self.features]
        f_class_label_ids = [f.class_label_id for f in self.features]
        all_start_positions = {}
        all_inform_slot_ids = {}
        all_refer_ids = {}
        all_diag_state = {}
        all_class_label_ids = {}
        for s in self.slot_list:
            all_start_positions[s] = torch.tensor([f[s] for f in f_start_pos], dtype=torch.long)
            all_inform_slot_ids[s] = torch.tensor([f[s] for f in f_inform_slot_ids], dtype=torch.long)
            all_refer_ids[s] = torch.tensor([f[s] for f in f_refer_ids], dtype=torch.long)
            all_diag_state[s] = torch.tensor([f[s] for f in f_diag_state], dtype=torch.long)
            all_class_label_ids[s] = torch.tensor([f[s] for f in f_class_label_ids], dtype=torch.long)
        data = {'input_ids': all_input_ids, 'input_mask': all_input_mask,
                'segment_ids': all_segment_ids, 'usr_mask': all_usr_mask,
                'start_pos': all_start_positions,
                'inform_slot_id': all_inform_slot_ids, 'refer_id': all_refer_ids,
                'diag_state': all_diag_state if not self.evaluate else {},
                'class_label_id': all_class_label_ids, 'example_id': all_example_index}
        for _, element in data.items():
            if isinstance(element, dict):
                assert all(self.size == tensor.size(0) for name, tensor in element.items()) # dict of tensors
            elif isinstance(element, list):
                assert all(self.size == tensor.size(0) for tensor in element) # list of tensors
            else:
                assert self.size == element.size(0) # tensor
        self.data = data

    def encode_slot_values(self, val_rep_mode="represent", val_rep="full"):
        def get_val_desc(val_rep, slot, value):
            if val_rep == "full":
                text = "%s is %s ." % (self.slot_dict[slot], value)
            elif val_rep == "v":
                text = value
            else:
                logger.error("Unknown val_rep (%s). Aborting." % (val_rep))
                exit(1)
            return text

        # Separate values by slots, because some slots share values
        self.encoded_slot_values = {}
        self.encoded_slot_values_variants = {} # For tagging only. Keep separate to not break rest of code
        self.encoded_dropout_slot_values = {} # For training with token dropout
        for slot in self.slot_dict:
            self.encoded_slot_values[slot] = {}
            self.encoded_slot_values_variants[slot] = {}
            self.encoded_dropout_slot_values[slot] = {}
            for value in self.value_list[slot]:
                # Encode value variants, if existent.
                if value in self.label_maps:
                    for variant in self.label_maps[value]:
                        text = get_val_desc(val_rep, slot, variant)
                        input_ids, input_mask = self._build_input(text)
                        encoded_slot_value, _ = self._encode_text(text, input_ids.unsqueeze(0), input_mask.unsqueeze(0), mode=val_rep_mode)
                        if isinstance(encoded_slot_value, dict):
                            encoded_slot_value = encoded_slot_value[slot] # Keep only slot-specific encoding
                        encoded_slot_value = encoded_slot_value.cpu()
                        self.encoded_slot_values_variants[slot][variant] = encoded_slot_value
                # Encode regular values.
                text = get_val_desc(val_rep, slot, value)
                input_ids, input_mask = self._build_input(text)
                encoded_slot_value, _ = self._encode_text(text, input_ids.unsqueeze(0), input_mask.unsqueeze(0), mode=val_rep_mode)
                if isinstance(encoded_slot_value, dict):
                    encoded_slot_value = encoded_slot_value[slot] # Keep only slot-specific encoding
                encoded_slot_value = encoded_slot_value.cpu()
                self.encoded_slot_values[slot][value] = encoded_slot_value
                # Encode dropped out values.
                if self.dropout_value_list is not None and slot in self.dropout_value_list and value in self.dropout_value_list[slot]:
                    for dropout_value_seq in self.dropout_value_list[slot][value]:
                        v_dropped_out = ''.join(self.tokenizer.convert_ids_to_tokens(dropout_value_seq))
                        if "\u0120" in v_dropped_out:
                            v_dropped_out = re.sub("\u0120", " ", v_dropped_out)
                            v_dropped_out = v_dropped_out.strip()
                        else:
                            v_dropped_out = re.sub("(^| )##", "", v_dropped_out)
                        assert "\u0122" not in value
                        v_tmp = re.sub(self.tokenizer.unk_token, "\u0122", v_dropped_out)
                        v_tmp = re.sub(" ", "", v_tmp)
                        text_dropped_out = get_val_desc(val_rep, slot, v_dropped_out)
                        input_ids_dropped_out, input_mask_dropped_out = self._build_input(text_dropped_out)
                        encoded_slot_value_dropped_out, _ = self._encode_text(text, input_ids.unsqueeze(0), input_mask.unsqueeze(0), mode=val_rep_mode)
                        if isinstance(encoded_slot_value_dropped_out, dict):
                            encoded_slot_value_dropped_out = encoded_slot_value_dropped_out[slot] # Keep only slot-specific encoding
                        encoded_slot_value_dropped_out = encoded_slot_value_dropped_out.cpu()
                        self.encoded_dropout_slot_values[slot][tuple(dropout_value_seq)] = encoded_slot_value_dropped_out
        logger.info("Slot values encoded")

    def save_encoded_slot_values(self, dir_name=""):
        file_name = "encoded_slot_values_%s.pickle" % self.dset
        pickle.dump(self.encoded_slot_values, open(os.path.join(dir_name, file_name), "wb"))
        logger.info("Saved encoded slot values to %s" % os.path.join(dir_name, file_name))

    def load_encoded_slot_values(self, dir_name=""):
        result = False
        file_name = os.path.join(dir_name, "encoded_slot_values_%s.pickle" % self.dset)
        if os.path.exists(file_name):
            result = True
            self.encoded_slot_values = pickle.load(open(file_name, "rb"))
        logger.info("Loaded encoded slot values from %s -> %s" % (file_name, result))
        return result

    def encode_slots(self, train=False):
        self.encoded_slots_pooled = {}
        self.encoded_slots_seq = {}
        self.encoded_slots_ids = {}
        for slot in self.slot_dict:
            text = slot + " . " + self.slot_dict[slot] + " ."
            input_ids, input_mask = self._build_input(text)
            encoded_slot_pooled, encoded_slot_seq = self._encode_text(text,
                                                                      input_ids.unsqueeze(0), input_mask.unsqueeze(0),
                                                                      mode="encode", train=train)
            self.encoded_slots_pooled[slot] = encoded_slot_pooled
            self.encoded_slots_seq[slot] = encoded_slot_seq
            self.encoded_slots_ids[slot] = (input_ids, input_mask)
        logger.info("Slots encoded")

    def save_encoded_slots(self, dir_name=""):
        pickle.dump(self.encoded_slots_pooled, open(os.path.join(dir_name, "encoded_slots_pooled.pickle"), "wb"))
        pickle.dump(self.encoded_slots_seq, open(os.path.join(dir_name, "encoded_slots_seq.pickle"), "wb"))
        pickle.dump(self.encoded_slots_ids, open(os.path.join(dir_name, "encoded_slots_ids.pickle"), "wb"))
        logger.info("Saved encoded slots to %s" % dir_name)

    def load_encoded_slots(self, dir_name=""):
        result = False
        try:
            self.encoded_slots_pooled = pickle.load(open(os.path.join(dir_name, "encoded_slots_pooled.pickle"), "rb"))
            self.encoded_slots_seq = pickle.load(open(os.path.join(dir_name, "encoded_slots_seq.pickle"), "rb"))
            self.encoded_slots_ids = pickle.load(open(os.path.join(dir_name, "encoded_slots_ids.pickle"), "rb"))
            result = True
        except FileNotFoundError:
            logger.warn("Loading encoded slots from %s failed" % dir_name)
        logger.info("Loaded encoded slots from %s" % dir_name)
        return result

    def compute_vectors(self):
        self.model.eval() # No dropout
        if not self.load_encoded_slots(self.args.output_dir):
            self.encode_slots()

    def distance(self, x, y):
        return torch.dist(x, y, p=2) # Euclidean/L2

    def query_values(self, turn_representation):
        def confidence(d, idx):
            d_lol = d[:idx] + d[idx + 1:]
            return max(1 - (d[idx] / ((sum(d_lol) + 1e-8) / (len(d_lol) + 1e-8))), 0)

        result = {}
        for slot in self.slot_list:
            result[slot] = []
            for e in turn_representation[slot]:
                distances = []
                keys = []
                for v in self.value_list[slot]:
                    distances.append(self.distance(self.encoded_slot_values[slot][v], e).item())
                    keys.append(v)
                idx = np.argmin(distances)
                conf = confidence(distances, idx)
                sorted_dists = sorted(zip(keys, distances), key = lambda t: t[1])
                result[slot].append((keys[idx], "%.4f" % distances[idx], "%.4f" % conf, sorted_dists))
        return result

    def tokenize_sequences(self, max_len=1, train=False):
        self.tokenized_sequences_ids = {}
        self.tokenized_sequences_list = []
        self.seqs_per_sample = {}
        for f in tqdm(self.features, desc="Tokenize sequences"):
            seq = f.input_ids[1:f.input_ids.index(self.tokenizer.pad_token_id) if self.tokenizer.pad_token_id in f.input_ids else -1]
            example_id = f.guid
            self.seqs_per_sample[example_id] = []

            # Consider full words for max_len, not just tokens.
            token_seq = self.tokenizer.convert_ids_to_tokens(seq)
            word_list = []
            idx_list = []
            for t_itr, t in enumerate(token_seq):
                if ("roberta" in self.args.model_type and t[0] == "\u0120") or t[0:2] != "##" or \
                    t in [self.tokenizer.unk_token, self.tokenizer.bos_token, self.tokenizer.eos_token,
                          self.tokenizer.sep_token, self.tokenizer.pad_token, self.tokenizer.cls_token,
                          self.tokenizer.mask_token] or t in self.tokenizer.additional_special_tokens:
                    word_list.append([t])
                    idx_list.append([t_itr])
                else:
                    word_list[-1].append(t)
                    idx_list[-1].append(t_itr)

            # Keep list of all sequences in each sample.
            seq_len = len(word_list)
            for start in range(seq_len):
                for offset in range(1, 1 + max_len):
                    if start + offset <= seq_len:
                        subseq = seq[idx_list[start][0]:idx_list[start + offset - 1][-1] + 1]
                        if self.tokenizer.sep_token_id in subseq:
                            continue
                        if tuple(subseq) not in self.tokenized_sequences_ids:
                            input_ids, input_mask = self._build_input(subseq, is_token_ids=True)
                            self.tokenized_sequences_ids[tuple(subseq)] = (input_ids.cpu(), input_mask.cpu())
                        self.seqs_per_sample[example_id].append(subseq)
        self.tokenized_sequences_list = list(self.tokenized_sequences_ids)

    def save_tokenized_sequences(self, dir_name="", overwrite=True):
        file_name = os.path.join(dir_name, "tokenized_sequences_ids_%s.pickle" % self.dset)
        if overwrite or not os.path.exists(file_name):
            pickle.dump(self.tokenized_sequences_ids, open(file_name, "wb"))
            logger.info("Saved tokenized sequences to %s" % dir_name)
        file_name = os.path.join(dir_name, "seqs_per_sample_%s.pickle" % self.dset)
        if overwrite or not os.path.exists(file_name):
            pickle.dump(self.seqs_per_sample, open(file_name, "wb"))

    def load_tokenized_sequences(self, dir_name=""):
        result = False
        file_name = os.path.join(dir_name, "tokenized_sequences_ids_%s.pickle" % self.dset)
        if os.path.exists(file_name):
            result = True
            self.tokenized_sequences_ids = pickle.load(open(file_name, "rb"))
            self.tokenized_sequences_list = list(self.tokenized_sequences_ids)
            logger.info("Loaded tokenized sequences from %s" % dir_name)
        file_name = os.path.join(dir_name, "seqs_per_sample_%s.pickle" % self.dset)
        if os.path.exists(file_name):
            self.seqs_per_sample = pickle.load(open(file_name, "rb"))
        return result

    def update_samples_for_proto(self, max_len=1):
        def list_in_list(a, lst):
            for i in range(len(lst) + 1 - len(a)):
                if lst[i:i + len(a)] == a:
                    return True
            return False

        if self.tokenized_sequences_ids is None:
            logger.warn("Updating negative samples, but values not encoded yet. Encoding now.")
            self.tokenize_sequences(max_len=max_len)
        result = {}
        self.positive_samples_for_proto_pos = {}
        self.positive_samples_for_proto_input_ids = {}
        self.positive_samples_for_proto_input_mask = {}
        self.negative_samples_for_proto_pos = {}
        self.negative_samples_for_proto_input_ids = {}
        self.negative_samples_for_proto_input_mask = {}
        for index in tqdm(range(self.size), desc="Update negative samples for proto training"):
            b_index = index % self.args.per_gpu_train_batch_size
            if b_index == 0:
                offset = min(self.args.per_gpu_train_batch_size, self.size - index + 1)
                for key, element in self.data.items():
                    if isinstance(element, dict):
                        result[key] = {k: v[index:index + offset] for k, v in element.items()}
                    elif isinstance(element, list):
                        result[key] = [v[index:index + offset] for v in element]
                    else:
                        result[key] = element[index:index + offset]

            input_ids = result['input_ids'][b_index].tolist()
            input_ids = input_ids[0:input_ids.index(1) if self.tokenizer.pad_token_id in input_ids else -1]

            # Pick a random sequence in the (entire) input as pos example
            guid = self.features[result['example_id'][b_index]].guid
            seq_list = self.seqs_per_sample[guid]
            seq = random.choice(seq_list)
            seq_len = len(seq)
            seq_tuple = tuple(seq)

            self.positive_samples_for_proto_input_ids[index] = self.tokenized_sequences_ids[seq_tuple][0]
            self.positive_samples_for_proto_input_mask[index] = self.tokenized_sequences_ids[seq_tuple][1]
            # Find all occurrences in input.
            self.positive_samples_for_proto_pos[index] = torch.zeros(self.args.max_seq_length, dtype=torch.long)
            for i in range(len(input_ids) + 1 - seq_len):
                if input_ids[i:i + seq_len] == seq:
                    self.positive_samples_for_proto_pos[index][i:i + seq_len] = 1

            # Pick a random sequence not in any location in the input as neg example
            subseq = random.choice(self.tokenized_sequences_list)
            while list_in_list(list(subseq), input_ids):
                subseq = random.choice(self.tokenized_sequences_list)
            self.negative_samples_for_proto_input_ids[index] = self.tokenized_sequences_ids[subseq][0]
            self.negative_samples_for_proto_input_mask[index] = self.tokenized_sequences_ids[subseq][1]
            self.negative_samples_for_proto_pos[index] = torch.zeros(self.args.max_seq_length, dtype=torch.long)
            if self.args.tag_none_target:
                self.negative_samples_for_proto_pos[index][1] = 1
        assert len(self.positive_samples_for_proto_pos) == self.size
        assert len(self.negative_samples_for_proto_pos) == self.size

    def dropout_input(self):
        if self.evaluate or self.mode != "default" or self.args.svd == 0.0:
            return

        # Preparation.
        self.data['input_ids_dropout'] = copy.deepcopy(self.data['input_ids'])
        joint_text_label = sum(list(self.data['start_pos'].values()))
        joint_text_label_noncat = torch.zeros(joint_text_label.size())
        joint_text_label_cat = torch.zeros(joint_text_label.size())
        for slot in self.slot_list:
            if slot in self.noncategorical:
                joint_text_label_noncat += self.data['start_pos'][slot]
            else:
                joint_text_label_cat += self.data['start_pos'][slot]
        rn = np.random.random_sample(joint_text_label.size())
        if self.args.use_td:
            assert self.args.td_ratio >= 0.0 and self.args.td_ratio <= 1.0
            top_n = int(self.tokenizer.vocab_size * self.args.td_ratio)
        svd_mask = (joint_text_label > 0) * (rn <= self.args.svd)
        svd_mask_noncat = (joint_text_label_noncat > 0) * (rn <= self.args.svd)
        svd_mask_cat = (joint_text_label_cat > 0) * (rn <= self.args.svd)

        self.dropout_value_seq = {slot: {} for slot in self.slot_list}
        self.dropout_value_list = {slot: {} for slot in self.slot_list}
        for i, input_ids_dropout in tqdm(enumerate(self.data['input_ids_dropout']), desc="Dropout inputs"):
            # Slot value dropout.
            if self.args.svd > 0.0:
                indices_to_drop_out = (svd_mask[i] == 1).nonzero(as_tuple=True)[0]
                indices_to_drop_out_noncat = (svd_mask_noncat[i] == 1).nonzero(as_tuple=True)[0]
                indices_to_drop_out_cat = (svd_mask_cat[i] == 1).nonzero(as_tuple=True)[0]
                if self.args.use_td:
                    random_token_id = random.sample(range(top_n), len(indices_to_drop_out_noncat))
                    while self.tokenizer.sep_token_id in random_token_id or \
                          self.tokenizer.pad_token_id in random_token_id or \
                          self.tokenizer.cls_token_id in random_token_id:
                        random_token_id = random.sample(range(top_n), len(indices_to_drop_out_noncat))
                    for k in range(len(indices_to_drop_out_noncat)):
                        input_ids_dropout[indices_to_drop_out_noncat[k]] = random_token_id[k]
                    input_ids_dropout[indices_to_drop_out_cat] = self.tokenizer.unk_token_id
                else:
                    input_ids_dropout[indices_to_drop_out] = self.tokenizer.unk_token_id

            # Remember dropped-out values.
            example_id = self.data['example_id'][i]
            for slot in self.slot_list:
                orig_value = self.features[example_id].values[slot]
                if orig_value not in self.dropout_value_list[slot]:
                    self.dropout_value_list[slot][orig_value] = []
                if not self.args.svd_for_all_slots and slot not in self.noncategorical:
                    continue
                value_indices = (self.data['start_pos'][slot][i] > 0).nonzero(as_tuple=True)[0]
                prev_si = None
                spans = []
                for si in value_indices:
                    if prev_si is None or si - prev_si > 1:
                        spans.append([])
                    spans[-1].append(si)
                    prev_si = si
                for s_itr in range(len(spans)):
                    spans[s_itr] = torch.stack(spans[s_itr])
                # In case of length variations, revert dropout (this might however never happen).
                # Else, make sure that all mentions of the same value are identically dropped out.
                if len(spans) > 1:
                    is_ambiguous = False
                    for span in spans[1:]:
                        if len(span) != len(spans[0]):
                            is_ambiguous = True
                            break
                    if is_ambiguous:
                        self.data['input_ids_dropout'][i][value_indices] = self.data['input_ids'][i][value_indices]
                    else:
                        for span in spans[1:]:
                            self.data['input_ids_dropout'][i][span] = self.data['input_ids_dropout'][i][spans[0]]
                # We only need to check if spans[0] differs from the original seqs, since all s in spans are identical now.
                if len(spans) > 0 and not torch.equal(self.data['input_ids'][i][spans[0]], self.data['input_ids_dropout'][i][spans[0]]):
                    self.dropout_value_seq[slot][i] = self.data['input_ids_dropout'][i][spans[0]].tolist()
                    if self.dropout_value_seq[slot][i] not in self.dropout_value_list[slot][orig_value]:
                        self.dropout_value_list[slot][orig_value].append(self.dropout_value_seq[slot][i])

    def __getitem__(self, index):
        result = {}
        # Static elements. Copy, because they will be modified below.
        for key, element in self.data.items():
            if isinstance(element, dict):
                result[key] = {k: v[index].detach().clone() for k, v in element.items()}
            elif isinstance(element, list):
                result[key] = [v[index].detach().clone() for v in element]
            else:
                result[key] = element[index].detach().clone()

        # For dropout, simply use pre-processed input_ids.
        if not self.evaluate and self.mode == "default" and self.args.svd > 0.0:
            result['input_ids'] = result['input_ids_dropout']

        # Dynamic elements.
        result['dropout_value_feat'] = {}
        result['value_labels'] = {}

        if self.mode == "proto":
            assert self.positive_samples_for_proto_pos is not None
            rn = random.random()
            if self.args.tag_none_target and rn <= self.args.proto_neg_sample_ratio:
                result['start_pos'] = self.negative_samples_for_proto_pos[index]
                result['slot_ids'] = self.negative_samples_for_proto_input_ids[index]
                result['slot_mask'] = self.negative_samples_for_proto_input_mask[index]
            else:
                result['start_pos'] = self.positive_samples_for_proto_pos[index]
                result['slot_ids'] = self.positive_samples_for_proto_input_ids[index]
                result['slot_mask'] = self.positive_samples_for_proto_input_mask[index]
        elif self.mode == "tag":
            value_reps = {}
            for slot in self.slot_list:
                value_name = self.features[result['example_id']].values[slot]
                if value_name not in self.encoded_slot_values[slot]:
                    value_rep = torch.zeros((1, self.model.config.hidden_size), dtype=torch.float)
                else:
                    value_rep = self.encoded_slot_values[slot][value_name]
                value_reps[slot] = value_rep
            result['value_reps'] = value_reps
        else:
            # History dropout
            if not self.evaluate and (self.args.hd > 0.0):
                hst_boundaries = self.features[result['example_id']].hst_boundaries
                if len(hst_boundaries) > 0:
                    rn = random.random()
                    if rn <= self.args.hd:
                        hst_dropout_idx = random.randint(0, len(hst_boundaries) - 1)
                        hst_dropout_start = hst_boundaries[hst_dropout_idx][0]
                        hst_dropout_end = hst_boundaries[-1][1]
                        result['input_ids'][hst_dropout_start] = result['input_ids'][hst_dropout_end]
                        result['input_ids'][hst_dropout_start + 1:hst_dropout_end + 1] = self.tokenizer.pad_token_id
                        result['input_mask'][hst_dropout_start + 1:hst_dropout_end + 1] = 0
                        result['segment_ids'][hst_dropout_start + 1:hst_dropout_end + 1] = 0
                        result['usr_mask'][hst_dropout_start:hst_dropout_end + 1] = 0
                        for slot in self.slot_list:
                            result['start_pos'][slot][hst_dropout_start + 1:hst_dropout_end + 1] = 0
            # Labels
            for slot in self.slot_list:
                token_is_pointable = result['start_pos'][slot].sum() > 0
                # If no sequence is present, attention should be on <none>
                if self.args.tag_none_target and not token_is_pointable:
                    result['start_pos'][slot][1] = 1
                pos_value = self.features[index].values[slot]
                # For value matching: Only the correct value has a weight, all (!) others automatically become negative samples.
                # TODO: Test subsampling negative samples.

                # For attention based value matching
                result['value_labels'][slot] = torch.zeros((len(self.encoded_slot_values[slot])), dtype=torch.float)
                result['dropout_value_feat'][slot] = torch.zeros((1, self.model.config.hidden_size), dtype=torch.float)
                # Only train value matching, if value is extractable
                if token_is_pointable and pos_value in self.encoded_slot_values[slot]:
                    result['value_labels'][slot][list(self.encoded_slot_values[slot].keys()).index(pos_value)] = 1.0
                    # In case of dropout, forward new representation as target for value matching instead.
                    if self.dropout_value_seq is not None:
                        if result['example_id'].item() in self.dropout_value_seq[slot]:
                            dropout_value_seq = tuple(self.dropout_value_seq[slot][result['example_id'].item()])
                            result['dropout_value_feat'][slot] = self.encoded_dropout_slot_values[slot][dropout_value_seq]
        return result

    def _encode_text(self, text, input_ids, input_mask, mode="represent", train=False):
        batch = {
            "input_ids": input_ids.to(self.args.device),
            "input_mask": input_mask.to(self.args.device),
            "encoded_slots_pooled": self.encoded_slots_pooled.copy() if self.encoded_slots_pooled is not None else None,
            "encoded_slots_seq": self.encoded_slots_seq.copy() if self.encoded_slots_seq is not None else None,
        }
        if train:
            self.model.train()
            encoded_text_pooled, encoded_text_seq, weights = self.model(batch, mode=mode)
            self.model.eval()
        else:
            self.model.eval()
            with torch.no_grad():
                encoded_text_pooled, encoded_text_seq, weights = self.model(batch, mode=mode)
        return encoded_text_pooled, encoded_text_seq

    def __len__(self):
        return self.size

    def _build_input(self, text, is_token_ids=False):
        if not is_token_ids:
            if "roberta" in self.args.model_type:
                tokens = self.tokenizer.tokenize(convert_to_unicode(' ' + text))
            else:
                tokens = self.tokenizer.tokenize(convert_to_unicode(text))
            input_id = self.tokenizer.convert_tokens_to_ids([self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token])
        else:
            input_id = [self.tokenizer.cls_token_id] + text + [self.tokenizer.sep_token_id]
        input_mask = [1] * len(input_id)
        while len(input_id) < self.args.max_seq_length:
            input_id.append(self.tokenizer.pad_token_id)
            input_mask.append(0)
        assert len(input_id) == self.args.max_seq_length
        return torch.tensor(input_id), torch.tensor(input_mask)

    def _convert_examples_to_features(self, examples, slot_list, class_list, model_type,
                                      max_seq_length, automatic_labels=None):
        """Loads a data file into a list of `InputBatch`s."""

        if model_type == 'roberta':
            model_specs = {'MODEL_TYPE': 'roberta',
                           'TOKEN_CORRECTION': 6}
        else:
            model_specs = {'MODEL_TYPE': 'bert',
                           'TOKEN_CORRECTION': 4}

        def _tokenize_text(text, text_label_dict, model_specs):
            token_to_subtoken = []
            tokens = []
            for token in text:
                token = convert_to_unicode(token)
                if model_specs['MODEL_TYPE'] == 'roberta':
                    # It seems the behaviour of the tokenizer changed in newer versions,
                    # which makes this case handling necessary.
                    if token != self.tokenizer.unk_token:
                        token = ' ' + token
                sub_tokens = self.tokenizer.tokenize(token) # Most time intensive step
                token_to_subtoken.append([token, sub_tokens])
                tokens.extend(sub_tokens)
            return tokens, token_to_subtoken

        def _label_tokenized_text(tokens, text_label_dict, slot):
            token_labels = []
            for element, token_label in zip(tokens, text_label_dict[slot]):
                token, sub_tokens = element
                token_labels.extend([token_label for _ in sub_tokens])
            return token_labels
        
        def _truncate_seq_pair(tokens_a, tokens_b, history, max_length):
            """Truncates a sequence pair in place to the maximum length.
            Copied from bert/run_classifier.py
            """
            # This is a simple heuristic which will always truncate the longer sequence
            # one token at a time. This makes more sense than truncating an equal percent
            # of tokens from each, since if one sequence is very short then each token
            # that's truncated likely contains more information than a longer sequence.
            while True:
                history_len = 0
                for hst in history:
                    for spk in hst:
                        history_len += len(spk)
                total_length = len(tokens_a) + len(tokens_b) + history_len
                if total_length <= max_length:
                    break
                if len(history) > 0:
                    history.pop() # Remove one entire turn from the history
                elif len(tokens_a) > len(tokens_b):
                    tokens_a.pop()
                else:
                    tokens_b.pop()

        def _truncate_length_and_warn(tokens_a, tokens_b, history, max_seq_length, model_specs, guid):
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP], [SEP] with "- 4" (BERT)
            # Account for <s>, </s></s>, </s></s>, </s> with "- 6" (RoBERTa)
            # Account for </s> after each history utterance (all models)
            history_len = 0
            for hst in history:
                for spk in hst:
                    history_len += len(spk)
            max_len = max_seq_length - model_specs['TOKEN_CORRECTION'] - len(history) * 2 - self.args.tag_none_target * int(model_specs['TOKEN_CORRECTION'] / 2)
            if len(tokens_a) + len(tokens_b) + history_len > max_len:
                logger.info("Truncate Example %s. Total len=%d." % (guid, len(tokens_a) + len(tokens_b) + history_len))
                input_text_too_long = True
            else:
                input_text_too_long = False
            _truncate_seq_pair(tokens_a, tokens_b, history, max_len)
            return input_text_too_long

        def _get_token_label_ids(token_labels_a, token_labels_b, token_labels_history, max_seq_length, model_specs):
            token_label_ids = {slot: [] for slot in token_labels_a}
            for slot in token_label_ids:
                if self.args.tag_none_target:
                    if model_specs['MODEL_TYPE'] == 'roberta':
                        labels = [0] + [0, 0, 0] + token_labels_a[slot] + [0] # <s> <none> </s> </s> ... </s>
                    else:
                        labels = [0] + [0, 0] + token_labels_a[slot] + [0] # [CLS] [NONE] [SEP] ... [SEP]
                else:
                    labels = [0] + token_labels_a[slot] + [0] # [CLS]/<s> ... [SEP]/</s>
                if model_specs['MODEL_TYPE'] == 'roberta':
                    labels.append(0) # </s>
                labels += token_labels_b[slot] + [0] # ... [SEP]/</s>
                if model_specs['MODEL_TYPE'] == 'roberta':
                    labels.append(0) # </s>
                token_label_ids[slot] = labels

            for hst in token_labels_history:
                (utt_a, utt_b) = hst
                for slot in token_label_ids:
                    token_label_ids[slot] += utt_a[slot] + [0] + utt_b[slot] + [0] # [SEP]/</s>

            for slot in token_label_ids:
                if len(token_label_ids[slot]) < max_seq_length:
                    token_label_ids[slot] += (max_seq_length - len(token_label_ids[slot])) * [0]

            return token_label_ids

        def _get_transformer_input(tokens_a, tokens_b, history, max_seq_length, model_specs):
            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids: 0     0   0   0  0     0 0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            cls = self.tokenizer.cls_token
            sep = self.tokenizer.sep_token
            if model_specs['MODEL_TYPE'] == 'roberta':
                if self.args.tag_none_target:
                    tokens = [cls] + ['<none>', sep, sep] + tokens_a + [sep] + [sep] + tokens_b + [sep] + [sep]
                    segment_ids = [0] + [0, 0, 0] + len(tokens_a) * [0] + 2 * [0] + len(tokens_b) * [0] + 2 * [0]
                    usr_mask = [0] + [1, 0, 0] + len(tokens_a) * [0 if self.args.swap_utterances else 1] + 2 * [0] + len(tokens_b) * [1 if self.args.swap_utterances else 0] + 2 * [0]
                else:
                    tokens = [cls] + tokens_a + [sep] + [sep] + tokens_b + [sep] + [sep]
                    segment_ids = [0] + len(tokens_a) * [0] + 2 * [0] + len(tokens_b) * [0] + 2 * [0]
                    usr_mask = [0] + len(tokens_a) * [0 if self.args.swap_utterances else 1] + 2 * [0] + len(tokens_b) * [1 if self.args.swap_utterances else 0] + 2 * [0]
            else:
                if self.args.tag_none_target:
                    tokens = [cls] + ['[NONE]', sep] + tokens_a + [sep] + tokens_b + [sep]
                    segment_ids = [0] + [0, 0] + len(tokens_a) * [0] + [0] + len(tokens_b) * [1] + [1]
                    usr_mask = [0] + [1, 0] + len(tokens_a) * [0 if self.args.swap_utterances else 1] + [0] + len(tokens_b) * [1 if self.args.swap_utterances else 0] + [0]
                else:
                    tokens = [cls] + tokens_a + [sep] + tokens_b + [sep]
                    segment_ids = [0] + len(tokens_a) * [0] + [0] + len(tokens_b) * [1] + [1]
                    usr_mask = [0] + len(tokens_a) * [0 if self.args.swap_utterances else 1] + [0] + len(tokens_b) * [1 if self.args.swap_utterances else 0] + [0]
            hst_boundaries = []
            for hst_itr in range(len(history)):
                hst_a, hst_b = history[hst_itr]
                hst_start = len(tokens)
                tokens += hst_a + [sep] + hst_b + [sep]
                hst_end = len(tokens)
                hst_boundaries.append([hst_start, hst_end])
                if model_specs['MODEL_TYPE'] == 'roberta':
                    segment_ids += [0] * (len(hst_a) + 1 + len(hst_b) + 1)
                else:
                    segment_ids += [1] * (len(hst_a) + 1 + len(hst_b) + 1)
                usr_mask += len(hst_a) * [0 if self.args.swap_utterances else 1] + [0] + len(hst_b) * [1 if self.args.swap_utterances else 0] + [0]
            tokens.append(sep)
            if model_specs['MODEL_TYPE'] == 'roberta':
                segment_ids.append(0)
            else:
                segment_ids.append(1)
            usr_mask.append(0)
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)
            # Zero-pad up to the sequence length.
            if len(input_ids) < max_seq_length:
                len_diff = max_seq_length - len(input_ids)
                input_ids += len_diff * [self.tokenizer.pad_token_id]
                input_mask += len_diff * [0]
                segment_ids += len_diff * [0]
                usr_mask += len_diff * [0]
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(usr_mask) == max_seq_length
            return tokens, input_ids, input_mask, segment_ids, usr_mask, hst_boundaries

        if automatic_labels is not None:
            logger.warning("USING AUTOMATIC LABELS TO REPLACE GROUNDTRUTH! BE SURE YOU KNOW WHAT YOU ARE DOING!")

        total_cnt = 0
        too_long_cnt = 0

        refer_list = list(slot_list.keys()) + ['none']

        # Tokenize turns
        tokens_dict = {}
        for (example_index, example) in enumerate(examples):
            if example_index % 1000 == 0:
                logger.info("Tokenizing turn %d of %d" % (example_index, len(examples)))
            re_guid = re.match("(.*)-([0-9]+)", example.guid)
            re_guid_diag = re_guid[1]
            re_guid_turn = int(re_guid[2])

            tokens_a, token_to_subtoken_a = _tokenize_text(
                example.text_a, example.text_a_label, model_specs)
            tokens_b, token_to_subtoken_b = _tokenize_text(
                example.text_b, example.text_b_label, model_specs)

            token_labels_a_dict = {}
            token_labels_b_dict = {}
            for slot in slot_list:
                token_labels_a_dict[slot] = _label_tokenized_text(token_to_subtoken_a, example.text_a_label, slot)
                token_labels_b_dict[slot] = _label_tokenized_text(token_to_subtoken_b, example.text_b_label, slot)

            # Use automatic labels (if provided)
            if automatic_labels is not None:
                for slot in slot_list:
                    # Case where <none> target was used during pre-training/tagging
                    if self.args.use_none_target_tags:
                        if model_specs['MODEL_TYPE'] == 'roberta':
                            a_start = 4
                        else:
                            a_start = 3
                    else:
                        a_start = 1
                    auto_lbl = automatic_labels[slot][example_index]
                    a_end = a_start + len(token_labels_a_dict[slot])
                    token_labels_a_dict[slot] = auto_lbl[a_start:a_end].int().tolist()
                    if model_specs['MODEL_TYPE'] == 'roberta':
                        b_start = a_end + 2
                    else:
                        b_start = a_end + 1
                    b_end = b_start + len(token_labels_b_dict[slot])
                    token_labels_b_dict[slot] = auto_lbl[b_start:b_end].int().tolist()

            tokens_dict[(re_guid_diag, re_guid_turn)] = [[tokens_a, token_labels_a_dict], [tokens_b, token_labels_b_dict]]

        # Build single example
        features = []
        for (example_index, example) in enumerate(examples):
            if example_index % 1000 == 0:
                logger.info("Writing example %d of %d" % (example_index, len(examples)))

            total_cnt += 1

            # Gather history
            re_guid = re.match("(.*)-([0-9]+)", example.guid)
            diag_id = re_guid[1]
            turn_id = int(re_guid[2])
            tokens_a, token_labels_a_dict = tokens_dict[(diag_id, turn_id)][0]
            tokens_b, token_labels_b_dict = tokens_dict[(diag_id, turn_id)][1]
            tokens_history = []
            token_labels_history_dict = []
            if not self.args.no_append_history:
                for hst_itr in range(turn_id - 1, -1, -1):
                    tokens_history.append([])
                    token_labels_history_dict.append([])
                    for spk_itr in range(len(tokens_dict[(diag_id, hst_itr)])):
                        tokens_h, token_labels_h_dict = tokens_dict[(diag_id, hst_itr)][spk_itr]
                        tokens_history[-1].append(tokens_h)
                        token_labels_history_dict[-1].append(token_labels_h_dict)
                for slot in slot_list:
                    if self.args.no_use_history_labels or example.slot_update[slot]:
                        for h in token_labels_history_dict:
                            for s in h:
                                s[slot] = len(s[slot]) * [0]

            input_text_too_long = _truncate_length_and_warn(
                tokens_a, tokens_b, tokens_history, max_seq_length, model_specs, example.guid)

            if input_text_too_long:
                too_long_cnt += 1

            tokens, input_ids, input_mask, segment_ids, usr_mask, hst_boundaries = _get_transformer_input(tokens_a,
                                                                                                          tokens_b,
                                                                                                          tokens_history,
                                                                                                          max_seq_length,
                                                                                                          model_specs)

            for slot in slot_list:
                token_labels_a_dict[slot] = token_labels_a_dict[slot][:len(tokens_a)]
                token_labels_b_dict[slot] = token_labels_b_dict[slot][:len(tokens_b)]
            token_labels_history_dict = token_labels_history_dict[:len(tokens_history)]

            token_label_ids = _get_token_label_ids(token_labels_a_dict,
                                                   token_labels_b_dict,
                                                   token_labels_history_dict,
                                                   max_seq_length,
                                                   model_specs)

            value_dict = {}
            inform_dict = {}
            inform_slot_dict = {}
            refer_id_dict = {}
            diag_state_dict = {}
            class_label_id_dict = {}
            start_pos_dict = {}
            for slot in slot_list:
                assert len(token_label_ids[slot]) == len(input_ids)

                value_dict[slot] = example.values[slot]
                inform_dict[slot] = example.inform_label[slot]

                start_pos_dict[slot] = token_label_ids[slot]

                inform_slot_dict[slot] = example.inform_slot_label[slot]
                refer_id_dict[slot] = refer_list.index(example.refer_label[slot])
                diag_state_dict[slot] = class_list.index(example.diag_state[slot])
                class_label_id_dict[slot] = class_list.index(example.class_label[slot])

            if example_index < 10:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("tokens: %s" % " ".join(tokens))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logger.info("usr_mask: %s" % " ".join([str(x) for x in usr_mask]))
                logger.info("start_pos: %s" % str(start_pos_dict))
                logger.info("values: %s" % str(value_dict))
                logger.info("inform: %s" % str(inform_dict))
                logger.info("inform_slot: %s" % str(inform_slot_dict))
                logger.info("refer_id: %s" % str(refer_id_dict))
                logger.info("diag_state: %s" % str(diag_state_dict))
                logger.info("class_label_id: %s" % str(class_label_id_dict))
                logger.info("hst_boundaries: %s" % " ".join([str(x) for x in hst_boundaries]))

            features.append(
                InputFeatures(
                    guid=example.guid,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    usr_mask=usr_mask,
                    start_pos=start_pos_dict,
                    values=value_dict,
                    inform=inform_dict,
                    inform_slot=inform_slot_dict,
                    refer_id=refer_id_dict,
                    diag_state=diag_state_dict,
                    class_label_id=class_label_id_dict,
                    hst_boundaries=hst_boundaries))

        logger.info("========== %d out of %d examples have text too long" % (too_long_cnt, total_cnt))

        return features


# From bert.tokenization (TF code)
def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")
