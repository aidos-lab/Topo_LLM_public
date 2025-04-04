# coding=utf-8
#
# Copyright 2020-2022 Heinrich Heine University Duesseldorf
#
# Part of this code is based on the source code of BERT-DST
# (arXiv:1907.03040)
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
import re

from utils_dst import (DSTExample, convert_to_unicode)


LABEL_MAPS = {} # Loaded from file
LABEL_FIX = {'areas': 'area', 'phone number': 'number', 'price range': 'price_range', 'center': 'centre', 'east side': 'east', 'corsican': 'corsica'}


def delex_utt(utt, values, unk_token="[UNK]"):
    utt_norm = utt.copy()
    for s, v in values.items():
        if v != 'none':
            v_norm = tokenize(v)
            v_len = len(v_norm)
            for i in range(len(utt_norm) + 1 - v_len):
                if utt_norm[i:i + v_len] == v_norm:
                    utt_norm[i:i + v_len] = [unk_token] * v_len
    return utt_norm


def get_token_pos(tok_list, label):
    find_pos = []
    found = False
    label_list = [item for item in map(str.strip, re.split("(\W+)", label)) if len(item) > 0]
    len_label = len(label_list)
    for i in range(len(tok_list) + 1 - len_label):
        if tok_list[i:i + len_label] == label_list:
            find_pos.append((i, i + len_label))  # start, exclusive_end
            found = True
    return found, find_pos


def check_label_existence(label, usr_utt_tok, sys_utt_tok):
    in_usr, usr_pos = get_token_pos(usr_utt_tok, label)
    if not in_usr and label in LABEL_MAPS:
        for tmp_label in LABEL_MAPS[label]:
            in_usr, usr_pos = get_token_pos(usr_utt_tok, tmp_label)
            if in_usr:
                break
    in_sys, sys_pos = get_token_pos(sys_utt_tok, label)
    if not in_sys and label in LABEL_MAPS:
        for tmp_label in LABEL_MAPS[label]:
            in_sys, sys_pos = get_token_pos(sys_utt_tok, tmp_label)
            if in_sys:
                break
    return in_usr, usr_pos, in_sys, sys_pos


def get_turn_label(label, sys_utt_tok, usr_utt_tok, slot_last_occurrence):
    usr_utt_tok_label = [0 for _ in usr_utt_tok]
    in_sys = False
    if label == 'none' or label == 'dontcare':
        class_type = label
    else:
        in_usr, usr_pos, in_sys, _ = check_label_existence(label, usr_utt_tok, sys_utt_tok)
        if in_usr:
            class_type = 'copy_value'
            if slot_last_occurrence:
                (s, e) = usr_pos[-1]
                for i in range(s, e):
                    usr_utt_tok_label[i] = 1
            else:
                for (s, e) in usr_pos:
                    for i in range(s, e):
                        usr_utt_tok_label[i] = 1
        elif in_sys:
            class_type = 'inform'
        else:
            class_type = 'unpointable'
    return usr_utt_tok_label, class_type, in_sys


def tokenize(utt):
    utt_lower = convert_to_unicode(utt).lower()
    utt_tok = utt_to_token(utt_lower)
    return utt_tok


def utt_to_token(utt):
    return [tok for tok in map(lambda x: re.sub(" ", "", x), re.split("(\W+)", utt)) if len(tok) > 0]


def create_examples(input_file, set_type, slot_list,
                    label_maps={},
                    asr=False,
                    no_label_value_repetitions=False,
                    swap_utterances=False,
                    delexicalize_sys_utts=False,
                    unk_token="[UNK]",
                    boolean_slots=True,
                    analyze=False):
    """Read a DST json file into a list of DSTExample."""
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)

    global LABEL_MAPS
    LABEL_MAPS = label_maps

    examples = []
    for entry in input_data:
        diag_seen_slots_dict = {}
        diag_seen_slots_value_dict = {slot: 'none' for slot in slot_list}
        diag_state = {slot: 'none' for slot in slot_list}
        sys_utt_tok = []
        sys_utt_tok_delex = []
        usr_utt_tok = []
        for turn in entry['dialogue']:
            sys_utt_tok_label_dict = {}
            usr_utt_tok_label_dict = {}
            inform_dict = {slot: 'none' for slot in slot_list}
            inform_slot_dict = {slot: 0 for slot in slot_list}
            referral_dict = {}
            class_type_dict = {}
            updated_slots = {slot: 0 for slot in slot_list}

            sys_utt_tok = tokenize(turn['system_transcript'])
            if asr:
                # The model always expects a non-empty user input.
                # This can not be guaranteed for ASR hypotheses.
                # In case of an empty hypo, use a generic hello.
                for asr_hypo in turn['asr']:
                    usr_utt_tok = tokenize(asr_hypo[0])
                    if len(usr_utt_tok) > 0:
                        break
                if len(usr_utt_tok) == 0:
                    usr_utt_tok = ['hello']
            else:
                usr_utt_tok = tokenize(turn['transcript'])
            turn_label = {LABEL_FIX.get(s.strip(), s.strip()): LABEL_FIX.get(v.strip(), v.strip()) for s, v in turn['turn_label']}

            guid = '%s-%s-%s' % (set_type, str(entry['dialogue_idx']), str(turn['turn_idx']))

            # Create delexicalized sys utterances.
            if delexicalize_sys_utts:
                delex_dict = {}
                for slot in slot_list:
                    delex_dict[slot] = 'none'
                    label = 'none'
                    if slot in turn_label:
                        label = turn_label[slot]
                    elif not no_label_value_repetitions and slot in diag_seen_slots_dict:
                        label = diag_seen_slots_value_dict[slot]
                    if label != 'none' and label != 'dontcare':
                        _, _, in_sys, _ = check_label_existence(label, usr_utt_tok, sys_utt_tok)
                        if in_sys:
                            delex_dict[slot] = label
                sys_utt_tok_delex = delex_utt(sys_utt_tok, delex_dict, unk_token)

            new_diag_state = diag_state.copy()
            for slot in slot_list:
                label = 'none'
                if slot in turn_label:
                    label = turn_label[slot]
                elif not no_label_value_repetitions and slot in diag_seen_slots_dict:
                    label = diag_seen_slots_value_dict[slot]

                (usr_utt_tok_label,
                 class_type,
                 is_informed) = get_turn_label(label,
                                               sys_utt_tok,
                                               usr_utt_tok,
                                               slot_last_occurrence=True)

                if class_type == 'inform':
                    inform_dict[slot] = label
                if is_informed and label != 'none':
                    inform_slot_dict[slot] = 1

                referral_dict[slot] = 'none' # Referral is not present in woz2 data

                # Generally don't use span prediction on sys utterance (but inform prediction instead).
                if delexicalize_sys_utts:
                    sys_utt_tok_label = [0 for _ in sys_utt_tok_delex]
                else:
                    sys_utt_tok_label = [0 for _ in sys_utt_tok]

                # Determine what to do with value repetitions.
                # If value is unique in seen slots, then tag it, otherwise not,
                # since correct slot assignment can not be guaranteed anymore.
                if not no_label_value_repetitions and slot in diag_seen_slots_dict:
                    if class_type == 'copy_value' and list(diag_seen_slots_value_dict.values()).count(label) > 1:
                        class_type = 'none'
                        usr_utt_tok_label = [0 for _ in usr_utt_tok_label]

                sys_utt_tok_label_dict[slot] = sys_utt_tok_label
                usr_utt_tok_label_dict[slot] = usr_utt_tok_label

                if diag_seen_slots_value_dict[slot] != label:
                    updated_slots[slot] = 1

                # For now, we map all occurences of unpointable slot values
                # to none. However, since the labels will still suggest
                # a presence of unpointable slot values, the task of the
                # DST is still to find those values. It is just not
                # possible to do that via span prediction on the current input.
                if class_type == 'unpointable':
                    class_type_dict[slot] = 'none'
                elif slot in diag_seen_slots_dict and class_type == diag_seen_slots_dict[slot] and class_type != 'copy_value' and class_type != 'inform':
                    # If slot has seen before and its class type did not change, label this slot a not present,
                    # assuming that the slot has not actually been mentioned in this turn.
                    # Exceptions are copy_value and inform. If a seen slot has been tagged as copy_value or inform,
                    # this must mean there is evidence in the original labels, therefore consider
                    # them as mentioned again.
                    class_type_dict[slot] = 'none'
                    referral_dict[slot] = 'none'
                else:
                    class_type_dict[slot] = class_type
                # Remember that this slot was mentioned during this dialog already.
                if class_type != 'none':
                    diag_seen_slots_dict[slot] = class_type
                    diag_seen_slots_value_dict[slot] = label
                    new_diag_state[slot] = class_type
                    # Unpointable is not a valid class, therefore replace with
                    # some valid class for now...
                    if class_type == 'unpointable':
                        new_diag_state[slot] = 'copy_value'

            if not swap_utterances:
                txt_a = usr_utt_tok
                if delexicalize_sys_utts:
                    txt_b = sys_utt_tok_delex
                else:
                    txt_b = sys_utt_tok
                txt_a_lbl = usr_utt_tok_label_dict
                txt_b_lbl = sys_utt_tok_label_dict
            else:
                if delexicalize_sys_utts:
                    txt_a = sys_utt_tok_delex
                else:
                    txt_a = sys_utt_tok
                txt_b = usr_utt_tok
                txt_a_lbl = sys_utt_tok_label_dict
                txt_b_lbl = usr_utt_tok_label_dict
            examples.append(DSTExample(
                guid=guid,
                text_a=txt_a,
                text_b=txt_b,
                text_a_label=txt_a_lbl,
                text_b_label=txt_b_lbl,
                values=diag_seen_slots_value_dict.copy(),
                inform_label=inform_dict,
                inform_slot_label=inform_slot_dict,
                refer_label=referral_dict,
                diag_state=diag_state,
                slot_update=updated_slots,
                class_label=class_type_dict))

            # Update some variables.
            diag_state = new_diag_state.copy()

    return examples


def get_value_list(input_file, slot_list):
    exclude = ['none', 'dontcare']
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)
    value_list = {slot: {} for slot in slot_list}
    for entry in input_data:
        for turn in entry['dialogue']:
            turn_label = {LABEL_FIX.get(s.strip(), s.strip()): LABEL_FIX.get(v.strip(), v.strip()) for s, v in turn['turn_label']}
            for slot in turn_label:
                if slot in slot_list and turn_label[slot] not in exclude:
                    if turn_label[slot] not in value_list[slot]:
                        value_list[slot][turn_label[slot]] = 0
                    value_list[slot][turn_label[slot]] += 1
    return value_list
