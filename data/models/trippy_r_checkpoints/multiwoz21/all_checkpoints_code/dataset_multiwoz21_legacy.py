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
from tqdm import tqdm

from utils_dst import (DSTExample)

from dataset_multiwoz21 import (ACTS_DICT,
                                tokenize, normalize_label,
                                get_turn_label, delex_utt)


# Loads the dialogue_acts.json and returns a list
# of slot-value pairs.
def load_acts(input_file, boolean_slots=True):
    with open(input_file) as f:
        acts = json.load(f)
    s_dict = {}
    for d in acts:
        for t in acts[d]:
            # Only process, if turn has annotation
            if isinstance(acts[d][t], dict):
                is_22_format = False
                if 'dialog_act' in acts[d][t]:
                    is_22_format = True
                    acts_list = acts[d][t]['dialog_act']
                    if int(t) % 2 == 0:
                        continue
                else:
                    acts_list = acts[d][t]
                for a in acts_list:
                    aa = a.lower().split('-')
                    if aa[1] in ['inform', 'recommend', 'select', 'book']:
                        for i in acts_list[a]:
                            s = i[0].lower()
                            v = i[1].lower().strip()
                            if s == 'none' or v == '?':
                                continue
                            if v == 'none':
                                if s in ['parking', 'internet']:
                                    if boolean_slots:
                                        v = 'true'
                                    else:
                                        v = s
                                else:
                                    continue
                            if d == 'hotel' and s == 'type':
                                if v in ['hotel', 'hotels']:
                                    if boolean_slots:
                                        v = 'true'
                                    else:
                                        v = 'hotel'
                                else:
                                    if boolean_slots:
                                        v = 'false'
                                    else:
                                        v = 'guest house'
                            slot = aa[0] + '-' + s
                            if slot in ACTS_DICT:
                                slot = ACTS_DICT[slot]
                            if is_22_format:
                                t_key = str(int(int(t) / 2 + 1))
                                d_key = d
                            else:
                                t_key = t
                                d_key = d + '.json'
                            key = d_key, t_key, slot
                            # INFO: Since the model has no mechanism to predict
                            # one among several informed value candidates, we
                            # keep only one informed value. For fairness, we
                            # apply a global rule:
                            # ... Option 1: Keep first informed value
                            if key not in s_dict:
                                s_dict[key] = list([v])
                            # ... Option 2: Keep last informed value
                            #s_dict[key] = list([v])
    return s_dict


def create_examples(input_file, acts_file, set_type, slot_list,
                    label_maps={}, not_domain=None,
                    no_label_value_repetitions=False,
                    swap_utterances=False,
                    delexicalize_sys_utts=False,
                    unk_token="[UNK]",
                    boolean_slots=True,
                    analyze=False):
    """Read a DST json file into a list of DSTExample."""

    sys_inform_dict = load_acts(acts_file, boolean_slots)

    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)

    examples = []
    for d_itr, dialog_id in enumerate(tqdm(input_data)):
        entry = input_data[dialog_id]
        goal = entry['goal']
        utterances = entry['log']

        # If a leave-one-out domain is provided, skip any dialogue containing this domain.
        if not_domain is not None:
            if not_domain in goal and goal[not_domain] != {}:
                continue

        # Collects all slot changes throughout the dialog
        cumulative_labels = {slot: 'none' for slot in slot_list}

        # First system utterance is empty, since multiwoz starts with user input
        utt_tok_list = [[]]
        mod_slots_list = [{}]

        # Collect all utterances and their metadata
        usr_sys_switch = True
        turn_itr = 0
        for utt in utterances:
            # Assert that system and user utterances alternate
            is_sys_utt = utt['metadata'] != {}
            if usr_sys_switch == is_sys_utt:
                print("WARN: Wrong order of system and user utterances. Skipping rest of dialog %s" % (dialog_id))
                break
            usr_sys_switch = is_sys_utt

            if is_sys_utt:
                turn_itr += 1

            # Delexicalize sys utterance
            if delexicalize_sys_utts and is_sys_utt:
                inform_dict = {slot: 'none' for slot in slot_list}
                for slot in slot_list:
                    if (str(dialog_id), str(turn_itr), slot) in sys_inform_dict:
                        inform_dict[slot] = sys_inform_dict[(str(dialog_id), str(turn_itr), slot)]
                utt_tok_list.append(delex_utt(utt['text'], inform_dict, unk_token)) # normalizes utterances
            else:
                utt_tok_list.append(tokenize(utt['text'])) # normalizes utterances

            modified_slots = {}

            # If sys utt, extract metadata (identify and collect modified slots)
            if is_sys_utt:
                for d in utt['metadata']:
                    booked = utt['metadata'][d]['book']['booked']
                    booked_slots = {}
                    # Check the booked section
                    if booked != []:
                        for s in booked[0]:
                            booked_slots[s] = normalize_label('%s-%s' % (d, s), booked[0][s], boolean_slots) # normalize labels
                    # Check the semi and the inform slots
                    for category in ['book', 'semi']:
                        for s in utt['metadata'][d][category]:
                            cs = '%s-book_%s' % (d, s) if category == 'book' else '%s-%s' % (d, s)
                            value_label = normalize_label(cs, utt['metadata'][d][category][s], boolean_slots) # normalize labels
                            # Prefer the slot value as stored in the booked section
                            if s in booked_slots:
                                value_label = booked_slots[s]
                            # Remember modified slots and entire dialog state
                            if cs in slot_list and cumulative_labels[cs] != value_label:
                                modified_slots[cs] = value_label
                                cumulative_labels[cs] = value_label

            mod_slots_list.append(modified_slots.copy())

        # Form proper (usr, sys) turns
        turn_itr = 0
        diag_seen_slots_dict = {}
        diag_seen_slots_value_dict = {slot: 'none' for slot in slot_list}
        diag_state = {slot: 'none' for slot in slot_list}
        sys_utt_tok = []
        usr_utt_tok = []
        for i in range(1, len(utt_tok_list) - 1, 2):
            sys_utt_tok_label_dict = {}
            usr_utt_tok_label_dict = {}
            value_dict = {}
            inform_dict = {}
            inform_slot_dict = {}
            referral_dict = {}
            class_type_dict = {}
            updated_slots = {slot: 0 for slot in slot_list}

            # Collect turn data
            sys_utt_tok = utt_tok_list[i - 1]
            usr_utt_tok = utt_tok_list[i]
            turn_slots = mod_slots_list[i + 1]

            guid = '%s-%s-%s' % (set_type, str(dialog_id), str(turn_itr))

            if analyze:
                print("%15s %2s %s ||| %s" % (dialog_id, turn_itr, ' '.join(sys_utt_tok), ' '.join(usr_utt_tok)))
                print("%15s %2s [" % (dialog_id, turn_itr), end='')

            new_diag_state = diag_state.copy()

            for slot in slot_list:
                value_label = 'none'
                if slot in turn_slots:
                    value_label = turn_slots[slot]
                    # We keep the original labels so as to not
                    # overlook unpointable values, as well as to not
                    # modify any of the original labels for test sets,
                    # since this would make comparison difficult.
                    value_dict[slot] = value_label
                elif not no_label_value_repetitions and slot in diag_seen_slots_dict:
                    value_label = diag_seen_slots_value_dict[slot]

                # Get dialog act annotations
                inform_label = list(['none'])
                inform_slot_dict[slot] = 0
                if (str(dialog_id), str(turn_itr), slot) in sys_inform_dict:
                    inform_label = list([normalize_label(slot, i, boolean_slots) for i in sys_inform_dict[(str(dialog_id), str(turn_itr), slot)]])
                    inform_slot_dict[slot] = 1
                elif (str(dialog_id), str(turn_itr), 'booking-' + slot.split('-')[1]) in sys_inform_dict:
                    inform_label = list([normalize_label(slot, i, boolean_slots) for i in sys_inform_dict[(str(dialog_id), str(turn_itr), 'booking-' + slot.split('-')[1])]])
                    inform_slot_dict[slot] = 1

                (informed_value,
                 referred_slot,
                 usr_utt_tok_label,
                 class_type) = get_turn_label(value_label,
                                              inform_label,
                                              sys_utt_tok,
                                              usr_utt_tok,
                                              slot,
                                              diag_seen_slots_value_dict,
                                              slot_last_occurrence=True,
                                              label_maps=label_maps)

                inform_dict[slot] = informed_value

                # Generally don't use span prediction on sys utterance (but inform prediction instead).
                sys_utt_tok_label = [0 for _ in sys_utt_tok]

                # Determine what to do with value repetitions.
                # If value is unique in seen slots, then tag it, otherwise not,
                # since correct slot assignment can not be guaranteed anymore.
                if not no_label_value_repetitions and slot in diag_seen_slots_dict:
                    if class_type == 'copy_value' and list(diag_seen_slots_value_dict.values()).count(value_label) > 1:
                        class_type = 'none'
                        usr_utt_tok_label = [0 for _ in usr_utt_tok_label]

                sys_utt_tok_label_dict[slot] = sys_utt_tok_label
                usr_utt_tok_label_dict[slot] = usr_utt_tok_label

                if diag_seen_slots_value_dict[slot] != value_label:
                    updated_slots[slot] = 1

                # For now, we map all occurences of unpointable slot values
                # to none. However, since the labels will still suggest
                # a presence of unpointable slot values, the task of the
                # DST is still to find those values. It is just not
                # possible to do that via span prediction on the current input.
                if class_type == 'unpointable':
                    class_type_dict[slot] = 'none'
                    referral_dict[slot] = 'none'
                    if analyze:
                        if slot not in diag_seen_slots_dict or value_label != diag_seen_slots_value_dict[slot]:
                            print("(%s): %s, " % (slot, value_label), end='')
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
                    referral_dict[slot] = referred_slot
                # Remember that this slot was mentioned during this dialog already.
                if class_type != 'none':
                    diag_seen_slots_dict[slot] = class_type
                    diag_seen_slots_value_dict[slot] = value_label
                    new_diag_state[slot] = class_type
                    # Unpointable is not a valid class, therefore replace with
                    # some valid class for now...
                    if class_type == 'unpointable':
                        new_diag_state[slot] = 'copy_value'

            if analyze:
                print("]")

            skip_example = False
            for slot in diag_seen_slots_value_dict:
                slot_domain, slot_name = slot.split("-")
                if slot_domain == not_domain and diag_seen_slots_value_dict[slot] != "none":
                    skip_example = True

            if not swap_utterances:
                txt_a = usr_utt_tok
                txt_b = sys_utt_tok
                txt_a_lbl = usr_utt_tok_label_dict
                txt_b_lbl = sys_utt_tok_label_dict
            else:
                txt_a = sys_utt_tok
                txt_b = usr_utt_tok
                txt_a_lbl = sys_utt_tok_label_dict
                txt_b_lbl = usr_utt_tok_label_dict
            if not skip_example:
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

            turn_itr += 1

        if analyze:
            print("----------------------------------------------------------------------")

    return examples


def get_value_list(input_file, slot_list, boolean_slots=True, not_domain=None):
    exclude = ['none', 'dontcare']
    if not boolean_slots:
        exclude += ['true', 'false']
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)
    value_list = {slot: {} for slot in slot_list}
    for dialog_id in input_data:
        goal = input_data[dialog_id]['goal']
        # If a leave-one-out domain is provided, skip any dialogue containing this domain.
        if not_domain is not None:
            if not_domain in goal and goal[not_domain] != {}:
                continue
        for utt in input_data[dialog_id]['log']:
            if utt['metadata'] != {}:
                for d in utt['metadata']:
                    booked = utt['metadata'][d]['book']['booked']
                    booked_slots = {}
                    # Check the booked section
                    if booked != []:
                        for s in booked[0]:
                            booked_slots[s] = normalize_label('%s-%s' % (d, s), booked[0][s], boolean_slots)
                    # Check the semi and the inform slots
                    for category in ['book', 'semi']:
                        for s in utt['metadata'][d][category]:
                            cs = '%s-book_%s' % (d, s) if category == 'book' else '%s-%s' % (d, s)
                            value_label = normalize_label(cs, utt['metadata'][d][category][s], boolean_slots)
                            # Prefer the slot value as stored in the booked section
                            if s in booked_slots:
                                value_label = booked_slots[s]
                            if cs in slot_list and value_label not in exclude:
                                cs_domain, _ = cs.split("-")
                                if cs_domain == not_domain:
                                    continue
                                if "|" not in value_label and "<" not in value_label and ">" not in value_label:
                                    if value_label not in value_list[cs]:
                                        value_list[cs][value_label] = 0
                                    value_list[cs][value_label] += 1
    return value_list
