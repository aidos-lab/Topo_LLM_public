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
import sys

from tqdm import tqdm
from utils_dst import DSTExample

# Append path to ConvLab3 to sys.path
sys.path.append(
    "$CONVLAB3_REPOSITORY_BASE_PATH",
)

try:
    from convlab.util import load_dataset, load_dst_data, load_ontology
except ModuleNotFoundError as e:
    print(e)
    print("Ignore this error if you don't intend to use the data processor for ConvLab3's unified data format.")
    print("Otherwise, make sure you have ConvLab3 installed and added to your PYTHONPATH.")


def get_ontology_slots(ontology):
    domains = [domain for domain in ontology["domains"]]
    ontology_slots = dict()
    for domain in domains:
        if domain not in ontology_slots:
            ontology_slots[domain] = dict()
        for slot in ontology["domains"][domain]["slots"]:
            ontology_slots[domain][slot] = ontology["domains"][domain]["slots"][slot]["description"]
    return ontology_slots


def get_slot_list(dataset_name):
    slot_list = {}
    ontology = load_ontology(dataset_name)
    dataset_slot_list = get_ontology_slots(ontology)
    for domain in dataset_slot_list:
        for slot in dataset_slot_list[domain]:
            slot_list["%s-%s" % (domain, slot)] = dataset_slot_list[domain][slot]
        slot_list["%s-none" % (domain)] = "the topic is %s" % (domain)
    # Some special intents are modeled as 'request' slots in TripPy
    if "bye" in ontology["intents"]:
        slot_list["general-bye"] = ontology["intents"]["bye"]["description"]
    if "thank" in ontology["intents"]:
        slot_list["general-thank"] = ontology["intents"]["thank"]["description"]
    if "greet" in ontology["intents"]:
        slot_list["general-greet"] = ontology["intents"]["greet"]["description"]
    return slot_list


def get_noncategorical(dataset_name):
    noncategorical_list = []
    ontology = load_ontology(dataset_name)
    for d in ontology["domains"]:
        for s in ontology["domains"][d]["slots"]:
            if not ontology["domains"][d]["slots"][s]["is_categorical"]:
                slot = "%s-%s" % (d, s)
                noncategorical_list.append(slot)
    return noncategorical_list


def get_boolean(dataset_name):
    boolean_list = []
    ontology = load_ontology(dataset_name)
    for d in ontology["domains"]:
        for s in ontology["domains"][d]["slots"]:
            if ontology["domains"][d]["slots"][s]["is_categorical"]:
                values = ontology["domains"][d]["slots"][s]["possible_values"]
                if set(values) == set(["True", "False"]):
                    slot = "%s-%s" % (d, s)
                    boolean_list.append(slot)
    return boolean_list


def get_value_list(dataset_name, slot_list):
    value_list = {slot: {} for slot in slot_list}
    ontology = load_ontology(dataset_name)
    for slot in slot_list:
        d, s = slot.split("-")
        if d in ontology["domains"]:
            if s in ontology["domains"][d]["slots"]:
                if ontology["domains"][d]["slots"][s]["is_categorical"]:
                    for v in ontology["domains"][d]["slots"][s]["possible_values"]:
                        value_list[slot][v] = 1
    return value_list


def create_examples(
    set_type,
    dataset_name="multiwoz21",
    class_types=[],
    slot_list=[],
    label_maps={},
    no_label_value_repetitions=False,
    swap_utterances=False,
    delexicalize_sys_utts=False,
    unk_token="[UNK]",
    boolean_slots=True,
    analyze=False,
):
    """Read a DST json file into a list of DSTExample."""

    # TODO: Make sure normalization etc. will be compatible with or suitable for SGD and
    # other datasets as well.
    if dataset_name == "multiwoz21":
        from dataset_multiwoz21 import delex_utt, get_turn_label, is_request, normalize_label, tokenize
    elif dataset_name == "sgd":
        from dataset_sgd import delex_utt, get_turn_label, is_request, normalize_label, tokenize
    else:
        raise ValueError("Unknown dataset_name.")

    dataset_args = {"dataset_name": dataset_name}
    dataset_dict = load_dataset(**dataset_args)

    if slot_list == []:
        slot_list = get_slot_list()

    data = load_dst_data(dataset_dict, data_split=set_type, speaker="all", dialogue_acts=True, split_to_turn=False)

    examples = []
    for d_itr, entry in enumerate(tqdm(data[set_type])):
        # if d_itr > 20:
        #    break # DEBUG
        dialog_id = entry["dialogue_id"]
        # dialog_id = entry['original_id']
        original_id = entry["original_id"]
        domains = entry["domains"]
        turns = entry["turns"]

        # Collects all slot changes throughout the dialog
        cumulative_labels = {slot: "none" for slot in slot_list}

        # First system utterance is empty, since multiwoz starts with user input
        utt_tok_list = [[]]
        mod_slots_list = [{}]
        inform_dict_list = [{}]
        user_act_dict_list = [{}]
        mod_domains_list = [{}]

        # Collect all utterances and their metadata
        usr_sys_switch = True
        for turn in turns:
            utterance = turn["utterance"]
            state = turn["state"] if "state" in turn else {}
            acts = [item for sublist in list(turn["dialogue_acts"].values()) for item in sublist]  # flatten list

            # Assert that system and user utterances alternate
            is_sys_utt = turn["speaker"] in ["sys", "system"]
            if usr_sys_switch == is_sys_utt:
                print("WARN: Wrong order of system and user utterances. Skipping rest of dialog %s" % (dialog_id))
                break
            usr_sys_switch = is_sys_utt

            # Extract metadata: identify modified slots and values informed by the system
            inform_dict = {}
            user_act_dict = {}
            modified_slots = {}
            modified_domains = set()
            for act in acts:
                slot = "%s-%s" % (act["domain"], act["slot"] if act["slot"] != "" else "none")
                if act["intent"] in ["bye", "thank", "hello"]:
                    slot = "general-%s" % (act["intent"])
                value_label = act["value"] if "value" in act else "yes" if act["slot"] != "" else "none"
                value_label = normalize_label(slot, value_label)
                modified_domains.add(act["domain"])  # Remember domains
                if is_sys_utt and act["intent"] in ["inform", "recommend", "select", "book"] and value_label != "none":
                    if slot not in inform_dict:
                        inform_dict[slot] = []
                    inform_dict[slot].append(value_label)
                elif not is_sys_utt:
                    if slot not in user_act_dict:
                        user_act_dict[slot] = []
                    user_act_dict[slot].append(act)
            # INFO: Since the model has no mechanism to predict
            # one among several informed value candidates, we
            # keep only one informed value. For fairness, we
            # apply a global rule:
            for e in inform_dict:
                # ... Option 1: Always keep first informed value
                inform_dict[e] = list([inform_dict[e][0]])
                # ... Option 2: Always keep last informed value
                # inform_dict[e] = list([inform_dict[e][-1]])
            for d in state:
                for s in state[d]:
                    slot = "%s-%s" % (d, s)
                    value_label = normalize_label(slot, state[d][s])
                    # Remember modified slots and entire dialog state
                    if slot in slot_list and cumulative_labels[slot] != value_label:
                        modified_slots[slot] = value_label
                        cumulative_labels[slot] = value_label
                        modified_domains.add(d)  # Remember domains

            # Delexicalize sys utterance
            if delexicalize_sys_utts and is_sys_utt:
                utt_tok_list.append(delex_utt(utterance, inform_dict, unk_token))  # normalizes utterances
            else:
                utt_tok_list.append(tokenize(utterance))  # normalizes utterances

            inform_dict_list.append(inform_dict.copy())
            user_act_dict_list.append(user_act_dict.copy())
            mod_slots_list.append(modified_slots.copy())
            modified_domains = list(modified_domains)
            modified_domains.sort()
            mod_domains_list.append(modified_domains)

        # Form proper (usr, sys) turns
        turn_itr = 0
        diag_seen_slots_dict = {}
        diag_seen_slots_value_dict = {slot: "none" for slot in slot_list}
        diag_state = {slot: "none" for slot in slot_list}
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
            turn_slots = mod_slots_list[i]
            inform_mem = inform_dict_list[i - 1]
            user_act = user_act_dict_list[i]
            turn_domains = mod_domains_list[i]

            guid = "%s-%s" % (dialog_id, turn_itr)

            if analyze:
                print("%15s %2s %s ||| %s" % (dialog_id, turn_itr, " ".join(sys_utt_tok), " ".join(usr_utt_tok)))
                print("%15s %2s [" % (dialog_id, turn_itr), end="")

            new_diag_state = diag_state.copy()
            for slot in slot_list:
                value_label = "none"
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
                inform_label = list(["none"])
                inform_slot_dict[slot] = 0
                if slot in inform_mem:
                    inform_label = inform_mem[slot]
                    inform_slot_dict[slot] = 1

                (informed_value, referred_slot, usr_utt_tok_label, class_type) = get_turn_label(
                    value_label,
                    inform_label,
                    sys_utt_tok,
                    usr_utt_tok,
                    slot,
                    diag_seen_slots_value_dict,
                    slot_last_occurrence=True,
                    label_maps=label_maps,
                )

                inform_dict[slot] = informed_value

                # Requestable slots, domain indicator slots and general slots
                # should have class_type 'request', if they ought to be predicted.
                # Give other class_types preference.
                if "request" in class_types:
                    if class_type in ["none", "unpointable"] and is_request(slot, user_act, turn_domains):
                        class_type = "request"

                # Generally don't use span prediction on sys utterance (but inform prediction instead).
                sys_utt_tok_label = [0 for _ in sys_utt_tok]

                # Determine what to do with value repetitions.
                # If value is unique in seen slots, then tag it, otherwise not,
                # since correct slot assignment can not be guaranteed anymore.
                if not no_label_value_repetitions and slot in diag_seen_slots_dict:
                    if class_type == "copy_value" and list(diag_seen_slots_value_dict.values()).count(value_label) > 1:
                        class_type = "none"
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
                if class_type == "unpointable":
                    class_type_dict[slot] = "none"
                    referral_dict[slot] = "none"
                    if analyze:
                        if slot not in diag_seen_slots_dict or value_label != diag_seen_slots_value_dict[slot]:
                            print("(%s): %s, " % (slot, value_label), end="")
                elif (
                    slot in diag_seen_slots_dict
                    and class_type == diag_seen_slots_dict[slot]
                    and class_type != "copy_value"
                    and class_type != "inform"
                ):
                    # If slot has seen before and its class type did not change, label this slot a not present,
                    # assuming that the slot has not actually been mentioned in this turn.
                    # Exceptions are copy_value and inform. If a seen slot has been tagged as copy_value or inform,
                    # this must mean there is evidence in the original labels, therefore consider
                    # them as mentioned again.
                    class_type_dict[slot] = "none"
                    referral_dict[slot] = "none"
                else:
                    class_type_dict[slot] = class_type
                    referral_dict[slot] = referred_slot
                # Remember that this slot was mentioned during this dialog already.
                if class_type != "none":
                    diag_seen_slots_dict[slot] = class_type
                    diag_seen_slots_value_dict[slot] = value_label
                    new_diag_state[slot] = class_type
                    # Unpointable is not a valid class, therefore replace with
                    # some valid class for now...
                    if class_type == "unpointable":
                        new_diag_state[slot] = "copy_value"

            if analyze:
                print("]")

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
            examples.append(
                DSTExample(
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
                    class_label=class_type_dict,
                )
            )

            # Update some variables.
            diag_state = new_diag_state.copy()

            turn_itr += 1

        if analyze:
            print("----------------------------------------------------------------------")

    return examples


def prediction_normalization(dataset_name, slot, value):
    if dataset_name == "multiwoz21":
        from dataset_multiwoz21 import prediction_normalization as pred_norm

        return pred_norm(slot, value)
    else:
        return value
