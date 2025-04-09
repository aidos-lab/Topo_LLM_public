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
from utils_dst import DSTExample, convert_to_unicode

# Required for mapping slot names in dialogue_acts.json file
# to proper designations.
ACTS_DICT = {
    "taxi-depart": "taxi-departure",
    "taxi-dest": "taxi-destination",
    "taxi-leave": "taxi-leaveAt",
    "taxi-arrive": "taxi-arriveBy",
    "train-depart": "train-departure",
    "train-dest": "train-destination",
    "train-leave": "train-leaveAt",
    "train-arrive": "train-arriveBy",
    "train-people": "train-book_people",
    "restaurant-price": "restaurant-pricerange",
    "restaurant-people": "restaurant-book_people",
    "restaurant-day": "restaurant-book_day",
    "restaurant-time": "restaurant-book_time",
    "hotel-price": "hotel-pricerange",
    "hotel-people": "hotel-book_people",
    "hotel-day": "hotel-book_day",
    "hotel-stay": "hotel-book_stay",
    "booking-people": "booking-book_people",
    "booking-day": "booking-book_day",
    "booking-stay": "booking-book_stay",
    "booking-time": "booking-book_time",
    "taxi-leaveat": "taxi-leaveAt",
    "taxi-arriveby": "taxi-arriveBy",
    "train-leaveat": "train-leaveAt",
    "train-arriveby": "train-arriveBy",
    "train-bookpeople": "train-book_people",
    "restaurant-bookpeople": "restaurant-book_people",
    "restaurant-bookday": "restaurant-book_day",
    "restaurant-booktime": "restaurant-book_time",
    "hotel-bookpeople": "hotel-book_people",
    "hotel-bookday": "hotel-book_day",
    "hotel-bookstay": "hotel-book_stay",
    "booking-bookpeople": "booking-book_people",
    "booking-bookday": "booking-book_day",
    "booking-bookstay": "booking-book_stay",
    "booking-booktime": "booking-book_time",
}


def prediction_normalization(slot, value):
    def _normalize_time(text):
        informed = False
        if text[:2] == "§§":
            informed = True
            text = text[2:]
        text = re.sub("noon", r"12:00", text)  # noon
        text = re.sub(r"(\d{1})(a\.?m\.?|p\.?m\.?)", r"\1 \2", text)  # am/pm without space
        text = re.sub(r"(^| )(\d{1,2}) ?[^0-9]? ?(\d{2})", r"\1\2:\3", text)  # Missing/wrong separator
        text = re.sub(r"(^| )(\d{1,2})( |$)", r"\1\2:00\3", text)  # normalize simple full hour time
        text = re.sub(r"(^| )(\d{1}:\d{2})", r"\g<1>0\2", text)  # Add missing leading 0
        # Map 12 hour times to 24 hour times
        text = re.sub(
            r"(\d{2})(:\d{2}) ?p\.?m\.?",
            lambda x: str(int(x.groups()[0]) + 12 if int(x.groups()[0]) < 12 else int(x.groups()[0])) + x.groups()[1],
            text,
        )
        text = re.sub(r"(^| )24:(\d{2})", r"\g<1>00:\2", text)  # Correct times that use 24 as hour
        final = re.match(r".*((before|after) \d{2}:\d{2})", text)
        result = text
        if final is not None:
            result = final[1]
        final = re.match(r".*(\d{2}:\d{2})", text)
        if final is not None:
            result = final[1]
        if informed and result[:2] != "§§":
            return "§§" + result
        return result

    def _normalize_value(text):
        text = re.sub(" ?' ?s", "s", text)
        return text

    if "leave" in slot or "arrive" in slot or "time" in slot:
        if isinstance(value, list):
            for e_itr in range(len(value)):
                for ee_itr in range(len(value[e_itr])):
                    tmp = list(value[e_itr][ee_itr])
                    tmp[0] = _normalize_time(tmp[0])
                    value[e_itr][ee_itr] = tuple(tmp)
        else:
            value = _normalize_time(value)
    else:
        value = _normalize_value(value)
    return value


def normalize_time(text):
    text = re.sub(r"(\d{1})(a\.?m\.?|p\.?m\.?)", r"\1 \2", text)  # am/pm without space
    text = re.sub(r"(^| )(\d{1,2}) (a\.?m\.?|p\.?m\.?)", r"\1\2:00 \3", text)  # am/pm short to long form
    text = re.sub(
        r"(^| )(at|from|by|until|after) ?(\d{1,2}) ?(\d{2})([^0-9]|$)", r"\1\2 \3:\4\5", text
    )  # Missing separator
    text = re.sub(r"(^| )(\d{2})[;.,](\d{2})", r"\1\2:\3", text)  # Wrong separator
    text = re.sub(
        r"(^| )(at|from|by|until|after) ?(\d{1,2})([;., ]|$)", r"\1\2 \3:00\4", text
    )  # normalize simple full hour time
    text = re.sub(r"(^| )(\d{1}:\d{2})", r"\g<1>0\2", text)  # Add missing leading 0
    # Map 12 hour times to 24 hour times
    text = re.sub(
        r"(\d{2})(:\d{2}) ?p\.?m\.?",
        lambda x: str(int(x.groups()[0]) + 12 if int(x.groups()[0]) < 12 else int(x.groups()[0])) + x.groups()[1],
        text,
    )
    text = re.sub(r"(^| )24:(\d{2})", r"\g<1>00:\2", text)  # Correct times that use 24 as hour
    return text


def normalize_text(text):
    text = normalize_time(text)
    text = re.sub("n't", " not", text)
    text = re.sub("(^| )zero(-| )star([s.,? ]|$)", r"\g<1>0 star\3", text)
    text = re.sub("(^| )one(-| )star([s.,? ]|$)", r"\g<1>1 star\3", text)
    text = re.sub("(^| )two(-| )star([s.,? ]|$)", r"\g<1>2 star\3", text)
    text = re.sub("(^| )three(-| )star([s.,? ]|$)", r"\g<1>3 star\3", text)
    text = re.sub("(^| )four(-| )star([s.,? ]|$)", r"\g<1>4 star\3", text)
    text = re.sub("(^| )five(-| )star([s.,? ]|$)", r"\g<1>5 star\3", text)
    text = re.sub("archaelogy", "archaeology", text)  # Systematic typo
    text = re.sub("guesthouse", "guest house", text)  # Normalization
    text = re.sub("(^| )b ?& ?b([.,? ]|$)", r"\1bed and breakfast\2", text)  # Normalization
    text = re.sub("bed & breakfast", "bed and breakfast", text)  # Normalization
    text = re.sub("\t", " ", text)  # Error
    text = re.sub("\n", " ", text)  # Error
    return text


# This should only contain label normalizations, no label mappings.
def normalize_label(slot, value_label, boolean_slots=True):
    # Normalization of capitalization
    if isinstance(value_label, str):
        value_label = value_label.lower().strip()
    elif isinstance(value_label, list):
        if len(value_label) > 1:
            value_label = value_label[
                0
            ]  # TODO: Workaround. Note that Multiwoz 2.2 supports variants directly in the labels.
        elif len(value_label) == 1:
            value_label = value_label[0]
        elif len(value_label) == 0:
            value_label = ""

    # Normalization of empty slots
    if value_label == "" or value_label == "not mentioned":
        return "none"

    # Normalization of 'dontcare'
    if value_label == "dont care":
        return "dontcare"

    # Normalization of time slots
    if "leave" in slot or "arrive" in slot or "time" in slot:
        return normalize_time(value_label)

    # Normalization
    if "type" in slot or "name" in slot or "destination" in slot or "departure" in slot:
        value_label = re.sub(" ?'s", "s", value_label)
        value_label = re.sub("guesthouse", "guest house", value_label)

    # Map to boolean slots
    if slot in ["hotel-parking", "hotel-internet"]:
        slot_name = slot.split("-")[1]
        if value_label in ["yes", "free"]:
            if boolean_slots:
                return "true"
            else:
                return slot_name
        if value_label == "no":
            if boolean_slots:
                return "false"
            else:
                return "no " + slot_name
    if slot == "hotel-type":
        if value_label in ["bed and breakfast", "guest houses"]:
            value_label = "guest house"
        if boolean_slots:
            if value_label == "hotel":
                return "true"
            if value_label == "guest house":
                return "false"

    return value_label


def get_token_pos(tok_list, value_label):
    find_pos = []
    found = False
    label_list = [item for item in map(str.strip, re.split(r"(\W+)", value_label)) if len(item) > 0]
    len_label = len(label_list)
    for i in range(len(tok_list) + 1 - len_label):
        if tok_list[i : i + len_label] == label_list:
            find_pos.append((i, i + len_label))  # start, exclusive_end
            found = True
    return found, find_pos


def check_label_existence(value_label, usr_utt_tok, label_maps={}):
    in_usr, usr_pos = get_token_pos(usr_utt_tok, value_label)
    # If no hit even though there should be one, check for value label variants
    if not in_usr and value_label in label_maps:
        for value_label_variant in label_maps[value_label]:
            in_usr, usr_pos = get_token_pos(usr_utt_tok, value_label_variant)
            if in_usr:
                break
    return in_usr, usr_pos


def check_slot_referral(value_label, slot, seen_slots, label_maps={}):
    referred_slot = "none"
    if slot == "hotel-stars" or slot == "hotel-internet" or slot == "hotel-parking":
        return referred_slot
    for s in seen_slots:
        # Avoid matches for slots that share values with different meaning.
        # hotel-internet and -parking are handled separately as Boolean slots.
        if s == "hotel-stars" or s == "hotel-internet" or s == "hotel-parking":
            continue
        if re.match("(hotel|restaurant)-book_people", s) and slot == "hotel-book_stay":
            continue
        if re.match("(hotel|restaurant)-book_people", slot) and s == "hotel-book_stay":
            continue
        if slot != s and (slot not in seen_slots or seen_slots[slot] != value_label):
            if seen_slots[s] == value_label:
                referred_slot = s
                break
            elif value_label in label_maps:
                for value_label_variant in label_maps[value_label]:
                    if seen_slots[s] == value_label_variant:
                        referred_slot = s
                        break
    return referred_slot


def is_in_list(tok, value):
    found = False
    tok_list = [item for item in map(str.strip, re.split(r"(\W+)", tok)) if len(item) > 0]
    value_list = [item for item in map(str.strip, re.split(r"(\W+)", value)) if len(item) > 0]
    tok_len = len(tok_list)
    value_len = len(value_list)
    for i in range(tok_len + 1 - value_len):
        if tok_list[i : i + value_len] == value_list:
            found = True
            break
    return found


def delex_utt(utt, values, unk_token="[UNK]"):
    utt_norm = tokenize(utt)
    for s, vals in values.items():
        for v in vals:
            if v != "none":
                v_norm = tokenize(v)
                v_len = len(v_norm)
                for i in range(len(utt_norm) + 1 - v_len):
                    if utt_norm[i : i + v_len] == v_norm:
                        utt_norm[i : i + v_len] = [unk_token] * v_len
    return utt_norm


# Fuzzy matching to label informed slot values
def check_slot_inform(value_label, inform_label, label_maps={}):
    result = False
    informed_value = "none"
    vl = " ".join(tokenize(value_label))
    for il in inform_label:
        if vl == il:
            result = True
        elif is_in_list(il, vl):
            result = True
        elif is_in_list(vl, il):
            result = True
        elif il in label_maps:
            for il_variant in label_maps[il]:
                if vl == il_variant:
                    result = True
                    break
                elif is_in_list(il_variant, vl):
                    result = True
                    break
                elif is_in_list(vl, il_variant):
                    result = True
                    break
        elif vl in label_maps:
            for value_label_variant in label_maps[vl]:
                if value_label_variant == il:
                    result = True
                    break
                elif is_in_list(il, value_label_variant):
                    result = True
                    break
                elif is_in_list(value_label_variant, il):
                    result = True
                    break
        if result:
            informed_value = il
            break
    return result, informed_value


def get_turn_label(
    value_label, inform_label, sys_utt_tok, usr_utt_tok, slot, seen_slots, slot_last_occurrence, label_maps={}
):
    usr_utt_tok_label = [0 for _ in usr_utt_tok]
    informed_value = "none"
    referred_slot = "none"
    if value_label == "none" or value_label == "dontcare" or value_label == "true" or value_label == "false":
        class_type = value_label
    else:
        in_usr, usr_pos = check_label_existence(value_label, usr_utt_tok, label_maps)
        is_informed, informed_value = check_slot_inform(value_label, inform_label, label_maps)
        if in_usr:
            class_type = "copy_value"
            if slot_last_occurrence:
                (s, e) = usr_pos[-1]
                for i in range(s, e):
                    usr_utt_tok_label[i] = 1
            else:
                for s, e in usr_pos:
                    for i in range(s, e):
                        usr_utt_tok_label[i] = 1
        elif is_informed:
            class_type = "inform"
        else:
            referred_slot = check_slot_referral(value_label, slot, seen_slots, label_maps)
            if referred_slot != "none":
                class_type = "refer"
            else:
                class_type = "unpointable"
    return informed_value, referred_slot, usr_utt_tok_label, class_type


# Requestable slots, general acts and domain indicator slots
def is_request(slot, user_act, turn_domains):
    if slot in user_act:
        if isinstance(user_act[slot], list):
            for act in user_act[slot]:
                if act["intent"] in ["request", "bye", "thank", "greet"]:
                    return True
        elif user_act[slot]["intent"] in ["request", "bye", "thank", "greet"]:
            return True
    do, sl = slot.split("-")
    if sl == "none" and do in turn_domains:
        return True
    return False


def tokenize(utt):
    utt_lower = convert_to_unicode(utt).lower()
    utt_lower = normalize_text(utt_lower)
    utt_tok = utt_to_token(utt_lower)
    return utt_tok


def utt_to_token(utt):
    return [tok for tok in map(lambda x: re.sub(" ", "", x), re.split(r"(\W+)", utt)) if len(tok) > 0]


def create_examples(
    input_file,
    set_type,
    class_types,
    slot_list,
    label_maps={},
    not_domain=None,
    no_label_value_repetitions=False,
    swap_utterances=False,
    delexicalize_sys_utts=False,
    unk_token="[UNK]",
    boolean_slots=True,
    analyze=False,
):
    """Read a DST json file into a list of DSTExample."""

    with open(input_file, "r", encoding="utf-8") as reader:
        input_data = json.load(reader)

    examples = []
    for d_itr, dialog_id in enumerate(tqdm(input_data)):
        entry = input_data[dialog_id]
        goal = entry["goal"]
        utterances = entry["log"]

        # If a leave-one-out domain is provided, skip any dialogue containing this domain.
        if not_domain is not None:
            if not_domain in goal and goal[not_domain] != {}:
                continue

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
        turn_itr = 0
        for utt in utterances:
            # Assert that system and user utterances alternate
            is_sys_utt = utt["metadata"] != {}
            if usr_sys_switch == is_sys_utt:
                print("WARN: Wrong order of system and user utterances. Skipping rest of dialog %s" % (dialog_id))
                break
            usr_sys_switch = is_sys_utt

            if is_sys_utt:
                turn_itr += 1

            # Extract dialog_act information for sys and usr utts.
            inform_dict = {}
            user_act_dict = {}
            modified_slots = {}
            modified_domains = set()
            if "dialog_act" in utt:
                for a in utt["dialog_act"]:
                    aa = a.lower().split("-")
                    for i in utt["dialog_act"][a]:
                        s = i[0].lower()
                        # Some special intents are modeled as slots in TripPy
                        if aa[0] == "general":
                            cs = "%s-%s" % (aa[0], aa[1])
                        else:
                            cs = "%s-%s" % (aa[0], s)
                        if cs in ACTS_DICT:
                            cs = ACTS_DICT[cs]
                        v = normalize_label(cs, i[1].lower().strip())
                        if cs in ["hotel-internet", "hotel-parking"]:
                            v = "true"
                        modified_domains.add(aa[0])  # Remember domains
                        if is_sys_utt and aa[1] in ["inform", "recommend", "select", "book"] and v != "none":
                            if cs not in inform_dict:
                                inform_dict[cs] = []
                            inform_dict[cs].append(v)
                        elif not is_sys_utt:
                            if cs not in user_act_dict:
                                user_act_dict[cs] = []
                            user_act_dict[cs] = {"domain": aa[0], "intent": aa[1], "slot": s, "value": v}
                # INFO: Since the model has no mechanism to predict
                # one among several informed value candidates, we
                # keep only one informed value. For fairness, we
                # apply a global rule:
                for e in inform_dict:
                    # ... Option 1: Always keep first informed value
                    inform_dict[e] = list([inform_dict[e][0]])
                    # ... Option 2: Always keep last informed value
                    # inform_dict[e] = list([inform_dict[e][-1]])
            else:
                print("WARN: dialogue %s is missing dialog_act information." % dialog_id)

            # If sys utt, extract metadata (identify and collect modified slots)
            if is_sys_utt:
                for d in utt["metadata"]:
                    booked = utt["metadata"][d]["book"]["booked"]
                    booked_slots = {}
                    # Check the booked section
                    if booked != []:
                        for s in booked[0]:
                            booked_slots[s] = normalize_label("%s-%s" % (d, s), booked[0][s], boolean_slots)
                    # Check the semi and the inform slots
                    for category in ["book", "semi"]:
                        for s in utt["metadata"][d][category]:
                            cs = "%s-book_%s" % (d, s) if category == "book" else "%s-%s" % (d, s)
                            value_label = normalize_label(cs, utt["metadata"][d][category][s], boolean_slots)
                            # Prefer the slot value as stored in the booked section
                            if s in booked_slots:
                                value_label = booked_slots[s]
                            # Remember modified slots and entire dialog state
                            if cs in slot_list and cumulative_labels[cs] != value_label:
                                modified_slots[cs] = value_label
                                cumulative_labels[cs] = value_label
                                modified_domains.add(cs.split("-")[0])  # Remember domains

            # Delexicalize sys utterance
            if delexicalize_sys_utts and is_sys_utt:
                utt_tok_list.append(delex_utt(utt["text"], inform_dict, unk_token))  # normalizes utterances
            else:
                utt_tok_list.append(tokenize(utt["text"]))  # normalizes utterances

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
            turn_slots = mod_slots_list[i + 1]
            inform_mem = inform_dict_list[i - 1]
            user_act = user_act_dict_list[i]
            turn_domains = mod_domains_list[i + 1]

            guid = "%s-%s-%s" % (set_type, str(dialog_id), str(turn_itr))

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
                booking_slot = "booking-" + slot.split("-")[1]
                if slot in inform_mem:
                    inform_label = inform_mem[slot]
                    inform_slot_dict[slot] = 1
                elif booking_slot in inform_mem:
                    inform_label = inform_mem[booking_slot]
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


def get_value_list(input_file, slot_list, boolean_slots=True, not_domain=None):
    exclude = ["none", "dontcare"]
    if not boolean_slots:
        exclude += ["true", "false"]
    with open(input_file, "r", encoding="utf-8") as reader:
        input_data = json.load(reader)
    value_list = {slot: {} for slot in slot_list}
    for dialog_id in input_data:
        goal = input_data[dialog_id]["goal"]
        # If a leave-one-out domain is provided, skip any dialogue containing this domain.
        if not_domain is not None:
            if not_domain in goal and goal[not_domain] != {}:
                continue
        for utt in input_data[dialog_id]["log"]:
            if utt["metadata"] != {}:
                for d in utt["metadata"]:
                    booked = utt["metadata"][d]["book"]["booked"]
                    booked_slots = {}
                    # Check the booked section
                    if booked != []:
                        for s in booked[0]:
                            booked_slots[s] = normalize_label("%s-%s" % (d, s), booked[0][s], boolean_slots)
                    # Check the semi and the inform slots
                    for category in ["book", "semi"]:
                        for s in utt["metadata"][d][category]:
                            cs = "%s-book_%s" % (d, s) if category == "book" else "%s-%s" % (d, s)
                            value_label = normalize_label(cs, utt["metadata"][d][category][s], boolean_slots)
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
