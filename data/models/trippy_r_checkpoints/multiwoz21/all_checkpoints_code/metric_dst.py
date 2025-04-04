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

import argparse
import glob
import json
import math
import re
import sys

import numpy as np


def load_dataset_config(dataset_config):
    with open(dataset_config, "r", encoding="utf-8") as f:
        raw_config = json.load(f)
    return (
        raw_config["class_types"],
        raw_config["slots"],
        raw_config["label_maps"],
        raw_config["noncategorical"],
        raw_config["boolean"],
    )


def tokenize(text):
    if "\u0120" in text:
        text = re.sub(" ", "", text)
        text = re.sub("\u0120", " ", text)
    else:
        text = re.sub(" ##", "", text)
    text = text.strip()
    return " ".join([tok for tok in map(str.strip, re.split("(\W+)", text)) if len(tok) > 0])


def filter_sequences(seqs, mode="first"):
    if mode == "first":
        return tokenize(seqs[0][0][0])
    elif mode == "max_first":
        max_conf = 0
        max_idx = 0
        for e_itr, e in enumerate(seqs[0]):
            if e[1] > max_conf:
                max_conf = e[1]
                max_idx = e_itr
        return tokenize(seqs[0][max_idx][0])
    elif mode == "max":
        max_conf = 0
        max_t_idx = 0
        for t_itr, t in enumerate(seqs):
            for e_itr, e in enumerate(t):
                if e[1] > max_conf:
                    max_conf = e[1]
                    max_t_idx = t_itr
                    max_idx = e_itr
        return tokenize(seqs[max_t_idx][max_idx][0])
    else:
        print("WARN: mode %s unknown. Aborting." % mode)
        exit()


def is_in_list(tok, value):
    found = False
    tok_list = [item for item in map(str.strip, re.split("(\W+)", tok)) if len(item) > 0]
    value_list = [item for item in map(str.strip, re.split("(\W+)", value)) if len(item) > 0]
    tok_len = len(tok_list)
    value_len = len(value_list)
    for i in range(tok_len + 1 - value_len):
        if tok_list[i : i + value_len] == value_list:
            found = True
            break
    return found


def check_slot_inform(value_label, inform_label, label_maps):
    value = inform_label
    if value_label == inform_label:
        value = value_label
    elif is_in_list(inform_label, value_label):
        value = value_label
    elif is_in_list(value_label, inform_label):
        value = value_label
    elif inform_label in label_maps:
        for inform_label_variant in label_maps[inform_label]:
            if value_label == inform_label_variant:
                value = value_label
                break
            elif is_in_list(inform_label_variant, value_label):
                value = value_label
                break
            elif is_in_list(value_label, inform_label_variant):
                value = value_label
                break
    elif value_label in label_maps:
        for value_label_variant in label_maps[value_label]:
            if value_label_variant == inform_label:
                value = value_label
                break
            elif is_in_list(inform_label, value_label_variant):
                value = value_label
                break
            elif is_in_list(value_label_variant, inform_label):
                value = value_label
                break
    return value


def match(gt, pd, label_maps):
    # We want to be as conservative as possible here.
    # We only allow maps according to label_maps and
    # tolerate the absence/presence of the definite article.
    if pd[:4] == "the " and gt == pd[4:]:
        return True
    if gt[:4] == "the " and gt[4:] == pd:
        return True
    if gt in label_maps:
        for variant in label_maps[gt]:
            if variant == pd:
                return True
    return False


def get_joint_slot_correctness(
    fp,
    args,
    class_types,
    label_maps,
    key_class_label_id="class_label_id",
    key_class_prediction="class_prediction",
    key_start_pos="start_pos",
    key_start_prediction="start_prediction",
    key_start_confidence="start_confidence",
    key_refer_id="refer_id",
    key_refer_prediction="refer_prediction",
    key_slot_groundtruth="slot_groundtruth",
    key_slot_prediction="slot_prediction",
    key_slot_dist_prediction="slot_dist_prediction",
    key_slot_dist_confidence="slot_dist_confidence",
    key_value_prediction="value_prediction",
    key_value_groundtruth="value_groundtruth",
    key_value_confidence="value_confidence",
    key_slot_value_prediction="slot_value_prediction",
    key_slot_value_confidence="slot_value_confidence",
    noncategorical=False,
    boolean=False,
):
    with open(fp) as f:
        preds = json.load(f)
        class_correctness = [[] for cl in range(len(class_types) + 1)]
        confusion_matrix = [[[] for cl_b in range(len(class_types))] for cl_a in range(len(class_types))]
        pos_correctness = []
        refer_correctness = []
        val_correctness = []
        total_correctness = []
        c_tp = {ct: 0 for ct in range(len(class_types))}
        c_tn = {ct: 0 for ct in range(len(class_types))}
        c_fp = {ct: 0 for ct in range(len(class_types))}
        c_fn = {ct: 0 for ct in range(len(class_types))}
        s_confidence_bins = {"%.1f" % (0.1 + b * 0.1): 0 for b in range(10)}
        s_confidence_cnts = {"%.1f" % (0.1 + b * 0.1): 0 for b in range(10)}
        confidence_bins = {"%.1f" % (0.1 + b * 0.1): 0 for b in range(10)}
        confidence_cnts = {"%.1f" % (0.1 + b * 0.1): 0 for b in range(10)}
        a_confidence_bins = {"%.1f" % (0.1 + b * 0.1): 0 for b in range(10)}
        a_confidence_cnts = {"%.1f" % (0.1 + b * 0.1): 0 for b in range(10)}

        value_match_cnt = 0
        for pred in preds:
            guid = pred["guid"]  # List: set_type, dialogue_idx, turn_idx
            turn_gt_class = pred[key_class_label_id]
            turn_pd_class = pred[key_class_prediction]
            gt_start_pos = pred[key_start_pos]
            pd_start_pos = pred[key_start_prediction]
            pd_start_conf = pred[key_start_confidence]
            gt_refer = pred[key_refer_id]
            pd_refer = pred[key_refer_prediction]
            gt_slot = tokenize(pred[key_slot_groundtruth])
            pd_slot = pred[key_slot_prediction]
            pd_slot_dist_pred = tokenize(pred[key_slot_dist_prediction])
            pd_slot_dist_conf = float(pred[key_slot_dist_confidence])
            pd_slot_value_pred = tokenize(pred[key_slot_value_prediction])
            pd_slot_value_conf = pred[key_slot_value_confidence]

            pd_slot_raw = pd_slot
            if isinstance(pd_slot, list):
                pd_slot = filter_sequences(pd_slot, mode="max")
            else:
                pd_slot = tokenize(pd_slot)

            # Make sure the true turn labels are contained in the prediction json file!
            joint_gt_slot = gt_slot

            # Sequence tagging confidence
            if len(pd_start_pos) > 0:
                avg_s_conf = np.mean(pd_start_conf)
                if avg_s_conf == 0.0:
                    avg_s_conf += 1e-8
                s_c_bin = "%.1f" % (math.ceil(avg_s_conf * 10) / 10)
                if gt_start_pos == pd_start_pos:
                    s_confidence_bins[s_c_bin] += 1
                s_confidence_cnts[s_c_bin] += 1

            # Distance based value matching confidence
            if pd_slot_dist_conf == 0.0:
                pd_slot_dist_conf += 1e-8
            c_bin = "%.1f" % (math.ceil(pd_slot_dist_conf * 10) / 10)
            if joint_gt_slot == pd_slot_dist_pred:
                confidence_bins[c_bin] += 1
            confidence_cnts[c_bin] += 1

            # Attention based value matching confidence
            if pd_slot_value_conf == 0.0:
                pd_slot_value_conf += 1e-8
            c_bin = "%.1f" % (math.ceil(pd_slot_value_conf * 10) / 10)
            if joint_gt_slot == pd_slot_value_pred:
                a_confidence_bins[c_bin] += 1
            a_confidence_cnts[c_bin] += 1

            if guid[-1] == "0":  # First turn, reset the slots
                joint_pd_slot = "none"

            # If turn_pd_class or a value to be copied is "none", do not update the dialog state.
            if turn_pd_class == class_types.index("none"):
                pass
            elif turn_pd_class == class_types.index("dontcare"):
                if not boolean:
                    joint_pd_slot = "dontcare"
            elif turn_pd_class == class_types.index("copy_value"):
                if not boolean:
                    if pd_slot not in ["< none >", "[ NONE ]"]:
                        joint_pd_slot = pd_slot
            elif "true" in class_types and turn_pd_class == class_types.index("true"):
                if boolean:
                    joint_pd_slot = "true"
            elif "false" in class_types and turn_pd_class == class_types.index("false"):
                if boolean:
                    joint_pd_slot = "false"
            elif "refer" in class_types and turn_pd_class == class_types.index("refer"):
                if not boolean:
                    if pd_slot[0:2] == "§§":
                        if pd_slot[2:].strip() != "none":
                            joint_pd_slot = check_slot_inform(joint_gt_slot, pd_slot[2:].strip(), label_maps)
                    elif pd_slot != "none":
                        joint_pd_slot = pd_slot
            elif "inform" in class_types and turn_pd_class == class_types.index("inform"):
                if not boolean:
                    if pd_slot[0:2] == "§§":
                        if pd_slot[2:].strip() != "none":
                            joint_pd_slot = check_slot_inform(joint_gt_slot, pd_slot[2:].strip(), label_maps)
            elif "request" in class_types and turn_pd_class == class_types.index("request"):
                pass
            else:
                print("ERROR: Unexpected class_type. Aborting.")
                exit()

            # Value matching
            if args.confidence_threshold < 1.0 and turn_pd_class == class_types.index("copy_value") and not boolean:
                # Treating categorical slots
                if not noncategorical:
                    max_conf = max(np.mean(pd_start_conf), pd_slot_dist_conf, pd_slot_value_conf)
                    if max_conf == pd_slot_dist_conf and max_conf > args.confidence_threshold:
                        joint_pd_slot = tokenize(pd_slot_dist_pred)
                        value_match_cnt += 1
                    elif max_conf == pd_slot_value_conf and max_conf > args.confidence_threshold:
                        joint_pd_slot = tokenize(pd_slot_value_pred)
                        value_match_cnt += 1
                # Treating all slots (including categorical slots)
                if pd_slot_dist_conf > args.confidence_threshold:
                    joint_pd_slot = tokenize(pd_slot_dist_pred)
                    value_match_cnt += 1

            total_correct = True

            # Check the per turn correctness of the class_type prediction
            if turn_gt_class == turn_pd_class:
                class_correctness[turn_gt_class].append(1.0)
                class_correctness[-1].append(1.0)
                c_tp[turn_gt_class] += 1
                # Only where there is a span, we check its per turn correctness
                if turn_gt_class == class_types.index("copy_value"):
                    if gt_start_pos == pd_start_pos:
                        pos_correctness.append(1.0)
                    else:
                        pos_correctness.append(0.0)
                # Only where there is a referral, we check its per turn correctness
                if "refer" in class_types and turn_gt_class == class_types.index("refer"):
                    if gt_refer == pd_refer:
                        refer_correctness.append(1.0)
                        print("  [%s] Correct referral: %s | %s" % (guid, gt_refer, pd_refer))
                    else:
                        refer_correctness.append(0.0)
                        print("  [%s] Incorrect referral: %s | %s" % (guid, gt_refer, pd_refer))
            else:
                if turn_gt_class == class_types.index("copy_value"):
                    pos_correctness.append(0.0)
                if "refer" in class_types and turn_gt_class == class_types.index("refer"):
                    refer_correctness.append(0.0)
                class_correctness[turn_gt_class].append(0.0)
                class_correctness[-1].append(0.0)
                confusion_matrix[turn_gt_class][turn_pd_class].append(1.0)
                c_fn[turn_gt_class] += 1
                c_fp[turn_pd_class] += 1
            for cc in range(len(class_types)):
                if cc != turn_gt_class and cc != turn_pd_class:
                    c_tn[cc] += 1

            # Check the joint slot correctness.
            # If the value label is not none, then we need to have a value prediction.
            # Even if the class_type is 'none', there can still be a value label,
            # it might just not be pointable in the current turn. It might however
            # be referrable and thus predicted correctly.
            if joint_gt_slot == joint_pd_slot:
                val_correctness.append(1.0)
            elif (
                joint_gt_slot != "none"
                and joint_gt_slot != "dontcare"
                and joint_gt_slot != "true"
                and joint_gt_slot != "false"
            ):
                is_match = match(joint_gt_slot, joint_pd_slot, label_maps)
                if not is_match:
                    val_correctness.append(0.0)
                    total_correct = False
                    print(
                        "  [%s] Incorrect value (variant): %s (turn class: %s) | %s (turn class: %s) | %.2f %s %.2f %s %s %s"
                        % (
                            guid,
                            joint_gt_slot,
                            turn_gt_class,
                            joint_pd_slot,
                            turn_pd_class,
                            np.mean(pd_start_conf),
                            pd_slot_raw,
                            pd_slot_dist_conf,
                            pd_slot_dist_pred,
                            "%.2f" % pd_slot_value_conf if pd_slot_value_pred != "" else "",
                            pd_slot_value_pred,
                        )
                    )
                else:
                    val_correctness.append(1.0)
            else:
                val_correctness.append(0.0)
                total_correct = False
                print(
                    "  [%s] Incorrect value: %s (turn class: %s) | %s (turn class: %s) | %.2f %s %.2f %s %s %s"
                    % (
                        guid,
                        joint_gt_slot,
                        turn_gt_class,
                        joint_pd_slot,
                        turn_pd_class,
                        np.mean(pd_start_conf),
                        pd_slot_raw,
                        pd_slot_dist_conf,
                        pd_slot_dist_pred,
                        "%.2f" % pd_slot_value_conf if pd_slot_value_pred != "" else "",
                        pd_slot_value_pred,
                    )
                )

            total_correctness.append(1.0 if total_correct else 0.0)

        # Account for empty lists (due to no instances of spans or referrals being seen)
        if pos_correctness == []:
            pos_correctness.append(1.0)
        if refer_correctness == []:
            refer_correctness.append(1.0)

        for ct in range(len(class_types)):
            if c_tp[ct] + c_fp[ct] > 0:
                precision = c_tp[ct] / (c_tp[ct] + c_fp[ct])
            else:
                precision = 1.0
            if c_tp[ct] + c_fn[ct] > 0:
                recall = c_tp[ct] / (c_tp[ct] + c_fn[ct])
            else:
                recall = 1.0
            if precision + recall > 0:
                f1 = 2 * ((precision * recall) / (precision + recall))
            else:
                f1 = 1.0
            if c_tp[ct] + c_tn[ct] + c_fp[ct] + c_fn[ct] > 0:
                acc = (c_tp[ct] + c_tn[ct]) / (c_tp[ct] + c_tn[ct] + c_fp[ct] + c_fn[ct])
            else:
                acc = 1.0
            print(
                "Performance for class '%s' (%s): Recall: %.2f (%d of %d), Precision: %.2f, F1: %.2f, Accuracy: %.2f (TP/TN/FP/FN: %d/%d/%d/%d)"
                % (
                    class_types[ct],
                    ct,
                    recall,
                    np.sum(class_correctness[ct]),
                    len(class_correctness[ct]),
                    precision,
                    f1,
                    acc,
                    c_tp[ct],
                    c_tn[ct],
                    c_fp[ct],
                    c_fn[ct],
                )
            )

        print("Confusion matrix:")
        for cl in range(len(class_types)):
            print("    %s" % (cl), end="")
        print("")
        for cl_a in range(len(class_types)):
            print("%s " % (cl_a), end="")
            for cl_b in range(len(class_types)):
                if len(class_correctness[cl_a]) > 0:
                    print("%.2f " % (np.sum(confusion_matrix[cl_a][cl_b]) / len(class_correctness[cl_a])), end="")
                else:
                    print("---- ", end="")
            print("")

        print("Confidence bins for sequence tagging:")
        print("  bin cor")
        for c in s_confidence_bins:
            print(
                "  %s %.2f (%d of %d)"
                % (c, s_confidence_bins[c] / (s_confidence_cnts[c] + 1e-8), s_confidence_bins[c], s_confidence_cnts[c])
            )

        print("Confidence bins for distance based value matching:")
        print("  bin cor")
        for c in confidence_bins:
            print(
                "  %s %.2f (%d of %d)"
                % (c, confidence_bins[c] / (confidence_cnts[c] + 1e-8), confidence_bins[c], confidence_cnts[c])
            )

        print("Confidence bins for attention based value matching:")
        print("  bin cor")
        for c in a_confidence_bins:
            print(
                "  %s %.2f (%d of %d)"
                % (c, a_confidence_bins[c] / (a_confidence_cnts[c] + 1e-8), a_confidence_bins[c], a_confidence_cnts[c])
            )

        print("Values replaced by value matching:", value_match_cnt)

        # Notes:
        # - We are adding the `dtype=object` to the np.asarray() calls to avoid and error from different length lists
        return (
            np.asarray(total_correctness),
            np.asarray(val_correctness),
            np.asarray(class_correctness, dtype=object),
            np.asarray(pos_correctness),
            np.asarray(refer_correctness),
            np.asarray(confusion_matrix, dtype=object),
            c_tp,
            c_tn,
            c_fp,
            c_fn,
            s_confidence_bins,
            s_confidence_cnts,
            confidence_bins,
            confidence_cnts,
            a_confidence_bins,
            a_confidence_cnts,
        )


if __name__ == "__main__":
    acc_list = []
    s_acc_list = []
    key_class_label_id = "class_label_id_%s"
    key_class_prediction = "class_prediction_%s"
    key_start_pos = "start_pos_%s"
    key_start_prediction = "start_prediction_%s"
    key_start_confidence = "start_confidence_%s"
    key_refer_id = "refer_id_%s"
    key_refer_prediction = "refer_prediction_%s"
    key_slot_groundtruth = "slot_groundtruth_%s"
    key_slot_prediction = "slot_prediction_%s"
    key_slot_dist_prediction = "slot_dist_prediction_%s"
    key_slot_dist_confidence = "slot_dist_confidence_%s"
    key_value_prediction = "value_prediction_%s"
    key_value_groundtruth = "value_label_id_%s"
    key_value_confidence = "value_confidence_%s"
    key_slot_value_prediction = "slot_value_prediction_%s"
    key_slot_value_confidence = "slot_value_confidence_%s"

    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--dataset_config", default=None, type=str, required=True, help="Dataset configuration file.")
    parser.add_argument("--file_list", default=None, type=str, required=True, help="List of input files.")

    # Other parameters
    parser.add_argument(
        "--confidence_threshold",
        default=1.0,
        type=float,
        help="Threshold for value matching confidence. 1.0 means no value matching is used.",
    )

    args = parser.parse_args()

    assert args.confidence_threshold >= 0.0 and args.confidence_threshold <= 1.0

    class_types, slots, label_maps, noncategorical, boolean = load_dataset_config(args.dataset_config)

    # Prepare label_maps
    label_maps_tmp = {}
    for v in label_maps:
        label_maps_tmp[tokenize(v)] = [tokenize(nv) for nv in label_maps[v]]
    label_maps = label_maps_tmp

    for fp in sorted(glob.glob(args.file_list)):
        # Infer slot list from data if not provided.
        if len(slots) == 0:
            with open(fp) as f:
                preds = json.load(f)
                for e in preds[0]:
                    slot = re.match("^slot_groundtruth_(.*)$", e)
                    slot = slot[1] if slot else None
                    if slot and slot not in slots:
                        slots.append(slot)
        print(fp)
        goal_correctness = 1.0
        cls_acc = [[] for cl in range(len(class_types))]
        cls_conf = [[[] for cl_b in range(len(class_types))] for cl_a in range(len(class_types))]
        c_tp = {ct: 0 for ct in range(len(class_types))}
        c_tn = {ct: 0 for ct in range(len(class_types))}
        c_fp = {ct: 0 for ct in range(len(class_types))}
        c_fn = {ct: 0 for ct in range(len(class_types))}
        s_confidence_bins = {"%.1f" % (0.1 + b * 0.1): 0 for b in range(10)}
        s_confidence_cnts = {"%.1f" % (0.1 + b * 0.1): 0 for b in range(10)}
        confidence_bins = {"%.1f" % (0.1 + b * 0.1): 0 for b in range(10)}
        confidence_cnts = {"%.1f" % (0.1 + b * 0.1): 0 for b in range(10)}
        a_confidence_bins = {"%.1f" % (0.1 + b * 0.1): 0 for b in range(10)}
        a_confidence_cnts = {"%.1f" % (0.1 + b * 0.1): 0 for b in range(10)}
        for slot in slots:
            (
                tot_cor,
                joint_val_cor,
                cls_cor,
                pos_cor,
                ref_cor,
                conf_mat,
                ctp,
                ctn,
                cfp,
                cfn,
                scbins,
                sccnts,
                cbins,
                ccnts,
                acbins,
                accnts,
            ) = get_joint_slot_correctness(
                fp,
                args,
                class_types,
                label_maps,
                key_class_label_id=(key_class_label_id % slot),
                key_class_prediction=(key_class_prediction % slot),
                key_start_pos=(key_start_pos % slot),
                key_start_prediction=(key_start_prediction % slot),
                key_start_confidence=(key_start_confidence % slot),
                key_refer_id=(key_refer_id % slot),
                key_refer_prediction=(key_refer_prediction % slot),
                key_slot_groundtruth=(key_slot_groundtruth % slot),
                key_slot_prediction=(key_slot_prediction % slot),
                key_slot_dist_prediction=(key_slot_dist_prediction % slot),
                key_slot_dist_confidence=(key_slot_dist_confidence % slot),
                key_value_prediction=(key_value_prediction % slot),
                key_value_groundtruth=(key_value_groundtruth % slot),
                key_value_confidence=(key_value_confidence % slot),
                key_slot_value_prediction=(key_slot_value_prediction % slot),
                key_slot_value_confidence=(key_slot_value_confidence % slot),
                noncategorical=slot in noncategorical,
                boolean=slot in boolean,
            )
            print(
                "%s: joint slot acc: %g, joint value acc: %g, turn class acc: %g, turn position acc: %g, turn referral acc: %g"
                % (
                    slot,
                    np.mean(tot_cor),
                    np.mean(joint_val_cor),
                    np.mean(cls_cor[-1]),
                    np.mean(pos_cor),
                    np.mean(ref_cor),
                )
            )
            goal_correctness *= tot_cor
            for cl_a in range(len(class_types)):
                cls_acc[cl_a] += cls_cor[cl_a]
                for cl_b in range(len(class_types)):
                    cls_conf[cl_a][cl_b] += list(conf_mat[cl_a][cl_b])
                c_tp[cl_a] += ctp[cl_a]
                c_tn[cl_a] += ctn[cl_a]
                c_fp[cl_a] += cfp[cl_a]
                c_fn[cl_a] += cfn[cl_a]
            for c in scbins:
                s_confidence_bins[c] += scbins[c]
                s_confidence_cnts[c] += sccnts[c]
            for c in cbins:
                confidence_bins[c] += cbins[c]
                confidence_cnts[c] += ccnts[c]
            for c in cbins:
                a_confidence_bins[c] += acbins[c]
                a_confidence_cnts[c] += accnts[c]

        for ct in range(len(class_types)):
            if c_tp[ct] + c_fp[ct] > 0:
                precision = c_tp[ct] / (c_tp[ct] + c_fp[ct])
            else:
                precision = 1.0
            if c_tp[ct] + c_fn[ct] > 0:
                recall = c_tp[ct] / (c_tp[ct] + c_fn[ct])
            else:
                recall = 1.0
            if precision + recall > 0:
                f1 = 2 * ((precision * recall) / (precision + recall))
            else:
                f1 = 1.0
            if c_tp[ct] + c_tn[ct] + c_fp[ct] + c_fn[ct] > 0:
                acc = (c_tp[ct] + c_tn[ct]) / (c_tp[ct] + c_tn[ct] + c_fp[ct] + c_fn[ct])
            else:
                acc = 1.0
            print(
                "Performance for class '%s' (%s): Recall: %.2f (%d of %d), Precision: %.2f, F1: %.2f, Accuracy: %.2f (TP/TN/FP/FN: %d/%d/%d/%d)"
                % (
                    class_types[ct],
                    ct,
                    recall,
                    np.sum(cls_acc[ct]),
                    len(cls_acc[ct]),
                    precision,
                    f1,
                    acc,
                    c_tp[ct],
                    c_tn[ct],
                    c_fp[ct],
                    c_fn[ct],
                )
            )

        print("Confusion matrix:")
        for cl in range(len(class_types)):
            print("    %s" % (cl), end="")
        print("")
        for cl_a in range(len(class_types)):
            print("%s " % (cl_a), end="")
            for cl_b in range(len(class_types)):
                if len(cls_acc[cl_a]) > 0:
                    print("%.2f " % (np.sum(cls_conf[cl_a][cl_b]) / len(cls_acc[cl_a])), end="")
                else:
                    print("---- ", end="")
            print("")

        print("Confidence bins for sequence tagging:")
        print("  bin cor")
        for c in s_confidence_bins:
            print(
                "  %s %.2f (%d of %d)"
                % (c, s_confidence_bins[c] / (s_confidence_cnts[c] + 1e-8), s_confidence_bins[c], s_confidence_cnts[c])
            )

        print("Confidence bins for distance based value matching:")
        print("  bin cor")
        for c in confidence_bins:
            print(
                "  %s %.2f (%d of %d)"
                % (c, confidence_bins[c] / (confidence_cnts[c] + 1e-8), confidence_bins[c], confidence_cnts[c])
            )

        print("Confidence bins for attention based value matching:")
        print("  bin cor")
        for c in a_confidence_bins:
            print(
                "  %s %.2f (%d of %d)"
                % (c, a_confidence_bins[c] / (a_confidence_cnts[c] + 1e-8), a_confidence_bins[c], a_confidence_cnts[c])
            )

        acc = np.mean(goal_correctness)
        acc_list.append((fp, acc))

    acc_list_s = sorted(acc_list, key=lambda tup: tup[1], reverse=True)
    for fp, acc in acc_list_s:
        print("Joint goal acc: %g, %s" % (acc, fp))
