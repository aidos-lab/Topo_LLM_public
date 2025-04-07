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

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import MultiheadAttention
import torch.nn.functional as F

from transformers import (BertModel, BertPreTrainedModel,
                          RobertaModel, RobertaPreTrainedModel,
                          ElectraModel, ElectraPreTrainedModel)

PARENT_CLASSES = {
    'bert': BertPreTrainedModel,
    'roberta': RobertaPreTrainedModel,
    'electra': ElectraPreTrainedModel
}

MODEL_CLASSES = {
    BertPreTrainedModel: BertModel,
    RobertaPreTrainedModel: RobertaModel,
    ElectraPreTrainedModel: ElectraModel
}

class ElectraPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        
    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


def TransformerForDST(parent_name):
    if parent_name not in PARENT_CLASSES:
        raise ValueError("Unknown model %s" % (parent_name))

    class TransformerForDST(PARENT_CLASSES[parent_name]):
        def __init__(self, config):
            assert config.model_type in PARENT_CLASSES
            super(TransformerForDST, self).__init__(config)
            self.model_type = config.model_type
            self.slot_list = config.slot_list
            self.noncategorical = config.noncategorical
            self.class_types = config.class_types
            self.class_labels = config.class_labels
            self.class_loss_ratio = config.class_loss_ratio
            self.slot_attention_heads = config.slot_attention_heads
            self.tag_none_target = config.tag_none_target
            self.value_matching_weight = config.value_matching_weight
            self.none_weight = config.none_weight
            self.proto_loss_function = config.proto_loss_function
            self.token_loss_function = config.token_loss_function
            self.value_loss_function = config.value_loss_function

            config.output_hidden_states = True

            # Make sure this module has the same name as in the pretrained checkpoint you want to load!
            self.add_module(self.model_type, MODEL_CLASSES[PARENT_CLASSES[self.model_type]](config))
            if self.model_type == "electra":
                self.pooler = ElectraPooler(config)

            self.dropout = nn.Dropout(config.dropout_rate)
            self.gelu = nn.GELU()

            # Only use refer loss if refer class is present in dataset.
            if 'refer' in self.class_types:
                self.refer_index = self.class_types.index('refer')
            else:
                self.refer_index = -1

            # Attention for slot gates
            self.class_att = MultiheadAttention(config.hidden_size, self.slot_attention_heads)

            # Conditioned sequence tagging
            self.token_att = MultiheadAttention(config.hidden_size, self.slot_attention_heads)
            self.refer_att = MultiheadAttention(config.hidden_size, self.slot_attention_heads)
            self.value_att = MultiheadAttention(config.hidden_size, self.slot_attention_heads)

            self.token_layer_norm_proto = nn.LayerNorm(config.hidden_size)
            self.token_layer_norm = nn.LayerNorm(config.hidden_size)
            self.class_layer_norm = nn.LayerNorm(config.hidden_size)

            # Conditioned slot gate
            self.h1c = nn.Linear(config.hidden_size, config.hidden_size)
            self.h2c = nn.Linear(config.hidden_size * 2, config.hidden_size * 2)
            self.llc = nn.Linear(config.hidden_size * 2, self.class_labels)

            # Conditioned refer gate
            self.h2r = nn.Linear(config.hidden_size * 2, config.hidden_size * 1)

            # Loss functions
            self.binary_cross_entropy = F.binary_cross_entropy
            self.mse = nn.MSELoss(reduction="none")
            self.refer_loss_fct = CrossEntropyLoss(reduction='none', ignore_index=len(self.slot_list)) # Ignore 'none' target
            if self.none_weight != 1.0:
                none_weight = self.none_weight
                weight_mass = none_weight + (self.class_labels - 1)
                none_weight /= weight_mass
                other_weights = 1 / weight_mass
                self.clweights = torch.tensor([other_weights] * self.class_labels)
                self.clweights[self.class_types.index('none')] = none_weight
                self.class_loss_fct = CrossEntropyLoss(weight=self.clweights, reduction='none')
            else:
                self.class_loss_fct = CrossEntropyLoss(reduction='none')

            self.init_weights()

        def forward(self, batch, step=None, mode=None):
            assert mode in [None, "proto", "tag", "encode", "represent"]

            # Required
            input_ids = batch['input_ids']
            input_mask = batch['input_mask']
            # Optional
            segment_ids = batch['segment_ids'] if 'segment_ids' in batch else None
            usr_mask = batch['usr_mask'] if 'usr_mask' in batch else None
            # For loss computation
            token_pos = batch['start_pos'] if 'start_pos' in batch else None
            refer_id = batch['refer_id'] if 'refer_id' in batch else None
            class_label_id = batch['class_label_id'] if 'class_label_id' in batch else None
            # Dynamic elements
            slot_ids = batch['slot_ids'] if 'slot_ids' in batch else None
            slot_mask = batch['slot_mask'] if 'slot_mask' in batch else None
            value_labels = batch['value_labels'] if 'value_labels' in batch else None
            dropout_value_feat = batch['dropout_value_feat'] if 'dropout_value_feat' in batch else None

            batch_input_mask = input_mask
            if slot_ids is not None and slot_mask is not None:
                input_ids = torch.cat((input_ids, slot_ids))
                input_mask = torch.cat((input_mask, slot_mask))

            outputs = getattr(self, self.model_type)(
                input_ids,
                attention_mask=input_mask,
                token_type_ids=None,
                position_ids=None,
                head_mask=None
            )

            sequence_output = outputs[0]
            if self.model_type == "electra":
                pooled_output = self.pooler(sequence_output)
            else:
                pooled_output = outputs[1]

            if slot_ids is not None and slot_mask is not None:
                encoded_slots_seq = sequence_output[-1 * len(slot_ids):, :, :]
                sequence_output = sequence_output[:-1 * len(slot_ids), :, :]
                encoded_slots_pooled = pooled_output[-1 * len(slot_ids):, :]
                pooled_output = pooled_output[:-1 * len(slot_ids), :]

            sequence_output = self.dropout(sequence_output)
            pooled_output = self.dropout(pooled_output)

            inverted_input_mask = ~(batch_input_mask.bool())
            if usr_mask is None:
                usr_mask = input_mask
            inverted_usr_mask = ~(usr_mask.bool())

            # Create vector representations only
            if mode == "encode": 
                return pooled_output, sequence_output, None

            # Proto-DST
            if mode == "proto":
                pos_vectors = {}
                pos_weights = {}
                pos_vectors, pos_weights = self.token_att(
                    query=encoded_slots_pooled.squeeze(1).unsqueeze(0),
                    key=sequence_output.transpose(0, 1),
                    value=sequence_output.transpose(0, 1),
                    key_padding_mask=inverted_input_mask,
                    need_weights=True)
                pos_vectors = pos_vectors.squeeze(0)
                pos_vectors = self.token_layer_norm_proto(pos_vectors)
                pos_weights = pos_weights.squeeze(1)

                pos_labels_clipped = torch.clamp(token_pos.float(), min=0, max=1)
                pos_labels_clipped_scaled = pos_labels_clipped / torch.clamp(pos_labels_clipped.sum(1).unsqueeze(1), min=1)
                if self.proto_loss_function == "mse":
                    pos_token_loss = self.mse(pos_weights, pos_labels_clipped_scaled) # MSE should be better for scaled targets
                else:
                    pos_token_loss = self.binary_cross_entropy(pos_weights, pos_labels_clipped_scaled, reduction="none")
                pos_token_loss = pos_token_loss.sum(1)

                per_example_loss = pos_token_loss
                total_loss = per_example_loss.sum()

                return (total_loss, pos_weights,)

            # Value tagging with proto-DST
            if mode == "tag":
                _, tag_weights = self.token_att(query=torch.stack(list(batch['value_reps'].values())).squeeze(2),
                                                key=sequence_output.transpose(0, 1),
                                                value=sequence_output.transpose(0, 1),
                                                key_padding_mask=inverted_input_mask + inverted_usr_mask,
                                                need_weights=True)
                return (tag_weights,)

            # Attention for sequence tagging
            vectors = {}
            weights = {}
            for s_itr, slot in enumerate(self.slot_list):
                if slot_ids is not None and slot_mask is not None:
                    encoded_slot_seq = encoded_slots_seq[s_itr]
                    encoded_slot_pooled = encoded_slots_pooled[s_itr]
                else:
                    encoded_slot_seq = batch['encoded_slots_seq'][slot]
                    encoded_slot_pooled = batch['encoded_slots_pooled'][slot]
                query = encoded_slot_pooled.expand(pooled_output.size()).unsqueeze(0)
                vectors[slot], weights[slot] = self.token_att(
                    query=query,
                    key=sequence_output.transpose(0, 1),
                    value=sequence_output.transpose(0, 1),
                    key_padding_mask=inverted_input_mask + inverted_usr_mask,
                    need_weights=True)
                vectors[slot] = vectors[slot].squeeze(0)
                vectors[slot] = self.token_layer_norm(vectors[slot])
                weights[slot] = weights[slot].squeeze(1)

            # Create vector representations only (alternative)
            if mode == "represent":
                return vectors, None, weights

            # ----
            # MAIN
            # ----

            total_loss = 0
            total_cl_loss = 0
            total_tk_loss = 0
            total_tp_loss = 0
            per_slot_per_example_loss = {}
            per_slot_per_example_cl_loss = {}
            per_slot_per_example_tk_loss = {}
            per_slot_per_example_tp_loss = {}
            per_slot_class_logits = {}
            per_slot_token_weights = {}
            per_slot_value_weights = {}
            per_slot_refer_logits = {}
            per_slot_att_weights = {}
            for s_itr, slot in enumerate(self.slot_list):
                if slot_ids is not None and slot_mask is not None:
                    encoded_slot_seq = encoded_slots_seq[s_itr]
                    encoded_slot_pooled = encoded_slots_pooled[s_itr]
                else:
                    encoded_slot_seq = batch['encoded_slots_seq'][slot]
                    encoded_slot_pooled = batch['encoded_slots_pooled'][slot]

                # Attention for slot gates
                query = encoded_slot_pooled.expand(pooled_output.size()).unsqueeze(0)
                att_output, c_weights = self.class_att(
                    query=query,
                    key=sequence_output.transpose(0, 1),
                    value=sequence_output.transpose(0, 1),
                    key_padding_mask=inverted_input_mask,
                    need_weights=True)
                att_output = self.class_layer_norm(att_output)
                per_slot_att_weights[slot] = c_weights.squeeze(1)

                # Conditioned slot gate
                slot_gate_feats = self.gelu(self.h1c(att_output.squeeze(0)))
                slot_gate_input = self.gelu(self.h2c(torch.cat((encoded_slot_pooled.expand(pooled_output.size()), slot_gate_feats), 1)))
                class_logits = self.llc(slot_gate_input)

                # Conditioned refer gate
                slot_refer_input = self.gelu(self.h2r(torch.cat((encoded_slot_pooled.expand(pooled_output.size()), slot_gate_feats), 1)))

                # Sequence tagging
                token_weights = weights[slot]

                # Value matching
                if self.value_matching_weight > 0.0:
                    slot_values = torch.stack(list(batch['encoded_slot_values'][slot].values()))
                    slot_values = slot_values.expand((-1, pooled_output.size(0), -1))
                    is_dropout_sample = (batch['dropout_value_feat'][slot].sum(2) > 0.0)
                    v_lbl = batch['value_labels'][slot] * is_dropout_sample
                    v_lbl_orig = v_lbl == 0
                    orig_feats = slot_values * v_lbl_orig.transpose(0, 1).unsqueeze(2)
                    v_lbl_dropout = v_lbl == 1
                    dropout_feats = batch['dropout_value_feat'][slot].expand(-1, slot_values.size(0), -1) * v_lbl_dropout.unsqueeze(2)
                    slot_values = orig_feats + dropout_feats.transpose(0, 1)
                    _, value_weights = self.value_att(
                        query=vectors[slot].unsqueeze(0),
                        key=slot_values,
                        value=slot_values,
                        need_weights=True)
                    value_weights = value_weights.squeeze(1)

                # Refer gate
                if slot_ids is not None and slot_mask is not None:
                    refer_slots = encoded_slots_pooled.unsqueeze(1).expand(-1, pooled_output.size()[0], -1)
                else:
                    refer_slots = torch.stack(list(batch['encoded_slots_pooled'].values())).expand(-1, pooled_output.size()[0], -1)
                _, refer_weights = self.refer_att(
                    query=slot_refer_input.unsqueeze(0),
                    key=refer_slots,
                    value=refer_slots,
                    need_weights=True)
                refer_weights = refer_weights.squeeze(1)
                refer_logits = refer_weights

                per_slot_class_logits[slot] = class_logits
                per_slot_token_weights[slot] = token_weights
                per_slot_refer_logits[slot] = refer_logits
                if self.value_matching_weight > 0.0:
                    per_slot_value_weights[slot] = value_weights

                # If there are no labels, don't compute loss
                if class_label_id is not None and token_pos is not None and refer_id is not None:
                    # If we are on multi-GPU, split add a dimension
                    if len(token_pos[slot].size()) > 1:
                        token_pos[slot] = token_pos[slot].squeeze(-1)

                    # Sequence tagging loss
                    labels_clipped = torch.clamp(token_pos[slot].float(), min=0, max=1)
                    labels_clipped_scaled = labels_clipped / torch.clamp(labels_clipped.sum(1).unsqueeze(1), min=1)
                    no_seq_mask = labels_clipped_scaled.sum(1) == 0
                    no_seq_w = 1 / batch_input_mask.sum(1)
                    labels_clipped_scaled += batch_input_mask * (no_seq_mask * no_seq_w).unsqueeze(1)
                    if self.token_loss_function == "mse":
                        token_loss = self.mse(token_weights, labels_clipped_scaled) # MSE should be better for scaled targets
                    else:
                        token_loss = self.binary_cross_entropy(token_weights, labels_clipped_scaled, reduction="none")

                    # TODO: subsample negative examples due to their large number?
                    token_loss = token_loss.sum(1)
                    token_is_pointable = (token_pos[slot].sum(1) > 0).float()
                    token_loss *= token_is_pointable

                    # Value matching loss
                    value_loss = torch.zeros(token_is_pointable.size(), device=token_is_pointable.device)
                    if self.value_matching_weight > 0.0:
                        value_labels_clipped = torch.clamp(value_labels[slot].float(), min=0, max=1)
                        value_labels_clipped /= torch.clamp(value_labels_clipped.sum(1).unsqueeze(1), min=1)
                        value_no_seq_mask = value_labels_clipped.sum(1) == 0
                        value_no_seq_w = 1 / value_labels_clipped.size(1)
                        value_labels_clipped += (value_no_seq_mask * value_no_seq_w).unsqueeze(1)
                        if self.value_loss_function == "mse":
                            value_loss = self.mse(value_weights, value_labels_clipped)
                        else:
                            value_loss = self.binary_cross_entropy(value_weights, value_labels_clipped, reduction="none")
                        value_loss = value_loss.sum(1)
                        token_is_matchable = token_is_pointable
                        if self.tag_none_target:
                            token_is_matchable *= (token_pos[slot][:, 1] == 0).float()
                        value_loss *= token_is_matchable

                    # Refer loss
                    # Re-definition is necessary here to make slot-independent prediction possible
                    self.refer_loss_fct = CrossEntropyLoss(reduction='none', ignore_index=len(self.slot_list)) # Ignore 'none' target
                    refer_loss = self.refer_loss_fct(refer_logits, refer_id[slot])
                    token_is_referrable = torch.eq(class_label_id[slot], self.refer_index).float()
                    refer_loss *= token_is_referrable

                    # Class loss (i.e., slot gate loss)
                    class_loss = self.class_loss_fct(class_logits, class_label_id[slot])

                    if self.refer_index > -1:
                        per_example_loss = (self.class_loss_ratio) * class_loss + ((1 - self.class_loss_ratio) / 2) * token_loss + ((1 - self.class_loss_ratio) / 2) * refer_loss + self.value_matching_weight * value_loss
                    else:
                        per_example_loss = self.class_loss_ratio * class_loss + (1 - self.class_loss_ratio) * token_loss + self.value_matching_weight * value_loss

                    total_loss += per_example_loss.sum()
                    total_cl_loss += class_loss.sum()
                    total_tk_loss += token_loss.sum()
                    total_tp_loss += value_loss.sum()
                    per_slot_per_example_loss[slot] = per_example_loss
                    per_slot_per_example_cl_loss[slot] = class_loss
                    per_slot_per_example_tk_loss[slot] = token_loss
                    per_slot_per_example_tp_loss[slot] = value_loss

            # add hidden states and attention if they are here
            outputs = (total_loss,
                       total_cl_loss,
                       total_tk_loss,
                       total_tp_loss,
                       per_slot_per_example_loss,
                       per_slot_per_example_cl_loss,
                       per_slot_per_example_tk_loss,
                       per_slot_per_example_tp_loss,
                       per_slot_class_logits,
                       per_slot_token_weights,
                       per_slot_value_weights,
                       per_slot_refer_logits,
                       per_slot_att_weights,) + (vectors,)

            return outputs

    return TransformerForDST
