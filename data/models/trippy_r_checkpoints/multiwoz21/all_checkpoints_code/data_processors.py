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
import os

import dataset_multiwoz21
import dataset_multiwoz21_legacy
import dataset_sim
import dataset_unified
import dataset_woz2


class DataProcessor(object):
    data_dir = ""
    dataset_name = ""
    class_types = []
    slot_list = {}
    noncategorical = []
    boolean = []
    label_maps = {}
    value_list = {"train": {}, "dev": {}, "test": {}}

    def __init__(self, dataset_config, data_dir, not_domain=None):
        self.data_dir = data_dir
        # Load dataset config file.
        with open(dataset_config, "r", encoding="utf-8") as f:
            raw_config = json.load(f)
        self.dataset_name = raw_config["dataset_name"] if "dataset_name" in raw_config else ""
        self.class_types = raw_config["class_types"]  # Required
        self.slot_list = raw_config["slots"] if "slots" in raw_config else {}
        self.noncategorical = raw_config["noncategorical"] if "noncategorical" in raw_config else []
        self.boolean = raw_config["boolean"] if "boolean" in raw_config else []
        self.label_maps = raw_config["label_maps"] if "label_maps" in raw_config else {}
        # If not slot list is provided, generate from data.
        if len(self.slot_list) == 0:
            self.slot_list = self._get_slot_list()

    def _add_dummy_value_to_value_list(self):
        for dset in self.value_list:
            for s in self.value_list[dset]:
                if len(self.value_list[dset][s]) == 0:
                    self.value_list[dset][s] = {"dummy": 1}

    def _remove_dummy_value_from_value_list(self):
        for dset in self.value_list:
            for s in self.value_list[dset]:
                if self.value_list[dset][s] == {"dummy": 1}:
                    self.value_list[dset][s] = {}

    def _merge_with_train_value_list(self, new_value_list):
        self._remove_dummy_value_from_value_list()
        for s in new_value_list:
            if s not in self.value_list["train"]:
                self.value_list["train"][s] = new_value_list[s]
            else:
                for v in new_value_list[s]:
                    if v not in self.value_list["train"][s]:
                        self.value_list["train"][s][v] = new_value_list[s][v]
                    else:
                        self.value_list["train"][s][v] += new_value_list[s][v]
        self._add_dummy_value_to_value_list()

    def _get_slot_list(self):
        raise NotImplementedError()

    def prediction_normalization(self, slot, value):
        return value

    def get_train_examples(self):
        raise NotImplementedError()

    def get_dev_examples(self):
        raise NotImplementedError()

    def get_test_examples(self):
        raise NotImplementedError()


class Woz2Processor(DataProcessor):
    def __init__(self, dataset_config, data_dir, not_domain=None):
        super(Woz2Processor, self).__init__(dataset_config, data_dir)
        self.value_list["train"] = dataset_woz2.get_value_list(
            os.path.join(self.data_dir, "woz_train_en.json"), self.slot_list
        )
        self.value_list["dev"] = dataset_woz2.get_value_list(
            os.path.join(self.data_dir, "woz_validate_en.json"), self.slot_list
        )
        self.value_list["test"] = dataset_woz2.get_value_list(
            os.path.join(self.data_dir, "woz_test_en.json"), self.slot_list
        )

    def get_train_examples(self, args):
        return dataset_woz2.create_examples(
            os.path.join(self.data_dir, "woz_train_en.json"), "train", self.slot_list, self.label_maps, **args
        )

    def get_dev_examples(self, args):
        return dataset_woz2.create_examples(
            os.path.join(self.data_dir, "woz_validate_en.json"), "dev", self.slot_list, self.label_maps, **args
        )

    def get_test_examples(self, args):
        return dataset_woz2.create_examples(
            os.path.join(self.data_dir, "woz_test_en.json"), "test", self.slot_list, self.label_maps, **args
        )


class Multiwoz21Processor(DataProcessor):
    def __init__(self, dataset_config, data_dir, not_domain=None):
        super(Multiwoz21Processor, self).__init__(dataset_config, data_dir)
        self.not_domain = not_domain
        self.value_list["train"] = dataset_multiwoz21.get_value_list(
            os.path.join(self.data_dir, "train_dials.json"), self.slot_list, self.not_domain
        )
        self.value_list["dev"] = dataset_multiwoz21.get_value_list(
            os.path.join(self.data_dir, "val_dials.json"), self.slot_list
        )
        self.value_list["test"] = dataset_multiwoz21.get_value_list(
            os.path.join(self.data_dir, "test_dials.json"), self.slot_list
        )
        self._add_dummy_value_to_value_list()

    def prediction_normalization(self, slot, value):
        return dataset_multiwoz21.prediction_normalization(slot, value)

    def get_train_examples(self, args):
        return dataset_multiwoz21.create_examples(
            os.path.join(self.data_dir, "train_dials.json"),
            "train",
            self.class_types,
            self.slot_list,
            self.label_maps,
            self.not_domain,
            **args,
        )

    def get_dev_examples(self, args):
        return dataset_multiwoz21.create_examples(
            os.path.join(self.data_dir, "val_dials.json"),
            "dev",
            self.class_types,
            self.slot_list,
            self.label_maps,
            **args,
        )

    def get_test_examples(self, args):
        return dataset_multiwoz21.create_examples(
            os.path.join(self.data_dir, "test_dials.json"),
            "test",
            self.class_types,
            self.slot_list,
            self.label_maps,
            **args,
        )


class Multiwoz21LegacyProcessor(DataProcessor):
    def __init__(self, dataset_config, data_dir, not_domain=None):
        super(Multiwoz21LegacyProcessor, self).__init__(dataset_config, data_dir)
        self.not_domain = not_domain
        self.value_list["train"] = dataset_multiwoz21_legacy.get_value_list(
            os.path.join(self.data_dir, "train_dials.json"), self.slot_list, self.not_domain
        )
        self.value_list["dev"] = dataset_multiwoz21_legacy.get_value_list(
            os.path.join(self.data_dir, "val_dials.json"), self.slot_list
        )
        self.value_list["test"] = dataset_multiwoz21_legacy.get_value_list(
            os.path.join(self.data_dir, "test_dials.json"), self.slot_list
        )
        self._add_dummy_value_to_value_list()

    def prediction_normalization(self, slot, value):
        return dataset_multiwoz21.prediction_normalization(slot, value)

    def get_train_examples(self, args):
        return dataset_multiwoz21_legacy.create_examples(
            os.path.join(self.data_dir, "train_dials.json"),
            os.path.join(self.data_dir, "dialogue_acts.json"),
            "train",
            self.slot_list,
            self.label_maps,
            self.not_domain,
            **args,
        )

    def get_dev_examples(self, args):
        return dataset_multiwoz21_legacy.create_examples(
            os.path.join(self.data_dir, "val_dials.json"),
            os.path.join(self.data_dir, "dialogue_acts.json"),
            "dev",
            self.slot_list,
            self.label_maps,
            **args,
        )

    def get_test_examples(self, args):
        return dataset_multiwoz21_legacy.create_examples(
            os.path.join(self.data_dir, "test_dials.json"),
            os.path.join(self.data_dir, "dialogue_acts.json"),
            "test",
            self.slot_list,
            self.label_maps,
            **args,
        )


class SimProcessor(DataProcessor):
    def __init__(self, dataset_config, data_dir, not_domain=None):
        super(SimProcessor, self).__init__(dataset_config, data_dir)
        self.value_list["train"] = dataset_sim.get_value_list(os.path.join(self.data_dir, "train.json"), self.slot_list)
        self.value_list["dev"] = dataset_sim.get_value_list(os.path.join(self.data_dir, "dev.json"), self.slot_list)
        self.value_list["test"] = dataset_sim.get_value_list(os.path.join(self.data_dir, "test.json"), self.slot_list)

    def get_train_examples(self, args):
        return dataset_sim.create_examples(os.path.join(self.data_dir, "train.json"), "train", self.slot_list, **args)

    def get_dev_examples(self, args):
        return dataset_sim.create_examples(os.path.join(self.data_dir, "dev.json"), "dev", self.slot_list, **args)

    def get_test_examples(self, args):
        return dataset_sim.create_examples(os.path.join(self.data_dir, "test.json"), "test", self.slot_list, **args)


class UnifiedDatasetProcessor(DataProcessor):
    def __init__(self, dataset_config, data_dir, not_domain=None):
        super(UnifiedDatasetProcessor, self).__init__(dataset_config, data_dir)
        self.not_domain = not_domain
        self.value_list["train"] = dataset_unified.get_value_list(self.dataset_name, self.slot_list)
        self.value_list["dev"] = dataset_unified.get_value_list(self.dataset_name, self.slot_list)
        self.value_list["test"] = dataset_unified.get_value_list(self.dataset_name, self.slot_list)
        self._add_dummy_value_to_value_list()

    def prediction_normalization(self, slot, value):
        return dataset_unified.prediction_normalization(self.dataset_name, slot, value)

    def _get_slot_list(self):
        return dataset_unified.get_slot_list(self.dataset_name)

    def get_train_examples(self, args):
        return dataset_unified.create_examples(
            set_type="train",
            dataset_name=self.dataset_name,
            class_types=self.class_types,
            slot_list=self.slot_list,
            label_maps=self.label_maps,
            # NOTE: We needed to comment out this additional argument
            # self.not_domain,
            **args,
        )

    def get_dev_examples(self, args):
        return dataset_unified.create_examples(
            "validation", self.dataset_name, self.class_types, self.slot_list, self.label_maps, **args
        )

    def get_test_examples(self, args):
        return dataset_unified.create_examples(
            "test", self.dataset_name, self.class_types, self.slot_list, self.label_maps, **args
        )


PROCESSORS = {
    "woz2": Woz2Processor,
    "sim-m": SimProcessor,
    "sim-r": SimProcessor,
    "multiwoz21": Multiwoz21Processor,
    "multiwoz21_legacy": Multiwoz21LegacyProcessor,
    "unified": UnifiedDatasetProcessor,
}
