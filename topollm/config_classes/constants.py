# coding=utf-8
#
# Copyright 2024
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Ruppik (ruppik@hhu.de)
# Julius von Rohrscheidt (julius.rohrscheidt@helmholtz-muenchen.de)
#
# Code generation tools and workflows:
# First versions of this code were potentially generated
# with the help of AI writing assistants including
# GitHub Copilot, ChatGPT, Microsoft Copilot, Google Gemini.
# Afterwards, the generated segments were manually reviewed and edited.
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

"""
Script for setting global variables for the config files.
"""


# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Imports

# System imports

# Third-party imports


# END Imports
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Globals

# This dictionary of prefixes allows us to
# easily change the prefixes for file paths and names
# in one place without modifying the functions itself,
# enhancing the maintainability of the code.
NAME_PREFIXES: dict[
    str,
    str,
] = {
    "aggregation": "agg-",
    "add_prefix_space": "add-prefix-space-",
    "center": "center-",
    "context": "ctxt-",
    "data": "data-",
    "dataloader_desc": "dataloader-",
    "epoch": "ep-",
    "global_step": "gs-",
    "label_map_description": "labelmap-",
    "layer": "layer-",
    "level": "lvl-",
    "metric": "metric-",
    "model": "model-",
    "model_parameters": "mparam-",
    "masking_mode": "mask-",
    "max_length": "max-len-",
    "normalization": "norm-",
    "n_neighbors": "n-neighbors-",
    "query": "query-",
    "transformation": "trans-",
    "use_canonical_values_from_dataset": "use-canonical-val-",
}


# END Globals
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
