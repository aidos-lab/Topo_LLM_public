# coding=utf-8
#
# Copyright 2024
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Ruppik (ruppik@hhu.de)
#
# This code was generated with the help of AI writing assistants
# including GitHub Copilot, ChatGPT, Bing Chat.
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

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Imports

# System imports
import os

# Third-party imports
from git import Repo

# END Imports
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def get_git_info() -> str:
    """Get the git info of the current branch and commit hash"""
    repo = Repo(
        os.path.dirname(os.path.realpath(__file__)),
        search_parent_directories=True,
    )
    branch_name = repo.active_branch.name
    commit_hex = repo.head.object.hexsha

    info = f"{branch_name}/{commit_hex}"
    return info
