# Copyright 2024
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Matthias Ruppik (mail@ruppik.net)
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

"""Get default prompts for masked and causal language modeling."""


def get_default_mlm_prompts(
    mask_token: str,
) -> list[str]:
    """Get default masked language model prompts."""
    prompts = [
        f"I am looking for a {mask_token}.",
        f"I am looking for a {mask_token}, can you help me?",
        f"Can you find me a {mask_token}?",
        f"I would like a {mask_token} hotel in the center of town, please.",
        f"{mask_token} is a cheap restaurant in the south of town.",
        f"The train should go to {mask_token}.",
        f"No, it should be {mask_token}, look again!",
        f"{mask_token} is a city in the south of England.",
        f"{mask_token} is the best city in the world.",
        f"I would like to invest in {mask_token}.",
        f"What is the best {mask_token} in town?",
        f"Can you recommend a good {mask_token}?",
    ]

    return prompts


def get_default_clm_prompts() -> list[str]:
    """Get default causal language model prompts."""
    prompts = [
        "I am looking for a",
        "I am looking for a ",  # with space at the end
        "Can you find me a",
        "Can you find me a ",  # with space at the end
        "I would like a hotel in the",
        "Nandos is a",
        "The train should go to",
        "No, it should be",
        "Cambridge is",
        "I would like to invest in",
        "What is the best",
        "Can you recommend a good",
    ]

    return prompts
