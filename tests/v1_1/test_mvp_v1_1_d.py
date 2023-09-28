# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from nemoguardrails import RailsConfig
from tests.utils import TestChat

CONFIGS_FOLDER = os.path.join(os.path.dirname(__file__), "../test_configs")


def test_1():
    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "mvp_v1_1_d"))

    chat = TestChat(
        config,
        llm_completions=[],
    )

    chat >> "hi"
    chat << "Hello world!"


def test_2():
    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "mvp_v1_1_d"))

    chat = TestChat(
        config,
        llm_completions=[],
    )

    chat >> "do something"
    (
        chat << "Why don't scientists trust atoms?"
        " Because they make up everything!\n"
        "I'm smiling."
    )
