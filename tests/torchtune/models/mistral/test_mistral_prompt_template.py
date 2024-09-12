# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from tests.test_utils import assert_dialogue_equal, MESSAGE_SAMPLE
from torchtune.data import Message
from torchtune.models.mistral import MistralChatTemplate


class TestMistralChatTemplate:
    expected_dialogue = [
        Message(
            role="user",
            content="[INST] Please briefly summarize this news article:\n\nAOL.com Video - Father Lets 8-Year-Old "
            "Drive On Icy Road\n\nDescription:Would you let your 8-year-old drive your car? "
            "How about on an icy road? Well one father in Russia did just that, and recorded "
            "the entire thing. To her credit, the child seemed to be doing a great job. "
            "(0:44)\n\nTags: 8-year-old driver , caught on camera , child driver , pix11\n\n"
            "Summary: [/INST] ",
        ),
        Message(
            role="assistant",
            content="A father in Russia allowed his 8-year-old child to drive his car on an "
            "icy road and recorded the event. The child appeared to be handling the situation well, "
            "showcasing their driving skills despite the challenging conditions.",
        ),
    ]

    def test_format(self):
        no_system_sample = MESSAGE_SAMPLE[1:]
        actual = MistralChatTemplate()(no_system_sample)
        assert_dialogue_equal(actual, self.expected_dialogue)

    def test_format_with_system_prompt_raises(self):
        with pytest.raises(
            ValueError, match="System prompts are not supported in MistralChatTemplate"
        ):
            _ = MistralChatTemplate()(MESSAGE_SAMPLE)
