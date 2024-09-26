# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Optional


class InstructTemplate(ABC):
    """
    Warning:
        This class is deprecated and will be removed in a future release. Please use
        :class:`~torchtune.data.PromptTemplate` for custom instruct templates.

    Interface for instruction templates. Each template should include the template
    prompt with placeholders for the data inputs.
    """

    template = ""

    @classmethod
    @abstractmethod
    def format(
        cls, sample: Mapping[str, Any], column_map: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Format the prompt template with the given arguments.

        Args:
            sample (Mapping[str, Any]): a single data sample with various fields
            column_map (Optional[Dict[str, str]]): a mapping from the expected
                placeholder names in the template to the column names in the sample.
                If None, assume these are identical. Note: if the sample output is not named
                as "output" in the dataset, you always need to map it to "output" in column_map.

        Returns:
            The formatted prompt
        """
        pass
