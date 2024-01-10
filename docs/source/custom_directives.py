# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import os
from pathlib import Path
from typing import List
from urllib.parse import quote, urlencode

import requests
from docutils import nodes
from docutils.parsers.rst import Directive, directives
from docutils.parsers.rst.directives.images import Image
from docutils.statemachine import StringList
from sphinx.util.docutils import SphinxDirective

_THIS_DIR = Path(__file__).parent

# Color palette from PyTorch Developer Day 2021 Presentation Template
YELLOW = "F9DB78"
GREEN = "70AD47"
BLUE = "00B0F0"
PINK = "FF71DA"
ORANGE = "FF8300"
TEAL = "00E5D1"
GRAY = "7F7F7F"


def _get_cache_path(key, ext):
    filename = f"{hashlib.sha256(key).hexdigest()}{ext}"
    cache_dir = _THIS_DIR / "gen_images"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / filename


def _download(url, path):
    response = requests.get(url)
    response.raise_for_status()
    with open(path, "wb") as file:
        file.write(response.content)


def _fetch_image(url):
    path = _get_cache_path(url.encode("utf-8"), ext=".svg")
    if not path.exists():
        _download(url, path)
    return os.sep + str(path.relative_to(_THIS_DIR))


def _get_relpath(target, base):
    target = os.sep + target
    base = os.sep + base
    target_path, filename = os.path.split(target)
    rel_path = os.path.relpath(target_path, os.path.dirname(base))
    return os.path.normpath(os.path.join(rel_path, filename))


class BaseShield(Image, SphinxDirective):
    def run(self, params, alt, section) -> List[nodes.Node]:
        url = f"https://img.shields.io/static/v1?{urlencode(params, quote_via=quote)}"
        path = _fetch_image(url)
        self.arguments = [path]
        self.options["alt"] = alt
        if "class" not in self.options:
            self.options["class"] = []
        self.options["class"].append("shield-badge")
        target = _get_relpath("supported_features.html", self.env.docname)
        self.options["target"] = f"{target}#{section}"
        return super().run()


def _parse_devices(arg: str):
    devices = sorted(arg.strip().split())

    valid_values = {"CPU", "CUDA"}
    if any(val not in valid_values for val in devices):
        raise ValueError(
            f"One or more device values are not valid. The valid values are {valid_values}. Given value: '{arg}'"
        )
    return ", ".join(sorted(devices))


def _parse_properties(arg: str):
    properties = sorted(arg.strip().split())

    valid_values = {"Autograd", "TorchScript"}
    if any(val not in valid_values for val in properties):
        raise ValueError(
            "One or more property values are not valid. "
            f"The valid values are {valid_values}. "
            f"Given value: '{arg}'"
        )
    return ", ".join(sorted(properties))


class SupportedDevices(BaseShield):
    """List the supported devices"""

    required_arguments = 1
    final_argument_whitespace = True

    def run(self) -> List[nodes.Node]:
        devices = _parse_devices(self.arguments[0])
        alt = f"This feature supports the following devices: {devices}"
        params = {
            "label": "Devices",
            "message": devices,
            "labelColor": GRAY,
            "color": BLUE,
            "style": "flat-square",
        }
        return super().run(params, alt, "devices")


class SupportedProperties(BaseShield):
    """List the supported properties"""

    required_arguments = 1
    final_argument_whitespace = True

    def run(self) -> List[nodes.Node]:
        properties = _parse_properties(self.arguments[0])
        alt = f"This API supports the following properties: {properties}"
        params = {
            "label": "Properties",
            "message": properties,
            "labelColor": GRAY,
            "color": GREEN,
            "style": "flat-square",
        }
        return super().run(params, alt, "properties")


_CARDLIST_START = """
.. raw:: html

   <div id="tutorial-cards-container">
     <nav class="navbar navbar-expand-lg navbar-light tutorials-nav col-12">
       <div class="tutorial-tags-container">
         <div id="dropdown-filter-tags">
           <div class="tutorial-filter-menu">
             <div class="tutorial-filter filter-btn all-tag-selected" data-tag="all">All</div>
           </div>
         </div>
       </div>
     </nav>

     <hr class="tutorials-hr">

     <div class="row">
       <div id="tutorial-cards">
         <div class="list">
"""

_CARD_TEMPLATE = """
.. raw:: html

   <div class="col-md-12 tutorials-card-container" data-tags={tags}>
     <div class="card tutorials-card">
       <a href="{link}">
         <div class="card-body">
           <div class="card-title-container">
             <h4>{header}</h4>
           </div>
           <p>Topics: <span class="tags">{tags}</span></p>
           <p class="card-summary">{card_description}</p>
           <div class="tutorials-image">{image}</div>
         </div>
       </a>
     </div>
   </div>
"""

_CARDLIST_END = """
.. raw:: html

         </div>
         <div class="pagination d-flex justify-content-center"></div>
       </div>
     </div>
   </div>
"""


class CustomCardStart(Directive):
    def run(self):
        para = nodes.paragraph()
        self.state.nested_parse(
            StringList(_CARDLIST_START.split("\n")), self.content_offset, para
        )
        return [para]


class CustomCardItem(Directive):
    option_spec = {
        "header": directives.unchanged,
        "image": directives.unchanged,
        "link": directives.unchanged,
        "card_description": directives.unchanged,
        "tags": directives.unchanged,
    }

    def run(self):
        for key in ["header", "card_description", "link"]:
            if key not in self.options:
                raise ValueError(f"Key: `{key}` is missing")

        header = self.options["header"]
        link = self.options["link"]
        card_description = self.options["card_description"]
        tags = self.options.get("tags", "")

        if "image" in self.options:
            image = "<img src='" + self.options["image"] + "'>"
        else:
            image = "_static/img/thumbnails/default.png"

        card_rst = _CARD_TEMPLATE.format(
            header=header,
            image=image,
            link=link,
            card_description=card_description,
            tags=tags,
        )
        card_list = StringList(card_rst.split("\n"))
        card = nodes.paragraph()
        self.state.nested_parse(card_list, self.content_offset, card)
        return [card]


class CustomCardEnd(Directive):
    def run(self):
        para = nodes.paragraph()
        self.state.nested_parse(
            StringList(_CARDLIST_END.split("\n")), self.content_offset, para
        )
        return [para]
