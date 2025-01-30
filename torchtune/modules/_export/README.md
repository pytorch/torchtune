# Export

This directory provides [exportable](https://pytorch.org/docs/stable/export.html) variants of torchtune modules.

Modules in this directory:

* Take the same arguments to `__init__()` and `forward()` as the corresponding reference modules in torchtune.
* Give the output as the reference module in torchtune (unless stated otherwise in the docstring).
* Are guaranteed to work out of the box with torch.export.export().
* Should work out of the box with torch.aot_compile().

All modules should be covered by unit tests (under `tests/torchtune/modules/_export/`) that runs daily and on PRs touching this directory.

These modules are subject to change so proceed with caution.

Contributors: @larryliu0820, @Jack-Khuu, @dvorjackz
