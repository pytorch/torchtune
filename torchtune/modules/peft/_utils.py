# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
from typing import Any, Dict, Generator, List, Literal, Optional, Protocol, Set

import torch
from torch import nn

# Modules from MultiHeadAttention that LoRA can be applied to
LORA_ATTN_MODULES = Literal["q_proj", "k_proj", "v_proj", "output_proj"]


class AdapterModule(Protocol):
    """
    Interface for an ``nn.Module`` containing adapter weights.
    Note that an adapter module does not have to explicitly implement this protocol,
    but it must define the ``adapter_params(self)`` method.
    """

    def adapter_params(self) -> List[str]:
        """
        Return a list of strings corresponding to the names of the ``nn.Parameter`` s in
        the model coming from the adapter.
        E.g. if an nn.Module has adapter ``self.proj = nn.Linear(in_dim, out_dim)``,
        then adapter_params should return ``['proj.weight', 'proj.bias']``.

        See LoRALinear's :func:`~torchtune.modules.peft.LoRALinear.adapter_params` for an example.
        """
        pass


def get_adapter_params(model: nn.Module) -> Dict[str, nn.Parameter]:
    """
    Return the subset of parameters from a model that correspond to an adapter.
    Assumes that any adapter class has defined the
    :func:`~torchtune.modules.peft.AdapterModule.adapter_params` method.

    Args:
        model (nn.Module): Instance of model class containing some adapter params.

    Returns:
        Dict[str, nn.Parameter]: the subset of model's state dict containing
        only adapter parameters.

    """
    adapter_params = {}
    for k, v in model.named_modules():
        if hasattr(v, "adapter_params") and callable(v.adapter_params):
            current_adapter_params = v.adapter_params()
            for n, p in v.named_parameters(recurse=True):
                if n in current_adapter_params:
                    full_key = f"{k}.{n}" if k else n
                    adapter_params.update({full_key: p})
                    current_adapter_params.remove(n)
            assert (
                current_adapter_params == []
            ), f"Adapter params {current_adapter_params} not converted"
    return adapter_params


def set_trainable_params(model: nn.Module, adapter_params: Dict[str, Any]) -> None:
    """
    Set trainable parameters for an nn.Module based on a state dict of adapter parameters.

    Args:
        model (nn.Module): Instance of model class containing some adapter params.
        adapter_params (Dict[str, Any]): State dict mapping adapter key names to their
            respective nn.Parameters (i.e. outputs of :func:`~torchtune.modules.peft.get_adapter_params`.)

    Returns:
        None
    """
    for k, v in model.named_parameters():
        v.requires_grad_(k in adapter_params)


def get_lora_module_names(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool,
    apply_lora_to_output: bool,
) -> List[str]:
    """
    Return a list of the names of modules in the model that have LoRA applied. Note that
    the names here are local to their modules and not the fully qualified names from the
    model state dict.


    Args:
        lora_attn_modules (List[LORA_ATTN_MODULES]): list of which linear layers
            LoRA should be applied to in each self-attention block. Options are
            ``{"q_proj", "k_proj", "v_proj", "output_proj"}``.
        apply_lora_to_mlp (bool): whether LoRA is applied to each MLP linear.
        apply_lora_to_output (bool): whether LoRA is applied to the final output projection.

    Returns:
        List[str]: list of module names in the model that have LoRA applied.
    """
    lora_module_keys = lora_attn_modules
    if apply_lora_to_mlp:
        lora_module_keys = lora_module_keys + ["w1", "w2", "w3"]
    if apply_lora_to_output:
        lora_module_keys.append("output")
    return lora_module_keys


def validate_state_dict_for_lora(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool,
    apply_lora_to_output: bool,
    full_model_state_dict_keys: List[str],
    lora_state_dict_keys: Optional[List[str]] = None,
    base_model_state_dict_keys: Optional[List[str]] = None,
) -> None:
    """
    Validate that the state dict keys for a LoRA model are as expected.

    (1) If lora_state_dict_keys are passed, this function will confirm that they match exactly the
        LoRA param names from the full model (as determined by lora_modules).
    (2) If base_model_state_dict_keys are passed, this function will confirm that they are exactly the
        complement of the LoRA param names from the full model.
    (3) If both lora_state_dict_keys and base_model_state_dict_keys are passed, this function will
        confirm that the full model's params are exactly their disjoint union.

    Args:
        lora_attn_modules (List[LORA_ATTN_MODULES]): list of which linear layers
            LoRA should be applied to in each self-attention block. Options are
            ``{"q_proj", "k_proj", "v_proj", "output_proj"}``.
        apply_lora_to_mlp (bool): whether LoRA is applied to each MLP linear.
        apply_lora_to_output (bool): whether LoRA is applied to the final output projection.
        full_model_state_dict_keys (List[str]): List of keys in the full model state dict.
        lora_state_dict_keys (Optional[List[str]]): List of keys in the LoRA state dict.
            If none, LoRA state dict keys will not be validated.
        base_model_state_dict_keys (Optional[List[str]]): List of keys in the base model state dict.
            If none, base model keys will not be validated.

    Returns:
        None

    Raises:
        AssertionError: If base model state dict is missing any non-LoRA params from the full model.
        AssertionError: If LoRA state dict is missing any LoRA params from the full model.
        AssertionError: If base model state dict has any LoRA params.
        AssertionError: If LoRA state dict has any non-LoRA params.
        AssertionError: If base model and LoRA state dicts have overlapping keys.
        AssertionError: If full model state dict is missing keys from either base model or LoRA state dict.

    """
    lora_modules = get_lora_module_names(
        lora_attn_modules, apply_lora_to_mlp, apply_lora_to_output
    )
    is_lora_param = lambda x: any(
        [
            ".".join([k, "lora"]) in x or ".".join([k, "magnitude"]) in x
            for k in lora_modules
        ]
    )
    for k in full_model_state_dict_keys:
        if not is_lora_param(k):
            if base_model_state_dict_keys is not None:
                if k not in base_model_state_dict_keys:
                    raise AssertionError(
                        f"Missing non-LoRA key {k} from base model state dict"
                    )
            if lora_state_dict_keys is not None:
                if k in lora_state_dict_keys:
                    raise AssertionError(f"Non-LoRA key {k} found in LoRA state dict")
        else:
            if base_model_state_dict_keys is not None:
                if k in base_model_state_dict_keys:
                    raise AssertionError(f"LoRA key {k} found in base model state dict")
            if lora_state_dict_keys is not None:
                if k not in lora_state_dict_keys:
                    raise AssertionError(f"Missing LoRA key {k} From LoRA state dict")

    # Full model is disjoint union of base model and LoRA weights
    if lora_state_dict_keys is not None and base_model_state_dict_keys is not None:
        combined_state_dict_keys = set(lora_state_dict_keys).union(
            base_model_state_dict_keys
        )
        shared_state_dict_keys = set(lora_state_dict_keys).intersection(
            base_model_state_dict_keys
        )
        assert (
            shared_state_dict_keys == set()
        ), "Base model and LoRA state dict have overlapping keys"
        assert combined_state_dict_keys == set(
            full_model_state_dict_keys
        ), "Extra keys not present in full model"


def _get_lora_modules(state_dict: Dict[str, Any]) -> Set[str]:
    """
    Get the keys from a state dict that correspond to LoRALinear modules.

    For example, if state_dict is the state dict of model and model.x.y.z is a
    LoRALinear, this method will return "model.x.y.z", not
    "model.x.y.z.lora_a.weight" or "model.x.y.z.lora_b.weight".

    Args:
        state_dict (Dict[str, Any]): State dict from a model.

    Returns:
        Set[str]: Set of keys in the state dict that correspond to LoRA modules.
    """
    lora_keys = [k for k in state_dict.keys() if "lora" in k or "magnitude" in k]
    return set(
        [
            k.replace(".lora_a.weight", "")
            .replace(".lora_b.weight", "")
            .replace(".magnitude", "")
            for k in lora_keys
        ]
    )


@torch.no_grad
def get_merged_lora_ckpt(
    state_dict: Dict[str, Any],
    rank: int,
    alpha: float,
) -> Dict[str, Any]:
    """
    Merge LoRA weights into the base model format for efficient inference.
    NOTE: This function modifies state_dict inplace. If you do not want to do that,
    make a copy prior to calling this function.

    For every LoRA module in the state dict, this function will convert its
    base weight then delete the LoRA-specific parameters.

    Args:
        state_dict (Dict[str, Any]): State dict from a model.
        rank (int): The rank of LoRA matrices.
        alpha (float): The alpha value used for scaling LoRA decompositions.

    Returns:
        Dict[str, Any]: The merged state dict.
    """
    lora_modules = _get_lora_modules(state_dict)
    for module in lora_modules:
        lora_a_weight = state_dict[f"{module}.lora_a.weight"]
        lora_b_weight = state_dict[f"{module}.lora_b.weight"]
        lora_magnitude = state_dict.get(f"{module}.magnitude", None)

        # If magnitude is present, calculate merged DoRA weight
        if lora_magnitude is not None:
            base_weight = state_dict[f"{module}.weight"].to(lora_a_weight.dtype)

            lora_weight = (alpha / rank) * lora_b_weight @ lora_a_weight
            merged_weight = base_weight + lora_weight
            weight_norm = torch.linalg.norm(base_weight + lora_weight, dim=1)
            mag_norm_scale = (lora_magnitude / weight_norm).view(-1, 1)
            merged_weight *= mag_norm_scale
            state_dict[f"{module}.weight"] = merged_weight
            del state_dict[f"{module}.magnitude"]

        # Otherwise it is just vanilla LoRA
        else:
            state_dict[f"{module}.weight"] += (
                (alpha / rank) * lora_b_weight @ lora_a_weight
            )

        del state_dict[f"{module}.lora_a.weight"]
        del state_dict[f"{module}.lora_b.weight"]

    return state_dict


@contextlib.contextmanager
def disable_adapter(model: nn.Module) -> Generator[None, None, None]:
    """
    Temporarily disable the adapters in a model. For example,
    this can be used in DPO for treating the LoRA adapters as the policy model
    and disabling it to treat the base model as the reference model.

    This context manager goes through all modules in the provided neural network model,
    and if a module has an ``adapter_params`` attribute that is callable and a ``disabled`` attribute,
    it sets ``disabled`` to True. Then, the control is given back to caller. When exiting the context manager,
    it sets ``disabled`` back to False for all modules that were temporarily disabled.

    Args:
        model (nn.Module): The model whose adapters are to be temporarily disabled.
    Yields:
        None: This function yields control back to the caller, with the adapters disabled.
    Example:
        >>> with disable_adapter(model):
        ...     # Perform operations with adapters disabled
        ...     pass

    """
    for _, module in model.named_modules():
        if (
            hasattr(module, "adapter_params")
            and callable(module.adapter_params)
            and hasattr(module, "disabled")
        ):
            module.disabled = True
    try:
        yield
    finally:
        for _, module in model.named_modules():
            if (
                hasattr(module, "adapter_params")
                and callable(module.adapter_params)
                and hasattr(module, "disabled")
            ):
                module.disabled = False


def validate_missing_and_unexpected_for_lora(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool,
    apply_lora_to_output: bool,
    base_missing: Optional[List[str]] = None,
    base_unexpected: Optional[List[str]] = None,
    lora_missing: Optional[List[str]] = None,
    lora_unexpected: Optional[List[str]] = None,
) -> None:
    """
    A more memory-efficient way to validate that LoRA state dict loading was done properly.

    Similar to :func:`validate_state_dict_for_lora`, this function uses a model's LoRA config to
    check that LoRA and/or base model weights are loaded into the full model correctly.
    Unlike that function, this method relies only on the values of missing and unexpected
    as returned by the load_state_dict API with strict=False. This allows us to do the
    validation without any additional calls to .state_dict(), which use additional memory.

    Args:
        lora_attn_modules (List[LORA_ATTN_MODULES]): list of which linear layers
            LoRA should be applied to in each self-attention block. Options are
            ``{"q_proj", "k_proj", "v_proj", "output_proj"}``.
        apply_lora_to_mlp (bool): whether LoRA is applied to each MLP linear.
        apply_lora_to_output (bool): whether LoRA is applied to the final output projection.
        base_missing (Optional[List[str]]): List of missing keys when loading base model weights.
            Default: None
        base_unexpected (Optional[List[str]]): List of unexpected keys when loading base model weights.
            Default: None
        lora_missing (Optional[List[str]]): List of missing keys when loading LoRA weights.
            Default: None
        lora_unexpected (Optional[List[str]]): List of unexpected keys when loading LoRA weights.
            Default: None

    Returns:
        None

    Raises:
        AssertionError: if base_missing contains any base model keys.
        AssertionError: if base_unexpected is nonempty.
        AssertionError: if lora_missing contains any LoRA keys.
        AssertionError: if lora_unexpected is nonempty.
    """
    lora_modules = get_lora_module_names(
        lora_attn_modules, apply_lora_to_mlp, apply_lora_to_output
    )
    is_lora_param = lambda x: any(
        [
            ".".join([k, "lora"]) in x or ".".join([k, "magnitude"]) in x
            for k in lora_modules
        ]
    )

    if base_missing:
        for k in base_missing:
            if not is_lora_param(k):
                raise AssertionError(f"Missing non-LoRA key {k} from base model dict")
    if base_unexpected:
        raise AssertionError("Unexpected key loading base model")
    if lora_missing:
        for k in lora_missing:
            if is_lora_param(k):
                raise AssertionError(f"Missing LoRA key {k} from adapter state dict")
    if lora_unexpected:
        raise AssertionError("Unexpected key loading adapter")


def load_dora_magnitudes(model: nn.Module) -> None:
    """
    For DoRA magnitude we use setattr to move from meta device
    """
    dora_parents = {
        n: p for n, p in model.named_modules() if hasattr(p, "adapter_params")
    }
    sd = {f"{n}.magnitude": p.magnitude for n, p in dora_parents.items()}
    model.load_state_dict(sd, strict=False, assign=True)
