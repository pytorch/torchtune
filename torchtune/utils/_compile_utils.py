import torch

def wrap_compile(model):
    """
    Wraps the model with torch.compile. This function will also
	register a state_dict post hook that allows state_dicts produced
	with torch.compile training to behave as regular eager mode models.
	In particular, it strips away a torch.compile specific prefix
	added to the state_dict by torch.compile.

	Args:
		model (nn.Module): model to wrap with compile.
    """
    model = torch.compile(model)
    model._register_state_dict_hook(_consume_torch_compile_prefix)
    return model


def _consume_torch_compile_prefix(model, state_dict, *args, prefix="_orig_mod.", **kwargs):
    keys = list(state_dict.keys())
    for key in keys:
        if key.startswith(prefix):
            newkey = key[len(prefix) :]
            state_dict[newkey] = state_dict.pop(key)
