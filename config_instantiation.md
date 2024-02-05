# [RFC] Instantiating objects from configs
Once we’ve correctly parsed user arguments from the config and command line,
we need to be able to create the appropriate TorchTune objects that the user
specified. These are usually models, tokenizers, datasets, optimizers, and other
core components necessary to run a fine-tuning script. In designing a system
that performs a mapping from a user-friendly string to one of these TorchTune
objects, we aim to optimize the user experience. Thus, there are two primary
requirements for this system:

1. It should minimize levels of abstractions between the user input and the
actual TorchTune objects. The more code in between the user and the desired
object, the harder it will be to understand and debug recipes.
2. It needs to scale to a large number of contributors to the library and a
large number of TorchTune objects. In this sense, we must consider memory
overhead, lookup overhead, and the dev experience.

With those constraints in mind, let’s briefly overview several common solutions
for this problem.

## Current approach
The codebase currently uses a combination of hardcoded dictionaries and getter
functions that call the dictionary and returns the requested object. This
approach does not scale with a large number of models, datasets, and any small
customizations to models that a user may need since they will have to manually
keep these dictionaries updated. It is also more error prone since it is easy to
miss the extra locations to update with a new TorchTune object.

```
ALL_MODELS = {"llama2_7b": llama2_7b}

def get_model(name: str, device: Union[str, torch.device], **kwargs) -> Module:
    """Get known supported models by name"""
    if name in ALL_MODELS:
        with get_device(device):
            model = ALL_MODELS[name](**kwargs)
        return model
    else:
        raise ValueError(
            f"Model not recognized. Expected one of {ALL_MODELS}, received {name}"
        )
```

## Registry
A registry is a central repository that holds all the possible TorchTune objects
that a user can specify, usually defined as a class. It does not store this
information explicitly hardcoded, instead we use decorators on classes and
functions to “register” them with a custom string during runtime.

```
class TuneRegistry:
    def __init__(self):
        self._models = {}
        self._tokenizers = {}
	def register_model(self, name, model_class):
        def decorator(model_class):
            self._models[name] = model_class
        return decorator
    def register_tokenizer(self, name, tokenizer_class):
        ...
    def get_model(self, model_name):
        if model_name not in self._models:
           	raise ValueError("Unknown model name")
       	return self._model[model_name]()
```

While this eliminates the need for hardcoded dictionaries, the mappings are
still created at runtime and incurs memory overhead. More importantly, any
changes to this central registry will require refactoring all recipes and
tightly couples many components.

## Hydra
Hydra allows configs to directly link with a Python object with the use of
`_target_` and instantiated via `hydra.utils.instantiate`. It is also simple to
enable some configurable parameters directly in the config.

```
model:
  _target_: torchtune.models.llama2_7b
  max_batch_size: 4
```

This approach provides the least abstraction between the user specified
TorchTune object and the object itself. However, the fact that Hydra is no
longer maintained is a hard blocker for adopting this approach. Moreover, this
approach may easily snowball into “coding via config” where configs become
highly verbose and replace operations that should be kept in python code.

## `getattr()`
This method is interesting because it also works on imported modules, which
means we can directly map a user provided string to the TorchTune object without
an explicit dictionary as long as it’s present in one of the library’s modules.
In fact, this is already done in the codebase with get_loss and get_optimizer.
One drawback is that it is highly dependent on what is in the namespace of the
module (which can include non-private imports other than classes and methods),
and it is non-intuitive that this needs to be updated when adding a TorchTune
object (for example, updating the __all__ in __init__.py). On the other hand,
this needs to be updated anyway for user-friendly imports of our components, so
it can serve a dual purpose. Outside of this, any new component that is added to
the libraries modules can automatically be retrieved via getattr() without
having to register it or update a dictionary.

```
from torch import nn

def get_loss(loss: str) -> nn.Module:
    try:
        return getattr(nn, loss)()
    except AttributeError as e:
        raise ValueError(f"{loss} is not a valid loss from torch.nn") from e
```

## Verdict
I contend that the `getattr()` approach achieves both the listed requirements of
being scalable with no additional memory overhead or code bloat and removing
one layer of abstraction between the config string and the actual object
compared to the current approach. As we gain more users and add significantly
more TorchTune objects, we can revisit this and consider a system like Hydra
where it’s easier to directly access these objects via the config. A quick
example of what this will look like is linked in the summary.
