# [RFC] Fused Models

**TLDR**
- Fused Models are two+ pre-trained models joined together and further tuned to work as one model. This is the approach used for most SOTA Multimodal models currently.
- This RFC proposes special modules and utils similar to our PEFT implementation to allow for easy manipulation of these fused model parameters and their state dicts.

**WHY**: To support a wide variety of multimodal models and multimodal recipes with a consistent single model API.
## Context
The majority of SOTA multimodal models coming out these days are not trained end to end as uniform models, but instead a separate model is trained for each modality and then these models are fused together. In the simplest case, you have an LLM trained on text and an image encoder such as CLIP. You then need to combine these pre-trained models and finetune them so that the LLM can understand the embeddings from the image encoder.

There are many different ways to combine them, and a detailed breakdown of these approaches can be found [here](https://arxiv.org/pdf/2405.17927). At a high level, there are two overarching architectures that most models can be grouped into. **EarlyFusion** and **DeepFusion**. **EarlyFusion** defines a model where a mapping is learned from encoder outputs to token embeddings. Then encoded inputs are converted to tokens and fed into the LLM the same as text tokens. On the other hand, in **DeepFusion**, the output of the encoder is injected directly into the intermediate layers of the LLM, typically with cross attention, following the same approach for encoder decoder language models.

To work with these models, there are several considerations. If you have two+ pre-trained models, you may want to be able to fuse them and add some additional learnable parameters (we will call **fusion parameters**). For this you want to be able to define the fused model but be able to load in the individual checkpoints. You also want to be able to determine which parameters/sub-modules to make trainable and which to keep frozen. Apart from directly fusing models, there are a lot of open models (like Llava, PaliGemma, Idefics, Phi Vision, etc) that have already been fused, but you might want to further finetune. In this case you want to be able to load in the full fused checkpoint and further finetuning on the models, either on all the weights or using PeFT techniques. To handle all of this, and keep uniform recipes/utils, it would be ideal if the fused models could behave as a single model but then easily decompose into their individual parts.

**TLDR**
- Multimodal models are primarily DeepFusion and EarlyFusion architectures
- Modules/utils for fusing and un-fusing MM models can allow for fine-tuning a variety of MM architectures in a uniform way.

## Proposal

**High Level List:**
- Introduce a modules/model_fusion folder for handling fused models
- fusion_models introduces two top level wrapper modules that combine a single encoder and decoder in two different ways. This allows the two models to be used as one model.
- fusion_layer and fusion_embed are fusion modules (similar to PeFT modules) that introduce extra trainable parameters to the pre-trained fused models
- fusion_utils is based on peft_utils and provides functions for accessing and modifying fusion parameters

**Folder Structure for new modules/functions**
```
torchtune/
	modules/
		model_fusion/
			fusion_models.py
				DeepFusionModel
				EarlyFusionModel
			fusion_layer.py
			fusion_embed.py
			fusion_utils.py
				register_fusion_module
				get_fusion_params
				set_trainable_params
```

**Pseudo code demonstrating usage of fused models**
```python
def flamingo_builder():
	return DeepFusionModel(
		encoder=CLIP(...),
		decoder=Llama3(...))

>>> model = flamingo_builder()

# Load full checkpoints
>>> model.load_state_dict(...)

# Or load pretrained individual models
>>> model.encoder.load_state_dict(...)
>>> model.decoder.load_state_dict(...)

# set only the fusion parameters to be trainable
>>> fusion_params = get_fusion_params(model)
>>> set_trainable_params(model, fusion_params)

# Do inference with the model the same as a text model
# Fusion models have the same signatures as TransformerDeocder
>>> data = flamingo_transform(...) # Message formated text + images
>>> model(**data)
tensor([-1.0885, ..., .0231], grad_fn=<ViewBackward0>)
```

> [!Note]
> This RFC proposes a complete design for supporting fusion models, but not all of the design is required to be built to support any one given fusion style model. This RFC gives us an opportunity to take a holistic view over the space, but it is recommended that each module/component is built as need.
### Fusion Models
DeepFusion requires that Decoder is already modified to have any additional layers for accepting encoder inputs (these should be wrapped with FusionLayer to be able to track these additional parameters and work with checkpoints). For inference all of the cache can be managed by the decoder, so these methods are simply passed through.

```python
class DeepFusionModel(nn.Module):
	def __init__(self,
		decoder: TransformerDecoder,
		encoder: nn.Module,
		...
	):

	def setup_caches(self, batch_size: int, dtype: torch.dtype) -> None:
        """Setup key value caches for attention calculation.

        Args:
            batch_size (int): batch size for the caches.
            dtype (torch.dtype): dtype for the caches.
        """
        self.decoder.setup_caches(batch_size, dtype)

	def caches_are_enabled(self) -> bool:
        """Check if the key value caches are setup."""
        return self.decoder.caches_are_enabled()

	def reset_caches(self):
        """Reset the key value caches."""
        self.decoder.reset_caches()

	def forward(
		self,
        tokens: Tensor,
        *,
        mask: Optional[Tensor] = None,
        encoder_input: Optional[Tensor] = None,
        encoder_mask: Optional[Tensor] = None,
        input_pos: Optional[Tensor] = None,
    ) -> Tensor:
	if 	encoder_input is not None:
		# during decoding, the output of the encoder
		# is cached in the decoder cross attention kv_cache.
		# so encoder_input is only called when there's a new image
		encoder_embed = self.encoder(**encoder_input)
	# input_pos is tracked by kv_cache now, but left in for
	# now to support sample packing and can be used here
	if input_pos is not None and encoder_mask is not None:
		encoder_mask = encoder_mask[input_pos]
	output = self.decoder(tokens, mask, encoder_embed, encoder_mask, input_pos)
	return output

 ```

EarlyFusion is more complicated from the fusion side, but does not require a special Decoder. EarlyFusion needs to call `decoder.tok_embeddings` directly so it can merge the embeddings with the encoded embeddings. As to not modify the `state_dict` it employs two hooks to keep `tok_embeddings` inside the decoder for the `state_dict`.

```python
class EarlyFusionModel(nn.Module):
	def __init__(self,
		decoder: TransformerDecoder,
		encoder: nn.Module,
		encoder_token: int,
	):
		self.decoder = decoder
		self.encoder = encoder
		self.encoder_token = encoder_token

		...

		# A little surgery in the decoder to give the
		# fusion module access to control the embeddings
		# The alternative is to pass a special tok_embeddings
		# module into TransformerDecoder builder that does the
		# merging there
		self.tok_embeddings = decoder.tok_embeddings
		decoder.tok_embeddings = nn.Identity()

		self.register_state_dict_post_hook(Test._state_dict_hook)
        # TODO: Switch to register_load_state_dict_pre_hook in v2.5
        self._register_load_state_dict_pre_hook(
	        Test._load_state_dict_hook,
	        with_module=True
	    )

	def _state_dict_hook(self, destination, prefix, keep_vars):
		"""
		Keep tok_embeddings inside of decoder state_dict

		[!Note] This update changes the order of the OrderedDict
		"""
		key = "tok_embeddings"
		decoder_key = "decoder.tok_embeddings"
		destination[decoder_key] = destination[key]
		del destination[key]

	def _load_state_dict_hook(self, state_dict, *args, **kwargs):
		""" Undo the change from _state_dict_hook """
		key = "tok_embeddings"
		decoder_key = "decoder.tok_embeddings"
		state_dict[key] = state_dict[decoder_key]
		del state_dict[decoder_key]

	def _merge_embeds(self, embeds, encoder_embeds, tok_mask):
		""" Combine embeds based on token mask """
		...

	def setup_caches(self, batch_size: int, dtype: torch.dtype) -> None:
        """Setup key value caches for attention calculation.

        Args:
            batch_size (int): batch size for the caches.
            dtype (torch.dtype): dtype for the caches.
        """
        self.decoder.setup_caches(batch_size, dtype)

	def caches_are_enabled(self) -> bool:
        """Check if the key value caches are setup."""
        return self.decoder.caches_are_enabled()

	def reset_caches(self):
        """Reset the key value caches."""
        self.decoder.reset_caches()

	def forward(
		self,
        tokens: Tensor,
        *,
        mask: Optional[Tensor] = None,
        encoder_input: Optional[Tensor] = None,
        input_pos: Optional[Tensor] = None,
        **kwargs, # no need for encoder_mask
    ) -> Tensor:

    if 	encoder_input is not None:
		encoder_embeds = self.encoder(**encoder_input)

	tok_mask = tok != self.encoder_token
	embeds = self.tok_embeddings(tokens[tok_mask])
	embeds = self._merge_embeds(embeds, encoder_embeds, tok_mask)

	output = self.decoder(tokens, mask, input_pos)
	return output

 ```


### Fusion Layer
This module's primary use case is for DeepFusion modules where you need to add new layers within the LLM decoder. Instead of adding additional layers directly, which would change layer counts and break checkpoint keys for the text model, FusionLayer fuses the new layer to an existing layer that it modifies. This fused layer is now treated as a single layer. The new parameters are returned with `fusion_params`. Two state dict hooks are used here so that the original layer doesn't have it's name modified in the `state_dict`.

```python

class FusionLayer(nn.Module):
    """Fusion layer as introduced in `Flamingo: a Visual Language Model for
    Few-Shot Learning <https://arxiv.org/abs/2204.14198>`_.

    Deep Fusion model architectures combine pretrained encoder models with
    pretrained language models by infusing the encoder outputs into the middle
    layers of the LLM. This allow the language model to interpret the encoder
    outputs as text, to "understand" different modalities that you can train an
    encoder for. To enable the language model to adapt to the encoder outputs,
    the FusionLayer inserts a new learnable layer between the decoder (language
    model) layers to learn to combine the encoder outputs and decoder
    activations. The fusion layer can be inserted before or after the decoder
    layer, in Flamingo they insert before.

    Args:
        layer (nn.Module): original decoder layer
        fusion_layer (nn.Module): new fusion layer
        fusion_first (bool): boolean to insert fusion layer before or after
	        the decoder layer.
    """

    def __init__(
        self,
        layer: nn.Module,
        fusion_layer: nn.Module,
        fusion_first: bool = True
    ):
        super().__init__()
        self.layer = layer
        self.fusion_layer = fusion_layer
        self.fusion_first = fusion_first
        self.register_state_dict_post_hook(Test._state_dict_hook)
        # TODO: Switch to register_load_state_dict_pre_hook in v2.5
        self._register_load_state_dict_pre_hook(
	        Test._load_state_dict_hook,
	        with_module=True
	    )

    def _state_dict_hook(self, destination, prefix, keep_vars):
	    """
		Remove "layer" from the original layer in the state_dict
		name. This keeps the orginal state dict name for the layer
		from before fusing with the second layer.

		[!Note] This update changes the order of the OrderedDict
		"""
        keys = list(destination.keys())
        for key in keys:
            if key.startswith("layer"):
                new_key = key.replace("layer.", "")
                destination[new_key] = destination[key]
                del destination[key]

    def _load_state_dict_hook(self, state_dict, *args, **kwargs):
	    """ Undo the change from _state_dict_hook """
        keys = list(state_dict.keys())
        for key in keys:
            if not key.startswith("fusion_layer"):
                new_key = "layer." + key
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

	def setup_cache(self, batch_size: int, dtype: torch.dtype) -> None:
		self.layer.setup_cache(batch_size, dtype)
		self.fusion_layer.setup_cache(batch_size, dtype)

    def cache_enabled(self) -> bool:
        return self.layer.cache_enabled()

    def reset_cache(self):
        self.layer.reset_cache()
        self.fusion_layer.reset_cache()

    def fusion_params(self) -> List[str]:
        """
        Return parameters of fusion layer.
        """
        fusion_params = [
            f"fusion_layer.{k}" for k, v in self.fusion_layer.named_parameters()
        ]
        return fusion_params

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """
        Args:
            x (Tensor): input tensor with shape
                [batch_size x seq_length x embed_dim]

        Returns:
            Tensor: output tensor with same shape as input
                [batch_size x seq_length x embed_dim]`

        """
        if self.fusion_first:
            x = self.fusion_layer(x, **kwargs)
            x = self.layer(x, **kwargs)
        else:
            x = self.layer(x, **kwargs)
            x = self.fusion_layer(x, **kwargs)
        return x
```

### Fusion Embedding
The reason for this layer is summarized in the docstring.

```python
class FusionEmbedding(nn.Module):
    def __init__(self,
	    vocab_size: int,
	    additional_tokens: int,
	    embed_dim: int
	) -> None:
        """Fusion embedding traines additional special tokens while
        keeping the original embedding frozen. When fusing new models with a
        language model, there may be some additional tokens needed to support
        the fused language model. The FusionEmbedding keeps the original
        embedding frozen while learning a much smaller second embedding for the
        additional tokens. During forward the module routes the tokens to the
        appropriate embedding table.

        Args:
            vocab_size (int): language model vocab size
            additional_tokens (int)): additional tokens for the fused model
            embed_dim (int): embedding dimension of the two embedding tables
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fusion_embedding = nn.Embedding(additional_tokens, embed_dim)
        self.dim = embed_dim
        # TODO: Support merging the embeddings after finetuning

    def fusion_params(self) -> List[str]:
        """
        Return fusion embedding parameters.
        """
        fusion_params = ["fusion_embedding.weight"]
        return fusion_params

    def _fused_embed(self, bs, seq_len):
        """
        Return an empty tensor the shape of the combined embedding.
        """
        device = self.embedding.weight.device
        dtype = self.embedding.weight.dtype
        return torch.empty(bs, seq_len, self.dim, device=device, dtype=dtype)

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input (Tensor): input integer tensor with shape
                [batch_size x seq_length]

        Returns:
            Tensor: output tensor embedding with shape
                [batch_size x seq_length x embed_dim]`

        """
        bs, seq_len = input.size()
        vocab_size = self.embedding.num_embeddings

        mask = input < vocab_size
        tokens = torch.masked_select(input, mask)
        additional_tokens = torch.masked_select(input, ~mask) - vocab_size

        embeds = self.embedding(tokens)
        additional_embeds = self.fusion_embedding(additional_tokens)

        out = self._fused_embed(bs, seq_len)
        mask = mask.unsqueeze(-1).expand(bs, seq_len, self.dim)
        out.masked_scatter_(mask, embeds)
        out.masked_scatter_(~mask, additional_embeds)
        return out
```

### Fusion Utils
These are mostly pulled directly from peft_utils as the process of getting special marked parameters and freezing/unfreezing them and handling the checkpointing is very similar to peft. These are kept separate to peft though as it's likely that multimodal models would need to be combined with peft recipes and we need to control peft and fusion parameters separately.

```python
def get_fusion_params(model: nn.Module) -> Dict[str, nn.Parameter]:
    """
    Return the subset of parameters from a model that correspond to fused
    modules. Assumes that any fusion class has defined the
    :func:`~torchtune.modules.model_fusion.FusionLayer.fusion_params` method.

    Args:
        model (nn.Module): Instance of model class containing some
        fusion params.

    Returns:
        Dict[str, nn.Parameter]: the subset of model's state dict containing
        only adapter parameters.

    """
    fusion_params = {}
    for k, v in model.named_modules():
        if hasattr(v, "fusion_params") and callable(v.fusion_params):
            current_fusion_params = v.fusion_params()
            for n, p in v.named_parameters(recurse=True):
                if n in current_fusion_params:
                    full_key = f"{k}.{n}" if k else n
                    fusion_params.update({full_key: p})
                    current_fusion_params.remove(n)
            assert (
                current_fusion_params == []
            ), f"Fusion params {current_adapter_params} not converted"
    return fusion_params


def set_trainable_params(
		model: nn.Module, fusion_params: Dict[str, Any]
	) -> None:
    """
    Set trainable parameters for an nn.Module based on a state dict of fusion
    parameters.

    Args:
        model (nn.Module): Instance of model class containing some adapter
	        params.
		fusion_params (Dict[str, Any]): State dict mapping adapter key names to
			their respective nn.Parameters (i.e. outputs of
			:func:`~torchtune.modules.model_fusion.get_fusion_params`.)

    Returns:
        None
    """
    for k, v in model.named_parameters():
        v.requires_grad_(k in fusion_params)
```

Additional function to convert any module into a fusion module. A primary use case would be for modifying the original encoder with additional layers to learn a mapping from the original encoder to the LLM space. These layers or defined as a full module, would be marked a fusion_module so their parameters could be controlled independently from the pre-trained encoder.
```python
def regsiter_fusion_module(module):
	""" Add the method fusion_params to to an nn.Module
		to mark them as fusion params
	"""
	def fusion_params(self) -> List[str]:
        """
        Return parameters of fusion layer.
        """
        return list(self.named_parameters().keys())
    module.fusion_params = fusion_params
```

## Discussion
- DeepFusionModel could just be called EncoderDecoderTransformer but that doesn't cast it in the context of fusing pre-trained models
- This design was intended to be robust over a large set of MM fusion designs, but it's a fast moving space and it's possible it's overfit to the current state of things. For example, there hasn't been any thought given to "MM in + MM out" models. Though I believe that the design makes minimal assumptions as to give us flexibility going forward. For example, the fusion_models all assume one encoder now for simplicity but can be updated pretty easily to support multiple encoders.
- These concepts are primarily meant to be exposed at the model builder stage and remain hidden at the recipe level. The only area they get exposed would be in checkpoint logic if we introduce recipes stitching together multiple pretrained models
