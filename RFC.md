# [RFC] TransformerDecoderLayer Refactor

**TLDR**
- Replace TransformerDecoderLayer with TransformerSelfAttention and TransformerCrossAttention
- Replace CausalSelfAttention with GroupedQueryAttention
- Support legacy Checkpoints

**WHY**: because we need to support a mix of cross attention and self attention layers to support encoders and deep fusion multimodal models in the library. Allowing different transformer layer types allows us to support most variants of transformers while keeping the complexity of the attention module with a reusable GroupedQueryAttention.# [RFC] TransformerDecoderLayer Refactor

## Context

Currently `TransformerDecoder` with `TransformerDecoderLayer` is a decoder only style transformer. As opposed to the original transformer (image below), decoder only style models don't take inputs from an encoder and remove the cross attention block (green) from the decoder block (`TransformerDecoderLayer`). Since most SOTA LLMs have been GPT style decoder only models for the last few years, only supporting Decoder layers have worked well until now.

<p align="center">
  <img src="https://github.com/pbontrager/torchtune/blob/304444545fdc658de1e239b05ef8928cdf21240b/labeled_attention.png?raw=true" width=50%/>
</p>

But looking forward at the advances in multimodal LLMs, it's clear that encoder-decoder model architectures will become important again [ref](https://scontent-iad3-1.xx.fbcdn.net/v/t39.2365-6/452387774_1036916434819166_4173978747091533306_n.pdf?_nc_cat=104&ccb=1-7&_nc_sid=3c67a6&_nc_ohc=t6egZJ8QdI4Q7kNvgGwBp2W&_nc_ht=scontent-iad3-1.xx&oh=00_AYC2OwSKC1f5iJMNAiAj48_5_u1cmYjC1nPcIizQmBu7HQ&oe=66A6EB8D). This brings with it several new requirements for TransformerDecoderLayers:
- The need to support self attention (decoder x decoder) and cross attention (decoder x encoder). Another way to think of this is that decoder blocks need to be able to support their primary input (tokens) and a conditional input (multimodal embeddings).
- The need to support both self attention masks and cross attention (aka conditional) masks. To work with encoders, layers also need to relax the assumption that self attention is always causal.
- The need to support mixtures of layer types where some might require conditional inputs, some might not.

**TLDR** Layers need to support a primary input and mask (current inputs) and an optional conditional input and mask. This would support any standard variation of TransformerBlock.

**Note on Attention** In the diagram above, it should be noted that all three attention implementations (red, green, and blue) use the same attention module, Multi Headed Attention. Most of the implementation complexity lie inside this module, and it's only getting more complex [#1193](https://github.com/pytorch/torchtune/pull/1193). The only difference between them is what inputs are provided as k, q, v, and what mask is provided. All of that to say, to support cross attention and self attention, the attention module doesn't change but the transformer block itself. See Figure 2, Deep Fusion, for an overview of some popular layer variations for multimodal models [ref](https://arxiv.org/pdf/2405.17927).
## Proposal
**High Level List:**
- Extend the TransformerDecoder forward signature to include `encoder_input` and `encoder_mask`. This sets a standard signature for all layers to follow so they can all be mixed and matched and stacked together.
- Modify TransformerDecoder to accept a list of layers instead of a layer and and a count. This allows us to support models where only some layers are conditional and other are not.
- Implement two initial layers `TransformerSelfAttentionLayer` and `TransformerCrossAttentionLayer`. `TransformerSelfAttentionLayer` would replace `TransformerDecoderLayer`. These would just be two possible layers, but the design would support more variations.
- Generalize `CausalSelfAttention` work in all three scenarios above (red, green, blue) and rename to `GroupedQueryAttention`. This includes changing the default mask behavior to not always be causal.
- **Extra**:
	- Add TanhGate module to support gated attention [ref](https://paperswithcode.com/method/attention-gate)
	- Modify `TransformerDecoder` to output it's hidden layers and make `model.output` optional to support the changes proposed [#1017](https://github.com/pytorch/torchtune/issues/1017) and make only one BC breaking change.
	- Implement legacy checkpoint conversion to support old checkpoints

### TransformerDecoder
TransformerDecoder only highlighting the changes, current module [here](https://github.com/pytorch/torchtune/blob/0057fe7cf83e14f0b62538a8d4d20719e0a88639/torchtune/modules/transformer.py#L99). Discussion in the comments.

```python
class TransformerDecoder(nn.Module):
    def __init__(
        self,
        *,
        ...
        # as opposed to layer and num_layers
        layers: List[nn.Module],
        ...
        # output is generalized and hidden_idx is added to address #1017
        output: Callable
        hidden_idx: Optional[List[int]] = None
    ) -> None:
        super().__init__()
        # We could keep an optional layer + num_layers args for BC
        # Otherwise this change requires builders to go from
        # TransformerDecoder(layer, 32)
        # To
        # for _ in range(32):
	    #    layers.append(layer)
	    # TransformerDecoder(layers)
        ...

    def forward(
        self,
        tokens: Tensor,
        *,
        mask: Optional[Tensor] = None,
        encoder_input: Optional[Tensor] = None,
        encoder_mask: Optional[Tensor] = None,
        input_pos: Optional[Tensor] = None,
    ) -> [Tensor, List[Tensor]]:
        ...

		# Move max seq length check from attention to here
        if seq_len > self.max_seq_len:
            raise ValueError("Too long")

        ...

		hidden = []
        for i, layer in enumerate(self.layers):
	        if i in self.hidden_idx:
		        hidden.append(h)
            h = layer(
                h,
                mask=mask,
                encoder_input=encoder_input,
                encoder_mask=encoder_mask,
                input_pos=input_pos,
            )

        # shape: [b, s, d]
        h = self.norm(h)

        output = self.output(h).float()
        # this unifies the output with ViT but breaks recipes
        # Alternative:
        # output = output if not hidden else hidden + [output]
        return output, hidden
```
@SalmanMohammadi does this design address all your needs from #1017
### TransformerSelfAttentionLayer
TransformerDecoderLayer only highlighting the changes, current module [here](https://github.com/pytorch/torchtune/blob/0057fe7cf83e14f0b62538a8d4d20719e0a88639/torchtune/modules/transformer.py#L15). Discussion in the comments.

```python
class TransformerSelfAttentionLayer(nn.Module):
    def __init__(
        self,
        attn: GroupedQueryAttention, # Updated Attn module
        mlp: nn.Module,
        *,
        # sa_norm renamed since it's no longer only self attention
        # this requires repo wide changes for sa_norm -> attn_norm
        attn_norm: Optional[nn.Module] = None,
        mlp_norm: Optional[nn.Module] = None,
        # New scale modules to allow scaling/gating attn output
        attn_scale: Optional[nn.Module] = None,
        mlp_scale: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.attn = attn
        self.mlp = mlp
        self.attn_norm = attn_norm or nn.Identity()
        self.mlp_norm = mlp_norm or nn.Identity()
        self.attn_scale = attn_scale or nn.Identity()
        self.mlp_scale = mlp_scale or nn.Identity()

    def forward(
        self,
        x: Tensor,
        *,
        mask: Optional[Tensor] = None,
        input_pos: Optional[Tensor] = None,
        **kwargs, # encoder_input and encoder_mask is not used for this module
    ) -> Tensor:
	    # Identical to TransformerDecoderLayer with addition of *_scale
        attn_out = self.attn(self.attn_norm(x), mask=mask, input_pos=input_pos)

        h = self.attn_scale(attn_out) + x

        mlp_out = self.mlp(self.mlp_norm(h))

        out = h + self.mlp_scale(mlp_out)
        return out
```

### TransformerCrossAttentionLayer
New layer. Discussion in the comments.
```python
class TransformerCrossAttentionLayer(nn.Module):
    def __init__(
		...
    ) -> None:
        super().__init__()
        assert (
            attn.pos_embeddings is None
        ), "Positions are not computed for encoder inputs"
        ...

    def _skip_mask(self, mask: Optional[Tensor]) -> Optional[Tensor]:
	    """Some tokens in x may not attend to any encoder inputs
        due to the cross attention mask (encoder_mask). This results in
        a full row of the attention matrix being masked out.

        In the example below, the word "the" is masked from every embedding.

        .. code-block:: text

            |emb||emb||emb|
        |The| x    x    x
        |red|      x
        |car| x

        This results in no inputs into the softmax layer which causes a NaN.
        The skip mask removes the output of the attention module and
        mlp resulting in the token being skipped.

        """
		# There is more discussion around this here
		# https://gist.github.com/drisspg/547648ded500d078961b7a3b3f11c310
		# We might be able to optimize this approach more
        if mask is None:
            return None
        if mask.dtype == torch.bool:
            mask = ~mask
        else:
            mask = torch.isneginf(mask)
        mask = torch.all(mask, dim=-1, keepdim=True)
        return mask

    def forward(
        self,
        x: Tensor,
        *,
        encoder_input: Optional[Tensor] = None,
        encoder_mask: Optional[Tensor] = None,
        input_pos: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        # Cross attention only layer (see red only) has no
        # purpose if no conditional input is provided
        if encoder_input is None:
            return x

        # A mask of tokens (x) with no encoder_input
        skip_mask = self._skip_mask(encoder_mask)

        # Input tensor and attention output have the same shape
        # [b, s, d]
        # Norm applied before self-attention
        attn_out = self.attn(
            self.attn_norm(x),
            encoder_input,
            mask=encoder_mask,
            input_pos=input_pos
        )
        if skip_mask is not None:
            attn_out.masked_fill_(skip_mask, 0)

        # Residual connection; shape: [batch_size, seq_length, embed_dim]
        h = self.attn_scale(attn_out) + x

        # Norm applied before the feedforward layer
        mlp_out = self.mlp(self.mlp_norm(h))
        if skip_mask is not None:
            mlp_out.masked_fill_(skip_mask, 0)

        # Residual connection; shape: [batch_size, seq_length, embed_dim]
        out = h + self.mlp_scale(mlp_out)
        return out
```
@Chillee the `_skip_mask` is an example of a fully masked out row. Is applying the mask this way the most efficient way to handle this?
### GroupedQueryAttention
CausalSelfAttention only highlighting the changes, current module [here](https://github.com/pytorch/torchtune/blob/main/torchtune/modules/attention.py). Discussion in the comments.
```python
class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        *,
        ...
        # made this optional to support encoders
        pos_embeddings: Optional[nn.Module] = None,
        # support qk normalization
        # https://arxiv.org/abs/2010.04245
        q_norm: Optional[nn.Module] = None,
		k_norm: Optional[nn.Module] = None,
		# sets the default mask wehn none is provided
		# to be causal or None
        default_causal_mask: bool = True,
    ) -> None:
        super().__init__()
        ...
        if bool(q_norm) ^ bool(k_norm):
            raise ValueError("q and k norm must be set together")
		...

    def forward(
        self,
        x: Tensor,
        # Optinal for cross attention but we could
        # make it required instead and self attention
        # would be GQA(x, x)
        y: Optional[Tensor] = None,
        *,
        mask: Optional[Tensor] = None,
        input_pos: Optional[Tensor] = None,
    ) -> Tensor:
        ...
        bsz, seq_len_x, _ = x.shape
        y = y if y is not None else x
        _, seq_len_y, _ = y.shape

        # q has shape [b, s, num_heads * head_dim]
        # k has shape [b, s, num_kv_heads * head_dim]
        # v has shape [b, s, num_kv_heads * head_dim]
        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        ...

        # Apply positional embeddings
        if self.pos_embeddings is not None:
            q = self.pos_embeddings(q, input_pos=input_pos)
            k = self.pos_embeddings(k, input_pos=input_pos)

        ...

        # Normalize k and q
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        ...

        output = nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.attn_dropout,
            # if you're using a kv_cache or a mask is provided
            # then don't apply a default causal mask
            # if there is no mask,
            # then only apply a causal mask if GQA has default_causal_mask
            is_causal=self.kv_cache is None and mask is None and self.is_causal,
        )

        ...

        return self.output_proj(output)
```

### Legacy Checkpointing

Since GQA changes sa_norm to attn_norm, this will break existing checkpoints that are in the tune format. It could be argued that since we don't really let users save checkpoints in the tune format by default, that this would not affect anyone. But to ensure that no checkpoint are broken, we can add conversion mappings to [convert_weights.py](https://github.com/pytorch/torchtune/blob/main/torchtune/models/convert_weights.py). These would be exactly the same as `_FROM_HF` or `_FROM_META` but instead map from older version to new.

```python
_FROM_TUNE_0_2_1 = {
    "layers.{}.sa_norm.scale": "layers.{}.attn_norm.scale",
}
```
The above code block is assuming that the mapping function is updated to treat missing keys as an identity

The state dict can be converted based on the provided version string `#.#.#`
```python
def _legacy_to_tune(
    state_dict: Dict[str, torch.Tensor], version: str
) -> Dict[str, torch.Tensor]:
    converted_state_dict = {}
    v1, v2, v3 = version.split(".")
    for key, value in state_dict.items():
        new_key = get_mapped_key(key, eval(f"_FROM_TUNE_{v1}_{v2}_{v3}"))
        converted_state_dict[new_key] = value

    return converted_state_dict

```

Finally, FullModelTorchTuneCheckpointer would be updated to check for signatures of older checkpoints, and then call `_legacy_to_tune`. This change is only applied to 'load_checkpoint' so that a users checkpoint is converted to the newer format but never back to the older format.
```python

class FullModelTorchTuneCheckpointer(_CheckpointerInterface):
	...

    def load_checkpoint(self, weights_only: bool = True) -> Dict[str, Any]:
        ...

		if "layers.0.sa_norm.scale" in model_state_dict:
			logger.info(
				"This is an older checkpoint format. Converting \
				to the current version."
			)
			model_state_dict = convert_weights._legacy_to_tune(
				model_state_dict, "0.2.1"
			)
			state_dict[utils.MODEL_KEY] = model_state_dict
		...

```
## Caveats

### Alternatives
@kartikayk expressed concerns over exposing SelfAttention and CrossAttention at the build level which might add complexity for users to parse. While we still requires users to pass in the attention module anyway, another way we could approach this would be to parametrize TransformerDecoderLayer to handle both cross attention and self attention.

```python
class TransformerDecoderLayer(nn.Module):
    def forward(
        self,
        x: Tensor,
        *,
        mask: Optional[Tensor] = None,
        encoder_input: Optional[Tensor] = None,
        encoder_mask: Optional[Tensor] = None,
        input_pos: Optional[Tensor] = None,
    ) -> Tensor:

        if encoder_input is None and self.cross_attn_layer:
            return x

		# return None if not self.cross_attn_layer
        skip_mask = self._skip_mask(encoder_mask)

		y = encoder_input if self.cross_attn_layer else x
		m = encoder_mask if self.cross_attn_layer else mask
        attn_out = self.attn(
            self.attn_norm(x),
            y,
            mask=m,
            input_pos=input_pos
        )
        if skip_mask is not None:
            attn_out.masked_fill_(skip_mask, 0)

        # Residual connection; shape: [batch_size, seq_length, embed_dim]
        h = self.attn_scale(attn_out) + x

        # Norm applied before the feedforward layer
        mlp_out = self.mlp(self.mlp_norm(h))
        if skip_mask is not None:
            mlp_out.masked_fill_(skip_mask, 0)

        out = h + self.mlp_scale(mlp_out)
        return out
```

This is a reasonable alternative but I do fear it'll get more and more complex to extend as we go forward and want to support more attention types.

### Risks/Concerns
- Attention module and model builders both get a bit more complex with this. The more variation we support the more complexity we introduce. I believe that supporting multiple transformer layers at least allows us to keep individual layers simpler.
- This change breaks BC both in the checkpoint itself but also in model builders. We can update all of our model builders, but if users have their own custom ones they will have to update them too. Do we have a mechanism for clearly communicating this in the release notes along with how to update a model builder?
- The cross attention layer here does not work with sample packing. Should we make layers that support sample packing through an error if they're used with packed data? @RdoubleA
- This proposal continues the use of the TransformerDecoder though technically it has been generalized that it could also be used for an encoder (it would need to ignore the kv cache stuff.). I think we should stick to the current name though as that is its primary purpose.
