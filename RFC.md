# [RFC] MOE design in Torchtune

## Background
This RFC proposes adding the MOE support in Torchtune. We want to design in a general way so that components can be easily swapped when implementing different MOE models. An MOE layer directly replaces the dense FFN layer in the transformer decoder layer and has two main components: router and experts.

## Expert
An expert is essentially an FFN layer similar to the original dense FFN layer in the transformer decoder layer. There are two kinds of experts: routed experts and shared experts. Each expert in the routed experts specializes in learning certain patterns/aspects, and only part of the routed experts will be activated. On the other hand, shared experts are always activated, aiming at capturing and consolidating common knowledge across varying contexts.

**Here's the proposed Experts design in torchtune:**
```python
class Experts(nn.Module):
    def __init__(self, dim_in, dim_out, num_experts=1, swiglu=True, nonlinearity=None):
        self.gate_proj = nn.Parameter(torch.empty(num_experts, dim_in, dim_out))
        self.down_proj = nn.Parameter(torch.empty(num_experts, dim_out, dim_in))
        if swiglu:
            self.up_proj = nn.Parameter(torch.empty(num_experts, dim_in, dim_out))
            self.act_fn = F.silu()
        else:
            self.up_proj = None
            self.act_fn = nonlinearity

    def forward(self, x, use_token_choice=False, num_local_tokens_per_expert=None):
        '''
        inputs:
            x: input tokens, shape [bs*slen*experts_per_token, hidden_dim] for TC or [num_experts*tokens_per_expert, hidden_dim] for EC
            use_token_choice: if we use token choice forward
            num_local_tokens_per_expert: number of tokens for each expert, only used for token choice forward
        outputs:
            out: output tokens, shape [bs*slen*experts_per_token, hidden_dim]
        '''
        if use_token_choice:
            # TODO: use cutlass groupGEMM instead of torch.matmul() to optimize performance
            assert num_local_tokens_per_expert is not None, "num_local_tokens_per_expert is needed for token choice expert forward"
            # x_expert_splits shape [num_experts, tokens_per_expert(varying), hidden_dim]
            x_expert_splits = torch.split(x, split_size_or_sections=num_local_tokens_per_expert, dim=0)
            out_expert_splits = []
            for expert_index, x_expert_split in enumerate(x_expert_splits):
                gate_proj = self.gate_proj[expert_index]
                down_proj = self.down_proj[expert_index]
                up_proj = None
                if self.up_proj is not None:
                    up_proj = self.up_proj[expert_index]

                h = self.act_fn(torch.matmul(x_expert_split, gate_proj))
                if up_proj is not None:
                    h = h * torch.matmul(x_expert_split, up_proj)
                # [tokens_per_expert, hidden_dim]
                h = torch.matmul(h, down_proj)

                out_expert_splits.append(h)
            # shape [num_experts * tokens_per_expert(varying), hidden_dim] = [bs*slen*experts_per_token, hidden_dim]
            out = torch.cat(out_expert_splits, dim=0)
        else:
            # x shape [num_experts, tokens_per_expert, hidden_dim]
            x = x.view(num_experts, -1, dim_in)
            h = self.act_fn(torch.bmm(x, self.gate_proj))
            if self.up_proj is not None:
                h = h * torch.bmm(x, self.up_proj)
            out = torch.bmm(h, self.down_proj).view(-1, dim_in)
        return out

# Expert builder for routed experts
def moe_experts(hidden_dim, model_dim, num_experts, swiglu=True, nonlinearity=None):
    return Experts(dim_in=hidden_dim, dim_out=model_dim, num_experts=num_experts, swiglu=swiglu, nonlinearity=nonlinearity)

# Single expert / shared expert
def moe_expert(hidden_dim, model_dim, swiglu=True, nonlinearity=None):
    return Experts(dim_in=hidden_dim, dim_out=model_dim, num_experts=1, swiglu=swiglu, nonlinearity=nonlinearity)
```

## Router
Router is a gating network that calculates router scores and learns token-to-expert affinity. There are two types of routing: token choice routing and expert choice routing.

Mixtral uses *token choice* topK routing, which means each token will select its topK experts. The router is implemented through a learnable gate function, whose outputs will go through softmax and topK. The router defines how tokens select experts / experts select tokens based on router scores.

**Here's the proposed Token Choice Routing and TokenChoiceMoeLayer design in torchtune:**
```python
class TokenChoiceTopKRouter(nn.Module):
    def __init__(self, hidden_dim, num_experts, experts_per_token):
        self.gate = nn.Linear(hidden_dim, num_experts)
        self.experts_per_token = experts_per_token

    def forward(self, x, use_sigmoid=False):
        '''
        input:
            x shape [bs*slen, hidden_dim]
        outputs:
            routed_input shape [bs*slen*experts_per_token, hidden_dim]
            num_local_tokens_per_expert shape [num_experts,]
        '''
        # scores shape [bs*slen, num_experts]
        scores = self.gate(x)
        if use_sigmoid:
            scores = torch.sigmoid(scores.to(sigmoid_dtype)).to(x.dtype)
        else:
            scores = F.softmax(scores.to(softmax_dtype), dim=1).to(x.dtype)

        # TODO: implement load balancing auxiliary loss for token choice routing
        # https://github.com/NVIDIA/Megatron-LM/blob/f1f039224584f0bc6ba89c21ef4f491d7136e3ce/megatron/core/transformer/moe/router.py#L162

        # router scores/indices shape [bs*slen, experts_per_token]
        top_scores, selected_experts_indices = torch.topk(scores, k=self.experts_per_token, dim=1)
        top_scores /= top_scores.sum(dim=-1, keep_dim=True).to(x.dtype)

        # shape [num_experts,]: how many tokens for each expert
        num_local_tokens_per_expert = torch.histc(selected_expert_indices.view(-1), bins=num_experts, min=0, max=num_experts)
        # shape [bs*slen*experts_per_token,]
        token_indices_experts_sorted = torch.argsort(selected_experts_indices.view(-1), stable=True)
        # top_scores shape [bs*slen*experts_per_token,]
        top_scores = top_scores.view(-1)[token_indices_experts_sorted]

        # token_indices_experts_sorted_expanded shape [bs*slen*experts_per_token, hidden_dim]
        token_indices_experts_sorted_expanded = token_indices_experts_sorted.reshape(-1, 1).expand(-1, hidden_dim)
        # routed_input shape [bs*slen*experts_per_token, hidden_dim]
        routed_input = torch.gather(x, dim=0, index=token_indices_experts_sorted_expanded)
        routed_input = routed_input * top_scores

        return routed_input, token_indices_experts_sorted_expanded, num_local_tokens_per_expert
 ```

However, token choice routing has several pitfalls according to the expert choice [paper](https://arxiv.org/pdf/2002.05202).
1. Poor load balance. Experts can become under or over-specialized. Load imbalance can hurt step latency / inference time.
2. Experts under specialization. Ideally the gating network will learn token-to-expert affinity such that similar or relevant tokens are routed to the same expert. However, a sub-optimal strategy can produce redundant experts and/or experts that are not sufficiently specialized.
3. Same compute for each token. Token choice will allocate a fixed number of experts to each token regardless of the importance of different tokens. Ideally an MOE model should flexibly allocate compute resources based on the complexity of the input.

Compared to **token choice**, **expert choice** topK routing lets experts select its top-k tokens. The ExpertChoiceMoeLayer class routes input tokens to different experts based on the routing algorithm, processes them through the experts and the shared expert, and then combines the output.

**Here's the proposed Expert Choice Routing and ExpertChoiceMoeLayer design in torchtune:**
```python
class ExpertChoiceTopKRouter(nn.Module):
	def __init__(self, hidden_dim, num_experts):
		self.gate = nn.Linear(hidden_dim, num_experts)
		self.tokens_per_expert = tokens_per_expert

	def forward(self, x, use_sigmoid=False):
        '''
        input:
            x: shape [bs*slen, hidden_dim]
        outputs:
            routed_input: shape [num_experts*tokens_per_expert, hidden_dim]
            num_local_tokens_per_expert: None
        '''
        # scores shape [num_experts, bs*slen]
        scores = self.gate(x).transpose(0,1)
        if use_sigmoid:
            scores = torch.sigmoid(scores.to(sigmoid_dtype)).to(x.dtype)
        else:
            scores = F.softmax(scores.to(softmax_dtype), dim=0).to(x.dtype)
        # router scores/indices shape [num_experts, tokens_per_expert]
        top_scores, selected_token_indices = torch.topk(scores, k=self.tokens_per_expert, dim=1)

        # apply the token preprocess function and then run experts forward
        selected_token_indices_expanded = selected_token_indices.reshape(-1, 1).expand(-1, D)
        # routed input shape [num_experts*tokens_per_expert, hidden_dim]
        routed_input = torch.gather(x, dim=0, index=selected_token_indices_expanded)
        routed_input = routed_input * top_scores.reshape(-1, 1)
        return routed_input, selected_token_indices_expanded, None,
```

## Moe Layer
An MOE layer consists of experts and routers.

**Here's the proposed MoeLayer design in torchtune:**
```python
class MoeLayer(nn.Module):
    def __init__(self, router="token_choice"):
        self.experts = moe_experts(hidden_dim, model_dim, num_experts=num_experts)
        self.shared_expert = moe_expert(hidden_dim, model_dim)
        if router == "token_choice":
            self.router = TokenChoiceTopKRouter(hidden_dim, num_experts, experts_per_token)
        elif router == "expert_choice":
            self.router = ExpertChoiceTopKRouter(hidden_dim, num_experts, tokens_per_expert)
        else:
            raise NotImplementedError("This router is not supported yet!")

    def forward(self, x, infernece=False):
        routed_input, token_indices, num_local_tokens_per_expert = self.router(x)

        # routed output shape [num_experts*tokens_per_expert, hidden_dim] for EC, [bs*slen*experts_per_token, hidden_dim] for TC
        routed_output = self.experts(routed_input, num_local_tokens_per_expert=num_local_tokens_per_expert)

        # shared expert
        if use_shared_expert:
            out = self.shared_expert(x)
        else:
            out = torch.zeros_like(x)

        # add experts output
        out.data = scatter_add_(
            out.data,
            routed_output,
            selected_indices,
        )
        return out
 ```

## Model builder
Besides the above components: experts, routers, and MOE layers, we would need a model builder to pull all pieces together to form the Transformer decoder layer and then Transformer decoder:

**Here's the proposed MOE model builder design in torchtune:**
```python
def moe(...) -> TransformerDecoder:
    # Build the decoder associated with the moe model. This includes
    # - Token embeddings
    # - num_layers number of TransfomerDecoderLayer block
    # - RMS Norm layer applied to the ouput of the transfomer
    # - Final projection into the token space'
    token_embeddings = nn.Embedding(vocab_size, embed_dim)
    self_attn = MultiHeadAttention()
    moe_layer = MoeLayer(router="token_choice") # or MoeLayer(router="expert_choice")
    norm = RMSNorm(dim=embed_dim)
    layer = TransformerSelfAttentionLayer(attn=self_attn, mlp=moe_layer, sa_norm=norm, mlp_norm=norm)
    output_proj = nn.Linear(embed_dim, vocab_size)
    return TransformerDecoder(
        tok_embeddings=tok_embeddings,
        layers=layer,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        norm=RMSNorm(dim=embed_dim),
        output=output_proj,
    )
```

**File changes for new modules/functions**
```
torchtune/
    modules/
        moe/
            moe_layers.py
                TokenChoiceTopKRouter()
                ExpertChoiceTopKRouter()
                MoeLayer()
            experts.py
                Experts()
    models/
        moe/
            _component_builders.py
                moe()
                moe_expert()
                moe_experts()
```
