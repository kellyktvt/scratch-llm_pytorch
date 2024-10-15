from typing import Tuple
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, Module, Parameter, Sequential

# Specify machine epsilon for 'torch.float32'
EPS = torch.finfo(torch.float32).eps


# Computes rotary positional encodings for each position in a sequence
class RotaryPositionalEncoding(Module):
    # Method to initialize the positional encoding with params
    # seq_len = sequence length; dim_emb = dimensionality of embeddings
    # base = base val for positional encoding; eps = small epsilon val to avoid division by 0 in scaling
    def __init__(self, seq_len: int, dim_emb: int, base: int = 10000, eps: float = EPS) -> None:
        super().__init__()

        self.dim_emb = dim_emb
        # Generates 'indices' as a torch tensor repping positions in the seq
        indices = torch.arange(0, seq_len, dtype=torch.float)
        # Computes'scale' vals for scaling rotations based on 'base' and 'dim_emb'
        scale = 1 / (base ** (torch.arange(0, dim_emb, 2, dtype=torch.float) / dim_emb) + eps)

        # Construct 'position' tensor by outer product of 'indices' and 'scale'
        position = torch.outer(indices, scale)
        # , concatenated along the last dimension
        position = torch.cat((position, position), dim=-1)

        # Compute 'position_cos' and 'position_sin' tensors using cosine and sine fxns applied to 'position'
        position_cos = torch.cos(position[None, None, :, :])  # (bs, num_heads, seq_len, dim_emb)
        position_sin = torch.sin(position[None, None, :, :])  # (bs, num_heads, seq_len, dim_emb)

        self.register_buffer("position_cos", position_cos)
        self.register_buffer("position_sin", position_sin)

    # Method to perform a specific rotation operation on a tensor 'x'
    def _rotate_half(self, x: Tensor) -> Tensor:
        # Split 'x' into 2 halves along the last dimension based on 'dim_emb'
        x1, x2 = x[..., : self.dim_emb // 2], x[..., self.dim_emb // 2 :]

        # Concatenate '(-x2, x1)' along the last dimension and return result
        return torch.cat((-x2, x1), dim=-1)

    # Method to apply rotary positional encodings to the input tensor 'x'
    def forward(self, x: Tensor) -> Tensor:
        # x is of shape  (bs, num_heads, seq_len, dim_emb)
        x = (x * self.position_cos) + (self._rotate_half(x) * self.position_sin)

        return x


# Implements RMS normalization for tensors
class RMSNorm(Module):
    # RMSnorm(x_i) = (x_i / RMS(x)) * g_i where RMS(x) = sqrt(1 / n *  sum a_i ** 2)
    # Initialize scaling factor, gain params, and epsilon
    def __init__(self, dim_last: int, eps: float = EPS):
        super().__init__()
        self.scale = dim_last**0.5
        self.gain = Parameter(torch.ones(dim_last), requires_grad=True)
        self.eps = eps

    # Method to apply RMS normalization to input tensor 'x'
    def forward(self, x: Tensor) -> Tensor:
        norm = torch.norm(x, 2, dim=-1, keepdim=True)
        x = self.scale * self.gain * x / (norm + self.eps)

        return x


# Implement SwiGLU activation function using single linear layer
class SwiGLU(Module):
    # SwiGLU(x) = (xW + b) ⊗ swish(xZ + c) where W, Z, b, c are learnable params
    # Initialize the linear transformation
    def __init__(self, dim_in: int, bias: bool = True) -> None:
        super().__init__()

        self.dim_in = dim_in
        self.linear = Linear(dim_in, 2 * dim_in, bias=bias)

    # Method to apply SwiGLU activation to input tensor 'x'
    def forward(self, x: Tensor) -> Tensor:
        # uses only one weight matrix instead of two
        x = self.linear(x)
        x = F.silu(x[..., : self.dim_in]) + x[..., self.dim_in :]

        return x


class MultiHeadAttention(Module):
    # Method to initialize the multi-head attention layer with params
    def __init__(
        self, seq_len: int, num_heads: int, dim_emb: int, dim_k: int = None, dim_v: int = None, causal=True
    ) -> None:
        super().__init__()

        # Ensure dim_emb is divisiblee by num_heads
        assert dim_emb % num_heads == 0, "num_heads must be a multiple of dim_emb"

        self.seq_len = seq_len
        self.num_heads = num_heads
        self.dim_head = dim_emb // num_heads
        self.dim_k = dim_k or dim_emb
        self.dim_v = dim_v or dim_emb
        self.causal = causal

        # positional encoding to be applied to query and key projections
        # self.positional_encoding = CosinePositionalEncoding(seq_len, dim_emb // num_heads)
        self.positional_encoding = RotaryPositionalEncoding(seq_len, dim_emb // num_heads)

        # Query, Key and Value projections batched into one linear layer
        self.proj_qkv = Linear(dim_emb, 3 * dim_emb, bias=False)
        self.proj_out = Linear(self.dim_v, self.dim_v, bias=False)

        # Build the causal mask, masking upper triangular part of attention scores
        self.register_buffer("causal_mask", torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool())

    # Method to perform forward pass of MultiHeadAttention layer
    def forward(self, x: Tensor, return_scores: bool = False) -> Tensor | Tuple[Tensor, Tensor]:
        # projects input to Q, K, V spaces
        qkv = self.proj_qkv(x)  # (bs, seq_len, 3 * dim_emb)

        # split into Q, K, V
        q, k, v = qkv.chunk(3, dim=-1)  # (bs, seq_len, dim_k), (bs, seq_len, dim_k), (bs, seq_len, dim_v)

        # split projections between heads -> (bs, num_heads, seq_len, dim_k)
        q = q.view(-1, self.seq_len, self.num_heads, self.dim_head).permute(0, 2, 1, 3)
        k = k.view(-1, self.seq_len, self.num_heads, self.dim_head).permute(0, 2, 1, 3)
        v = v.view(-1, self.seq_len, self.num_heads, self.dim_head).permute(0, 2, 1, 3)

        # apply positional encoding to projections, for each heads
        q = self.positional_encoding(q)  # (bs, num_heads, seq_len, dim_k)
        k = self.positional_encoding(k)  # (bs, num_heads, seq_len, dim_k)

        # Compute the correlation between a query q_i and all the keys, for every q_i
        attn_scores = (q @ k.permute(0, 1, 3, 2)) * self.dim_k**-0.5  # (bs, num_heads, seq_len, seq_len)

        # Fill the upper triangular part of the attention scores with -inf to inhibit them in the softmax
        if self.causal:
            attn_scores.masked_fill_(self.causal_mask[None, None, ...], -torch.inf)

        # attention scores are used to build a weighted linear combination of values vectors
        attn_scores = torch.softmax(attn_scores, dim=-1)  # (bs, num_heads, seq_len, seq_len)
        out = attn_scores @ v  # (bs, num_heads, seq_len, dim_v)

        # Merge heads by reshaping and permuting output tensor
        out = out.permute(0, 2, 1, 3).contiguous().view(-1, self.seq_len, self.dim_v)  # (bs, seq_len, dim_v)

        # projects to the output space
        out = self.proj_out(out)  # (bs, seq_len, dim_v)

        # Return output tensor and optionally the attention scores if 'return_scores' is True
        if return_scores:
            return out, attn_scores
        else:
            return out


# Defines a feedforward nn layer using 'Sequential' to chain together multiple layers
class FeedForward(Sequential):
    # Method to initialize the feed forward layer with params
    def __init__(self, dim_in: int, dim_hidden: int, bias: bool = False) -> None:
        super().__init__(
            # First Linear layer
            Linear(dim_in, dim_hidden, bias=bias),
            # SwiGLU activation
            SwiGLU(dim_hidden),
            # Second Linear layer
            Linear(dim_hidden, dim_in, bias=bias),
        )


# Defines transformer block that includes multi-head attention, feedforward layers, and normalization
class TransformerBlock(Module):
    # Method to initialize the TransformerBlock with params
    def __init__(
        self,
        seq_len: int, # length of sequence
        dim_emb: int, # dimension of embedding
        attn_num_heads: int, # number of attention heads
        ffn_hidden_dim: int, # dimensionality of hidden layer in feedforward network
        ffn_bias: bool = False, # whether to use bias in feedforward network
        attn_causal: bool = True, # whether to use causal mask in attention layer
    ) -> None:
        super().__init__()

        # Follows LLama 2 architecture:
        # - positional encoding on every head of the multi-head attention query and keys projections
        # - RMS pre-normalization instead of layer normalization
        # - SwiGLU activation for the feedforward
        self.norm_attn = RMSNorm(dim_emb)
        self.multihead_attn = MultiHeadAttention(seq_len, attn_num_heads, dim_emb, causal=attn_causal)
        self.norm_ffn = RMSNorm(dim_emb)
        self.feed_forward = FeedForward(dim_emb, ffn_hidden_dim, bias=ffn_bias)

    # Method to perform forward pass of TransformerBlock
    def forward(self, x: Tensor) -> Tensor:
        x = x + self.multihead_attn(self.norm_attn(x))  # (bs, seq_len, dim_in)
        x = x + self.feed_forward(self.norm_ffn(x))  # (bs, seq_len, dim_in)

        # Return transformed tensor
        return x  # (bs, seq_len, dim_in)
