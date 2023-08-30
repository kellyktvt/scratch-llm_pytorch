import torch
from torch import Tensor
from torch.nn import Module, ModuleList, Embedding, Linear, Dropout
import torch.nn.functional as F

from model.transformer import TransformerBlock


# TODO: implement the auto-regressive generation
class LLM(Module):
    def __init__(
        self,
        vocab_size: int,
        dim_emb: int,
        num_layers: int,
        num_heads: int,
        dim_k: int,
        causal: bool = True,
        emb_dropout: float = 0.0,
        ffd_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.token_embedding = Embedding(vocab_size, dim_emb)
        self.emb_dropout = Dropout(emb_dropout)
        self.transformer = ModuleList(
            [TransformerBlock(dim_emb, num_heads, dim_k, causal, ffd_dropout=ffd_dropout) for _ in range(num_layers)]
        )
        self.projection_head = Linear(dim_emb, vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.token_embedding(x)  # (bs, seq_len, dim_emb)
        x = self.emb_dropout(x)  # (bs, seq_len, dim_emb)
        x = self.transformer(x)  # (bs, seq_len, dim_emb)
        x = self.projection_head(x)  # (bs, seq_len, vocab_size)

        return x

    @torch.inference_mode()
    def generate(self, inputs: Tensor, max_seq_len: int) -> Tensor:
        pass
