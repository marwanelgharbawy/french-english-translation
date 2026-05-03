import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

Tensor = torch.Tensor


@dataclass
class TransformerConfig:
    # Assignment table: hidden size = 32
    d_model: int = 32
    n_heads: int = 4
    n_encoder_layers: int = 3
    n_decoder_layers: int = 3
    d_ff: int = 32 * 4
    max_seq_len: int = 32
    dropout: float = 0.1


class TokenPositionalEmbedding(nn.Module):
    """Embedding = token embedding + learned positional embedding."""

    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.max_seq_len = max_seq_len

    def forward(self, token_ids: Tensor) -> Tensor:
        # token_ids: (B, T)
        seq_len = token_ids.size(1)
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}")

        token_vecs = self.token_emb(token_ids)  # (B, T, d)
        pos_ids = torch.arange(seq_len, device=token_ids.device)  # (T,)
        pos_vecs = self.pos_emb(pos_ids).unsqueeze(0)  # (1, T, d)
        return token_vecs + pos_vecs


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # no dropout on attention weights.
        # We still apply hidden dropout to the attention output (post projection).
        self.out_dropout = nn.Dropout(dropout)

    def _to_heads(self, x: Tensor) -> Tensor:
        # (B, T, d_model) -> (B, H, T, d_head)
        # [Tarsh] each token’s 32-dim vector is split into 4 chunks (heads), each chunk size 8.
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.n_heads, self.d_head).transpose(1, 2)

    def _from_heads(self, x: Tensor) -> Tensor:
        # (B, H, T, d_head) -> (B, T, d_model)
        bsz, _heads, seq_len, _d = x.shape
        return x.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        key_padding_mask: Optional[Tensor] = None,
        causal_mask: Optional[Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Vectorized multi-head attention.

        Args:
            query: (B, Tq, d)
            key: (B, Tk, d)
            value: (B, Tk, d)
            key_padding_mask: (B, Tk) bool, True for real tokens, False for PAD.
            causal_mask: (Tq, Tk) bool, True where attention is allowed.
            need_weights: return attention weights (B, H, Tq, Tk)
        """
        q = self._to_heads(self.q_proj(query))  # (B, H, Tq, dH)
        k = self._to_heads(self.k_proj(key))  # (B, H, Tk, dH)
        v = self._to_heads(self.v_proj(value))  # (B, H, Tk, dH)

        # (B, H, Tq, d_head) @ (B, H, d_head, Tk) = (B, H, Tq, Tk)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B, H, Tq, Tk)

        if key_padding_mask is not None:
            if key_padding_mask.dtype != torch.bool:
                key_padding_mask = key_padding_mask.to(torch.bool)
            # (B, Tk) -> (B, 1, 1, Tk)
            # key_padding_mask: True for real tokens, False for PAD.
            scores = scores.masked_fill((~key_padding_mask)[:, None, None, :], -1e9)

        if causal_mask is not None:
            if causal_mask.dtype != torch.bool:
                causal_mask = causal_mask.to(torch.bool)
            # (Tq, Tk) -> (1, 1, Tq, Tk)
            # causal_mask: True where attention is allowed.
            scores = scores.masked_fill((~causal_mask)[None, None, :, :], -1e9)

        attn_weights = torch.softmax(scores, dim=-1)  # (B, H, Tq, Tk)
        attn_out = torch.matmul(attn_weights, v)  # (B, H, Tq, dH)

        out = self._from_heads(attn_out)  # (B, Tq, d_model)
        out = self.out_dropout(self.out_proj(out))

        return out, (attn_weights if need_weights else None)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, *, src_padding_mask: Optional[Tensor] = None) -> Tensor:
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=src_padding_mask)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        *,
        memory: Tensor,
        tgt_padding_mask: Optional[Tensor] = None,
        memory_padding_mask: Optional[Tensor] = None,
        causal_mask: Optional[Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        weights: Dict[str, Tensor] = {}
        ########################### masked multi head attention ###########################
        self_out, self_w = self.self_attn(
            x,
            x,
            x,
            key_padding_mask=tgt_padding_mask,
            causal_mask=causal_mask,
            need_weights=need_weights,
        )
        x = self.norm1(x + self_out)
        if self_w is not None:
            weights["self"] = self_w

        ########################### cross multi head attention ###########################
        cross_out, cross_w = self.cross_attn(
            x,
            memory,
            memory,
            key_padding_mask=memory_padding_mask,
            need_weights=need_weights,
        )
        x = self.norm2(x + cross_out)
        if cross_w is not None:
            weights["cross"] = cross_w

        ############################ feed forward network ###########################
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)

        return x, weights


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size: int, cfg: TransformerConfig):
        super().__init__()
        self.embed = TokenPositionalEmbedding(vocab_size, cfg.d_model, cfg.max_seq_len)
        self.layers = nn.ModuleList(
            [EncoderLayer(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout) for _ in range(cfg.n_encoder_layers)]
        )

    def forward(self, src_ids: Tensor, *, src_padding_mask: Optional[Tensor] = None) -> Tensor:
        x = self.embed(src_ids)
        for layer in self.layers:
            x = layer(x, src_padding_mask=src_padding_mask)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size: int, cfg: TransformerConfig):
        super().__init__()
        self.embed = TokenPositionalEmbedding(vocab_size, cfg.d_model, cfg.max_seq_len)
        self.layers = nn.ModuleList(
            [DecoderLayer(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout) for _ in range(cfg.n_decoder_layers)]
        )

    def forward(
        self,
        tgt_input_ids: Tensor,
        *,
        memory: Tensor,
        tgt_padding_mask: Optional[Tensor] = None,
        memory_padding_mask: Optional[Tensor] = None,
        causal_mask: Optional[Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        x = self.embed(tgt_input_ids)
        all_weights = []
        for layer in self.layers:
            x, weights = layer(
                x,
                memory=memory,
                tgt_padding_mask=tgt_padding_mask,
                memory_padding_mask=memory_padding_mask,
                causal_mask=causal_mask,
                need_weights=need_weights,
            )
            if need_weights:
                all_weights.append(weights)
        return x, {"layers": all_weights}


class TransformerNMT(nn.Module):
    def __init__(
        self,
        *,
        src_vocab_size: int,
        tgt_vocab_size: int,
        cfg: Optional[TransformerConfig] = None,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
    ):
        super().__init__()
        self.cfg = cfg or TransformerConfig()
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        self.encoder = TransformerEncoder(src_vocab_size, self.cfg)
        self.decoder = TransformerDecoder(tgt_vocab_size, self.cfg)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def encode(self, src_ids: torch.Tensor, *, src_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.encoder(src_ids, src_padding_mask=src_padding_mask)

    def decode(
        self,
        tgt_input_ids: torch.Tensor,
        *,
        memory: torch.Tensor,
        tgt_padding_mask: Optional[torch.Tensor] = None,
        memory_padding_mask: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        return self.decoder(
            tgt_input_ids,
            memory=memory,
            tgt_padding_mask=tgt_padding_mask,
            memory_padding_mask=memory_padding_mask,
            causal_mask=causal_mask,
            need_weights=need_weights,
        )

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_input_ids: torch.Tensor,
        *,
        src_padding_mask: Optional[torch.Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        memory = self.encode(src_ids, src_padding_mask=src_padding_mask)
        dec_out, weights = self.decode(
            tgt_input_ids,
            memory=memory,
            tgt_padding_mask=tgt_padding_mask,
            memory_padding_mask=src_padding_mask,
            causal_mask=causal_mask,
            need_weights=need_weights,
        )

        # Weight tying: output projection uses decoder token embedding matrix.
        logits = torch.matmul(dec_out, self.decoder.embed.token_emb.weight.t())  # (B, T, V)
        return logits, weights
