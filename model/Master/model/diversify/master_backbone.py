import torch
import torch.nn as nn

from model.MASTER import Gate, PositionalEncoding, SAttention, TAttention, TemporalAttention


class MasterBackbone(nn.Module):
    def __init__(
        self,
        d_feat: int,
        gate_feat_dim: int = 100,
        d_model: int = 256,
        t_nhead: int = 4,
        s_nhead: int = 2,
        t_dropout: float = 0.5,
        s_dropout: float = 0.5,
        beta: float = 1.0,
    ):
        super().__init__()
        self.raw_feat_dim = d_feat
        self.gate_feat_dim = gate_feat_dim
        self.out_dim = d_model

        self.feature_gate = Gate(d_input=gate_feat_dim, d_output=d_feat, beta=beta)
        self.input_proj = nn.Linear(d_feat, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.temporal_attn = TAttention(d_model=d_model, nhead=t_nhead, dropout=t_dropout)
        self.spatial_attn = SAttention(d_model=d_model, nhead=s_nhead, dropout=s_dropout)
        self.temporal_pool = TemporalAttention(d_model=d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        single_batch = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            single_batch = True
        elif x.dim() != 4:
            raise ValueError(f"Unsupported input shape {tuple(x.shape)} for MasterBackbone")

        feat_dim = x.size(-1)
        expected_dim = self.raw_feat_dim + self.gate_feat_dim
        if feat_dim != expected_dim:
            raise ValueError(
                f"Expected feature dim {expected_dim} (raw {self.raw_feat_dim} + gate {self.gate_feat_dim}), got {feat_dim}"
            )

        raw_feat = x[..., : self.raw_feat_dim]
        gate_input = x[..., -1, self.raw_feat_dim :]
        gate_weights = self.feature_gate(gate_input)
        if gate_weights.dim() == 2:
            gated_feat = raw_feat * gate_weights.unsqueeze(1)
        else:
            gated_feat = raw_feat * gate_weights.unsqueeze(2)

        h = self.input_proj(gated_feat)
        h = self.pos_encoding(h)
        h = self.temporal_attn(h)
        h = self.spatial_attn(h)
        h = self.temporal_pool(h)

        if single_batch:
            return h.squeeze(0)
        return h
