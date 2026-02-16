import math
import torch
import torch.nn as nn


class Gate(nn.Module):
    def __init__(self, d_input: int, d_output: int, beta: float = 1.0):
        super().__init__()
        self.trans = nn.Linear(d_input, d_output)
        self.d_output = d_output
        self.t = beta if beta is not None else 1.0

    def forward(self, gate_input: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.trans(gate_input) / self.t, dim=-1)
        return self.d_output * weights


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(-2)
        pe = self.pe[:seq_len]
        if x.dim() == 3:
            return x + pe
        if x.dim() == 4:
            return x + pe.view(1, 1, seq_len, -1)
        raise ValueError(f"Unsupported input shape {tuple(x.shape)} for positional encoding")


class TAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.ModuleList([nn.Dropout(p=dropout) for _ in range(nhead)]) if dropout > 0 else None

        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(p=dropout),
        )

    # [N, T, F]
    # [N, T, F] * [N, F, T] -> [N, T, T]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        single_batch = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            single_batch = True
        elif x.dim() != 4:
            raise ValueError(f"Unsupported input shape {tuple(x.shape)} for TAttention")

        x_norm = self.norm1(x)
        q = self.qtrans(x_norm)
        k = self.ktrans(x_norm)
        v = self.vtrans(x_norm)

        B, N, T, D = q.shape
        dim = D // self.nhead

        q = q.reshape(B * N, T, D)
        k = k.reshape(B * N, T, D)
        v = v.reshape(B * N, T, D)

        att_output = []
        for i in range(self.nhead):
            if i == self.nhead - 1:
                qh = q[:, :, i * dim :]
                kh = k[:, :, i * dim :]
                vh = v[:, :, i * dim :]
            else:
                qh = q[:, :, i * dim : (i + 1) * dim]
                kh = k[:, :, i * dim : (i + 1) * dim]
                vh = v[:, :, i * dim : (i + 1) * dim]

            scores = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)), dim=-1)
            if self.attn_dropout is not None:
                scores = self.attn_dropout[i](scores)
            att_output.append(torch.matmul(scores, vh))

        att_output = torch.cat(att_output, dim=-1).reshape(B, N, T, D)
        xt = x + att_output
        xt = self.norm2(xt)
        out = xt + self.ffn(xt)

        if single_batch:
            return out.squeeze(0)
        return out


class SAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.temperature = math.sqrt(self.d_model / nhead)

        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.ModuleList([nn.Dropout(p=dropout) for _ in range(nhead)]) if dropout > 0 else None

        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        single_batch = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            single_batch = True
        elif x.dim() != 4:
            raise ValueError(f"Unsupported input shape {tuple(x.shape)} for SAttention")

        x_norm = self.norm1(x)
        q = self.qtrans(x_norm)
        k = self.ktrans(x_norm)
        v = self.vtrans(x_norm)

        B, N, T, D = q.shape
        dim = D // self.nhead

        # reshape to (B*T, N, D) so each time step attends across stocks independently
        q = q.permute(0, 2, 1, 3).reshape(B * T, N, D)
        k = k.permute(0, 2, 1, 3).reshape(B * T, N, D)
        v = v.permute(0, 2, 1, 3).reshape(B * T, N, D)

        att_output = []
        for i in range(self.nhead):
            if i == self.nhead - 1:
                qh = q[:, :, i * dim :]
                kh = k[:, :, i * dim :]
                vh = v[:, :, i * dim :]
            else:
                qh = q[:, :, i * dim : (i + 1) * dim]
                kh = k[:, :, i * dim : (i + 1) * dim]
                vh = v[:, :, i * dim : (i + 1) * dim]

            scores = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)) / self.temperature, dim=-1)
            if self.attn_dropout is not None:
                scores = self.attn_dropout[i](scores)
            att_output.append(torch.matmul(scores, vh))

        att_output = torch.cat(att_output, dim=-1).reshape(B, T, N, D).permute(0, 2, 1, 3)
        xt = x + att_output
        xt = self.norm2(xt)
        out = xt + self.ffn(xt)

        if single_batch:
            return out.squeeze(0)
        return out


class TemporalAttention(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.trans = nn.Linear(d_model, d_model, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 3:
            z_in = z
        elif z.dim() == 4:
            B, N, T, D = z.shape
            z_in = z.view(B * N, T, D)
        else:
            raise ValueError(f"Unsupported input shape {tuple(z.shape)} for TemporalAttention")

        h = self.trans(z_in)
        query = h[:, -1, :].unsqueeze(-1)
        lam = torch.matmul(h, query).squeeze(-1)
        lam = torch.softmax(lam, dim=1).unsqueeze(1)
        output = torch.matmul(lam, z_in).squeeze(1)

        if z.dim() == 3:
            return output
        return output.view(B, N, -1)


class MASTER(nn.Module):
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

        self.feature_gate = Gate(d_input=gate_feat_dim, d_output=d_feat, beta=beta)

        self.input_proj = nn.Linear(d_feat, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.temporal_attn = TAttention(d_model=d_model, nhead=t_nhead, dropout=t_dropout)
        self.spatial_attn = SAttention(d_model=d_model, nhead=s_nhead, dropout=s_dropout)
        self.temporal_pool = TemporalAttention(d_model=d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        single_batch = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            single_batch = True
        elif x.dim() != 4:
            raise ValueError(f"Unsupported input shape {tuple(x.shape)} for MASTER")

        # split raw features and gate features
        feat_dim = x.size(-1)
        expected_dim = self.raw_feat_dim + self.gate_feat_dim
        if feat_dim != expected_dim:
            raise ValueError(f"Expected feature dim {expected_dim} (raw {self.raw_feat_dim} + gate {self.gate_feat_dim}), got {feat_dim}")

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
        out = self.head(h).squeeze(-1)

        if single_batch:
            return out.squeeze(0)
        return out






# [N, T, F] -> [N, T, H]