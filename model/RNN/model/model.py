import torch
import torch.nn as nn


class TemporalModel(nn.Module):
    """Basic GRU encoder that uses the last hidden state for regression."""

    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.backbone = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        out, _ = self.backbone(x)
        last = out[:, -1, :]
        pred = self.head(last).squeeze(-1)
        return pred


class TemporalAttention(nn.Module):
    """Temporal attention pooling to aggregate sequence features."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, H]
        scores = self.attn(x)  # [B, T, 1]
        weights = torch.softmax(scores, dim=1)
        return (x * weights).sum(dim=1)  # [B, H]


class TemporalAttentionModel(nn.Module):
    """Bidirectional GRU encoder with temporal attention pooling."""

    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.input_drop = nn.Dropout(dropout)
        self.backbone = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        hidden_out = hidden_size * 2
        self.attention = TemporalAttention(hidden_out)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_out),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_out, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_drop(x)
        out, _ = self.backbone(x)
        pooled = self.attention(out)
        pred = self.head(pooled).squeeze(-1)
        return pred
