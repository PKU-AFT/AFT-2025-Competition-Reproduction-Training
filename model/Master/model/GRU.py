import torch
import torch.nn as nn


class TemporalModel(nn.Module):
    """
    简单时序模型：GRU 编码时间序列，取最后一步 hidden 做回归。
    输入:
      - [N, T, F] 单组
      - [B, N, T, F] 分组批
    输出:
      - [N] 或 [B, N] 对应标量预测
    """

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
        """
        支持两种形状:
        - [N, T, F] 单组输入
        - [B, N, T, F] 分组批次
        """
        if x.dim() == 4:
            B, N, T, F = x.shape
            x_flat = x.reshape(B * N, T, F)
        elif x.dim() == 3:
            B = 1
            N, T, F = x.shape
            x_flat = x
        else:
            raise ValueError(f"Unsupported input shape {tuple(x.shape)}")

        out, _ = self.backbone(x_flat)  # [B*N, T, H]
        last = out[:, -1, :]  # 取最后一个时间步
        pred = self.head(last).squeeze(-1)  # [B*N]

        if x.dim() == 4:
            return pred.view(B, N)
        return pred  # [N]
