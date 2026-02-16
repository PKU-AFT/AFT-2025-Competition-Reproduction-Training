"""
多种时序模型的实现
包括：GRU, LSTM, GRU+Attention, Transformer, MLP, TCN
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseModel, register_model


# ============ GRU 模型 ============

@register_model("gru")
class GRUModel(BaseModel):
    """
    GRU 时序模型：取最后一步 hidden 做回归
    输入: [batch, T, F] -> 输出: [batch]
    """

    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__(input_dim)
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
        out, _ = self.backbone(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


# ============ GRU + Attention ============

class TemporalAttention(nn.Module):
    """时序注意力层"""

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


@register_model("gru_attn")
class GRUAttentionModel(BaseModel):
    """
    GRU + 时序注意力模型
    使用双向 GRU + Attention Pooling
    """

    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__(input_dim)
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
        return self.head(pooled).squeeze(-1)


# ============ LSTM 模型 ============

@register_model("lstm")
class LSTMModel(BaseModel):
    """
    LSTM 时序模型：取最后一步 hidden 做回归
    输入: [batch, T, F] -> 输出: [batch]
    """

    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__(input_dim)
        self.backbone = nn.LSTM(
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
        out, _ = self.backbone(x)
        return self.head(out[:, -1, :]).squeeze(-1)


# ============ LSTM + Attention ============

@register_model("lstm_attn")
class LSTMAttentionModel(BaseModel):
    """
    LSTM + 时序注意力模型
    使用双向 LSTM + Attention Pooling
    """

    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__(input_dim)
        self.input_drop = nn.Dropout(dropout)
        self.backbone = nn.LSTM(
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
        return self.head(pooled).squeeze(-1)


# ============ Transformer 模型 ============

class PositionalEncoding(nn.Module):
    """Transformer 位置编码"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


@register_model("transformer")
class TransformerModel(BaseModel):
    """
    Transformer Encoder 用于时序回归
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        nhead: int = 4,
        dim_feedforward: int = None,
    ):
        super().__init__(input_dim)
        dim_feedforward = dim_feedforward or hidden_size * 4

        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    @classmethod
    def from_config(cls, input_dim: int, cfg) -> "TransformerModel":
        return cls(
            input_dim=input_dim,
            hidden_size=cfg.HIDDEN_SIZE,
            num_layers=cfg.NUM_LAYERS,
            dropout=cfg.DROPOUT,
            nhead=getattr(cfg, "NHEAD", 4),
            dim_feedforward=getattr(cfg, "DIM_FEEDFORWARD", None),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        return self.head(x[:, -1, :]).squeeze(-1)


# ============ MLP 基线模型 ============

@register_model("mlp")
class MLPModel(BaseModel):
    """
    简单 MLP 基线，将序列展平后输入
    适用于快速验证和对比
    """

    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float, seq_len: int = 10):
        super().__init__(input_dim)
        self.seq_len = seq_len
        flat_dim = input_dim * seq_len

        layers = []
        in_dim = flat_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_size

        layers.append(nn.Linear(hidden_size, 1))
        self.network = nn.Sequential(*layers)

    @classmethod
    def from_config(cls, input_dim: int, cfg) -> "MLPModel":
        return cls(
            input_dim=input_dim,
            hidden_size=cfg.HIDDEN_SIZE,
            num_layers=cfg.NUM_LAYERS,
            dropout=cfg.DROPOUT,
            seq_len=cfg.STEP_LEN,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.network(x).squeeze(-1)


# ============ TCN 模型 ============

class Chomp1d(nn.Module):
    """移除因果卷积产生的填充（确保只使用历史信息）"""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T] 卷积后的输出
        Returns:
            [B, C, T - chomp_size] 移除右侧填充
        """
        if self.chomp_size <= 0:
            return x
        return x[:, :, : -self.chomp_size]


class TemporalBlock(nn.Module):
    """
    TCN 基本块：因果卷积 + 膨胀卷积 + 残差连接
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()

        # 因果卷积的 padding 数量
        padding = (kernel_size - 1) * dilation

        # 两个卷积层，使用 weight_norm 加速收敛
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        ))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.LeakyReLU(0.01)  # 使用 LeakyReLU 防止神经元死亡
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        ))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.LeakyReLU(0.01)
        self.dropout2 = nn.Dropout(dropout)

        # 组合成序列
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )

        # 下采样层（如果输入输出通道数不同）
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        )
        self.relu = nn.LeakyReLU(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_channels, T]
        Returns:
            [B, out_channels, T]
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


@register_model("tcn")
class TCNModel(BaseModel):
    """
    Temporal Convolutional Network (TCN)
    使用膨胀因果卷积捕捉时序依赖

    特点：
    - 并行计算效率高
    - 感受野指数增长：RF = 1 + (kernel-1) * sum(2^i for i in range(L))
    - 保证因果性（不使用未来信息）

    示例：
        kernel_size=2, num_layers=3: RF = 1 + 1*(1+2+4) = 8 (AFT-TCN配置，推荐)
        kernel_size=3, num_layers=3: RF = 1 + 2*(1+2+4) = 15

    注意：kernel_size=2时padding=dilation，效率显著优于kernel_size=3
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        kernel_size: int = 3,
    ):
        super().__init__(input_dim)
        self.kernel_size = kernel_size

        # 构建 TCN 层
        layers = []
        num_levels = num_layers
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else hidden_size
            out_channels = hidden_size

            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    dilation=dilation_size,
                    dropout=dropout,
                )
            )

        self.network = nn.Sequential(*layers)

        # 回归头 (简化版，类似AFT-TCN)
        self.head = nn.Linear(hidden_size, 1)

    @classmethod
    def from_config(cls, input_dim: int, cfg) -> "TCNModel":
        return cls(
            input_dim=input_dim,
            hidden_size=cfg.HIDDEN_SIZE,
            num_layers=getattr(cfg, "TCN_NUM_LAYERS", cfg.NUM_LAYERS),
            dropout=cfg.DROPOUT,
            kernel_size=getattr(cfg, "TCN_KERNEL_SIZE", 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, F]
        Returns:
            [B] 预测值
        """
        # 转换为 [B, F, T] 供 Conv1d 使用
        x = x.transpose(1, 2).contiguous()

        # 通过 TCN 网络
        x = self.network(x)  # [B, hidden_size, T]

        # 取最后时间步
        x = x[:, :, -1]  # [B, hidden_size]

        # 通过回归头
        return self.head(x).squeeze(-1)

    def get_receptive_field(self) -> int:
        """计算感受野大小
        TCN感受野公式: RF = 1 + (kernel_size - 1) * sum(2^i for i in range(num_layers))
        例如: kernel_size=3, num_layers=2 -> RF = 1 + 2 * (1 + 2) = 7
        """
        num_layers = len(self.network)
        return 1 + (self.kernel_size - 1) * sum(2**i for i in range(num_layers))


# ============ Griffin 模型 ============

class RMSNorm(nn.Module):
    """根均方归一化 (Root Mean Square Normalization)"""

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, H] 或 [B, H]
        rms = (x.square().mean(dim=-1, keepdim=True) + self.eps).sqrt()
        return x / rms * self.g


class RG_LRU(nn.Module):
    """
    Relaxed Gated Linear Recurrent Unit

    Griffin 的核心循环单元，使用门控线性递归实现高效的时序建模。
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.c = 8.0  # 数值稳定常数

        # 门控参数
        self.Wa = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.Wx = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.ba = nn.Parameter(torch.zeros(hidden_size))
        self.bx = nn.Parameter(torch.zeros(hidden_size))

        # 衰减参数 Λ
        self.Lambda = nn.Parameter(torch.empty(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        """初始化参数"""
        nn.init.xavier_uniform_(self.Wa, gain=0.01)
        nn.init.xavier_uniform_(self.Wx, gain=0.01)
        nn.init.zeros_(self.ba)
        nn.init.zeros_(self.bx)
        # 初始化 Λ 使得衰减在合理范围
        self.Lambda.data.uniform_(-2, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: [B, T, H] 输入序列

        Returns:
            [B, T, H] 输出序列
        """
        B, T, H = x.shape

        # 计算门控
        rt = torch.sigmoid(F.linear(x, self.Wa, self.ba))  # [B, T, H]
        it = torch.sigmoid(F.linear(x, self.Wx, self.bx))  # [B, T, H]

        # 计算衰减系数 (log space 保证数值稳定)
        log_a = -F.softplus(-self.Lambda)  # [H]
        a = torch.exp(log_a.unsqueeze(0).unsqueeze(0) * rt / self.c)  # [B, T, H]

        # 循环计算
        gate_x = it * x  # [B, T, H]
        scale = torch.sqrt(1 - a ** 2 + 1e-8)  # [B, T, H]

        h_list = []
        h = torch.zeros(B, H, device=x.device, dtype=x.dtype)
        for t in range(T):
            h = a[:, t] * h + scale[:, t] * gate_x[:, t]
            h_list.append(h)
        return torch.stack(h_list, dim=1)  # [B, T, H]


class GatedMLPBlock(nn.Module):
    """门控 MLP 块"""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.gate = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, H]
        gate = torch.sigmoid(self.gate(x))
        value = F.gelu(self.value(x))
        return self.output(gate * value)


class GriffinRecurrentBlock(nn.Module):
    """
    Griffin 循环块

    结合了：
    1. 线性投影 + GELU 激活
    2. 时序深度可分离卷积
    3. RG_LRU 循环单元
    """

    def __init__(self, input_dim: int, rnn_width: int, kernel_size: int = 3):
        super().__init__()
        self.rnn_width = rnn_width

        # 路径1: GELU 分支
        self.gelu_proj = nn.Linear(input_dim, rnn_width)

        # 路径2: 循环分支
        self.linear_proj = nn.Linear(input_dim, rnn_width)
        self.temporal_conv = nn.Conv1d(
            in_channels=rnn_width,
            out_channels=rnn_width,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=rnn_width,  # 深度可分离卷积
        )
        self.rg_lru = RG_LRU(rnn_width)

        # 输出投影
        self.output_proj = nn.Linear(rnn_width, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, input_dim]
        x_gelu = F.gelu(self.gelu_proj(x))  # [B, T, rnn_width]

        # 循环路径
        x_rnn = self.linear_proj(x)  # [B, T, rnn_width]
        x_rnn = x_rnn.transpose(1, 2)  # [B, rnn_width, T]
        x_rnn = self.temporal_conv(x_rnn)  # [B, rnn_width, T]
        x_rnn = x_rnn.transpose(1, 2)  # [B, T, rnn_width]
        x_rnn = self.rg_lru(x_rnn)  # [B, T, rnn_width]

        # 元素乘积融合
        combined = x_gelu * x_rnn
        return self.output_proj(combined)  # [B, T, input_dim]


class GriffinResidualBlock(nn.Module):
    """Griffin 残差块"""

    def __init__(self, input_dim: int, expansion_factor: int, rnn_width: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(input_dim)
        self.recurrent = GriffinRecurrentBlock(input_dim, rnn_width)
        self.norm2 = RMSNorm(input_dim)
        self.mlp = GatedMLPBlock(input_dim, input_dim * expansion_factor)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, H]
        residual = x
        x = self.norm1(x)
        x = self.recurrent(x)
        x = self.dropout(x)
        residual = residual + x

        x = self.norm2(residual)
        x = self.mlp(x)
        x = self.dropout(x)
        return residual + x


@register_model("griffin")
class GriffinModel(BaseModel):
    """
    Griffin 时序回归模型

    结合门控线性循环 (RG_LRU) 和局部时序卷积的高效时序模型。
    适配时序回归任务，输入 [B, T, F]，输出 [B]。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        expansion_factor: int = 2,
        rnn_width: int = None,
    ):
        super().__init__(input_dim)
        rnn_width = rnn_width or hidden_size

        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_size)

        # Griffin 残差块堆叠
        self.layers = nn.ModuleList([
            GriffinResidualBlock(hidden_size, expansion_factor, rnn_width, dropout)
            for _ in range(num_layers)
        ])

        # 输出层（取最后时间步做回归）
        self.norm = RMSNorm(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    @classmethod
    def from_config(cls, input_dim: int, cfg) -> "GriffinModel":
        return cls(
            input_dim=input_dim,
            hidden_size=cfg.HIDDEN_SIZE,
            num_layers=cfg.NUM_LAYERS,
            dropout=cfg.DROPOUT,
            expansion_factor=getattr(cfg, "GRIFFIN_EXPANSION", 2),
            rnn_width=getattr(cfg, "GRIFFIN_RNN_WIDTH", None),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        x = self.input_proj(x)  # [B, T, H]

        # 通过 Griffin 层（每一层都有外部残差连接）
        for layer in self.layers:
            x = x + layer(x)

        # 取最后时间步
        x = self.norm(x[:, -1, :])  # [B, H]

        # 回归头
        return self.head(x).squeeze(-1)  # [B]


# ============ 向后兼容别名 ============

# 保持与原代码的兼容性
TemporalModel = GRUModel
TemporalAttentionModel = GRUAttentionModel
