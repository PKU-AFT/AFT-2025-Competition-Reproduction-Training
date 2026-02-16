"""
模型模块初始化
导出统一的模型创建接口和所有可用模型
"""

# ============ 核心接口 ============
from .base import (
    BaseModel,
    MODEL_REGISTRY,
    register_model,
    get_model,
    list_models,
    create_model,
)

# ============ 导入所有模型以触发注册 ============
from . import models

# ============ 向后兼容导出 ============
# 原有代码中使用的类名
from .models import (
    TemporalModel,           # 原 GRU 模型
    TemporalAttentionModel,  # 原 GRU+Attention 模型
    GRUModel,                # 新名称
    GRUAttentionModel,       # 新名称
    LSTMModel,
    LSTMAttentionModel,
    TransformerModel,
    TCNModel,
    MLPModel,
)

__all__ = [
    # 核心接口
    "BaseModel",
    "MODEL_REGISTRY",
    "register_model",
    "get_model",
    "list_models",
    "create_model",
    # 向后兼容
    "TemporalModel",
    "TemporalAttentionModel",
    "GRUModel",
    "GRUAttentionModel",
    "LSTMModel",
    "LSTMAttentionModel",
    "TransformerModel",
    "TCNModel",
    "MLPModel",
]
