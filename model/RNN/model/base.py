"""
模型基类和注册机制
支持统一的模型接口和动态模型创建
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
import torch.nn as nn


class BaseModel(ABC, nn.Module):
    """
    所有时序模型的基类，定义统一接口。

    输入: [batch, seq_len, features]
    输出: [batch] (回归值)
    """

    def __init__(self, input_dim: int, **kwargs):
        super().__init__()
        self.input_dim = input_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: [B, T, F] 输入序列

        Returns:
            pred: [B] 预测值
        """
        pass

    @classmethod
    def from_config(cls, input_dim: int, cfg: Any) -> "BaseModel":
        """
        从配置创建模型实例

        Args:
            input_dim: 输入特征维度
            cfg: 配置模块（config.py）

        Returns:
            模型实例
        """
        return cls(
            input_dim=input_dim,
            hidden_size=cfg.HIDDEN_SIZE,
            num_layers=cfg.NUM_LAYERS,
            dropout=cfg.DROPOUT,
        )

    def get_num_params(self) -> int:
        """返回可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============ 模型注册机制 ============

MODEL_REGISTRY: Dict[str, type] = {}


def register_model(name: str):
    """
    模型注册装饰器

    用法:
        @register_model("gru")
        class GRUModel(BaseModel):
            ...
    """
    def decorator(cls):
        if name in MODEL_REGISTRY:
            raise ValueError(f"模型名称 '{name}' 已被注册")
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_model(name: str) -> type:
    """
    根据名称获取模型类

    Args:
        name: 模型名称

    Returns:
        模型类

    Raises:
        ValueError: 如果模型名称未注册
    """
    if name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"未知模型: '{name}'。可用模型: {available}")
    return MODEL_REGISTRY[name]


def list_models() -> list:
    """返回所有已注册的模型名称"""
    return list(MODEL_REGISTRY.keys())


def create_model(name: str, input_dim: int, cfg: Any) -> BaseModel:
    """
    工厂函数：根据名称创建模型实例

    Args:
        name: 模型名称（如 "gru", "lstm", "transformer"）
        input_dim: 输入特征维度
        cfg: 配置模块

    Returns:
        模型实例
    """
    model_cls = get_model(name)
    return model_cls.from_config(input_dim, cfg)
