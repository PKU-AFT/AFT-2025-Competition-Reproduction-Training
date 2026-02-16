"""Factor registry for the onboarding exercise."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import pandas as pd

from factors import sample_factors

FactorFunction = Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class FactorDefinition:
    """Metadata for a factor function."""

    name: str
    function: FactorFunction
    description: str
    is_intentional_leak: bool = False


# ============================================================
# 装饰器注册系统
# ============================================================

class FactorRegistry:
    """
    因子注册表（装饰器模式）

    使用方法:
        >>> registry = FactorRegistry()
        >>>
        >>> @registry.register
        ... def my_factor(df):
        ...     '''我的因子'''
        ...     return df['close'].shift(1)
        >>>
        >>> factors = registry.get_all()
    """

    def __init__(self, prefix: str = ""):
        self._factors: Dict[str, FactorDefinition] = {}
        self.prefix = prefix

    def register(
        self,
        func: Optional[Callable] = None,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        is_intentional_leak: bool = False
    ):
        """
        装饰器：注册因子

        支持两种用法:
            @registry.register
            def my_factor(df): ...

            @registry.register(description="自定义描述")
            def my_factor(df): ...
        """
        def decorator(fn):
            factor_name = name or fn.__name__
            if self.prefix and not factor_name.startswith(self.prefix):
                factor_name = f"{self.prefix}{factor_name}"

            factor_desc = description or (
                fn.__doc__.strip().split('\n')[0] if fn.__doc__ else factor_name
            )

            self._factors[factor_name] = FactorDefinition(
                name=factor_name,
                function=fn,
                description=factor_desc,
                is_intentional_leak=is_intentional_leak
            )
            return fn

        if func is not None:
            return decorator(func)
        return decorator

    def get_all(self) -> Dict[str, FactorDefinition]:
        """获取所有注册的因子"""
        return self._factors.copy()

    def get(self, name: str) -> Optional[FactorDefinition]:
        """获取单个因子"""
        return self._factors.get(name)

    def list_names(self) -> List[str]:
        """列出所有因子名称"""
        return list(self._factors.keys())

    def clear(self):
        """清空注册表"""
        self._factors.clear()

    def __len__(self):
        return len(self._factors)

    def __contains__(self, name: str):
        return name in self._factors


# 全局注册表实例（用于装饰器方式）
global_registry = FactorRegistry()
register = global_registry.register  # 快捷方式


def get_registered_factors() -> Dict[str, FactorDefinition]:
    """Return the factor definitions that ship with the starter project."""

    return {
        "factor_momentum_ema": FactorDefinition(
            name="factor_momentum_ema",
            function=sample_factors.factor_momentum_ema,
            description="Lagged EMA of 1-day returns (should be safe)",
        ),
        "factor_volume_mean5": FactorDefinition(
            name="factor_volume_mean5",
            function=sample_factors.factor_volume_mean5,
            description="Lagged mean volume over a 5-day window",
        ),
        "factor_future_return": FactorDefinition(
            name="factor_future_return",
            function=sample_factors.factor_future_return,
            description="Intentional leak: aligns future return onto today's row",
            is_intentional_leak=True,
        ),
        # Additional safe factors
        "factor_ret_1d_lag1": FactorDefinition(
            name="factor_ret_1d_lag1",
            function=sample_factors.factor_ret_1d_lag1,
            description="Yesterday's 1-day return (lagged)",
        ),
        "factor_sma_5": FactorDefinition(
            name="factor_sma_5",
            function=sample_factors.factor_sma_5,
            description="5-day SMA of close, lagged by 1 day",
        ),
        "factor_volatility_10": FactorDefinition(
            name="factor_volatility_10",
            function=sample_factors.factor_volatility_10,
            description="10-day rolling std of 1d returns, lagged",
        ),
        "factor_rsi_14": FactorDefinition(
            name="factor_rsi_14",
            function=sample_factors.factor_rsi_14,
            description="14-period RSI on close, lagged",
        ),
        "factor_obv_lag1": FactorDefinition(
            name="factor_obv_lag1",
            function=sample_factors.factor_obv_lag1,
            description="On-balance volume cumulative line, lagged by 1 day",
        ),
        # Additional intentional leaks
        "factor_future_close": FactorDefinition(
            name="factor_future_close",
            function=sample_factors.factor_future_close,
            description="Intentional leak: tomorrow's close aligned to today",
            is_intentional_leak=True,
        ),
        "factor_future_volume": FactorDefinition(
            name="factor_future_volume",
            function=sample_factors.factor_future_volume,
            description="Intentional leak: tomorrow's volume aligned to today",
            is_intentional_leak=True,
        ),
    }
