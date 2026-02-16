"""
示例：使用装饰器注册因子

这个文件展示如何使用装饰器模式注册因子，相比传统方式更简洁优雅。

使用方法:
    from factors.sample_factors_decorator import decorator_registry

    # 获取所有装饰器注册的因子
    factors = decorator_registry.get_all()
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from factors import FactorRegistry

# ============================================================
# 创建装饰器注册表
# ============================================================

# 方式 1: 使用本地注册表（推荐用于模块化管理）
decorator_registry = FactorRegistry(prefix="")

# 方式 2: 也可以使用全局注册表
# from factors import global_registry, register


# ============================================================
# 安全因子 - 使用装饰器注册
# ============================================================

@decorator_registry.register
def factor_momentum_ema_v2(df: pd.DataFrame) -> pd.Series:
    """Lagged exponential moving average of 1-day returns (decorator version)."""
    ret_1d = df.groupby("symbol")["close"].pct_change()
    ema = ret_1d.groupby(df["symbol"]).transform(
        lambda x: x.ewm(span=10, adjust=False).mean()
    )
    return ema.groupby(df["symbol"]).shift(1)


@decorator_registry.register
def factor_volume_mean5_v2(df: pd.DataFrame) -> pd.Series:
    """Lagged mean volume over a 5-day window (decorator version)."""
    vol_mean = df.groupby("symbol")["volume"].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    return vol_mean.groupby(df["symbol"]).shift(1)


@decorator_registry.register
def factor_ret_1d_lag1_v2(df: pd.DataFrame) -> pd.Series:
    """Yesterday's 1-day return (decorator version)."""
    ret = df.groupby("symbol")["close"].pct_change()
    return ret.groupby(df["symbol"]).shift(1)


@decorator_registry.register
def factor_sma_5_v2(df: pd.DataFrame) -> pd.Series:
    """5-day simple moving average of close, lagged by 1 day (decorator version)."""
    sma = df.groupby("symbol")["close"].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    return sma.groupby(df["symbol"]).shift(1)


@decorator_registry.register
def factor_volatility_10_v2(df: pd.DataFrame) -> pd.Series:
    """10-day rolling standard deviation of 1d returns (decorator version)."""
    ret = df.groupby("symbol")["close"].pct_change()
    vol = ret.groupby(df["symbol"]).transform(
        lambda x: x.rolling(window=10, min_periods=2).std()
    )
    return vol.groupby(df["symbol"]).shift(1)


@decorator_registry.register(description="RSI指标（装饰器版本）- 14期RSI滞后1天")
def factor_rsi_14_v2(df: pd.DataFrame) -> pd.Series:
    """14-period RSI on close, lagged (decorator version with custom description)."""
    def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    rsi = df.groupby("symbol")["close"].transform(compute_rsi)
    return rsi.groupby(df["symbol"]).shift(1)


@decorator_registry.register
def factor_obv_lag1_v2(df: pd.DataFrame) -> pd.Series:
    """On-balance volume cumulative line, lagged by 1 day (decorator version)."""
    def compute_obv(group: pd.DataFrame) -> pd.Series:
        price_change = group["close"].diff()
        obv = (
            (price_change > 0).astype(int) * group["volume"]
            - (price_change < 0).astype(int) * group["volume"]
        )
        return obv.cumsum()

    obv = df.groupby("symbol", group_keys=False).apply(compute_obv)
    return obv.groupby(df["symbol"]).shift(1)


# ============================================================
# 故意泄露因子 - 使用装饰器注册
# ============================================================

@decorator_registry.register(
    description="故意泄露：明天的收益率对齐到今天",
    is_intentional_leak=True
)
def factor_future_return_v2(df: pd.DataFrame) -> pd.Series:
    """Intentional leak: aligns tomorrow's return onto today's row (decorator version)."""
    ret_1d = df.groupby("symbol")["close"].pct_change()
    return ret_1d.groupby(df["symbol"]).shift(-1)


@decorator_registry.register(is_intentional_leak=True)
def factor_future_close_v2(df: pd.DataFrame) -> pd.Series:
    """Intentional leak: tomorrow's close aligned to today (decorator version)."""
    return df.groupby("symbol")["close"].shift(-1)


@decorator_registry.register(is_intentional_leak=True)
def factor_future_volume_v2(df: pd.DataFrame) -> pd.Series:
    """Intentional leak: tomorrow's volume aligned to today (decorator version)."""
    return df.groupby("symbol")["volume"].shift(-1)


# ============================================================
# 便捷函数
# ============================================================

def get_decorator_factors():
    """获取所有装饰器注册的因子"""
    return decorator_registry.get_all()


def list_decorator_factor_names():
    """列出所有装饰器注册的因子名称"""
    return decorator_registry.list_names()


# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("装饰器注册因子示例")
    print("=" * 70)

    print(f"\n注册了 {len(decorator_registry)} 个因子:")
    for name in decorator_registry.list_names():
        factor_def = decorator_registry.get(name)
        leak_marker = " [LEAK]" if factor_def.is_intentional_leak else ""
        print(f"  • {name}{leak_marker}")
        print(f"    {factor_def.description}")

    print("\n" + "=" * 70)
    print("装饰器的优势:")
    print("=" * 70)
    print("1. 自动注册 - 定义时即完成注册，无需手动维护注册字典")
    print("2. 代码更简洁 - 减少样板代码")
    print("3. 可读性强 - 因子定义和注册信息在一起")
    print("4. 支持参数 - 可自定义描述、泄露标记等")
    print("5. 灵活性高 - 可创建多个注册表，分模块管理")

    print("\n" + "=" * 70)
    print("使用方法:")
    print("=" * 70)
    print("""
# 方法 1: 导入本地注册表
from factors.sample_factors_decorator import decorator_registry
factors = decorator_registry.get_all()

# 方法 2: 使用全局注册表
from factors import global_registry
factors = global_registry.get_all()

# 方法 3: 在自己的代码中使用
from factors import FactorRegistry

registry = FactorRegistry()

@registry.register
def my_factor(df):
    return df['close'].shift(1)

factors = registry.get_all()
    """)
