"""因子批量注册工具

提供多种便捷的因子注册方式，简化外部因子测试流程。
"""
from __future__ import annotations

import inspect
import importlib
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import pandas as pd

# 添加项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factors import FactorDefinition


# ============================================================
# 方式 1: 简单批量注册
# ============================================================

def register_factors(
    factors: Dict[str, Callable[[pd.DataFrame], pd.Series]],
    descriptions: Optional[Dict[str, str]] = None,
    prefix: str = "",
    suffix: str = "",
) -> Dict[str, FactorDefinition]:
    """
    从函数字典批量创建 FactorDefinition

    参数:
        factors: {因子名: 因子函数} 字典
        descriptions: {因子名: 描述} 字典（可选）
        prefix: 因子名前缀
        suffix: 因子名后缀

    返回:
        Dict[str, FactorDefinition]

    示例:
        >>> factors = register_factors({
        ...     'momentum': momentum_func,
        ...     'volume': volume_func,
        ... })
    """
    descriptions = descriptions or {}
    result = {}

    for name, func in factors.items():
        full_name = f"{prefix}{name}{suffix}"
        desc = descriptions.get(name, func.__doc__ or name)

        result[full_name] = FactorDefinition(
            name=full_name,
            function=func,
            description=desc.strip().split('\n')[0] if desc else name
        )

    return result


def register_factors_simple(
    *factor_funcs: Callable[[pd.DataFrame], pd.Series],
    prefix: str = "factor_"
) -> Dict[str, FactorDefinition]:
    """
    从函数列表批量注册（使用函数名作为因子名）

    参数:
        *factor_funcs: 因子函数（可变参数）
        prefix: 因子名前缀

    返回:
        Dict[str, FactorDefinition]

    示例:
        >>> def momentum_10d(df): ...
        >>> def volume_ratio(df): ...
        >>> factors = register_factors_simple(momentum_10d, volume_ratio)
        # 结果: {'factor_momentum_10d': ..., 'factor_volume_ratio': ...}
    """
    result = {}

    for func in factor_funcs:
        name = func.__name__
        if not name.startswith(prefix):
            name = f"{prefix}{name}"

        result[name] = FactorDefinition(
            name=name,
            function=func,
            description=func.__doc__.strip().split('\n')[0] if func.__doc__ else name
        )

    return result


# ============================================================
# 方式 2: 装饰器注册
# ============================================================

class FactorRegistry:
    """
    因子注册表（装饰器模式）

    示例:
        >>> registry = FactorRegistry()
        >>>
        >>> @registry.register
        ... def momentum_10d(df):
        ...     '''10日动量因子'''
        ...     return df.groupby('symbol')['close'].pct_change(10).shift(1)
        >>>
        >>> @registry.register(description="成交量比率")
        ... def volume_ratio(df):
        ...     return ...
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

        可以不带参数使用:
            @registry.register
            def my_factor(df): ...

        也可以带参数使用:
            @registry.register(description="我的因子")
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

        # 支持 @registry.register 和 @registry.register(...) 两种方式
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


# 全局注册表实例
global_registry = FactorRegistry()
register = global_registry.register  # 快捷方式


# ============================================================
# 方式 3: 从模块自动发现
# ============================================================

def discover_factors_from_module(
    module_path: str,
    pattern: str = "factor_*",
    exclude: Optional[List[str]] = None
) -> Dict[str, FactorDefinition]:
    """
    从模块自动发现并注册因子函数

    参数:
        module_path: 模块路径，如 'my_factors.momentum'
        pattern: 函数名匹配模式，支持 * 通配符
        exclude: 要排除的函数名列表

    返回:
        Dict[str, FactorDefinition]

    示例:
        >>> # 自动发现 my_factors 模块中所有 factor_ 开头的函数
        >>> factors = discover_factors_from_module('my_factors', pattern='factor_*')
    """
    import fnmatch

    exclude = exclude or []
    result = {}

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"无法导入模块 {module_path}: {e}")

    for name, obj in inspect.getmembers(module):
        # 检查是否是函数
        if not inspect.isfunction(obj):
            continue

        # 检查名称是否匹配模式
        if not fnmatch.fnmatch(name, pattern):
            continue

        # 检查是否在排除列表
        if name in exclude:
            continue

        # 检查签名（应该接受 DataFrame 参数）
        sig = inspect.signature(obj)
        if len(sig.parameters) < 1:
            continue

        # 注册
        result[name] = FactorDefinition(
            name=name,
            function=obj,
            description=obj.__doc__.strip().split('\n')[0] if obj.__doc__ else name
        )

    return result


def discover_factors_from_file(
    file_path: str,
    pattern: str = "factor_*",
    exclude: Optional[List[str]] = None
) -> Dict[str, FactorDefinition]:
    """
    从 Python 文件自动发现并注册因子函数

    参数:
        file_path: Python 文件路径
        pattern: 函数名匹配模式
        exclude: 要排除的函数名列表

    返回:
        Dict[str, FactorDefinition]

    示例:
        >>> factors = discover_factors_from_file('my_factors/momentum.py')
    """
    file_path = Path(file_path).resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    # 动态导入
    spec = importlib.util.spec_from_file_location("dynamic_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # 复用模块发现逻辑
    import fnmatch
    exclude = exclude or []
    result = {}

    for name in dir(module):
        obj = getattr(module, name)

        if not inspect.isfunction(obj):
            continue
        if not fnmatch.fnmatch(name, pattern):
            continue
        if name in exclude:
            continue

        sig = inspect.signature(obj)
        if len(sig.parameters) < 1:
            continue

        result[name] = FactorDefinition(
            name=name,
            function=obj,
            description=obj.__doc__.strip().split('\n')[0] if obj.__doc__ else name
        )

    return result


# ============================================================
# 方式 4: 从配置批量创建
# ============================================================

@dataclass
class FactorSpec:
    """因子规格"""
    name: str
    function: Callable[[pd.DataFrame], pd.Series]
    description: str = ""
    is_intentional_leak: bool = False


def register_from_specs(
    specs: List[FactorSpec]
) -> Dict[str, FactorDefinition]:
    """
    从规格列表批量创建

    示例:
        >>> specs = [
        ...     FactorSpec('momentum', momentum_func, '动量因子'),
        ...     FactorSpec('volume', volume_func, '成交量因子'),
        ... ]
        >>> factors = register_from_specs(specs)
    """
    return {
        spec.name: FactorDefinition(
            name=spec.name,
            function=spec.function,
            description=spec.description or spec.name,
            is_intentional_leak=spec.is_intentional_leak
        )
        for spec in specs
    }


def register_from_tuples(
    factor_tuples: List[Tuple[str, Callable, str]]
) -> Dict[str, FactorDefinition]:
    """
    从元组列表批量创建

    参数:
        factor_tuples: [(名称, 函数, 描述), ...]

    示例:
        >>> factors = register_from_tuples([
        ...     ('momentum', momentum_func, '动量因子'),
        ...     ('volume', volume_func, '成交量因子'),
        ... ])
    """
    return {
        name: FactorDefinition(
            name=name,
            function=func,
            description=desc
        )
        for name, func, desc in factor_tuples
    }


# ============================================================
# 方式 5: 链式构建器
# ============================================================

class FactorBuilder:
    """
    链式因子构建器

    示例:
        >>> factors = (FactorBuilder()
        ...     .add('momentum', momentum_func, '动量因子')
        ...     .add('volume', volume_func, '成交量因子')
        ...     .add_leak('future_ret', future_func, '测试泄露')
        ...     .build())
    """

    def __init__(self):
        self._factors: List[Tuple[str, Callable, str, bool]] = []

    def add(
        self,
        name: str,
        func: Callable[[pd.DataFrame], pd.Series],
        description: str = ""
    ) -> 'FactorBuilder':
        """添加因子"""
        self._factors.append((name, func, description or name, False))
        return self

    def add_leak(
        self,
        name: str,
        func: Callable[[pd.DataFrame], pd.Series],
        description: str = ""
    ) -> 'FactorBuilder':
        """添加故意泄露的因子（用于测试）"""
        self._factors.append((name, func, description or name, True))
        return self

    def build(self) -> Dict[str, FactorDefinition]:
        """构建因子字典"""
        return {
            name: FactorDefinition(
                name=name,
                function=func,
                description=desc,
                is_intentional_leak=is_leak
            )
            for name, func, desc, is_leak in self._factors
        }


# ============================================================
# 便捷函数
# ============================================================

def quick_register(**kwargs) -> Dict[str, FactorDefinition]:
    """
    最简单的批量注册方式

    示例:
        >>> factors = quick_register(
        ...     momentum=momentum_func,
        ...     volume=volume_func,
        ...     volatility=volatility_func
        ... )
    """
    return {
        name: FactorDefinition(
            name=name,
            function=func,
            description=func.__doc__.strip().split('\n')[0] if func.__doc__ else name
        )
        for name, func in kwargs.items()
    }


def merge_factors(*factor_dicts: Dict[str, FactorDefinition]) -> Dict[str, FactorDefinition]:
    """
    合并多个因子字典

    示例:
        >>> all_factors = merge_factors(momentum_factors, volume_factors, custom_factors)
    """
    result = {}
    for d in factor_dicts:
        result.update(d)
    return result


# ============================================================
# 使用示例汇总
# ============================================================

def show_examples():
    """展示所有注册方式的示例"""

    print("=" * 70)
    print("因子批量注册方式示例")
    print("=" * 70)

    # 示例因子函数
    def momentum_10d(df):
        """10日动量因子"""
        return df.groupby('symbol')['close'].pct_change(10).shift(1)

    def volume_ratio(df):
        """成交量比率因子"""
        return df.groupby('symbol')['volume'].transform(
            lambda x: x / x.rolling(20).mean()
        ).shift(1)

    def volatility_20d(df):
        """20日波动率因子"""
        return df.groupby('symbol')['close'].pct_change().transform(
            lambda x: x.rolling(20).std()
        ).shift(1)

    # 方式 1: 简单批量注册
    print("\n方式 1: register_factors()")
    factors1 = register_factors({
        'momentum': momentum_10d,
        'volume': volume_ratio,
    })
    print(f"  注册了 {len(factors1)} 个因子: {list(factors1.keys())}")

    # 方式 2: 更简单的方式
    print("\n方式 2: register_factors_simple()")
    factors2 = register_factors_simple(momentum_10d, volume_ratio, volatility_20d)
    print(f"  注册了 {len(factors2)} 个因子: {list(factors2.keys())}")

    # 方式 3: 装饰器
    print("\n方式 3: FactorRegistry 装饰器")
    registry = FactorRegistry()

    @registry.register
    def my_momentum(df):
        """我的动量因子"""
        return df.groupby('symbol')['close'].pct_change().shift(1)

    @registry.register(description="自定义描述")
    def my_volume(df):
        return df['volume'].shift(1)

    print(f"  注册了 {len(registry)} 个因子: {registry.list_names()}")

    # 方式 4: 最简单的方式
    print("\n方式 4: quick_register()")
    factors4 = quick_register(
        momentum=momentum_10d,
        volume=volume_ratio
    )
    print(f"  注册了 {len(factors4)} 个因子: {list(factors4.keys())}")

    # 方式 5: 链式构建器
    print("\n方式 5: FactorBuilder 链式构建")
    factors5 = (FactorBuilder()
        .add('momentum', momentum_10d, '动量因子')
        .add('volume', volume_ratio, '成交量因子')
        .build())
    print(f"  注册了 {len(factors5)} 个因子: {list(factors5.keys())}")

    # 方式 6: 从元组列表
    print("\n方式 6: register_from_tuples()")
    factors6 = register_from_tuples([
        ('momentum', momentum_10d, '动量因子'),
        ('volume', volume_ratio, '成交量因子'),
    ])
    print(f"  注册了 {len(factors6)} 个因子: {list(factors6.keys())}")

    print("\n" + "=" * 70)
    print("完整代码示例见上方各函数的 docstring")
    print("=" * 70)


if __name__ == "__main__":
    show_examples()
