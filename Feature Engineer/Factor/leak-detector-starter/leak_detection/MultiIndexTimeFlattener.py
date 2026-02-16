"""
MultiIndex 时间展平器

将多层 MultiIndex 展平为单层连续时间索引，便于时间序列因子计算。

典型场景：
- date_id (日期ID) + bucket_in_seconds (日内时间桶) → 连续时间戳
- trading_day + intraday_period → 完整时间序列
"""

from __future__ import annotations
from typing import Optional, Callable, Dict, Any, List, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@dataclass
class MultiIndexFlattenConfig:
    """MultiIndex 展平配置

    Parameters
    ----------
    base_date : str or datetime
        基准日期，用于将相对日期ID转换为实际日期
        示例: '2020-01-01' 或 datetime(2020, 1, 1)

    time_unit : str, default 'seconds'
        时间单位，支持 'days', 'hours', 'minutes', 'seconds'
        用于将时间偏移量转换为 timedelta

    index_name : str, default 'timestamp'
        生成的新索引名称

    drop_multiindex : bool, default True
        是否删除原始 MultiIndex（转换为普通列）

    sort_index : bool, default True
        是否对生成的时间索引排序

    tz : str, optional
        时区设置，如 'Asia/Shanghai', 'UTC'

    verbose : bool, default False
        是否输出详细日志
    """
    base_date: str | datetime = '2020-01-01'
    time_unit: str = 'seconds'
    index_name: str = 'timestamp'
    drop_multiindex: bool = True
    sort_index: bool = True
    tz: Optional[str] = None
    verbose: bool = False


class MultiIndexTimeFlattener:
    """
    MultiIndex 时间展平器

    将多层 MultiIndex（如 date_id + bucket）展平为连续的单层时间索引。

    使用场景
    --------
    场景1: 高频交易数据
        - Level 0: date_id (0, 1, 2, ..., 480) - 交易日序号
        - Level 1: bucket_in_seconds (0, 60, 120, ..., 54*60) - 日内时间桶

    场景2: 分钟级数据
        - Level 0: trading_day (日期)
        - Level 1: minute_of_day (0-389) - 交易分钟数

    场景3: 自定义时间层级
        - 任意两层或多层时间表示
        - 通过自定义转换函数实现

    Examples
    --------
    基础使用（date_id + bucket_in_seconds）:

    >>> df = pd.DataFrame({
    ...     'symbol': ['AAPL'] * 100,
    ...     'close': np.random.randn(100)
    ... })
    >>> df.index = pd.MultiIndex.from_product([
    ...     range(2),      # date_id: 0, 1
    ...     range(0, 50)   # bucket_in_seconds: 0-49
    ... ], names=['date_id', 'bucket_in_seconds'])
    >>>
    >>> flattener = MultiIndexTimeFlattener(
    ...     level_names=['date_id', 'bucket_in_seconds'],
    ...     base_date='2023-01-01'
    ... )
    >>> df_flat = flattener.transform(df)
    >>> # 结果: 单层 DatetimeIndex

    自定义转换:

    >>> def custom_converter(levels):
    ...     date_id, bucket = levels
    ...     base = pd.Timestamp('2023-01-01')
    ...     return base + pd.Timedelta(days=date_id) + pd.Timedelta(seconds=bucket)
    >>>
    >>> flattener = MultiIndexTimeFlattener(
    ...     level_names=['date_id', 'bucket_in_seconds'],
    ...     custom_converter=custom_converter
    ... )
    """

    def __init__(
        self,
        level_names: List[str],
        base_date: str | datetime = '2020-01-01',
        time_unit: str = 'seconds',
        config: Optional[MultiIndexFlattenConfig] = None,
        custom_converter: Optional[Callable] = None,
    ):
        """
        初始化 MultiIndex 时间展平器

        Parameters
        ----------
        level_names : list of str
            MultiIndex 的层级名称列表，按从左到右的顺序
            示例: ['date_id', 'bucket_in_seconds']

        base_date : str or datetime
            基准日期（仅在使用默认转换器时需要）

        time_unit : str
            时间单位（仅在使用默认转换器时需要）
            支持: 'days', 'hours', 'minutes', 'seconds'

        config : MultiIndexFlattenConfig, optional
            详细配置对象

        custom_converter : callable, optional
            自定义转换函数
            签名: func(level_values: Tuple) -> pd.Timestamp or datetime
            输入: 各层级的值组成的元组
            输出: 时间戳
        """
        self.level_names = level_names
        self.custom_converter = custom_converter

        # 配置
        if config is None:
            config = MultiIndexFlattenConfig(
                base_date=base_date,
                time_unit=time_unit
            )
        self.config = config

        # 验证
        if len(level_names) < 2:
            raise ValueError(
                f"MultiIndex 至少需要2层，当前只有 {len(level_names)} 层: {level_names}"
            )

        # 状态变量
        self.detected_levels_: Optional[List[str]] = None
        self.original_index_: Optional[pd.MultiIndex] = None

    def _log(self, message: str):
        """输出日志"""
        if self.config.verbose:
            print(f"[MultiIndexTimeFlattener] {message}")

    def _validate_multiindex(self, df: pd.DataFrame):
        """验证 DataFrame 是否有合适的 MultiIndex"""
        if not isinstance(df.index, pd.MultiIndex):
            raise TypeError(
                f"DataFrame 的索引必须是 MultiIndex，当前是 {type(df.index).__name__}\n"
                f"提示: 使用 df.set_index([col1, col2]) 创建 MultiIndex"
            )

        # 检查层级数量
        if df.index.nlevels < 2:
            raise ValueError(
                f"MultiIndex 至少需要2层，当前只有 {df.index.nlevels} 层"
            )

        # 检查层级名称
        actual_names = list(df.index.names)
        for expected_name in self.level_names:
            if expected_name not in actual_names:
                raise ValueError(
                    f"MultiIndex 中找不到层级 '{expected_name}'\n"
                    f"期望的层级: {self.level_names}\n"
                    f"实际的层级: {actual_names}\n"
                    f"提示: 检查 level_names 参数是否正确"
                )

        self._log(f"✅ MultiIndex 验证通过: {actual_names}")

    def _default_converter_2levels(self, date_id: int, time_offset: int) -> pd.Timestamp:
        """
        默认的两层转换器（最常见的情况）

        Parameters
        ----------
        date_id : int
            日期ID（从0开始的相对天数）
        time_offset : int
            时间偏移量（单位由 config.time_unit 决定）

        Returns
        -------
        pd.Timestamp
            转换后的时间戳
        """
        # 解析基准日期
        if isinstance(self.config.base_date, str):
            base = pd.Timestamp(self.config.base_date)
        else:
            base = pd.Timestamp(self.config.base_date)

        # 计算日期偏移
        date_offset = pd.Timedelta(days=date_id)

        # 计算时间偏移
        time_offset_td = pd.Timedelta(**{self.config.time_unit: time_offset})

        # 合并
        timestamp = base + date_offset + time_offset_td

        return timestamp

    def _default_converter_multilevel(self, level_values: Tuple) -> pd.Timestamp:
        """
        默认的多层转换器（>2层）

        假设：
        - 第一层是日期相关（days）
        - 后续层级按 time_unit 累加

        Parameters
        ----------
        level_values : tuple
            各层级的值

        Returns
        -------
        pd.Timestamp
        """
        # 解析基准日期
        if isinstance(self.config.base_date, str):
            base = pd.Timestamp(self.config.base_date)
        else:
            base = pd.Timestamp(self.config.base_date)

        # 第一层作为日期
        date_id = level_values[0]
        timestamp = base + pd.Timedelta(days=date_id)

        # 后续层级累加
        for i, value in enumerate(level_values[1:], start=1):
            time_offset = pd.Timedelta(**{self.config.time_unit: value})
            timestamp += time_offset

        return timestamp

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        执行 MultiIndex 展平

        Parameters
        ----------
        df : pd.DataFrame
            输入 DataFrame，必须有 MultiIndex

        Returns
        -------
        pd.DataFrame
            展平后的 DataFrame，索引为单层 DatetimeIndex

        Examples
        --------
        >>> df.index.names
        ['date_id', 'bucket_in_seconds']
        >>> df_flat = flattener.transform(df)
        >>> df_flat.index.name
        'timestamp'
        >>> type(df_flat.index)
        pandas.core.indexes.datetimes.DatetimeIndex
        """
        self._log(f"开始展平 MultiIndex: {self.level_names}")

        # 验证
        self._validate_multiindex(df)

        # 保存原始索引
        self.original_index_ = df.index.copy()
        self.detected_levels_ = list(df.index.names)

        # 复制 DataFrame
        _df = df.copy()

        # 提取需要的层级值
        level_values = []
        for level_name in self.level_names:
            level_values.append(_df.index.get_level_values(level_name))

        self._log(f"提取层级数据: {self.level_names}")

        # 转换为时间戳
        timestamps = self._convert_to_timestamps(level_values)

        self._log(f"生成 {len(timestamps)} 个时间戳")

        # 时区处理
        if self.config.tz:
            try:
                timestamps = timestamps.tz_localize(self.config.tz)
                self._log(f"设置时区: {self.config.tz}")
            except TypeError:
                timestamps = timestamps.tz_convert(self.config.tz)
                self._log(f"转换时区: {self.config.tz}")

        # 重置索引（如果需要保留原始层级作为列）
        if self.config.drop_multiindex:
            # 只保留新的时间索引
            _df.index = timestamps
            _df.index.name = self.config.index_name
        else:
            # 将 MultiIndex 重置为列，然后设置新索引
            _df = _df.reset_index()
            _df.index = timestamps
            _df.index.name = self.config.index_name

        # 排序
        if self.config.sort_index:
            _df = _df.sort_index()
            self._log("索引已排序")

        self._log(f"✅ 展平完成")

        return _df

    def _convert_to_timestamps(self, level_values: List[pd.Index]) -> pd.DatetimeIndex:
        """
        将多个层级的值转换为时间戳

        Parameters
        ----------
        level_values : list of pd.Index
            各层级的值列表

        Returns
        -------
        pd.DatetimeIndex
            时间戳索引
        """
        n_rows = len(level_values[0])
        timestamps = []

        # 选择转换器
        if self.custom_converter is not None:
            # 使用自定义转换器
            converter = self.custom_converter
            self._log("使用自定义转换函数")
        elif len(level_values) == 2:
            # 使用默认两层转换器（最优化）
            converter = self._default_converter_2levels
            self._log(f"使用默认两层转换器 (base_date={self.config.base_date}, unit={self.config.time_unit})")
        else:
            # 使用默认多层转换器
            converter = self._default_converter_multilevel
            self._log(f"使用默认多层转换器 ({len(level_values)}层)")

        # 转换
        if len(level_values) == 2 and self.custom_converter is None:
            # 优化的两层转换（向量化）
            timestamps = self._vectorized_convert_2levels(level_values[0], level_values[1])
        else:
            # 通用转换（逐行）
            for i in range(n_rows):
                values_tuple = tuple(level_values[j][i] for j in range(len(level_values)))
                ts = converter(*values_tuple) if self.custom_converter is None else converter(values_tuple)
                timestamps.append(ts)

            timestamps = pd.DatetimeIndex(timestamps)

        return timestamps

    def _vectorized_convert_2levels(self, level0: pd.Index, level1: pd.Index) -> pd.DatetimeIndex:
        """
        向量化的两层转换（性能优化）

        Parameters
        ----------
        level0 : pd.Index
            第一层级（date_id）
        level1 : pd.Index
            第二层级（time_offset）

        Returns
        -------
        pd.DatetimeIndex
        """
        # 解析基准日期
        if isinstance(self.config.base_date, str):
            base = pd.Timestamp(self.config.base_date)
        else:
            base = pd.Timestamp(self.config.base_date)

        # 向量化计算
        date_offsets = pd.to_timedelta(level0, unit='D')
        time_offsets = pd.to_timedelta(level1, unit=self.config.time_unit)

        timestamps = base + date_offsets + time_offsets

        return pd.DatetimeIndex(timestamps)

    def get_flatten_report(self) -> str:
        """
        获取展平报告

        Returns
        -------
        str
            格式化的报告
        """
        if self.detected_levels_ is None:
            return "尚未执行展平操作"

        lines = [
            "=" * 60,
            "MultiIndex 展平报告",
            "=" * 60,
            f"原始层级: {self.detected_levels_}",
            f"展平层级: {self.level_names}",
            f"基准日期: {self.config.base_date}",
            f"时间单位: {self.config.time_unit}",
            f"新索引名: {self.config.index_name}",
            f"时区设置: {self.config.tz or 'None'}",
            f"是否排序: {self.config.sort_index}",
            "=" * 60
        ]

        return "\n".join(lines)


# ============================================================================
# 便捷函数
# ============================================================================

def flatten_date_bucket_index(
    df: pd.DataFrame,
    date_level: str = 'date_id',
    bucket_level: str = 'bucket_in_seconds',
    base_date: str = '2020-01-01',
    time_unit: str = 'seconds',
    **kwargs
) -> pd.DataFrame:
    """
    便捷函数：展平 date_id + bucket 类型的 MultiIndex

    Parameters
    ----------
    df : pd.DataFrame
        输入 DataFrame
    date_level : str
        日期层级名称
    bucket_level : str
        时间桶层级名称
    base_date : str
        基准日期
    time_unit : str
        时间单位
    **kwargs
        其他传给 MultiIndexTimeFlattener 的参数

    Returns
    -------
    pd.DataFrame
        展平后的 DataFrame

    Examples
    --------
    >>> df_flat = flatten_date_bucket_index(
    ...     df,
    ...     date_level='date_id',
    ...     bucket_level='bucket_in_seconds',
    ...     base_date='2023-01-01'
    ... )
    """
    flattener = MultiIndexTimeFlattener(
        level_names=[date_level, bucket_level],
        base_date=base_date,
        time_unit=time_unit,
        **kwargs
    )
    return flattener.transform(df)


def flatten_custom_multiindex(
    df: pd.DataFrame,
    level_names: List[str],
    converter: Callable,
    **kwargs
) -> pd.DataFrame:
    """
    便捷函数：使用自定义转换器展平 MultiIndex

    Parameters
    ----------
    df : pd.DataFrame
        输入 DataFrame
    level_names : list of str
        层级名称列表
    converter : callable
        自定义转换函数
    **kwargs
        其他参数

    Returns
    -------
    pd.DataFrame
        展平后的 DataFrame

    Examples
    --------
    >>> def my_converter(levels):
    ...     date_id, bucket = levels
    ...     base = pd.Timestamp('2023-01-01')
    ...     return base + pd.Timedelta(days=date_id, seconds=bucket)
    >>>
    >>> df_flat = flatten_custom_multiindex(
    ...     df,
    ...     level_names=['date_id', 'bucket_in_seconds'],
    ...     converter=my_converter
    ... )
    """
    flattener = MultiIndexTimeFlattener(
        level_names=level_names,
        custom_converter=converter,
        **kwargs
    )
    return flattener.transform(df)
