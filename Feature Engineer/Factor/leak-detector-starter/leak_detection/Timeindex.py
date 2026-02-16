import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional, Set, Union, Callable


class AutoTimeIndexer:
    """
    一个用于自动检测DataFrame中的离散时间列（年、月、日、时等），
    将它们“展平”为一个标准时间戳，并设置为排序后的索引。
    """

    # 1. 定义启发式规则（您可以按需添加更多）
    # 键是 pd.to_datetime 能识别的标准名称
    # 值是该标准名称在实际数据中可能的列名（将全部转为小写进行比较）
    HEURISTICS_MAP = {
        'year':   {'year', 'yr', 'y', '年'},
        'month':  {'month', 'mon', 'm', '月'},
        'day':    {'day', 'd', '日'},
        'hour':   {'hour', 'hr', 'h', '时', '小時'},
        'minute': {'minute', 'min', 'mi', '分', '分鐘'},
        'second': {'second', 'sec', 's', '秒'}
    }

    # 在检测 'day' (日期) 时，需要排除的明确的非 "day-of-month" (几号) 的列
    DAY_EXCLUSIONS = {'dayofweek', 'dayofyear', 'weekday', '星期'}

    def __init__(self,
                 column_mapping: Optional[Dict[str, str]] = None,
                 tz: Optional[str] = None,
                 index_name: str = 'timestamp',
                 drop_originals: bool = True,
                 use_regex: bool = False,
                 use_inference: bool = False,
                 custom_heuristics: Optional[Dict[str, Set[str]]] = None,
                 verbose: bool = False,
                 custom_time_levels: Optional[List[str]] = None,
                 custom_converter: Optional[Callable[[pd.DataFrame], pd.Series]] = None):
        """
        初始化 AutoTimeIndexer (增强版).

        参数:
        - column_mapping (dict, optional): 手动指定列名映射，格式为 {'year': 'actual_col_name', ...}。
                                          手动映射优先级最高，会覆盖自动检测。
        - tz (str, optional): 目标时区，例如 'UTC' 或 'Asia/Shanghai'。
                              如果为 None，则创建 "naive" (无时区) 的时间戳。
        - index_name (str): 设置新索引的名称。
        - drop_originals (bool): 是否在处理后删除自动检测到的原始时间列。
        - use_regex (bool): 是否使用正则表达式进行模糊匹配（如匹配 'date_year' 到 'year'）。
        - use_inference (bool): 是否根据数据值范围智能推断时间列（当列名无法识别时）。
        - custom_heuristics (dict, optional): 自定义启发式规则，会与默认规则合并。
        - verbose (bool): 是否输出详细的检测过程信息（用于调试）。
        - custom_time_levels (list, optional): 自定义时间层级列名列表，如 ['date_id', 'second_in_buckets']。
                                               使用此参数时必须同时提供 custom_converter。
        - custom_converter (callable, optional): 自定义转换函数，签名为 func(df) -> pd.Series。
                                                 输入为包含自定义时间列的 DataFrame，
                                                 输出为 pd.Series 的时间戳。

        自定义时间层级使用示例:
        ```python
        def my_converter(df):
            # df 包含 date_id 和 second_in_buckets 列
            base_date = pd.Timestamp('2020-01-01')
            timestamps = (base_date +
                         pd.to_timedelta(df['date_id'], unit='D') +
                         pd.to_timedelta(df['second_in_buckets'], unit='s'))
            return timestamps

        indexer = AutoTimeIndexer(
            custom_time_levels=['date_id', 'second_in_buckets'],
            custom_converter=my_converter
        )
        ```
        """
        self.column_mapping = column_mapping or {}
        self.tz = tz
        self.index_name = index_name
        self.drop_originals = drop_originals
        self.use_regex = use_regex
        self.use_inference = use_inference
        self.verbose = verbose

        # 自定义时间层级
        self.custom_time_levels = custom_time_levels
        self.custom_converter = custom_converter

        # 验证自定义时间层级配置
        if self.custom_time_levels is not None:
            if self.custom_converter is None:
                raise ValueError(
                    "使用 custom_time_levels 时必须提供 custom_converter 函数。\n"
                    "custom_converter 应接受 DataFrame 并返回 pd.Series 时间戳。"
                )

        # 合并自定义启发式规则
        self.heuristics_map = self._merge_heuristics(custom_heuristics)

        # 这些属性将在 transform 过程中被填充，方便您查看检测结果
        self.detected_mapping_: Dict[str, str] = {}
        self.original_columns_: List[str] = []
        self.detection_method_: Dict[str, str] = {}  # 记录每个字段的检测方法
        self.is_custom_mode_: bool = False  # 标记是否使用自定义模式

    def _merge_heuristics(self, custom: Optional[Dict[str, Set[str]]]) -> Dict[str, Set[str]]:
        """合并默认和自定义启发式规则"""
        merged = {k: v.copy() for k, v in self.HEURISTICS_MAP.items()}
        if custom:
            for standard_name, variants in custom.items():
                if standard_name in merged:
                    merged[standard_name].update(variants)
                else:
                    merged[standard_name] = set(variants)
        return merged

    def _log(self, message: str):
        """输出日志信息（当 verbose=True 时）"""
        if self.verbose:
            print(f"[AutoTimeIndexer] {message}")

    def _match_by_exact(self, df_columns: pd.Index, skip_existing: bool = False) -> Dict[str, str]:
        """精确匹配（原始方法）"""
        mapping = {}
        df_cols_lower = {col: col.lower() for col in df_columns}

        for standard_name, possible_names in self.heuristics_map.items():
            # 如果需要跳过已存在的映射
            if skip_existing and standard_name in self.detection_method_:
                continue

            for actual_col, col_lower in df_cols_lower.items():
                if col_lower in possible_names:
                    if standard_name == 'day' and col_lower in self.DAY_EXCLUSIONS:
                        continue
                    mapping[standard_name] = actual_col
                    if standard_name not in self.detection_method_:
                        self.detection_method_[standard_name] = 'exact_match'
                    self._log(f"精确匹配: {standard_name} -> {actual_col}")
                    break
        return mapping

    def _match_by_regex(self, df_columns: pd.Index, existing_mapping: Dict[str, str]) -> Dict[str, str]:
        """正则表达式模糊匹配"""
        mapping = existing_mapping.copy()

        # 构建正则模式
        regex_patterns = {
            'year':   r'.*year.*|.*yr.*|.*年.*',
            'month':  r'.*month.*|.*mon.*|.*月.*',
            'day':    r'.*day.*|.*日.*(?<!周)(?<!星期)',  # 排除周、星期
            'hour':   r'.*hour.*|.*hr.*|.*时.*',
            'minute': r'.*minute.*|.*min.*|.*分.*',
            'second': r'.*second.*|.*sec.*|.*秒.*'
        }

        for standard_name, pattern in regex_patterns.items():
            if standard_name in mapping:  # 已经找到了
                continue
            for col in df_columns:
                col_lower = col.lower()
                if re.match(pattern, col_lower):
                    # 额外检查：确保不是排除的列
                    if standard_name == 'day' and any(excl in col_lower for excl in self.DAY_EXCLUSIONS):
                        continue
                    mapping[standard_name] = col
                    self.detection_method_[standard_name] = 'regex_match'
                    self._log(f"正则匹配: {standard_name} -> {col}")
                    break

        return mapping

    def _infer_by_values(self, df: pd.DataFrame, existing_mapping: Dict[str, str]) -> Dict[str, str]:
        """根据数据值范围智能推断"""
        mapping = existing_mapping.copy()

        # 定义推断规则
        inference_rules = {
            'year': lambda s: (s.dtype in [int, np.int64, float, np.float64] and
                              1900 <= s.min() <= 2200 and s.max() <= 2200),
            'month': lambda s: (s.dtype in [int, np.int64, float, np.float64] and
                               1 <= s.min() and s.max() <= 12),
            'day': lambda s: (s.dtype in [int, np.int64, float, np.float64] and
                             1 <= s.min() and s.max() <= 31),
            'hour': lambda s: (s.dtype in [int, np.int64, float, np.float64] and
                              0 <= s.min() and s.max() <= 23),
            'minute': lambda s: (s.dtype in [int, np.int64, float, np.float64] and
                                0 <= s.min() and s.max() <= 59),
            'second': lambda s: (s.dtype in [int, np.int64, float, np.float64] and
                                0 <= s.min() and s.max() <= 59)
        }

        # 优先级顺序（避免歧义）
        priority_order = ['year', 'month', 'day', 'hour', 'minute', 'second']
        used_columns = set(mapping.values())

        for standard_name in priority_order:
            if standard_name in mapping:
                continue

            for col in df.columns:
                if col in used_columns:
                    continue

                try:
                    if inference_rules[standard_name](df[col]):
                        # 额外验证：避免将 month 误判为 day
                        if standard_name == 'day' and 'month' not in mapping:
                            # 如果还没找到 month，暂时跳过
                            continue

                        mapping[standard_name] = col
                        self.detection_method_[standard_name] = 'value_inference'
                        used_columns.add(col)
                        self._log(f"值推断: {standard_name} -> {col} (范围: {df[col].min()}-{df[col].max()})")
                        break
                except Exception:
                    continue

        return mapping

    def _detect_columns(self, df_columns: pd.Index, df: Optional[pd.DataFrame] = None) -> Dict[str, str]:
        """
        (内部方法) 根据启发式规则，从DataFrame的列中检测时间组件。
        支持多种检测策略：手动映射 > 精确匹配 > 正则匹配 > 值推断
        """
        self._log(f"开始检测时间列，共 {len(df_columns)} 列")
        self._log(f"列名: {list(df_columns)}")

        # 1. 优先使用手动映射
        mapping = {}
        for standard_name, actual_col in self.column_mapping.items():
            if actual_col in df_columns:
                mapping[standard_name] = actual_col
                self.detection_method_[standard_name] = 'manual_mapping'
                self._log(f"手动映射: {standard_name} -> {actual_col}")
            else:
                raise ValueError(
                    f"手动映射错误：指定的列 '{actual_col}' (用于 '{standard_name}') 不存在于 DataFrame 中"
                )

        # 2. 精确匹配（跳过已手动映射的）
        exact_mapping = self._match_by_exact(df_columns, skip_existing=True)
        for standard_name, actual_col in exact_mapping.items():
            if standard_name not in mapping:
                mapping[standard_name] = actual_col

        # 3. 如果启用了正则匹配
        if self.use_regex:
            mapping = self._match_by_regex(df_columns, mapping)

        # 4. 如果启用了值推断
        if self.use_inference and df is not None:
            mapping = self._infer_by_values(df, mapping)

        # 5. 检查：基础的时间组件（年、月、日）是必需的
        missing_components = {'year', 'month', 'day'} - set(mapping.keys())
        if missing_components:
            error_msg = [
                f"自动检测失败：缺少必需的时间组件: {missing_components}",
                f"已找到的映射: {mapping}",
                f"DataFrame 列名: {list(df_columns)}",
                "\n建议解决方案:",
                "1. 使用 column_mapping 参数手动指定列映射",
                "2. 启用 use_regex=True 进行模糊匹配",
                "3. 启用 use_inference=True 根据数据值推断"
            ]
            if self.verbose:
                error_msg.append(f"4. 使用 verbose=True 查看详细检测过程")
            raise ValueError("\n".join(error_msg))

        self._log(f"检测成功: {mapping}")
        return mapping

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        执行"检查、展平、设置索引和排序"的完整流程。

        参数:
        - df (pd.DataFrame): 包含离散时间列的输入 DataFrame。

        返回:
        - pd.DataFrame: 一个以统一时间戳为索引的新 DataFrame。
        """

        # 复制以避免修改原始 DataFrame
        _df = df.copy()

        # 判断是否使用自定义模式
        if self.custom_time_levels is not None:
            self.is_custom_mode_ = True
            ts_series = self._transform_custom(_df)
        else:
            self.is_custom_mode_ = False
            ts_series = self._transform_standard(_df)

        # 4. 处理时区
        if self.tz:
            try:
                # 本地化（如果时间是 "naive"）
                ts_series = ts_series.dt.tz_localize(self.tz)
                self._log(f"时区本地化: {self.tz}")
            except TypeError:
                # 转换（如果时间已经是 "aware"）
                ts_series = ts_series.dt.tz_convert(self.tz)
                self._log(f"时区转换: {self.tz}")

        # 5. 统一一个 Index (设置为索引)
        _df.set_index(ts_series, inplace=True)
        _df.index.name = self.index_name

        # 6. （可选）删除原始列
        if self.drop_originals:
            _df = _df.drop(columns=self.original_columns_)
            self._log(f"已删除原始时间列: {self.original_columns_}")

        # 7. 统一一个有时间顺序的index (排序)
        _df = _df.sort_index()
        self._log("索引排序完成")

        return _df

    def _transform_custom(self, df: pd.DataFrame) -> pd.Series:
        """
        使用自定义时间层级进行转换。

        参数:
        - df (pd.DataFrame): 输入 DataFrame。

        返回:
        - pd.Series: 时间戳序列。
        """
        self._log(f"使用自定义时间层级模式: {self.custom_time_levels}")

        # 检查自定义时间列是否存在
        missing_cols = [col for col in self.custom_time_levels if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"自定义时间列不存在于 DataFrame 中: {missing_cols}\n"
                f"DataFrame 列名: {list(df.columns)}"
            )

        # 记录检测结果
        for col in self.custom_time_levels:
            self.detected_mapping_[col] = col
            self.detection_method_[col] = 'custom_level'

        self.original_columns_ = list(self.custom_time_levels)

        # 提取自定义时间列
        time_df = df[self.custom_time_levels].copy()

        # 使用用户提供的转换函数
        try:
            self._log("正在使用自定义转换函数...")
            ts_series = self.custom_converter(time_df)

            # 验证返回类型
            if not isinstance(ts_series, pd.Series):
                raise TypeError(
                    f"custom_converter 必须返回 pd.Series，但返回了 {type(ts_series)}"
                )

            # 尝试转换为时间戳（如果还不是）
            if not pd.api.types.is_datetime64_any_dtype(ts_series):
                ts_series = pd.to_datetime(ts_series)

            self._log(f"成功创建 {len(ts_series)} 个时间戳")

        except Exception as e:
            print(f"自定义转换函数执行出错: {e}")
            print(f"自定义时间列数据:\n{time_df.head()}")
            raise e

        return ts_series

    def _transform_standard(self, df: pd.DataFrame) -> pd.Series:
        """
        使用标准时间层级（year/month/day 等）进行转换。

        参数:
        - df (pd.DataFrame): 输入 DataFrame。

        返回:
        - pd.Series: 时间戳序列。
        """
        # 1. 检查其时间结构 (自动检测列)
        self.detected_mapping_ = self._detect_columns(df.columns, df)
        self.original_columns_ = list(self.detected_mapping_.values())

        self._log(f"原始列: {self.original_columns_}")
        self._log(f"检测方法: {self.detection_method_}")

        # 2. 将其展平 (创建标准时间 DataFrame)
        time_df = pd.DataFrame()
        for standard_name, actual_name in self.detected_mapping_.items():
            time_df[standard_name] = df[actual_name]

        # 3. 将其展平 (转换为时间戳序列)
        try:
            # 填充任何缺失的可选组件（时、分、秒）为0，否则 pd.to_datetime 会报错
            for component in ['hour', 'minute', 'second']:
                if component not in time_df.columns:
                    time_df[component] = 0

            self._log("正在创建时间戳...")
            ts_series = pd.to_datetime(time_df)
            self._log(f"成功创建 {len(ts_series)} 个时间戳")
        except Exception as e:
            print(f"创建时间戳时出错。请检查数据中是否有无效日期（如2月30日）。")
            print(f"时间组件数据:\n{time_df.head()}")
            raise e

        return ts_series

    def get_detection_report(self) -> str:
        """
        获取检测报告，显示每个时间组件的检测结果和方法。

        返回:
        - str: 格式化的检测报告
        """
        if not self.detected_mapping_:
            return "尚未执行检测。请先调用 transform() 方法。"

        report_lines = [
            "=" * 60,
            "AutoTimeIndexer 检测报告",
            "=" * 60,
            f"索引名称: {self.index_name}",
            f"时区设置: {self.tz or 'None (naive)'}",
            f"删除原始列: {self.drop_originals}",
            f"模式: {'自定义时间层级' if self.is_custom_mode_ else '标准时间组件'}",
            "",
            "检测到的时间组件:",
            "-" * 60
        ]

        if self.is_custom_mode_:
            # 自定义模式：显示自定义时间层级
            for col in self.custom_time_levels:
                if col in self.detected_mapping_:
                    method = self.detection_method_.get(col, 'unknown')
                    report_lines.append(
                        f"  {col:20s} [{method}]"
                    )
        else:
            # 标准模式：显示 year/month/day 等
            for standard_name in ['year', 'month', 'day', 'hour', 'minute', 'second']:
                if standard_name in self.detected_mapping_:
                    actual_col = self.detected_mapping_[standard_name]
                    method = self.detection_method_.get(standard_name, 'unknown')
                    report_lines.append(
                        f"  {standard_name:8s} -> {actual_col:20s} [{method}]"
                    )

        report_lines.append("=" * 60)
        return "\n".join(report_lines)