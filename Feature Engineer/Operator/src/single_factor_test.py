"""
文件名：signal_factor_backtest.py (Polars 加速版 v4)
功能：提供因子回测功能，使用 Polars 极大加速 IC/RankIC 的计算。
"""

import pandas as pd
import numpy as np
import polars as pl
from tqdm.auto import tqdm
from typing import List, Tuple, Optional

GROUP_KEYS = ['date_id', 'seconds_in_bucket']
DEFAULT_EXCLUDE_COLS = [
    'stock_id',
    'date_id',
    'seconds_in_bucket',
    'target',
    'time_id',
    'row_id',
]
POLARS_NUM_DTYPES = {
    pl.Float32, pl.Float64,
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
}

# -------------------------------------------
#                   Tools                   |
# -------------------------------------------

def _melt_metric(df: pl.DataFrame, value_cols: List[str], suffix: str, value_name: str) -> pl.DataFrame:
    """将宽表的相关系数列展开为“因子-桶”长表。"""
    if not value_cols:
        return pl.DataFrame({
            "factor": pl.Series([], dtype=pl.Utf8),
            value_name: pl.Series([], dtype=pl.Float64),
        })

    return (
        df.select(value_cols)
        .melt(variable_name="factor", value_name=value_name)
        .with_columns(
            pl.col("factor").str.replace(f"{suffix}$", "", literal=False)
        )
        .drop_nulls(value_name)
        .filter(pl.col(value_name).is_finite())
    )


def _aggregate_metric(long_df: pl.DataFrame, value_name: str, mean_name: str) -> pl.DataFrame:
    """按因子聚合均值和标准差，匹配 pandas 的 dropna 逻辑。"""
    if long_df.height == 0:
        return pl.DataFrame({
            "factor": pl.Series([], dtype=pl.Utf8),
            mean_name: pl.Series([], dtype=pl.Float64),
            f"{mean_name}_std": pl.Series([], dtype=pl.Float64),
        })

    return long_df.group_by("factor").agg([
        pl.col(value_name).mean().alias(mean_name),
        pl.col(value_name).std().alias(f"{mean_name}_std"),
    ])


def _safe_ratio(num: pd.Series, denom: pd.Series) -> pd.Series:
    """计算安全的均值/标准差比值，避免 0 或 NaN 带来的异常。"""
    denom_array = pd.to_numeric(denom, errors='coerce').to_numpy(dtype=float)
    num_array = pd.to_numeric(num, errors='coerce').to_numpy(dtype=float)
    valid = (~np.isnan(denom_array)) & (denom_array != 0)
    result = np.full_like(num_array, np.nan, dtype=float)
    result[valid] = num_array[valid] / denom_array[valid]
    return pd.Series(result, index=num.index)


def _clip_factor_inf_values(pl_df: pl.DataFrame, factor_cols: List[str]) -> pl.DataFrame:
    """
    检查每个因子是否包含 ±inf，若有则打印提示，
    并在每个 [date_id, seconds_in_bucket] 组内将其裁剪到组内有限值的最大/最小。
    """
    inf_factors: List[str] = []
    for col in factor_cols:
        has_inf = pl_df.select(pl.col(col).is_infinite().any()).item()
        if has_inf:
            print(f"[提示] 因子 '{col}' 存在无穷值，正在按 {GROUP_KEYS} 进行裁剪...")
            inf_factors.append(col)

    if not inf_factors:
        return pl_df

    agg_exprs = []
    for col in inf_factors:
        agg_exprs.extend([
            pl.col(col)
            .filter(pl.col(col).is_finite())
            .max()
            .alias(f"__{col}_finite_max"),
            pl.col(col)
            .filter(pl.col(col).is_finite())
            .min()
            .alias(f"__{col}_finite_min"),
        ])

    finite_stats = pl_df.group_by(GROUP_KEYS, maintain_order=True).agg(agg_exprs)
    pl_df = pl_df.join(finite_stats, on=GROUP_KEYS, how="left")

    replace_exprs = []
    for col in inf_factors:
        max_col = f"__{col}_finite_max"
        min_col = f"__{col}_finite_min"
        replace_exprs.append(
            pl.when(pl.col(col) == float('inf'))
            .then(pl.col(max_col).fill_null(pl.col(col)))
            .when(pl.col(col) == float('-inf'))
            .then(pl.col(min_col).fill_null(pl.col(col)))
            .otherwise(pl.col(col))
            .alias(col)
        )

    pl_df = pl_df.with_columns(replace_exprs)

    drop_cols = []
    for col in inf_factors:
        drop_cols.extend([f"__{col}_finite_max", f"__{col}_finite_min"])
    pl_df = pl_df.drop(drop_cols)

    return pl_df


def _prepare_inputs_pd(
    df: pd.DataFrame,
    factor_cols: List[str],
    start_date: int = 0,
    end_date: int = 400,
) -> Tuple[List[str], pd.DataFrame]:
    """Pandas 分支：完成类型检查、日期过滤与列裁剪。"""
    factor_cols = list(factor_cols)
    if len(factor_cols) == 0:
        raise ValueError("`factor_cols` 列表不能为空。")

    if 'target' not in df.columns:
        raise ValueError("DataFrame中未找到 'target' 列。")

    missing_cols = [col for col in factor_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame中缺少以下列: {missing_cols}")

    if start_date > end_date:
        raise ValueError("`start_date` 不能大于 `end_date`。")

    exclude_cols = set(DEFAULT_EXCLUDE_COLS)
    filtered_factor_cols = [
        col for col in factor_cols
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
    ]
    if len(filtered_factor_cols) == 0:
        raise ValueError("过滤后没有可用的数值因子列。")

    missing_group_cols = [col for col in GROUP_KEYS if col not in df.columns]
    if missing_group_cols:
        raise ValueError(f"DataFrame中缺少分组列: {missing_group_cols}")

    required_cols = filtered_factor_cols + ['target'] + GROUP_KEYS

    date_series = df['date_id']
    mask = (date_series >= start_date) & (date_series <= end_date)
    filtered_df = df.loc[mask]
    if filtered_df.empty:
        raise ValueError(f"在 [{start_date}, {end_date}] 范围内没有可用数据。")

    return filtered_factor_cols, filtered_df[required_cols].copy()


def _prepare_inputs_pl(
    df: pl.DataFrame,
    factor_cols: List[str],
    start_date: int = 0,
    end_date: int = 400,
) -> Tuple[List[str], pl.DataFrame]:
    """Polars 分支：完成类型检查、日期过滤与列裁剪。"""
    factor_cols = list(factor_cols)
    if len(factor_cols) == 0:
        raise ValueError("`factor_cols` 列表不能为空。")

    cols_set = set(df.columns)
    if 'target' not in cols_set:
        raise ValueError("DataFrame中未找到 'target' 列。")

    missing_cols = [col for col in factor_cols if col not in cols_set]
    if missing_cols:
        raise ValueError(f"DataFrame中缺少以下列: {missing_cols}")

    if start_date > end_date:
        raise ValueError("`start_date` 不能大于 `end_date`。")

    exclude_cols = set(DEFAULT_EXCLUDE_COLS)
    schema = df.schema
    filtered_factor_cols = [
        col for col in factor_cols
        if (col in schema) and (schema[col] in POLARS_NUM_DTYPES) and (col not in exclude_cols)
    ]
    if len(filtered_factor_cols) == 0:
        raise ValueError("过滤后没有可用的数值因子列。")

    missing_group_cols = [col for col in GROUP_KEYS if col not in cols_set]
    if missing_group_cols:
        raise ValueError(f"DataFrame中缺少分组列: {missing_group_cols}")

    required_cols = filtered_factor_cols + ['target'] + GROUP_KEYS

    filtered_df = df.filter(
        (pl.col('date_id') >= start_date) & (pl.col('date_id') <= end_date)
    )
    if filtered_df.height == 0:
        raise ValueError(f"在 [{start_date}, {end_date}] 范围内没有可用数据。")

    # 仅保留所需列，保持列顺序
    filtered_df = filtered_df.select(required_cols)
    return filtered_factor_cols, filtered_df


def _prepare_inputs(
    df,  # 可为 pandas.DataFrame 或 polars.DataFrame
    factor_cols: List[str],
    start_date: int = 0,
    end_date: int = 400,
):
    """根据 df 类型分派到 Pandas/Polars 分支。"""
    if isinstance(df, pd.DataFrame):
        return _prepare_inputs_pd(df, factor_cols, start_date=start_date, end_date=end_date)
    if isinstance(df, pl.DataFrame):
        return _prepare_inputs_pl(df, factor_cols, start_date=start_date, end_date=end_date)
    raise TypeError("`df` 必须是 pandas.DataFrame 或 polars.DataFrame")

# -------------------------------------------
#               Backtest Methods            |
# -------------------------------------------

def basic_backtest(
    pl_df: pl.DataFrame,
    factor_cols: List[str],
) -> pd.DataFrame:
    """
    按 [date_id, seconds_in_bucket] 分组，计算因子与 'target' 的 IC 均值、RankIC 均值、ICIR 和 RankICIR。

    参数:
    pl_df (pl.DataFrame): 
        经过预处理的 Polars DataFrame（已过滤日期、列并处理无穷值）。
    factor_cols (List[str]): 
        需要进行回测的因子列名列表。

    返回:
    pd.DataFrame: 
        包含所有因子指标的DataFrame，按 'ic' 降序排列。
        列: ['factor', 'ic', 'rankic', 'icir', 'rankicir']
    """
    ic_exprs = [
        pl.corr(col, "target", method="pearson").alias(f"{col}_ic")
        for col in factor_cols
    ]
    
    rank_ic_exprs = [
        pl.corr(col, "target", method="spearman").alias(f"{col}_rank_ic")
        for col in factor_cols
    ]

    print("\n正在计算basic统计指标...")
    
    # 捕获空组可能引发的Polars计算错误
    try:
        ic_df_pl = pl_df.group_by(GROUP_KEYS, maintain_order=False).agg(
            ic_exprs + rank_ic_exprs
        )
    except pl.ComputeError as e:
        print(f"Polars 计算错误: {e}")
        print("这可能是由于某些组中方差为零（例如 factor_flat）。正在尝试过滤掉这些组...")
        # 过滤掉方差为零的组（这会慢一些，但更稳健）
        ic_df_pl = pl_df.group_by(GROUP_KEYS, maintain_order=False).filter(
            pl.col('target').std().is_not_null() & (pl.col('target').std() > 0)
        ).agg(
            ic_exprs + rank_ic_exprs
        )

    ic_cols = [f"{col}_ic" for col in factor_cols]
    rank_ic_cols = [f"{col}_rank_ic" for col in factor_cols]

    ic_long = _melt_metric(ic_df_pl, ic_cols, "_ic", "ic")
    rank_ic_long = _melt_metric(ic_df_pl, rank_ic_cols, "_rank_ic", "rankic")

    ic_stats = _aggregate_metric(ic_long, "ic", "ic")
    rank_stats = _aggregate_metric(rank_ic_long, "rankic", "rankic")

    stats_pl = (
        pl.DataFrame({"factor": factor_cols})
        .join(ic_stats, on="factor", how="left")
        .join(rank_stats, on="factor", how="left")
    )
    stats_df = stats_pl.to_pandas()

    stats_df['icir'] = _safe_ratio(stats_df['ic'], stats_df['ic_std'])
    stats_df['rankicir'] = _safe_ratio(stats_df['rankic'], stats_df['rankic_std'])

    results_df = stats_df[['factor', 'ic', 'rankic', 'icir', 'rankicir']].copy()

    del pl_df, ic_df_pl, ic_long, rank_ic_long, stats_pl
    
    return results_df

# -------------------------------------------
#                 Main Function             |
# -------------------------------------------

def backtest(
    df: pd.DataFrame,
    factor_cols: List[str],
    threshold: float = 0,
    start_date: int = 0,
    end_date: int = 400,
    check_inf: bool = False,
) -> pd.DataFrame:

    prepared_factors, filtered_df = _prepare_inputs(
        df,
        factor_cols,
        start_date=start_date,
        end_date=end_date,
    )

    print(f"开始回测 {len(prepared_factors)} 个因子，日期范围 [{start_date}, {end_date}]...")
    # 若已是 Polars 则直接使用，否则再从 Pandas 转换
    if isinstance(filtered_df, pl.DataFrame):
        pl_df = filtered_df
    else:
        pl_df = pl.from_pandas(filtered_df)
    del filtered_df

    if check_inf: # 检查并处理Inf
        pl_df = _clip_factor_inf_values(pl_df, prepared_factors)

    raw_results = basic_backtest(
        pl_df,
        prepared_factors,
    )

    processed = raw_results.copy()
    processed['abs_ic'] = processed['ic'].abs()
    processed_sorted = processed.sort_values(by='ic', ascending=False).reset_index(drop=True)

    accepted = processed[processed['abs_ic'] >= threshold].copy() # 这里设定过检标准
    accepted = accepted.sort_values(by='abs_ic', ascending=False).reset_index(drop=True)

    print(f"\n回测完成。总共 {len(processed)} 个因子，其中 {len(accepted)} 个因子IC绝对值 >= {threshold}")
    print(f"\n--- IC 绝对值 > {threshold} 的过审因子排名 (按绝对值IC降序) ---")
    if not accepted.empty:
        print(accepted[['factor', 'ic', 'rankic', 'icir', 'rankicir']].to_string())
    else:
        print("没有因子通过阈值。")

    return processed_sorted[['factor', 'ic', 'rankic', 'icir', 'rankicir']]
