"""
Shared utility helpers for factor preparation workflows.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns

DEFAULT_GROUP_KEYS: Sequence[str] = ("date_id", "seconds_in_bucket")
DEFAULT_EXCLUDE_COLS: Sequence[str] = ("stock_id", "date_id", "seconds_in_bucket", "target", "time_id", "row_id")
DEFAULT_COLS: Sequence[str] = (
    'stock_id', 'date_id', 'seconds_in_bucket', 'imbalance_size',
    'imbalance_buy_sell_flag', 'reference_price', 'matched_size',
    'far_price', 'near_price', 'bid_price', 'bid_size', 'ask_price',
    'ask_size', 'wap', 'target', 'time_id', 'row_id'
)

# -------------------------------------------
#                   Tools                   |
# -------------------------------------------

def _infer_factor_cols_pd(df: pd.DataFrame, group_keys: Sequence[str]) -> List[str]:
    return [
        col
        for col in df.columns
        if col not in group_keys and pd.api.types.is_numeric_dtype(df[col])
    ]


def _infer_factor_cols_pl(df: pl.DataFrame, group_keys: Sequence[str]) -> List[str]:
    numeric_dtypes = {
        pl.Float32,
        pl.Float64,
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
    }
    return [
        col
        for col, dtype in zip(df.columns, df.dtypes)
        if col not in group_keys and dtype in numeric_dtypes
    ]


def _clip_inf_polars(
    df: pl.DataFrame,
    factor_cols: Iterable[str],
    group_keys: Sequence[str],
) -> pl.DataFrame:
    inf_factors: List[str] = []
    for col in factor_cols:
        has_inf = df.select(pl.col(col).is_infinite().any()).item()
        if has_inf:
            inf_factors.append(col)

    if not inf_factors:
        return df

    print(f"[clip_extreme_values] Polars columns adjusted: {', '.join(inf_factors)}")

    agg_exprs = []
    for col in inf_factors:
        agg_exprs.extend(
            [
                pl.col(col)
                .filter(pl.col(col).is_finite())
                .max()
                .alias(f"__{col}_finite_max"),
                pl.col(col)
                .filter(pl.col(col).is_finite())
                .min()
                .alias(f"__{col}_finite_min"),
            ]
        )

    finite_stats = df.group_by(group_keys, maintain_order=True).agg(agg_exprs)
    df = df.join(finite_stats, on=group_keys, how="left")

    replace_exprs = []
    for col in inf_factors:
        max_col = f"__{col}_finite_max"
        min_col = f"__{col}_finite_min"
        replace_exprs.append(
            pl.when(pl.col(col) == float("inf"))
            .then(pl.col(max_col).fill_null(pl.col(col)))
            .when(pl.col(col) == float("-inf"))
            .then(pl.col(min_col).fill_null(pl.col(col)))
            .otherwise(pl.col(col))
            .alias(col)
        )

    df = df.with_columns(replace_exprs)
    drop_cols = []
    for col in inf_factors:
        drop_cols.extend([f"__{col}_finite_max", f"__{col}_finite_min"])
    return df.drop(drop_cols)


def _clip_inf_pandas(
    df: pd.DataFrame,
    factor_cols: Iterable[str],
    group_keys: Sequence[str],
) -> pd.DataFrame:
    result = df.copy(deep=True)
    groupers = [result[key] for key in group_keys]

    adjusted_cols: List[str] = []
    for col in factor_cols:
        series = result[col]
        has_inf = np.isinf(series.to_numpy()).any()
        if not has_inf:
            continue

        finite_series = series.where(np.isfinite(series))
        group_max = finite_series.groupby(groupers).transform("max")
        group_min = finite_series.groupby(groupers).transform("min")

        result.loc[series == np.inf, col] = group_max[series == np.inf]
        result.loc[series == -np.inf, col] = group_min[series == -np.inf]
        adjusted_cols.append(col)

    if adjusted_cols:
        print(f"[clip_extreme_values] Pandas columns adjusted: {', '.join(adjusted_cols)}")

    return result


def _to_pandas_subset(
    df: pd.DataFrame | pl.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    """
    提取指定列并转换为 pandas.DataFrame，便于统一统计。
    """
    columns = list(dict.fromkeys(columns))
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"DataFrame 缺少所需列: {missing}")

    if isinstance(df, pd.DataFrame):
        return df.loc[:, columns].copy()
    if isinstance(df, pl.DataFrame):
        return df.select(columns).to_pandas()
    raise TypeError("df 必须是 pandas 或 polars DataFrame。")


def _groupwise_corr_mean(
    data: pd.DataFrame,
    factor_cols: Sequence[str],
    target_col: str,
    group_cols: Sequence[str],
    corr_method: str,
) -> dict[str, float]:
    grouped = data.groupby(list(group_cols), dropna=False)
    collected: dict[str, list[float]] = {col: [] for col in factor_cols}
    for _, group in grouped:
        tgt = group[target_col]
        if tgt.nunique(dropna=False) < 2:
            continue
        for factor in factor_cols:
            series = group[factor]
            if series.nunique(dropna=False) < 2:
                continue
            value = series.corr(tgt, method=corr_method)
            if pd.notna(value):
                collected[factor].append(float(value))

    result: dict[str, float] = {}
    for factor, values in collected.items():
        result[factor] = float(np.nanmean(values)) if values else np.nan
    return result


def _aggregate_corr_matrices(
    data: pd.DataFrame,
    cols: Sequence[str],
    group_cols: Sequence[str],
    corr_method: str,
) -> pd.DataFrame:
    size = len(cols)
    if size == 0:
        return pd.DataFrame()

    # 用 float32 累积，减少中间矩阵内存占用
    sum_matrix = np.zeros((size, size), dtype=np.float32)
    count_matrix = np.zeros((size, size), dtype=np.float32)

    grouped = data.groupby(list(group_cols), dropna=False)
    for _, group in grouped:
        if len(group) < 2:
            continue
        corr = group[list(cols)].corr(method=corr_method)
        if corr.empty:
            continue
        arr = corr.to_numpy(dtype=np.float32)
        mask = ~np.isnan(arr)
        sum_matrix[mask] += arr[mask]
        count_matrix[mask] += np.float32(1.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        averaged = np.divide(
            sum_matrix,
            count_matrix,
            where=count_matrix > 0,
            out=np.zeros_like(sum_matrix),
        )

    result = pd.DataFrame(averaged, index=cols, columns=cols)
    diag_mask = np.eye(size, dtype=bool)
    diag_counts = count_matrix[diag_mask]
    result.values[diag_mask & (diag_counts > 0)] = 1.0
    return result


def _compute_factor_corr_matrix(
    df: pd.DataFrame | pl.DataFrame,
    cols: Sequence[str],
    method: str,
    corr_method: str = "pearson",
    date_col: str = "date_id",
    bucket_col: str = "seconds_in_bucket",
    stock_col: str = "stock_id",
) -> pd.DataFrame:
    method_key = method.lower()
    valid_methods = {"corr_csall", "corr_cs1", "corr_cs2", "corr_ts"}
    if method_key not in valid_methods:
        raise ValueError(f"method 必须属于 {valid_methods}")

    required_cols = list(cols)
    if method_key in {"corr_cs1", "corr_cs2"}:
        required_cols.append(date_col)
    if method_key == "corr_cs2":
        required_cols.append(bucket_col)
    if method_key == "corr_ts":
        required_cols.append(stock_col)

    data = _to_pandas_subset(df, required_cols)
    if data.empty:
        return pd.DataFrame(columns=cols, index=cols, dtype=np.float32)

    # Use a random 25% subsample to approximate correlations instead of full data.
    sample_frac = 0.25
    sample_size = max(1, int(len(data) * sample_frac))
    if sample_size < len(data):
        data = data.sample(n=sample_size, random_state=42)

    # 将参与相关性计算的列统一转换为 float32，避免 float64 占用
    factor_cols = list(cols)
    for col in factor_cols:
        if col in data.columns:
            data[col] = data[col].astype(np.float32, copy=False)
    if method_key == "corr_csall":
        corr = data[factor_cols].corr(method=corr_method)
        return corr.astype(np.float32, copy=False)

    group_map = {
        "corr_cs1": [date_col],
        "corr_cs2": [date_col, bucket_col],
        "corr_ts": [stock_col],
    }
    corr_matrix = _aggregate_corr_matrices(data, factor_cols, group_map[method_key], corr_method)
    return corr_matrix.astype(np.float32, copy=False)


def _render_corr_heatmap(corr_matrix: pd.DataFrame, cols: Sequence[str], title: str) -> None:
    if corr_matrix.empty:
        return
    corr_subset = corr_matrix.loc[cols, cols]
    mask = corr_subset.abs() < 0.7
    plt.figure(figsize=(max(6, len(cols) * 0.5), max(6, len(cols) * 0.5)))
    sns.heatmap(
        corr_subset,
        mask=mask,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        cbar_kws={"shrink": 0.75},
    )
    plt.title(title)
    plt.tight_layout()
    plt.show()


def _ic_value(col_name: str, ic_map: dict[str, float]) -> float:
    value = ic_map.get(col_name, np.nan)
    if pd.notna(value):
        return abs(float(value))
    return 0


def _count_connected_components(matrix: np.ndarray) -> int:
    if matrix.size == 0:
        return 0
    n = matrix.shape[0]
    visited = np.zeros(n, dtype=bool)
    count = 0
    for start in range(n):
        if visited[start]:
            continue
        count += 1
        stack = [start]
        visited[start] = True
        while stack:
            node = stack.pop()
            neighbors = np.where(matrix[node] > 0)[0]
            for nb in neighbors:
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
    return count


def _select_by_corr_threshold(
    corr_matrix: pd.DataFrame,
    cols: Sequence[str],
    ic_map: dict[str, float],
    corr_threshold: float,
) -> tuple[list[str], list[str]]:
    to_drop: set[str] = set()
    comparisons: list[tuple[float, str, str]] = []
    for i, col_i in enumerate(cols):
        if col_i not in corr_matrix.columns:
            continue
        for j in range(i + 1, len(cols)):
            col_j = cols[j]
            if col_j not in corr_matrix.columns:
                continue
            corr_value = corr_matrix.loc[col_i, col_j]
            if pd.isna(corr_value):
                continue
            if corr_value > corr_threshold:
                comparisons.append((corr_value, col_i, col_j))

    comparisons.sort(reverse=True, key=lambda item: item[0])
    for _, col_i, col_j in comparisons:
        if col_i in to_drop or col_j in to_drop:
            continue
        ic_i = _ic_value(col_i, ic_map)
        ic_j = _ic_value(col_j, ic_map)
        if ic_i >= ic_j:
            to_drop.add(col_j)
        else:
            to_drop.add(col_i)

    remaining = [col for col in cols if col not in to_drop]
    removed = [col for col in cols if col in to_drop]
    return remaining, removed


def _select_by_hierarchical_clustering(
    corr_matrix: pd.DataFrame,
    cols: Sequence[str],
    ic_map: dict[str, float],
    corr_threshold: float,
) -> tuple[list[str], list[str]]:
    """
    先基于 |corr| 做层次聚类，再在每个簇内按阈值和 IC 进行筛选。
    只在簇内去重，不跨簇删除，可保留跨簇的高相关因子。
    """
    if len(cols) <= 1:
        return list(cols), []

    corr_abs = corr_matrix.loc[cols, cols].fillna(0.0)
    np.fill_diagonal(corr_abs.values, 1.0)

    def _choose_labels_silhouette(corr_abs: pd.DataFrame, taus: tuple[float, ...]) -> np.ndarray:
        """基于 d=1-|corr| 的 average linkage，枚举相关性下限 tau，用 silhouette 选最优簇。"""
        dist = 1.0 - corr_abs.to_numpy()
        np.fill_diagonal(dist, 0.0)
        dist = np.clip(dist, 0.0, 2.0)
        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method="average")
        best_labels: np.ndarray | None = None
        best_score = -1.0
        for tau in taus:
            t = max(0.0, 1.0 - tau)
            labels = fcluster(Z, t=t, criterion="distance")
            if len(np.unique(labels)) < 2:
                continue
            score = silhouette_score(dist, labels, metric="precomputed")
            if score > best_score:
                best_score = score
                best_labels = labels
        if best_labels is None:
            return np.ones(len(cols), dtype=int)
        return best_labels

    candidate_taus = (0.2, 0.3, 0.4, 0.5)
    labels = _choose_labels_silhouette(corr_abs, candidate_taus)
    labels_array = np.asarray(labels)
    unique_labels, counts = np.unique(labels_array, return_counts=True)
    print(f"[cluster] n_clusters={len(unique_labels)}, sizes={dict(zip(unique_labels.tolist(), counts.tolist()))}")

    remaining: list[str] = []
    removed: list[str] = []
    for label in unique_labels:
        idx = np.where(labels_array == label)[0]
        members = [cols[i] for i in idx]
        if len(members) == 1:
            remaining.append(members[0])
            continue
        sub_corr = corr_abs.loc[members, members]
        keep, drop = _select_by_corr_threshold(sub_corr, members, ic_map, corr_threshold)
        remaining.extend(keep)
        removed.extend(drop)

    return remaining, removed

# -------------------------------------------
#                  Functions                |
# -------------------------------------------

def clip_extreme_values(
    df: pd.DataFrame | pl.DataFrame | pl.LazyFrame,
    factor_cols: Iterable[str] | None = None,
    group_keys: Sequence[str] = DEFAULT_GROUP_KEYS,
):
    """
    Clamp +/-inf values per group so that they match the finite extrema observed within that group.
    """
    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    if isinstance(df, pl.DataFrame):
        cols = list(factor_cols) if factor_cols is not None else _infer_factor_cols_pl(df, group_keys)
        if not cols:
            return df
        return _clip_inf_polars(df, cols, group_keys)

    if isinstance(df, pd.DataFrame):
        cols = list(factor_cols) if factor_cols is not None else _infer_factor_cols_pd(df, group_keys)
        if not cols:
            return df.copy(deep=True)
        return _clip_inf_pandas(df, cols, group_keys)

    raise TypeError("df must be a pandas DataFrame, polars DataFrame, or polars LazyFrame.")


def compute_factor_correlations(
    df: pd.DataFrame | pl.DataFrame,
    factor_cols: Sequence[str],
    target_col: str = "target",
    method: str = "corr_cs1",
    corr_method: str = "pearson",
    date_col: str = "date_id",
    bucket_col: str = "seconds_in_bucket",
    stock_col: str = "stock_id",
) -> pd.Series:
    """
    计算因子与目标列的整体/截面/时序相关性。

    参数:
        method 可选:
            - "corr_csall": 全量计算；
            - "corr_cs1": 先按 date_id 分组，得到各天相关性再取平均；
            - "corr_cs2": 先按 (date_id, seconds_in_bucket) 分组再取平均；
            - "corr_ts": 按 stock_id 分组，描述单只股票的时序相关性。
    返回:
        pandas.Series，索引为因子名，name 为 method。
    """
    if not factor_cols:
        raise ValueError("factor_cols 不能为空。")

    method_key = method.lower()
    valid = {"corr_csall", "corr_cs1", "corr_cs2", "corr_ts"}
    if method_key not in valid:
        raise ValueError(f"method 必须为 {valid}")

    required_cols: list[str] = [target_col, *factor_cols]
    if method_key in {"corr_cs1", "corr_cs2"} and date_col not in required_cols:
        required_cols.append(date_col)
    if method_key == "corr_cs2" and bucket_col not in required_cols:
        required_cols.append(bucket_col)
    if method_key == "corr_ts" and stock_col not in required_cols:
        required_cols.append(stock_col)

    data = _to_pandas_subset(df, required_cols)

    if method_key == "corr_csall":
        corr_values = data[factor_cols].corrwith(
            data[target_col], method=corr_method
        ).to_dict()
    else:
        grouping = {
            "corr_cs1": [date_col],
            "corr_cs2": [date_col, bucket_col],
            "corr_ts": [stock_col],
        }[method_key]
        corr_values = _groupwise_corr_mean(
            data, factor_cols, target_col, grouping, corr_method
        )

    return pd.Series(corr_values, name=method_key)


def filter_correlated_factors(
    df: pd.DataFrame | pl.DataFrame,
    cols: Sequence[str],
    ic_df: pd.DataFrame | pl.DataFrame,
    corr_threshold: float = 0.07,
    factor_col: str = "factor",
    ic_col: str = "ic",
    method: str = "corr_csall",
    corr_method: str = "pearson",
    date_col: str = "date_id",
    bucket_col: str = "seconds_in_bucket",
    stock_col: str = "stock_id",
) -> tuple[list[str], list[str]]:
    """
    根据相关性和 IC 高低筛选因子，可选整体 / 截面 / 时序的相关性定义。

    步骤:
        1. 计算 df 中指定列的相关性矩阵（method 决定口径）；
        2. 从 ic_df 获取每个因子的 IC 数值；
        3. 对任意 |corr| > corr_threshold 的因子对，仅保留 IC 较高者。

    返回:
        (剩余因子列表, 被剔除因子列表)
    """
    if not cols:
        return [], []

    cols = [col for col in cols if col not in DEFAULT_EXCLUDE_COLS]
    if not cols:
        return [], []

    if not isinstance(df, (pd.DataFrame, pl.DataFrame)):
        raise TypeError("df 必须是 pandas 或 polars DataFrame。")

    if isinstance(ic_df, pl.DataFrame):
        ic_frame = ic_df.select([factor_col, ic_col]).to_pandas()
    elif isinstance(ic_df, pd.DataFrame):
        ic_frame = ic_df[[factor_col, ic_col]].copy()
    else:
        raise TypeError("ic_df 必须是 pandas 或 polars DataFrame。")

    ic_map = ic_frame.set_index(factor_col)[ic_col].to_dict()

    corr_matrix = _compute_factor_corr_matrix(
        df,
        cols,
        method=method,
        corr_method=corr_method,
        date_col=date_col,
        bucket_col=bucket_col,
        stock_col=stock_col,
    )
    if corr_matrix.empty:
        return list(cols), []

    # _render_corr_heatmap(
    #     corr_matrix,
    #     cols,
    #     f"Factor Correlation Heatmap ({method}, |corr| >= 0.7 显示)",
    # )

    corr_abs = corr_matrix.abs()
    return _select_by_hierarchical_clustering(corr_abs, cols, ic_map, corr_threshold)


def filter_correlated_factors_with_cluster(
    df: pd.DataFrame | pl.DataFrame,
    cols: Sequence[str],
    ic_df: pd.DataFrame | pl.DataFrame,
    corr_threshold: float = 0.07,
    factor_col: str = "factor",
    ic_col: str = "ic",
    method: str = "corr_csall",
    corr_method: str = "pearson",
    date_col: str = "date_id",
    bucket_col: str = "seconds_in_bucket",
    stock_col: str = "stock_id",
) -> tuple[list[str], list[str]]:
    """
    基于指定相关性口径构建谱聚类图结构，簇内保留 IC 最高的因子。
    """
    if not cols:
        return [], []

    cols = [col for col in cols if col not in DEFAULT_EXCLUDE_COLS]
    if not cols:
        return [], []

    if not isinstance(df, (pd.DataFrame, pl.DataFrame)):
        raise TypeError("df 必须是 pandas 或 polars DataFrame。")

    if isinstance(ic_df, pl.DataFrame):
        ic_frame = ic_df.select([factor_col, ic_col]).to_pandas()
    elif isinstance(ic_df, pd.DataFrame):
        ic_frame = ic_df[[factor_col, ic_col]].copy()
    else:
        raise TypeError("ic_df 必须是 pandas 或 polars DataFrame。")

    ic_map = ic_frame.set_index(factor_col)[ic_col].to_dict()

    corr_matrix = _compute_factor_corr_matrix(
        df,
        cols,
        method=method,
        corr_method=corr_method,
        date_col=date_col,
        bucket_col=bucket_col,
        stock_col=stock_col,
    )
    if corr_matrix.empty:
        return list(cols), []

    _render_corr_heatmap(
        corr_matrix,
        cols,
        f"Factor Correlation Heatmap (cluster, {method})",
    )

    corr_abs = corr_matrix.abs()
    return _select_by_hierarchical_clustering(corr_abs, cols, ic_map, corr_threshold)


def select_columns(df: pd.DataFrame | pl.DataFrame, remaining_cols: Sequence[str]):
    """
    仅保留 DEFAULT_COLS 与 remaining_cols 中出现的列。
    """
    if remaining_cols is None:
        remaining_cols = []

    keep: list[str] = []
    seen: set[str] = set()

    def _maybe_add(column: str):
        if column in seen:
            return
        if column not in df.columns:
            return
        keep.append(column)
        seen.add(column)

    for column in DEFAULT_COLS:
        _maybe_add(column)
    for column in remaining_cols:
        _maybe_add(column)

    if isinstance(df, pl.DataFrame):
        return df.select(keep)
    if isinstance(df, pd.DataFrame):
        return df.loc[:, keep].copy()
    raise TypeError("df must be a pandas or polars DataFrame.")
