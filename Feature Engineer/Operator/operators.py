"""
Polars版本的 Operators 类 - 用于快速生成大量特征
Date: 2025/11/17
"""

import pandas as pd
import polars as pl
import numpy as np
from numba import njit
import itertools
import statsmodels.api as sm
from scipy.stats import kurtosis, skew, mstats
from sklearn.linear_model import LinearRegression


EPS = 1e-9


def _winsorize_and_standardize_block(values, winsor_limit: float = 0.01) -> np.ndarray:
    """
    Apply cross-sectional winsorization + standardization to a 2-D numpy array.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    arr = arr.copy()
    arr[~np.isfinite(arr)] = np.nan

    winsor_limit = 0.0 if winsor_limit is None else float(winsor_limit)
    if winsor_limit < 0.0:
        winsor_limit = 0.0
    if winsor_limit > 0.5:
        winsor_limit = 0.5

    if winsor_limit > 0.0:
        lower_pct = winsor_limit * 100.0
        upper_pct = (1.0 - winsor_limit) * 100.0
        for idx in range(arr.shape[1]):
            col = arr[:, idx]
            finite_mask = np.isfinite(col)
            if not np.any(finite_mask):
                continue
            lo = np.nanpercentile(col[finite_mask], lower_pct)
            hi = np.nanpercentile(col[finite_mask], upper_pct)
            if not np.isfinite(lo) or not np.isfinite(hi):
                continue
            arr[finite_mask, idx] = np.clip(col[finite_mask], lo, hi)

    means = np.nanmean(arr, axis=0)
    stds = np.nanstd(arr, axis=0)
    means = np.where(np.isfinite(means), means, 0.0)
    stds = np.where(stds < EPS, 1.0, stds)
    standardized = (arr - means) / stds
    standardized = np.where(np.isfinite(standardized), standardized, 0.0)
    return standardized


@njit(cache=True, fastmath=True)
def _gram_schmidt_numba(matrix: np.ndarray) -> np.ndarray:
    rows, cols = matrix.shape
    result = np.zeros((rows, cols), dtype=np.float64)
    for j in range(cols):
        v = matrix[:, j].copy()
        for i in range(j):
            proj = 0.0
            for k in range(rows):
                proj += result[k, i] * v[k]
            for k in range(rows):
                v[k] -= proj * result[k, i]
        norm = 0.0
        for k in range(rows):
            norm += v[k] * v[k]
        norm = np.sqrt(norm)
        if norm < EPS:
            for k in range(rows):
                result[k, j] = 0.0
        else:
            inv = 1.0 / norm
            for k in range(rows):
                result[k, j] = v[k] * inv
    return result


def _gram_schmidt_columns(matrix: np.ndarray) -> np.ndarray:
    """Column-wise Gram-Schmidt orthogonalization accelerated with Numba."""
    matrix64 = np.asarray(matrix, dtype=np.float64)
    if matrix64.ndim != 2:
        raise ValueError("matrix must be 2-D")
    return _gram_schmidt_numba(matrix64)


def _symmetric_orthogonalize(matrix: np.ndarray) -> np.ndarray:
    """
    Symmetric orthogonalization (X @ (X^T X)^(-1/2)).
    """
    if matrix.size == 0:
        return np.zeros_like(matrix, dtype=np.float64)
    gram = matrix.T @ matrix
    if gram.size == 0:
        return np.zeros_like(matrix, dtype=np.float64)
    eigvals, eigvecs = np.linalg.eigh(gram)
    eigvals = np.clip(eigvals, EPS, None)
    inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    return matrix @ inv_sqrt


def _sanitize_positive_ints(values):
    """过滤并去重正整数窗口/period 参数"""
    if not values:
        return []
    cleaned = []
    seen = set()
    for value in values:
        try:
            intval = int(value)
        except (TypeError, ValueError):
            continue
        if intval <= 0 or intval in seen:
            continue
        seen.add(intval)
        cleaned.append(intval)
    return cleaned


def _series_to_float_array(series):
    """将数值列转换为 float64 numpy 数组"""
    if not isinstance(series, pl.Series):
        raise TypeError("仅支持 Polars Series 转换为 numpy 数组")
    arr = series.to_numpy()
    if arr.dtype.kind in {'i', 'u'}:
        arr = arr.astype(np.float64)
    elif arr.dtype == np.dtype('O'):
        arr = arr.astype(np.float64)
    return arr


@njit(cache=True)
def _rolling_slope_numba(values, window):
    """Numba slope加速"""
    n = values.shape[0]
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan
    if window <= 0:
        return result
    for idx in range(n):
        start = idx - window + 1
        if start < 0:
            start = 0
        segment = values[start:idx + 1]
        mask = ~np.isnan(segment)
        valid = segment[mask]
        if valid.size < 2:
            continue
        x = np.arange(valid.size, dtype=np.float64)
        mean_x = x.mean()
        mean_y = valid.mean()
        denom = np.sum((x - mean_x) ** 2)
        if denom == 0:
            continue
        numer = np.sum((x - mean_x) * (valid - mean_y))
        result[idx] = numer / denom
    return result


@njit(fastmath=True, nogil=True)
def _rolling_rank_numba(values, window, out):
    n = len(values)
    for i in range(n):
        start_idx = max(0, i - window + 1)
        end_idx = i + 1
        window_size = end_idx - start_idx
        if window_size == 0:
            out[i] = np.nan
        else:
            current_val = values[i]
            rank_sum = 0.0
            for j in range(start_idx, end_idx):
                if values[j] <= current_val:
                    rank_sum += 1.0
            out[i] = rank_sum / window_size
    return out


@njit(fastmath=True, nogil=True)
def _rolling_beta_stateful(symbol_idx, low, close, window):
    n = low.shape[0]
    n_symbols = np.max(symbol_idx) + 1

    buffer_low = np.empty((n_symbols, window), dtype=np.float64)
    buffer_close = np.empty((n_symbols, window), dtype=np.float64)
    counts = np.zeros(n_symbols, dtype=np.int64)
    positions = np.zeros(n_symbols, dtype=np.int64)
    sum_low = np.zeros(n_symbols, dtype=np.float64)
    sum_close = np.zeros(n_symbols, dtype=np.float64)
    sum_low_sq = np.zeros(n_symbols, dtype=np.float64)
    sum_low_close = np.zeros(n_symbols, dtype=np.float64)

    out = np.empty(n, dtype=np.float64)

    for i in range(n):
        sid = symbol_idx[i]
        pos = positions[sid]
        count = counts[sid]

        if count == window:
            old_low = buffer_low[sid, pos]
            old_close = buffer_close[sid, pos]
            sum_low[sid] -= old_low
            sum_close[sid] -= old_close
            sum_low_sq[sid] -= old_low * old_low
            sum_low_close[sid] -= old_low * old_close
            count += 1
            counts[sid] = count

        cur_low = low[i]
        cur_close = close[i]
        buffer_low[sid, pos] = cur_low
        buffer_close[sid, pos] = cur_close
        sum_low[sid] += cur_low
        sum_close[sid] += cur_close
        sum_low_sq[sid] += cur_low * cur_low
        sum_low_close[sid] += cur_low * cur_close
        pos += 1
        if pos == window:
            pos = 0
        positions[sid] = pos

        if count < 2:
            out[i] = np.nan
            continue

        inv_count = 1.0 / count
        cov = (sum_low_close[sid] - (sum_low[sid] * sum_close[sid]) * inv_count) / (count - 1.0)
        var_low = (sum_low_sq[sid] - (sum_low[sid] * sum_low[sid]) * inv_count) / (count - 1.0)

        if var_low < 1e-6:
            out[i] = 0.0
        else:
            out[i] = cov / var_low

    return out


class Operators:
    """
    统一的算子类，包含截面算子(CS)和时序算子(TS)
    量纲/去量纲提示可查阅 DIMENSION_GUIDE 常量
    
    使用方法:
        df = Operators.cs_add(df, ['col1', 'col2'])
        df = Operators.ts_mean(df, ['col1'], windows=[5, 10, 20])
    """
    # ==================== 原子操作算子 (Atomic Operators) ====================

    @staticmethod
    def at_abs(df, cols):
        """绝对值"""
        exprs = [pl.col(col).abs().alias(f'at_abs_{col}') for col in cols]
        return df.with_columns(exprs)

    @staticmethod
    def at_log(df, cols):
        """对数变换 (加1防止对0取对数)"""
        exprs = [(pl.col(col) + 1.0).log().alias(f'at_log_{col}') for col in cols ]
        return df.with_columns(exprs)

    @staticmethod
    def at_sign(df, cols):
        """符号函数"""
        exprs = [pl.col(col).sign().alias(f'at_sign_{col}') for col in cols ]
        return df.with_columns(exprs)
    
    @staticmethod
    def at_signlog(df, cols):
        """带符号的对数"""
        exprs = [
            (pl.col(col).sign() * (pl.col(col).abs() + 1.0).log()).alias(f'at_signlog_{col}')
            for col in cols
        ]
        return df.with_columns(exprs)

    @staticmethod
    def at_signsqrt(df, cols):
        """
        带符号的平方根，先按 time_id 做 1% 截尾，再除以截尾均值的绝对值后取符号开方
        """
        group_col = "time_id"
        exprs = []
        for col in cols:
            lower = pl.col(col).quantile(0.01).over(group_col)
            upper = pl.col(col).quantile(0.99).over(group_col)
            wins = (
                pl.when(pl.col(col) > upper)
                .then(upper)
                .when(pl.col(col) < lower)
                .then(lower)
                .otherwise(pl.col(col))
            )
            scale = wins.mean().over(group_col).abs()
            safe_scale = pl.when(scale < EPS).then(pl.lit(1.0)).otherwise(scale)
            norm = wins / safe_scale
            exprs.append((norm.sign() * norm.abs().sqrt()).alias(f'at_signsqrt_{col}'))
        return df.with_columns(exprs)
    
    @staticmethod
    def at_sigmoid(df: pl.DataFrame, cols):
        """Sigmoid 函数"""
        exprs = [
            (1 / (1 + (-pl.col(col)).exp())).alias(f"at_sigmoid_{col}")
            for col in cols
        ]
        return df.with_columns(exprs)

    @staticmethod
    def at_square(df: pl.DataFrame, cols):
        """
        平方：先按 time_id 做 1% 截尾并除以截尾均值绝对值，再平方
        """
        group_col = "time_id"
        exprs = []
        for col in cols:
            lower = pl.col(col).quantile(0.01).over(group_col)
            upper = pl.col(col).quantile(0.99).over(group_col)
            wins = (
                pl.when(pl.col(col) > upper)
                .then(upper)
                .when(pl.col(col) < lower)
                .then(lower)
                .otherwise(pl.col(col))
            )
            scale = wins.mean().over(group_col).abs()
            safe_scale = pl.when(scale < EPS).then(pl.lit(1.0)).otherwise(scale)
            norm = wins / safe_scale
            exprs.append((norm ** 2).alias(f"at_square_{col}"))
        return df.with_columns(exprs)

    @staticmethod
    def at_cube(df: pl.DataFrame, cols):
        """
        立方：先按 time_id 做 1% 截尾并除以截尾均值绝对值，再立方
        """
        group_col = "time_id"
        exprs = []
        for col in cols:
            lower = pl.col(col).quantile(0.01).over(group_col)
            upper = pl.col(col).quantile(0.99).over(group_col)
            wins = (
                pl.when(pl.col(col) > upper)
                .then(upper)
                .when(pl.col(col) < lower)
                .then(lower)
                .otherwise(pl.col(col))
            )
            scale = wins.mean().over(group_col).abs()
            safe_scale = pl.when(scale < EPS).then(pl.lit(1.0)).otherwise(scale)
            norm = wins / safe_scale
            exprs.append((norm ** 3).alias(f"at_cube_{col}"))
        return df.with_columns(exprs)

    # ==================== 截面算子 (Cross Section Operators) ====================
    @staticmethod
    def cs_add(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
        """截面加法 - 先按 time_id 对每列做 zscore，再两两相加"""
        cols = list(cols or [])
        if len(cols) < 2:
            return df

        group_col = "time_id"
        temp_cols = []
        z_exprs = []
        for col in cols:
            temp = f"__cs_z_{col}"
            temp_cols.append(temp)
            col_expr = pl.col(col)
            mean_expr = col_expr.mean().over(group_col)
            std_expr = col_expr.std(ddof=1).over(group_col)
            z_exprs.append(
                pl.when(std_expr.is_null() | (std_expr == 0))
                .then(None)
                .otherwise((col_expr - mean_expr) / std_expr)
                .alias(temp)
            )

        df = df.with_columns(z_exprs)
        exprs = [
            (pl.col(f"__cs_z_{c1}") + pl.col(f"__cs_z_{c2}")).alias(f'cs_add_{c1}_{c2}')
            for c1, c2 in itertools.combinations(cols, 2)
        ]
        df = df.with_columns(exprs) if exprs else df
        return df.drop(temp_cols)
    
    @staticmethod
    def cs_subtract(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
        """截面减法 - 先按 time_id 对每列做 zscore，再两两相减"""
        cols = list(cols or [])
        if len(cols) < 2:
            return df

        group_col = "time_id"
        temp_cols = []
        z_exprs = []
        for col in cols:
            temp = f"__cs_z_{col}"
            temp_cols.append(temp)
            col_expr = pl.col(col)
            mean_expr = col_expr.mean().over(group_col)
            std_expr = col_expr.std(ddof=1).over(group_col)
            z_exprs.append(
                pl.when(std_expr.is_null() | (std_expr == 0))
                .then(None)
                .otherwise((col_expr - mean_expr) / std_expr)
                .alias(temp)
            )

        df = df.with_columns(z_exprs)
        exprs = [
            (pl.col(f"__cs_z_{c1}") - pl.col(f"__cs_z_{c2}")).alias(f"cs_subtract_{c1}_{c2}")
            for c1, c2 in itertools.combinations(cols, 2)
        ]
        df = df.with_columns(exprs) if exprs else df
        return df.drop(temp_cols)
    
    @staticmethod
    def cs_multiply(df: pl.DataFrame, cols: list[str], col_pairs: list[tuple[str, str]] | None = None) -> pl.DataFrame:
        """截面乘法 - 对列的两两组合进行乘法；支持自定义列对 col_pairs"""
        pairs = col_pairs or itertools.combinations(cols, 2)
        exprs = [
            (pl.col(c1) * pl.col(c2)).alias(f"cs_multiply_{c1}_{c2}")
            for c1, c2 in pairs
            if c1 in df.columns and c2 in df.columns and c1 != c2
        ]
        return df.with_columns(exprs) if exprs else df
    
    @staticmethod
    def cs_divide(df: pl.DataFrame, cols: list[str], col_pairs: list[tuple[str, str]] | None = None) -> pl.DataFrame:
        """截面除法 - 对列的两两组合进行除法；支持自定义列对 col_pairs"""
        pairs = col_pairs or itertools.combinations(cols, 2)
        exprs = [
            (
                pl.col(c1)
                / pl.when(pl.col(c2) == 0).then(pl.lit(EPS)).otherwise(pl.col(c2))
            ).alias(f"cs_divide_{c1}_{c2}")
            for c1, c2 in pairs
            if c1 in df.columns and c2 in df.columns and c1 != c2
        ]
        return df.with_columns(exprs) if exprs else df

    @staticmethod
    def cs_max(df, cols):
        """截面最大值 - 每行取多列最大值"""
        return df.with_columns(
            pl.max_horizontal([pl.col(c) for c in cols]).alias("cs_max")
        )

    @staticmethod
    def cs_min(df, cols):
        """截面最小值 - 每行取多列最小值"""
        return df.with_columns(
            pl.min_horizontal([pl.col(c) for c in cols]).alias("cs_min")
        )

    @staticmethod
    def cs_range(df, cols):
        """截面范围 = max - min"""
        return df.with_columns([
            (pl.max_horizontal([pl.col(c) for c in cols]) -
             pl.min_horizontal([pl.col(c) for c in cols])
            ).alias("cs_range")
        ])

    @staticmethod
    def cs_mean(df, cols):
        """截面均值"""
        return df.with_columns(
            pl.mean_horizontal([pl.col(c) for c in cols]).alias("cs_mean")
        )

    @staticmethod
    def cs_std(df, cols):
        """截面标准差（横向 std）"""
        return df.with_columns([
            pl.concat_list([pl.col(c) for c in cols])
            .list.std()
            .alias("cs_std")
        ])

    @staticmethod
    def cs_cross_feature(df, cols, col_pairs: list[tuple[str, str]] | None = None):
        """交叉特征 - 生成列之间的交叉乘积；支持指定 col_pairs"""
        pairs = col_pairs or itertools.combinations(cols, 2)
        exprs = [
            (pl.col(c0) * pl.col(c1)).alias(f'cs_cross_{c0}_{c1}')
            for c0, c1 in pairs
            if c0 in df.columns and c1 in df.columns and c0 != c1
        ]
        return df.with_columns(exprs) if exprs else df

    @staticmethod
    def cs_zscore_add(df, cols, group_col='time_id'):
        """先对每列进行截面标准化，再进行两两加法"""
        zscore_exprs = [
            (
                (
                    pl.col(col)
                    - pl.col(col)
                    .filter(pl.col(col).is_not_null())
                    .mean()
                    .over(group_col)
                )
                / pl.col(col)
                .filter(pl.col(col).is_not_null())
                .std()
                .over(group_col)
            ).alias(f"cs_zscore_{col}")
            for col in cols
        ]

        df = df.with_columns(zscore_exprs)

        add_exprs = [
            (pl.col(f"cs_zscore_{c0}") + pl.col(f"cs_zscore_{c1}")).alias(f'cs_zscore_add_{c0}_{c1}')
            for c0, c1 in itertools.combinations(cols, 2)
        ]

        return df.with_columns(add_exprs).select(
            pl.exclude([f"cs_zscore_{col}" for col in cols])
        )
    
    @staticmethod
    def cs_rank_add(df, cols, group_col='time_id'):
        """先对每列进行截面排名，再进行两两加法"""
        rank_exprs = [
            (
                pl.col(col)
                .rank(method='average')
                .over(group_col)
                / pl.col(col).count().over(group_col)
            ).alias(f"cs_rank_{col}")
            for col in cols
        ]
        df = df.with_columns(rank_exprs)

        add_exprs = [
            (pl.col(f"cs_rank_{c0}") + pl.col(f"cs_rank_{c1}")).alias(f'cs_rank_add_{c0}_{c1}')
            for c0, c1 in itertools.combinations(cols, 2)
        ]
        return df.with_columns(add_exprs)

    @staticmethod
    def cs_zscore(df, cols, group_col='time_id'):
        """对每列进行截面标准化"""
        exprs = []
        for col in cols:
            col_expr = pl.col(col)
            mean_expr = col_expr.mean().over(group_col)
            std_expr = col_expr.std(ddof=1).over(group_col)
            exprs.append(
                pl.when(std_expr.is_null() | (std_expr == 0))
                .then(None)
                .otherwise((col_expr - mean_expr) / std_expr)
                .alias(f'cs_zscore_{col}')
            )
        return df.with_columns(exprs)

    @staticmethod
    def cs_rank(df: pl.DataFrame, cols, group_col: str = "time_id"):
        """截面排名 - 在每个时间点内对指定列进行排名"""
        exprs = [
            (pl.col(col).rank(method="average").over(group_col) / pl.count().over(group_col)).alias(f"cs_rank_{col}")
            for col in cols
        ]
        return df.with_columns(exprs)

    @staticmethod
    def cs_deviation_from_mean(df: pl.DataFrame, cols, group_col: str = "time_id"):
        """偏离均值 - 计算每个值与组内均值的偏离"""
        exprs = [
            (pl.col(col) - pl.col(col).mean().over(group_col)).alias(f"cs_dev_mean_{col}")
            for col in cols
        ]
        return df.with_columns(exprs)
    
    @staticmethod
    def cs_deviation_from_max(df: pl.DataFrame, cols, group_col: str = "time_id"):
        """偏离最大值 - 计算每个值与组内最大值的偏离"""
        exprs = [
            (pl.col(col) - pl.col(col).max().over(group_col)).alias(f"cs_dev_max_{col}")
            for col in cols
        ]
        return df.with_columns(exprs)
    
    @staticmethod
    def cs_deviation_from_min(df: pl.DataFrame, cols, group_col: str = "time_id"):
        """偏离最小值 - 计算每个值与组内最小值的偏离"""
        exprs = [
            (pl.col(col) - pl.col(col).min().over(group_col)).alias(f"cs_dev_min_{col}")
            for col in cols
        ]
        return df.with_columns(exprs)

    @staticmethod
    def _cs_orthogonal_add(
        df: pl.DataFrame,
        cols,
        group_col: str = "time_id",
        winsor_limit: float = 0.01,
        method: str = "schmidt",
    ):
        """Helper for cross-sectional orthogonalization-then-sum operators."""
        cols = [cols] if isinstance(cols, str) else list(cols)
        if len(cols) < 2:
            return df
        missing = [col for col in cols if col not in df.columns]
        if missing:
            raise KeyError(f"missing columns for cs orthogonal add: {missing}")
        if group_col not in df.columns:
            raise KeyError(f"group_col '{group_col}' not found in DataFrame")

        temp = df.select([pl.arange(0, pl.len()).alias("__row_idx"), group_col, *cols])
        pdf = temp.to_pandas()
        values = np.full(df.height, np.nan, dtype=np.float32)
        grouped = pdf.groupby(group_col, sort=False)
        for _, block in grouped:
            row_idx = block["__row_idx"].to_numpy()
            matrix = block[cols].to_numpy(dtype=np.float64, copy=True)
            if matrix.size == 0:
                continue
            prepped = _winsorize_and_standardize_block(matrix, winsor_limit=winsor_limit)
            if method == "schmidt":
                ortho = _gram_schmidt_columns(prepped)
            elif method == "symmetric":
                ortho = _symmetric_orthogonalize(prepped)
            else:
                raise ValueError(f"unsupported orthogonalization method: {method}")
            summed = np.sum(ortho, axis=1)
            values[row_idx] = summed.astype(np.float32, copy=False)

        prefix = "cs_schmidt_add" if method == "schmidt" else "cs_sym_ortho_add"
        feature_name = f"{prefix}_{'_'.join(cols)}"
        return df.with_columns(pl.Series(feature_name, values))

    @staticmethod
    def cs_schmidt_add(
        df: pl.DataFrame,
        cols,
        group_col: str = "time_id",
        winsor_limit: float = 0.01,
    ):
        """Schmidt orthogonalization + sum (winsorize -> zscore -> Gram-Schmidt)."""
        return Operators._cs_orthogonal_add(
            df,
            cols=cols,
            group_col=group_col,
            winsor_limit=winsor_limit,
            method="schmidt",
        )

    @staticmethod
    def cs_symmetric_orth_add(
        df: pl.DataFrame,
        cols,
        group_col: str = "time_id",
        winsor_limit: float = 0.01,
    ):
        """Symmetric orthogonalization (X(X'X)^(-1/2)) + sum."""
        return Operators._cs_orthogonal_add(
            df,
            cols=cols,
            group_col=group_col,
            winsor_limit=winsor_limit,
            method="symmetric",
        )

    @staticmethod
    def _cs_decompose(df, cols, n=4, group_col='time_id', activation='sigmoid'):
        """
        Helper that builds cross-sectional decompositions using a single column list.
        Every ordered pair (x, y) from cols is transformed as
        multiply(x, activation(standardize(winsorize(y)))).
        """
        cols = [cols] if isinstance(cols, str) else list(cols)
        if not cols:
            print('缺少必要行')
            return df
        available_cols = set(df.columns)
        has_wap = 'wap' in cols and 'wap' in available_cols
        if not has_wap:
            return df

        ts_group_col = 'stock_id' if 'stock_id' in available_cols else group_col
        target_cols = []
        mean_exprs = []
        for window in (5, 10):
            target_name = f'wap_{window}mean'
            target_cols.append(target_name)
            mean_exprs.append(
                pl.col('wap')
                .rolling_mean(window_size=window, min_periods=1)
                .over(ts_group_col)
                .alias(target_name)
            )
        df = df.with_columns(mean_exprs)

        processed_cols = []
        seen = set()
        for col in cols + target_cols:
            if col in available_cols or col in target_cols:
                if col not in seen:
                    processed_cols.append(col)
                    seen.add(col)
        cols = processed_cols

        # 只有 target_cols 需要做截尾/标准化，other_cols 只是参与乘法
        target_cols = [col for col in target_cols if col in cols]
        other_cols = [col for col in cols if col not in target_cols]
        if not target_cols or not other_cols:
            return df

        median_cols = []
        mad_cols = []
        wins_cols = []
        mean_cols = []
        std_cols = []
        lookup = {}

        median_exprs = []
        for idx, col in enumerate(target_cols):
            median_col = f'__cs_decompose_median_{idx}'
            median_cols.append(median_col)
            median_exprs.append(pl.col(col).median().over(group_col).alias(median_col))
        df = df.with_columns(median_exprs)

        mad_exprs = []
        for idx, col in enumerate(target_cols):
            median_col = median_cols[idx]
            mad_col = f'__cs_decompose_mad_{idx}'
            mad_cols.append(mad_col)
            mad_exprs.append(
                (pl.col(col) - pl.col(median_col))
                .abs()
                .median()
                .over(group_col)
                .alias(mad_col)
            )
        df = df.with_columns(mad_exprs)

        wins_exprs = []
        for idx, col in enumerate(target_cols):
            median_col = median_cols[idx]
            mad_col = mad_cols[idx]
            wins_col = f'__cs_decompose_wins_{idx}'
            wins_cols.append(wins_col)
            safe_mad = pl.when(pl.col(mad_col) < EPS).then(pl.lit(EPS)).otherwise(pl.col(mad_col))
            lower = pl.col(median_col) - pl.lit(n) * safe_mad
            upper = pl.col(median_col) + pl.lit(n) * safe_mad
            wins_exprs.append(
                pl.when(pl.col(col) > upper)
                .then(upper)
                .when(pl.col(col) < lower)
                .then(lower)
                .otherwise(pl.col(col))
                .alias(wins_col)
            )
        df = df.with_columns(wins_exprs)

        moment_exprs = []
        for idx, wins_col in enumerate(wins_cols):
            mean_col = f'__cs_decompose_mean_{idx}'
            std_col = f'__cs_decompose_std_{idx}'
            mean_cols.append(mean_col)
            std_cols.append(std_col)
            lookup[target_cols[idx]] = (wins_col, mean_col, std_col)
            moment_exprs.extend(
                [
                    pl.col(wins_col).mean().over(group_col).alias(mean_col),
                    pl.col(wins_col).std().over(group_col).alias(std_col),
                ]
            )
        df = df.with_columns(moment_exprs)

        def apply_activation(expr):
            if activation == 'sigmoid':
                return 1.0 / (1.0 + (-expr).exp())
            if activation == 'tanh':
                return expr.tanh()
            raise ValueError(f"Unsupported activation: {activation}")

        final_exprs = []
        suffix = activation
        for target_col in target_cols:
            wins_col, mean_col, std_col = lookup[target_col]
            standardized = (pl.col(wins_col) - pl.col(mean_col)) / (pl.col(std_col) + EPS)
            activated = apply_activation(standardized)
            for x_col in other_cols:
                final_exprs.append(
                    (pl.col(x_col) * activated).alias(f'cs_decompose_{suffix}_{x_col}_{target_col}')
                )

        df = df.with_columns(final_exprs)
        temp_cols = median_cols + mad_cols + wins_cols + mean_cols + std_cols
        df = df.drop(temp_cols)
        return df

    @staticmethod
    def cs_decompose_sigmoid(df, cols, n=4, group_col='time_id'):
        """
        Cross-sectional decomposition with sigmoid activation:
        multiply X by sigmoid(standardize(winsorize(Y))).
        """
        return Operators._cs_decompose(
            df,
            cols=cols,
            n=n,
            group_col=group_col,
            activation='sigmoid',
        )

    @staticmethod
    def cs_decompose_tanh(df, cols, n=4, group_col='time_id'):
        """
        Cross-sectional decomposition with tanh activation:
        multiply X by tanh(standardize(winsorize(Y))).
        """
        return Operators._cs_decompose(
            df,
            cols=cols,
            n=n,
            group_col=group_col,
            activation='tanh',
        )
    
    # ==================== 时序算子 (Time Series Operators) ====================
    def ts_max(df, cols, windows=[], group_col='stock_id'):
        """时序最大值 - 计算滚动窗口最大值"""
        exprs = [
            pl.col(col).rolling_max(window_size=window).over(group_col).alias(f"ts_max_{col}_{window}")
            for col in cols
            for window in windows
        ]
        return df.with_columns(exprs)
    
    @staticmethod
    def ts_min(df, cols, windows=[], group_col='stock_id'):
        """时序最小值 - 计算滚动窗口最小值"""
        exprs = [
            pl.col(col).rolling_min(window_size=window).over(group_col).alias(f"ts_min_{col}_{window}")
            for col in cols
            for window in windows
        ]
        return df.with_columns(exprs)

    @staticmethod
    def ts_sum(
        df: pl.DataFrame,
        cols,
        windows=[],
        group_col='stock_id',
        order_col=None
    ):

        # 构造表达式
        exprs = [
            pl.col(col)
            .rolling_sum(window_size=window, min_samples=1)
            .over(group_col)
            .alias(f"ts_sum_{col}_{window}")
            for col in cols
            for window in windows
        ]

        return df.with_columns(exprs) if exprs else df
    
    @staticmethod
    def ts_diff(df, cols, periods=None, group_col='stock_id'):
        """时序差分 - 计算滞后差值"""
        exprs = [
            pl.col(col).diff(n=period).over(group_col).alias(f'ts_diff_{col}_{period}')
            for col in cols
            for period in periods
        ]
        return df.with_columns(exprs)

    @staticmethod
    def ts_pct_change(df, cols, periods=None, group_col='stock_id'):
        """时序百分比变化率"""
        exprs = []
        for col in cols:
            column = pl.col(col)
            for period in periods:
                shifted = column.shift(period).over(group_col)
                exprs.append(((column / shifted) - 1.0).alias(f'ts_pct_change_{col}_{period}'))
        return df.with_columns(exprs)

    @staticmethod
    def ts_log_return(df, cols, periods=None, group_col='stock_id'):
        """时序对数收益"""
        exprs = []
        for col in cols:
            column = pl.col(col)
            for period in periods:
                shifted = column.shift(period).over(group_col)
                exprs.append((column / shifted).log().alias(f'ts_log_return_{col}_{period}'))
        return df.with_columns(exprs)

    @staticmethod
    def ts_zscore(df, cols, windows=None, group_col='stock_id'):
        """时序标准化 - 滚动窗口内标准化"""
        exprs = [
            (
                (pl.col(col) - pl.col(col).rolling_mean(window_size=window, min_periods=1))
                / pl.col(col).rolling_std(window_size=window, min_periods=1).fill_nan(0)
            )
            .over(group_col)
            .alias(f"ts_zscore_{col}_{window}")
            for col in cols
            for window in windows
        ]
        return df.with_columns(exprs)

    @staticmethod
    def ts_rank(df, cols, windows=None, group_col='stock_id'):
        """时序排名 - 计算滚动窗口内的相对排名"""
        exprs = [
            pl.col(col)
            .rolling_rank(window_size=window, method='average')
            .over(group_col)
            .alias(f"ts_rank_{col}_{window}")
            for col in cols
            for window in windows
        ]
        return df.with_columns(exprs)

    # new added
    @staticmethod
    def ts_rolling_rank(df, cols, windows=None, group_col='stock_id'):
        """时序滚动排名 - 使用 Numba 计算窗口内排名占比"""
        windows = _sanitize_positive_ints(windows)
        cols = [cols] if isinstance(cols, str) else list(cols or [])
        if not windows or not cols:
            return df

        helper_col = '__ts_rolling_rank_order'

        def _apply(group: pl.DataFrame) -> pl.DataFrame:
            new_series = []
            for col in cols:
                if col not in group.columns:
                    continue
                values = _series_to_float_array(group[col])
                length = values.shape[0]
                for window in windows:
                    result = np.empty(length, dtype=np.float32)
                    _rolling_rank_numba(values, window, result)
                    new_series.append(
                        pl.Series(f"ts_rolling_rank_{col}_{window}", result)
                    )
            return group.with_columns(new_series) if new_series else group

        return (
            df.with_row_index(helper_col)
            .group_by(group_col, maintain_order=True)
            .map_groups(_apply)
            .sort(helper_col)
            .drop(helper_col)
        )

    @staticmethod
    def ts_corr(df, col_pairs, windows=None, group_col='stock_id'):
        """时序相关系数 - 计算两列的滚动相关性"""
        exprs = []
        for col1, col2 in col_pairs:
            for window in windows:
                exprs.append(
                    pl.col(col1)
                    .rolling_corr(pl.col(col2), window_size=window, min_samples=1)
                    .over(group_col)
                    .alias(f'ts_corr_{col1}_{col2}_{window}')
                )
        return df.with_columns(exprs)

    @staticmethod
    def ts_slope(df, cols, windows=None, group_col='stock_id'):
        """时序斜率 - 滚动窗口线性回归斜率"""
        helper_col = '__ts_slope_order'
        def _apply(group: pl.DataFrame) -> pl.DataFrame:
            new_series = []
            for col in cols:
                if col not in group.columns:
                    continue
                values = _series_to_float_array(group[col])
                for window in windows:
                    arr = _rolling_slope_numba(values, window)
                    new_series.append(pl.Series(f"ts_slope_{col}_{window}", arr))
            return group.with_columns(new_series) if new_series else group

        return (
            df.with_row_index(helper_col)
            .group_by(group_col, maintain_order=True)
            .map_groups(_apply)
            .sort(helper_col)
            .drop(helper_col)
        )

    # new added
    @staticmethod
    def ts_regbeta(df, cols=None, windows=None, group_col='stock_id', target_col='wap'):
        """时序回归 Beta - 将每个 cols 与 target_col 的滚动均值(5、10)配对计算回归系数"""
        cols = [cols] if isinstance(cols, str) else list(cols or [])
        if not windows or not cols:
            return df

        available = set(df.columns)
        if target_col not in available:
            return df

        cols = [col for col in cols if col in available and col != target_col]
        if not cols:
            return df

        helper_col = '__ts_regbeta_order'
        target_mean_5_col = '__ts_regbeta_target_mean_5'
        target_mean_10_col = '__ts_regbeta_target_mean_10'
        work_df = (
            df.with_row_index(helper_col)
            .with_columns(
                pl.col(target_col)
                .rolling_mean(5, min_periods=1)
                .over(group_col)
                .alias(target_mean_5_col),
                pl.col(target_col)
                .rolling_mean(10, min_periods=1)
                .over(group_col)
                .alias(target_mean_10_col),
            )
        )

        if group_col not in work_df.columns:
            raise ValueError(f"group_col '{group_col}' 不存在于 DataFrame 中")

        group_series = work_df[group_col]
        if group_series.dtype != pl.Categorical:
            group_series = group_series.cast(pl.Utf8, strict=False).cast(pl.Categorical)
        else:
            group_series = group_series.clone()

        symbol_codes = (
            group_series.to_physical().to_numpy().astype(np.int64, copy=False)
        )

        target_mean_5 = work_df[target_mean_5_col].to_numpy().astype(np.float64, copy=False)
        target_mean_10 = work_df[target_mean_10_col].to_numpy().astype(np.float64, copy=False)
        target_variants = [
            ('5mean', target_mean_5),
            ('10mean', target_mean_10),
        ]

        new_series = []
        for x_col in cols:
            if x_col not in work_df.columns:
                continue
            x_values = work_df[x_col].to_numpy().astype(np.float64, copy=False)
            for window in windows:
                for suffix, target_values in target_variants:
                    beta = _rolling_beta_stateful(symbol_codes, x_values, target_values, window)
                    new_series.append(
                        pl.Series(f"ts_regbeta_{x_col}_{target_col}_{suffix}_{window}", beta)
                    )

        if not new_series:
            return df

        return (
            work_df.with_columns(new_series)
            .sort(helper_col)
            .drop(helper_col, target_mean_5_col, target_mean_10_col)
        )

    @staticmethod
    def ts_mean(df, cols, windows=None, group_col='stock_id'):
        """时序均值 - 计算滚动窗口均值"""
        windows = _sanitize_positive_ints(windows)
        cols = [cols] if isinstance(cols, str) else list(cols or [])
        if not windows or not cols:
            return df

        available_cols = set(df.columns)
        exprs = []
        for col in cols:
            if col not in available_cols:
                continue
            base = pl.col(col)
            for window in windows:
                exprs.append(
                    base.rolling_mean(window_size=window, min_periods=1)
                    .over(group_col)
                    .alias(f"ts_mean_{col}_{window}")
                )
        return df.with_columns(exprs) if exprs else df
    
    @staticmethod
    def ts_std(df, cols, windows=None, group_col='stock_id'):
        """时序标准差 - 计算滚动窗口标准差"""
        windows = _sanitize_positive_ints(windows)
        cols = [cols] if isinstance(cols, str) else list(cols or [])
        if not windows or not cols:
            return df

        available_cols = set(df.columns)
        exprs = []
        for col in cols:
            if col not in available_cols:
                continue
            base = pl.col(col)
            for window in windows:
                exprs.append(
                    base.rolling_std(window_size=window, min_periods=1, ddof=1)
                    .over(group_col)
                    .alias(f"ts_std_{col}_{window}")
                )
        return df.with_columns(exprs) if exprs else df
    
    @staticmethod
    def ts_skew(df, cols, windows=None, group_col='stock_id'):
        """时序偏度 - 计算滚动窗口偏度"""
        windows = _sanitize_positive_ints(windows)
        cols = [cols] if isinstance(cols, str) else list(cols or [])
        if not windows or not cols:
            return df

        available_cols = set(df.columns)
        exprs = []
        for col in cols:
            if col not in available_cols:
                continue
            base = pl.col(col)
            for window in windows:
                exprs.append(
                    base.rolling_skew(window_size=window, min_samples=3, bias=False)
                    .over(group_col)
                    .alias(f"ts_skew_{col}_{window}")
                )
        return df.with_columns(exprs) if exprs else df

    @staticmethod
    def ts_kurt(df, cols, windows=None, group_col='stock_id'):
        """时序峰度 - 计算滚动窗口峰度"""
        windows = _sanitize_positive_ints(windows)
        cols = [cols] if isinstance(cols, str) else list(cols or [])
        if not windows or not cols:
            return df

        available_cols = set(df.columns)
        exprs = []
        for col in cols:
            if col not in available_cols:
                continue
            base = pl.col(col)
            for window in windows:
                exprs.append(
                    base.rolling_kurtosis(window_size=window, min_samples=4, fisher=True)
                    .over(group_col)
                    .alias(f"ts_kurt_{col}_{window}")
                )
        return df.with_columns(exprs) if exprs else df

    @staticmethod
    def ts_lag(df, cols, periods=None, group_col='stock_id'):
        """生成滞后特征"""
        if not periods:
            return df
        exprs = [
            pl.col(col).shift(period).over(group_col).alias(f'ts_lag_{col}_{period}')
            for col in cols
            for period in periods
        ]
        return df.with_columns(exprs)

    @staticmethod
    def ts_momentum(df, cols, periods=None, group_col='stock_id'):
        """时序动量 - 与ts_diff一致，此处保留独立命名"""
        return Operators.ts_diff(df, cols, periods=periods, group_col=group_col)

    @staticmethod
    def ts_ema(df, cols, spans=None, group_col='stock_id'):
        """时序指数移动平均 (EMA)"""
        spans = spans or []
        exprs = [
            pl.col(col)
            .ewm_mean(span=span, adjust=False)
            .over(group_col)
            .alias(f'ts_ema_{col}_{span}')
            for col in cols
            for span in spans
        ]
        df = df.with_columns(exprs) if exprs else df
        return df

    @staticmethod
    def ts_rsi(df: pl.DataFrame, cols, period: int = 14, group_col: str = "stock_id"):
        """(RSI)"""
        cols = [cols] if isinstance(cols, str) else list(cols)
        if not cols:
            return df

        delta_exprs = [
            pl.col(col).diff().over(group_col).alias(f"__delta_{col}") for col in cols
        ]
        df = df.with_columns(delta_exprs)

        gain_loss_exprs = []
        for col in cols:
            delta = pl.col(f"__delta_{col}")
            gain_loss_exprs.extend(
                [
                    delta.clip(lower_bound=0)
                    .rolling_mean(period)
                    .over(group_col)
                    .alias(f"__gain_{col}"),
                    (-delta)
                    .clip(lower_bound=0)
                    .rolling_mean(period)
                    .over(group_col)
                    .alias(f"__loss_{col}"),
                ]
            )
        df = df.with_columns(gain_loss_exprs)

        rsi_exprs = [
            (
                100
                - 100
                / (1 + pl.col(f"__gain_{col}") / pl.col(f"__loss_{col}"))
            ).alias(f"ts_rsi_{col}_{period}")
            for col in cols
        ]
        df = df.with_columns(rsi_exprs)

        drop_cols = (
            [f"__delta_{col}" for col in cols]
            + [f"__gain_{col}" for col in cols]
            + [f"__loss_{col}" for col in cols]
        )
        return df.drop(drop_cols)

    @staticmethod
    def ts_macd(
        df: pl.DataFrame,
        cols,
        short_window: int = 12,
        long_window: int = 26,
        signal_window: int = 9,
        group_col: str = "stock_id",
    ):
        """ (MACD)"""
        cols = [cols] if isinstance(cols, str) else list(cols)
        if not cols:
            return df

        ema_exprs = [
            pl.col(col)
            .ewm_mean(span=span, adjust=False)
            .over(group_col)
            .alias(f"__macd_{label}_{col}")
            for col in cols
            for span, label in ((short_window, "short"), (long_window, "long"))
        ]
        df = df.with_columns(ema_exprs)

        macd_exprs = [
            (
                pl.col(f"__macd_short_{col}") - pl.col(f"__macd_long_{col}")
            ).alias(f"ts_macd_{col}")
            for col in cols
        ]
        df = df.with_columns(macd_exprs)

        signal_exprs = [
            pl.col(f"ts_macd_{col}")
            .ewm_mean(span=signal_window, adjust=False)
            .over(group_col)
            .alias(f"ts_macd_signal_{col}")
            for col in cols
        ]
        df = df.with_columns(signal_exprs)

        hist_exprs = [
            (
                pl.col(f"ts_macd_{col}") - pl.col(f"ts_macd_signal_{col}")
            ).alias(f"ts_macd_hist_{col}")
            for col in cols
        ]
        df = df.with_columns(hist_exprs)

        drop_cols = [
            f"__macd_short_{col}" for col in cols
        ] + [f"__macd_long_{col}" for col in cols]
        return df.drop(drop_cols)

    @staticmethod
    def ts_bollinger_bands(
        df: pl.DataFrame,
        cols,
        window: int = 20,
        num_sd: int = 2,
        group_col: str = "stock_id",
    ):
        """(Bollinger Bands)"""
        cols = [cols] if isinstance(cols, str) else list(cols)
        if not cols:
            return df

        stats_exprs = []
        for col in cols:
            stats_exprs.extend(
                [
                    pl.col(col)
                    .rolling_mean(window)
                    .over(group_col)
                    .alias(f"__bb_ma_{col}"),
                    pl.col(col)
                    .rolling_std(window, ddof=1)
                    .over(group_col)
                    .alias(f"__bb_std_{col}"),
                ]
            )
        df = df.with_columns(stats_exprs)

        middle_exprs = []
        for col in cols:
            ma_col = pl.col(f"__bb_ma_{col}")
            std_col = pl.col(f"__bb_std_{col}")
            middle_exprs.extend(
                [
                    ma_col.alias(f"ts_bb_ma_{col}_{window}"),
                    (ma_col + std_col * num_sd).alias(f"__bb_upper_{col}"),
                    (ma_col - std_col * num_sd).alias(f"__bb_lower_{col}"),
                ]
            )
        df = df.with_columns(middle_exprs)

        band_exprs = []
        for col in cols:
            width = pl.col(f"__bb_upper_{col}") - pl.col(f"__bb_lower_{col}")
            band_exprs.extend(
                [
                    pl.col(f"__bb_upper_{col}").alias(f"ts_bb_upper_{col}_{window}"),
                    pl.col(f"__bb_lower_{col}").alias(f"ts_bb_lower_{col}_{window}"),
                    width.alias(f"ts_bb_width_{col}_{window}"),
                    pl.when(width != 0)
                    .then((pl.col(col) - pl.col(f"__bb_lower_{col}")) / width)
                    .otherwise(None)
                    .alias(f"ts_bb_pct_{col}_{window}"),
                ]
            )
        df = df.with_columns(band_exprs)

        drop_cols = (
            [f"__bb_ma_{col}" for col in cols]
            + [f"__bb_std_{col}" for col in cols]
            + [f"__bb_upper_{col}" for col in cols]
            + [f"__bb_lower_{col}" for col in cols]
        )
        return df.drop(drop_cols)
