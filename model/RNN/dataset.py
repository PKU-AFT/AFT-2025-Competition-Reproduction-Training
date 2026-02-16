import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.interpolate import interp1d

from data import TSDataset, DailyBatchSamplerRandom


def _fit_label_scaler(train_tgt: pd.Series) -> Dict[str, float]:
    mean = float(train_tgt.mean())
    std = float(train_tgt.std(ddof=0))
    std = std if std > 1e-8 else 1.0
    return {"mean": mean, "std": std}


def _apply_label_scaler(series: pd.Series, scaler: Dict[str, float]) -> pd.Series:
    return (series - scaler["mean"]) / scaler["std"]


# =============================================================================
# Rank Gauss 标准化
# =============================================================================

def _fit_rank_gauss_scaler(train_tgt: pd.Series) -> Dict:
    """
    Rank Gauss 标准化：将任意分布转换为标准正态分布

    原理：
    1. 计算每个样本的秩（rank）
    2. 将秩映射到 (0, 1) 均匀分布
    3. 通过逆 CDF 映射到标准正态分布

    优点：
    - 完全消除原始分布形态影响
    - 对极端值鲁棒
    - 适合尖峰、肥尾分布
    - 保持样本间的相对顺序

    Args:
        train_tgt: 训练集标签

    Returns:
        scaler_info: 包含映射信息的字典，用于转换 val/test
    """
    n = len(train_tgt)

    # 1. 计算秩（平均法处理并列值）
    ranks = train_tgt.rank(method='average')

    # 2. 转换为 (0, 1) 均匀分布，避免无穷大
    # 使用 (ranks - 0.5) / n 确保映射到 (0, 1) 开区间
    uniform = (ranks - 0.5) / n

    # 3. 转换为标准正态分布
    normalized = stats.norm.ppf(uniform.values)

    # 4. 保存训练集的分位数映射（用于 val/test）
    # 存储排序后的原始值和对应的累积概率
    sorted_values = np.sort(train_tgt.values)
    # 对应的累积概率 P(X <= x)
    probabilities = np.linspace(0.5 / n, 1 - 0.5 / n, n)

    scaler_info = {
        "type": "rank_gauss",
        "sorted_values": sorted_values,
        "probabilities": probabilities,
        # 保存训练集的基本统计（用于可能的边界处理）
        "train_min": float(train_tgt.min()),
        "train_max": float(train_tgt.max()),
    }

    return scaler_info


def _apply_rank_gauss_scaler(series: pd.Series, scaler_info: Dict) -> pd.Series:
    """
    使用训练集的 Rank Gauss 映射转换新数据

    处理超出训练集范围的值：
    - 小于训练集最小值 → 映射到 -3（约 0.13% 分位数）
    - 大于训练集最大值 → 映射到 +3（约 99.87% 分位数）

    Args:
        series: 要转换的数据（val/test）
        scaler_info: _fit_rank_gauss_scaler 返回的 scaler 信息

    Returns:
        转换后的标准正态分布数据
    """
    sorted_values = scaler_info["sorted_values"]
    probabilities = scaler_info["probabilities"]

    # 创建插值函数：value -> percentile
    # bounds_error=False 允许超出范围，使用 fill_value 外推
    f = interp1d(
        sorted_values,
        probabilities,
        kind='linear',
        bounds_error=False,
        fill_value=(0.001, 0.999)  # 超出训练集范围时映射到边界概率
    )

    # value -> percentile
    percentiles = f(series.values)

    # 处理可能的 NaN（如果输入有 NaN）
    percentiles = np.nan_to_num(percentiles, nan=0.5)

    # 裁剪到有效范围，避免 ppf 产生无穷大
    percentiles = np.clip(percentiles, 0.001, 0.999)

    # percentile -> standard normal
    normalized = stats.norm.ppf(percentiles)

    return pd.Series(normalized, index=series.index)


def _inverse_rank_gauss_scaler(normalized: pd.Series, scaler_info: Dict) -> pd.Series:
    """
    将 Rank Gauss 标准化的数据反变换回原始尺度

    Args:
        normalized: 标准正态分布的数据（模型预测值）
        scaler_info: _fit_rank_gauss_scaler 返回的 scaler 信息

    Returns:
        反变换后的原始尺度数据
    """
    sorted_values = scaler_info["sorted_values"]
    probabilities = scaler_info["probabilities"]

    # 创建插值函数：percentile -> value
    f = interp1d(
        probabilities,
        sorted_values,
        kind='linear',
        bounds_error=False,
        fill_value=(sorted_values[0], sorted_values[-1])
    )

    # standard normal -> percentile
    percentiles = stats.norm.cdf(normalized.values)

    # percentile -> value
    denormalized = f(percentiles)

    return pd.Series(denormalized, index=normalized.index)


# =============================================================================


def read_data(cfg) -> Tuple[pd.DataFrame, pd.Series]:
    """
    读取原始特征并拆分 target，输出为 pandas MultiIndex 以适配 data.py 的 TSDataset。
    """
    path = Path(cfg.DATA_PATH)

    df = pd.read_parquet(path)
    df["time_id"] = df["date_id"] * 10000 + df["seconds_in_bucket"]
    df.set_index(["time_id", "stock_id"], inplace=True)
    df.drop(columns=["seconds_in_bucket", "date_id"], inplace=True)
    df = df.astype(np.float32)
    target = df.pop(cfg.TARGET_COLUMN).astype(np.float32)

    # # 删掉缺失标签并对特征做零填充，避免后续张量包含 NaN
    mask = target.notna()
    target = target.loc[mask]
    features = df.loc[mask].fillna(0.0)
    # features = df

    features = features.sort_index()
    target = target.sort_index()
    return features, target


def _winsorize_train_only(
    train_tgt: pd.Series,
    lower: float = 0.01,
    upper: float = 0.99,
) -> Tuple[pd.Series, Dict[str, float]]:
    """
    只对训练集标签进行缩尾处理（Winsorization）

    只裁剪训练集，验证集和测试集保持原样。
    这样评估时看到的是模型在真实数据分布上的表现。

    Args:
        train_tgt: 训练集标签
        lower: 下分位数（默认 0.01）
        upper: 上分位数（默认 0.99）

    Returns:
        (train_winsorized, bounds_dict)
        bounds_dict 包含 {"lower": 下界值, "upper": 上界值}
    """
    # 只用训练集计算分位数边界
    lower_bound = train_tgt.quantile(lower)
    upper_bound = train_tgt.quantile(upper)

    bounds = {"lower": float(lower_bound), "upper": float(upper_bound)}

    # 只裁剪训练集
    train_winsorized = train_tgt.clip(lower_bound, upper_bound)

    return train_winsorized, bounds


def _subset_by_time(
    features: pd.DataFrame, target: pd.Series, start_time, end_time
) -> Tuple[pd.DataFrame, pd.Series]:
    mask = features.index.get_level_values("time_id")
    sel = (mask >= start_time) & (mask <= end_time)
    feat = features.loc[sel]
    tgt = target.loc[sel] if target is not None else None
    return feat, tgt


def generate_rolling_windows(times: np.ndarray, cfg) -> List[Dict]:
    """
    Generate rolling windows based on *trading days* (date_id), not per time_id bucket.
    Assumes time_id = date_id * 10000 + seconds_in_bucket, so integer division by 10000 recovers date_id.

    支持 FINAL_TEST_DAYS 模式：预留最后 N 天作为独立测试集，前面用于滚动窗口训练/验证。
    """
    time_ids = np.asarray(times)
    date_ids = time_ids // 10000

    # Build per-day min/max time_id so we can translate day windows back to exact time_id boundaries.
    day_bounds: Dict[int, Tuple[int, int]] = {}
    prev_date = None
    day_start = None
    for t, d in zip(time_ids, date_ids):
        if d != prev_date:
            if prev_date is not None:
                day_bounds[prev_date] = (day_start, prev_time)
            day_start = t
            prev_date = d
        prev_time = t
    if prev_date is not None:
        day_bounds[prev_date] = (day_start, prev_time)

    unique_dates = np.array(sorted(day_bounds.keys()))

    # 检查是否启用独立最终测试集模式
    final_test_days = getattr(cfg, "FINAL_TEST_DAYS", None)
    use_final_test = final_test_days is not None and final_test_days > 0

    windows = []

    if use_final_test:
        # 独立最终测试集模式：预留最后 N 天
        if len(unique_dates) < cfg.TRAIN_DAYS + cfg.VAL_DAYS + final_test_days:
            return []  # 数据不足

        # 计算滚动窗口的有效日期范围（排除最终测试集）
        rolling_end_idx = len(unique_dates) - final_test_days

        # 滚动步长 = VAL_DAYS（或配置的 STEP_DAYS），确保验证集不重叠
        step = getattr(cfg, "STEP_DAYS", cfg.VAL_DAYS)

        for start in range(0, rolling_end_idx - cfg.TRAIN_DAYS - cfg.VAL_DAYS + 1, step):
            train_start_date = unique_dates[start]
            train_end_date = unique_dates[start + cfg.TRAIN_DAYS - 1]
            val_start_date = unique_dates[start + cfg.TRAIN_DAYS]
            val_end_date = unique_dates[start + cfg.TRAIN_DAYS + cfg.VAL_DAYS - 1]

            # 测试集使用固定的最后 N 天
            test_start_date = unique_dates[-final_test_days]
            test_end_date = unique_dates[-1]

            windows.append(
                {
                    "train_start": day_bounds[train_start_date][0],
                    "train_end": day_bounds[train_end_date][1],
                    "val_start": day_bounds[val_start_date][0],
                    "val_end": day_bounds[val_end_date][1],
                    "test_start": day_bounds[test_start_date][0],
                    "test_end": day_bounds[test_end_date][1],
                }
            )
    else:
        # 原有滚动窗口模式（每个窗口有自己的 test set）
        total_days = cfg.TRAIN_DAYS + cfg.VAL_DAYS + cfg.TEST_DAYS
        step = cfg.TEST_DAYS
        for start in range(0, len(unique_dates) - total_days + 1, step):
            train_start_date = unique_dates[start]
            train_end_date = unique_dates[start + cfg.TRAIN_DAYS - 1]
            val_start_date = unique_dates[start + cfg.TRAIN_DAYS]
            val_end_date = unique_dates[start + cfg.TRAIN_DAYS + cfg.VAL_DAYS - 1]
            test_start_date = unique_dates[start + cfg.TRAIN_DAYS + cfg.VAL_DAYS]
            test_end_date = unique_dates[start + total_days - 1]

            windows.append(
                {
                    "train_start": day_bounds[train_start_date][0],
                    "train_end": day_bounds[train_end_date][1],
                    "val_start": day_bounds[val_start_date][0],
                    "val_end": day_bounds[val_end_date][1],
                    "test_start": day_bounds[test_start_date][0],
                    "test_end": day_bounds[test_end_date][1],
                }
            )
    return windows


def build_datasets_for_window(
    features: pd.DataFrame,
    target: pd.Series,
    cfg,
    window: Dict,
    label_scaler: Optional[Dict[str, float]] = None,
) -> Tuple[TSDataset, TSDataset, TSDataset, Optional[StandardScaler], Optional[Dict[str, float]]]:
    """
    构建滚动窗口的 train/val/test 数据集，并进行标准化。

    Returns:
        train_ds, val_ds, test_ds, feature_scaler, label_scaler
        scaler 可用于 inference 时对新数据做相同的标准化
    """
    train_feat, train_tgt = _subset_by_time(features, target, window["train_start"], window["train_end"])
    val_feat, val_tgt = _subset_by_time(features, target, window["val_start"], window["val_end"])
    test_feat, test_tgt = _subset_by_time(features, target, window["test_start"], window["test_end"])

    if train_feat.empty or val_feat.empty or test_feat.empty:
        raise ValueError("存在空的数据切片，请检查时间窗口或原始数据是否覆盖该范围")

    # ========== label 缩尾处理（Winsorization）==========
    # 只对训练集进行缩尾处理，验证集和测试集保持原样
    # 这样评估时看到的是模型在真实数据分布上的表现
    winsorize_bounds: Optional[Dict[str, float]] = None
    if getattr(cfg, "LABEL_WINSORIZE", False):
        lower_q = getattr(cfg, "LABEL_WINSORIZE_LOWER", 0.01)
        upper_q = getattr(cfg, "LABEL_WINSORIZE_UPPER", 0.99)
        train_tgt, winsorize_bounds = _winsorize_train_only(
            train_tgt, lower=lower_q, upper=upper_q
        )
    # ==================================================

    # ========== 滚动窗口全局标准化 ==========
    # 只用 train 数据 fit，防止数据泄露
    scaler = StandardScaler()
    scaler.fit(train_feat.values)

    # 标准化 train/val/test
    train_feat_scaled = pd.DataFrame(
        scaler.transform(train_feat.values),
        index=train_feat.index,
        columns=train_feat.columns
    )
    val_feat_scaled = pd.DataFrame(
        scaler.transform(val_feat.values),
        index=val_feat.index,
        columns=val_feat.columns
    )
    test_feat_scaled = pd.DataFrame(
        scaler.transform(test_feat.values),
        index=test_feat.index,
        columns=test_feat.columns
    )
    # ==========================================

    # ========== label 标准化 ==========
    label_scaler_out: Optional[Dict] = None

    # 获取标准化方法类型
    standardize_method = getattr(cfg, "LABEL_STANDARDIZE_METHOD", "zscore").lower()

    if getattr(cfg, "LABEL_STANDARDIZE", False) or label_scaler is not None:
        if standardize_method == "rank_gauss":
            # Rank Gauss 标准化（适合尖峰肥尾分布）
            # 注意：Rank Gauss 会自动处理极端值，不需要 Winsorize
            label_scaler_out = label_scaler or _fit_rank_gauss_scaler(train_tgt)
            train_tgt = _apply_rank_gauss_scaler(train_tgt, label_scaler_out)
            val_tgt = _apply_rank_gauss_scaler(val_tgt, label_scaler_out)
            test_tgt = _apply_rank_gauss_scaler(test_tgt, label_scaler_out)
        else:
            # 默认 Z-score 标准化
            label_scaler_out = label_scaler or _fit_label_scaler(train_tgt)
            train_tgt = _apply_label_scaler(train_tgt, label_scaler_out)
            val_tgt = _apply_label_scaler(val_tgt, label_scaler_out)
            test_tgt = _apply_label_scaler(test_tgt, label_scaler_out)

    # 合并 winsorize_bounds 到 label_scaler_out，方便统一传递
    if winsorize_bounds is not None and label_scaler_out is not None:
        label_scaler_out["winsorize_bounds"] = winsorize_bounds
    elif winsorize_bounds is not None:
        # 只有 winsorize，没有标准化
        label_scaler_out = {"winsorize_bounds": winsorize_bounds, "mean": None, "std": None, "type": "winsorize"}
    # ==========================================

    train_ds = TSDataset(
        feature_df=train_feat_scaled,
        target=train_tgt,
        start=window["train_start"],
        end=window["train_end"],
        step_len=cfg.STEP_LEN,
        fillna_type=cfg.FILLNA_TYPE,
    )
    val_ds = TSDataset(
        feature_df=val_feat_scaled,
        target=val_tgt,
        start=window["val_start"],
        end=window["val_end"],
        step_len=cfg.STEP_LEN,
        fillna_type=cfg.FILLNA_TYPE,
    )
    test_ds = TSDataset(
        feature_df=test_feat_scaled,
        target=test_tgt,
        start=window["test_start"],
        end=window["test_end"],
        step_len=cfg.STEP_LEN,
        fillna_type=cfg.FILLNA_TYPE,
    )
    return train_ds, val_ds, test_ds, scaler, label_scaler_out


def make_loader(dataset, cfg, shuffle: bool) -> DataLoader:
    daily_sampler = DailyBatchSamplerRandom(dataset, shuffle=shuffle)
    batch_days = getattr(cfg, "BATCH_DAYS", 1)
    if batch_days > 1:
        sampler = GroupedDailyBatchSampler(daily_sampler, batch_days=batch_days, drop_last=False)
    else:
        sampler = daily_sampler
    max_daily = int(daily_sampler.daily_count.max())
    collate_fn = partial(collate_pad_to_max, max_daily=max_daily)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        collate_fn=collate_fn,
        persistent_workers=cfg.NUM_WORKERS > 0,
        prefetch_factor=getattr(cfg, "PREFETCH_FACTOR", 2) if cfg.NUM_WORKERS > 0 else None,
    )
    return loader


def make_test_loader(dataset, cfg) -> DataLoader:
    daily_sampler = DailyBatchSamplerRandom(dataset, shuffle=False)
    batch_days = getattr(cfg, "BATCH_DAYS", 1)
    if batch_days > 1:
        sampler = GroupedDailyBatchSampler(daily_sampler, batch_days=batch_days, drop_last=False)
    else:
        sampler = daily_sampler
    max_daily = int(daily_sampler.daily_count.max())
    collate_fn = partial(collate_pad_to_max, max_daily=max_daily)
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        collate_fn=collate_fn,
        persistent_workers=cfg.NUM_WORKERS > 0,
        prefetch_factor=getattr(cfg, "PREFETCH_FACTOR", 2) if cfg.NUM_WORKERS > 0 else None,
    )


def _pad_day_batch(x_day: torch.Tensor, y_day: torch.Tensor, max_daily: int, pad_val: float):
    """
    Pad one day's samples to max_daily and build a day mask.
    x_day: [day_len, T, F], y_day: [day_len]
    returns: x [max_daily, T, F], y [max_daily], mask [max_daily]
    """
    day_len = x_day.size(0)
    # mask 初始化为 max_daily 长度，前 day_len 个为 True
    mask = torch.zeros(max_daily, dtype=torch.bool)
    mask[:day_len] = True

    if day_len < max_daily:
        pad_n = max_daily - day_len
        x_day = F.pad(x_day, (0, 0, 0, 0, 0, pad_n), value=pad_val)
        y_day = F.pad(y_day, (0, pad_n), value=pad_val)
        # 注意: mask 已经是 max_daily 长度，不需要再 pad
    elif day_len > max_daily:
        # 截断（防御性处理，理论上不应该发生）
        x_day = x_day[:max_daily]
        y_day = y_day[:max_daily]
        mask[:] = True

    return x_day, y_day, mask


def collate_pad_to_max(batch, max_daily: int, pad_val: float = 0.0):
    """
    优化版 collate 函数 - 使用预分配 tensor 减少内存操作

    Pad samples of the same time_id to max_daily and stack along a day dimension.
    - If the batch_sampler is not grouped (BATCH_DAYS=1), batch is per-sample within one day: [T, F]/scalar.
    - If grouped (BATCH_DAYS>1), batch is a list of days, each [day_len, T, F]/[day_len].
    Output: x [batch_days, max_daily, T, F], y [batch_days, max_daily], mask [batch_days, max_daily].
    """
    x0, y0 = batch[0]

    # Single-day, per-sample: stack into one day then pad
    if x0.dim() == 2:
        xs, ys = zip(*batch)  # per-sample list
        x_stack = torch.stack(xs)  # [day_len, T, F]
        y_stack = torch.stack(ys)  # [day_len]

        batch_days = 1
        day_len = x_stack.size(0)
        T, F = x_stack.shape[1], x_stack.shape[2]

        # 预分配输出 tensor (优化：一次性分配，避免多次 stack)
        x_out = torch.full((batch_days, max_daily, T, F), pad_val, dtype=x_stack.dtype)
        y_out = torch.full((batch_days, max_daily), pad_val, dtype=y_stack.dtype)
        mask_out = torch.zeros((batch_days, max_daily), dtype=torch.bool)

        # 填充有效数据
        x_out[0, :day_len] = x_stack
        y_out[0, :day_len] = y_stack
        mask_out[0, :day_len] = True

        return x_out, y_out, mask_out

    # Grouped days: each element is [day_len, T, F]
    # 优化：预分配输出 tensor，避免 list append + torch.stack
    batch_days = len(batch)
    T, F = batch[0][0].shape[1], batch[0][0].shape[2]

    # 预分配输出 tensor
    x_out = torch.full((batch_days, max_daily, T, F), pad_val, dtype=torch.float32)
    y_out = torch.full((batch_days, max_daily), pad_val, dtype=torch.float32)
    mask_out = torch.zeros((batch_days, max_daily), dtype=torch.bool)

    # 填充每天的数据
    for i, (x_day, y_day) in enumerate(batch):
        day_len = x_day.size(0)
        x_out[i, :day_len] = x_day
        y_out[i, :day_len] = y_day
        mask_out[i, :day_len] = True

    return x_out, y_out, mask_out

class GroupedDailyBatchSampler:

    """
    Wrap DailyBatchSamplerRandom to group multiple days into one batch.
    Yields a list of per-day index arrays of length batch_days (last batch may be shorter if not drop_last).
    """

    def __init__(self, daily_sampler, batch_days: int, drop_last: bool = False):
        assert batch_days >= 1
        self.daily_sampler = daily_sampler
        self.batch_days = batch_days
        self.drop_last = drop_last
        self.daily_count = daily_sampler.daily_count

    def __iter__(self):
        bucket = []
        for day_idx in self.daily_sampler:
            bucket.append(day_idx)
            if len(bucket) == self.batch_days:
                yield bucket
                bucket = []
        if bucket and not self.drop_last:
            yield bucket

    def __len__(self):
        n = len(self.daily_sampler)
        if self.drop_last:
            return n // self.batch_days
        return (n + self.batch_days - 1) // self.batch_days
