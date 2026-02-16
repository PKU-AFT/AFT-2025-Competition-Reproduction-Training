from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

# =============================================================================
# Section 1: pandas/NumPy helpers
# =============================================================================
def _is_lexsorted(mi: pd.MultiIndex) -> bool:
    if hasattr(mi, "is_lexsorted"):
        return mi.is_lexsorted()
    return mi._lexsort_depth == mi.nlevels

def lazy_sort_index(df: pd.DataFrame, axis: int = 0) -> pd.DataFrame:
    idx = df.index if axis == 0 else df.columns
    if isinstance(idx, pd.MultiIndex) and (not idx.is_monotonic_increasing or not _is_lexsorted(idx)):
    # 同时满足MultiIndex，索引非单调递增，或者索引没有按字典序排列的情况下，继续排序
        return df.sort_index(axis=axis)
    return df

def np_ffill(arr: np.ndarray) -> np.ndarray:
    mask = np.isnan(arr.astype(float))
    idx = np.where(~mask, np.arange(mask.shape[0]), 0)
    np.maximum.accumulate(idx, out=idx)
    return arr[idx]

def np_ffill_2d(arr: np.ndarray) -> np.ndarray:
    mask = np.isnan(arr.astype(float))
    idx = np.where(~mask, np.arange(arr.shape[0])[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    return arr[idx, np.arange(arr.shape[1])]

def mad_zscore_ts(arr: np.ndarray, k: float = 5.0, eps: float = 1e-6) -> np.ndarray:
    if arr.size == 0:
        return arr
    median = np.median(arr, axis=-1, keepdims=True)
    mad = np.median(np.abs(arr - median), axis=-1, keepdims=True)
    mad = np.where(mad < eps, 1.0, mad)
    lower = median - k * mad
    upper = median + k * mad
    clipped = np.clip(arr, lower, upper)
    mean = clipped.mean(axis=-1, keepdims=True)
    std = clipped.std(axis=-1, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return (clipped - mean) / std

def mad_zscore_xs(arr: np.ndarray, valid_mask: Optional[np.ndarray] = None, k: float = 5.0, eps: float = 1e-6) -> np.ndarray:
    if arr.size == 0:
        return arr
    if valid_mask is not None:
        if valid_mask.shape == (arr.shape[1], arr.shape[0]):
            valid = valid_mask.T
        elif valid_mask.shape == (arr.shape[0], arr.shape[1]):
            valid = valid_mask
        else:
            raise ValueError("valid_mask shape mismatch")
        masked = np.where(valid[..., None], arr, np.nan)
        median = np.nanmedian(masked, axis=0, keepdims=True)
        mad = np.nanmedian(np.abs(masked - median), axis=0, keepdims=True)
        mad = np.where(mad < eps, 1.0, mad)
        lower = median - k * mad
        upper = median + k * mad
        clipped = np.clip(masked, lower, upper)
        mean = np.nanmean(clipped, axis=0, keepdims=True)
        std = np.nanstd(clipped, axis=0, keepdims=True)
        std = np.where(std < eps, 1.0, std)
        out = (clipped - mean) / std
        out = np.nan_to_num(out, nan=0.0)
        return out.astype(arr.dtype, copy=False)
    median = np.median(arr, axis=0, keepdims=True)
    mad = np.median(np.abs(arr - median), axis=0, keepdims=True)
    mad = np.where(mad < eps, 1.0, mad)
    lower = median - k * mad
    upper = median + k * mad
    clipped = np.clip(arr, lower, upper)
    mean = clipped.mean(axis=0, keepdims=True)
    std = clipped.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return ((clipped - mean) / std).astype(arr.dtype, copy=False)

def _get_time_ids_from_index(index: pd.MultiIndex) -> np.ndarray:
    if "time_id" in index.names:
        return index.get_level_values("time_id").to_numpy()
    return index.get_level_values(0).to_numpy()

def fit_section_med_scaler(target: pd.Series) -> Dict[str, object]:
    if isinstance(target, pd.DataFrame) and target.shape[1] == 1:
        target = target.iloc[:, 0]
    if not isinstance(target.index, pd.MultiIndex):
        raise ValueError("target index must be MultiIndex with time_id")
    time_ids = _get_time_ids_from_index(target.index)
    values = target.to_numpy(dtype=np.float64)
    valid = np.isfinite(values)
    if valid.any():
        global_median = float(np.nanmedian(values))
        global_mad = float(np.nanmedian(np.abs(values - global_median)))
    else:
        global_median = 0.0
        global_mad = 1.0
    if not np.isfinite(global_mad) or global_mad <= 0:
        global_mad = 1.0

    df = pd.DataFrame({"time_id": time_ids, "target": values})
    grouped = df.groupby("time_id")["target"]
    med = grouped.median()
    mad = grouped.apply(lambda x: np.nanmedian(np.abs(x - np.nanmedian(x))))
    mad = mad.replace(0.0, np.nan)
    return {
        "median": med.to_dict(),
        "mad": mad.to_dict(),
        "global": (global_median, global_mad),
    }

def apply_section_med_scaler(target: pd.Series, scaler: Dict[str, object]) -> pd.Series:
    if isinstance(target, pd.DataFrame) and target.shape[1] == 1:
        target = target.iloc[:, 0]
    if not isinstance(target.index, pd.MultiIndex):
        raise ValueError("target index must be MultiIndex with time_id")
    time_ids = _get_time_ids_from_index(target.index)
    med_map = scaler["median"]
    mad_map = scaler["mad"]
    global_median, global_mad = scaler["global"]
    med = pd.Series(time_ids).map(med_map).to_numpy()
    mad = pd.Series(time_ids).map(mad_map).to_numpy()
    med = np.where(np.isfinite(med), med, global_median)
    mad = np.where(np.isfinite(mad) & (mad > 0), mad, global_mad)
    values = target.to_numpy(dtype=np.float64)
    scaled = (values - med) / mad
    return pd.Series(scaled, index=target.index, name=target.name)

def invert_section_med_scaler(values: np.ndarray, time_ids: np.ndarray, scaler: Dict[str, object]) -> np.ndarray:
    med_map = scaler["median"]
    mad_map = scaler["mad"]
    global_median, global_mad = scaler["global"]
    time_ids = np.asarray(time_ids).reshape(-1)
    med = pd.Series(time_ids).map(med_map).to_numpy()
    mad = pd.Series(time_ids).map(mad_map).to_numpy()
    med = np.where(np.isfinite(med), med, global_median)
    mad = np.where(np.isfinite(mad) & (mad > 0), mad, global_mad)
    med = med.reshape(values.shape)
    mad = mad.reshape(values.shape)
    return values * mad + med

# =============================================================================
# Section 2: Core dataset
# =============================================================================
def build_panel_index(data: pd.DataFrame) -> pd.DataFrame:
    idx_df = pd.Series(range(data.shape[0]), index=data.index, dtype=object)
    idx_df = lazy_sort_index(idx_df.unstack())
    idx_df = lazy_sort_index(idx_df, axis=1)
    return idx_df

class PanelDataset(Dataset):
    """
    Panel Dataset that yields per-time slices with [N, T, F] windows.
    Optionally applies per-time cross-sectional MAD/Z-score over N.
    """

    def __init__(
        self,
        feature_df: pd.DataFrame,
        target: Optional[Union[pd.Series, pd.DataFrame]],
        start,
        end,
        step_len: int,
        fillna_type: str = "ffill+bfill",
        mask: Optional[Union[pd.Series, pd.DataFrame]] = None,
        feature_dtype=None,
        transform=None,
        target_transform=None,
        return_index: bool = False,
        return_mask: bool = True,
        apply_mad_zscore: bool = True,
        mad_k: float = 5.0,
        cache_data: bool = True,
    ):
        super().__init__()
        self.step_len = step_len
        self.fillna_type = fillna_type
        self.return_index = return_index
        self.return_mask = return_mask
        self.apply_mad_zscore = apply_mad_zscore
        self.mad_k = mad_k
        self.transform = transform
        self.target_transform = target_transform

        self.data = lazy_sort_index(feature_df)
        kwargs = {"object": self.data}
        if feature_dtype is not None:
            kwargs["dtype"] = feature_dtype
        self.data_arr = np.array(**kwargs)
        self.data_arr = np.append(
            self.data_arr,
            np.zeros((1, self.data_arr.shape[1]), dtype=self.data_arr.dtype),
            axis=0,
        )
        self.nan_idx = -1

        self.idx_df = build_panel_index(self.data)
        self.idx_arr = np.array(self.idx_df.values, dtype=np.float64)
        self.time_index = self.idx_df.index
        self.stock_index = self.idx_df.columns
        self.index_names = self.data.index.names
        self.data_index = self.data.index.copy()

        self.valid_arr = ~np.isnan(self.idx_arr)
        if mask is not None:
            if isinstance(mask, pd.DataFrame):
                assert len(mask.columns) == 1
                mask = mask.iloc[:, 0]
            row_mask = mask.reindex(self.data_index).fillna(False).to_numpy(dtype=bool)
            valid = np.zeros_like(self.idx_arr, dtype=bool)
            existing = self.valid_arr
            valid[existing] = row_mask[self.idx_arr[existing].astype(int)]
            self.valid_arr = valid
            self.idx_arr[~self.valid_arr] = np.nan

        self.start_pos, self.end_pos = self.time_index.slice_locs(start=start, end=end)
        self.panel_index = pd.MultiIndex.from_product(
            [self.time_index[self.start_pos : self.end_pos], self.stock_index],
            names=self.index_names,
        )

        if target is None:
            self.targets = None
        else:
            if isinstance(target, pd.DataFrame) and target.shape[1] == 1:
                target = target.iloc[:, 0]
            aligned = target.reindex(self.data_index)
            self.targets = aligned.to_numpy()

        del self.data
        self.cache_data = cache_data
        self._cache = {} if cache_data else None

    def get_index(self):
        return self.panel_index

    def get_time_index(self):
        return self.time_index[self.start_pos : self.end_pos]

    @property
    def empty(self):
        return len(self) == 0

    def _get_row(self, idx: Union[int, np.integer]) -> int:
        if not isinstance(idx, (int, np.integer)):
            raise NotImplementedError("Index type not supported")
        row = self.start_pos + int(idx)
        if not (self.start_pos <= row < self.end_pos):
            raise KeyError(f"{row} is out of [{self.start_pos}, {self.end_pos})")
        return row

    def _get_indices(self, row: int) -> Tuple[np.ndarray, np.ndarray]:
        indices = self.idx_arr[max(row - self.step_len + 1, 0) : row + 1, :]
        valid = ~np.isnan(indices)
        if indices.shape[0] < self.step_len:
            pad = np.full((self.step_len - indices.shape[0], indices.shape[1]), np.nan)
            indices = np.concatenate([pad, indices], axis=0)
            valid = np.concatenate([np.zeros_like(pad, dtype=bool), valid], axis=0)

        if self.fillna_type == "ffill":
            indices = np_ffill_2d(indices)
        elif self.fillna_type == "ffill+bfill":
            indices = np_ffill_2d(np_ffill_2d(indices)[::-1])[::-1]
        else:
            assert self.fillna_type == "none"
        return indices, valid

    def __len__(self):
        return self.end_pos - self.start_pos

    def __getitem__(self, idx: Union[int, np.integer]):
        row = self._get_row(idx)
        cache_key = int(idx)
        if self.cache_data:
            cached = self._cache.get(cache_key)
            if cached is not None:
                data, y_arr, valid = cached
                x = torch.from_numpy(data).float()
                if self.transform:
                    x = self.transform(x.clone())
                y = None
                if y_arr is not None:
                    y = torch.from_numpy(y_arr)
                    if self.target_transform:
                        y = self.target_transform(y.clone())

                if self.return_index:
                    time_val = self.time_index[row]
                    idx_pd = pd.MultiIndex.from_product([[time_val], self.stock_index], names=self.index_names)
                    if y is None:
                        return x, idx_pd
                    if self.return_mask:
                        return x, y, torch.from_numpy(valid), idx_pd
                    return x, y, idx_pd

                if y is None:
                    return x
                if self.return_mask:
                    return x, y, torch.from_numpy(valid)
                return x, y

        indices, valid_mask = self._get_indices(row)
        indices = np.nan_to_num(indices.astype(np.float64), nan=self.nan_idx).astype(int)
        data = self.data_arr[indices]
        data = np.transpose(data, (1, 0, 2))

        if self.apply_mad_zscore:
            data = mad_zscore_xs(data, valid_mask=valid_mask, k=self.mad_k)

        x = torch.from_numpy(data).float()

        row_indices = self.idx_arr[row, :]
        valid = ~np.isnan(row_indices)
        y = None
        y_arr = None
        if self.targets is not None:
            y_arr = np.zeros((row_indices.shape[0],), dtype=self.targets.dtype)
            if valid.any():
                y_arr[valid] = self.targets[row_indices[valid].astype(int)]
            y = torch.from_numpy(y_arr)

        if self.cache_data:
            self._cache[cache_key] = (data.astype(np.float32, copy=False), y_arr, valid)

        if self.transform:
            x = self.transform(x)
        if self.target_transform and y is not None:
            y = self.target_transform(y)

        if self.return_index:
            time_val = self.time_index[row]
            idx_pd = pd.MultiIndex.from_product([[time_val], self.stock_index], names=self.index_names)
            if y is None:
                return x, idx_pd
            if self.return_mask:
                return x, y, torch.from_numpy(valid), idx_pd
            return x, y, idx_pd

        if y is None:
            return x
        if self.return_mask:
            return x, y, torch.from_numpy(valid)
        return x, y

# =============================================================================
# Section 3: 数据读取
# =============================================================================
def read_data(cfg) -> Tuple[pd.DataFrame, pd.Series]:
    """
    ????????? target???? pandas MultiIndex ??? PanelDataset
    """
    def _load_one(path_like: Path) -> pd.DataFrame:
        df_single = pd.read_parquet(Path(path_like))
        df_single["time_id"] = df_single["date_id"] * 10000 + df_single["seconds_in_bucket"]
        df_single.set_index(["time_id", "stock_id"], inplace=True)
        df_single.drop(columns=["seconds_in_bucket", "date_id"], inplace=True)
        return df_single

    paths = cfg.DATA_PATH if isinstance(cfg.DATA_PATH, (list, tuple)) else [cfg.DATA_PATH]
    dfs = [_load_one(p) for p in paths]

    base = dfs[0].astype(np.float32)
    target = base.pop(cfg.TARGET_COLUMN).astype(np.float32)

    extra_feats = []
    for extra_df in dfs[1:]:
        extra_df = extra_df.astype(np.float32)
        if cfg.TARGET_COLUMN in extra_df.columns:
            extra_df = extra_df.drop(columns=[cfg.TARGET_COLUMN])
        extra_feats.append(extra_df)

    if extra_feats:
        # ?????????????????
        base = pd.concat([base] + extra_feats, axis=1, join="inner")
        target = target.reindex(base.index)

    # ?????????????
    mask = target.notna()
    target = target.loc[mask].astype(np.float32)
    features = base.loc[mask].fillna(0.0)

    features = features.sort_index()
    target = target.sort_index()
    return features, target

# =============================================================================
# Section 4: 窗口切分
# =============================================================================
def _subset_by_time(
    features: pd.DataFrame, target: pd.Series, start_time, end_time
) -> Tuple[pd.DataFrame, pd.Series]:
    # ? time_id ??????????
    mask = features.index.get_level_values("time_id")
    sel = (mask >= start_time) & (mask <= end_time)
    feat = features.loc[sel]
    tgt = target.loc[sel] if target is not None else None
    return feat, tgt

def generate_rolling_windows(times: np.ndarray, cfg) -> List[Dict]:
    """
    ?? time_id ???????train/val/test ?????????? test ?????
    """
    windows = []
    train_len = cfg.TRAIN_DAYS * cfg.BUCKETS_PER_DAY
    val_len = cfg.VAL_DAYS * cfg.BUCKETS_PER_DAY
    test_len = cfg.TEST_DAYS * cfg.BUCKETS_PER_DAY
    total = train_len + val_len + test_len
    step = test_len  # ???????? test ????
    for start in range(0, len(times) - total + 1, step):
        train_start = times[start]
        train_end = times[start + train_len - 21]
        val_start = times[start + train_len]
        val_end = times[start + train_len + val_len - 21]
        test_start = times[start + train_len + val_len]
        test_end = times[start + total - 1]
        windows.append(
            {
                "train_start": train_start,
                "train_end": train_end,
                "val_start": val_start,
                "val_end": val_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )
    return windows

# =============================================================================
# Section 5: ?????
# =============================================================================
def build_datasets_for_window(
    features: pd.DataFrame,
    target: pd.Series,
    cfg,
    window: Dict,
    return_index: bool = False,
) -> Tuple[PanelDataset, PanelDataset, PanelDataset]:
    train_feat, train_tgt = _subset_by_time(features, target, window["train_start"], window["train_end"])
    val_feat, val_tgt = _subset_by_time(features, target, window["val_start"], window["val_end"])
    test_feat, test_tgt = _subset_by_time(features, target, window["test_start"], window["test_end"])

    if train_feat.empty or val_feat.empty or test_feat.empty:
        raise ValueError("????????????????????????????")

    train_ds = PanelDataset(
        feature_df=train_feat,
        target=train_tgt,
        start=window["train_start"],
        end=window["train_end"],
        step_len=cfg.STEP_LEN,
        fillna_type=cfg.FILLNA_TYPE,
        return_index=return_index,
    )
    val_ds = PanelDataset(
        feature_df=val_feat,
        target=val_tgt,
        start=window["val_start"],
        end=window["val_end"],
        step_len=cfg.STEP_LEN,
        fillna_type=cfg.FILLNA_TYPE,
        return_index=return_index,
    )
    test_ds = PanelDataset(
        feature_df=test_feat,
        target=test_tgt,
        start=window["test_start"],
        end=window["test_end"],
        step_len=cfg.STEP_LEN,
        fillna_type=cfg.FILLNA_TYPE,
        return_index=return_index,
    )
    return train_ds, val_ds, test_ds

# =============================================================================
# Section 6: DataLoader 构建
# =============================================================================
def _panel_collate(batch):
    if not batch:
        return batch
    first = batch[0]
    if isinstance(first, (list, tuple)) and len(first) > 0:
        last = first[-1]
        if isinstance(last, pd.MultiIndex):
            transposed = list(zip(*batch))
            collated = [default_collate(items) for items in transposed[:-1]]
            collated.append(list(transposed[-1]))
            return tuple(collated)
    return default_collate(batch)

def make_panel_loader(dataset, cfg, shuffle: bool, batch_size: int = None) -> DataLoader:
    group_bs = batch_size if batch_size is not None else getattr(cfg, "GROUP_BATCH_SIZE", 1)
    return DataLoader(
        dataset,
        batch_size=group_bs,
        shuffle=shuffle,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        collate_fn=_panel_collate,
    )

def make_loader(dataset, cfg, shuffle: bool, batch_size: int = None) -> DataLoader:
    return make_panel_loader(dataset, cfg, shuffle=shuffle, batch_size=batch_size)

def make_test_loader(dataset, cfg) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        collate_fn=_panel_collate,
    )

if __name__ == '__main__':
    import config as cfg

    import time

    t_all = time.perf_counter()
    print("Loading data via read_data(cfg)...")
    t0 = time.perf_counter()
    features, target = read_data(cfg)
    print(f"[time] read_data: {time.perf_counter() - t0:.4f}s")

    t0 = time.perf_counter()
    print("################### manual window ###################")

    print(features.shape)
    print(target.shape)

    import numpy as np
    import pandas as pd
    import torch

    # Pick a time_id and stock_id
    time_ids_sorted = features.index.get_level_values("time_id").unique().sort_values()
    stock_ids_sorted = features.index.get_level_values("stock_id").unique().sort_values()
    print(time_ids_sorted, stock_ids_sorted)
    t_sel = time_ids_sorted[9]
    s_sel = stock_ids_sorted[0]
    s_pos = stock_ids_sorted.get_loc(s_sel)

    # Build a naive window for this single stock
    pos = np.searchsorted(time_ids_sorted, t_sel)
    window_times = time_ids_sorted[max(0, pos - (cfg.STEP_LEN - 1)) : pos + 1]

    # Reindex window and ffill/bfill along time
    idx = pd.MultiIndex.from_product([window_times, [s_sel]], names=["time_id", "stock_id"])
    window_df = features.reindex(idx).sort_index()
    window_df = window_df.ffill().bfill()  # matches fillna_type="ffill+bfill"
    print("window_df raw values:")
    for row_idx, row in window_df.iterrows():
        print(f"{row_idx}: {row.to_numpy()}")

    # Build a naive cross-sectional window for all stocks
    window_idx_all = pd.MultiIndex.from_product(
        [window_times, stock_ids_sorted], names=["time_id", "stock_id"]
    )
    window_df_all = features.reindex(window_idx_all).sort_index()
    valid_mask = ~window_df_all.isna().all(axis=1)
    valid_mask = valid_mask.to_numpy().reshape(len(window_times), len(stock_ids_sorted))
    window_df_all = window_df_all.groupby(level="stock_id").ffill().bfill()
    window_df_all = window_df_all.unstack("stock_id")
    window_df_all = window_df_all.swaplevel(0, 1, axis=1)
    col_index = pd.MultiIndex.from_product(
        [stock_ids_sorted, features.columns], names=["stock_id", "feature"]
    )
    window_df_all = window_df_all.reindex(columns=col_index)
    x_all = window_df_all.to_numpy(dtype=np.float32)
    x_all = x_all.reshape(len(window_times), len(stock_ids_sorted), len(features.columns))
    x_all = np.transpose(x_all, (1, 0, 2))  # [N, T, F]
    x_all_raw = x_all.copy()
    x_all = mad_zscore_xs(x_all, valid_mask=valid_mask, k=5.0)
    t_pos = np.searchsorted(window_times, t_sel)
    f_sel = features.columns[0]
    f_pos = features.columns.get_loc(f_sel)
    raw_cs = x_all_raw[:, t_pos, f_pos]
    norm_cs = x_all[:, t_pos, f_pos]
    print(f"xs debug: time_id={t_sel}, feature={f_sel}, N={len(stock_ids_sorted)}")
    print(f"xs raw[s_sel@{s_pos}]={raw_cs[s_pos]} -> norm={norm_cs[s_pos]}")
    print("xs valid at s_sel:", bool(valid_mask[t_pos, s_pos]))
    print("xs raw head:", raw_cs[:5])
    print("xs norm head:", norm_cs[:5])
    x_naive = x_all[s_pos]
    print(f"naive window (xs mad_zscore): time_id={t_sel}, stock_id={s_sel}, shape={x_naive.shape}")
    print(x_naive)
    print(f"[time] manual window: {time.perf_counter() - t0:.4f}s")

    t0 = time.perf_counter()
    print("################### compare Naive vs PanelDataset ###################")

    time_ids = features.index.get_level_values("time_id")
    ds = PanelDataset(
        feature_df=features,
        target=target,
        start=time_ids.min(),
        end=time_ids.max(),
        step_len=cfg.STEP_LEN,
        fillna_type=cfg.FILLNA_TYPE,
        return_mask=True,
    )

    time_index = ds.get_time_index()
    pos_ds = time_index.get_loc(t_sel)
    x_ds, y_ds, mask_ds = ds[pos_ds]
    s_pos = ds.stock_index.get_loc(s_sel)
    x_ds_sel = x_ds[s_pos]
    y_ds_sel = y_ds[s_pos]
    print(f"dataset window: time_id={t_sel}, stock_id={s_sel}, shape={x_ds_sel.shape}")
    print(x_ds_sel)
    print("Naive vs Dataset match:", torch.allclose(torch.from_numpy(x_naive), x_ds_sel))

    key = (t_sel, s_sel)
    if key in target.index:
        y_naive = torch.as_tensor(target.loc[key])
        print("Target match:", torch.allclose(y_naive, y_ds_sel))
    else:
        print("Target missing at selected key; mask:", bool(mask_ds[s_pos]))
    print(f"[time] naive vs panel: {time.perf_counter() - t0:.4f}s")

    t0 = time.perf_counter()
    print("################### compare Dataset vs DataLoader ###################")
    batch_size = 2
    loader = make_panel_loader(ds, cfg, shuffle=False, batch_size=batch_size)
    batch_id = pos_ds // batch_size
    offset = pos_ds % batch_size
    for b_id, batch in enumerate(loader):
        if b_id != batch_id:
            continue
        x_b, y_b, mask_b = batch
        print(f"batch_id={b_id} shapes: x={x_b.shape}, y={y_b.shape}, mask={mask_b.shape}")
        print("Batch sample matches dataset:", torch.allclose(x_b[offset], x_ds))
        print("Batch target matches dataset:", torch.allclose(y_b[offset], y_ds))
        print("Batch mask matches dataset:", torch.equal(mask_b[offset], mask_ds))
        break
    print(f"[time] dataloader compare: {time.perf_counter() - t0:.4f}s")
    print(f"[time] total: {time.perf_counter() - t_all:.4f}s")

