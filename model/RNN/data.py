import bisect
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler, DataLoader


def _is_lexsorted(mi: pd.MultiIndex) -> bool:
    if hasattr(mi, "is_lexsorted"):
        return mi.is_lexsorted()
    return mi._lexsort_depth == mi.nlevels


def lazy_sort_index(df: pd.DataFrame, axis: int = 0) -> pd.DataFrame:
    idx = df.index if axis == 0 else df.columns
    if isinstance(idx, pd.MultiIndex) and (not idx.is_monotonic_increasing or not _is_lexsorted(idx)):
        return df.sort_index(axis=axis)
    return df


def np_ffill(arr: np.ndarray) -> np.ndarray:
    mask = np.isnan(arr.astype(float))
    idx = np.where(~mask, np.arange(mask.shape[0]), 0)
    np.maximum.accumulate(idx, out=idx)
    return arr[idx]


class TSDataset(Dataset):
    """
    Standalone Dataset for panel tabular time series.
    Produces sliding windows of length step_len and corresponding targets.
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
    ):
        super().__init__()
        self.step_len = step_len
        self.fillna_type = fillna_type
        self.return_index = return_index
        self.transform = transform
        self.target_transform = target_transform

        self.data = lazy_sort_index(feature_df)
        kwargs = {"object": self.data}
        if feature_dtype is not None:
            kwargs["dtype"] = feature_dtype
        self.data_arr = np.array(**kwargs)
        # 用零做序列填充，避免缺失步长产生 NaN
        self.data_arr = np.append(self.data_arr, np.zeros((1, self.data_arr.shape[1]), dtype=self.data_arr.dtype), axis=0)
        self.nan_idx = -1

        self.idx_df, self.idx_map = self.build_index(self.data)
        self.data_index = self.data.index.copy()

        if mask is not None:
            if isinstance(mask, pd.DataFrame):
                assert len(mask.columns) == 1
                mask = mask.iloc[:, 0]
            mask = mask.reindex(self.data_index).fillna(False).astype(np.bool_)
            self.mask_arr = mask.values
            self.idx_map = self.flt_idx_map(self.mask_arr, self.idx_map)
            self.data_index = self.data_index[np.where(self.mask_arr)[0]]
        self.idx_map = self.idx_map2arr(self.idx_map)

        self.start_idx, self.end_idx = self.data_index.slice_locs(start=start, end=end)
        self.idx_arr = np.array(self.idx_df.values, dtype=np.float64)

        if target is None:
            self.targets = None
        else:
            if isinstance(target, pd.DataFrame) and target.shape[1] == 1:
                target = target.iloc[:, 0]
            aligned = target.reindex(self.data_index)
            self.targets = aligned.to_numpy()

        del self.data

    @staticmethod
    def idx_map2arr(idx_map):
        dtype = np.int64
        no_existing_idx = (np.iinfo(dtype).max, np.iinfo(dtype).max)
        max_idx = max(idx_map.keys())
        arr_map = []
        for i in range(max_idx + 1):
            arr_map.append(idx_map.get(i, no_existing_idx))
        return np.array(arr_map, dtype=dtype)

    @staticmethod
    def flt_idx_map(flt_data, idx_map):
        idx = 0
        new_idx_map = {}
        for i, exist in enumerate(flt_data):
            if exist:
                new_idx_map[idx] = idx_map[i]
                idx += 1
        return new_idx_map

    @staticmethod
    def build_index(data: pd.DataFrame):
        idx_df = pd.Series(range(data.shape[0]), index=data.index, dtype=object)
        idx_df = lazy_sort_index(idx_df.unstack())
        idx_df = lazy_sort_index(idx_df, axis=1)

        idx_map = {}
        for i, (_, row) in enumerate(idx_df.iterrows()):
            for j, real_idx in enumerate(row):
                if not np.isnan(real_idx):
                    idx_map[real_idx] = (i, j)
        return idx_df, idx_map

    def get_index(self):
        return self.data_index[self.start_idx : self.end_idx]

    @property
    def empty(self):
        return len(self) == 0

    def _get_row_col(self, idx):
        if isinstance(idx, (int, np.integer)):
            real_idx = self.start_idx + idx
            if self.start_idx <= real_idx < self.end_idx:
                i, j = self.idx_map[real_idx]
            else:
                raise KeyError(f"{real_idx} is out of [{self.start_idx}, {self.end_idx})")
        elif isinstance(idx, tuple):
            date, inst = idx
            date = pd.Timestamp(date)
            i = bisect.bisect_right(self.idx_df.index, date) - 1
            j = bisect.bisect_left(self.idx_df.columns, inst)
        else:
            raise NotImplementedError("Index type not supported")
        return i, j

    def _get_indices(self, row: int, col: int) -> np.ndarray:
        indices = self.idx_arr[max(row - self.step_len + 1, 0) : row + 1, col]
        if len(indices) < self.step_len:
            indices = np.concatenate([np.full((self.step_len - len(indices),), np.nan), indices])

        if self.fillna_type == "ffill":
            indices = np_ffill(indices)
        elif self.fillna_type == "ffill+bfill":
            indices = np_ffill(np_ffill(indices)[::-1])[::-1]
        else:
            assert self.fillna_type == "none"
        return indices

    def _fetch_target(self, idx: Union[int, Sequence[int]]):
        if self.targets is None:
            return None
        if isinstance(idx, (list, np.ndarray, tuple)):
            idx_arr = np.asarray(idx, dtype=int)
            loc = self.start_idx + idx_arr
            return self.targets[loc]
        loc = self.start_idx + int(idx)
        return self.targets[loc]

    def __len__(self):
        return self.end_idx - self.start_idx

    def __getitem__(self, idx: Union[int, Sequence[int]]):
        multi = (list, np.ndarray)
        if isinstance(idx, multi):
            indices = [self._get_indices(*self._get_row_col(i)) for i in idx]
            indices = np.concatenate(indices)
        else:
            indices = self._get_indices(*self._get_row_col(idx))

        indices = np.nan_to_num(indices.astype(np.float64), nan=self.nan_idx).astype(int)
        data = self.data_arr[indices]
        if isinstance(idx, multi):
            data = data.reshape(-1, self.step_len, *data.shape[1:])

        x = torch.from_numpy(data).float()
        y = self._fetch_target(idx)
        if y is not None:
            y = torch.as_tensor(y)

        if self.transform:
            x = self.transform(x)
        if self.target_transform and y is not None:
            y = self.target_transform(y)

        if self.return_index:
            idx_arr = np.atleast_1d(np.asarray(idx if isinstance(idx, (list, np.ndarray)) else [idx]))
            indices_pd = self.data_index[self.start_idx + idx_arr]
            if y is None:
                return x, indices_pd
            return x, y, indices_pd

        if y is None:
            return x
        return x, y


class DailyBatchSamplerRandom(Sampler):
    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        # calculate number of samples in each batch
        self.daily_count = pd.Series(index=self.data_source.get_index()).groupby("time_id").size().values
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)  # calculate begin index of each batch
        self.daily_index[0] = 0

    def __iter__(self):
        if self.shuffle:
            index = np.arange(len(self.daily_count))
            np.random.shuffle(index)
            for i in index:
                yield np.arange(self.daily_index[i], self.daily_index[i] + self.daily_count[i])
        else:
            for idx, count in zip(self.daily_index, self.daily_count):
                yield np.arange(idx, idx + count)

    def __len__(self):
        return len(self.data_source)
