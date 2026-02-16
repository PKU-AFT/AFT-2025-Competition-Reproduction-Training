import logging
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from lossfuncs import RegressionLoss



def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return RegressionLoss.mae(pred, target)


def ensure_dirs(*paths: Iterable[Path]) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def get_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("runner")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def _format_float_tag(value: float) -> str:
    if value == 0:
        return "0"
    sci = f"{value:.6e}"
    mantissa, exp = sci.split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    sign = "-" if exp.startswith("-") else ""
    exp_num = exp[1:] if exp[0] in "+-" else exp
    exp_num = exp_num.lstrip("0") or "0"
    return f"{mantissa}e{sign}{exp_num}"


def build_run_tag(args, model_name: str = "MASTER") -> str:
    lr_tag = _format_float_tag(args.lr)
    wd_tag = _format_float_tag(args.weight_decay)
    return f"{model_name}_lr{lr_tag}_bs{args.group_batch_size}_wd{wd_tag}"


def build_diversify_run_tag(args) -> str:
    lr_tag = _format_float_tag(args.lr)
    wd_tag = _format_float_tag(args.weight_decay)
    return (
        f"diversify_lr{lr_tag}_bs{args.group_batch_size}_wd{wd_tag}_"
        f"ld{args.latent_domain_num}"
    )


def _compute_decay_from_records(
    records: List[Dict], window_days: Optional[int] = 40, segment_days: int = 10
) -> Tuple[Optional[float], List[float]]:
    if not records:
        return None, []
    df = pd.DataFrame.from_records(records)
    if df.empty or df["target"].isna().all():
        return None, []
    df["date_id"] = df["time_id"] // 10000
    df["abs_err"] = (df["pred"] - df["target"]).abs()
    daily_mae = df.groupby("date_id")["abs_err"].mean().sort_index()
    if window_days is not None and len(daily_mae) > window_days:
        daily_mae = daily_mae.iloc[:window_days]
    seg_maes: List[float] = []
    for start in range(0, len(daily_mae), segment_days):
        seg_maes.append(float(daily_mae.iloc[start : start + segment_days].mean()))
    decay = None
    if len(seg_maes) >= 2:
        decay = float(seg_maes[0] - seg_maes[-1])
    return decay, seg_maes


def _assign_decay_weights(decays: List[Optional[float]]) -> List[float]:
    if not decays:
        return []
    valid = [i for i, d in enumerate(decays) if d is not None and not np.isnan(d)]
    if not valid:
        return [1.0 / len(decays)] * len(decays)

    decay_vals = np.array([decays[i] for i in valid], dtype=float)
    if len(decay_vals) == 1:
        weights = [0.0] * len(decays)
        weights[valid[0]] = 1.0
        return weights

    d_min = float(decay_vals.min())
    d_max = float(decay_vals.max())
    if np.isclose(d_min, d_max):
        weights = [0.0] * len(decays)
        share = 1.0 / len(valid)
        for idx in valid:
            weights[idx] = share
        return weights

    normalized = (decay_vals - d_min) / (d_max - d_min)
    scores = 1.0 - normalized
    total = float(scores.sum())
    if total <= 0:
        weights = [0.0] * len(decays)
        share = 1.0 / len(valid)
        for idx in valid:
            weights[idx] = share
        return weights

    weights = [0.0] * len(decays)
    for idx, score in zip(valid, scores):
        weights[idx] = float(score / total)
    return weights



