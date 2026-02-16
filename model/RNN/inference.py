import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.interpolate import interp1d

import config as cfg
from dataset import (
    build_datasets_for_window,
    generate_rolling_windows,
    make_test_loader,
    read_data,
)
from model import create_model
from utils import ensure_dirs, get_logger, set_seed


def _maybe_denorm(t: Optional[torch.Tensor], label_scaler: Optional[Dict]):
    if label_scaler is None or t is None:
        return t

    # 检查标准化类型
    scaler_type = label_scaler.get("type", "zscore")

    if scaler_type == "rank_gauss":
        # Rank Gauss 反标准化
        sorted_values = label_scaler["sorted_values"]
        probabilities = label_scaler["probabilities"]

        # 创建插值函数：percentile -> value
        f = interp1d(
            probabilities,
            sorted_values,
            kind='linear',
            bounds_error=False,
            fill_value=(sorted_values[0], sorted_values[-1])
        )

        # standard normal -> percentile
        t_np = t.cpu().numpy().flatten()

        percentiles = stats.norm.cdf(t_np)

        # percentile -> value
        denormalized = f(percentiles)

        return torch.from_numpy(denormalized).reshape(t.shape)

    else:
        # Z-score 反标准化
        if label_scaler.get("std") is None or label_scaler.get("mean") is None:
            return t
        return t * label_scaler["std"] + label_scaler["mean"]


def load_model(model_path: Path, input_dim: int, state: Optional[dict] = None):
    if state is None:
        state = torch.load(model_path, map_location=cfg.DEVICE, weights_only=True)
    input_dim = state.get("input_dim", input_dim)

    # 优先使用保存的 model_type，向后兼容 use_attention
    model_type = state.get("model_type", None)
    if model_type is None:
        use_attention = state.get("use_attention", getattr(cfg, "USE_ATTENTION", False))
        model_type = "gru_attn" if use_attention else "gru"

    model = create_model(model_type, input_dim, cfg).to(cfg.DEVICE)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model


def predict(model, loader, dataset, label_scaler: Optional[Dict[str, float]] = None) -> Tuple[Optional[float], List[Dict]]:
    records: List[Dict] = []
    idx_list = list(dataset.get_index())
    offset = 0
    total_abs = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                x, y, mask = batch
            else:
                x, y = batch
                mask = torch.ones_like(y, dtype=torch.bool)
            if x.dim() == 4:
                b, d, t, f = x.shape
                x = x.view(b * d, t, f)
                y = y.view(-1)
                mask = mask.view(-1)
            else:
                mask = mask.view(-1)
            bs_valid = int(mask.sum().item())
            x = x.to(cfg.DEVICE)
            pred = model(x).cpu()
            batch_idx = idx_list[offset : offset + bs_valid]
            tgt = y.cpu() if y is not None else None
            mask_cpu = mask.cpu()
            pred_valid = pred[mask_cpu]
            tgt_valid = tgt[mask_cpu] if tgt is not None else None

            if tgt_valid is not None:
                pred_eval = _maybe_denorm(pred_valid, label_scaler)
                tgt_eval = _maybe_denorm(tgt_valid, label_scaler)
                total_abs += torch.sum(torch.abs(pred_eval.view(-1) - tgt_eval.view(-1))).item()
                total_count += tgt_eval.numel()
            else:
                pred_eval = _maybe_denorm(pred_valid, label_scaler)
                tgt_eval = None

            targets_iter = tgt_eval.view(-1) if tgt_eval is not None else [None] * bs_valid
            for meta, p, t in zip(batch_idx, pred_eval.view(-1), targets_iter):
                records.append(
                    {
                        "time_id": int(meta[0]),
                        "stock_id": int(meta[1]),
                        "pred": float(p.item()),
                        "target": float(t.item()) if t is not None else None,
                    }
                )
            offset += bs_valid
    mae_val = total_abs / max(total_count, 1) if total_count > 0 else None
    return mae_val, records


def resolve_last_window(features: pd.DataFrame) -> Dict:
    unique_times = np.array(sorted(features.index.get_level_values("time_id").unique()))
    windows = generate_rolling_windows(unique_times, cfg)
    if not windows:
        raise RuntimeError("时间跨度不足以构造推理窗口")
    return windows[-1]


def main(model_path: Optional[str] = None):
    set_seed(cfg.SEED)
    ensure_dirs(cfg.PRED_DIR, cfg.LOG_DIR)
    logger = get_logger(cfg.LOG_DIR / "inference.log")

    logger.info("=" * 60)
    logger.info(f"推理输出目录: {cfg.PRED_DIR}")
    logger.info("=" * 60)

    features, target = read_data(cfg)
    window = resolve_last_window(features)
    logger.info(
        f"推理窗口: train {window['train_start']}~{window['train_end']}, "
        f"val {window['val_start']}~{window['val_end']}, "
        f"test {window['test_start']}~{window['test_end']}"
    )

    if model_path is None:
        # 优先查找新格式: {model_type}_window_*.pth
        candidates = sorted(cfg.MODEL_DIR.glob("*_window_*.pth"))
        # 兼容旧格式: window_*.pth
        if not candidates:
            candidates = sorted(cfg.MODEL_DIR.glob("window_*.pth"))
        if not candidates:
            raise FileNotFoundError(
                f"未找到模型权重，请先运行 train.py。查找路径: {cfg.MODEL_DIR}"
            )
        model_path = candidates[-1]
    else:
        model_path = Path(model_path)

    state = torch.load(model_path, map_location=cfg.DEVICE, weights_only=True)

    # 显示加载的模型信息
    loaded_model_type = state.get("model_type", "unknown")
    logger.info(f"加载的模型类型: {loaded_model_type}")
    logger.info(f"模型权重: {model_path}")

    # 恢复 label_scaler（支持 Z-score 和 Rank Gauss）
    label_scaler = None
    scaler_type = state.get("label_scaler_type", "zscore")

    if scaler_type == "rank_gauss":
        # Rank Gauss scaler
        sorted_values = state.get("label_scaler_sorted_values")
        probabilities = state.get("label_scaler_probabilities")
        if sorted_values is not None and probabilities is not None:
            label_scaler = {
                "type": "rank_gauss",
                "sorted_values": sorted_values.numpy() if isinstance(sorted_values, torch.Tensor) else sorted_values,
                "probabilities": probabilities.numpy() if isinstance(probabilities, torch.Tensor) else probabilities,
            }
            logger.info("已加载 Rank Gauss label scaler")
    else:
        # Z-score scaler
        if state.get("label_mean") is not None and state.get("label_std") is not None:
            label_scaler = {
                "type": "zscore",
                "mean": float(state["label_mean"]),
                "std": float(state["label_std"])
            }
            logger.info(f"已加载 Z-score label scaler (mean={label_scaler['mean']:.6f}, std={label_scaler['std']:.6f})")

    _, _, test_ds, _, label_scaler = build_datasets_for_window(
        features, target, cfg, window, label_scaler=label_scaler
    )
    test_loader = make_test_loader(test_ds, cfg)

    input_dim = state.get("input_dim", test_ds.data_arr.shape[1])

    logger.info(f"加载模型: {model_path}")
    model = load_model(model_path, input_dim, state=state)

    test_mae, records = predict(model, test_loader, test_ds, label_scaler=label_scaler)
    if test_mae is not None:
        logger.info(f"Test MAE: {test_mae:.6f}")

    out_path = cfg.PRED_DIR / f"infer_{model_path.stem}.csv"
    pd.DataFrame.from_records(records).to_csv(out_path, index=False)
    logger.info(f"推理结果已保存至: {out_path}")


if __name__ == "__main__":
    cli_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(cli_path)
