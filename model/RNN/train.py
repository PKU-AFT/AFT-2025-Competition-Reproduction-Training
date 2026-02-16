import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from scipy import stats
from scipy.interpolate import interp1d

import config as cfg
from dataset import (
    build_datasets_for_window,
    generate_rolling_windows,
    make_loader,
    make_test_loader,
    read_data,
)
from model import create_model, list_models
from utils import ensure_dirs, get_logger, set_seed
from losses import create_loss, get_loss_info


# ============ 多GPU支持 ============

def setup_distributed():
    """初始化分布式训练环境"""
    # 检查是否在分布式环境中（通过环境变量判断）
    # 只有通过 torchrun/python -m torch.distributed.launch 启动时才会设置这些变量
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        world_size = int(os.environ["WORLD_SIZE"])

        # 设置当前设备
        torch.cuda.set_device(local_rank)

        # 初始化进程组
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        return rank, local_rank, world_size
    # 单卡模式或非分布式启动
    return 0, 0, 1


def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    """判断是否为主进程"""
    return rank == 0


# ============ 原有函数 ============

def _resolve_model_type():
    """解析实际使用的模型类型"""
    if getattr(cfg, "USE_ATTENTION", False):
        return "gru_attn"
    return getattr(cfg, "MODEL_TYPE", "gru")


def _create_loss_fn():
    """
    创建损失函数实例

    根据配置中的 LOSS_TYPE 和相关参数创建损失函数
    """
    loss_type = getattr(cfg, "LOSS_TYPE", "mae")

    # 根据损失函数类型准备参数
    loss_params = {}

    if loss_type in ("huber", "smooth_l1"):
        loss_params["delta"] = getattr(cfg, "HUBER_DELTA", 1.0)
    elif loss_type in ("quantile", "pinball"):
        loss_params["quantile"] = getattr(cfg, "QUANTILE", 0.5)
    elif loss_type == "focal_mae":
        loss_params["gamma"] = getattr(cfg, "FOCAL_GAMMA", 2.0)
    elif loss_type == "weighted_mae":
        weight_type = getattr(cfg, "WEIGHTED_MAE_TYPE", "magnitude")
        loss_params["weight_type"] = weight_type
        loss_params["weight_params"] = {}

        if weight_type == "error":
            loss_params["weight_params"]["alpha"] = getattr(cfg, "WEIGHTED_MAE_ALPHA", 1.0)
        elif weight_type == "threshold":
            loss_params["weight_params"]["threshold"] = getattr(cfg, "WEIGHTED_MAE_THRESHOLD", 5.0)
            loss_params["weight_params"]["high_weight"] = getattr(cfg, "WEIGHTED_MAE_HIGH_WEIGHT", 2.0)
            loss_params["weight_params"]["low_weight"] = getattr(cfg, "WEIGHTED_MAE_LOW_WEIGHT", 1.0)
        elif weight_type == "smooth_threshold":
            loss_params["weight_params"]["threshold"] = getattr(cfg, "WEIGHTED_MAE_THRESHOLD", 5.0)
            loss_params["weight_params"]["max_weight"] = getattr(cfg, "WEIGHTED_MAE_MAX_WEIGHT", 3.0)
            loss_params["weight_params"]["smooth_width"] = getattr(cfg, "WEIGHTED_MAE_SMOOTH_WIDTH", 2.0)
    elif loss_type == "adaptive_mae":
        loss_params["warmup_epochs"] = getattr(cfg, "ADAPTIVE_WARMUP_EPOCHS", 5)
        loss_params["max_gamma"] = getattr(cfg, "ADAPTIVE_MAX_GAMMA", 3.0)

    return create_loss(loss_type, **loss_params)


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    grad_clip_norm: float,
    loss_fn=None,
    amp_scaler: Optional[GradScaler] = None,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    use_amp = amp_scaler is not None

    # 如果没有提供损失函数，使用默认MAE（向后兼容）
    if loss_fn is None:
        loss_fn = _create_loss_fn()

    for batch in loader:
        if len(batch) == 3:
            x, y, mask = batch
        else:
            x, y = batch
            mask = torch.ones_like(y, dtype=torch.bool)
        if x.dim() == 4:  # [B, max_daily, T, F] -> flatten batch/day
            b, d, t, f = x.shape
            x = x.view(b * d, t, f)
            y = y.view(-1)
            mask = mask.view(-1)
        # align lengths defensively (should already match)
        min_len = min(x.shape[0], y.shape[0], mask.numel())
        if y.shape[0] != min_len or mask.numel() != min_len:
            y = y[:min_len]
            mask = mask.view(-1)[:min_len]
        x = x.to(device)
        y = y.to(device).view(-1)
        mask = mask.to(device).view(-1)

        optimizer.zero_grad()

        # 混合精度前向传播
        with autocast(device_type='cuda' if use_amp else 'cpu', enabled=use_amp):
            pred = model(x)
            # 使用模块化的损失函数
            loss = loss_fn(pred, y, mask)

        # 反向传播和优化
        if use_amp:
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            if grad_clip_norm and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            amp_scaler.step(optimizer)
            amp_scaler.update()
        else:
            loss.backward()
            if grad_clip_norm and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        # 累积损失（使用MAE作为标准指标，便于比较）
        with torch.no_grad():
            mae_loss = torch.abs(pred - y)
            if mask is not None:
                mask_f = mask.float()
                mae_loss = mae_loss * mask_f
                denom = mask_f.sum().clamp_min(1.0)
            else:
                denom = pred.numel()
            total_loss += mae_loss.sum().item()
            total_count += denom.item()

    return total_loss / max(total_count, 1)


def _maybe_denorm(t: torch.Tensor, label_scaler):
    if label_scaler is None:
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
        if isinstance(t, torch.Tensor):
            t_np = t.cpu().numpy().flatten()
        else:
            t_np = np.array(t).flatten()

        percentiles = stats.norm.cdf(t_np)

        # percentile -> value
        denormalized = f(percentiles)

        return torch.from_numpy(denormalized).to(t.device).reshape(t.shape) if isinstance(t, torch.Tensor) else denormalized

    else:
        # Z-score 反标准化
        if label_scaler.get("std") is None or label_scaler.get("mean") is None:
            return t
        return t * label_scaler["std"] + label_scaler["mean"]


def evaluate(model, loader, device, label_scaler=None) -> float:
    model.eval()
    total_abs = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                x, y, mask = batch
            else:
                x, y = batch
                mask = torch.ones_like(y, dtype=torch.bool)
            if x.dim() == 4:  # [B, max_daily, T, F]
                b, d, t, f = x.shape
                x = x.view(b * d, t, f)
                y = y.view(-1)
                mask = mask.view(-1)
            min_len = min(x.shape[0], y.shape[0], mask.numel())
            if y.shape[0] != min_len or mask.numel() != min_len:
                y = y[:min_len]
                mask = mask.view(-1)[:min_len]
            x = x.to(device)
            y = y.to(device).view(-1)
            mask = mask.to(device).view(-1)
            pred = model(x)
            mask_f = mask.float()
            pred_eval = _maybe_denorm(pred, label_scaler)
            y_eval = _maybe_denorm(y, label_scaler)
            total_abs += torch.sum(torch.abs(pred_eval - y_eval) * mask_f).item()
            total_count += mask_f.sum().item()
    return total_abs / max(total_count, 1)


def predict(model, loader, dataset, device, label_scaler=None) -> Tuple[float, List[Dict]]:
    model.eval()
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
            min_len = min(x.shape[0], y.shape[0], mask.numel())
            if y.shape[0] != min_len or mask.numel() != min_len:
                y = y[:min_len]
                mask = mask.view(-1)[:min_len]
            else:
                mask = mask.view(-1)
            bs_valid = int(mask.sum().item())
            x = x.to(device)
            pred = model(x).cpu()
            y_cpu = y.cpu() if y is not None else None
            mask_cpu = mask.cpu()
            pred_valid = pred[mask_cpu]
            y_valid = y_cpu[mask_cpu] if y_cpu is not None else None
            pred_eval = _maybe_denorm(pred_valid, label_scaler)
            y_eval = _maybe_denorm(y_valid, label_scaler) if y_valid is not None else None
            batch_idx = idx_list[offset : offset + bs_valid]

            if y_valid is not None:
                total_abs += torch.sum(torch.abs(pred_eval.view(-1) - y_eval.view(-1))).item()
                total_count += y_eval.numel()

            targets_iter = y_eval.view(-1) if y_eval is not None else [None] * bs_valid
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


def build_scheduler(optimizer):
    if cfg.LR_SCHEDULER == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg.LR_FACTOR,
            patience=cfg.LR_PATIENCE,
            min_lr=cfg.LR_MIN,
        )
    if cfg.LR_SCHEDULER == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=cfg.LR_T0, T_mult=cfg.LR_T_MULT, eta_min=cfg.LR_MIN
        )
    return None


def get_current_lr(optimizer) -> float:
    if not optimizer.param_groups:
        return cfg.LEARNING_RATE
    return optimizer.param_groups[0].get("lr", cfg.LEARNING_RATE)


def main():
    # 初始化分布式训练
    rank, local_rank, world_size = setup_distributed()
    is_main = is_main_process(rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    set_seed(cfg.SEED + rank)  # 每个进程使用不同的种子

    if is_main:
        ensure_dirs(cfg.MODEL_DIR, cfg.PRED_DIR, cfg.LOG_DIR)
        logger = get_logger(cfg.LOG_DIR / "train.log")
    else:
        # 非主进程禁用日志
        import logging
        logger = logging.getLogger()
        logger.addHandler(logging.NullHandler())

    # 显示当前模型和输出路径
    model_type = _resolve_model_type()
    if is_main:
        logger.info("=" * 60)
        logger.info(f"模型类型: {model_type}")
        logger.info(f"使用 GPU: {world_size} 张")
        logger.info(f"输出目录配置:")
        logger.info(f"  MODEL_DIR: {cfg.MODEL_DIR}")
        logger.info(f"  PRED_DIR:  {cfg.PRED_DIR}")
        logger.info(f"  LOG_DIR:   {cfg.LOG_DIR}")
        logger.info("=" * 60)

    if is_main:
        logger.info("加载数据...")
    features, target = read_data(cfg)
    unique_times = np.array(sorted(features.index.get_level_values("time_id").unique()))
    windows = generate_rolling_windows(unique_times, cfg)
    if not windows:
        if is_main:
            logger.error("时间跨度不足以生成滚动窗口，请检查数据或调整 TRAIN/VAL/TEST 天数")
        return

    if is_main:
        logger.info(f"生成 {len(windows)} 个滚动窗口")

    for win_id, window in enumerate(windows):
        if is_main:
            logger.info(
                f"[Window {win_id}] "
                f"train: {window['train_start']}~{window['train_end']}, "
                f"val: {window['val_start']}~{window['val_end']}, "
                f"test: {window['test_start']}~{window['test_end']}"
            )

        try:
            train_ds, val_ds, test_ds, scaler, label_scaler = build_datasets_for_window(
                features, target, cfg, window
            )
        except ValueError as e:
            if is_main:
                logger.warning(f"跳过窗口 {win_id}: {e}")
            continue

        # 多GPU: 使用 Subset 让每个进程处理不同的数据子集
        if world_size > 1:
            # 创建 DistributedSampler 获取当前 rank 的索引
            temp_sampler = DistributedSampler(
                train_ds,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True
            )
            # 获取当前 rank 对应的索引
            indices = list(temp_sampler)
            # 创建 Subset，保持原有的 make_loader 逻辑
            from torch.utils.data import Subset
            train_ds_subset = Subset(train_ds, indices)
            train_loader = make_loader(train_ds_subset, cfg, shuffle=True)
        else:
            train_loader = make_loader(train_ds, cfg, shuffle=True)

        val_loader = make_loader(val_ds, cfg, shuffle=False)
        test_loader = make_test_loader(test_ds, cfg)

        input_dim = train_ds.data_arr.shape[1]

        # 创建模型并包装 DDP
        model = create_model(model_type, input_dim, cfg).to(device)
        if world_size > 1:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)

        if is_main:
            # 获取实际模型参数量（去除 DDP wrapper）
            actual_model = model.module if hasattr(model, 'module') else model
            logger.info(f"[Window {win_id}] 模型: {model_type}, 参数量: {actual_model.get_num_params():,}")

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
        scheduler = build_scheduler(optimizer)

        if is_main:
            writer = SummaryWriter(log_dir=str(cfg.LOG_DIR / f"window_{win_id}"))

        # 初始化混合精度训练器（仅 CUDA 设备且 USE_AMP=True 时启用）
        use_amp = getattr(cfg, "USE_AMP", False) and torch.cuda.is_available()
        amp_scaler = GradScaler(device='cuda') if use_amp else None
        if is_main and use_amp:
            logger.info(f"[Window {win_id}] 混合精度训练 (AMP) 已启用")

        best_val = float("inf")
        best_epoch = 0
        best_path = cfg.MODEL_DIR / f"{model_type}_window_{win_id}.pth"

        # 创建损失函数
        loss_fn = _create_loss_fn()
        if is_main:
            loss_info = get_loss_info(cfg.LOSS_TYPE)
            logger.info(f"[Window {win_id}] 损失函数: {loss_info.get('name', cfg.LOSS_TYPE)}")

        # 检查是否需要设置epoch（用于AdaptiveMAELoss）
        use_adaptive_loss = hasattr(loss_fn, 'set_epoch')

        for epoch in range(1, cfg.NUM_EPOCHS + 1):
            # 设置 epoch 用于 DistributedSampler
            if world_size > 1 and hasattr(train_loader.batch_sampler, 'set_epoch'):
                train_loader.batch_sampler.set_epoch(epoch)

            # 对于自适应损失函数，设置当前epoch
            if use_adaptive_loss:
                loss_fn.set_epoch(epoch)

            train_loss = train_one_epoch(model, train_loader, optimizer, device, cfg.GRAD_CLIP_NORM, loss_fn, amp_scaler)

            # 验证只在主进程进行
            if is_main:
                val_mae = evaluate(model, val_loader, device, label_scaler=label_scaler)

                writer.add_scalar("train/mae", train_loss, epoch)
                writer.add_scalar("val/mae", val_mae, epoch)
                current_lr = get_current_lr(optimizer)
                writer.add_scalar("train/lr", current_lr, epoch)

                logger.info(
                    f"[Window {win_id}] Epoch {epoch} | train_mae={train_loss:.6f} | "
                    f"val_mae={val_mae:.6f} | lr={current_lr:.6e}"
                )

                if scheduler is not None:
                    if cfg.LR_SCHEDULER == "plateau":
                        scheduler.step(val_mae)
                    else:
                        scheduler.step()

                if val_mae + cfg.EARLY_STOP_MIN_DELTA < best_val:
                    best_val = val_mae
                    best_epoch = epoch
                    # 保存模型参数和 scaler（用于 inference）
                    # 获取原始模型（去除 DDP wrapper）
                    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                    checkpoint = {
                        "model_state": model_state,
                        "model_type": model_type,
                        "input_dim": input_dim,
                        "use_attention": cfg.USE_ATTENTION,
                        "scaler_mean": torch.from_numpy(scaler.mean_),
                        "scaler_scale": torch.from_numpy(scaler.scale_),
                        "best_val_mae": float(best_val),
                    }

                    # 保存 label_scaler 信息
                    if label_scaler is not None:
                        scaler_type = label_scaler.get("type", "zscore")
                        checkpoint["label_scaler_type"] = scaler_type

                        if scaler_type == "rank_gauss":
                            checkpoint["label_scaler_sorted_values"] = torch.from_numpy(label_scaler["sorted_values"])
                            checkpoint["label_scaler_probabilities"] = torch.from_numpy(label_scaler["probabilities"])
                            checkpoint["label_mean"] = None
                            checkpoint["label_std"] = None
                        else:
                            checkpoint["label_mean"] = label_scaler.get("mean")
                            checkpoint["label_std"] = label_scaler.get("std")
                    else:
                        checkpoint["label_mean"] = None
                        checkpoint["label_std"] = None

                    torch.save(checkpoint, best_path)

                if epoch - best_epoch >= cfg.EARLY_STOP_PATIENCE:
                    logger.info(
                        f"[Window {win_id}] Early stopping at epoch {epoch} "
                        f"(best val_mae={best_val:.6f} at epoch {best_epoch})"
                    )
                    break

        # 测试与预测保存（仅主进程）
        if is_main:
            if best_path.exists():
                state = torch.load(best_path, map_location=device, weights_only=True)
                model_to_load = model.module if hasattr(model, 'module') else model
                model_to_load.load_state_dict(state["model_state"])

            test_mae, records = predict(model, test_loader, test_ds, device, label_scaler=label_scaler)
            writer.add_scalar("test/mae", test_mae if test_mae is not None else float("nan"), 0)
            writer.flush()
            writer.close()

            logger.info(f"[Window {win_id}] test_mae={test_mae:.6f}" if test_mae is not None else "无 test 标签，未计算 MAE")

            pred_path = cfg.PRED_DIR / f"{model_type}_window_{win_id}_preds.csv"
            pd.DataFrame.from_records(records).to_csv(pred_path, index=False)
            logger.info(f"已保存预测到 {pred_path}")

        # 同步所有进程
        if world_size > 1:
            dist.barrier()

        # 清理显存
        del model, optimizer, train_loader, val_loader, test_loader
        del train_ds, val_ds, test_ds
        if amp_scaler is not None:
            del amp_scaler
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    # 清理分布式环境
    cleanup_distributed()


if __name__ == "__main__":
    main()
