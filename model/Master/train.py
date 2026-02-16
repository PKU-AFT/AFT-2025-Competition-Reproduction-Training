import os
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

import config as cfg
from datatools import (
    build_datasets_for_window,
    generate_rolling_windows,
    make_loader,
    make_test_loader,
    PanelDataset,
    read_data,
    fit_section_med_scaler,
    apply_section_med_scaler,
    invert_section_med_scaler,
)
from model.MASTER import MASTER
from utils import _assign_decay_weights, _compute_decay_from_records, build_run_tag, ensure_dirs, get_logger, set_seed


# 目录
def prepare_run_dirs(run_name: str):
    model_dir = cfg.MODEL_DIR / run_name
    pred_dir = cfg.PRED_DIR / run_name
    log_dir = cfg.LOG_DIR / run_name
    ensure_dirs(model_dir, pred_dir, log_dir)
    return model_dir, pred_dir, log_dir


# 将优化器状态移动到指定device
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=cfg.NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=cfg.LEARNING_RATE)
    parser.add_argument("--weight_decay", type=float, default=cfg.WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    parser.add_argument("--group_batch_size", type=int, default=cfg.GROUP_BATCH_SIZE)
    parser.add_argument("--master_d_model", type=int, default=256)
    parser.add_argument("--master_t_nhead", type=int, default=4)
    parser.add_argument("--master_s_nhead", type=int, default=2)
    parser.add_argument("--master_t_dropout", type=float, default=0.5)
    parser.add_argument("--master_s_dropout", type=float, default=0.5)
    parser.add_argument("--master_gate_feat_dim", type=int, default=100)
    parser.add_argument("--master_gate_beta", type=float, default=1.0)
    parser.add_argument("--use_section_med_norm", type=bool, default=False)
    parser.add_argument(
        "--resume", action="store_true", help="Resume basical_train from the last saved checkpoint"
    )
    parser.add_argument(
        "--resume_path",
        type=str,
        default=None,
        help="Optional checkpoint path for basical_train resume (defaults to models/basical_train/basic_split_last_<run_tag>.pth)",
    )
    return parser.parse_args()


# Trainer
class TrainRunner:
    def __init__(self, args, device, logger):
        self.args = args
        self.device = device
        self.logger = logger
        self.section_scaler = None

    def set_section_scaler(self, scaler: Dict[str, object]) -> None:
        self.section_scaler = scaler

    @staticmethod
    def _extract_time_ids(index) -> Optional[np.ndarray]:
        if index is None:
            return None
        if isinstance(index, pd.MultiIndex):
            return index.get_level_values("time_id").to_numpy()
        if isinstance(index, (list, tuple)):
            time_list = [idx.get_level_values("time_id").to_numpy() for idx in index]
            return np.stack(time_list, axis=0)
        return None

    def _inverse_scale(self, tensor: torch.Tensor, index) -> torch.Tensor:
        if self.section_scaler is None or index is None:
            return tensor
        time_ids = self._extract_time_ids(index)
        if time_ids is None:
            return tensor
        values = tensor.detach().cpu().numpy()
        time_ids = np.asarray(time_ids)
        if time_ids.shape != values.shape:
            time_ids = np.reshape(time_ids, values.shape)
        inv = invert_section_med_scaler(values, time_ids, self.section_scaler)
        return torch.from_numpy(inv).to(tensor.device).type_as(tensor)

    @staticmethod
    def _move_optimizer_state_to_device(optimizer, device):
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    def train_one_epoch(self, model, loader, optimizer) -> float:
        model.train()
        total_count = 0
        total_abs_raw = 0.0
        for batch in loader:
            if len(batch) == 4:
                x, y, mask, index = batch
            elif len(batch) == 3:
                x, y, mask = batch
                index = None
            else:
                x, y = batch
                mask = None
                index = None
            x = x.to(self.device)
            y = y.to(self.device)
            if mask is not None:
                mask = mask.to(self.device)

            x_group = x.unsqueeze(0) if x.dim() == 3 else x
            y_group = y.unsqueeze(0) if y.dim() == 1 else y
            mask_group = mask.unsqueeze(0) if mask is not None and mask.dim() == 2 and x.dim() == 3 else mask

            optimizer.zero_grad()
            pred = model(x_group)
            if pred.dim() == 1:
                pred = pred.unsqueeze(0)
            abs_err = torch.abs(pred - y_group)
            if mask_group is not None:
                valid_per_group = mask_group.sum(dim=1).clamp(min=1)
                per_group = (abs_err * mask_group).sum(dim=1) / valid_per_group
            else:
                per_group = abs_err.mean(dim=1)
            loss = per_group.mean()
            loss.backward()
            optimizer.step()

            pred_raw = pred.detach()
            y_raw = y_group
            raw_abs = torch.abs(pred_raw - y_raw)
            if mask_group is not None:
                valid_per_group = mask_group.sum(dim=1).clamp(min=1)
                group_mae_raw = (raw_abs * mask_group).sum(dim=1) / valid_per_group
            else:
                group_mae_raw = raw_abs.mean(dim=1)
            total_abs_raw += group_mae_raw.sum().item()
            total_count += group_mae_raw.numel()
        return total_abs_raw / max(total_count, 1)

    def evaluate(self, model, loader) -> float:
        model.eval()
        total_abs = 0.0
        total_count = 0
        with torch.no_grad():
            for batch in loader:
                if len(batch) == 3:
                    x, y, mask = batch
                else:
                    x, y = batch
                    mask = None
                x = x.to(self.device)
                y = y.to(self.device)
                if mask is not None:
                    mask = mask.to(self.device)

                x_group = x.unsqueeze(0) if x.dim() == 3 else x
                y_group = y.unsqueeze(0) if y.dim() == 1 else y
                mask_group = mask.unsqueeze(0) if mask is not None and mask.dim() == 2 and x.dim() == 3 else mask

                pred = model(x_group)
                if pred.dim() == 1:
                    pred = pred.unsqueeze(0)
                pred_raw = pred
                y_raw = y_group
                raw_abs = torch.abs(pred_raw - y_raw)
                if mask_group is not None:
                    valid_per_group = mask_group.sum(dim=1).clamp(min=1)
                    group_mae = (raw_abs * mask_group).sum(dim=1) / valid_per_group
                else:
                    group_mae = raw_abs.mean(dim=1)
                total_abs += group_mae.sum().item()
                total_count += group_mae.numel()
        return total_abs / max(total_count, 1)

    def predict(self, model, loader, dataset) -> Tuple[Optional[float], List[Dict]]:
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
                    mask = None
                x = x.to(self.device)
                y_cpu = y.cpu() if y is not None else None
                mask_cpu = mask.cpu() if mask is not None else None

                x_group = x.unsqueeze(0) if x.dim() == 3 else x
                pred = model(x_group).cpu()
                if pred.dim() == 1:
                    pred = pred.unsqueeze(0)

                pred_raw = pred
                y_raw = y_cpu.view_as(pred) if y_cpu is not None else None

                flat_pred = pred_raw.view(-1)
                if y_raw is not None:
                    if mask_cpu is not None:
                        mask_view = mask_cpu.view_as(pred_raw)
                        valid_per_group = mask_view.sum(dim=1).clamp(min=1)
                        group_mae = (torch.abs(pred_raw - y_raw) * mask_view).sum(dim=1) / valid_per_group
                    else:
                        group_mae = torch.abs(pred_raw - y_raw).mean(dim=1)
                    total_abs += group_mae.sum().item()
                    total_count += group_mae.numel()

                if y_raw is not None:
                    flat_y = y_raw.view(-1)
                    if mask_cpu is not None:
                        flat_mask = mask_cpu.view(-1)
                        flat_y = torch.where(flat_mask, flat_y, torch.tensor(float("nan")))
                else:
                    flat_y = None

                batch_idx = idx_list[offset : offset + flat_pred.numel()]
                for meta, p, t in zip(
                    batch_idx,
                    flat_pred,
                    flat_y if flat_y is not None else [torch.tensor(float("nan"))] * flat_pred.numel(),
                ):
                    records.append(
                        {
                            "time_id": int(meta[0]),
                            "stock_id": int(meta[1]),
                            "pred": float(p.item()),
                            "target": float(t.item()),
                        }
                    )
                offset += flat_pred.numel()

        mae_val = total_abs / max(total_count, 1) if total_count > 0 else None
        return mae_val, records

    def _maybe_resume(self, model, optimizer, checkpoint_path: Path, raw_dim: int, gate_dim: int, log_prefix: str, warn_missing_resume: bool) -> Tuple[float, int, int]:
        best_val = float("inf")
        best_epoch = 0
        start_epoch = 0
        if self.args.resume:
            if checkpoint_path.exists():
                ckpt = torch.load(checkpoint_path, map_location="cpu")
                if ckpt.get("raw_dim") == raw_dim and ckpt.get("gate_dim") == gate_dim:
                    model.load_state_dict(ckpt["model_state"])
                    optimizer.load_state_dict(ckpt["optimizer_state"])
                    self._move_optimizer_state_to_device(optimizer, self.device)
                    best_val = ckpt.get("best_val", best_val)
                    best_epoch = ckpt.get("best_epoch", best_epoch)
                    start_epoch = ckpt.get("epoch", -1) + 1
                    self.logger.info(
                        f"{log_prefix} Resumed from epoch {start_epoch} (best val_mae={best_val:.6f} at epoch {best_epoch})"
                    )
                else:
                    self.logger.warning(f"{log_prefix} Checkpoint feature dims mismatch; skip resume.")
                del ckpt
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            elif warn_missing_resume:
                self.logger.warning(
                    f"{log_prefix} Resume requested but checkpoint not found at {checkpoint_path}; start fresh."
                )
        return best_val, best_epoch, start_epoch

    def fit(
        self,
        model,
        optimizer,
        train_loader,
        val_loader,
        writer,
        checkpoint_path: Path,
        best_path: Path,
        input_dim: int,
        raw_dim: int,
        gate_dim: int,
        log_prefix: str,
        warn_missing_resume: bool = False,
    ):
        best_val, best_epoch, start_epoch = self._maybe_resume(
            model, optimizer, checkpoint_path, raw_dim, gate_dim, log_prefix, warn_missing_resume
        )
        if start_epoch > self.args.num_epochs:
            self.logger.info(
                f"{log_prefix} Resume epoch {start_epoch} exceeds num_epochs={self.args.num_epochs}; skip training."
            )
            return {
                "best_val": best_val,
                "best_epoch": best_epoch,
                "start_epoch": start_epoch,
                "skipped": True,
            }

        for epoch in range(start_epoch, self.args.num_epochs + 1):
            train_mae = self.train_one_epoch(model, train_loader, optimizer)
            val_mae = self.evaluate(model, val_loader)

            if writer is not None:
                writer.add_scalar("train/mae", train_mae, epoch)
                writer.add_scalar("val/mae", val_mae, epoch)
            self.logger.info(f"{log_prefix} Epoch {epoch} | train_mae={train_mae:.6f} | val_mae={val_mae:.6f}")

            if val_mae + cfg.EARLY_STOP_MIN_DELTA < best_val:
                best_val = val_mae
                best_epoch = epoch
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "input_dim": input_dim,
                        "raw_dim": raw_dim,
                        "gate_dim": gate_dim,
                    },
                    best_path,
                )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_val": best_val,
                    "best_epoch": best_epoch,
                    "input_dim": input_dim,
                    "raw_dim": raw_dim,
                    "gate_dim": gate_dim,
                },
                checkpoint_path,
            )

            if epoch - best_epoch >= cfg.EARLY_STOP_PATIENCE:
                self.logger.info(
                    f"{log_prefix} Early stopping at epoch {epoch} (best val_mae={best_val:.6f} at epoch {best_epoch})"
                )
                break

        return {
            "best_val": best_val,
            "best_epoch": best_epoch,
            "start_epoch": start_epoch,
            "skipped": False,
        }

    def load_best(self, model, best_path: Path) -> bool:
        if best_path.exists():
            state = torch.load(best_path, map_location=self.device, weights_only=True)
            model.load_state_dict(state["model_state"])
            return True
        return False

    def load_best_or_last(self, model, best_path: Path, last_path: Optional[Path]) -> Optional[str]:
        if self.load_best(model, best_path):
            return "best"
        if last_path is not None and last_path.exists():
            state = torch.load(last_path, map_location=self.device)
            model.load_state_dict(state["model_state"])
            return "last"
        return None


# 基本划分训练
def basical_train():
    """
    按 date_id 划分：最后 81 天为 test，其余日期按 4:1 划分 train/val。
    """
    args = parse_args()
    set_seed(args.seed)
    run_tag = build_run_tag(args, MASTER.__name__)
    model_dir, pred_dir, log_dir = prepare_run_dirs("basical_train")
    logger = get_logger(log_dir / f"train_basic_{run_tag}.log")
    runner = TrainRunner(args, cfg.DEVICE, logger)

    logger.info("加载数据...")
    features, target = read_data(cfg)
    unique_times = np.array(sorted(features.index.get_level_values("time_id").unique()))
    date_ids = unique_times // 10000
    change_idx = np.where(np.diff(date_ids) != 0)[0]
    first_idx = np.concatenate(([0], change_idx + 1))
    last_idx = np.concatenate((change_idx, [len(unique_times) - 1]))
    unique_dates = date_ids[first_idx]
    date_start_times = unique_times[first_idx]
    date_end_times = unique_times[last_idx]

    test_date_count = 81
    if len(unique_dates) < test_date_count + 2:
        logger.error("date_id 数量不足以生成 train/val/test，请检查数据量")
        return

    train_val_dates = len(unique_dates) - test_date_count
    split = max(int(train_val_dates * 0.8), 1)
    if split >= train_val_dates:
        logger.error("4:1 划分无效，请检查 date_id 数量")
        return

    test_start_idx = len(unique_dates) - test_date_count
    train_val_end_idx = test_start_idx - 1

    window = {
        "train_start": date_start_times[0],
        "train_end": date_end_times[split - 1],
        "val_start": date_start_times[split],
        "val_end": date_end_times[train_val_end_idx],
        "test_start": date_start_times[test_start_idx],
        "test_end": date_end_times[-1],
    }

    logger.info(
        f"[Basic] train: {window['train_start']}~{window['train_end']}, "
        f"val: {window['val_start']}~{window['val_end']}, "
        f"test: {window['test_start']}~{window['test_end']}"
    )

    use_section = args.use_section_med_norm
    return_index = False
    if use_section:
        time_index = target.index.get_level_values("time_id")
        train_sel = (time_index >= window["train_start"]) & (time_index <= window["train_end"])
        train_tgt = target.loc[train_sel]
        scaler = fit_section_med_scaler(train_tgt)
        runner.set_section_scaler(scaler)
        target_scaled = apply_section_med_scaler(target, scaler)
        return_index = True
    else:
        target_scaled = target
    train_ds, val_ds, test_ds = build_datasets_for_window(features, target_scaled, cfg, window, return_index=return_index)
    cfg.GROUP_BATCH_SIZE = args.group_batch_size
    train_loader = make_loader(train_ds, cfg, shuffle=True)
    val_loader = make_loader(val_ds, cfg, shuffle=False)
    test_loader = make_test_loader(test_ds, cfg)

    input_dim = train_ds.data_arr.shape[1]
    gate_dim = args.master_gate_feat_dim
    raw_dim = input_dim - gate_dim
    model = MASTER(
        d_feat=raw_dim,
        gate_feat_dim=gate_dim,
        d_model=args.master_d_model,
        t_nhead=args.master_t_nhead,
        s_nhead=args.master_s_nhead,
        t_dropout=args.master_t_dropout,
        s_dropout=args.master_s_dropout,
        beta=args.master_gate_beta,
    ).to(cfg.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    checkpoint_path = (
        Path(args.resume_path) if args.resume_path else model_dir / f"basic_split_last_{run_tag}.pth"
    )
    writer = SummaryWriter(log_dir=str(log_dir / f"basic_split_{run_tag}"))
    best_path = model_dir / f"basic_split_{run_tag}.pth"

    train_state = runner.fit(
        model,
        optimizer,
        train_loader,
        val_loader,
        writer,
        checkpoint_path,
        best_path,
        input_dim,
        raw_dim,
        gate_dim,
        log_prefix="[Basic]",
        warn_missing_resume=True,
    )
    if train_state["skipped"]:
        return

    runner.load_best(model, best_path)

    test_mae, records = runner.predict(model, test_loader, test_ds)
    writer.add_scalar("test/mae", test_mae if test_mae is not None else float("nan"), 0)
    writer.flush()
    writer.close()

    logger.info(f"[Basic] test_mae={test_mae:.6f}" if test_mae is not None else "无 test 标签，未计算 MAE")

    pred_path = pred_dir / f"basic_split_preds_{run_tag}.csv"
    pd.DataFrame.from_records(records).to_csv(pred_path, index=False)
    logger.info(f"已保存预测到 {pred_path}")


# k折训练
def kfold_train():
    """
    K 折训练（时间序列）：
    - 最后 81 天作为固定测试集。
    - 其余日期按时间顺序 4 等分，每次 3 折训练，1 折验证。
    - 训练 4 个模型；在固定测试集上预测并取平均，作为最终验证指标。
    """
    args = parse_args()
    set_seed(args.seed)
    run_tag = build_run_tag(args, MASTER.__name__)
    model_dir, pred_dir, log_dir = prepare_run_dirs("kfold_train")
    logger = get_logger(log_dir / f"train_kfold_{run_tag}.log")
    runner = TrainRunner(args, cfg.DEVICE, logger)

    logger.info("加载数据...")
    features, target = read_data(cfg)
    unique_times = np.array(sorted(features.index.get_level_values("time_id").unique()))

    date_ids = unique_times // 10000
    change_idx = np.where(np.diff(date_ids) != 0)[0]
    first_idx = np.concatenate(([0], change_idx + 1))
    last_idx = np.concatenate((change_idx, [len(unique_times) - 1]))
    unique_dates = date_ids[first_idx]
    date_start_times = unique_times[first_idx]
    date_end_times = unique_times[last_idx]

    fixed_test_days = 81
    if len(unique_dates) < fixed_test_days + 4:
        logger.error("date_id 数量不足以构建 4 折 + 81 天固定测试集，请检查数据量")
        return

    fixed_test_start_idx = len(unique_dates) - fixed_test_days
    fixed_test_start = date_start_times[fixed_test_start_idx]
    fixed_test_end = date_end_times[-1]

    pre_dates = unique_dates[:fixed_test_start_idx]

    folds = np.array_split(pre_dates, 4)
    if any(len(f) == 0 for f in folds):
        logger.error("存在空折，无法进行 4 折训练")
        return

    logger.info(
        f"[KFold] fixed test: {fixed_test_start}~{fixed_test_end}, "
        f"pre_dates: {pre_dates[0]}~{pre_dates[-1]} ({len(pre_dates)} days)"
    )

    time_index = features.index.get_level_values("time_id")
    date_index = (time_index // 10000).to_numpy()

    all_test_preds = []
    use_section = args.use_section_med_norm

    for fold_id, val_dates in enumerate(folds):
        train_dates = np.concatenate([f for i, f in enumerate(folds) if i != fold_id])
        train_dates = np.unique(train_dates)

        fold_path = pred_dir / f"fold{fold_id}_preds_{run_tag}.csv"
        if args.resume and fold_path.exists():
            logger.info(f"[Fold {fold_id}] 检测到已有预测文件，跳过本折训练。")
            all_test_preds.append(pd.read_csv(fold_path))
            continue

        train_sel = np.isin(date_index, train_dates)
        val_sel = np.isin(date_index, val_dates)
        test_sel = (time_index >= fixed_test_start) & (time_index <= fixed_test_end)

        train_feat = features.loc[train_sel]
        train_tgt = target.loc[train_sel]
        val_feat = features.loc[val_sel]
        val_tgt = target.loc[val_sel]
        test_feat = features.loc[test_sel]
        test_tgt = target.loc[test_sel]

        if train_feat.empty or val_feat.empty or test_feat.empty:
            logger.warning(f"[Fold {fold_id}] 训练/验证/测试数据为空，跳过")
            continue
        if use_section:
            scaler = fit_section_med_scaler(train_tgt)
            runner.set_section_scaler(scaler)
            train_tgt = apply_section_med_scaler(train_tgt, scaler)
            val_tgt = apply_section_med_scaler(val_tgt, scaler)
            test_tgt = apply_section_med_scaler(test_tgt, scaler)
        train_start = train_feat.index.get_level_values("time_id").min()
        train_end = train_feat.index.get_level_values("time_id").max()
        val_start = val_feat.index.get_level_values("time_id").min()
        val_end = val_feat.index.get_level_values("time_id").max()

        train_ds = PanelDataset(
            feature_df=train_feat,
            target=train_tgt,
            start=train_start,
            end=train_end,
            step_len=cfg.STEP_LEN,
            fillna_type=cfg.FILLNA_TYPE,
            return_index=use_section,
        )
        val_ds = PanelDataset(
            feature_df=val_feat,
            target=val_tgt,
            start=val_start,
            end=val_end,
            step_len=cfg.STEP_LEN,
            fillna_type=cfg.FILLNA_TYPE,
            return_index=use_section,
        )
        test_ds = PanelDataset(
            feature_df=test_feat,
            target=test_tgt,
            start=fixed_test_start,
            end=fixed_test_end,
            step_len=cfg.STEP_LEN,
            fillna_type=cfg.FILLNA_TYPE,
            return_index=use_section,
        )

        if train_ds.empty or val_ds.empty or test_ds.empty:
            logger.warning(f"[Fold {fold_id}] 训练/验证/测试数据为空，跳过")
            continue

        cfg.GROUP_BATCH_SIZE = args.group_batch_size
        train_loader = make_loader(train_ds, cfg, shuffle=True)
        val_loader = make_loader(val_ds, cfg, shuffle=False)
        test_loader = make_test_loader(test_ds, cfg)

        input_dim = train_ds.data_arr.shape[1]
        gate_dim = args.master_gate_feat_dim
        raw_dim = input_dim - gate_dim
        model = MASTER(
            d_feat=raw_dim,
            gate_feat_dim=gate_dim,
            d_model=args.master_d_model,
            t_nhead=args.master_t_nhead,
            s_nhead=args.master_s_nhead,
            t_dropout=args.master_t_dropout,
            s_dropout=args.master_s_dropout,
            beta=args.master_gate_beta,
        ).to(cfg.DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        checkpoint_path = model_dir / f"fold{fold_id}_last_{run_tag}.pth"
        writer = SummaryWriter(log_dir=str(log_dir / f"fold_{fold_id}_{run_tag}"))
        best_path = model_dir / f"fold{fold_id}_{run_tag}.pth"

        runner.fit(
            model,
            optimizer,
            train_loader,
            val_loader,
            writer,
            checkpoint_path,
            best_path,
            input_dim,
            raw_dim,
            gate_dim,
            log_prefix=f"[Fold {fold_id}]",
        )
        runner.load_best_or_last(model, best_path, checkpoint_path)

        test_mae, records = runner.predict(model, test_loader, test_ds)
        writer.add_scalar("test/mae", test_mae if test_mae is not None else float("nan"), 0)
        writer.flush()
        writer.close()

        logger.info(
            f"[Fold {fold_id}] test_mae={test_mae:.6f}" if test_mae is not None else "[Fold {fold_id}] 无 test 标签"
        )

        fold_df = pd.DataFrame.from_records(records)
        fold_df.to_csv(fold_path, index=False)
        all_test_preds.append(fold_df)

        del train_ds, val_ds, test_ds
        del train_loader, val_loader, test_loader
        del model, optimizer, writer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"[Fold {fold_id}] 已保存测试预测到 {fold_path}")

    if not all_test_preds:
        logger.error("未得到任何折的测试预测，退出。")
        return

    combined = pd.concat(all_test_preds, ignore_index=True)
    grouped = combined.groupby(["time_id", "stock_id"])
    avg_pred = grouped["pred"].mean()
    target_first = grouped["target"].first()
    final_df = avg_pred.reset_index().rename(columns={"pred": "pred_avg"})
    final_df["target"] = target_first.values

    final_mae = None
    if final_df["target"].notna().any():
        final_df["abs_err"] = (final_df["pred_avg"] - final_df["target"]).abs()
        final_mae = final_df["abs_err"].mean()
        logger.info(f"[KFold] avg_pred_mae={final_mae:.6f}")
    else:
        logger.info("[KFold] 测试集无标签，无法计算平均预测指标")

    final_path = pred_dir / f"kfold_test_preds_{run_tag}.csv"
    final_df[["time_id", "stock_id", "pred_avg", "target"]].to_csv(final_path, index=False)
    logger.info(f"[KFold] 已保存测试预测到 {final_path}")

    return final_mae


# 拓展窗口训练
def expanding_with_fixed_window_train():
    """
    Expanding training on pre-test dates with a fixed 40-day validation window.
    Trains 3 models on first 120/240/360 days and ensembles on fixed test.
    """
    args = parse_args()
    set_seed(args.seed)
    run_tag = build_run_tag(args, MASTER.__name__)
    model_dir, pred_dir, log_dir = prepare_run_dirs("expanding_with_fixed_window_train")
    logger = get_logger(log_dir / f"train_expanding_fixed_{run_tag}.log")
    runner = TrainRunner(args, cfg.DEVICE, logger)

    logger.info("Loading data...")
    features, target = read_data(cfg)
    unique_times = np.array(sorted(features.index.get_level_values("time_id").unique()))

    date_ids = unique_times // 10000
    change_idx = np.where(np.diff(date_ids) != 0)[0]
    first_idx = np.concatenate(([0], change_idx + 1))
    last_idx = np.concatenate((change_idx, [len(unique_times) - 1]))
    unique_dates = date_ids[first_idx]
    date_start_times = unique_times[first_idx]
    date_end_times = unique_times[last_idx]

    fixed_test_days = 81
    train_lengths = [120, 240, 360]
    val_days = 40

    min_required = max(train_lengths) + val_days + fixed_test_days
    if len(unique_dates) < min_required:
        logger.error(
            f"[ExpFixed] Not enough dates: need at least {min_required}, got {len(unique_dates)}."
        )
        return

    fixed_test_start_idx = len(unique_dates) - fixed_test_days
    fixed_test_start = date_start_times[fixed_test_start_idx]
    fixed_test_end = date_end_times[-1]

    pre_dates = unique_dates[:fixed_test_start_idx]

    logger.info(
        f"[ExpFixed] fixed test: {fixed_test_start}~{fixed_test_end}, "
        f"pre_dates: {pre_dates[0]}~{pre_dates[-1]} ({len(pre_dates)} days)"
    )

    time_index = features.index.get_level_values("time_id")
    date_index = (time_index // 10000).to_numpy()

    all_test_preds: List[pd.DataFrame] = []
    val_decays: List[Optional[float]] = []
    model_tags: List[str] = []
    use_section = args.use_section_med_norm

    for train_len in train_lengths:
        if train_len + val_days > len(pre_dates):
            logger.warning(f"[ExpFixed train{train_len}] Not enough dates for validation; skip.")
            continue

        train_dates = pre_dates[:train_len]
        val_dates = pre_dates[train_len : train_len + val_days]

        logger.info(
            f"[ExpFixed train{train_len}] train_dates: {train_dates[0]}~{train_dates[-1]}, "
            f"val_dates: {val_dates[0]}~{val_dates[-1]}"
        )

        train_sel = np.isin(date_index, train_dates)
        val_sel = np.isin(date_index, val_dates)
        test_sel = (time_index >= fixed_test_start) & (time_index <= fixed_test_end)

        train_feat = features.loc[train_sel]
        train_tgt = target.loc[train_sel]
        val_feat = features.loc[val_sel]
        val_tgt = target.loc[val_sel]
        test_feat = features.loc[test_sel]
        test_tgt = target.loc[test_sel]

        if train_feat.empty or val_feat.empty or test_feat.empty:
            logger.warning(f"[ExpFixed train{train_len}] Empty train/val/test data; skip.")
            continue
        if use_section:
            scaler = fit_section_med_scaler(train_tgt)
            runner.set_section_scaler(scaler)
            train_tgt = apply_section_med_scaler(train_tgt, scaler)
            val_tgt = apply_section_med_scaler(val_tgt, scaler)
            test_tgt = apply_section_med_scaler(test_tgt, scaler)
        train_start = train_feat.index.get_level_values("time_id").min()
        train_end = train_feat.index.get_level_values("time_id").max()
        val_start = val_feat.index.get_level_values("time_id").min()
        val_end = val_feat.index.get_level_values("time_id").max()

        train_ds = PanelDataset(
            feature_df=train_feat,
            target=train_tgt,
            start=train_start,
            end=train_end,
            step_len=cfg.STEP_LEN,
            fillna_type=cfg.FILLNA_TYPE,
            return_index=use_section,
        )
        val_ds = PanelDataset(
            feature_df=val_feat,
            target=val_tgt,
            start=val_start,
            end=val_end,
            step_len=cfg.STEP_LEN,
            fillna_type=cfg.FILLNA_TYPE,
            return_index=use_section,
        )
        test_ds = PanelDataset(
            feature_df=test_feat,
            target=test_tgt,
            start=fixed_test_start,
            end=fixed_test_end,
            step_len=cfg.STEP_LEN,
            fillna_type=cfg.FILLNA_TYPE,
            return_index=use_section,
        )

        if train_ds.empty or val_ds.empty or test_ds.empty:
            logger.warning(f"[ExpFixed train{train_len}] Empty dataset after build; skip.")
            continue

        cfg.GROUP_BATCH_SIZE = args.group_batch_size
        train_loader = make_loader(train_ds, cfg, shuffle=True)
        val_loader = make_loader(val_ds, cfg, shuffle=False)
        test_loader = make_test_loader(test_ds, cfg)

        input_dim = train_ds.data_arr.shape[1]
        gate_dim = args.master_gate_feat_dim
        raw_dim = input_dim - gate_dim
        model = MASTER(
            d_feat=raw_dim,
            gate_feat_dim=gate_dim,
            d_model=args.master_d_model,
            t_nhead=args.master_t_nhead,
            s_nhead=args.master_s_nhead,
            t_dropout=args.master_t_dropout,
            s_dropout=args.master_s_dropout,
            beta=args.master_gate_beta,
        ).to(cfg.DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        model_tag = f"train{train_len}"
        checkpoint_path = model_dir / f"exp_fixed_{model_tag}_last_{run_tag}.pth"
        writer = SummaryWriter(log_dir=str(log_dir / f"exp_fixed_{model_tag}_{run_tag}"))
        best_path = model_dir / f"exp_fixed_{model_tag}_{run_tag}.pth"

        runner.fit(
            model,
            optimizer,
            train_loader,
            val_loader,
            writer,
            checkpoint_path,
            best_path,
            input_dim,
            raw_dim,
            gate_dim,
            log_prefix=f"[ExpFixed {model_tag}]",
        )
        runner.load_best_or_last(model, best_path, checkpoint_path)

        val_mae_final, val_records = runner.predict(model, val_loader, val_ds)
        decay, seg_maes = _compute_decay_from_records(val_records, window_days=val_days, segment_days=10)
        if val_mae_final is not None:
            logger.info(
                f"[ExpFixed {model_tag}] val_mae={val_mae_final:.6f} decay={decay} seg_mae={seg_maes}"
            )
        else:
            logger.info(f"[ExpFixed {model_tag}] val_mae=None decay={decay} seg_mae={seg_maes}")

        test_mae, records = runner.predict(model, test_loader, test_ds)
        writer.add_scalar("test/mae", test_mae if test_mae is not None else float("nan"), 0)
        writer.flush()
        writer.close()

        logger.info(
            f"[ExpFixed {model_tag}] test_mae={test_mae:.6f}"
            if test_mae is not None
            else f"[ExpFixed {model_tag}] test has no labels"
        )

        pred_path = pred_dir / f"exp_fixed_{model_tag}_preds_{run_tag}.csv"
        model_df = pd.DataFrame.from_records(records)
        model_df.to_csv(pred_path, index=False)
        logger.info(f"[ExpFixed {model_tag}] Saved test predictions to {pred_path}")

        all_test_preds.append(model_df)
        val_decays.append(decay)
        model_tags.append(model_tag)

        del train_ds, val_ds, test_ds
        del train_loader, val_loader, test_loader
        del model, optimizer, writer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not all_test_preds:
        logger.error("[ExpFixed] No model predictions were generated.")
        return

    combined = pd.concat(all_test_preds, ignore_index=True)
    grouped = combined.groupby(["time_id", "stock_id"])
    avg_pred = grouped["pred"].mean()
    target_first = grouped["target"].first()
    mean_df = avg_pred.reset_index().rename(columns={"pred": "pred_avg"})
    mean_df["target"] = target_first.values

    mean_mae = None
    if mean_df["target"].notna().any():
        mean_df["abs_err"] = (mean_df["pred_avg"] - mean_df["target"]).abs()
        mean_mae = mean_df["abs_err"].mean()
        logger.info(f"[ExpFixed] mean_pred_mae={mean_mae:.6f}")
    else:
        logger.info("[ExpFixed] mean_pred has no target labels.")

    mean_path = pred_dir / f"exp_fixed_mean_preds_{run_tag}.csv"
    mean_df[["time_id", "stock_id", "pred_avg", "target"]].to_csv(mean_path, index=False)
    logger.info(f"[ExpFixed] Saved mean predictions to {mean_path}")

    weights = _assign_decay_weights(val_decays)
    logger.info(f"[ExpFixed] decay weights: {list(zip(model_tags, val_decays, weights))}")

    weighted_frames = []
    for df, w in zip(all_test_preds, weights):
        df_w = df.copy()
        df_w["pred"] = df_w["pred"] * w
        weighted_frames.append(df_w)

    combined_w = pd.concat(weighted_frames, ignore_index=True)
    grouped_w = combined_w.groupby(["time_id", "stock_id"])
    weighted_pred = grouped_w["pred"].sum()
    target_first = grouped_w["target"].first()
    weighted_df = weighted_pred.reset_index().rename(columns={"pred": "pred_weighted"})
    weighted_df["target"] = target_first.values

    weighted_mae = None
    if weighted_df["target"].notna().any():
        weighted_df["abs_err"] = (weighted_df["pred_weighted"] - weighted_df["target"]).abs()
        weighted_mae = weighted_df["abs_err"].mean()
        logger.info(f"[ExpFixed] weighted_pred_mae={weighted_mae:.6f}")
    else:
        logger.info("[ExpFixed] weighted_pred has no target labels.")

    weighted_path = pred_dir / f"exp_fixed_weighted_preds_{run_tag}.csv"
    weighted_df[["time_id", "stock_id", "pred_weighted", "target"]].to_csv(weighted_path, index=False)
    logger.info(f"[ExpFixed] Saved weighted predictions to {weighted_path}")

    return {"mean_mae": mean_mae, "weighted_mae": weighted_mae, "weights": weights}


# 拓展全量窗口训练
def expanding_with_total_window_train():
    """
    Expanding training on pre-test dates with full remaining validation window.
    Trains 3 models on first 120/240/360 days and ensembles on fixed test.
    """
    args = parse_args()
    set_seed(args.seed)
    run_tag = build_run_tag(args, MASTER.__name__)
    model_dir, pred_dir, log_dir = prepare_run_dirs("expanding_with_ftotal_window_train")
    logger = get_logger(log_dir / f"train_expanding_ftotal_{run_tag}.log")
    runner = TrainRunner(args, cfg.DEVICE, logger)

    logger.info("Loading data...")
    features, target = read_data(cfg)
    unique_times = np.array(sorted(features.index.get_level_values("time_id").unique()))

    date_ids = unique_times // 10000
    change_idx = np.where(np.diff(date_ids) != 0)[0]
    first_idx = np.concatenate(([0], change_idx + 1))
    last_idx = np.concatenate((change_idx, [len(unique_times) - 1]))
    unique_dates = date_ids[first_idx]
    date_start_times = unique_times[first_idx]
    date_end_times = unique_times[last_idx]

    fixed_test_days = 81
    train_lengths = [120, 240, 360]
    val_days = 40

    min_required = max(train_lengths) + 1 + fixed_test_days
    if len(unique_dates) < min_required:
        logger.error(
            f"[ExpFtotal] Not enough dates: need at least {min_required}, got {len(unique_dates)}."
        )
        return

    fixed_test_start_idx = len(unique_dates) - fixed_test_days
    fixed_test_start = date_start_times[fixed_test_start_idx]
    fixed_test_end = date_end_times[-1]

    pre_dates = unique_dates[:fixed_test_start_idx]

    logger.info(
        f"[ExpFtotal] fixed test: {fixed_test_start}~{fixed_test_end}, "
        f"pre_dates: {pre_dates[0]}~{pre_dates[-1]} ({len(pre_dates)} days)"
    )

    time_index = features.index.get_level_values("time_id")
    date_index = (time_index // 10000).to_numpy()

    all_test_preds: List[pd.DataFrame] = []
    val_decays: List[Optional[float]] = []
    model_tags: List[str] = []
    use_section = args.use_section_med_norm

    for train_len in train_lengths:
        if train_len >= len(pre_dates):
            logger.warning(f"[ExpFtotal train{train_len}] Not enough dates for validation; skip.")
            continue

        train_dates = pre_dates[:train_len]
        val_dates = pre_dates[train_len:]

        logger.info(
            f"[ExpFtotal train{train_len}] train_dates: {train_dates[0]}~{train_dates[-1]}, "
            f"val_dates: {val_dates[0]}~{val_dates[-1]} ({len(val_dates)} days)"
        )

        train_sel = np.isin(date_index, train_dates)
        val_sel = np.isin(date_index, val_dates)
        test_sel = (time_index >= fixed_test_start) & (time_index <= fixed_test_end)

        train_feat = features.loc[train_sel]
        train_tgt = target.loc[train_sel]
        val_feat = features.loc[val_sel]
        val_tgt = target.loc[val_sel]
        test_feat = features.loc[test_sel]
        test_tgt = target.loc[test_sel]

        if train_feat.empty or val_feat.empty or test_feat.empty:
            logger.warning(f"[ExpFtotal train{train_len}] Empty train/val/test data; skip.")
            continue
        if use_section:
            scaler = fit_section_med_scaler(train_tgt)
            runner.set_section_scaler(scaler)
            train_tgt = apply_section_med_scaler(train_tgt, scaler)
            val_tgt = apply_section_med_scaler(val_tgt, scaler)
            test_tgt = apply_section_med_scaler(test_tgt, scaler)
        train_start = train_feat.index.get_level_values("time_id").min()
        train_end = train_feat.index.get_level_values("time_id").max()
        val_start = val_feat.index.get_level_values("time_id").min()
        val_end = val_feat.index.get_level_values("time_id").max()

        train_ds = PanelDataset(
            feature_df=train_feat,
            target=train_tgt,
            start=train_start,
            end=train_end,
            step_len=cfg.STEP_LEN,
            fillna_type=cfg.FILLNA_TYPE,
            return_index=use_section,
        )
        val_ds = PanelDataset(
            feature_df=val_feat,
            target=val_tgt,
            start=val_start,
            end=val_end,
            step_len=cfg.STEP_LEN,
            fillna_type=cfg.FILLNA_TYPE,
            return_index=use_section,
        )
        test_ds = PanelDataset(
            feature_df=test_feat,
            target=test_tgt,
            start=fixed_test_start,
            end=fixed_test_end,
            step_len=cfg.STEP_LEN,
            fillna_type=cfg.FILLNA_TYPE,
            return_index=use_section,
        )

        if train_ds.empty or val_ds.empty or test_ds.empty:
            logger.warning(f"[ExpFtotal train{train_len}] Empty dataset after build; skip.")
            continue

        cfg.GROUP_BATCH_SIZE = args.group_batch_size
        train_loader = make_loader(train_ds, cfg, shuffle=True)
        val_loader = make_loader(val_ds, cfg, shuffle=False)
        test_loader = make_test_loader(test_ds, cfg)

        input_dim = train_ds.data_arr.shape[1]
        gate_dim = args.master_gate_feat_dim
        raw_dim = input_dim - gate_dim
        model = MASTER(
            d_feat=raw_dim,
            gate_feat_dim=gate_dim,
            d_model=args.master_d_model,
            t_nhead=args.master_t_nhead,
            s_nhead=args.master_s_nhead,
            t_dropout=args.master_t_dropout,
            s_dropout=args.master_s_dropout,
            beta=args.master_gate_beta,
        ).to(cfg.DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        model_tag = f"train{train_len}"
        checkpoint_path = model_dir / f"exp_ftotal_{model_tag}_last_{run_tag}.pth"
        writer = SummaryWriter(log_dir=str(log_dir / f"exp_ftotal_{model_tag}_{run_tag}"))
        best_path = model_dir / f"exp_ftotal_{model_tag}_{run_tag}.pth"

        runner.fit(
            model,
            optimizer,
            train_loader,
            val_loader,
            writer,
            checkpoint_path,
            best_path,
            input_dim,
            raw_dim,
            gate_dim,
            log_prefix=f"[ExpFtotal {model_tag}]",
        )
        runner.load_best_or_last(model, best_path, checkpoint_path)

        val_mae_final, val_records = runner.predict(model, val_loader, val_ds)
        decay, seg_maes = _compute_decay_from_records(val_records, window_days=decay_days, segment_days=10)
        if val_mae_final is not None:
            logger.info(
                f"[ExpFtotal {model_tag}] val_mae={val_mae_final:.6f} "
                f"decay={decay} seg_mae={seg_maes} (decay_days={decay_days})"
            )
        else:
            logger.info(
                f"[ExpFtotal {model_tag}] val_mae=None decay={decay} seg_mae={seg_maes} "
                f"(decay_days={decay_days})"
            )

        test_mae, records = runner.predict(model, test_loader, test_ds)
        writer.add_scalar("test/mae", test_mae if test_mae is not None else float("nan"), 0)
        writer.flush()
        writer.close()

        logger.info(
            f"[ExpFtotal {model_tag}] test_mae={test_mae:.6f}"
            if test_mae is not None
            else f"[ExpFtotal {model_tag}] test has no labels"
        )

        pred_path = pred_dir / f"exp_ftotal_{model_tag}_preds_{run_tag}.csv"
        model_df = pd.DataFrame.from_records(records)
        model_df.to_csv(pred_path, index=False)
        logger.info(f"[ExpFtotal {model_tag}] Saved test predictions to {pred_path}")

        all_test_preds.append(model_df)
        val_decays.append(decay)
        model_tags.append(model_tag)

        del train_ds, val_ds, test_ds
        del train_loader, val_loader, test_loader
        del model, optimizer, writer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not all_test_preds:
        logger.error("[ExpFtotal] No model predictions were generated.")
        return

    combined = pd.concat(all_test_preds, ignore_index=True)
    grouped = combined.groupby(["time_id", "stock_id"])
    avg_pred = grouped["pred"].mean()
    target_first = grouped["target"].first()
    mean_df = avg_pred.reset_index().rename(columns={"pred": "pred_avg"})
    mean_df["target"] = target_first.values

    mean_mae = None
    if mean_df["target"].notna().any():
        mean_df["abs_err"] = (mean_df["pred_avg"] - mean_df["target"]).abs()
        mean_mae = mean_df["abs_err"].mean()
        logger.info(f"[ExpFtotal] mean_pred_mae={mean_mae:.6f}")
    else:
        logger.info("[ExpFtotal] mean_pred has no target labels.")

    mean_path = pred_dir / f"exp_ftotal_mean_preds_{run_tag}.csv"
    mean_df[["time_id", "stock_id", "pred_avg", "target"]].to_csv(mean_path, index=False)
    logger.info(f"[ExpFtotal] Saved mean predictions to {mean_path}")

    weights = _assign_decay_weights(val_decays)
    logger.info(f"[ExpFtotal] decay weights: {list(zip(model_tags, val_decays, weights))}")

    weighted_frames = []
    for df, w in zip(all_test_preds, weights):
        df_w = df.copy()
        df_w["pred"] = df_w["pred"] * w
        weighted_frames.append(df_w)

    combined_w = pd.concat(weighted_frames, ignore_index=True)
    grouped_w = combined_w.groupby(["time_id", "stock_id"])
    weighted_pred = grouped_w["pred"].sum()
    target_first = grouped_w["target"].first()
    weighted_df = weighted_pred.reset_index().rename(columns={"pred": "pred_weighted"})
    weighted_df["target"] = target_first.values

    weighted_mae = None
    if weighted_df["target"].notna().any():
        weighted_df["abs_err"] = (weighted_df["pred_weighted"] - weighted_df["target"]).abs()
        weighted_mae = weighted_df["abs_err"].mean()
        logger.info(f"[ExpFtotal] weighted_pred_mae={weighted_mae:.6f}")
    else:
        logger.info("[ExpFtotal] weighted_pred has no target labels.")

    weighted_path = pred_dir / f"exp_ftotal_weighted_preds_{run_tag}.csv"
    weighted_df[["time_id", "stock_id", "pred_weighted", "target"]].to_csv(weighted_path, index=False)
    logger.info(f"[ExpFtotal] Saved weighted predictions to {weighted_path}")

    return {"mean_mae": mean_mae, "weighted_mae": weighted_mae, "weights": weights}


if __name__ == '__main__':
    # basical_train()
    kfold_train()
    # expanding_with_fixed_window_train()
    # expanding_with_total_window_train()


























