"""
模型集成脚本
将多个滚动窗口训练的模型进行集成，在最终测试集上评估效果
"""

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch

import config as cfg
from dataset import (
    build_datasets_for_window,
    generate_rolling_windows,
    make_test_loader,
    read_data,
)
from model import create_model
from utils import get_logger, set_seed


def _maybe_denorm(t: torch.Tensor, label_scaler: Optional[Dict[str, float]]):
    if label_scaler is None or t is None:
        return t
    if label_scaler.get("std") is None or label_scaler.get("mean") is None:
        return t
    return t * label_scaler["std"] + label_scaler["mean"]


def load_model(model_path: Path, device: str) -> Tuple:
    """加载模型权重，返回 (model, state_dict)"""
    state = torch.load(model_path, map_location=device, weights_only=True)

    # 优先使用保存的 model_type
    model_type = state.get("model_type", None)
    if model_type is None:
        use_attention = state.get("use_attention", getattr(cfg, "USE_ATTENTION", False))
        model_type = "gru_attn" if use_attention else "gru"

    input_dim = state.get("input_dim", None)
    model = create_model(model_type, input_dim, cfg).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()

    # 提取 label_scaler
    label_scaler = None
    if state.get("label_mean") is not None and state.get("label_std") is not None:
        label_scaler = {"mean": float(state["label_mean"]), "std": float(state["label_std"])}

    return model, state, label_scaler


def predict_single_model(
    model, loader, dataset, device, label_scaler: Optional[Dict] = None
) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    单个模型预测
    返回: (predictions, targets, indices)
    """
    records = []
    predictions = []
    targets = []
    idx_list = list(dataset.get_index())
    offset = 0

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

            bs_valid = int(mask.sum().item())
            x = x.to(device)
            pred = model(x).cpu()
            tgt = y.cpu() if y is not None else None
            mask_cpu = mask.cpu()

            pred_valid = pred[mask_cpu]
            tgt_valid = tgt[mask_cpu] if tgt is not None else None

            # 反标准化
            if label_scaler is not None:
                pred_valid = _maybe_denorm(pred_valid, label_scaler)
                if tgt_valid is not None:
                    tgt_valid = _maybe_denorm(tgt_valid, label_scaler)

            batch_idx = idx_list[offset : offset + bs_valid]

            predictions.append(pred_valid.numpy())
            if tgt_valid is not None:
                targets.append(tgt_valid.numpy())

            for meta, p, t in zip(batch_idx, pred_valid.numpy(), tgt_valid.numpy() if tgt_valid is not None else [None] * bs_valid):
                records.append({
                    "time_id": int(meta[0]),
                    "stock_id": int(meta[1]),
                    "pred": float(p),
                    "target": float(t) if t is not None else None,
                })

            offset += bs_valid

    preds = np.concatenate(predictions) if predictions else np.array([])
    targs = np.concatenate(targets) if targets else np.array([])

    return preds, targs, records


def calculate_weights(
    n_windows: int,
    method: str = "uniform",
    val_maes: Optional[np.ndarray] = None,
    temperature: float = 1.0,
) -> np.ndarray:
    """
    计算集成权重

    Args:
        n_windows: 窗口数量
        method: 权重计算方式
            - "uniform": 等权重
            - "inverse_mae": 验证集 MAE 的倒数加权
            - "exponential_mae": 指数衰减的 MAE 倒数
            - "time_decay": 时间衰减（越近权重越高）
            - "time_mae": 时间衰减 * MAE 加权
        val_maes: 各窗口验证集 MAE
        temperature: 温度参数，控制权重分布的尖锐程度

    Returns:
        归一化的权重数组
    """
    if method == "uniform":
        weights = np.ones(n_windows)

    elif method == "time_decay":
        # 时间衰减：最近窗口权重最高
        # 使用指数衰减: w_i = exp(i * temperature / n_windows)
        indices = np.arange(n_windows)
        weights = np.exp(indices * temperature / n_windows)

    elif method == "inverse_mae":
        if val_maes is None:
            raise ValueError("val_maes 不能为空")
        # MAE 越小权重越高: w_i = 1 / mae_i
        weights = 1.0 / (val_maes + 1e-8)

    elif method == "exponential_mae":
        if val_maes is None:
            raise ValueError("val_maes 不能为空")
        # 指数形式的 MAE 倒数: w_i = exp(-mae_i / temperature)
        weights = np.exp(-val_maes / temperature)

    elif method == "time_mae":
        if val_maes is None:
            raise ValueError("val_maes 不能为空")
        # 时间衰减 * MAE 加权
        indices = np.arange(n_windows)
        time_weights = np.exp(indices * temperature / n_windows)
        mae_weights = 1.0 / (val_maes + 1e-8)
        weights = time_weights * mae_weights

    else:
        raise ValueError(f"未知的权重计算方式: {method}")

    # 归一化
    weights = weights / weights.sum()
    return weights


def ensemble_predictions(
    all_preds: Dict[int, np.ndarray],
    method: str = "mean",
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    集成多个模型的预测

    Args:
        all_preds: {window_id: predictions_array}
        method: 集成方式
            - "mean": 算术平均
            - "weighted_mean": 加权平均
            - "median": 中位数
            - "trim_mean": 截尾平均（去掉最高最低各 10%）
        weights: 加权平均时的权重

    Returns:
        集成后的预测数组
    """
    pred_matrix = np.stack([all_preds[i] for i in sorted(all_preds.keys())], axis=0)

    if method == "mean":
        return pred_matrix.mean(axis=0)

    elif method == "weighted_mean":
        if weights is None:
            raise ValueError("weighted_mean 需要提供 weights")
        return (pred_matrix * weights[:, None]).sum(axis=0)

    elif method == "median":
        return np.median(pred_matrix, axis=0)

    elif method == "trim_mean":
        from scipy.stats import trim_mean
        return trim_mean(pred_matrix, 0.1, axis=0)

    else:
        raise ValueError(f"未知的集成方式: {method}")


def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict:
    """计算评估指标"""
    errors = predictions - targets
    abs_errors = np.abs(errors)
    squared_errors = errors ** 2

    return {
        "mae": float(np.mean(abs_errors)),
        "rmse": float(np.sqrt(np.mean(squared_errors))),
        "bias": float(np.mean(errors)),
        "median_ae": float(np.median(abs_errors)),
        "std_error": float(np.std(errors)),
    }


def main():
    set_seed(cfg.SEED)
    logger = get_logger(Path("output/logs/ensemble.log"))

    logger.info("=" * 60)
    logger.info("模型集成分析")
    logger.info("=" * 60)

    # 1. 加载数据
    logger.info("加载数据...")
    features, target = read_data(cfg)
    unique_times = np.array(sorted(features.index.get_level_values("time_id").unique()))
    windows = generate_rolling_windows(unique_times, cfg)

    if not windows:
        logger.error("无法生成滚动窗口，请检查配置")
        return

    n_windows = len(windows)
    logger.info(f"共有 {n_windows} 个滚动窗口")

    # 2. 查找所有模型文件
    model_dir = cfg.MODEL_DIR
    model_files = sorted(model_dir.glob("*_window_*.pth"))

    if not model_files:
        logger.error(f"未找到模型文件: {model_dir}")
        return

    logger.info(f"找到 {len(model_files)} 个模型文件")

    # 3. 解析模型文件，获取验证集 MAE
    window_val_maes = {}
    for model_path in model_files:
        state = torch.load(model_path, map_location=cfg.DEVICE, weights_only=True)
        # 从文件名提取 window_id
        stem = model_path.stem
        if "window_" in stem:
            window_id = int(stem.split("window_")[-1])
            # 从 TensorBoard 日志或保存的指标中获取 val_mae
            # 这里简化处理，使用文件加载时的信息
            window_val_maes[window_id] = state.get("best_val_mae", 1.0)

    logger.info(f"加载了 {len(window_val_maes)} 个窗口的模型")

    # 4. 对每个窗口加载模型并预测
    all_predictions = {}  # {window_id: predictions_array}
    all_targets = None
    all_records = {}      # {window_id: records_list}

    # 使用最后一个窗口的测试集
    test_window = windows[-1]
    logger.info(f"测试集范围: {test_window['test_start']} ~ {test_window['test_end']}")

    # 构建测试数据集（使用最后一个窗口）
    _, _, test_ds, _, _ = build_datasets_for_window(features, target, cfg, test_window)
    test_loader = make_test_loader(test_ds, cfg)

    for window_id, window in enumerate(windows):
        # 查找对应的模型文件
        model_path = None
        for f in model_files:
            if f"window_{window_id}" in f.name:
                model_path = f
                break

        if model_path is None:
            logger.warning(f"未找到 window_{window_id} 的模型文件，跳过")
            continue

        try:
            logger.info(f"加载 window_{window_id} 模型: {model_path.name}")
            model, state, label_scaler = load_model(model_path, cfg.DEVICE)

            preds, targs, records = predict_single_model(model, test_loader, test_ds, cfg.DEVICE, label_scaler)

            all_predictions[window_id] = preds
            all_records[window_id] = records

            if all_targets is None and len(targs) > 0:
                all_targets = targs

            # 计算单模型 MAE
            if len(targs) > 0:
                mae = np.mean(np.abs(preds - targs))
                logger.info(f"  Window {window_id} MAE: {mae:.6f}")

        except Exception as e:
            logger.error(f"处理 window_{window_id} 时出错: {e}")
            continue

    if not all_predictions:
        logger.error("没有成功加载任何模型")
        return

    logger.info(f"\n成功加载 {len(all_predictions)} 个模型的预测")

    # 5. 计算各种权重
    logger.info("\n" + "=" * 60)
    logger.info("权重计算")
    logger.info("=" * 60)

    weight_methods = ["uniform", "time_decay", "inverse_mae", "exponential_mae", "time_mae"]
    all_weights = {}

    for method in weight_methods:
        try:
            val_maes = np.array([window_val_maes.get(wid, 1.0) for wid in sorted(all_predictions.keys())])
            weights = calculate_weights(len(all_predictions), method=method, val_maes=val_maes)
            all_weights[method] = weights
            logger.info(f"\n{method} 权重:")
            for i, wid in enumerate(sorted(all_predictions.keys())):
                logger.info(f"  Window {wid}: {weights[i]:.4f}")
        except Exception as e:
            logger.warning(f"计算 {method} 权重失败: {e}")

    # 6. 集成预测
    logger.info("\n" + "=" * 60)
    logger.info("集成结果")
    logger.info("=" * 60)

    ensemble_methods = ["mean", "weighted_mean", "median", "trim_mean"]
    results = []

    for ens_method in ensemble_methods:
        logger.info(f"\n{ens_method} 集成:")

        for weight_name, weights in all_weights.items():
            if ens_method == "mean" and weight_name != "uniform":
                continue
            if ens_method == "weighted_mean" and weight_name == "uniform":
                continue

            try:
                if ens_method == "weighted_mean":
                    preds = ensemble_predictions(all_predictions, method="weighted_mean", weights=weights)
                    name = f"{ens_method}_{weight_name}"
                else:
                    preds = ensemble_predictions(all_predictions, method=ens_method)
                    name = ens_method

                metrics = calculate_metrics(preds, all_targets)
                results.append({
                    "method": name,
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "bias": metrics["bias"],
                    "median_ae": metrics["median_ae"],
                })

                logger.info(f"  {name}: MAE={metrics['mae']:.6f}, RMSE={metrics['rmse']:.6f}, Bias={metrics['bias']:.6f}")

            except Exception as e:
                logger.warning(f"  {name}_{weight_name} 失败: {e}")

    # 7. 汇总对比
    logger.info("\n" + "=" * 60)
    logger.info("结果汇总 (按 MAE 排序)")
    logger.info("=" * 60)

    results_df = pd.DataFrame(results).sort_values("mae")
    logger.info("\n" + results_df.to_string(index=False))

    # 保存结果
    output_dir = Path("output/ensemble")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_dir / "ensemble_results.csv", index=False)
    logger.info(f"\n结果已保存到 {output_dir / 'ensemble_results.csv'}")

    # 8. 保存最佳集成预测
    if results_df.empty:
        logger.warning("没有有效的集成结果")
        return

    best_method = results_df.iloc[0]["method"]

    # 重新计算最佳预测
    if best_method.startswith("weighted_mean"):
        weight_name = best_method.split("_", 1)[1] if "_" in best_method else "time_mae"
        weights = all_weights.get(weight_name, all_weights.get("time_decay"))
        best_preds = ensemble_predictions(all_predictions, method="weighted_mean", weights=weights)
    else:
        best_preds = ensemble_predictions(all_predictions, method=best_method)

    # 构建预测记录
    records = all_records[list(all_predictions.keys())[0]]
    ensemble_records = []
    for i, rec in enumerate(records):
        rec_copy = rec.copy()
        rec_copy["pred"] = float(best_preds[i])
        ensemble_records.append(rec_copy)

    pred_df = pd.DataFrame.from_records(ensemble_records)
    pred_df.to_csv(output_dir / f"ensemble_{best_method}_preds.csv", index=False)
    logger.info(f"最佳集成预测已保存到 {output_dir / f'ensemble_{best_method}_preds.csv'}")

    # 9. 单模型对比
    logger.info("\n" + "=" * 60)
    logger.info("单模型 vs 集成模型对比")
    logger.info("=" * 60)

    single_model_maes = []
    for wid in sorted(all_predictions.keys()):
        preds = all_predictions[wid]
        mae = np.mean(np.abs(preds - all_targets))
        single_model_maes.append(mae)

    logger.info(f"单模型平均 MAE: {np.mean(single_model_maes):.6f} ± {np.std(single_model_maes):.6f}")
    logger.info(f"单模型最佳 MAE: {np.min(single_model_maes):.6f}")
    logger.info(f"集成最佳 MAE: {results_df.iloc[0]['mae']:.6f}")
    logger.info(f"集成提升: {(np.mean(single_model_maes) - results_df.iloc[0]['mae']) / np.mean(single_model_maes) * 100:.2f}%")

    # 10. 生成对比图
    try:
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 单模型 MAE 分布
        axes[0].bar(range(len(single_model_maes)), single_model_maes, alpha=0.7, label='单模型')
        axes[0].axhline(y=results_df.iloc[0]['mae'], color='red', linestyle='--', linewidth=2, label=f'最佳集成 ({results_df.iloc[0]["method"]})')
        axes[0].axhline(y=np.mean(single_model_maes), color='green', linestyle='--', linewidth=2, label='单模型平均')
        axes[0].set_title('单模型 vs 集成模型 MAE 对比')
        axes[0].set_xlabel('窗口 ID')
        axes[0].set_ylabel('MAE')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)

        # 不同集成方式对比
        top_results = results_df.head(10)
        axes[1].barh(top_results['method'], top_results['mae'], alpha=0.7)
        axes[1].set_title('不同集成方式 MAE 对比')
        axes[1].set_xlabel('MAE')
        axes[1].grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "ensemble_comparison.png", dpi=150, bbox_inches='tight')
        logger.info(f"\n对比图已保存到 {output_dir / 'ensemble_comparison.png'}")

    except ImportError:
        logger.warning("matplotlib 未安装，跳过绘图")

    # 11. 输出最终集成测试集 Loss
    logger.info("\n" + "=" * 60)
    logger.info("最终集成测试集 Loss")
    logger.info("=" * 60)
    logger.info(f"最佳集成方法: {best_method}")
    logger.info(f"测试集 MAE:  {results_df.iloc[0]['mae']:.6f}")
    logger.info(f"测试集 RMSE: {results_df.iloc[0]['rmse']:.6f}")
    logger.info(f"测试集 Bias: {results_df.iloc[0]['bias']:.6f}")
    logger.info(f"测试集 Median_AE: {results_df.iloc[0]['median_ae']:.6f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
