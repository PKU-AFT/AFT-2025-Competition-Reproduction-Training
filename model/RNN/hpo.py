"""
超参数优化框架（增强版 v4.0 - 固定窗口模式）

支持多种调优策略：
1. optuna - 贝叶斯优化 (TPE，推荐)
2. random - 随机搜索
3. grid - 网格搜索

当前搜索目标：基础 GRU 模型
- 模型类型：gru
- 搜索参数：hidden_size, num_layers, dropout, learning_rate, weight_decay, batch_days, step_len, grad_clip

v4.0 更新：
1. ✅ 固定窗口模式 - 前320天训练 + 后80天验证 + 最后81天测试
2. ✅ 消除窗口选择误差 - 所有试验使用相同数据划分
3. ✅ Optuna + TPE 采样 - 10-100x 效率提升
4. ✅ 增量剪枝 - 50-80% 时间节省
5. ✅ TensorBoard 可视化 - 实时监控训练曲线
6. ✅ 断点续传 - 支持恢复优化

数据划分:
    训练集: 前 320 个交易日
    验证集: 接下来 80 个交易日
    测试集: 最后 81 个交易日

使用方式:
    # Optuna 贝叶斯优化（推荐）
    python hpo.py --method optuna --max-trials 100

    # 随机搜索
    python hpo.py --method random --max-trials 20

    # 指定数据划分
    python hpo.py --method optuna --max-trials 100 --train-days 320 --val-days 80 --test-days 81

    # 断点续传
    python hpo.py --method optuna --max-trials 100 --storage sqlite:///hpo.db

    # 查看可视化结果
    tensorboard --logdir output/hpo/GRU/tensorboard
"""

import argparse
import copy
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import random
from itertools import product

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from scipy import stats
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns

# Optuna 导入（可选）
try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

import config as cfg
from dataset import (
    read_data,
    generate_rolling_windows,
    build_datasets_for_window,
    make_loader,
    make_test_loader,
)
from model import create_model, list_models
from utils import set_seed, ensure_dirs, get_logger
from losses import create_loss, get_loss_info


# =============================================================================
# 超参数搜索空间定义
# =============================================================================

@dataclass
class SearchSpace:
    """超参数搜索空间定义"""

    # 模型架构 - 仅基础 GRU
    model_type: List[str] = field(default_factory=lambda: ["gru"])

    # GRU 网络结构
    hidden_size: List[int] = field(default_factory=lambda: [64, 128, 256])  # 移除512，减少显存
    num_layers: List[int] = field(default_factory=lambda: [1, 2, 3])  # 移除4层
    dropout: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.5])

    # 训练超参数
    learning_rate: List[float] = field(default_factory=lambda: [5e-5, 1e-4, 5e-4, 1e-3, 5e-3])
    weight_decay: List[float] = field(default_factory=lambda: [0, 1e-6, 1e-5, 1e-4, 1e-3])
    batch_days: List[int] = field(default_factory=lambda: [256])  # 从512降到256，减少显存

    # 序列相关
    step_len: List[int] = field(default_factory=lambda: [5, 7, 10, 15, 20])  # 移除30

    # 梯度裁剪
    grad_clip: List[float] = field(default_factory=lambda: [0.5, 1.0, 5.0])

    # 保留兼容性（暂不使用）
    attention_heads: List[int] = field(default_factory=lambda: [1])
    attention_dropout: List[float] = field(default_factory=lambda: [0.0])
    tcn_kernel_size: List[int] = field(default_factory=lambda: [3, 5, 7])


# 默认搜索空间
DEFAULT_SEARCH_SPACE = SearchSpace()


# =============================================================================
# 配置生成器
# =============================================================================

class TrialConfig:
    """单次试验配置"""

    def __init__(self, **kwargs):
        # 从默认配置复制
        self.learning_rate = cfg.LEARNING_RATE
        self.weight_decay = cfg.WEIGHT_DECAY
        self.hidden_size = cfg.HIDDEN_SIZE
        self.num_layers = cfg.NUM_LAYERS
        self.dropout = cfg.DROPOUT
        self.model_type = cfg.MODEL_TYPE
        self.batch_days = cfg.BATCH_DAYS
        self.step_len = cfg.STEP_LEN
        self.tcn_kernel_size = cfg.TCN_KERNEL_SIZE

        # 新增：GRU 变体参数
        self.attention_heads = 1
        self.attention_dropout = 0.0
        self.grad_clip = cfg.GRAD_CLIP_NORM if hasattr(cfg, 'GRAD_CLIP_NORM') else 1.0

        # 覆盖试验参数
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # 同步大写属性 (模型创建需要)
        self._sync_uppercase_attrs()

    def _sync_uppercase_attrs(self):
        """同步大写属性，供 create_model 使用"""
        self.HIDDEN_SIZE = self.hidden_size
        self.NUM_LAYERS = self.num_layers
        self.DROPOUT = self.dropout
        self.TCN_KERNEL_SIZE = self.tcn_kernel_size
        self.NHEAD = getattr(cfg, 'NHEAD', 4)  # Transformer 专用
        self.DIM_FEEDFORWARD = getattr(cfg, 'DIM_FEEDFORWARD', None)
        self.BATCH_DAYS = self.batch_days
        self.STEP_LEN = self.step_len

    def __setattr__(self, name, value):
        """拦截属性设置，同步大写版本"""
        super().__setattr__(name, value)
        # 同步大写属性
        if name in ('hidden_size', 'num_layers', 'dropout', 'tcn_kernel_size', 'batch_days', 'step_len'):
            upper_name = name.upper()
            super().__setattr__(upper_name, value)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'model_type': self.model_type,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'batch_days': self.batch_days,
            'step_len': self.step_len,
            'attention_heads': self.attention_heads,
            'attention_dropout': self.attention_dropout,
            'grad_clip': self.grad_clip,
            'tcn_kernel_size': self.tcn_kernel_size,
        }

    def __repr__(self) -> str:
        return f"TrialConfig({self.to_dict()})"


# =============================================================================
# 调优策略
# =============================================================================

class SearchStrategy:
    """搜索策略基类"""

    def __init__(self, search_space: SearchSpace, seed: int = 42):
        self.search_space = search_space
        self.seed = seed
        set_seed(seed)

    def suggest(self) -> TrialConfig:
        """建议下一个试验配置"""
        raise NotImplementedError


class GridSearch(SearchStrategy):
    """网格搜索 - 遍历所有组合"""

    def __init__(self, search_space: SearchSpace, seed: int = 42):
        super().__init__(search_space, seed)
        self.indices = {}
        self.keys = ['model_type', 'hidden_size', 'num_layers', 'dropout',
                     'learning_rate', 'weight_decay', 'batch_days']
        for key in self.keys:
            self.indices[key] = 0
        self.total_trials = self._calculate_total()

    def _calculate_total(self) -> int:
        total = 1
        for key in self.keys:
            values = getattr(self.search_space, key)
            total *= len(values)
        return total

    def suggest(self) -> Optional[TrialConfig]:
        """生成下一个配置"""
        # 检查是否还有更多组合
        for key in self.keys:
            values = getattr(self.search_space, key)
            if self.indices[key] >= len(values):
                return None

        # 生成当前配置
        config = TrialConfig(
            model_type=self.search_space.model_type[self.indices['model_type']],
            hidden_size=self.search_space.hidden_size[self.indices['hidden_size']],
            num_layers=self.search_space.num_layers[self.indices['num_layers']],
            dropout=self.search_space.dropout[self.indices['dropout']],
            learning_rate=self.search_space.learning_rate[self.indices['learning_rate']],
            weight_decay=self.search_space.weight_decay[self.indices['weight_decay']],
            batch_days=self.search_space.batch_days[self.indices['batch_days']],
        )

        # 递增索引
        for key in reversed(self.keys):
            self.indices[key] += 1
            if self.indices[key] < len(getattr(self.search_space, key)):
                break
            self.indices[key] = 0

        return config


class RandomSearch(SearchStrategy):
    """随机搜索 - 随机采样"""

    def suggest(self) -> TrialConfig:
        """随机生成配置"""
        return TrialConfig(
            model_type=random.choice(self.search_space.model_type),
            hidden_size=random.choice(self.search_space.hidden_size),
            num_layers=random.choice(self.search_space.num_layers),
            dropout=random.choice(self.search_space.dropout),
            learning_rate=random.choice(self.search_space.learning_rate),
            weight_decay=random.choice(self.search_space.weight_decay),
            batch_days=random.choice(self.search_space.batch_days),
            step_len=random.choice(self.search_space.step_len),
            tcn_kernel_size=random.choice(self.search_space.tcn_kernel_size),
            attention_heads=random.choice(self.search_space.attention_heads),
            attention_dropout=random.choice(self.search_space.attention_dropout),
            grad_clip=random.choice(self.search_space.grad_clip),
        )


class OptunaSearch(SearchStrategy):
    """Optuna 贝叶斯优化 - TPE 采样"""

    def __init__(self, search_space: SearchSpace, seed: int = 42,
                 storage: Optional[str] = None, study_name: str = "hpo",
                 n_startup_trials: int = 5, n_warmup_steps: int = 5,
                 multivariate: bool = True):
        super().__init__(search_space, seed)
        self.storage = storage
        self.study_name = study_name
        self.n_startup_trials = n_startup_trials
        self.n_warmup_steps = n_warmup_steps
        self.multivariate = multivariate

        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna 未安装，请运行: pip install optuna")

        # 创建 Study
        self.study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            sampler=TPESampler(seed=seed, multivariate=multivariate),
            pruner=MedianPruner(n_startup_trials=n_startup_trials, n_warmup_steps=n_warmup_steps),
            storage=storage,
            load_if_exists=True,
        )

    def suggest(self, trial=None) -> Optional[TrialConfig]:
        """生成下一个配置"""
        if trial is None:
            # 非集成模式，创建临时 trial
            trial = self.study.ask()

        # 模型类型选择
        model_type = trial.suggest_categorical("model_type", self.search_space.model_type)

        # 网络结构
        hidden_size = trial.suggest_categorical("hidden_size",
            [int(x) for x in self.search_space.hidden_size])
        num_layers = trial.suggest_int("num_layers",
            min(self.search_space.num_layers),
            max(self.search_space.num_layers))
        dropout = trial.suggest_float("dropout",
            min(self.search_space.dropout),
            max(self.search_space.dropout))

        # 注意力参数（条件采样）
        attention_heads = 1
        attention_dropout = 0.0
        if "attn" in model_type:
            attention_heads = trial.suggest_categorical("attention_heads",
                [int(x) for x in self.search_space.attention_heads])
            attention_dropout = trial.suggest_float("attention_dropout",
                min(self.search_space.attention_dropout),
                max(self.search_space.attention_dropout))

        # 训练超参数（对数尺度）
        lr_min, lr_max = min(self.search_space.learning_rate), max(self.search_space.learning_rate)
        learning_rate = trial.suggest_float("learning_rate", lr_min, lr_max, log=True)

        # weight_decay 可能包含 0，特殊处理
        wd_values = self.search_space.weight_decay
        if 0 in wd_values:
            # 过滤掉 0，使用最小正数作为下界
            wd_positive = [x for x in wd_values if x > 0]
            if wd_positive:
                wd_min = min(wd_positive)
                wd_max = max(wd_positive)
                weight_decay = trial.suggest_float("weight_decay", wd_min, wd_max, log=True)
            else:
                weight_decay = 0.0
        else:
            wd_min, wd_max = min(wd_values), max(wd_values)
            weight_decay = trial.suggest_float("weight_decay", wd_min, wd_max, log=True)

        # 序列和批处理
        batch_days = trial.suggest_categorical("batch_days",
            [int(x) for x in self.search_space.batch_days])
        step_len = trial.suggest_categorical("step_len",
            [int(x) for x in self.search_space.step_len])

        # 梯度裁剪
        grad_clip = trial.suggest_categorical("grad_clip",
            [float(x) for x in self.search_space.grad_clip])

        # TCN 参数（兼容）
        tcn_kernel_size = trial.suggest_categorical("tcn_kernel_size",
            [int(x) for x in self.search_space.tcn_kernel_size])

        return TrialConfig(
            model_type=model_type,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            attention_heads=attention_heads,
            attention_dropout=attention_dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_days=batch_days,
            step_len=step_len,
            grad_clip=grad_clip,
            tcn_kernel_size=tcn_kernel_size,
        )

    def tell(self, trial, value: float):
        """报告试验结果"""
        if value is not None and not (isinstance(value, float) and value == float("inf")):
            self.study.tell(trial, value)
        else:
            # 失败的试验
            self.study.tell(trial, float("inf"))

    def get_best_trial(self):
        """获取最佳试验"""
        return self.study.best_trial

    def get_trials_df(self):
        """获取所有试验作为 DataFrame"""
        return self.study.trials_dataframe()

    def get_param_importance(self):
        """获取参数重要性"""
        return optuna.importance.get_param_importances(self.study)


# =============================================================================
# 试验管理器
# =============================================================================

@dataclass
class TrialResult:
    """试验结果"""
    trial_id: int
    config: Dict[str, Any]
    val_mae: float
    test_mae: Optional[float] = None
    train_time: float = 0.0
    epoch: int = 0
    timestamp: str = ""
    # Multi-window support
    val_mae_windows: List[float] = field(default_factory=list)
    # Training curve for TensorBoard
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)


class ExperimentTracker:
    """实验跟踪器"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.results: List[TrialResult] = []
        output_dir.mkdir(parents=True, exist_ok=True)

        # CSV 文件
        self.csv_path = output_dir / "trials.csv"
        self.json_path = output_dir / "trials.json"

        # TensorBoard writer
        self.tb_dir = output_dir / "tensorboard"
        self.tb_writer = SummaryWriter(log_dir=str(self.tb_dir))

    def add_result(self, result: TrialResult):
        """添加试验结果"""
        self.results.append(result)

        # Log to TensorBoard
        for key, value in result.config.items():
            self.tb_writer.add_text(
                f"trial_{result.trial_id}/config",
                f"{key}: {value}",
                global_step=result.trial_id
            )

        self.tb_writer.add_scalar(
            "hyperparameters/val_mae",
            result.val_mae,
            result.trial_id
        )
        self.tb_writer.add_scalar(
            "hyperparameters/test_mae",
            result.test_mae or 0.0,
            result.trial_id
        )
        self.tb_writer.add_scalar(
            "hyperparameters/train_time",
            result.train_time,
            result.trial_id
        )

        # Log training curves
        for epoch, (train_loss, val_loss) in enumerate(zip(result.train_losses, result.val_losses)):
            self.tb_writer.add_scalars(
                f"trial_{result.trial_id}/loss",
                {'train': train_loss, 'val': val_loss},
                epoch
            )

        self.tb_writer.flush()
        self._save()

    def _save(self):
        """保存结果到文件"""
        # 保存 CSV
        df = pd.DataFrame([{
            'trial_id': r.trial_id,
            'val_mae': r.val_mae,
            'test_mae': r.test_mae,
            'train_time': r.train_time,
            'epoch': r.epoch,
            **r.config
        } for r in self.results])
        df.to_csv(self.csv_path, index=False)

        # 保存 JSON
        with open(self.json_path, 'w') as f:
            json.dump([{
                'trial_id': r.trial_id,
                'config': r.config,
                'val_mae': r.val_mae,
                'test_mae': r.test_mae,
                'train_time': r.train_time,
                'epoch': r.epoch,
                'timestamp': r.timestamp,
                'val_mae_windows': r.val_mae_windows,
            } for r in self.results], f, indent=2)

    def get_best(self) -> Optional[TrialResult]:
        """获取最佳试验"""
        if not self.results:
            return None
        return min(self.results, key=lambda r: r.val_mae)

    def print_summary(self):
        """打印摘要"""
        if not self.results:
            print("还没有试验结果")
            return

        print("\n" + "=" * 60)
        print("实验摘要")
        print("=" * 60)
        print(f"总试验数: {len(self.results)}")

        best = self.get_best()
        print(f"\n最佳试验 (Trial {best.trial_id}):")
        print(f"  Val MAE: {best.val_mae:.6f}")
        if best.val_mae_windows:
            print(f"  各窗口 MAE: {best.val_mae_windows}")
        print(f"  配置: {json.dumps(best.config, indent=4)}")

        # Top 5
        sorted_results = sorted(self.results, key=lambda r: r.val_mae)[:5]
        print(f"\nTop 5 试验:")
        for r in sorted_results:
            print(f"  Trial {r.trial_id}: Val MAE = {r.val_mae:.6f}, Epoch = {r.epoch}")

        # TensorBoard 提示
        print(f"\n📊 TensorBoard 可视化:")
        print(f"   tensorboard --logdir {self.tb_dir}")

    def analyze_parameter_importance(self):
        """分析参数重要性（使用方差分析 F-test）"""
        if len(self.results) < 5:
            print("\n⚠️  样本量不足，无法进行参数重要性分析（需要至少 5 个试验）")
            return

        df = pd.DataFrame([{
            **r.config,
            'val_mae': r.val_mae
        } for r in self.results])

        print("\n" + "=" * 60)
        print("参数重要性分析 (ANOVA F-test)")
        print("=" * 60)

        importance_scores = {}

        for param in df.columns:
            if param == 'val_mae':
                continue

            # Group by parameter value
            groups = df.groupby(param)['val_mae'].apply(list)

            if len(groups) < 2:
                continue

            # Perform one-way ANOVA
            try:
                f_stat, p_value = stats.f_oneway(*groups.values)

                # Calculate effect size (eta-squared)
                grand_mean = df['val_mae'].mean()
                ss_between = sum([len(g) * (np.mean(g) - grand_mean)**2 for g in groups.values])
                ss_total = sum((df['val_mae'] - grand_mean)**2)
                eta_squared = ss_between / ss_total if ss_total > 0 else 0

                importance_scores[param] = {
                    'f_stat': f_stat,
                    'p_value': p_value,
                    'eta_squared': eta_squared,
                    'importance': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
                }
            except Exception as e:
                print(f"  ⚠️  {param}: 分析失败 - {e}")

        # Sort by eta-squared
        sorted_params = sorted(importance_scores.items(), key=lambda x: x[1]['eta_squared'], reverse=True)

        print(f"\n{'参数':<20} {'F-stat':<10} {'p-value':<12} {'η²':<10} {'重要性'}")
        print("-" * 70)
        for param, scores in sorted_params:
            print(f"{param:<20} {scores['f_stat']:<10.3f} {scores['p_value']:<12.6f} {scores['eta_squared']:<10.4f} {scores['importance']}")

        # Save importance plot
        self._plot_parameter_importance(importance_scores)

    def _plot_parameter_importance(self, importance_scores):
        """绘制参数重要性图"""
        try:
            params = list(importance_scores.keys())
            scores = [importance_scores[p]['eta_squared'] for p in params]

            plt.figure(figsize=(10, 6))
            sorted_idx = np.argsort(scores)[::-1]
            plt.barh(range(len(params)), np.array(scores)[sorted_idx])
            plt.yticks(range(len(params)), np.array(params)[sorted_idx])
            plt.xlabel('Eta-squared (Effect Size)')
            plt.title('Parameter Importance Analysis')
            plt.tight_layout()

            plot_path = self.output_dir / "parameter_importance.png"
            plt.savefig(plot_path, dpi=150)
            print(f"\n📈 参数重要性图已保存: {plot_path}")
            plt.close()
        except Exception as e:
            print(f"\n⚠️  绘图失败: {e}")

    def plot_comparison_matrix(self):
        """绘制参数对比热力图"""
        if len(self.results) < 5:
            return

        try:
            df = pd.DataFrame([{
                **r.config,
                'val_mae': r.val_mae
            } for r in self.results])

            # 只选择分类参数
            categorical_params = []
            for col in df.columns:
                if col == 'val_mae':
                    continue
                if df[col].nunique() <= 5:  # 最多 5 个不同值
                    categorical_params.append(col)

            if len(categorical_params) < 2:
                return

            # 绘制热力图矩阵
            fig, axes = plt.subplots(1, min(3, len(categorical_params)), figsize=(15, 4))
            if len(categorical_params) == 1:
                axes = [axes]

            for idx, param in enumerate(categorical_params[:3]):
                ax = axes[idx]
                pivot_table = df.groupby(param)['val_mae'].agg(['mean', 'std', 'count'])
                pivot_table.plot(kind='bar', y='mean', yerr='std', ax=ax, capsize=3)
                ax.set_title(f'{param} vs Val MAE')
                ax.set_ylabel('Val MAE')
                ax.set_xlabel(param)
                ax.tick_params(axis='x', rotation=45)

            plt.tight_layout()
            plot_path = self.output_dir / "parameter_comparison.png"
            plt.savefig(plot_path, dpi=150)
            print(f"📊 参数对比图已保存: {plot_path}")
            plt.close()
        except Exception as e:
            print(f"⚠️  参数对比图绘制失败: {e}")

    def close(self):
        """关闭 TensorBoard writer"""
        if hasattr(self, 'tb_writer'):
            self.tb_writer.close()


# =============================================================================
# 简化训练器（用于 HPO）
# =============================================================================

def _maybe_denorm(t: torch.Tensor, label_scaler):
    """逆标准化（支持 Z-score 和 Rank Gauss）"""
    if label_scaler is None:
        return t

    scaler_type = label_scaler.get("type", "zscore")

    if scaler_type == "rank_gauss":
        sorted_values = label_scaler["sorted_values"]
        probabilities = label_scaler["probabilities"]

        f = interp1d(
            probabilities,
            sorted_values,
            kind='linear',
            bounds_error=False,
            fill_value=(sorted_values[0], sorted_values[-1])
        )

        if isinstance(t, torch.Tensor):
            t_np = t.cpu().numpy().flatten()
        else:
            t_np = np.array(t).flatten()

        percentiles = stats.norm.cdf(t_np)
        denormalized = f(percentiles)

        return torch.from_numpy(denormalized).to(t.device).reshape(t.shape) if isinstance(t, torch.Tensor) else denormalized
    else:
        if label_scaler.get("std") is None or label_scaler.get("mean") is None:
            return t
        return t * label_scaler["std"] + label_scaler["mean"]


def train_single_trial(
    config: TrialConfig,
    train_loader,
    val_loader,
    test_loader,
    device: str,
    logger=None,
    trial_early_stop_threshold: float = None,
    trial=None,  # Optuna trial（用于剪枝）
    label_scaler=None,  # 用于逆标准化
) -> TrialResult:
    """训练单次试验"""

    # 清理显存缓存
    if device == "cuda":
        torch.cuda.empty_cache()

    # 创建模型
    input_dim = train_loader.dataset.data_arr.shape[1]
    model = create_model(config.model_type, input_dim, config).to(device)

    # 优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # 混合精度
    use_amp = cfg.DEVICE == "cuda"
    from torch.amp import autocast, GradScaler
    amp_scaler = GradScaler() if use_amp else None

    # 创建损失函数
    loss_type = getattr(cfg, "LOSS_TYPE", "mae")
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

    loss_fn = create_loss(loss_type, **loss_params)
    use_adaptive_loss = hasattr(loss_fn, 'set_epoch')

    # 训练
    best_val = float("inf")
    best_epoch = 0
    start_time = time.time()

    # 记录训练曲线
    train_losses = []
    val_losses = []

    # Trial early stopping: 如果前几个 epoch 表现太差，提前终止
    early_stop_check_epochs = 5
    early_stop_baseline = None

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        # 对于自适应损失函数，设置当前epoch
        if use_adaptive_loss:
            loss_fn.set_epoch(epoch)

        # Train
        model.train()
        train_loss = 0.0
        train_count = 0

        for batch in train_loader:
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

            x = x.to(device)
            y = y.to(device).view(-1)
            mask = mask.to(device).view(-1)

            optimizer.zero_grad()
            with autocast(device_type='cuda' if use_amp else 'cpu', enabled=use_amp):
                pred = model(x)
                # 使用模块化的损失函数
                loss = loss_fn(pred, y, mask)

            if use_amp:
                amp_scaler.scale(loss).backward()
                amp_scaler.unscale_(optimizer)
                if config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                amp_scaler.step(optimizer)
                amp_scaler.update()
            else:
                loss.backward()
                if config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()

            # 使用MAE作为标准指标
            with torch.no_grad():
                mae = torch.abs(pred - y)
                if mask is not None:
                    mask_f = mask.float()
                    mae = mae * mask_f
                    denom = mask_f.sum().clamp_min(1.0)
                else:
                    denom = pred.numel()
                train_loss += mae.sum().item()
                train_count += denom.item()

        train_loss = train_loss / max(train_count, 1)
        train_losses.append(train_loss)

        # Val
        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
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

                x = x.to(device)
                y = y.to(device).view(-1)
                mask = mask.to(device).view(-1)

                pred = model(x)
                mask_f = mask.float()
                # 逆标准化后计算 MAE
                pred_denorm = _maybe_denorm(pred, label_scaler)
                y_denorm = _maybe_denorm(y, label_scaler)
                val_loss += torch.sum(torch.abs(pred_denorm - y_denorm) * mask_f).item()
                val_count += mask_f.sum().item()

        val_loss = val_loss / max(val_count, 1)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        # ===== Optuna 剪枝检查 =====
        if trial is not None:
            # 报告中间结果
            trial.report(val_loss, epoch)
            # 检查是否应该剪枝
            if trial.should_prune():
                if logger:
                    logger.info(f"  ⚠️  Trial 被 Optuna 剪枝于 epoch {epoch}, val_loss={val_loss:.6f}")
                raise optuna.TrialPruned()
        # ===================================

        # Trial early stopping check（传统方法）
        if trial_early_stop_threshold is not None:
            if epoch == early_stop_check_epochs:
                early_stop_baseline = val_loss
                if logger:
                    logger.info(f"  Early stop baseline (epoch {epoch}): {early_stop_baseline:.6f}")

            if epoch > early_stop_check_epochs and early_stop_baseline is not None:
                # 如果当前 loss 超过 baseline 的阈值倍数，提前终止
                if val_loss > early_stop_baseline * trial_early_stop_threshold:
                    if logger:
                        logger.info(f"  ⚠️  Trial early stopping: {val_loss:.6f} > {early_stop_baseline * trial_early_stop_threshold:.6f}")
                    break

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch

        if epoch - best_epoch >= cfg.EARLY_STOP_PATIENCE:
            break

    train_time = time.time() - start_time

    # Test
    model.eval()
    test_loss = 0.0
    test_count = 0
    with torch.no_grad():
        for batch in test_loader:
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

            x = x.to(device)
            y = y.to(device).view(-1)
            mask = mask.to(device).view(-1)

            pred = model(x)
            mask_f = mask.float()
            # 逆标准化后计算 MAE
            pred_denorm = _maybe_denorm(pred, label_scaler)
            y_denorm = _maybe_denorm(y, label_scaler)
            test_loss += torch.sum(torch.abs(pred_denorm - y_denorm) * mask_f).item()
            test_count += mask_f.sum().item()

    test_loss = test_loss / max(test_count, 1) if test_count > 0 else None

    if logger:
        logger.info(f"Trial 完成: Val MAE={best_val:.6f}, Test MAE={test_loss:.6f}, Epoch={best_epoch}, Time={train_time:.1f}s")

    return TrialResult(
        trial_id=0,  # 由外部设置
        config=config.to_dict(),
        val_mae=best_val,
        test_mae=test_loss,
        train_time=train_time,
        epoch=best_epoch,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        train_losses=train_losses,
        val_losses=val_losses,
    )


# =============================================================================
# 主流程
# =============================================================================

def run_optuna_hpo(search: OptunaSearch, validation_windows: List[int],
                    window_datasets: Dict, tracker: ExperimentTracker,
                    logger, max_trials: int, trial_early_stop_threshold: float = None):
    """使用 Optuna 运行超参数优化"""

    def objective_wrapper(trial):
        """Optuna 目标函数包装器"""
        # 生成配置
        config = search.suggest(trial)

        logger.info(f"\n{'='*60}")
        logger.info(f"Trial {trial.number}: {config.model_type}, "
                   f"hs={config.hidden_size}, layers={config.num_layers}, "
                   f"lr={config.learning_rate:.2e}, drop={config.dropout:.2f}")
        logger.info(f"{'='*60}")

        # 更新全局配置
        original_batch_days = cfg.BATCH_DAYS
        original_step_len = cfg.STEP_LEN
        cfg.BATCH_DAYS = config.batch_days
        cfg.STEP_LEN = config.step_len

        try:
            # Multi-window validation
            val_mae_windows = []
            total_train_time = 0
            first_result = None

            for win_id in validation_windows:
                if win_id not in window_datasets:
                    continue

                win_data = window_datasets[win_id]

                # 重新创建数据加载器
                train_loader = make_loader(win_data['train_ds'], cfg, shuffle=True)
                val_loader = make_loader(win_data['val_ds'], cfg, shuffle=False)
                test_loader = make_test_loader(win_data['test_ds'], cfg)

                logger.info(f"  训练窗口 {win_id}...")

                result = train_single_trial(
                    config, train_loader, val_loader, test_loader,
                    cfg.DEVICE, logger, trial_early_stop_threshold, trial
                )

                if first_result is None:
                    first_result = result

                val_mae_windows.append(result.val_mae)
                total_train_time += result.train_time

                logger.info(f"  窗口 {win_id} 完成: Val MAE={result.val_mae:.6f}, Epoch={result.epoch}")

            # Compute average validation MAE across windows
            avg_val_mae = np.mean(val_mae_windows)
            std_val_mae = np.std(val_mae_windows)

            logger.info(f"  多窗口平均: Val MAE={avg_val_mae:.6f} (+/- {std_val_mae:.6f})")

            # 记录结果
            if first_result:
                first_result.val_mae = avg_val_mae
                first_result.val_mae_windows = val_mae_windows
                first_result.train_time = total_train_time
                first_result.trial_id = trial.number

                tracker.add_result(first_result)

            return avg_val_mae

        except optuna.TrialPruned:
            logger.info(f"  Trial {trial.number} 被剪枝")
            raise
        except RuntimeError as e:
            # 捕获 CUDA OOM 错误
            if "out of memory" in str(e) or "CUDA out of memory" in str(e):
                logger.warning(f"Trial {trial.number} 显存溢出 (OOM): hs={config.hidden_size}, layers={config.num_layers}, batch_days={config.batch_days}")
                # 清理显存
                if cfg.DEVICE == "cuda":
                    torch.cuda.empty_cache()
                return float("inf")
            else:
                logger.error(f"Trial {trial.number} 失败: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return float("inf")
        except Exception as e:
            logger.error(f"Trial {trial.number} 失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return float("inf")
        finally:
            # 恢复配置
            cfg.BATCH_DAYS = original_batch_days
            cfg.STEP_LEN = original_step_len

    # 运行优化
    search.study.optimize(objective_wrapper, n_trials=max_trials)


def run_traditional_hpo(search: SearchStrategy, validation_windows: List[int],
                        window_datasets: Dict, tracker: ExperimentTracker,
                        logger, max_trials: int, trial_early_stop_threshold: float = None):
    """使用传统方法（Grid/Random）运行超参数优化"""

    trial_id = 0
    while trial_id < max_trials:
        config = search.suggest()
        if config is None:
            logger.info("已遍历所有搜索空间组合")
            break

        trial_id += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"Trial {trial_id}: {config}")
        logger.info(f"{'='*60}")

        # 更新全局配置
        original_batch_days = cfg.BATCH_DAYS
        original_step_len = cfg.STEP_LEN
        cfg.BATCH_DAYS = config.batch_days
        cfg.STEP_LEN = config.step_len

        try:
            # Multi-window validation
            val_mae_windows = []
            total_train_time = 0
            first_result = None

            for win_id in validation_windows:
                if win_id not in window_datasets:
                    continue

                win_data = window_datasets[win_id]

                # 重新创建数据加载器
                train_loader = make_loader(win_data['train_ds'], cfg, shuffle=True)
                val_loader = make_loader(win_data['val_ds'], cfg, shuffle=False)
                test_loader = make_test_loader(win_data['test_ds'], cfg)

                logger.info(f"  训练窗口 {win_id}...")
                result = train_single_trial(
                    config, train_loader, val_loader, test_loader,
                    cfg.DEVICE, logger, trial_early_stop_threshold, trial=None
                )

                if first_result is None:
                    first_result = result

                val_mae_windows.append(result.val_mae)
                total_train_time += result.train_time

                logger.info(f"  窗口 {win_id} 完成: Val MAE={result.val_mae:.6f}, Epoch={result.epoch}")

            # Compute average validation MAE across windows
            avg_val_mae = np.mean(val_mae_windows)
            std_val_mae = np.std(val_mae_windows)

            # Update result with multi-window metrics
            if first_result:
                first_result.val_mae = avg_val_mae
                first_result.val_mae_windows = val_mae_windows
                first_result.train_time = total_train_time
                first_result.trial_id = trial_id

                logger.info(f"  多窗口平均: Val MAE={avg_val_mae:.6f} (+/- {std_val_mae:.6f})")

                tracker.add_result(first_result)

        except Exception as e:
            logger.error(f"Trial {trial_id} 失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            # 恢复配置
            cfg.BATCH_DAYS = original_batch_days
            cfg.STEP_LEN = original_step_len


def run_optuna_hpo_fixed(search: OptunaSearch, train_ds, val_ds, test_ds,
                          scaler, label_scaler, features, target, window,
                          tracker: ExperimentTracker, logger, max_trials: int):
    """使用 Optuna 运行超参数优化 (固定窗口模式)"""

    def objective_wrapper(trial):
        """Optuna 目标函数包装器"""
        # 生成配置
        config = search.suggest(trial)

        logger.info(f"\n{'='*60}")
        logger.info(f"Trial {trial.number}: {config.model_type}, "
                   f"hs={config.hidden_size}, layers={config.num_layers}, "
                   f"lr={config.learning_rate:.2e}, drop={config.dropout:.2f}")
        logger.info(f"{'='*60}")

        # 更新全局配置
        original_batch_days = cfg.BATCH_DAYS
        original_step_len = cfg.STEP_LEN
        cfg.BATCH_DAYS = config.batch_days
        cfg.STEP_LEN = config.step_len

        try:
            # 重新创建数据加载器
            train_loader = make_loader(train_ds, cfg, shuffle=True)
            val_loader = make_loader(val_ds, cfg, shuffle=False)
            test_loader = make_loader(test_ds, cfg, shuffle=False)

            # 训练
            result = train_single_trial(
                config, train_loader, val_loader, test_loader,
                cfg.DEVICE, logger, trial_early_stop_threshold=None, trial=trial,
                label_scaler=label_scaler
            )

            result.trial_id = trial.number
            tracker.add_result(result)

            logger.info(f"Trial {trial.number} 完成: Val MAE={result.val_mae:.6f}, "
                       f"Test MAE={result.test_mae:.6f}, Epoch={result.epoch}")

            return result.val_mae

        except optuna.TrialPruned:
            logger.info(f"  Trial {trial.number} 被剪枝")
            raise
        except RuntimeError as e:
            # 捕获 CUDA OOM 错误
            if "out of memory" in str(e) or "CUDA out of memory" in str(e):
                logger.warning(f"Trial {trial.number} 显存溢出 (OOM): hs={config.hidden_size}, layers={config.num_layers}, batch_days={config.batch_days}")
                # 清理显存
                if cfg.DEVICE == "cuda":
                    torch.cuda.empty_cache()
                return float("inf")
            else:
                logger.error(f"Trial {trial.number} 失败: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return float("inf")
        except Exception as e:
            logger.error(f"Trial {trial.number} 失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return float("inf")
        finally:
            # 恢复配置
            cfg.BATCH_DAYS = original_batch_days
            cfg.STEP_LEN = original_step_len

    # 运行优化
    search.study.optimize(objective_wrapper, n_trials=max_trials)


def run_traditional_hpo_fixed(search: SearchStrategy, train_ds, val_ds, test_ds,
                               scaler, label_scaler, features, target, window,
                               tracker: ExperimentTracker, logger, max_trials: int):
    """使用传统方法（Grid/Random）运行超参数优化 (固定窗口模式)"""

    trial_id = 0
    while trial_id < max_trials:
        config = search.suggest()
        if config is None:
            logger.info("已遍历所有搜索空间组合")
            break

        trial_id += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"Trial {trial_id}: {config}")
        logger.info(f"{'='*60}")

        # 更新全局配置
        original_batch_days = cfg.BATCH_DAYS
        original_step_len = cfg.STEP_LEN
        cfg.BATCH_DAYS = config.batch_days
        cfg.STEP_LEN = config.step_len

        try:
            # 重新创建数据加载器
            train_loader = make_loader(train_ds, cfg, shuffle=True)
            val_loader = make_loader(val_ds, cfg, shuffle=False)
            test_loader = make_loader(test_ds, cfg, shuffle=False)

            # 训练
            result = train_single_trial(
                config, train_loader, val_loader, test_loader,
                cfg.DEVICE, logger, trial_early_stop_threshold=None, trial=None,
                label_scaler=label_scaler
            )

            result.trial_id = trial_id
            tracker.add_result(result)

            logger.info(f"Trial {trial_id} 完成: Val MAE={result.val_mae:.6f}, "
                       f"Test MAE={result.test_mae:.6f}, Epoch={result.epoch}")

        except Exception as e:
            logger.error(f"Trial {trial_id} 失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            # 恢复配置
            cfg.BATCH_DAYS = original_batch_days
            cfg.STEP_LEN = original_step_len


def create_fixed_window(features, target, train_days=320, val_days=80, test_days=81):
    """
    创建固定窗口划分

    按时间顺序划分数据：
    - 训练集: 前 train_days 个交易日
    - 验证集: 接下来 val_days 个交易日
    - 测试集: 最后 test_days 个交易日

    Returns:
        window: dict with train_start, train_end, val_start, val_end, test_start, test_end
    """
    # 提取日期信息
    time_ids = features.index.get_level_values('time_id').unique()
    date_ids = (time_ids // 10000).unique()
    date_ids = np.sort(date_ids)

    total_days = len(date_ids)
    required_days = train_days + val_days + test_days

    if total_days < required_days:
        raise ValueError(f"数据不足: 需要 {required_days} 天，实际只有 {total_days} 天")

    print(f"总天数: {total_days}, 使用: {required_days} 天")

    # 计算边界
    train_end_idx = train_days - 1
    val_end_idx = train_days + val_days - 1
    test_end_idx = total_days - 1  # 使用最后所有天数作为测试

    # 获取对应的时间ID边界
    train_end_date = date_ids[train_end_idx]
    val_start_date = date_ids[train_end_idx + 1]
    val_end_date = date_ids[val_end_idx]
    test_start_date = date_ids[val_end_idx + 1]

    # 获取时间ID边界
    train_mask = (time_ids // 10000) <= train_end_date
    val_mask = ((time_ids // 10000) >= val_start_date) & ((time_ids // 10000) <= val_end_date)
    test_mask = (time_ids // 10000) >= test_start_date

    train_times = time_ids[train_mask]
    val_times = time_ids[val_mask]
    test_times = time_ids[test_mask]

    window = {
        'train_start': int(train_times.min()),
        'train_end': int(train_times.max()),
        'val_start': int(val_times.min()),
        'val_end': int(val_times.max()),
        'test_start': int(test_times.min()),
        'test_end': int(test_times.max()),
        'fixed_window': True,
        'train_days': train_days,
        'val_days': val_days,
        'test_days': test_days,
    }

    return window


def main():
    parser = argparse.ArgumentParser(description="超参数优化 - 固定窗口模式")
    parser.add_argument("--method", choices=["grid", "random", "optuna"], default="optuna",
                        help="搜索方法 (optuna=贝叶斯优化, random=随机搜索, grid=网格搜索)")
    parser.add_argument("--max-trials", type=int, default=100,
                        help="最大试验次数")
    parser.add_argument("--output", type=str, default="output/hpo",
                        help="输出目录")

    # 固定窗口参数
    parser.add_argument("--train-days", type=int, default=320,
                        help="训练集天数 (默认: 320)")
    parser.add_argument("--val-days", type=int, default=80,
                        help="验证集天数 (默认: 80)")
    parser.add_argument("--test-days", type=int, default=81,
                        help="测试集天数 (默认: 81)")

    # Optuna 参数
    parser.add_argument("--storage", type=str, default=None,
                        help="Optuna 存储路径 (如 sqlite:///hpo.db)，用于断点续传")
    parser.add_argument("--study-name", type=str, default="gru_hpo_fixed",
                        help="Optuna study 名称")
    parser.add_argument("--n-startup-trials", type=int, default=5,
                        help="Optuna 剪枝启动前试验数")
    parser.add_argument("--n-warmup-steps", type=int, default=5,
                        help="Optuna 剪枝 warmup 步数")

    args = parser.parse_args()

    set_seed(cfg.SEED)

    # 创建输出目录
    model_type = cfg.MODEL_TYPE.upper().replace("-", "_")
    output_dir = Path(args.output) / model_type
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(output_dir / "hpo.log")
    logger.info("=" * 60)
    logger.info("超参数优化启动 (固定窗口模式 v4.0)")
    logger.info("=" * 60)
    logger.info(f"方法: {args.method.upper()}")
    logger.info(f"模型类型: {model_type}")
    logger.info(f"最大试验次数: {args.max_trials}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"数据划分: 训练 {args.train_days}天 | 验证 {args.val_days}天 | 测试 {args.test_days}天")

    if args.method == "optuna":
        logger.info(f"存储: {args.storage or '内存 (in-memory)'}")
        logger.info(f"剪枝: n_startup={args.n_startup_trials}, n_warmup={args.n_warmup_steps}")

    # 检查 Optuna
    if args.method == "optuna" and not OPTUNA_AVAILABLE:
        print("⚠️  Optuna 未安装，回退到 random search")
        print("   安装: pip install optuna")
        args.method = "random"

    # 创建搜索策略
    search_space = DEFAULT_SEARCH_SPACE

    if args.method == "optuna":
        search = OptunaSearch(
            search_space,
            seed=cfg.SEED,
            storage=args.storage,
            study_name=args.study_name,
            n_startup_trials=args.n_startup_trials,
            n_warmup_steps=args.n_warmup_steps,
        )
    elif args.method == "grid":
        search = GridSearch(search_space)
    else:
        search = RandomSearch(search_space)

    # 创建跟踪器
    tracker = ExperimentTracker(output_dir)

    # 加载数据
    logger.info("加载数据...")
    features, target = read_data(cfg)

    # 创建固定窗口
    logger.info("创建固定窗口划分...")
    window = create_fixed_window(features, target, args.train_days, args.val_days, args.test_days)

    logger.info(f"窗口划分:")
    logger.info(f"  训练集: {window['train_start']} ~ {window['train_end']} ({args.train_days}天)")
    logger.info(f"  验证集: {window['val_start']} ~ {window['val_end']} ({args.val_days}天)")
    logger.info(f"  测试集: {window['test_start']} ~ {window['test_end']} ({args.test_days}天)")

    # 构建数据集
    logger.info("构建数据集...")
    train_ds, val_ds, test_ds, scaler, label_scaler = build_datasets_for_window(
        features, target, cfg, window
    )
    logger.info(f"  训练集: {len(train_ds):,} 样本")
    logger.info(f"  验证集: {len(val_ds):,} 样本")
    logger.info(f"  测试集: {len(test_ds):,} 样本")

    # 运行试验
    if args.method == "optuna":
        run_optuna_hpo_fixed(
            search, train_ds, val_ds, test_ds, scaler, label_scaler,
            features, target, window, tracker, logger, args.max_trials
        )
    else:
        run_traditional_hpo_fixed(
            search, train_ds, val_ds, test_ds, scaler, label_scaler,
            features, target, window, tracker, logger, args.max_trials
        )

    # 打印摘要和分析
    tracker.print_summary()

    # 参数重要性分析
    if len(tracker.results) >= 5:
        tracker.analyze_parameter_importance()
        tracker.plot_comparison_matrix()

    # Optuna 特有：获取最佳参数和重要性
    if args.method == "optuna" and OPTUNA_AVAILABLE:
        print("\n" + "=" * 60)
        print("Optuna 最佳参数:")
        print("=" * 60)
        best_trial = search.get_best_trial()
        print(f"  Value (Val MAE): {best_trial.value:.6f}")
        print(f"  Params:")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")

        # 保存 Optuna 最佳配置
        best_config_path = output_dir / "best_optuna_config.json"
        with open(best_config_path, 'w') as f:
            json.dump({
                'value': best_trial.value,
                'params': best_trial.params,
                'data_split': {
                    'train_days': args.train_days,
                    'val_days': args.val_days,
                    'test_days': args.test_days,
                }
            }, f, indent=2)
        logger.info(f"Optuna 最佳配置已保存: {best_config_path}")

    # 保存最佳配置（兼容格式）
    best = tracker.get_best()
    if best:
        best_config_path = output_dir / "best_config.json"
        with open(best_config_path, 'w') as f:
            json.dump(best.config, f, indent=2)
        logger.info(f"最佳配置已保存到: {best_config_path}")

    # 关闭 TensorBoard writer
    tracker.close()


if __name__ == "__main__":
    main()
