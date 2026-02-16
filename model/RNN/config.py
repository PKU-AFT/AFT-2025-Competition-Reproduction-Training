"""
时序回归预测配置文件
所有超参数和路径的统一配置中心
"""

from pathlib import Path
import torch

# =============================================================================
# 数据配置
# =============================================================================

# 数据路径
DATA_PATH = Path(r"D:\汇丰生活\模型\selected_features_top100_new.parquet")

# 目标列名
TARGET_COLUMN = "target"

# 是否对目标值进行标准化（True 时预测会自动反标准化）
LABEL_STANDARDIZE = True

# 标准化方法
# 可选: "zscore" - 标准 Z-score 标准化 (mean/std)
#       "rank_gauss" - Rank Gauss 标准化（推荐用于尖峰肥尾分布）
LABEL_STANDARDIZE_METHOD = "rank_gauss"

# 是否对目标值进行缩尾处理（Winsorization，去除极端值）
# 只在训练集上计算分位数边界，避免数据泄漏
# 注意：使用 Rank Gauss 时可以关闭 Winsorize，因为 Rank Gauss 自带处理极端值能力
LABEL_WINSORIZE = False            # 是否启用缩尾处理
LABEL_WINSORIZE_LOWER = 0.01      # 下分位数（默认 1%）
LABEL_WINSORIZE_UPPER = 0.99      # 上分位数（默认 99%）


# =============================================================================
# 序列构建配置
# =============================================================================

# 输入序列长度（滑动窗口时间步数）
# TCN推荐: 使用较短的序列(5-15)以避免dilated conv效率问题
# GRU推荐: 使用较长的序列(10-30)以捕捉更多时序信息
STEP_LEN = 10           # 每个样本包含的历史时间步数量

# 缺失值填充策略
FILLNA_TYPE = "ffill+bfill"  # "ffill"前向填充 / "ffill+bfill"前向后结合 / "none"不填充


# =============================================================================
# 滚动窗口配置（按交易日划分）
# =============================================================================

# 独立最终测试集模式：预留最后 N 天作为固定测试集
# 设置为 None 使用原有滚动窗口模式（每个窗口有自己的 test set）
# 设置为正整数（如 81）则预留最后 N 天，所有窗口共用此测试集
FINAL_TEST_DAYS = 81          # 独立测试集天数（None 或正整数）

# 滚动窗口的步长（控制窗口滑动幅度）
# - FINAL_TEST_DAYS=None 时，步长=TEST_DAYS，确保每个窗口的测试集不重叠
# - FINAL_TEST_DAYS>0 时，步长=STEP_DAYS，确保验证集不重叠
STEP_DAYS = 20                # 滚动步长（默认等于 VAL_DAYS）

TRAIN_DAYS = 160        # 训练集交易日数量
VAL_DAYS = 40          # 验证集交易日数量
TEST_DAYS = 40         # 测试集交易日数量（仅在 FINAL_TEST_DAYS=None 时使用）


# =============================================================================
# 批处理配置
# =============================================================================

# 训练批处理：将 N 个交易日的样本聚合为一个 batch
BATCH_DAYS =    256    # 每个 batch 包含的交易日数
# 测试批处理
TEST_BATCH_SIZE = 512  # 测试时的 batch 大小

USE_ATTENTION = False

# =============================================================================
# 训练超参数
# =============================================================================

# 随机种子（保证可复现）
SEED = 42

# 优化器配置
LEARNING_RATE = 1e-3   # 初始学习率
WEIGHT_DECAY = 1e-5    # L2 正则化系数

# 学习率调度器
LR_SCHEDULER = "plateau"  # 调度器类型: "plateau" 指数衰减 / "cosine" 余弦退火 / "none" 不使用
LR_FACTOR = 0.5           # Plateau: 学习率衰减因子
LR_PATIENCE = 5           # Plateau: 性能不提升的等待轮数
LR_MIN = 1e-6             # 最小学习率
LR_T0 = 10                # Cosine: 初始重启周期
LR_T_MULT = 2             # Cosine: 周期倍增系数

# 梯度裁剪（防止梯度爆炸，RNN/GRU/LSTM 尤其重要）
GRAD_CLIP_NORM = 1.0   # 梯度范数裁剪阈值（0 表示不裁剪）

# 训练轮数
NUM_EPOCHS = 50       # 最大训练轮数（受早停影响可能提前结束）


# =============================================================================
# 模型架构通用参数
# =============================================================================

# 隐藏层维度
HIDDEN_SIZE = 128      # 模型隐藏层大小（影响参数量和容量）

# 网络层数
NUM_LAYERS = 2         # RNN/TCN/Transformer 的层数（过多可能导致过拟合）

# Dropout 概率
DROPOUT = 0.2          # 0.1=10% 随机丢弃（值越大正则化越强）


# =============================================================================
# 模型选择
# =============================================================================
# 可选模型:
#   "gru"         - 单向 GRU（基线模型，训练快）
#   "gru_attn"    - 双向 GRU + 时序注意力（性能更好）
#   "lstm"        - 单向 LSTM（与 GRU 类似）
#   "lstm_attn"   - 双向 LSTM + 时序注意力
#   "transformer" - Transformer Encoder（并行度高，适合长序列）
#   "tcn"         - 时序卷积网络（因果卷积，并行高效）
#   "mlp"         - 简单全连接（快速基线对比）
#   "griffin"     - Griffin 混合架构（RG_LRU + 时序卷积，高效长序列建模）
# =============================================================================

MODEL_TYPE = "gru"


# =============================================================================
# 输出路径配置（不同模型的输出分开保存）
# =============================================================================
# 目录结构:
#   output/
#   ├── models/
#   │   ├── GRU/window_0.pth
#   │   ├── LSTM/window_0.pth
#   │   └── TCN/window_0.pth
#   ├── preds/
#   │   ├── GRU/window_0_preds.csv
#   │   ├── LSTM/window_0_preds.csv
#   │   └── TCN/window_0_preds.csv
#   └── logs/
#       ├── GRU/window_0/
#       ├── LSTM/window_0/
#       └── TCN/window_0/

# 模型类型子目录名称（将 gru_attn -> GRU_ATTN）、
OUTPUT_DIR = Path("output")
_MODEL_SUBDIR = MODEL_TYPE.upper().replace("-", "_")
MODEL_DIR = OUTPUT_DIR / "models" / _MODEL_SUBDIR
PRED_DIR = OUTPUT_DIR / "preds" / _MODEL_SUBDIR
LOG_DIR = OUTPUT_DIR / "logs" / _MODEL_SUBDIR




# =============================================================================
# 模型特定参数
# =============================================================================

# ---------- Transformer ----------
NHEAD = 8              # 注意力头数（必须能整除 HIDDEN_SIZE，如 128 可用 4/8/16）
DIM_FEEDFORWARD = None # FFN 中间层维度（None 则为 HIDDEN_SIZE * 4）

# ---------- TCN ----------
# AFT-TCN配置: kernel_size=2 效率更高（padding更小）
TCN_KERNEL_SIZE = 2    # kernel=2时 padding=dilation，效率显著优于kernel=3
TCN_NUM_LAYERS = 3     # kernel=2, L=3 -> RF=8 >= STEP_LEN=7 ✓
TCN_NUM_CHANNELS = [64, 128, 128]  # 递增通道数（可选，当前仍用固定HIDDEN_SIZE）

# ---------- MLP ----------
# MLP 使用 STEP_LEN 和 INPUT_DIM 自动计算输入维度

# ---------- GRU/LSTM ----------
# 使用通用 HIDDEN_SIZE、NUM_LAYERS、DROPOUT 参数

# ---------- Griffin ----------
# Griffin 混合架构特定参数
GRIFFIN_EXPANSION = 2         # MLP 扩展因子（内部隐藏层 = hidden_size * expansion）
GRIFFIN_RNN_WIDTH = None      # RNN 宽度（None 则等于 hidden_size）


# =============================================================================
# 硬件与数据加载配置
# =============================================================================

# 设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 混合精度训练（仅 CUDA 设备有效）
USE_AMP = True         # 是否使用自动混合精度（AMP）- 加速训练、节省显存

# 数据加载器配置
NUM_WORKERS = 16        # 数据加载进程数（Windows 必须设为 0，否则 DataLoader 初始化会卡死）
PIN_MEMORY = True      # 是否锁页内存（GPU 训练时建议 True）
PREFETCH_FACTOR = 2    # 每个 worker 预取的 batch 数（提高 GPU 利用率）


# =============================================================================
# 损失函数配置
# =============================================================================

# 损失函数类型
# 可选: "mae", "mse", "huber", "quantile", "focal_mae", "weighted_mae",
#       "adaptive_mae", "log_cosh", "tweedie"
# 别名: "l1"=mae, "l2"=mse, "smooth_l1"=huber, "pinball"=quantile
LOSS_TYPE = "mae"

# 损失函数参数
# Huber Loss
HUBER_DELTA = 1.0              # Huber损失切换点

# Quantile Loss
QUANTILE = 0.5                # 目标分位数 (0.5=中位数, 0.1/0.9=90%置信区间)

# Focal MAE Loss
FOCAL_GAMMA = 2.0             # 聚焦参数，越大越关注难样本

# Weighted MAE Loss
WEIGHTED_MAE_TYPE = "magnitude"  # 加权类型: "magnitude", "threshold", "smooth_threshold", "error", "inverse_magnitude"
WEIGHTED_MAE_ALPHA = 1.0          # 误差加权强度 (仅 weight_type="error" 时)
# 阈值加权参数 (用于 threshold 和 smooth_threshold 模式)
WEIGHTED_MAE_THRESHOLD = 5.0      # 加权阈值（超过此值的目标值权重增加）
WEIGHTED_MAE_HIGH_WEIGHT = 2.0     # 超阈值样本的权重
WEIGHTED_MAE_LOW_WEIGHT = 1.0      # 低于阈值样本的权重
WEIGHTED_MAE_MAX_WEIGHT = 3.0      # 最大权重 (仅 smooth_threshold 模式)
WEIGHTED_MAE_SMOOTH_WIDTH = 2.0   # 平滑过渡宽度 (仅 smooth_threshold 模式)

# Adaptive MAE Loss
ADAPTIVE_WARMUP_EPOCHS = 5    # 预热轮数
ADAPTIVE_MAX_GAMMA = 3.0      # 最大聚焦参数


# =============================================================================
# 早停配置
# =============================================================================

EARLY_STOP_PATIENCE = 10      # 验证集性能不提升的最大等待轮数
EARLY_STOP_MIN_DELTA = 0.0    # 视为提升的最小性能变化量
