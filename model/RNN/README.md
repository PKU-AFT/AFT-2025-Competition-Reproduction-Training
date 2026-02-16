# 项目说明

时间序列回归项目，基于滑动窗口 (80 天训练 / 10 天验证 / 10 天测试) 的滚动训练流程，输入为 `[batch, T, F]` 的特征序列，输出单步回归预测。

## 目录结构
- `config.py`：全局超参数与路径配置（数据路径、滑窗长度、训练/验证/测试天数、模型尺寸、输出目录等）。
- `data.py`：核心数据集与采样工具
  - `TSDataset`：将按 `time_id, stock_id` 排序的面板数据转换为滑动窗口张量，可返回目标和索引。
  - `DailyBatchSamplerRandom`：按天分批的采样器（可选 shuffle），与 `DataLoader` 配合。
- `dataset.py`：数据读取与数据集构建
  - `read_data`：读取 Parquet，构造 `time_id`（`date_id * 10000 + seconds_in_bucket`），拆分出 `target`，返回 pandas MultiIndex 的特征与标签。
  - `generate_rolling_windows`：基于唯一 `time_id` 生成 80/20/20 的滚动窗口（步长为测试窗口）。
  - `build_datasets_for_window`：按窗口切片构造 train/val/test 的 `TSDataset`。
  - `make_loader` / `make_test_loader`：构造按天分批的 DataLoader。
- `model/`
  - `model.py`：`TemporalModel`，GRU 序列编码，取最后一步 hidden，经 MLP 头输出回归值。
- `utils.py`：常用工具：`set_seed`、`ensure_dirs`、`get_logger`、`mae`（MAE 度量）。
- `train.py`：训练主流程（滚动窗口、TensorBoard 记录、保存最佳权重与预测）。
- `inference.py`：推理脚本（加载模型，对最后窗口测试切片预测并导出 CSV）。
- `output/`：运行时生成
  - `models/`：各窗口最佳权重。
  - `preds/`：预测结果 CSV。
  - `logs/`：训练/推理日志与 TensorBoard 日志。

## 数据要求
- Parquet 文件需包含列：`date_id`, `seconds_in_bucket`, `stock_id`, 其他特征列，以及 `target` 列。
- `time_id = date_id * 10000 + seconds_in_bucket`，索引用 MultiIndex(`time_id`, `stock_id`)。
- 数据需覆盖至少 120 个 `time_id` 才能生成一个完整的 80/20/20 滑窗；不足时可调整 `TRAIN_DAYS/VAL_DAYS/TEST_DAYS`。

## 训练方式（滚动窗口）
- 每个窗口：80 天训练 / 20 天验证 / 20 天测试，步长为 20（测试窗口长度）。
- 模型：GRU（长度 `STEP_LEN`，默认 10），取最后一步做回归。
- 优化：Adam，损失与度量均为 MAE。
- 每个窗口保存验证集最优的模型权重至 `output/models/window_*.pth`，并在对应测试切片上评估，预测导出到 `output/preds/window_*_preds.csv`。

## 快速开始
1. 配置数据路径与超参  
   - 修改 `config.py` 中的 `DATA_PATH`，按需调整其他超参。
2. 运行训练  
   ```bash
   python train.py
   ```
   - TensorBoard 日志在 `output/logs/window_*`，可用 `tensorboard --logdir output/logs` 查看。
3. 运行推理  
   - 默认加载最新权重：`python inference.py`  
   - 或指定权重：`python inference.py output/models/window_0.pth`  
   - 结果输出到 `output/preds/`。

## 备注
- 当前仅依赖 pandas 读取数据，无需 Polars。
- 若内存有限，可在外部先对 Parquet 做列/时间范围筛选后再运行脚本。 
