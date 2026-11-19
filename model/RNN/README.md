# RNN 时间序列模型

本目录包含面板时间序列数据处理、多种模型，以及滚动训练、推理和集成代码。

## 运行准备

训练需要补齐 `utils.py` 和 `losses.py`；推理也需要 `utils.py`。项目根目录的 `uv` 配置提供 `training` 依赖分组，安装方法见[项目首页](../../README.md#python-环境)。修改 `config.py` 中的 `DATA_PATH`，使其指向实际数据文件。

## 主要文件

- `config.py`：数据、窗口、模型及输出配置。
- `dataset.py`、`data.py`：数据读取与序列构建。
- `model/`：模型实现与注册。
- `train.py`、`inference.py`、`ensemble.py`、`hpo.py`：训练、推理、集成和超参数搜索。

## 数据与配置

输入为 Parquet 文件，包含 `date_id`、`seconds_in_bucket`、`stock_id`、`target` 和数值型特征列。代码以 `date_id * 10000 + seconds_in_bucket` 构造 `time_id`，并建立 `(time_id, stock_id)` 索引；目标缺失的行会被删除，特征缺失值填零。

默认模型为 GRU，输入序列长度为 10；每个窗口使用 160 个交易日训练、40 个交易日验证，步长为 20 个交易日，并将最后 81 个交易日作为固定测试集。设置 `FINAL_TEST_DAYS = None` 后，改用长度为 40 个交易日的滚动测试窗口。参数可在 `config.py` 中调整。

运行目录下的 `output/models/`、`output/preds/` 和 `output/logs/` 分别存放权重、预测和日志，并按模型类型分目录。补齐文件、依赖和数据后，可在本目录使用 `uv run --locked --group training python train.py` 尝试训练。
