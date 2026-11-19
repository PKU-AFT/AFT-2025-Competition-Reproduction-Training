# AFT 2025 竞赛复现与训练

PKU-AFT 的金融时间序列预测项目，围绕 Kaggle 竞赛中的纳斯达克收盘价预测，探索特征工程和模型训练。

## 项目目录

| 路径 | 内容 |
| --- | --- |
| [Feature Engineer/Factor/](Feature%20Engineer/Factor/) | 因子构造、单因子测试和泄漏检测示例 |
| [Feature Engineer/Operator/](Feature%20Engineer/Operator/) | 特征算子及辅助代码 |
| [model/RNN/](model/RNN/) | 序列模型、滚动训练和推理代码 |
| [model/Master/](model/Master/) | MASTER 模型及训练代码 |
| [model/TCN/](model/TCN/)、[model/GNN/](model/GNN/)、[model/时序Transformer/](model/时序Transformer/) | 模型实验 Notebook |

项目流程见 [项目概览](docs/README.md)，RNN 数据格式和配置见 [RNN 说明](model/RNN/README.md)。

## Python 环境

项目使用 Python 3.11 和 `uv` 管理环境。安装 uv 后，在项目根目录执行：

```bash
uv sync --locked                     # 基础数据处理环境
uv sync --locked --group training    # 模型训练依赖
uv sync --locked --group features    # 特征工程依赖
uv sync --locked --group hpo         # 调参与训练依赖
uv sync --locked --group monitor     # 监控与训练依赖
uv sync --locked --group dev         # 测试依赖
```

需要多个分组时可重复指定 `--group`，或使用 `uv sync --locked --all-groups`。`uv.lock` 固定依赖版本；切换分组时，uv 会同步环境并移除未选中的依赖。

运行泄漏检测示例的测试：

```bash
cd "Feature Engineer/Factor/leak-detector-starter"
uv run --locked --group dev pytest tests
```

## 使用准备

RNN 训练需要先补齐仓库中缺失的 `model/RNN/utils.py` 和 `model/RNN/losses.py`。依赖清单未列出代码使用的 `scipy` 和 `scikit-learn`，安装时需一并考虑。

RNN 默认数据路径指向 Windows 本地文件，MASTER 默认路径指向 Google Drive。使用前请分别修改 `model/RNN/config.py` 和 `model/Master/config.py` 中的数据及输出路径。

## 上游项目

[PKU-AFT/AFT-2025-Competition-Reproduction-Training](https://github.com/PKU-AFT/AFT-2025-Competition-Reproduction-Training)。上游 README 标注 MIT License，仓库未包含独立许可证文件。
