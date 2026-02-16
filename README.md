# 金融时序预测系统

基于kaggle竞赛的金融时序预测项目，专注于纳斯达克收盘价预测，目前是初版，之后会整理成更加可复用的模型库。

## 项目概述

本项目结合金融特征工程和深度学习模型，用于预测金融资产的时序价格变动。支持多种模型架构和滚动窗口训练策略。

### 核心特性

- 特征工程：因子拓展、算子生成、特征重要性筛选
- 多种模型架构：GRU、LSTM、Transformer、TCN
- 滚动窗口训练：避免数据泄漏
- 集成学习：多模型融合

## 项目结构

```
.
├── Feature Engineer/      # 特征工程模块
├── model/
│   ├── RNN/              # RNN 时序模型（主要模块）
│   ├── Master/           # MASTER Transformer 模型
│   ├── TCN/           # MASTER Transformer 模型
│   ├── 时序Transformer/           # MASTER Transformer 模型
│   └── GNN/              # 图神经网络
└── output/               # 模型权重、预测结果、日志
```

## 快速开始

### 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ 

### 安装依赖

```bash
cd model/RNN
pip install -r requirements.txt
```

### 训练模型

```bash
python train.py 
```


## 特征工程

位于 [Feature Engineer/Factor/](Feature Engineer/Factor/)，包含：

- **特征提取**：拍卖流动性、价格发现、订单簿压力等
- **算子库**：时序算子、横截面算子、组合算子
- **特征筛选**：重要性排序、相关性过滤、Top-K 选择

## 模型

位于 [model/](model/)，包含：

- **RNN**：主要训练模块，支持多种时序模型
- **Master**：Transformer 混合架构
- **GNN**：图神经网络实验



## 训练策略

采用滚动窗口训练策略，测试集不重叠，避免数据泄漏：

```
窗口1: [0-160训练] [160-200验证] [200-240测试]
窗口2: [40-200训练] [200-240验证] [240-280测试]
...
```


## 许可证

MIT License
