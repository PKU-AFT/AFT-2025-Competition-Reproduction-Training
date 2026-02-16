方法层面的风险

  - 交叉截面上下文丢失，导致误报：重放时只给单一 symbol 的前缀子集（而非“截至
    as_of 的全市场切片”）。凡是用到同日横截面统计（如按 date 排名/标准化、行业中性
    化、同日均值/分位）都会在重放中失真，被当成“泄露”误判。
  - 索引对齐不严格：仅校验 len(series)==len(df)，随后用 full_series.iloc[idx]
    做位置访问；若因子在内部重排或重建索引，位置将错位。应强制
    series.index.equals(df.index) 或先 .reindex(df.index)，再比较。
  - 回放值定位存在掩盖风险：replay_val = sub_series.loc[idx] if idx in
    sub_series.index else sub_series.iloc[-1]。如果因子在子集内重排，这个后备
    策略会把“最后一行”的值错当目标行，产生偶然一致或不稳定误差；应直接对齐到
    (entity_col,time_col) 后再取值，若无法对齐应标记为 index_mismatch 而非静默
    回退。
  - 比较容差过严且不含相对误差：仅用绝对容差 1e-9，对大数/缩放敏感；应使用
    np.isclose(a,b, rtol=..., atol=...) 并显式处理无穷大。
  - 类型兼容性不足：比较函数强制 float(a)-float(b)。布尔/整数可行，但分类、字符
    串、时间戳会报错并中断检测。应按 dtype 分派：数值用 isclose，布尔/分类用相等，
    时间戳比较纳秒差。
  - 错误与泄露信号混淆：factor_error、replay_error 以 confidence=0.0 也被加入
    violations，下游若把任何 violation 当失败会噪声很大。应区分“实现问题”和“泄露嫌
    疑”的严重级别/类别。
  - 示例细节缺失：examples 中的 replay 永远是 None，难以复现与解释。应在同一重放上
    下文中回算并填入 replay 与 delta。

  边界场景

  - 多实体依赖/市场聚合：因子若依赖全市场/行业聚合（当日横截面均值、分位、截面回
    归），当前“单实体子集重放”一定产生高错配率（系统性误报）。
  - 多索引/重复时间：同一 symbol 下 time_col 重复或非唯一排序稳定性问题；建议在排
    序后检测并记录重复时间点，或以 (entity,time,position) 唯一定位。
  - 因子返回非常规形态：返回 DataFrame、与输入长度对齐但索引不一致、或返回延迟/前
    移后的对齐缺失，当前仅以长度判断会放过对齐问题或误判。
  - 非确定性因子：内部使用随机数、外部状态（全局缓存、时区）、或依赖未固定的第三方
    库版本，重放将出现不稳定错配（非泄露）。
  - 小样本/起始窗口：min_periods、初期 NaN 边界在不同实现可略有差异；对“近零值”的
    绝对容差比较会放大误差比例。
  - 合并引入的隐式未来：通过 merge 同日跨表拿到“未来填充”的列（如对未来已知的财报
    数据进行前填充）在当前框架可能被识别为泄露，但需要更具体的解释与定位（建议静态
    规则辅助识别如 shift(-1), forward fill, asof merge）。

  性能与可扩展性

  - 复杂度 O(E · T^2)：每个实体逐步扩大的重放对大数据集不可扩展；还频繁 .copy()，
    内存/CPU 开销高。
  - 无早停与采样：在错配率明显升高时仍全量重放，浪费计算。

  改进建议（优先级从高到低）

  - 改为“全市场 as-of 重放”：对每个检查点 as_of（可按唯一日期采样），构造
    df[df[time_col] <= as_of] 全量切片运行因子，再取目标行的值比较。这样保留横截面
    上下文，显著降低误报。
  - 严格对齐与比较策略：
      - 先强制 series = series.reindex(work_df.index)，否则标记 index_mismatch。
      - 数值用 np.isclose（可设 rtol=1e-6, atol=1e-9，并可从 FactorDefinition 覆
        盖），其他 dtype 用类型相等或专用比较。
  - 回放取值以键对齐：通过 (entity_col, time_col) merge 定位同一行，不再使用
    iloc[-1] 回退。
  - 结果结构化与示例完善：
      - violations 增加 severity/category（implementation_error / shape_mismatch /
        suspected_leak）。
      - examples 填入 replay、abs_diff、rel_diff 与用于重放的 as_of 时间。
  - 引入静态规则的轻量扫描：inspect.getsource 中搜索 shift(-、lead、asof merge +
    forward fill 等关键词，作为独立“静态嫌疑”信号（低到中等置信度），与动态重放交
    叉验证。
  - 控制成本与稳定性：
      - 仅在每个实体上抽样 K 个时间点（或全局前/中/后等距采样、或首个非 NaN 后的固
        定间隔）。
      - 设置早停：当错配率超过阈值（如 20%）且在多个实体/时间段一致时，可停止进一
        步重放。
      - 在函数外固定随机种子（np.random.seed、random.seed）并鼓励因子纯函数化。
  - 训练/验证一致性更有信息量：报告错配率的置信区间（Wilson/正态近似），而不是再次
    生成一条 consistency_check violation。
  - 容错与可观测性：
      - 对 factor_error/replay_error 标记为 “不可判定（inconclusive）”，并在 notes
        给出建议（如需要全市场上下文或索引修复）。
      - 记录每个因子被检查的样本量，避免小样本上的偶然误差被高估。