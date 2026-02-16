import pandas as pd
import numpy as np
from typing import Sequence
try:
    import numba
except Exception:  # Fallback if numba is unavailable
    class _DummyNumba:
        def jit(self, *args, **kwargs):
            def deco(f):
                return f
            return deco
    numba = _DummyNumba()


class Factor:
    # ===================================================================
    # 你的原始静态函数 (保持不变)
    # ===================================================================
    @staticmethod
    def auction_liquidity_profile(df, group_col='stock_id'):
        """
        【原始】拍卖流动性特征 (主要关注 L1 和匹配效率)
        """
        # 总流动性 (拍卖池成交+未成交)
        # 使用abs()确保imbalance为正，如果它是带符号的
        df['total_auction_volume'] = df['matched_size'] + df['imbalance_size'].abs() 

        # 买卖方流动性不平衡 (L1)
        df['l1_bid_ask_liquidity_imbalance'] = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size']).replace(0, np.nan)

        # 拍卖订单占比(相对于连续市场L1)
        continuous_liquidity = df['bid_size'] + df['ask_size']
        df['auction_continuous_ratio'] = df['total_auction_volume'] / continuous_liquidity.replace(0, np.nan)

        # 匹配效率(能够匹配的比例)
        df['matching_efficiency'] = df['matched_size'] / df['total_auction_volume'].replace(0, np.nan)

        return df

    @staticmethod
    def auction_price_discovery(df, group_col='stock_id'):
        """
        【原始】价格发现过程特征 (关注 near/far/ref 价格)
        """
        # near 和 far 价格的平均值
        df['near_far_mid'] = (df['near_price'] + df['far_price']) / 2

        # reference_price 与 near_far_mid 的偏离
        df['ref_nearfar_deviation'] = df['reference_price'] - df['near_far_mid']

        # near_price 与 reference_price 的偏离
        df['near_ref_deviation'] = df['near_price'] - df['reference_price']
        
        # (使用 reference_price 作为分母可能更稳定)
        df['near_ref_deviation_pct'] = df['near_ref_deviation'] / df['reference_price'].replace(0, np.nan)

        # far_price 与 reference_price 的偏离
        df['far_ref_deviation'] = df['far_price'] - df['reference_price']
        df['far_ref_deviation_pct'] = df['far_ref_deviation'] / df['reference_price'].replace(0, np.nan)

        # 价格一致性(三个价格的标准差)
        df['price_consistency'] = df[['near_price', 'far_price', 'reference_price']].std(axis=1)

        return df

    @staticmethod
    def auction_order_book_pressure(df, group_col='stock_id'):
        """
        【原始】订单簿压力特征 (主要关注 L1)
        """
        # 买卖价差
        df['bid_ask_spread'] = df['ask_price'] - df['bid_price']
        df['bid_ask_spread_pct'] = df['bid_ask_spread'] / df['wap'].replace(0, np.nan)

        # 买卖量比 (L1)
        df['l1_bid_ask_size_ratio'] = df['bid_size'] / df['ask_size'].replace(0, np.nan)
        df['l1_bid_ask_size_imbalance'] = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size']).replace(0, np.nan)

        # WAP 与买卖中间价的偏离
        df['bid_ask_mid'] = (df['bid_price'] + df['ask_price']) / 2
        df['wap_mid_deviation'] = df['wap'] - df['bid_ask_mid']
        df['wap_mid_deviation_pct'] = df['wap_mid_deviation'] / df['bid_ask_mid'].replace(0, np.nan)

        # WAP 在买卖价差中的位置
        df['wap_position'] = (df['wap'] - df['bid_price']) / df['bid_ask_spread'].replace(0, np.nan)

        return df

    # ===================================================================
    # 拓展 1: 时间紧迫性因子 (Time Urgency Factors)
    # ===================================================================
    @staticmethod
    def auction_time_urgency(df, group_col='stock_id'):
        """
        基于 seconds_in_bucket 的时间紧迫性特征
        临近收盘时（seconds_in_bucket 接近 600），市场行为会发生变化
        """
        # 归一化时间进度 (0-1)
        df['time_progress'] = df['seconds_in_bucket'] / 600.0

        # 剩余时间（反向）
        df['time_to_close'] = 600 - df['seconds_in_bucket']
        df['time_to_close_normalized'] = df['time_to_close'] / 600.0

        # 时间紧迫性加权的不平衡（越接近收盘，不平衡越重要）
        df['time_weighted_imbalance'] = df['imbalance_size'] * df['time_progress']

        # 时间紧迫性加权的价差（临近收盘，价差压缩或扩大）
        df['time_weighted_spread'] = (df['ask_price'] - df['bid_price']) * df['time_progress']

        # 时间段标记（早期/中期/晚期）
        df['time_stage_early'] = (df['seconds_in_bucket'] < 200).astype(int)
        df['time_stage_mid'] = ((df['seconds_in_bucket'] >= 200) & (df['seconds_in_bucket'] < 400)).astype(int)
        df['time_stage_late'] = (df['seconds_in_bucket'] >= 400).astype(int)

        # 紧急度指标（非线性，临近收盘加速增长）
        df['urgency_score'] = np.exp(df['time_progress'] * 2) - 1

        return df

    # ===================================================================
    # 拓展 2: 不平衡方向性因子 (Imbalance Directionality Factors)
    # ===================================================================
    @staticmethod
    def auction_imbalance_directionality(df, group_col='stock_id'):
        """
        利用 imbalance_buy_sell_flag 的方向性特征
        区分买方压力和卖方压力
        """
        # 带符号的不平衡（买方为正，卖方为负）
        # imbalance_buy_sell_flag: 1=买方, -1=卖方, 0=平衡
        df['signed_imbalance'] = df['imbalance_size'] * df['imbalance_buy_sell_flag']

        # 相对不平衡强度（相对于总成交量）
        total_volume = df['matched_size'] + df['imbalance_size'].abs()
        df['relative_signed_imbalance'] = df['signed_imbalance'] / total_volume.replace(0, np.nan)

        # 买方主导度（买方不平衡占比）
        df['buy_dominance'] = ((df['imbalance_buy_sell_flag'] == 1).astype(int) *
                               df['imbalance_size'] / total_volume.replace(0, np.nan))

        # 卖方主导度（卖方不平衡占比）
        df['sell_dominance'] = ((df['imbalance_buy_sell_flag'] == -1).astype(int) *
                                df['imbalance_size'] / total_volume.replace(0, np.nan))

        # 不平衡方向与价格偏离的交互
        # 如果买方不平衡且 near_price > reference_price，可能预示上涨压力
        df['imbalance_price_alignment'] = (
            df['signed_imbalance'] * (df['near_price'] - df['reference_price'])
        )

        # 不平衡方向与订单簿压力的交互
        df['imbalance_book_pressure'] = (
            df['signed_imbalance'] * ((df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size']).replace(0, np.nan))
        )

        return df

    # ===================================================================
    # 拓展 3: 价格冲击和效率因子 (Price Impact & Efficiency Factors)
    # ===================================================================
    @staticmethod
    def auction_price_impact(df, group_col='stock_id'):
        """
        价格冲击和市场效率特征
        衡量订单流对价格的影响程度
        """
        # 1. 价格冲击强度（near_price 偏离 far_price 的幅度）
        # near_price 包含连续市场订单，far_price 仅拍卖订单
        df['price_impact'] = df['near_price'] - df['far_price']
        df['price_impact_pct'] = df['price_impact'] / df['far_price'].replace(0, np.nan)

        # 2. 单位不平衡的价格冲击（每单位不平衡导致的价格变化）
        df['price_impact_per_imbalance'] = (
            df['price_impact'] / df['imbalance_size'].replace(0, np.nan)
        )

        # 3. WAP 的价格发现效率（WAP 与 reference_price 的偏离）
        df['wap_price_discovery_efficiency'] = (
            abs(df['wap'] - df['reference_price']) / df['reference_price'].replace(0, np.nan)
        )

        # 4. 三价格的价格发现收敛度（标准差越小，价格发现越有效）
        price_mean = df[['near_price', 'far_price', 'reference_price']].mean(axis=1)
        df['price_convergence'] = (
            df[['near_price', 'far_price', 'reference_price']].std(axis=1) / price_mean.replace(0, np.nan)
        )

        # 5. 有效价差（考虑拍卖价格后的综合价差）
        df['effective_spread'] = abs(df['near_price'] - df['far_price']) + (df['ask_price'] - df['bid_price'])
        df['effective_spread_pct'] = df['effective_spread'] / df['wap'].replace(0, np.nan)

        # 6. 价格弹性（价格对订单流的敏感度）
        total_depth = df['bid_size'] + df['ask_size']
        df['price_elasticity'] = df['price_impact'] / total_depth.replace(0, np.nan)

        return df

    # ===================================================================
    # 拓展 4: 微观结构因子 (Microstructure Factors)
    # ===================================================================
    @staticmethod
    def auction_microstructure(df, group_col='stock_id'):
        """
        订单簿微观结构特征
        关注深度、分布和成交效率
        """
        # 1. 订单簿深度不对称性 (已在 auction_liquidity_profile 中计算为 l1_bid_ask_liquidity_imbalance，此处不重复)

        # 2. 价格-深度比率（价格相对于深度的比率）
        df['price_to_depth_bid'] = df['bid_price'] / df['bid_size'].replace(0, np.nan)
        df['price_to_depth_ask'] = df['ask_price'] / df['ask_size'].replace(0, np.nan)

        # 3. 成交压力指标（matched 相对于总订单的比例）
        total_order_flow = df['matched_size'] + df['imbalance_size'].abs()
        df['execution_pressure'] = df['matched_size'] / total_order_flow.replace(0, np.nan)

        # 4. 不成交风险（imbalance 相对于总订单的比例）
        df['non_execution_risk'] = df['imbalance_size'].abs() / total_order_flow.replace(0, np.nan)

        # 5. 订单簿厚度（深度相对于价格的比率）
        df['book_thickness'] = (df['bid_size'] + df['ask_size']) / df['wap'].replace(0, np.nan)

        # 6. 买卖方市值差异
        df['bid_notional'] = df['bid_price'] * df['bid_size']
        df['ask_notional'] = df['ask_price'] * df['ask_size']
        df['notional_imbalance'] = (df['bid_notional'] - df['ask_notional']) / (df['bid_notional'] + df['ask_notional']).replace(0, np.nan)

        # 7. 流动性成本（已在 auction_order_book_pressure 中计算为 bid_ask_spread_pct，此处不重复）

        # 8. Kyle's Lambda（价格冲击的市场深度调整版本）
        df['kyle_lambda'] = (df['near_price'] - df['far_price']) / df['imbalance_size'].replace(0, np.nan)

        return df

    # ===================================================================
    # 拓展 5: 高级交叉特征 (Advanced Cross Features)
    # ===================================================================
    @staticmethod
    def auction_cross_features(df, group_col='stock_id'):
        """
        现有特征的非线性组合和交互特征
        捕捉复杂的市场动态
        """
        # 1. 不平衡-流动性交互（不平衡在低流动性时更重要）
        total_liquidity = df['bid_size'] + df['ask_size'] + df['matched_size']
        df['imbalance_liquidity_ratio'] = df['imbalance_size'] / total_liquidity.replace(0, np.nan)

        # 2. 价差-波动性交互（价差调整的价格波动）
        price_std = df[['near_price', 'far_price', 'reference_price']].std(axis=1)
        df['spread_adjusted_volatility'] = price_std / (df['ask_price'] - df['bid_price']).replace(0, np.nan)

        # 3. 匹配效率-时间交互（临近收盘时匹配效率的重要性）
        time_progress = df['seconds_in_bucket'] / 600.0
        matching_eff = df['matched_size'] / (df['matched_size'] + df['imbalance_size'].abs()).replace(0, np.nan)
        df['time_adjusted_matching_efficiency'] = matching_eff * (1 + time_progress)

        # 4. 不平衡强度平方（捕捉非线性效应）
        df['imbalance_intensity_squared'] = (
            df['imbalance_size'] / total_liquidity.replace(0, np.nan)
        ) ** 2

        # 5. 综合压力指标（多维度压力的组合）
        l1_imbalance = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size']).replace(0, np.nan)
        bid_notional = df['bid_price'] * df['bid_size']
        ask_notional = df['ask_price'] * df['ask_size']
        notional_imb = (bid_notional - ask_notional) / (bid_notional + ask_notional).replace(0, np.nan)
        signed_imb = df['imbalance_size'] * df['imbalance_buy_sell_flag']

        df['composite_pressure'] = (
            l1_imbalance * 0.3 +
            notional_imb * 0.3 +
            (signed_imb / total_liquidity.replace(0, np.nan)) * 0.4
        )

        # 6. 市场质量指标（综合流动性、效率和价格发现）
        # 使用已计算的 bid_ask_spread_pct (等同于 liquidity_cost)
        price_mean = df[['near_price', 'far_price', 'reference_price']].mean(axis=1)
        price_conv = df[['near_price', 'far_price', 'reference_price']].std(axis=1) / price_mean.replace(0, np.nan)

        df['market_quality_score'] = (
            matching_eff * 0.4 +
            (1 - df['bid_ask_spread_pct'].fillna(0)) * 0.3 +
            (1 / (1 + price_conv.fillna(0))) * 0.3
        )

        # 7. 风险调整的不平衡（不平衡除以价格波动）
        df['risk_adjusted_imbalance'] = (
            df['imbalance_size'] / price_std.replace(0, np.nan)
        )

        # 8. 信息不对称指标（价格偏离和订单不平衡的综合）
        df['information_asymmetry'] = (
            abs(df['near_price'] - df['reference_price']) *
            abs(signed_imb)
        )

        return df

    # ===================================================================
    # 拓展 6: 融合静态因子（原有）
    # ===================================================================
    @staticmethod
    def auction_fusion_factors(df, group_col='stock_id'):
        """
        (实现我们讨论的 'intensity', 'profile', 'pressure' 融合版)
        """
        print("生成融合（L1+拍卖池）静态因子...")
        
        epsilon = 1e-8 # 用于安全除法

        # 1. 计算融合深度
        # (假设你的df中包含 'auc_bid_size' 和 'auc_ask_size')
        df['l1_depth'] = df['bid_size'] + df['ask_size']
        df['auction_depth'] = df['auc_bid_size'] + df['auc_ask_size']
        df['total_fused_depth'] = df['l1_depth'] + df['auction_depth']

        # 2. auction_imbalance_intensity (不平衡强度)
        # 逻辑: 拍卖池不平衡量 / 市场总深度 (L1+拍卖池)
        df['auction_imbalance_intensity'] = df['imbalance_size'] / (df['total_fused_depth'] + epsilon)

        # 3. auction_liquidity_profile_fused (流动性画像)
        # 逻辑: 拍卖池流动性 / 市场总深度 (L1+拍卖池)
        df['auction_liquidity_profile_fused'] = df['auction_depth'] / (df['total_fused_depth'] + epsilon)

        # 4. auction_order_book_pressure (融合订单簿压力)
        # 逻辑: log(总买方名义价值) - log(总卖方名义价值)
        df['total_buy_money'] = (df['bid_size'] * df['bid_price']) + (df['auc_bid_size'] * df['reference_price'])
        df['total_sell_money'] = (df['ask_size'] * df['ask_price']) + (df['auc_ask_size'] * df['reference_price'])
        
        df['auction_order_book_pressure'] = (
            np.log(df['total_buy_money'] + epsilon) - 
            np.log(df['total_sell_money'] + epsilon)
        )

        return df
        
    # ===================================================================
    # 肖力丹
    @staticmethod
    def lidan_price(df, group_col='stock_id'):
        """
        肖力丹价格因子 - 价格偏离和市场压力特征
        """
        # near_price 与 bid_price 的偏离（near 在买盘方向偏高 → 短期更可能回落 / 偏低 → 更可能上涨）
        df['near_bid_impact'] = df['near_price'] - df['bid_price']

        # near_price 与 ask_price 的偏离（near 靠近卖盘代表上压力减弱）
        df['near_ask_impact'] = df['ask_price'] - df['near_price']


        # -------------------------
        # reference / price family imbalance
        # 所有 “imb1_xxx_yyy” 都是 (A - B) / (A + B) 的标准化不平衡度
        # -------------------------
        df['mid_price'] = (df['ask_price']+df['bid_price'])/2
        # 参考价格相对卖盘 ask_price 的不平衡程度（参考价偏高→看涨）
        df['imb1_reference_price_ask_price'] = (
            df['reference_price'] - df['ask_price']
        ) / (df['reference_price'] + df['ask_price'])

        # 参考价格相对 mid_price 的偏离（参考价格高于/低于理论公平价）
        df['imb1_reference_price_mid_price'] = (
            df['reference_price'] - df['mid_price']
        ) / (df['reference_price'] + df['mid_price'])

        # wap 相对 mid_price 的偏离（最近成交均价是否偏离订单簿理论中间点）
        df['imb1_wap_mid_price'] = (
            df['wap'] - df['mid_price']
        ) / (df['wap'] + df['mid_price'])

        # 参考价 vs 买盘 bid_price 的偏离程度（参考价低→后续更可能上涨）
        df['imb1_reference_price_bid_price'] = (
            df['reference_price'] - df['bid_price']
        ) / (df['reference_price'] + df['bid_price'])

        # near_price vs wap 的标准化偏离（near 是否偏离真实成交均价）
        df['imb1_near_price_wap'] = (
            df['near_price'] - df['wap']
        ) / (df['near_price'] + df['wap'])

        # far_price vs wap（不含连续盘订单的清算价是否偏离 wap）
        df['imb1_far_price_wap'] = (
            df['far_price'] - df['wap']
        ) / (df['far_price'] + df['wap'])

        # -------------------------
        # Orderbook imbalance (size & price side)
        # -------------------------

        # 订单簿挂单量不平衡（正=卖盘挂单更多；负=买盘挂单更多）
        df['imb1_ask_size_bid_size'] = (
            df['ask_size'] - df['bid_size']
        ) / (df['ask_size'] + df['bid_size'])

        # 价格不平衡（ask_price 偏高 or bid_price 偏高）
        df['imb1_ask_price_bid_price'] = (
            df['ask_price'] - df['bid_price']
        ) / (df['ask_price'] + df['bid_price'])


        # -------------------------
        # Price pressure & urgency
        # -------------------------

        # 价格压力：价差 × 不平衡挂单量（越大表示 orderbook 容易被冲破）
        df['price_pressure'] = (df['ask_price'] - df['bid_price']) * df['imbalance_size']

        # signed price pressure：带买卖方向的信息（正=卖压；负=买压）
        df['signed_price_pressure'] = (
            df['ask_price'] - df['bid_price']
        ) * df['imbalance_size'] * df['imbalance_buy_sell_flag']

        # market_urgency：价差 × 深度不平衡（spread 越大 & 深度偏斜越严重 → 越容易发生快速移动）
        df['market_urgency'] = (
            df['ask_price'] - df['bid_price']
        ) * df['imb1_ask_size_bid_size']

        # market_urgency_v3：价格不平衡 × 深度不平衡（双重共振 → 最强的单边方向信号）
        df['market_urgency_v3'] = (
            df['imb1_ask_price_bid_price'] * df['imb1_ask_size_bid_size']
        )

        # -------------------------
        # reference vs wap
        # -------------------------

        # reference_price vs wap 的标准化偏离（最强 auction signal 之一）
        df['imb1_reference_price_wap'] = (
            df['reference_price'] - df['wap']
        ) / (df['reference_price'] + df['wap'])
        # ===============================================`====================
        return df 

    @staticmethod
    def auction_static_price_extensions(df, group_col='stock_id', date_col='date_id'):
        """
        静态价格扩展特征
        """
        # 确保 bid_ask_mid 存在
        if 'bid_ask_mid' not in df.columns:
            df['bid_ask_mid'] = (df['bid_price'] + df['ask_price']) / 2

        # near / far 相对中间价的偏离
        df['near_mid_deviation'] = df['near_price'] - df['bid_ask_mid']
        df['far_mid_deviation'] = df['far_price'] - df['bid_ask_mid']
        df['near_mid_deviation_pct'] = df['near_mid_deviation'] / df['bid_ask_mid'].replace(0, np.nan)
        df['far_mid_deviation_pct'] = df['far_mid_deviation'] / df['bid_ask_mid'].replace(0, np.nan)

        # 泄露的我删掉了

        return df


    @staticmethod
    def auction_price_pressure(df, group_col='stock_id'):
        """
        拍卖价格压力 - near_price 和 far_price 的差异

        near_price: 包含连续市场订单的清算价
        far_price: 仅包含拍卖订单的清算价
        差异反映了连续市场订单对价格的影响

        Args:
            df: DataFrame
            group_col: 分组列

        Returns:
            df: 添加了 auction_price_pressure 列
        """
        # 价格压力(绝对值)
        df['auction_price_pressure'] = df['near_price'] - df['far_price']

        # 价格压力(相对值,基于 wap)
        df['auction_price_pressure_pct'] = df['auction_price_pressure'] / df['wap']

        # 价格压力方向
        df['auction_price_pressure_sign'] = np.sign(df['auction_price_pressure'])

        return df

    @staticmethod
    def auction_imbalance_intensity(df, group_col='stock_id'):
        """
        拍卖不平衡强度

        imbalance_size: 未匹配的金额
        matched_size: 可匹配的金额
        比率反映了市场失衡的严重程度

        Args:
            df: DataFrame
            group_col: 分组列

        Returns:
            df: 添加了不平衡强度相关列
        """
        # 不平衡比率
        df['imbalance_ratio'] = df['imbalance_size'] / (df['matched_size'] + df['imbalance_size']).replace(0, np.nan)

        # 带符号的不平衡比率
        df['signed_imbalance_ratio'] = df['imbalance_ratio'] * df['imbalance_buy_sell_flag']

        # 不平衡金额占总成交量的比例
        total_volume = df['bid_size'] + df['ask_size']
        df['imbalance_volume_ratio'] = df['imbalance_size'] / total_volume.replace(0, np.nan)

        return df

    @staticmethod
    def auction_reference_price_quality(df, group_col='stock_id'):
        """
        参考价格质量指标

        reference_price 是使配对最大化、不平衡最小化的价格
        分析其与市场价格的关系

        Args:
            df: DataFrame
            group_col: 分组列

        Returns:
            df: 添加了参考价格质量相关列
        """
        # 参考价格与 WAP 的偏离
        df['ref_price_deviation'] = df['reference_price'] - df['wap']
        df['ref_price_deviation_pct'] = df['ref_price_deviation'] / df['wap']

        # 参考价格与买卖中间价的偏离
        df['bid_ask_mid'] = (df['bid_price'] + df['ask_price']) / 2
        df['ref_price_mid_deviation'] = df['reference_price'] - df['bid_ask_mid']

        # 参考价格在买卖价差中的位置 (0-1, 0=bid, 1=ask)
        bid_ask_spread = df['ask_price'] - df['bid_price']
        df['ref_price_position'] = (df['reference_price'] - df['bid_price']) / bid_ask_spread.replace(0, np.nan)

        return df

    @staticmethod
    def auction_dynamic_trend(
        df,
        group_col='stock_id',
        fast_span: int = 10,
        slow_span: int = 30,
        slope_window: int = 20,
        signal_span: int = 9,
        momentum_windows: Sequence[int] = (5, 10, 20),
        zscore_windows: Sequence[int] = (15, 30),
        mock_periods: Sequence[int] = (1, 3),
        macd_spans: Sequence[int] = (3, 6, 12, 24, 48),
    ):
        """Compute a compact set of EMA/MACD, momentum and mock-based trend diagnostics."""

        slope_window = max(2, slope_window)
        date_col = 'date_id' if 'date_id' in df.columns else None
        bucket_col = 'seconds_in_bucket' if 'seconds_in_bucket' in df.columns else None
        sort_cols = [group_col]
        if date_col:
            sort_cols.append(date_col)
        if bucket_col:
            sort_cols.append(bucket_col)
        df.sort_values(sort_cols, inplace=True)

        group_cols_day = [group_col] + ([date_col] if date_col else [])
        base_group = df.groupby(group_col)
        day_group = df.groupby(group_cols_day)

        def _group_series(col: str, intraday: bool = False) -> pd.core.groupby.generic.SeriesGroupBy:
            """Helper to access grouped series; intraday=True prevents cross-day windows."""

            return (day_group if intraday else base_group)[col]

        def _assign(columns: dict[str, pd.Series], note: str) -> None:
            """Attach a batch of Series to df with logging."""

            nonlocal df
            if not columns:
                return
            print(f"[trend] assigning {note} ({len(columns)} columns)")
            for name in columns:
                print(f"[trend] generated factor: {name}")
            df = df.assign(**columns)

        def _rolling_slope(values: np.ndarray) -> float:
            if len(values) == 0:
                return np.nan
            idx = np.arange(len(values), dtype=float)
            y = values.astype(float)
            denom = len(values) * np.sum(idx ** 2) - np.sum(idx) ** 2
            if denom == 0:
                return 0.0
            num = len(values) * np.sum(idx * y) - np.sum(idx) * np.sum(y)
            return num / denom

        # ------------------------------------------------------------------ #
        # EMA / DIF / RATIO / SIGNAL / HIST
        # ------------------------------------------------------------------ #
        fast_series = _group_series('wap').transform(lambda x: x.ewm(span=fast_span, adjust=False).mean())
        slow_series = _group_series('wap').transform(lambda x: x.ewm(span=slow_span, adjust=False).mean())
        diff_series = fast_series - slow_series
        ratio_series = (fast_series / slow_series.replace(0, np.nan)) - 1

        ema_columns = {
            f'wap_trend_ema_fast_{fast_span}': fast_series,
            f'wap_trend_ema_slow_{slow_span}': slow_series,
            f'wap_trend_diff_{fast_span}_{slow_span}': diff_series,
            f'wap_trend_ratio_{fast_span}_{slow_span}': ratio_series,
        }
        _assign(ema_columns, "ema/diff/ratio")

        signal_group_keys = [df[group_col]]
        if date_col:
            signal_group_keys.append(df[date_col])
        signal_series = diff_series.groupby(signal_group_keys).transform(
            lambda x: x.ewm(span=signal_span, adjust=False).mean()
        )
        hist_series = diff_series - signal_series
        _assign(
            {
                f'wap_trend_signal_{signal_span}': signal_series,
                f'wap_trend_hist_{fast_span}_{slow_span}_{signal_span}': hist_series,
            },
            "signal/hist",
        )

        # ------------------------------------------------------------------ #
        # Slopes / Velocity / Acceleration / Momentum
        # ------------------------------------------------------------------ #
        slope_columns = {
            f'wap_trend_slope_{slope_window}': _group_series('wap', intraday=True).transform(
                lambda x: x.rolling(window=slope_window, min_periods=2).apply(_rolling_slope, raw=True)
            ),
            f'near_price_trend_slope_{slope_window}': _group_series('near_price', intraday=True).transform(
                lambda x: x.rolling(window=slope_window, min_periods=2).apply(_rolling_slope, raw=True)
            ),
        }
        slope_columns[f'wap_trend_vs_near_slope_{slope_window}'] = (
            slope_columns[f'wap_trend_slope_{slope_window}'] - slope_columns[f'near_price_trend_slope_{slope_window}']
        )
        _assign(slope_columns, "rolling slopes")

        velocity_series = _group_series('wap', intraday=True).pct_change(fill_method=None)
        accel_series = _group_series('wap', intraday=True).transform(
            lambda x: x.pct_change(fill_method=None).diff()
        )
        _assign(
            {
                'wap_trend_velocity': velocity_series,
                'wap_trend_acceleration': accel_series,
            },
            "velocity/acceleration",
        )

        for window in list(momentum_windows)[:2]:
            _assign(
                {
                    f'wap_trend_momentum_{window}': _group_series('wap', intraday=True).pct_change(
                        periods=window, fill_method=None
                    ),
                    f'wap_trend_log_return_{window}': _group_series('wap', intraday=True).transform(
                        lambda x, w=window: np.log(x.replace(0, np.nan)).diff(w)
                    ),
                },
                f"momentum_{window}",
            )

        rolling_std = _group_series('wap', intraday=True).transform(
            lambda x: x.rolling(window=slope_window, min_periods=2).std()
        )
        _assign(
            {
                f'wap_trend_strength_{slope_window}': diff_series / (rolling_std + 1e-9),
                f'wap_trend_signal_strength_{signal_span}': signal_series / (rolling_std + 1e-9),
                f'wap_trend_hist_z_{signal_span}': hist_series / (rolling_std + 1e-9),
            },
            "trend strength",
        )

        for window in zscore_windows:
            mean_series = _group_series('wap', intraday=True).transform(
                lambda x, w=window: x.rolling(window=w, min_periods=1).mean()
            )
            std_series = _group_series('wap', intraday=True).transform(
                lambda x, w=window: x.rolling(window=w, min_periods=2).std()
            )
            _assign(
                {f'wap_trend_zscore_{window}': (df['wap'] - mean_series) / (std_series + 1e-9)},
                f"zscore_{window}",
            )

        _assign(
            {
                f'near_price_trend_{slope_window}': _group_series('near_price', intraday=True).pct_change(
                    periods=slope_window, fill_method=None
                ),
                f'reference_price_trend_{slope_window}': _group_series('reference_price', intraday=True).pct_change(
                    periods=slope_window, fill_method=None
                ),
            },
            "price pct-change",
        )

        # ------------------------------------------------------------------ #
        # Mock ratios / deviations
        # ------------------------------------------------------------------ #
        weight_series = df['weight'] if 'weight' in df.columns else pd.Series(1.0, index=df.index)
        weight_series = weight_series.fillna(0.0)
        bucket_index = None
        bucket_weight_sum = None
        if date_col and bucket_col:
            bucket_index = pd.MultiIndex.from_arrays([df[date_col].values, df[bucket_col].values])
            bucket_weight_sum = (
                pd.Series(weight_series.values, index=bucket_index)
                .groupby(level=[0, 1])
                .transform('sum')
                .to_numpy()
            )

        mock_columns: dict[str, pd.Series] = {}
        for period in mock_periods:
            shifted = df.groupby(group_cols_day)['wap'].shift(period)
            ratio = shifted / df['wap'].replace(0, np.nan)
            if bucket_index is not None:
                weighted_sum = (
                    pd.Series((ratio * weight_series).values, index=bucket_index)
                    .groupby(level=[0, 1])
                    .transform('sum')
                    .to_numpy()
                )
                benchmark = weighted_sum / (bucket_weight_sum + 1e-9)
                deviation = (ratio - benchmark) * 10000.0
            else:
                deviation = pd.Series(np.nan, index=df.index)
            mock_columns[f'wap_trend_mock_ratio_{period}'] = ratio
            mock_columns[f'wap_trend_mock_{period}'] = deviation
        _assign(mock_columns, "mock inertia")

        # ------------------------------------------------------------------ #
        # MACD family for price spreads
        # ------------------------------------------------------------------ #
        if 'near_price' in df.columns and 'far_price' in df.columns and 'mid_price_near_far' not in df.columns:
            df['mid_price_near_far'] = (df['near_price'] + df['far_price']) / 2
        if 'reference_price' in df.columns and 'wap' in df.columns and 'imb1_reference_price_wap' not in df.columns:
            df['imb1_reference_price_wap'] = df['reference_price'] - df['wap']

        macd_pairs = list(zip(macd_spans[:-1], macd_spans[1:])) if len(macd_spans) > 1 else []
        ewm_data = {}
        macd_specs: list[tuple[str, str, str, str, int, int]] = []
        for col in ('mid_price_near_far', 'imb1_reference_price_wap', 'near_price'):
            if col not in df.columns:
                continue
            for span in macd_spans:
                ewm_col = f'wap_trend_ewm_{span}_{col}'
                ewm_data[ewm_col] = df.groupby(group_cols_day)[col].transform(
                    lambda x, s=span: x.ewm(span=s, adjust=False).mean()
                )
            for w1, w2 in macd_pairs:
                dif_col = f'wap_trend_macd_dif_{col}_{w1}_{w2}'
                dea_col = f'wap_trend_macd_dea_{col}_{w1}_{w2}'
                macd_col = f'wap_trend_macd_{col}_{w1}_{w2}'
                macd_specs.append((dif_col, dea_col, macd_col, col, w1, w2))

        _assign(ewm_data, "ewm price spreads")

        dif_data = {}
        for dif_col, _, _, col, w1, w2 in macd_specs:
            base1 = f'wap_trend_ewm_{w1}_{col}'
            base2 = f'wap_trend_ewm_{w2}_{col}'
            if base1 in df.columns and base2 in df.columns:
                dif_data[dif_col] = df[base1] - df[base2]
        _assign(dif_data, "macd dif")

        dea_data = {}
        for dif_col, dea_col, _, _, _, _ in macd_specs:
            if dif_col in df.columns:
                dea_data[dea_col] = df.groupby(group_cols_day)[dif_col].transform(
                    lambda x, s=signal_span: x.ewm(span=s, adjust=False).mean()
                )
        _assign(dea_data, "macd dea")

        macd_data = {}
        for dif_col, dea_col, macd_col, _, _, _ in macd_specs:
            if dif_col in df.columns and dea_col in df.columns:
                macd_data[macd_col] = df[dif_col] - df[dea_col]
        _assign(macd_data, "macd hist")

        return df

    # ===================================================================
    @staticmethod
    def auction_all_features(df, group_col='stock_id'):
        """
        一键生成所有 Optiver 拍卖相关【静态】特征

        Args:
            df: DataFrame (必须包含 L1, 拍卖池, near/far 价格数据)
            group_col: 分组列

        Returns:
            df: 添加了所有静态拍卖特征的 DataFrame
        """
        initial_columns = len(df.columns)

        # 1. 生成基础静态因子 (L1, 价格, 匹配效率)
        print("1/7: 生成基础静态特征 (L1, 价格, 匹配效率)...")
        df = Factor.auction_liquidity_profile(df, group_col)
        df = Factor.auction_price_discovery(df, group_col)
        df = Factor.auction_order_book_pressure(df, group_col)

        # 2. 时间紧迫性因子
        print("2/7: 生成时间紧迫性因子...")
        df = Factor.auction_time_urgency(df, group_col)

        # 3. 不平衡方向性因子
        print("3/7: 生成不平衡方向性因子...")
        df = Factor.auction_imbalance_directionality(df, group_col)

        # 4. 价格冲击和效率因子
        print("4/7: 生成价格冲击和效率因子...")
        df = Factor.auction_price_impact(df, group_col)

        # 5. 微观结构因子
        print("5/7: 生成微观结构因子...")
        df = Factor.auction_microstructure(df, group_col)

        # 6. 高级交叉特征
        print("6/7: 生成高级交叉特征...")
        df = Factor.auction_cross_features(df, group_col)

        # 7. 融合静态因子 (需要拍卖池数据)
        print("7/7: 生成融合（L1+拍卖池）静态因子...")
        # (安全检查，确保数据集中有拍卖池深度数据)
        if 'auc_bid_size' in df.columns and 'auc_ask_size' in df.columns:
            df = Factor.auction_fusion_factors(df, group_col)
        else:
            print("警告: 缺少 'auc_bid_size' 或 'auc_ask_size'，跳过 'auction_fusion_factors' 融合因子。")

        # 8 肖力丹价格因子
        print("1/1: 价格因子1")
        df = Factor.lidan_price(df, group_col)

        # 9 赛琳娜价格因子
        
        print("生成拍卖价格压力特征...")
        df = Factor.auction_price_pressure(df, group_col)

        print("生成拍卖不平衡强度特征...")
        df = Factor.auction_imbalance_intensity(df, group_col)

        print("生成参考价格质量特征...")
        df = Factor.auction_reference_price_quality(df, group_col)

        new_features = len(df.columns) - initial_columns
        print(f"\n完成！共新增 {new_features} 个静态因子，当前总列数: {len(df.columns)}")

        return df