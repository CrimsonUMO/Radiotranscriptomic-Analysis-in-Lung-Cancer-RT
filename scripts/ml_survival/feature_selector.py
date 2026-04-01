"""
特征选择模块

包含特征选择管道：方差过滤 → 相关性过滤 → 单变量Cox选择
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.feature_selection import VarianceThreshold
from sksurv.metrics import concordance_index_censored

from .config import Config


class CorrelationFilter:
    """相关性过滤器"""

    def __init__(self, threshold: float = 0.9, method: str = "pearson"):
        """
        初始化相关性过滤器

        参数:
            threshold: 相关系数阈值，超过此阈值的特征将被移除
            method: 相关系数计算方法 ('pearson', 'spearman', 'kendall')
        """
        self.threshold = threshold
        self.method = method
        self.keep_features_: list[str] | None = None

    def fit(self, X: pd.DataFrame) -> 'CorrelationFilter':
        """
        拟合相关性过滤器

        参数:
            X: 特征 DataFrame

        返回:
            self: 拟合后的过滤器
        """
        corr = X.corr(method=self.method).abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [
            col for col in upper.columns if any(upper[col] > self.threshold)
        ]
        self.keep_features_ = [col for col in X.columns if col not in to_drop]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        应用特征过滤

        参数:
            X: 特征 DataFrame

        返回:
            过滤后的特征 DataFrame
        """
        if self.keep_features_ is None:
            raise ValueError("Must fit before transform.")
        return X[self.keep_features_]

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        拟合并应用特征过滤

        参数:
            X: 特征 DataFrame

        返回:
            过滤后的特征 DataFrame
        """
        return self.fit(X).transform(X)


class UnivariateCoxSelector:
    """单变量Cox选择器"""

    def __init__(self, alpha: float = 0.05, k: int | None = None):
        """
        初始化单变量Cox选择器

        参数:
            alpha: p值显著性水平
            k: 保留的前K个特征（如果没有特征通过alpha阈值）
        """
        self.alpha = alpha
        self.k = k
        self.selected_features_: list[str] = []
        self.feature_pvals_: dict[str, float] = {}
        self.feature_cindex_: dict[str, float] = {}
        self.summary_df_: pd.DataFrame | None = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> 'UnivariateCoxSelector':
        """
        拟合单变量Cox选择器

        参数:
            X: 特征 DataFrame
            y: 生存数据 DataFrame (包含 events, OS, patient 列)

        返回:
            self: 拟合后的选择器
        """
        # 标准化 y 为 DataFrame
        if isinstance(y, pd.DataFrame):
            df_y = y.copy()
        else:
            # 关键修复：使用 X.index 确保拼接时 index 匹配
            df_y = pd.DataFrame({"events": y["events"], "OS": y["OS"]}, index=X.index)

        df_y = df_y.set_index('patient')

        # 只保留必要的列（events, OS），移除其他临床变量（如 SEX, AGE, stage 等）
        df_y = df_y[['events', 'OS']]

        print("X 中 NaN 总数:", X.isna().sum().sum())
        print("y 中 NaN 总数:", df_y.isna().sum().sum())
        if X.isna().sum().sum() > 0:
            print("X 中包含 NaN 的列:", X.columns[X.isna().any()].tolist())

        pvals = {}
        cindices = {}  # 保存每个特征的 C-index

        for col in X.columns:
            df = pd.concat([X[[col]], df_y], axis=1)
            try:
                cph = CoxPHFitter(penalizer=0.1)
                cph.fit(df, duration_col="OS", event_col="events", show_progress=False)
                pvals[col] = cph.summary.loc[col, "p"]
                risk_scores = cph.predict_partial_hazard(df).values.ravel()
                ci = concordance_index_censored(
                    df_y["events"].astype(bool),
                    df_y["OS"],
                    risk_scores
                )[0]
                cindices[col] = ci if not np.isnan(ci) else 0.5  # NaN 时设为随机水平

            except Exception as e:
                # 拟合失败：p = NaN, C-index = 0.5（无判别力）
                pvals[col] = np.nan
                cindices[col] = 0.5
                print(e)
                print('Failed')

        self.feature_pvals_ = pvals
        self.feature_cindex_ = cindices

        # 创建摘要
        summary_data = {
            'p_value': pvals,
            'c_index': cindices
        }
        self.summary_df_ = pd.DataFrame(summary_data).sort_values(by='c_index', ascending=False)
        print("\n=== Univariate Cox 分析摘要 (describe()) ===")
        print(self.summary_df_.describe())
        print("\n=== Top 5 特征 (by C-index) ===")
        print(self.summary_df_.head(5))
        print("=" * 50)

        # 第一步：筛选 p < alpha 的显著特征
        significant = [f for f, p in pvals.items() if not pd.isna(p) and p < self.alpha]

        if significant:
            print(f"{len(significant)}个特征在 α={self.alpha} 下显著。")
            # 按 C-index 降序排序（越高越好）
            significant.sort(key=lambda f: cindices[f], reverse=True)

            # 如果显著特征数 >= k，取前 k 个
            if len(significant) >= self.k:
                self.selected_features_ = significant[:self.k]
            else:
                # 如果显著特征数 < k，补充 C-index 最高的非显著特征
                print(f"警告：显著特征数({len(significant)}) < k({self.k})，补充TOP特征")
                remaining = [f for f in cindices.keys() if f not in significant]
                remaining.sort(key=lambda f: cindices[f], reverse=True)
                needed = self.k - len(significant)
                self.selected_features_ = significant + remaining[:needed]
        else:
            # 无显著特征：fallback 到 top-k by C-index
            self.selected_features_ = []
            print(f"警告：无特征在 α={self.alpha} 下显著。回退到TOP特征")
            top_k_by_cindex = sorted(cindices.keys(), key=lambda f: cindices[f], reverse=True)[:self.k]
            self.selected_features_ = top_k_by_cindex

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        应用特征选择

        参数:
            X: 特征 DataFrame

        返回:
            选择后的特征 DataFrame
        """
        if not self.selected_features_:
            return X[[]]
        return X[self.selected_features_]

    def fit_transform(self, X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        """
        拟合并应用特征选择

        参数:
            X: 特征 DataFrame
            y: 生存数据 DataFrame

        返回:
            选择后的特征 DataFrame
        """
        return self.fit(X, y).transform(X)

    def get_summary(self) -> pd.DataFrame:
        """
        返回包含 p_value 和 c_index 的完整DataFrame

        返回:
            摘要 DataFrame
        """
        if self.summary_df_ is None:
            raise ValueError("尚未调用 fit() 方法。")
        return self.summary_df_.copy()


class FeatureSelector:
    """特征选择管道"""

    def __init__(self, config: Config):
        """
        初始化特征选择管道

        参数:
            config: 配置对象
        """
        self.config = config
        self.variance_selector = VarianceThreshold(threshold=config.variance_threshold)
        self.corr_filter: CorrelationFilter | None = None
        self.cox_selector: UnivariateCoxSelector | None = None
        self.selected_features_: list[str] = []

    def fit_transform(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_test: pd.DataFrame | None = None
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        """
        拟合特征选择器并转换数据

        参数:
            X_train: 训练集特征矩阵
            y_train: 训练集生存目标
            X_test: 可选的测试集特征矩阵

        返回:
            如果 X_test 为 None: X_selected
            如果 X_test 不为 None: (X_train_selected, X_test_selected)
        """
        # 方差过滤
        self.variance_selector.fit(X_train)
        X_train_var = X_train.loc[:, self.variance_selector.get_support()]

        if self.config.diagnostic:
            print(f"  [诊断] 方差过滤后特征数: {X_train_var.shape[1]}")

        # 相关性过滤
        self.corr_filter = CorrelationFilter(threshold=self.config.corr_threshold)
        X_train_corr = self.corr_filter.fit_transform(X_train_var)

        if self.config.diagnostic:
            print(f"  [诊断] 相关性过滤后特征数: {X_train_corr.shape[1]}")

        # 单变量Cox
        if self.config.enable_cox:
            print(f"→ Applying UnivariateCoxSelector (alpha={self.config.cox_alpha}, k={self.config.cox_k})")
            self.cox_selector = UnivariateCoxSelector(
                alpha=self.config.cox_alpha,
                k=self.config.cox_k
            )
            X_train_final = self.cox_selector.fit_transform(X_train_corr, y_train)
        else:
            print("→ Skipping Univariate Cox selection (disabled)")
            X_train_final = X_train_corr

        print(f"最终特征数: {X_train_final.shape[1]}")

        if self.config.diagnostic:
            if X_train_final.shape[1] == 0:
                print("  [诊断] 警告：最终特征数为0！")
            else:
                print(f"  [诊断] 特征统计:")
                print(f"    - 形状: {X_train_final.shape}")
                print(f"    - 范围: [{X_train_final.min().min():.4f}, {X_train_final.max().max():.4f}]")
                print(f"    - 均值: {X_train_final.mean().mean():.4f}")
                print(f"    - 标准差: {X_train_final.values.flatten().std():.4f}")

        # 保存选中的特征列表
        self.selected_features_ = X_train_final.columns.tolist()

        # 处理测试集
        if X_test is not None:
            X_test_final = X_test[self.selected_features_]
            return X_train_final, X_test_final

        return X_train_final

    def get_selected_features(self) -> list[str]:
        """
        获取选中的特征列表

        返回:
            选中的特征名称列表
        """
        return self.selected_features_

    def get_feature_importance(self) -> pd.DataFrame | None:
        """
        获取特征重要性（如果有Cox结果）

        返回:
            特征重要性 DataFrame，如果没有Cox结果则返回 None
        """
        if self.cox_selector is not None:
            return self.cox_selector.get_summary()
        return None
