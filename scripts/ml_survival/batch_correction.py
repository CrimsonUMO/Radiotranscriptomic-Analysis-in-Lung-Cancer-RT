"""
批次效应校正模块

负责去除训练集和测试集之间的批次效应。
"""

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .config import Config


class BatchCorrector:
    """批次效应校正器"""

    def __init__(self, config: Config):
        """
        初始化批次校正器

        参数:
            config: 配置对象
        """
        self.config = config

    def extract_batch_from_patient(self, patient_ids: list) -> pd.Series:
        """
        从患者ID提取批次标签

        参数:
            patient_ids: 患者ID列表（可能是4/5位数字、TCGA格式或p+数字格式）

        返回:
            pd.Series: 批次标签，索引为患者ID
        """
        batches = {}
        for pid in patient_ids:
            pid_str = str(pid)

            # TCGA格式 → 单一批次'TCGA'
            if pid_str.startswith('TCGA-'):
                batches[pid] = 'TCGA'
            # p+数字格式（如p003308）→ 单一批次'Train'
            elif pid_str.startswith('p'):
                batches[pid] = 'Train'
            # 纯数字ID → 补零后提取前2位作为中心代码
            else:
                pid_num = int(float(pid_str))
                pid_str_padded = str(pid_num).zfill(5)
                batch = pid_str_padded[:2]
                batches[pid] = batch

        return pd.Series(batches, name='batch')

    def remove_batch_effects(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        train_batches: pd.Series,
        test_batches: pd.Series
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        去除训练集和验证集的批次效应
        训练集作为reference不被修改，只校正验证集

        参数:
            X_train: 训练集特征 DataFrame（索引为患者ID）
            X_test: 验证集特征 DataFrame（索引为患者ID）
            train_batches: 训练集批次标签 Series
            test_batches: 验证集批次标签 Series

        返回:
            X_train_corrected, X_test_corrected: 校正后的特征矩阵
        """
        # 保存校正前的数据副本（用于评估）
        X_train_before = X_train.copy()
        X_test_before = X_test.copy()
        print(f"{X_train_before.shape = }, {X_test_before.shape = }")

        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_before)
        print(f"{X_train_scaled.shape = }")
        X_test_scaled = scaler.transform(X_test_before)

        # 获取批次信息
        train_unique_batches = train_batches.unique()
        test_unique_batches = test_batches.unique()

        if self.config.diagnostic:
            print(f"\n  [批次信息]")
            print(f"    方法: {self.config.batch_method}")
            print(f"    训练集批次: {train_unique_batches[0]} (reference, 不被修改)")
            print(f"    验证集批次数: {len(test_unique_batches)}")
            print(f"    验证集批次分布:")
            for batch in sorted(test_unique_batches):
                n_samples = (test_batches == batch).sum()
                pct = 100 * n_samples / len(test_batches)
                print(f"      {batch}: {n_samples} 样本 ({pct:.1f}%)")

        # 根据方法选择校正算法
        if self.config.batch_method == 'combat':
            X_train_corrected, X_test_corrected = self._combat_correction(
                pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns),
                pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns),
                train_batches, test_batches
            )
        elif self.config.batch_method == 'mean_centering':
            X_train_corrected, X_test_corrected = self._mean_centering_correction(
                pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns),
                pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns),
                train_batches, test_batches
            )
        else:
            raise ValueError(f"不支持的批次校正方法: {self.config.batch_method}")

        if self.config.diagnostic:
            print(f"  [校正后统计]")
            print(f"    训练集: 范围=[{X_train_corrected.min().min():.2f}, "
                  f"{X_train_corrected.max().max():.2f}] (未修改)")
            print(f"    验证集: 范围=[{X_test_corrected.min().min():.2f}, "
                  f"{X_test_corrected.max().max():.2f}] (已校正)")

        # 自动评估批次校正效果
        if self.config.output_dir is not None:
            self.evaluate_correction(
                X_train_before, X_test_before,
                X_train_corrected, X_test_corrected,
                train_batches, test_batches
            )

        return X_train_corrected, X_test_corrected

    def _combat_correction(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        train_batches: pd.Series,
        test_batches: pd.Series
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        ComBat批次校正（经验贝叶斯方法）
        训练集作为reference不被修改，只校正验证集

        参数:
            X_train: 训练集特征 DataFrame（reference）
            X_test: 验证集特征 DataFrame（需要校正）
            train_batches: 训练集批次标签 Series
            test_batches: 验证集批次标签 Series

        返回:
            X_train_corrected, X_test_corrected: 校正后的特征矩阵
        """
        # 训练集保持不变
        X_train_corrected = X_train.copy()

        # 转换为numpy数组
        X_test_array = X_test.values.astype(float)
        train_batch_array = train_batches.values
        test_batch_array = test_batches.values

        # 获取训练集批次
        train_unique_batches = np.unique(train_batch_array)
        test_unique_batches = np.unique(test_batch_array)

        n_features = X_test.shape[1]
        X_test_corrected = X_test_array.copy()

        # 获取训练集作为reference的统计量
        train_array = X_train.values.astype(float)

        # 选择样本最多的批次作为 reference（而不是第一个）
        batch_counts = pd.Series(train_batch_array).value_counts()
        ref_batch = batch_counts.idxmax()
        ref_mask = train_batch_array == ref_batch

        diagnostic = self.config.diagnostic
        if diagnostic:
            print(f"  [ComBat] 训练集批次分布:")
            for batch, count in batch_counts.items():
                ref_marker = " (reference)" if batch == ref_batch else ""
                print(f"    {batch}: {count} 样本{ref_marker}")
            print(f"  [ComBat] 验证集批次数: {len(test_unique_batches)}")
            for batch in test_unique_batches:
                count = (test_batch_array == batch).sum()
                print(f"    {batch}: {count} 样本")

        # 检查参考批次样本数是否足够
        min_ref_samples = 3  # 最小参考样本数
        if ref_mask.sum() < min_ref_samples:
            logging.error(f"  [ComBat 错误] Reference批次样本数({ref_mask.sum()}) < {min_ref_samples}，无法进行校正")
            logging.error(f"  [ComBat 跳过] 跳过ComBat校正，返回原始数据")
            return X_train_corrected, X_test.copy()

        # 跳过的特征计数
        skipped_features = 0
        corrected_features = 0

        # 对每个特征进行校正
        for feature_idx in range(n_features):
            # 训练集（reference）的均值和标准差
            ref_values = train_array[ref_mask, feature_idx]

            # 检查 reference 是否有效
            if np.isnan(ref_values).any() or np.isinf(ref_values).any():
                if diagnostic and skipped_features == 0:
                    print(f"  [警告] 特征 '{X_test.columns[feature_idx]}' reference 包含 NaN/Inf，跳过校正")
                skipped_features += 1
                continue

            ref_mean = np.mean(ref_values)
            ref_std = np.std(ref_values, ddof=1) + 1e-10

            # 对验证集的每个批次进行校正
            for batch in test_unique_batches:
                batch_mask = test_batch_array == batch
                batch_values = X_test_array[batch_mask, feature_idx]

                if len(batch_values) == 0:
                    continue

                # 检查批次是否有效
                if np.isnan(batch_values).any() or np.isinf(batch_values).any():
                    if diagnostic and corrected_features == 0:
                        print(f"  [警告] 特征 '{X_test.columns[feature_idx]}' 批次 {batch} 包含 NaN/Inf，保持原值")
                    continue

                batch_mean = np.mean(batch_values)
                batch_std = np.std(batch_values, ddof=1) + 1e-10

                # 校正公式
                corrected = (
                    (batch_values - batch_mean) / batch_std
                ) * ref_std + ref_mean

                # 裁剪 Inf/NaN 到合理范围
                corrected = np.nan_to_num(
                    corrected,
                    nan=ref_mean,
                    posinf=ref_mean + 3 * ref_std,
                    neginf=ref_mean - 3 * ref_std
                )

                X_test_corrected[batch_mask, feature_idx] = corrected

            corrected_features += 1

        if diagnostic:
            print(f"  [ComBat校正完成] 校正特征数: {corrected_features}, 跳过特征数: {skipped_features}")

        return X_train_corrected, pd.DataFrame(X_test_corrected, columns=X_test.columns, index=X_test.index)

    def _mean_centering_correction(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        train_batches: pd.Series,
        test_batches: pd.Series
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        简单均值中心化批次校正
        训练集作为reference不被修改，只校正验证集

        参数:
            X_train: 训练集特征 DataFrame（reference）
            X_test: 验证集特征 DataFrame（需要校正）
            train_batches: 训练集批次标签 Series
            test_batches: 验证集批次标签 Series

        返回:
            X_train_corrected, X_test_corrected: 校正后的特征矩阵
        """
        # 训练集保持不变
        X_train_corrected = X_train.copy()

        # 计算训练集的全局均值（reference分布）
        train_array = X_train.values.astype(float)

        # 检查训练集是否有 NaN/Inf
        if np.isnan(train_array).any() or np.isinf(train_array).any():
            if self.config.diagnostic:
                print(f"  [警告] 训练集包含 NaN/Inf，跳过均值中心化校正")
            return X_train_corrected, X_test.copy()

        train_mean = np.mean(train_array, axis=0)

        X_test_array = X_test.values.astype(float)
        test_batch_array = test_batches.values
        test_unique_batches = np.unique(test_batch_array)

        X_test_corrected = X_test_array.copy()

        skipped_features = 0
        corrected_features = 0

        # 对验证集的每个批次进行中心化
        for feature_idx in range(X_test.shape[1]):
            feature_name = X_test.columns[feature_idx]

            # 检查训练集该特征是否有效
            if np.isnan(train_mean[feature_idx]) or np.isinf(train_mean[feature_idx]):
                if self.config.diagnostic and skipped_features == 0:
                    print(f"  [警告] 特征 '{feature_name}' train_mean 为 NaN/Inf，跳过")
                skipped_features += 1
                continue

            for batch in test_unique_batches:
                batch_mask = test_batch_array == batch
                batch_data = X_test_array[batch_mask, feature_idx]

                # 检查批次数据是否有效
                if np.isnan(batch_data).any() or np.isinf(batch_data).any():
                    if self.config.diagnostic and corrected_features == 0:
                        print(f"  [警告] 特征 '{feature_name}' 批次 {batch} 包含 NaN/Inf，保持原值")
                    continue

                batch_mean = np.mean(batch_data)

                # 中心化: batch_value - batch_mean + train_mean
                corrected = batch_data - batch_mean + train_mean[feature_idx]

                # 裁剪可能的 Inf/NaN
                corrected = np.nan_to_num(
                    corrected,
                    nan=train_mean[feature_idx],
                    posinf=train_mean[feature_idx] + 3 * np.std(train_array[:, feature_idx]),
                    neginf=train_mean[feature_idx] - 3 * np.std(train_array[:, feature_idx])
                )

                X_test_corrected[batch_mask, feature_idx] = corrected

            corrected_features += 1

        if self.config.diagnostic:
            print(f"  [均值中心化校正完成] 校正特征数: {corrected_features}, 跳过特征数: {skipped_features}")

        return X_train_corrected, pd.DataFrame(X_test_corrected, columns=X_test.columns, index=X_test.index)

    def evaluate_correction(
        self,
        X_train_before: pd.DataFrame,
        X_test_before: pd.DataFrame,
        X_train_after: pd.DataFrame,
        X_test_after: pd.DataFrame,
        train_batches: pd.Series,
        test_batches: pd.Series
    ) -> dict[str, Any]:
        """
        评估批次效应校正效果

        参数:
            X_train_before: 校正前训练集特征
            X_test_before: 校正前验证集特征
            X_train_after: 校正后训练集特征
            X_test_after: 校正后验证集特征
            train_batches: 训练集批次标签
            test_batches: 验证集批次标签

        返回:
            evaluation_results: 评估结果字典
        """
        # 使用 Agg 后端
        matplotlib.use('Agg')

        # 创建 batch 子目录
        batch_dir = self.config.output_dir / "batch"
        batch_dir.mkdir(exist_ok=True, parents=True)

        print(f"\n[批次效应评估] 保存结果至: {batch_dir}")

        # 准备数据
        X_before = pd.concat([X_train_before, X_test_before])
        X_after = pd.concat([X_train_after, X_test_after])
        labels = [0] * len(X_train_before) + [1] * len(X_test_before)

        results = {}

        # 1. 数据范围统计
        print(f"  [数据范围对比]")
        stats = {
            'train_before_min': X_train_before.min().min(),
            'train_before_max': X_train_before.max().max(),
            'train_before_mean': X_train_before.mean().mean(),
            'train_before_std': X_train_before.std().mean(),
            'test_before_min': X_test_before.min().min(),
            'test_before_max': X_test_before.max().max(),
            'test_before_mean': X_test_before.mean().mean(),
            'test_before_std': X_test_before.std().mean(),
            'test_after_min': X_test_after.min().min(),
            'test_after_max': X_test_after.max().max(),
            'test_after_mean': X_test_after.mean().mean(),
            'test_after_std': X_test_after.std().mean(),
        }
        results['statistics'] = stats

        print(f"    训练集: [{stats['train_before_min']:.2f}, {stats['train_before_max']:.2f}] (未修改)")
        print(f"    验证集(校正前): [{stats['test_before_min']:.2f}, {stats['test_before_max']:.2f}]")
        print(f"    验证集(校正后): [{stats['test_after_min']:.2f}, {stats['test_after_max']:.2f}]")

        # 2. PCA 可视化
        print(f"  [生成 PCA 可视化]")
        pca = PCA(n_components=2)
        X_pca_before = pca.fit_transform(X_before)

        pca_after = PCA(n_components=2)
        X_pca_after = pca_after.fit_transform(X_after)

        fig, axes = plt.subplots(1, 2, figsize=(6, 3))

        # 校正前
        axes[0].scatter(X_pca_before[:len(X_train_before), 0], X_pca_before[:len(X_train_before), 1],
                       c='blue', alpha=0.6, s=50, label='Train')
        axes[0].scatter(X_pca_before[len(X_train_before):, 0], X_pca_before[len(X_train_before):, 1],
                       c='red', alpha=0.6, s=50, label='Test')
        axes[0].set_title('Before Correction', fontsize=14, fontweight='bold')
        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        axes[0].legend()

        # 校正后
        axes[1].scatter(X_pca_after[:len(X_train_after), 0], X_pca_after[:len(X_train_after), 1],
                       c='blue', alpha=0.6, s=50, label='Train')
        axes[1].scatter(X_pca_after[len(X_train_after):, 0], X_pca_after[len(X_train_after):, 1],
                       c='red', alpha=0.6, s=50, label='Test')
        axes[1].set_title('After Correction', fontsize=14, fontweight='bold')
        axes[1].set_xlabel(f'PC1 ({pca_after.explained_variance_ratio_[0]*100:.1f}%)')
        axes[1].set_ylabel(f'PC2 ({pca_after.explained_variance_ratio_[1]*100:.1f}%)')
        axes[1].legend()

        plt.tight_layout()
        pca_file = batch_dir / f"pca_comparison.{self.config.figure_format}"
        dpi_param = 300 if self.config.figure_format == 'png' else None
        plt.savefig(pca_file, dpi=dpi_param, bbox_inches='tight', format=self.config.figure_format)
        plt.close()
        print(f"    ✓ 保存: {pca_file.name}")

        # 3. Silhouette Score
        print(f"  [计算 Silhouette Score]")
        pca_10d = PCA(n_components=10)
        X_pca_10d_before = pca_10d.fit_transform(X_before)

        pca_10d_after = PCA(n_components=10)
        X_pca_10d_after = pca_10d_after.fit_transform(X_after)

        score_before = silhouette_score(X_pca_10d_before, labels)
        score_after = silhouette_score(X_pca_10d_after, labels)
        score_change = score_after - score_before

        results['silhouette_score'] = {
            'before': float(score_before),
            'after': float(score_after),
            'change': float(score_change)
        }

        print(f"    校正前: {score_before:.4f}")
        print(f"    校正后: {score_after:.4f}")
        print(f"    变化: {score_change:+.4f}")

        # 4. 结论
        if score_after < score_before - 0.05:
            verdict = "✓ 批次混合度显著改善"
            verdict_code = "significant_improvement"
        elif score_after < score_before:
            verdict = "~ 批次混合度略有改善"
            verdict_code = "slight_improvement"
        else:
            verdict = "✗ 批次混合度未改善"
            verdict_code = "no_improvement"

        results['verdict'] = verdict_code
        results['verdict_message'] = verdict

        print(f"  [结论] {verdict}")

        # 5. 保存结果到 JSON
        results_file = batch_dir / "evaluation_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  ✓ 保存评估结果: {results_file.name}")

        return results
