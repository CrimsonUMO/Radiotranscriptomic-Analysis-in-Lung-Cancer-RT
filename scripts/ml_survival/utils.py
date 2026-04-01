"""
工具函数模块

包含随机性控制、数据加载、CV划分等通用工具函数。
"""

import json
import logging
import os
import random

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any

from sklearn.model_selection import StratifiedKFold
from sksurv.metrics import concordance_index_censored


def lock_random(seed: int = 42) -> np.random.Generator:
    """
    锁定所有随机性来源，确保完全可重复

    包括：
    - Python random, NumPy 随机数生成器
    - 多线程库（OpenMP, OpenBLAS, MKL等）
    - Python hash seed
    - scikit-learn 全局随机状态
    """
    # 1. 基础随机数生成器
    rng = np.random.default_rng(seed)
    random.seed(seed)
    np.random.seed(seed)

    # 2. 环境变量（控制多线程库）
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    # 3. scikit-learn 全局随机状态（新增）
    try:
        import sklearn
        sklearn.random_state = seed
    except Exception:
        pass  # 版本兼容性处理

    return rng


def load_cv_splits(cv_json_path: Path) -> list[dict[str, Any]]:
    """
    从JSON文件加载CV划分

    参数:
        cv_json_path: CV划分JSON文件路径

    返回:
        cv_splits: CV划分列表，每个元素包含 fold_id, train_patients, val_patients
    """
    with open(cv_json_path, 'r', encoding='utf-8') as f:
        cv_splits = json.load(f)
        logging.info(f"  √ 已加载 {len(cv_splits)} 个 fold")

    for i, fold in enumerate(cv_splits):
        assert "train_patients" in fold, f"Fold {i} 缺少 'train_patients'"
        assert "val_patients" in fold, f"Fold {i} 缺少 'val_patients'"

    return cv_splits


def create_cv_splits(
    df_meta: pd.DataFrame,
    n_splits: int = 5,
    random_state: int | None = None,
    cv_json_path: Path | None = None
) -> list[dict[str, Any]]:
    """
    创建新的CV划分

    参数:
        df_meta: 包含 patient 和 events 列的元数据 DataFrame
        n_splits: 交叉验证折数
        random_state: 随机种子
        cv_json_path: 保存CV划分的JSON文件路径（如果提供）

    返回:
        cv_splits: CV划分列表
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_splits = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df_meta, df_meta["events"])):
        train_pids = df_meta.iloc[train_idx]["patient"].tolist()
        val_pids = df_meta.iloc[val_idx]["patient"].tolist()
        cv_splits.append({
            "fold_id": fold_idx,
            "train_patients": train_pids,
            "val_patients": val_pids
        })

    if cv_json_path is not None:
        with open(cv_json_path, 'w') as f:
            json.dump(cv_splits, f, indent=4)

    return cv_splits


def align_features(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    确保训练集和验证集具有相同的特征列

    参数:
        X_train_raw: 训练集特征 DataFrame (索引为患者ID)
        X_test_raw: 验证集特征 DataFrame (索引为患者ID)

    返回:
        X_train_aligned, X_test_aligned: 对齐后的特征 DataFrame
    """
    # 获取共同特征列
    train_features = set(X_train_raw.columns)
    test_features = set(X_test_raw.columns)

    common_features = train_features & test_features

    if len(common_features) < len(train_features) or len(common_features) < len(test_features):
        missing_in_test = train_features - test_features
        missing_in_train = test_features - train_features

        if missing_in_test:
            logging.info(f"警告：验证集缺少 {len(missing_in_test)} 个特征")
        if missing_in_train:
            logging.info(f"警告：训练集缺少 {len(missing_in_train)} 个特征")

        logging.info(f"使用 {len(common_features)} 个共同特征")

    common_features = sorted(list(common_features))

    # 直接选择共同特征列，索引已经是患者ID
    X_train_aligned = X_train_raw[common_features]
    X_test_aligned = X_test_raw[common_features]

    return X_train_aligned, X_test_aligned


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算并显示特征统计摘要

    参数:
        df: 特征 DataFrame

    返回:
        stats_summary: 统计摘要 DataFrame
    """
    variances = df.var()  # 默认ddof=1（样本方差）

    # 计算每列的最小值和最大值（分布区间）
    min_values = df.min()
    max_values = df.max()

    # 合并结果
    stats_summary = pd.DataFrame({
        '方差': variances,
        '最小值': min_values,
        '最大值': max_values,
        '极差': max_values - min_values
    })

    print("各列方差和分布区间:")
    print(stats_summary)

    return stats_summary


def unpack_single_result(result: dict) -> pd.DataFrame:
    """
    解包从 train_and_validate 函数返回的单个结果

    参数:
        result: 训练结果字典 {model_name: {'train': {...}, 'vali': {...}}}

    返回:
        summary_df: 多层索引的摘要 DataFrame
    """
    # 选择评价指标名
    metric_names = [
        list(ds_result.keys())
        for model_result in result.values()
        for ds_result in model_result.values()
    ]
    metric_names = set(metric_names[0]).intersection(*metric_names[1:])
    metric_names = list(metric_names)
    metric_names = [i for i in metric_names if 'risk' not in i]
    metric_names.sort()
    print(f"Metric names: {metric_names}")

    # 解包结果字典
    data = []
    index_tuples = []  # 用于多层索引
    for model_name, model_result in result.items():
        for ds_name, ds_result in model_result.items():
            # 创建数据行
            row_data = {metric: ds_result.get(metric) for metric in metric_names}
            data.append(row_data)
            index_tuples.append((model_name, ds_name))

    # 创建多层索引
    multi_index = pd.MultiIndex.from_tuples(
        index_tuples,
        names=['Model', 'Dataset']
    )
    # 创建完整DataFrame
    summary_df = pd.DataFrame(data, index=multi_index, columns=metric_names)
    return summary_df


def cindex_score(y_true: pd.DataFrame, y_pred: np.ndarray) -> float:
    """
    计算C-index

    参数:
        y_true: 真实生存数据（包含 events, OS 列）
        y_pred: 预测的风险评分

    返回:
        C-index 值
    """
    return concordance_index_censored(y_true["events"], y_true["OS"], y_pred)[0]
