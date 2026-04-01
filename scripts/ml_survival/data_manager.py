"""
数据管理模块

负责数据加载、预处理、对齐等数据相关操作。
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sksurv.util import Surv

from .config import Config
from .utils import align_features, summary
from .validator import DataValidator


class DataManager:
    """数据管理器"""

    def __init__(self, config: Config):
        """
        初始化数据管理器

        参数:
            config: 配置对象
        """
        self.config = config
        self.X_train: pd.DataFrame | None = None
        self.X_test: pd.DataFrame | None = None
        self.y_train: pd.DataFrame | None = None
        self.y_test: pd.DataFrame | None = None
        self.train_patients: list[str] = []
        self.test_patients: list[str] = []
        self.validator = DataValidator(strict=True)

    def load_data(self) -> None:
        """加载训练集和测试集数据"""
        logging.info(">>> 加载数据")

        # 加载训练集
        df_train_features = pd.read_csv(self.config.train_features_path)
        df_train_features = df_train_features.filter(regex="^(?!.*diagnostic)")
        df_train_survival = pd.read_csv(self.config.train_survival_path)
        logging.info('训练集数据已加载')

        # 加载测试集
        df_test_features = pd.read_csv(self.config.test_features_path)
        df_test_features = df_test_features.filter(regex="^(?!.*diagnostic)")
        df_test_survival = pd.read_csv(self.config.test_survival_path)
        logging.info('测试集数据已加载')

        # 验证数据格式
        logging.info("验证数据格式...")
        self.validator.validate_dataset(
            df_train_features, df_train_survival, "训练集"
        )
        self.validator.validate_dataset(
            df_test_features, df_test_survival, "测试集"
        )
        logging.info("数据格式验证通过")

        # 预处理数据
        self.X_train, self.y_train = self._prepare_single_dataset(
            df_train_features, df_train_survival
        )
        self.X_test, self.y_test = self._prepare_single_dataset(
            df_test_features, df_test_survival
        )

        # 保存预处理后的数据
        self._save_processed_data()

    def _prepare_single_dataset(
        self,
        df_feature: pd.DataFrame,
        df_surv: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        预处理单个数据集

        参数:
            df_feature: 特征 DataFrame
            df_surv: 生存数据 DataFrame

        返回:
            (X, y): 预处理后的特征和生存数据
        """
        # 准备生存数据
        df_surv['OS'] = df_surv['OS'].astype(float)

        # 检查NA值
        na_mask = df_surv[["events", "OS"]].isna().any(axis=1)

        # 检查生存时间为0
        if self.config.remove_zero:
            zero_mask = df_surv["OS"] <= 0
        else:
            zero_mask = pd.Series([False] * len(df_surv), index=df_surv.index)

        # 合并需要移除的患者
        remove_mask = na_mask | zero_mask
        patients_with_na = df_surv.loc[na_mask, "patient"].tolist()
        patients_with_zero = df_surv.loc[zero_mask, "patient"].tolist() if self.config.remove_zero else []

        if remove_mask.any():
            if patients_with_na:
                logging.warning(f"警告：发现 {len(patients_with_na)} 个患者的生存数据包含NA值")
            if patients_with_zero:
                logging.warning(f"警告：发现 {len(patients_with_zero)} 个患者的生存时间 <= 0")
            logging.info(f"这些患者将从分析中排除")

            # 过滤掉问题患者
            df_surv = df_surv[~remove_mask].reset_index(drop=True)
        else:
            logging.info(f"没有患者被过滤")

        # 选择共有患者
        logging.info(f"纳入样本量：{len(df_surv)}, 事件率: {df_surv['events'].astype(bool).values.sum() / len(df_surv):.2%}")
        patients = list(set(df_surv["patient"]) & set(df_feature["patient"]))
        df_surv = df_surv.set_index("patient").loc[patients].reset_index()

        event = df_surv["events"].astype(bool).values
        time = df_surv["OS"].astype(float).values
        logging.info(f"与特征矩阵共有的样本量：{len(patients)}, 事件率: {event.sum() / len(event):.2%}")

        # 准备特征数据
        X = df_feature.set_index("patient").loc[patients].reset_index()
        X.columns = [col.replace('-', '_') for col in X.columns]
        X = X.drop(columns=["patient", "Num", "Date"], errors="ignore")

        # 保留患者列表作为索引，返回纯数值特征矩阵
        df_feature = X.select_dtypes(include=[np.number]).fillna(0)
        df_feature.index = patients

        # 只保留生存分析必要的列
        df_surv_clean = df_surv[["patient", "events", "OS"]].copy()

        return df_feature, df_surv_clean

    def clean_outliers(self, X: pd.DataFrame, method: str = 'clip') -> pd.DataFrame:
        """
        处理极端异常值

        参数:
            X: 特征 DataFrame
            method: 'clip' 截断, 'log' 对数变换

        返回:
            处理后的特征 DataFrame
        """
        X_clean = X.copy()
        logging.info('  原始数据：')
        summary(X)
        logging.info(f'clean_outliers method is {method}')

        percentile = (1, 99)

        if method == 'clip':
            # 对每列进行分位数截断
            for col in X_clean.columns:
                lower = X_clean[col].quantile(percentile[0] / 100)
                upper = X_clean[col].quantile(percentile[1] / 100)
                X_clean[col] = X_clean[col].clip(lower=lower, upper=upper)

        elif method == 'log':
            # 对数变换（对每列独立偏移确保为正）
            for col in X_clean.columns:
                col_min = X_clean[col].min()
                if col_min <= 0:
                    # 仅对该列进行平移
                    X_clean[col] = X_clean[col] - col_min + 1
                # 使用 log1p 保证数值稳定性
                X_clean[col] = np.log1p(X_clean[col])

        logging.info('  处理后：')
        summary(X_clean)
        return X_clean

    def prepare_data(self) -> None:
        """数据预处理（清洗、对齐、索引设置）"""
        logging.info(">>> 数据预处理")

        # 初始化 raw 变量
        X_train_raw = self.X_train.copy()
        X_test_raw = self.X_test.copy()

        # 清理异常值
        if self.config.clean_outliers:
            logging.info("清理极端异常值...")
            X_train_raw = self.clean_outliers(
                self.X_train,
                method=self.config.outlier_method
            )
            X_test_raw = self.clean_outliers(
                self.X_test,
                method=self.config.outlier_method
            )
            if self.config.diagnostic:
                logging.info(f"  [诊断] 训练集异常值处理后: 范围=[{X_train_raw.min().min():.2f}, {X_train_raw.max().max():.2f}]")
                logging.info(f"  [诊断] 验证集异常值处理后: 范围=[{X_test_raw.min().min():.2f}, {X_test_raw.max().max():.2f}]")

        # 特征对齐
        logging.info("特征对齐中...")
        self.X_train, self.X_test = align_features(X_train_raw, X_test_raw)
        logging.info("✓ 特征对齐完成\n")

    def _save_processed_data(self) -> None:
        """保存预处理后的数据"""
        data_folder = self.config.output_dir / 'data'
        data_folder.mkdir(exist_ok=True, parents=True)

        self.X_train.to_csv(data_folder / "X_train.csv")
        self.y_train.to_csv(data_folder / "y_train.csv", index=False)
        self.X_test.to_csv(data_folder / "X_test.csv")
        self.y_test.to_csv(data_folder / "y_test.csv", index=False)

        logging.info(f"预处理后的数据已保存至: {data_folder}")

    def get_train_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        获取训练数据

        返回:
            (X_train, y_train): 训练集特征和生存数据
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("训练数据尚未加载，请先调用 load_data()")
        return self.X_train, self.y_train

    def get_test_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        获取测试数据

        返回:
            (X_test, y_test): 测试集特征和生存数据
        """
        if self.X_test is None or self.y_test is None:
            raise ValueError("测试数据尚未加载，请先调用 load_data()")
        return self.X_test, self.y_test
