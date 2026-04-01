"""
模型工厂模块

负责创建和管理模型，加载/调优超参数。
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.svm import FastSurvivalSVM

from .config import Config
from .utils import cindex_score


class ModelRegistry:
    """模型注册表（单例）"""

    _models: dict[str, type] = {}

    @classmethod
    def register(cls, name: str, model_class: type) -> None:
        """
        注册新模型

        参数:
            name: 模型名称
            model_class: 模型类
        """
        cls._models[name] = model_class

    @classmethod
    def get(cls, name: str) -> type:
        """
        获取模型类

        参数:
            name: 模型名称

        返回:
            模型类
        """
        if name not in cls._models:
            raise ValueError(f"未知模型: {name}")
        return cls._models[name]

    @classmethod
    def list_models(cls) -> list[str]:
        """
        列出所有已注册模型

        返回:
            模型名称列表
        """
        return list(cls._models.keys())


class ModelFactory:
    """模型工厂"""

    def __init__(self, config: Config, rng: np.random.Generator):
        """
        初始化模型工厂

        参数:
            config: 配置对象
            rng: 随机数生成器
        """
        self.config = config
        self.rng = rng
        self.param_cache: dict[str, dict] = {}

    def create_pipelines(
        self,
        X: pd.DataFrame,
        y: np.ndarray
    ) -> dict[str, Pipeline]:
        """
        创建所有模型的pipeline

        参数:
            X: 特征矩阵
            y: 生存数据（sksurv 格式）

        返回:
            模型名称到 Pipeline 的字典
        """
        # RSF 超参数
        rsf_params = self.load_or_tune_params(
            "RandomSurvivalForest",
            RandomSurvivalForest(random_state=self.config.random_state),
            {"n_estimators": [100, 200], "max_depth": [5, 10], "max_features": ["sqrt", None]},
            X, y
        )
        print(">>> Selected RSF params:", rsf_params)

        # GBM 超参数
        gbm_params = self.load_or_tune_params(
            "GradientBoostingSurvivalAnalysis",
            GradientBoostingSurvivalAnalysis(random_state=self.config.random_state),
            {"learning_rate": [0.05, 0.1], "n_estimators": [100, 200], "max_depth": [3, 4]},
            X, y
        )
        print(">>> Selected GBM params:", gbm_params)

        return {
            "Cox_Net_Lasso": Pipeline([
                ("scaler", StandardScaler()),
                ("model", CoxnetSurvivalAnalysis(
                    alpha_min_ratio=0.01,
                    n_alphas=50,
                    max_iter=10000
                ))
            ]),
            "Cox_PH_Ridge": Pipeline([
                ("scaler", StandardScaler()),
                ("model", CoxPHSurvivalAnalysis(alpha=1.0))
            ]),
            "Random_Survival_Forest": Pipeline([
                ("model", RandomSurvivalForest(
                    **rsf_params,
                    random_state=self.config.random_state
                ))
            ]),
            "Gradient_Boosting": Pipeline([
                ("model", GradientBoostingSurvivalAnalysis(
                    **gbm_params,
                    random_state=self.config.random_state
                ))
            ]),
            "Survival_SVM": Pipeline([
                ("scaler", StandardScaler()),
                ("model", FastSurvivalSVM(
                    alpha=1.0,
                    max_iter=1000,
                    random_state=self.config.random_state
                ))
            ]),
        }

    def load_or_tune_params(
        self,
        model_name: str,
        estimator: Any,
        param_grid: dict,
        X: pd.DataFrame,
        y: np.ndarray
    ) -> dict:
        """
        加载或调优超参数

        参数:
            model_name: 模型名称
            estimator: 估计器实例
            param_grid: 参数网格
            X: 特征矩阵
            y: 生存数据

        返回:
            最佳参数字典
        """
        cache_path = self.config.best_param_path

        # 尝试从缓存加载
        if cache_path is not None and cache_path.exists():
            with open(cache_path) as f:
                all_params = json.load(f)
            if model_name in all_params:
                print(f"加载缓存参数: {model_name}")
                return all_params[model_name]

        # 执行网格搜索
        print(f"正在调参: {model_name}")
        scorer = make_scorer(cindex_score, greater_is_better=True)
        cv = KFold(
            n_splits=3,
            shuffle=True,
            random_state=self.config.random_state
        )
        grid = GridSearchCV(
            estimator, param_grid, cv=cv, scoring=scorer, n_jobs=-1, verbose=1
        )
        grid.fit(X, y)
        best_params = grid.best_params_

        # 保存到缓存
        all_params = {}
        if cache_path is not None:
            if cache_path.exists():
                with open(cache_path) as f:
                    all_params = json.load(f)
            all_params[model_name] = best_params
            with open(cache_path, "w") as f:
                json.dump(all_params, f, indent=4)

        return best_params

    def get_model_param_grid(self, model_name: str) -> dict:
        """
        获取模型的参数网格

        参数:
            model_name: 模型名称

        返回:
            参数网格字典
        """
        param_grids = {
            "RandomSurvivalForest": {
                "n_estimators": [100, 200],
                "max_depth": [5, 10],
                "max_features": ["sqrt", None]
            },
            "GradientBoostingSurvivalAnalysis": {
                "learning_rate": [0.05, 0.1],
                "n_estimators": [100, 200],
                "max_depth": [3, 4]
            },
        }

        if model_name not in param_grids:
            raise ValueError(f"未知模型: {model_name}")

        return param_grids[model_name]


# 导入 pandas（用于类型注解）
import pandas as pd
