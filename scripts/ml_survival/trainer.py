"""
训练器模块

负责模型训练、评估、结果保存。
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sksurv.compare import compare_survival
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc

from .config import Config
from .data_manager import DataManager
from .feature_selector import FeatureSelector
from .models import ModelFactory


class Evaluator:
    """评估器"""

    def __init__(self, config: Config):
        """
        初始化评估器

        参数:
            config: 配置对象
        """
        self.config = config

    def calculate_evaluate(
        self,
        pipeline: Pipeline,
        X: pd.DataFrame,
        y: pd.DataFrame,
        y_train: pd.DataFrame
    ) -> dict[str, Any]:
        """
        计算评估指标（C-index, timeAUC, KM p-value）

        参数:
            pipeline: 训练好的模型 pipeline
            X: 特征矩阵
            y: 生存数据（包含 events, OS 列）
            y_train: 训练集生存数据（用于IPCW计算）

        返回:
            包含评估指标的字典
        """
        # 确保 y 和 y_train 是 DataFrame 格式（处理可能的结构化数组输入）
        if not isinstance(y, pd.DataFrame):
            y = pd.DataFrame({
                'events': y['events'],
                'OS': y['OS']
            })
        if not isinstance(y_train, pd.DataFrame):
            y_train = pd.DataFrame({
                'events': y_train['events'],
                'OS': y_train['OS']
            })
        print(y)
        
        # 计算风险评分
        risk = pipeline.predict(X)
        logging.info(f"风险评分范围=[{risk.min():.4f}, {risk.max():.4f}], 唯一数={np.unique(risk).shape[0]}")

        # 时间依赖性ROC
        max_ref = y_train.loc[y_train["events"],'OS'].max()
        max_eval = y.loc[y["events"]]["OS"].max()
        logging.info(f"{max_ref = } months, {max_eval = } months")

        if max_eval > max_ref:
            mask = (y["OS"] <= max_ref).values
            # print(mask)
            # print(mask.shape)
            # print(f"{X.shape = }")
            # print(f"{X.index = }")
            X_roc = X.iloc[mask]
            y_roc = y.iloc[mask]
            risk_roc = risk[mask]
            logging.info(f'    y_roc 实际最大随访时间: {y_roc["OS"].max()}')
        else:
            X_roc = X.copy()
            y_roc = y.copy()
            risk_roc = risk.copy()
            logging.info('    未对y进行截断')

        endpoint = int(min(max_ref, y_roc["OS"].max(), 36) // 6)
        eval_times = np.array(range(1, endpoint + 1)) * 6
        logging.info(f'    评估时间节点为：{eval_times = }')

        # 转换为 sksurv 格式
        y_train_struct = np.array(
            [(bool(e), t) for e, t in zip(y_train["events"], y_train["OS"])],
            dtype=[('events', '?'), ('OS', '<f8')]
        )
        logging.info('y_train_struct转换成功')

        y_roc_struct = np.array(
            [(bool(e), t) for e, t in zip(y_roc["events"], y_roc["OS"])],
            dtype=[('events', '?'), ('OS', '<f8')]
        )
        logging.info('y_roc_struct转换成功')

        aucs, _ = cumulative_dynamic_auc(y_train_struct, y_roc_struct, risk_roc, eval_times)
        auc_dict = {f"timeAUC_{t}m": float(aucs[i]) for i, t in enumerate(eval_times)}

        # logrank检验
        print(len(risk_roc))
        print(len(y_roc_struct))
        KM_pvalue = self.calculate_km_pvalue(risk_roc, y_roc_struct)

        # 风险评分 - Dataframe
        risk_df = pd.DataFrame({
            "patient": X.index,
            "risk_score": risk,
            "OS": y["OS"],
            "events": y["events"].astype(int)
        })

        # 聚合结果
        result = {
            'risk': risk,
            "c_index": float(cidx),
            **auc_dict,
            'KM_pvalue': float(KM_pvalue) if KM_pvalue is not None else None,
            'risk_df': risk_df,
        }
        return result

    @staticmethod
    def cindex_score(y_true: pd.DataFrame, y_pred: np.ndarray) -> float:
        """
        计算C-index

        参数:
            y_true: 真实生存数据（包含 events, OS 列）
            y_pred: 预测的风险评分

        返回:
            C-index 值
        """
        # 处理结构化数组输入
        if not isinstance(y_true, pd.DataFrame):
            events = y_true['events']
            os = y_true['OS']
        else:
            events = y_true["events"].astype(bool)
            os = y_true["OS"]

        return concordance_index_censored(
            events,
            os,
            y_pred
        )[0]

    @staticmethod
    def calculate_km_pvalue(risk_scores: np.ndarray, y_true: pd.DataFrame) -> float | None:
        """
        计算 log-rank 检验 P 值（不绘图）

        参数:
            risk_scores: 风险评分数组
            y_true: 包含 OS, events 的生存数据

        返回:
            pval: log-rank 检验的 P 值，如果无法计算则返回 None
        """
        df = pd.DataFrame({
            "score": risk_scores,
            "OS": y_true["OS"],
            "events": y_true["events"].astype(bool)
        })

        if df["score"].nunique() <= 1:
            logging.info(f"  警告：所有风险评分相同 ({df['score'].iloc[0]:.6f})，无法计算 log-rank P 值")
            return None

        try:
            df["group"] = pd.qcut(df["score"], q=2, labels=["Low", "High"], duplicates="drop")
        except (ValueError, IndexError) as e:
            logging.info(f"  警告：qcut 分组失败 ({e})，尝试使用中位数分组")
            median_score = df["score"].median()
            df["group"] = df["score"].apply(
                lambda x: "Low" if x <= median_score else "High"
            )

        if df["group"].nunique() < 2:
            logging.info(f"  警告：风险评分分布过于集中，无法区分高低风险组")
            return None
        
        _, pval = compare_survival(y_true, df["group"])
        return pval


class ModelTrainer:
    """模型训练器"""

    def __init__(
        self,
        config: Config,
        data_manager: DataManager,
        feature_selector: FeatureSelector,
        model_factory: ModelFactory
    ):
        """
        初始化模型训练器

        参数:
            config: 配置对象
            data_manager: 数据管理器
            feature_selector: 特征选择器
            model_factory: 模型工厂
        """
        self.config = config
        self.data_manager = data_manager
        self.feature_selector = feature_selector
        self.model_factory = model_factory
        self.evaluator = Evaluator(config)
        self.trained_models: dict[str, Pipeline] = {}
        self.results: dict[str, dict] = {}

    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame
    ) -> tuple[dict[str, dict], dict[str, Pipeline]]:
        """
        训练所有模型

        参数:
            X_train: 训练集特征矩阵
            y_train: 训练集生存数据
            X_test: 测试集特征矩阵
            y_test: 测试集生存数据

        返回:
            (results, trained_models): 训练结果和已训练模型的字典
        """
        logging.info(">>>>" * 5 + '进行训练与验证')

        # 数据转换
        y_train_struct = np.array(
            [(bool(e), t) for e, t in zip(y_train["events"], y_train["OS"])],
            dtype=[('events', '?'), ('OS', '<f8')]
        )
        y_test_struct = np.array(
            [(bool(e), t) for e, t in zip(y_test["events"], y_test["OS"])],
            dtype=[('events', '?'), ('OS', '<f8')]
        )

        # 创建模型 pipelines
        logging.info('加载模型并调参')
        pipelines = self.model_factory.create_pipelines(X_train, y_train_struct)

        # 训练前诊断
        if self.config.diagnostic:
            print(f"  [诊断] 训练集:")
            print(f"    - X_train 形状: {X_train.shape}")
            print(f"    - y_train: 事件数={y_train_struct['events'].sum()}/{len(y_train_struct)}")
            print(f"  [诊断] 验证集:")
            print(f"    - X_test 形状: {X_test.shape}")
            print(f"    - y_test: 事件数={y_test_struct['events'].sum()}/{len(y_test_struct)}")

        results = {}
        trained_pipelines = {}

        for name, pipe in pipelines.items():
            try:
                logging.info(f'>>> 训练模型: {name}')
                pipe.fit(X_train, y_train_struct)
                trained_pipelines[name] = pipe

                # 训练后诊断
                if self.config.diagnostic:
                    model = pipe.named_steps.get('model')
                    logging.info('训练后诊断')
                    if hasattr(model, 'coef_'):
                        coef = model.coef_
                        if coef.ndim == 1:
                            logging.info(f"  [诊断] 模型系数:")
                            logging.info(f"    - 系数形状: {coef.shape}")
                            logging.info(f"    - 非零系数: {(coef != 0).sum()}/{len(coef)}")
                            logging.info(f"    - 系数范围: [{coef.min():.4f}, {coef.max():.4f}]")
                        elif coef.ndim == 2:
                            if hasattr(model, 'alphas_'):
                                final_coef = coef[:, -1]
                                logging.info(f"  [诊断] CoxNet 路径 - 最终 alpha ({model.alphas_[-1]:.4f}):")
                                logging.info(f"    - 非零系数: {np.count_nonzero(final_coef)}/{len(final_coef)}")
                                logging.info(f"    - 系数范围: [{final_coef.min():.4f}, {final_coef.max():.4f}]")

                # 性能评估
                logging.info(f'>>> 性能评估')
                logging.info("  训练集")
                result_train = self.evaluator.calculate_evaluate(pipe, X_train, y_train, y_train)
                logging.info("  验证集")
                result_test = self.evaluator.calculate_evaluate(pipe, X_test, y_test, y_train)
                logging.info(f'>>> 性能评估完毕: {result_train.keys()}')

            except Exception as e:
                logging.exception(f"  × 模型 {name} 失败")

            results[name] = {
                'train': result_train,
                'vali': result_test
            }

        self.results = results
        self.trained_models = trained_pipelines

        return results, trained_pipelines

    def save_results(self) -> None:
        """保存训练结果和模型"""
        import pickle

        # 保存 results 字典
        results_path = self.config.output_dir / 'train_results.pkl'
        with open(results_path, 'wb') as f:
            pickle.dump(self.results, f)
        logging.info(f"  √ 训练结果已保存: {results_path}")

        # 保存 trained_pipelines
        pipelines_path = self.config.output_dir / 'trained_pipelines.pkl'
        with open(pipelines_path, 'wb') as f:
            pickle.dump(self.trained_models, f)
        logging.info(f"  √ 训练模型已保存: {pipelines_path}")
        logging.info(f"    包含 {len(self.trained_models)} 个模型: {list(self.trained_models.keys())}")

    def run_cv_mode(
        self,
        cv_splits: list[dict[str, Any]],
        X: pd.DataFrame,
        y: pd.DataFrame
    ) -> dict[str, dict]:
        """
        运行交叉验证模式

        参数:
            cv_splits: CV划分列表
            X: 对齐后的特征矩阵
            y: 生存数据

        返回:
            CV结果字典
        """
        cv_folder = self.config.output_dir / 'CV'
        cv_folder.mkdir(exist_ok=True, parents=True)
        results = {}

        for fold_idx, fold in enumerate(cv_splits):
            logging.info(f"Running fold {fold_idx}")

            # 数据集划分
            train_patients = fold['train_patients']
            vali_patients = fold['val_patients']
            X_train_fold = X.loc[train_patients]
            X_vali_fold = X.loc[vali_patients]
            y_train_fold = y.set_index('Name').loc[train_patients].reset_index()
            y_vali_fold = y.set_index('Name').loc[vali_patients].reset_index()

            # 特征选择
            X_train_selected = self.feature_selector.fit_transform(
                X_train_fold, y_train_fold
            )
            print(f"{X_train_selected.shape}")
            X_vali_selected = X_vali_fold[X_train_selected.columns]
            print(f"{X_vali_selected.shape}")

            # 训练
            results_fold, _ = self.train_and_validate_single_fold(
                X_train_selected, y_train_fold,
                X_vali_selected, y_vali_fold
            )
            results[f'fold{fold_idx}'] = results_fold

        return results

    def train_and_validate_single_fold(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_vali: pd.DataFrame,
        y_vali: pd.DataFrame
    ) -> tuple[dict[str, dict], dict[str, Pipeline]]:
        """
        训练和验证单个fold

        参数:
            X_train: 训练集特征
            y_train: 训练集生存数据
            X_vali: 验证集特征
            y_vali: 验证集生存数据

        返回:
            (results, trained_pipelines): 结果和模型字典
        """
        # 数据转换
        y_train_struct = np.array(
            [(bool(e), t) for e, t in zip(y_train["events"], y_train["OS"])],
            dtype=[('events', '?'), ('OS', '<f8')]
        )
        y_vali_struct = np.array(
            [(bool(e), t) for e, t in zip(y_vali["events"], y_vali["OS"])],
            dtype=[('events', '?'), ('OS', '<f8')]
        )

        # 创建模型 pipelines
        pipelines = self.model_factory.create_pipelines(X_train, y_train_struct)

        results = {}
        trained_pipelines = {}

        for name, pipe in pipelines.items():
            try:
                logging.info(f'>>> 训练模型: {name}')
                pipe.fit(X_train, y_train_struct)
                trained_pipelines[name] = pipe

                # 性能评估
                result_train = self.evaluator.calculate_evaluate(pipe, X_train, y_train, y_train)
                result_vali = self.evaluator.calculate_evaluate(pipe, X_vali, y_vali, y_train)

            except Exception as e:
                logging.exception(f"  × 模型 {name} 失败")

            results[name] = {
                'train': result_train,
                'vali': result_vali
            }

        return results, trained_pipelines
