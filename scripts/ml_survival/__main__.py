"""
主流程控制器和命令行入口

协调所有组件，管理整体流程。
"""
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .batch_correction import BatchCorrector
from .config import Config, ConfigManager
from .data_manager import DataManager
from .feature_selector import FeatureSelector
from .models import ModelFactory
from .trainer import ModelTrainer
from .utils import (
    create_cv_splits,
    load_cv_splits,
    lock_random,
    unpack_single_result,
)


class SurvivalAnalysisPipeline:
    """生存分析主流程控制器"""

    def __init__(self, config: Config):
        """
        初始化生存分析流程

        参数:
            config: 配置对象
        """
        self.config = config
        self.rng = lock_random(seed=config.random_state)

        # 初始化各组件
        self.data_manager = DataManager(config)
        self.feature_selector = FeatureSelector(config)
        self.model_factory = ModelFactory(config, self.rng)
        self.trainer = ModelTrainer(
            config,
            self.data_manager,
            self.feature_selector,
            self.model_factory
        )
        self.batch_corrector = BatchCorrector(config) if config.remove_batch_effects else None

        # 数据
        self.X_train_selected: pd.DataFrame | None = None
        self.X_test_selected: pd.DataFrame | None = None
        logging.info('初始化完毕')

    def run(self) -> None:
        """运行完整流程"""
        # 1. 数据准备
        self._prepare_data()

        # 2. 特征选择
        self._apply_feature_selection()

        # 3. CV模式（如果启用）
        if self.config.run_cv:
            self._run_cv_mode()

        # 4. 训练-测试模式
        results, trained_models = self._run_train_test_mode()
        self.trainer.results = results
        self.trainer.trained_models = trained_models

        # 5. 保存结果
        self._save_results(results, trained_models)

        # 6. 可视化
        if self.config.plot:
            self._visualize_results()

    def _prepare_data(self) -> None:
        """数据准备阶段"""
        logging.info("=" * 60)
        logging.info("数据集预处理")
        logging.info("=" * 60)

        # 加载数据
        self.data_manager.load_data()

        # 预处理数据
        self.data_manager.prepare_data()

    def _apply_feature_selection(self) -> None:
        """特征选择阶段"""
        # 获取数据
        X_train, y_train = self.data_manager.get_train_data()
        X_test, y_test = self.data_manager.get_test_data()

        # 批次校正（如果启用）
        if self.batch_corrector is not None:
            logging.info("=" * 60)
            logging.info("批次效应去除")
            logging.info("=" * 60)

            # 从患者ID提取批次标签
            train_batches = self.batch_corrector.extract_batch_from_patient(X_train.index)
            test_batches = self.batch_corrector.extract_batch_from_patient(X_test.index)

            # 执行批次校正
            X_train, X_test = self.batch_corrector.remove_batch_effects(
                X_train, X_test,
                train_batches, test_batches
            )
            logging.info("✓ 批次效应去除完成\n")

        # 特征选择
        logging.info('现在开始进行特征选择')
        logging.info(f'{X_train.shape}')
        
        X_train_selected, X_test_selected = self.feature_selector.fit_transform(
            X_train, y_train, X_test
        )

        logging.info(f"{X_train_selected.shape = }")
        logging.info(f"{X_test_selected.shape = }")

        self.X_train_selected = X_train_selected
        self.X_test_selected = X_test_selected

    def _run_cv_mode(self) -> None:
        """运行交叉验证模式"""
        logging.info("=" * 60)
        logging.info("    增量运行: CV模式")
        logging.info("=" * 60)

        # 读取CV数据
        if os.path.exists(self.config.cv_json_path):
            logging.info(f"CV Splits Exists, loading from {self.config.cv_json_path}")
            cv_splits = load_cv_splits(self.config.cv_json_path)
        else:
            logging.info(f"CV Splits NOT Exists, creating and save to {self.config.cv_json_path}")
            y_train = self.data_manager.y_train
            cv_splits = create_cv_splits(
                y_train,
                random_state=self.config.random_state,
                cv_json_path=self.config.cv_json_path
            )

        # CV结果的保存路径
        cv_folder = self.config.output_dir / 'CV'
        cv_folder.mkdir(exist_ok=True, parents=True)

        # 获取对齐后的数据
        X_train, y_train = self.data_manager.get_train_data()

        results = {}
        for fold_idx, fold in enumerate(cv_splits):
            logging.info(f"Running fold {fold_idx}")

            # 数据集划分
            train_patients = fold['train_patients']
            vali_patients = fold['val_patients']
            X_train_fold = X_train.loc[train_patients]
            X_vali_fold = X_train.loc[vali_patients]
            y_train_fold = y_train.set_index('patient').loc[train_patients].reset_index()
            y_vali_fold = y_train.set_index('patient').loc[vali_patients].reset_index()

            # 特征选择
            X_train_selected = self.feature_selector.fit_transform(
                X_train_fold, y_train_fold
            )
            print(f"{X_train_selected.shape}")
            X_vali_selected = X_vali_fold[X_train_selected.columns]
            print(f"{X_vali_selected.shape}")

            # 训练
            results_fold, _ = self.trainer.train_and_validate_single_fold(
                X_train_selected, y_train_fold,
                X_vali_selected, y_vali_fold
            )
            results[f'fold{fold_idx}'] = results_fold

        # 汇总结果
        self._save_cv_results(results)

        logging.info("=" * 60)
        logging.info("    CV模式运行完毕！")
        logging.info("=" * 60)

    def _run_train_test_mode(self) -> tuple[dict, dict]:
        """运行训练-测试模式"""
        logging.info("=" * 60)
        logging.info("    模式: Train-Test模式")
        logging.info("=" * 60)

        # 获取数据
        y_train = self.data_manager.y_train
        y_test = self.data_manager.y_test

        # 训练所有模型
        results, trained_models = self.trainer.train_all_models(
            self.X_train_selected, y_train,
            self.X_test_selected, y_test
        )

        logging.info("=" * 60)
        logging.info("    Train-Test模式运行完毕！")
        logging.info("=" * 60)

        return results, trained_models

    def _save_results(self, results: dict, trained_models: dict) -> None:
        """保存训练结果和模型"""
        logging.info("=" * 60)
        logging.info("    保存训练结果和模型")
        logging.info("=" * 60)

        # 保存 results 字典
        results_path = self.config.output_dir / 'train_results.pkl'
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        logging.info(f"  √ 训练结果已保存: {results_path}")

        # 保存 trained_pipelines
        pipelines_path = self.config.output_dir / 'trained_pipelines.pkl'
        with open(pipelines_path, 'wb') as f:
            pickle.dump(trained_models, f)
        logging.info(f"  √ 训练模型已保存: {pipelines_path}")
        logging.info(f"    包含 {len(trained_models)} 个模型: {list(trained_models.keys())}")

        # 结果解析
        summary_df = unpack_single_result(results)
        csv_path = self.config.output_dir / 'summary.csv'
        summary_df.to_csv(csv_path, index=True)

        # 保存筛选后的特征矩阵
        if self.X_train_selected is not None:
            train_features_path = self.config.output_dir / 'X_train_selected.csv'
            self.X_train_selected.to_csv(train_features_path, index=True)
            logging.info(f"  √ 筛选后训练集特征矩阵已保存: {train_features_path}")
            logging.info(f"    形状: {self.X_train_selected.shape}, 特征数: {self.X_train_selected.shape[1]}")

        if self.X_test_selected is not None:
            test_features_path = self.config.output_dir / 'X_test_selected.csv'
            self.X_test_selected.to_csv(test_features_path, index=True)
            logging.info(f"  √ 筛选后测试集特征矩阵已保存: {test_features_path}")
            logging.info(f"    形状: {self.X_test_selected.shape}, 特征数: {self.X_test_selected.shape[1]}")

    def _save_cv_results(self, results: dict) -> None:
        """保存CV结果"""
        from .visualizer import VisualizationManager

        cv_folder = self.config.output_dir / 'CV'
        cv_folder.mkdir(exist_ok=True, parents=True)

        # 汇总结果
        result_summarised = {
            fold_idx: unpack_single_result(fold_result)
            for fold_idx, fold_result in results.items()
        }

        combined_data = []
        for fold_idx, fold_df in result_summarised.items():
            df_copy = fold_df.copy()
            df_copy['Fold'] = fold_idx
            combined_data.append(df_copy.reset_index())

        # 合并所有数据
        combined_df = pd.concat(combined_data, ignore_index=True)
        combined_df.set_index(['Fold', 'Model', 'Dataset'], inplace=True)
        combined_df = combined_df.sort_index()

        csv_path = cv_folder / 'CV_combined_df.csv'
        combined_df.to_csv(csv_path, index=True)
        logging.info('Fold Result Unpacked')
        print(combined_df)

        # 生成CV统计和可视化
        metric_names = combined_df.columns.to_list()
        logging.info(f"{metric_names = }")
        model_names = list(set(combined_df.index.get_level_values('Model').to_list()))
        model_names.sort()
        logging.info(f'{model_names = }')

        CV_summary = []
        (self.config.figure_dir / 'CV').mkdir(exist_ok=True, parents=True)
        cv_figure_format = self.config.figure_format

        visualizer = VisualizationManager(self.config, results, {})

        for metric in metric_names:
            for ds in ['train', 'vali']:
                df_sub = combined_df.xs(ds, level='Dataset')[metric]
                df_sub = df_sub.unstack(level='Fold')
                print(df_sub)

                # 计算均值
                metric_mean = np.mean(df_sub, axis=1)
                print(metric_mean)

                # 绘图
                save_path = self.config.figure_dir / 'CV' / f"CV_{metric}_{ds}.{cv_figure_format}"
                if metric == 'c_index':
                    title_metric = "C-index"
                else:
                    title_metric = metric
                visualizer.plot_fold_boxplot(
                    df_sub, save_path,
                    title=f"{title_metric} in {ds}ing cohort",
                    figure_format=cv_figure_format
                )

                # 保存
                csv_path = cv_folder / f'CV_{metric}_{ds}.csv'
                df_sub.to_csv(csv_path, index=True)

                # 添加到最终结果文件
                CV_summary.append(pd.DataFrame(metric_mean, columns=[f'{metric}_{ds}']))

        CV_summary_df = pd.concat(CV_summary, axis=1)
        csv_path = cv_folder / 'CV_summary.csv'
        CV_summary_df.to_csv(csv_path, index=True)

    def _visualize_results(self) -> None:
        """可视化阶段"""
        from .visualizer import VisualizationManager

        logging.info("=" * 60)
        logging.info("    可视化")
        logging.info("=" * 60)

        y_train = self.data_manager.y_train

        visualizer = VisualizationManager(
            self.config,
            self.trainer.results,
            self.trainer.trained_models
        )

        visualizer.plot_all(self.X_train_selected, y_train, self.rng)


def main():
    """命令行入口"""
    # 解析参数
    args = ConfigManager.parse_args()

    # 创建配置
    config = ConfigManager.from_args(args)

    # 验证配置
    ConfigManager.validate_config(config)

    # 打印配置摘要
    ConfigManager.print_summary(config)

    # 创建输出目录
    config.output_dir.mkdir(exist_ok=True, parents=True)
    config.figure_dir.mkdir(exist_ok=True, parents=True)

    # 运行流程
    pipeline = SurvivalAnalysisPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
