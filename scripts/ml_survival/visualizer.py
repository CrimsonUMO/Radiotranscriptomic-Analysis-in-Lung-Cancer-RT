"""
可视化模块

负责生成所有可视化图表（KM曲线、ROC、SHAP等）。
"""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times
from sklearn.metrics import roc_curve
from sksurv.compare import compare_survival
from sksurv.metrics import cumulative_dynamic_auc

from .config import Config


class VisualizationManager:
    """可视化管理器"""

    def __init__(
        self,
        config: Config,
        results: dict[str, dict],
        trained_models: dict[str, Any]
    ):
        """
        初始化可视化管理器

        参数:
            config: 配置对象
            results: 训练结果字典
            trained_models: 已训练模型字典
        """
        self.config = config
        self.results = results
        self.trained_models = trained_models

    def plot_all(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        rng: np.random.Generator
    ) -> None:
        """
        生成所有可视化

        参数:
            X_train: 训练集特征（特征选择后的）
            y_train: 训练集生存数据
            rng: 随机数生成器（用于 SHAP 可重复性）
        """
        if not self.config.plot:
            return

        logging.info("=" * 60)
        logging.info("    可视化")
        logging.info("=" * 60)

        figure_format = self.config.figure_format

        for model_name, model_result in self.results.items():
            logging.info("=" * 60)
            logging.info(f" Visualizing {model_name}")
            logging.info("=" * 60)

            # 准备文件夹
            model_figure_dir = self.config.figure_dir / model_name
            model_figure_dir.mkdir(exist_ok=True, parents=True)

            # 1. KM曲线
            logging.info(f"plotting KM curve for {model_name}")
            train_risk_df = model_result['train']['risk_df']
            save_path = model_figure_dir / f"KM_train.{figure_format}"
            logging.info(f"Saving train KM to {save_path}")
            if model_result['train']['KM_pvalue'] is not None:
                self.plot_km_curves(train_risk_df, save_path, figure_format)

            vali_risk_df = model_result['vali']['risk_df']
            save_path = model_figure_dir / f"KM_test.{figure_format}"
            logging.info(f"Saving test KM to {save_path}")
            if model_result['vali']['KM_pvalue'] is not None:
                self.plot_km_curves(vali_risk_df, save_path, figure_format)

            # 2. timeROC
            logging.info(f"plotting time ROC curve for {model_name}")
            logging.info("    Training set")
            save_path = model_figure_dir / f"ROC_train.{figure_format}"
            self.plot_time_dependent_roc(
                y_train, train_risk_df,
                self.config.evaluation_times, save_path, figure_format
            )
            logging.info("    Test set")
            save_path = model_figure_dir / f"ROC_test.{figure_format}"
            self.plot_time_dependent_roc(
                y_train, vali_risk_df,
                self.config.evaluation_times, save_path, figure_format
            )
            logging.info(f"ROC curves generated")

            # 3. 风险评分分布图
            logging.info(f"plotting risk distribution for {model_name}")
            save_path = model_figure_dir / f"risk_distribution_train.{figure_format}"
            self.plot_risk_distribution(train_risk_df, save_path, figure_format)
            save_path = model_figure_dir / f"risk_distribution_test.{figure_format}"
            self.plot_risk_distribution(vali_risk_df, save_path, figure_format)

            # 4. SHAP解释图
            if model_name in ['Survival_SVM', 'Gradient_Boosting', 'Random_Survival_Forest']:
                logging.info(f"跳过 SHAP: {model_name}")
                continue

            save_path = model_figure_dir / f"SHAP.{figure_format}"
            model_trained = self.trained_models[model_name]
            self.plot_shap_values(model_trained, X_train, y_train, save_path, rng, figure_format)

        # 5. C-index 对比图（所有模型绘制在一起）
        logging.info("plotting C-index comparison across models")
        save_path = self.config.figure_dir / f"cindex_comparison.{figure_format}"
        self.plot_cindex_comparison(save_path, figure_format)

    def plot_km_curves(
        self,
        risk_df: pd.DataFrame,
        save_path: Path,
        figure_format: str = 'png'
    ) -> None:
        """
        绘制 Kaplan-Meier 生存曲线并保存图片

        参数:
            risk_df: 包含 risk_score, OS, events 的 DataFrame
            save_path: 图片保存路径
            figure_format: 图片格式
        """
        # 检查风险评分是否全相同
        if risk_df["risk_score"].nunique() <= 1:
            logging.info(f"  警告：所有风险评分相同，无法绘制 KM 曲线")
            return

        # 准备生存数据结构
        y_true = risk_df[["OS", "events"]].copy()
        y_true["events"] = y_true["events"].astype(bool)

        # 分组
        try:
            df = risk_df.copy()
            df["group"] = pd.qcut(df["risk_score"], q=2, labels=["Low", "High"], duplicates="drop")
        except (ValueError, IndexError) as e:
            logging.info(f"  警告：qcut 分组失败 ({e})，尝试使用中位数分组")
            median_score = risk_df["risk_score"].median()
            df = risk_df.copy()
            df["group"] = df["risk_score"].apply(
                lambda x: "Low" if x <= median_score else "High"
            )

        # 检查是否成功创建了两个组
        if df["group"].nunique() < 2:
            logging.info(f"  警告：风险评分分布过于集中，无法区分高低风险组")
            return

        # 计算 log-rank P 值
        _, pval = compare_survival(
            np.array([(e, t) for e, t in zip(y_true["events"], y_true["OS"])],
                     dtype=[('events', '?'), ('OS', '<f8')]),
            df["group"].values
        )

        # 绘制 KM 曲线
        plt.figure(figsize=(3, 3))
        kmf = KaplanMeierFitter()

        for group, color in zip(["Low", "High"], ["green", "red"]):
            mask = df["group"] == group
            kmf.fit(df.loc[mask, "OS"], event_observed=df.loc[mask, "events"],
                    label=f"{group} Score (n={mask.sum()})")
            median_time = kmf.median_survival_time_
            kmf.plot(color=color, ci_show=True)

            # 输出中位OS和95%CI
            ci_df = median_survival_times(kmf.confidence_interval_)
            if not ci_df.empty:
                lower = ci_df.iloc[0, 0]
                upper = ci_df.iloc[0, 1]
            else:
                lower = upper = None

            logging.info(f"    {group}组: 中位生存期 = {median_time:.2f}, 95% CI = [{lower:.2f}, {upper:.2f}]")

        plt.text(0.7, 0.05, f"log-rank\np = {pval:.4f}", transform=plt.gca().transAxes, fontsize=8)
        plt.xlabel("OS (months)", fontsize=8)
        plt.ylabel("Survival Probability", fontsize=8)
        plt.legend(loc="upper right", fontsize=8)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)

        dpi_param = 300 if figure_format == 'png' else None
        plt.savefig(save_path, format=figure_format, dpi=dpi_param, bbox_inches="tight")
        plt.close()

    def plot_time_dependent_roc(
        self,
        y_train: pd.DataFrame,
        risk_df: pd.DataFrame,
        times: list[int],
        save_path: Path,
        figure_format: str = 'png'
    ) -> None:
        """
        绘制时间依赖ROC曲线

        参数:
            y_train: 训练集生存数据（用于IPCW计算）
            risk_df: 验证集 risk_df（包含 risk_score, OS, events）
            times: 评估时间点列表
            save_path: 保存路径
            figure_format: 图片格式
        """
        plt.figure(figsize=(3, 3))
        colors = sns.color_palette("Dark2", n_colors=len(times))

        # 从 risk_df 提取 survival_df
        risk_scores = risk_df["risk_score"].values
        y_test = risk_df[["OS", "events"]].copy()
        y_test["events"] = y_test["events"].astype(bool)

        # 转换 y_train 为 sksurv 格式
        y_train_struct = np.array(
            [(bool(e), t) for e, t in zip(y_train["events"], y_train["OS"])],
            dtype=[('events', '?'), ('OS', '<f8')]
        )

        # 转换 y_test 为 sksurv 格式
        y_test_struct = np.array(
            [(bool(e), t) for e, t in zip(y_test["events"], y_test["OS"])],
            dtype=[('events', '?'), ('OS', '<f8')]
        )

        # 计算最大事件时间
        max_ref = y_train_struct[y_train_struct["events"]]["OS"].max()
        max_eval = y_test_struct[y_test_struct["events"]]["OS"].max()

        if max_eval > max_ref:
            mask = y_test_struct["OS"] <= max_ref
            y_roc = y_test_struct[mask]
            risk_roc = risk_scores[mask]
        else:
            y_roc = y_test_struct.copy()
            risk_roc = risk_scores.copy()

        endpoint = int(min(max_ref, max_eval, 36) // 12)
        eval_times = np.array(range(1, endpoint + 1)) * 12

        try:
            auc_vals, _ = cumulative_dynamic_auc(y_train_struct, y_roc, risk_roc, eval_times)
        except Exception as e:
            logging.info(f"AUC 计算失败: {e}")
            auc_vals = [0.5] * len(times)

        for i, t in enumerate(times):
            mask = (y_test["OS"] >= t) | y_test["events"]
            if mask.sum() < 2:
                continue
            y_bin = ((y_test["OS"] <= t) & y_test["events"]).astype(int)[mask]
            scores = risk_scores[mask]
            fpr, tpr, _ = roc_curve(y_bin, scores)
            plt.plot(fpr, tpr, color=colors[i], lw=2, label=f'{t}m (AUC={auc_vals[i]:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
        plt.xlabel('False Positive Rate', fontsize=8)
        plt.ylabel('True Positive Rate', fontsize=8)
        plt.legend(loc="lower right", fontsize=8)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        dpi_param = 300 if figure_format == 'png' else None
        plt.savefig(save_path, format=figure_format, dpi=dpi_param, bbox_inches='tight')
        plt.close()

    def plot_shap_values(
        self,
        trained_pipeline: Any,
        X: pd.DataFrame,
        y: pd.DataFrame,
        save_path: Path,
        rng: np.random.Generator,
        figure_format: str = 'png'
    ) -> None:
        """
        为单个已训练模型生成 SHAP 解释图并保存SHAP值表格

        参数:
            trained_pipeline: 已训练的单个模型 pipeline
            X: 特征数据
            y: 生存数据
            save_path: 图片保存路径
            rng: 随机数生成器
            figure_format: 图片格式（仅用于确定文件扩展名，savefig自动推断格式）
        """
        # 临时抑制 SHAP 内部日志
        shap_logger = logging.getLogger("shap")
        original_level = shap_logger.level
        shap_logger.setLevel(logging.WARNING)

        try:
            logging.info(f"  >>> 计算 SHAP (KernelExplainer)")

            # 定义预测函数
            def predict_fn(x):
                if not isinstance(x, pd.DataFrame):
                    x = pd.DataFrame(x, columns=X.columns)
                return trained_pipeline.predict(x)

            # 使用 KernelExplainer
            explainer = shap.KernelExplainer(predict_fn, X, link="identity")

            # 计算 SHAP 值
            shap_values = explainer.shap_values(X, nsamples="auto")

            # 保存 SHAP 值表格到对应模型目录
            shap_df = pd.DataFrame(shap_values, columns=X.columns, index=X.index)
            csv_path = save_path.parent / "SHAP_values.csv"
            shap_df.to_csv(csv_path, index=True)
            logging.info(f"    √ SHAP值表格已保存: {csv_path.name}")

            # 绘图
            plt.figure(figsize=(8, max(4, len(X.columns) * 0.3)))
            shap.summary_plot(shap_values, X, show=False,
                            plot_size=(8, max(4, len(X.columns) * 0.3)), rng=rng)
            plt.tight_layout()

            # 保存图片（format参数冗余，移除；savefig自动从文件扩展名推断格式）
            dpi_param = 300 if figure_format == 'png' else None
            plt.savefig(save_path, dpi=dpi_param, bbox_inches='tight')
            plt.close()

            logging.info(f"    √ SHAP图已保存: {save_path.name}")

        except Exception as e:
            logging.error(f"  × SHAP 失败: {e}")

    def plot_risk_distribution(
        self,
        risk_df: pd.DataFrame,
        save_path: Path,
        figure_format: str = 'png'
    ) -> None:
        """
        绘制风险评分分布图（区分事件/非事件）

        参数:
            risk_df: 包含 risk_score, OS, events 的 DataFrame
            save_path: 图片保存路径
            figure_format: 图片格式
        """
        # 检查数据有效性
        if risk_df["risk_score"].nunique() <= 1:
            logging.warning("  风险评分无变异，跳过分布图绘制")
            return

        # 分离事件和非事件患者
        event_mask = risk_df["events"] == True
        risk_event = risk_df.loc[event_mask, "risk_score"].values
        risk_no_event = risk_df.loc[~event_mask, "risk_score"].values

        # 绘图
        plt.figure(figsize=(8, 5))

        # 绘制直方图
        plt.hist(risk_no_event, bins=30, alpha=0.6, color='blue',
                label=f'No Event (n={len(risk_no_event)})', edgecolor='black', linewidth=0.5)
        plt.hist(risk_event, bins=30, alpha=0.6, color='red',
                label=f'Event (n={len(risk_event)})', edgecolor='black', linewidth=0.5)

        # 添加垂直线（均值）
        plt.axvline(risk_no_event.mean(), color='blue', linestyle='--',
                   linewidth=2, alpha=0.7, label=f'Mean (No Event): {risk_no_event.mean():.3f}')
        plt.axvline(risk_event.mean(), color='red', linestyle='--',
                   linewidth=2, alpha=0.7, label=f'Mean (Event): {risk_event.mean():.3f}')

        # 标签和标题
        plt.xlabel('Risk Score', fontsize=10)
        plt.ylabel('Frequency', fontsize=10)
        plt.title('Risk Score Distribution by Event Status', fontsize=12, fontweight='bold')
        plt.legend(loc='upper right', fontsize=9)
        plt.xticks(fontsize=9)
        plt.yticks(fontsize=9)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()

        # 保存
        dpi_param = 300 if figure_format == 'png' else None
        plt.savefig(save_path, format=figure_format, dpi=dpi_param, bbox_inches='tight')
        plt.close()

        logging.info(f"    √ 风险分布图已保存: {save_path.name}")

    def plot_fold_boxplot(
        self,
        re_df: pd.DataFrame,
        save_path: Path,
        title: str = '',
        figsize: tuple = (4, 3),
        dpi: int = 300,
        figure_format: str = 'png'
    ) -> None:
        """
        绘制交叉验证箱线图

        参数:
            re_df: 结果 DataFrame
            save_path: 保存路径
            title: 图表标题
            figsize: 图像尺寸
            dpi: 分辨率
            figure_format: 图片格式
        """
        # 重置索引，将模型名变为列
        df_reset = re_df.reset_index()
        df_reset.rename(columns={'Model': 'model'}, inplace=True)

        # 处理NA值
        fold_cols = [col for col in df_reset.columns if col.startswith('fold')]

        if not fold_cols:
            raise ValueError("DataFrame 中未找到 'foldX' 列")

        valid_fold_cols = []
        for col in fold_cols:
            if not df_reset[col].isna().any():
                valid_fold_cols.append(col)

        if not valid_fold_cols:
            raise ValueError("所有fold列都包含NA值，无法绘制箱线图")

        # 转换为长格式
        df_long = pd.melt(
            df_reset,
            id_vars=['model'],
            value_vars=valid_fold_cols,
            var_name='fold',
            value_name='value'
        )
        df_long['model'] = df_long['model'].replace('_', '')

        # 绘图
        plt.figure(figsize=figsize)

        ax = sns.boxplot(
            data=df_long,
            x='model',
            y='value',
            hue='model',
            palette='Set2',
            legend=False,
            fliersize=0
        )

        sns.stripplot(
            data=df_long,
            x='model',
            y='value',
            color='gray',
            alpha=0.5,
            jitter=True,
            size=4,
            linewidth=0.5,
            edgecolor='black'
        )

        plt.xticks(rotation=15, ha='right')
        plt.ylabel('Value')
        plt.tight_layout()
        plt.title(title)

        dpi_param = dpi if figure_format == 'png' else None
        plt.savefig(save_path, format=figure_format, dpi=dpi_param, bbox_inches='tight')
        plt.close()
        print(f"箱线图已保存至: {save_path}")

    def plot_cindex_comparison(
        self,
        save_path: Path,
        figure_format: str = 'png'
    ) -> None:
        """
        绘制 C-index 在训练集和测试集对比的箱线图/条形图

        - 横轴：不同的模型
        - 纵轴：C-index
        - 颜色：根据数据集来源（训练集/测试集）进行区分
        - 图像尺寸：4 x 3 英寸

        参数:
            save_path: 图片保存路径
            figure_format: 图片格式
        """
        # 提取数据
        data = []
        for model_name, model_result in self.results.items():
            # 训练集 C-index
            train_cindex = model_result.get('train', {}).get('c_index')
            if train_cindex is not None:
                data.append({
                    'Model': model_name,
                    'Dataset': 'Train',
                    'C-index': train_cindex
                })

            # 测试集 C-index
            test_cindex = model_result.get('vali', {}).get('c_index')
            if test_cindex is not None:
                data.append({
                    'Model': model_name,
                    'Dataset': 'Test',
                    'C-index': test_cindex
                })

        if not data:
            logging.warning("没有可用的 C-index 数据，跳过绘图")
            return

        df = pd.DataFrame(data)

        # 绘图
        plt.figure(figsize=(4, 3))

        # 定义颜色
        palette = {'Train': '#3498db', 'Test': '#e74c3c'}

        # 绘制条形图（每个模型只有一个 C-index 值时使用条形图）
        ax = sns.barplot(
            data=df,
            x='Model',
            y='C-index',
            hue='Dataset',
            palette=palette,
            alpha=0.8
        )

        # 添加数值标签
        for p in ax.patches:
            height = p.get_height()
            if height > 0:  # 只为有效的柱子添加标签
                ax.annotate(
                    f'{height:.3f}',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center',
                    va='bottom',
                    fontsize=7,
                    xytext=(0, 1),
                    textcoords='offset points'
                )

        # 设置图表属性
        plt.xlabel('Model', fontsize=9)
        plt.ylabel('C-index', fontsize=9)
        plt.title('C-index: Train vs Test', fontsize=10, fontweight='bold')
        plt.legend(loc='lower right', fontsize=8)
        plt.grid(True, linestyle='--', alpha=0.3, axis='y')

        # 设置 y 轴范围
        plt.ylim(0.4, 1.0)

        # 旋转 x 轴标签
        plt.xticks(rotation=15, ha='right', fontsize=8)
        plt.yticks(fontsize=8)

        plt.tight_layout()

        # 保存
        dpi_param = 300 if figure_format == 'png' else None
        plt.savefig(save_path, format=figure_format, dpi=dpi_param, bbox_inches='tight')
        plt.close()

        logging.info(f"    √ C-index 对比图已保存: {save_path.name}")
