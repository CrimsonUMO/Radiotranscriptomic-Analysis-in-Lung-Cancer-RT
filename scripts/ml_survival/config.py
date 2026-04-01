"""
配置管理模块

负责解析命令行参数、管理配置、验证配置有效性。
"""

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Config:
    """配置数据类"""

    # 数据路径
    train_features_path: Path
    train_survival_path: Path
    test_features_path: Path
    test_survival_path: Path
    cv_json_path: Path
    output_dir: Path
    figure_dir: Path

    # 特征选择参数
    variance_threshold: float = 1e-2
    corr_threshold: float = 0.9
    enable_cox: bool = False
    cox_alpha: float = 0.05
    cox_k: int | None = 5

    # 数据处理参数
    random_state: int = 42
    remove_zero: bool = False
    clean_outliers: bool = False
    outlier_method: str = 'clip'
    remove_batch_effects: bool = False
    batch_method: str = 'combat'

    # 运行模式
    run_cv: bool = False
    plot: bool = False
    figure_format: str = 'png'
    diagnostic: bool = False

    # 超参数缓存
    best_param_path: Path | None = None

    # 其他配置
    cv_splits: int = 5
    evaluation_times: list[int] = field(default_factory=lambda: [12, 24, 36])


class ConfigManager:
    """配置管理器"""

    @staticmethod
    def parse_args() -> argparse.Namespace:
        """解析命令行参数"""
        parser = argparse.ArgumentParser(
            description="Survival Model Evaluation with Configurable Preprocessing"
        )

        # 特征选择参数
        parser.add_argument("--var", type=float, help="Variance threshold for feature selection")
        parser.add_argument("--cor", type=float, help="Correlation threshold for removing highly correlated features")
        parser.add_argument("--enable_cox", action="store_true", help="Enable univariate Cox-based feature selection")
        parser.add_argument("--alpha", type=float, help="Significance level (alpha) for Cox p-value filtering")
        parser.add_argument("--k", type=int, help="Top-K features to keep if no features pass alpha threshold")

        # 数据处理参数
        parser.add_argument("--random", type=int, help="Random state for reproducibility")
        parser.add_argument("--remove_zero", action="store_true", help="删除生存时间为0的患者")
        parser.add_argument("--diagnostic", action="store_true", help="启用诊断模式，输出详细调试信息")
        parser.add_argument("--clean-outliers", action="store_true", help="在特征选择前清理极端异常值")
        parser.add_argument("--outlier-method", choices=['clip', 'log'], default='clip',
                           help="异常值处理方法: clip=分位数截断, log=对数变换")
        parser.add_argument("--remove-batch-effects", action="store_true",
                           help="启用批次效应去除（多中心验证时使用）")
        parser.add_argument("--batch-method", choices=['combat', 'mean_centering'],
                           default='combat', help="批次校正方法: combat=经验贝叶斯, mean_centering=均值中心化")

        # 运行模式参数
        parser.add_argument("--plot", action="store_true", default=False, help="启用可视化图表绘制（默认不启用）")
        parser.add_argument("--run_cv", action='store_true', help="是否运行交叉验证模式")
        parser.add_argument("--figure-format", choices=['png', 'svg'], default='png', help="图片保存格式 (default: png)")

        # 路径参数
        parser.add_argument("--cv_json", type=str, required=True, help="CV 划分 JSON 文件路径")
        parser.add_argument("--tr", type=str, help="Training set features CSV path")
        parser.add_argument("--ts", type=str, help="Test/Validation set features CSV path")
        parser.add_argument("--trs", type=str, help="Training set survival data CSV/Excel path")
        parser.add_argument("--tss", type=str, help="Test/Validation set survival data CSV/Excel path")
        parser.add_argument("--output", type=str, help="Output directory path (default: ./results)")
        parser.add_argument("--best_param", type=str, default=None,
                           help="Path to save/load best hyperparameters (relative to --output)")

        return parser.parse_args()

    @staticmethod
    def from_args(args: argparse.Namespace) -> Config:
        """
        从命令行参数创建配置对象

        参数:
            args: 解析后的命令行参数

        返回:
            config: 配置对象
        """
        # 创建配置字典
        config_dict = {
            "variance_threshold": args.var if args.var is not None else 1e-2,
            "corr_threshold": args.cor if args.cor is not None else 0.9,
            "enable_cox": args.enable_cox,
            "cox_alpha": args.alpha if args.alpha is not None else 0.05,
            "cox_k": args.k if args.k is not None else 5,
            "random_state": args.random if args.random is not None else 42,
            "remove_zero": args.remove_zero,
            "diagnostic": args.diagnostic,
            "clean_outliers": args.clean_outliers,
            "outlier_method": args.outlier_method,
            "remove_batch_effects": args.remove_batch_effects,
            "batch_method": args.batch_method,
            "run_cv": args.run_cv,
            "plot": args.plot,
            "figure_format": args.figure_format,
            "train_features_path": Path(args.tr),
            "test_features_path": Path(args.ts),
            "train_survival_path": Path(args.trs),
            "test_survival_path": Path(args.tss),
            "cv_json_path": Path(args.cv_json),
        }

        # 输出目录
        if args.output:
            output_dir = Path(args.output)
        else:
            output_dir = Path("./results")

        config_dict["output_dir"] = output_dir
        config_dict["figure_dir"] = output_dir / "figures"

        # 超参数缓存路径
        if args.best_param is not None:
            config_dict["best_param_path"] = output_dir / args.best_param
        else:
            config_dict["best_param_path"] = None

        # 创建配置对象
        config = Config(**config_dict)

        # 保存参数到文件
        config.output_dir.mkdir(exist_ok=True, parents=True)
        params_file = config.output_dir / "params.json"
        ConfigManager._save_params(config, params_file)

        return config

    @staticmethod
    def _save_params(config: Config, params_file: Path) -> None:
        """
        保存配置参数到JSON文件

        参数:
            config: 配置对象
            params_file: 参数文件路径
        """
        config_dict = {
            'train_features_path': str(config.train_features_path),
            'test_features_path': str(config.test_features_path),
            'train_survival_path': str(config.train_survival_path),
            'test_survival_path': str(config.test_survival_path),
            'cv_json_path': str(config.cv_json_path),
            'output_dir': str(config.output_dir),
            'figure_dir': str(config.figure_dir),
            'variance_threshold': config.variance_threshold,
            'corr_threshold': config.corr_threshold,
            'enable_cox': config.enable_cox,
            'cox_alpha': config.cox_alpha,
            'cox_k': config.cox_k,
            'random_state': config.random_state,
            'remove_zero': config.remove_zero,
            'diagnostic': config.diagnostic,
            'clean_outliers': config.clean_outliers,
            'outlier_method': config.outlier_method,
            'remove_batch_effects': config.remove_batch_effects,
            'batch_method': config.batch_method,
            'run_cv': config.run_cv,
            'plot': config.plot,
            'figure_format': config.figure_format,
            'best_param_path': str(config.best_param_path) if config.best_param_path else None,
        }

        with open(params_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

    @staticmethod
    def validate_config(config: Config) -> None:
        """
        验证配置有效性

        参数:
            config: 配置对象

        异常:
            ValueError: 配置无效时抛出
        """
        # 验证数据路径存在
        if not config.train_features_path.exists():
            raise ValueError(f"训练集特征文件不存在: {config.train_features_path}")
        if not config.train_survival_path.exists():
            raise ValueError(f"训练集生存数据文件不存在: {config.train_survival_path}")
        if not config.test_features_path.exists():
            raise ValueError(f"测试集特征文件不存在: {config.test_features_path}")
        if not config.test_survival_path.exists():
            raise ValueError(f"测试集生存数据文件不存在: {config.test_survival_path}")

        # 验证参数范围
        if config.variance_threshold < 0:
            raise ValueError(f"方差阈值必须 >= 0，当前值: {config.variance_threshold}")
        if not (0 < config.corr_threshold < 1):
            raise ValueError(f"相关性阈值必须在 (0, 1) 范围内，当前值: {config.corr_threshold}")
        if config.cox_alpha < 0 or config.cox_alpha > 1:
            raise ValueError(f"Cox alpha 必须在 [0, 1] 范围内，当前值: {config.cox_alpha}")
        if config.cox_k is not None and config.cox_k <= 0:
            raise ValueError(f"Cox k 必须大于 0，当前值: {config.cox_k}")

        # 验证枚举值
        if config.outlier_method not in ['clip', 'log']:
            raise ValueError(f"异常值方法必须是 'clip' 或 'log'，当前值: {config.outlier_method}")
        if config.batch_method not in ['combat', 'mean_centering']:
            raise ValueError(f"批次校正方法必须是 'combat' 或 'mean_centering'，当前值: {config.batch_method}")
        if config.figure_format not in ['png', 'svg']:
            raise ValueError(f"图片格式必须是 'png' 或 'svg'，当前值: {config.figure_format}")

    @staticmethod
    def print_summary(config: Config) -> None:
        """
        打印本次运行的详细参数配置

        参数:
            config: 配置对象
        """
        print("\n" + "="*60)
        print("运行参数配置")
        print("="*60)

        # 基本参数
        print(f"[基本参数]")
        print(f"  随机种子: {config.random_state}")
        print(f"  交叉验证折数: {config.cv_splits}")
        print(f"  评估时间点(月): {config.evaluation_times}")
        print(f"  输出目录: {config.output_dir}")

        # 特征选择参数
        print(f"\n[特征选择参数]")
        print(f"  方差阈值: {config.variance_threshold}")
        print(f"  相关性阈值: {config.corr_threshold}")
        print(f"  启用单变量Cox: {config.enable_cox}")
        if config.enable_cox:
            print(f"    Cox p值阈值(alpha): {config.cox_alpha}")
            print(f"    Cox 选择特征数(k): {config.cox_k}")

        # 数据处理参数
        print(f"\n[数据处理参数]")
        print(f"  移除生存时间<=0的患者: {config.remove_zero}")
        print(f"  清理极端异常值: {config.clean_outliers}")
        if config.clean_outliers:
            print(f"    异常值处理方法: {config.outlier_method}")
        print(f"  批次效应去除: {config.remove_batch_effects}")
        if config.remove_batch_effects:
            print(f"    批次校正方法: {config.batch_method}")

        # 诊断模式
        print(f"\n[诊断]")
        print(f"  诊断模式: {config.diagnostic}")

        # 数据路径
        print(f"\n[数据路径]")
        print(f"  训练集特征: {config.train_features_path}")
        print(f"  验证集特征: {config.test_features_path}")
        print(f"  训练集生存数据: {config.train_survival_path}")
        print(f"  验证集生存数据: {config.test_survival_path}")

        print("="*60 + "\n")
