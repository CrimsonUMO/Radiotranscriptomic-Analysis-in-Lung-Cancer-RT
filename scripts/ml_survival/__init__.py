"""
ml_survival - 生存分析模块化包

用于癌症患者的生存分析，支持多种机器学习模型。
"""

from .config import Config, ConfigManager
from .data_manager import DataManager
from .feature_selector import FeatureSelector, CorrelationFilter, UnivariateCoxSelector
from .models import ModelFactory, ModelRegistry
from .trainer import ModelTrainer, Evaluator
from .visualizer import VisualizationManager
from .batch_correction import BatchCorrector
from .utils import lock_random, load_cv_splits, create_cv_splits
from .validator import (
    DataValidator,
    DataValidationError,
    MissingColumnError,
    InvalidDataTypeError,
    InvalidValueError,
    PatientMismatchError
)
from .__main__ import main

__version__ = "2.0.0"
__all__ = [
    "Config", "ConfigManager",
    "DataManager",
    "FeatureSelector", "CorrelationFilter", "UnivariateCoxSelector",
    "ModelFactory", "ModelRegistry",
    "ModelTrainer", "Evaluator",
    "VisualizationManager",
    "BatchCorrector",
    "lock_random", "load_cv_splits", "create_cv_splits",
    "DataValidator",
    "DataValidationError",
    "MissingColumnError",
    "InvalidDataTypeError",
    "InvalidValueError",
    "PatientMismatchError",
    "main",
]
