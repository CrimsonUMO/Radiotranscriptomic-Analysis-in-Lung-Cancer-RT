"""
数据验证模块

提供输入数据格式验证功能，确保数据符合ml_survival包的要求。
"""

import logging
from typing import Any

import numpy as np
import pandas as pd


# =============================================================================
# Custom Exceptions
# =============================================================================

class DataValidationError(Exception):
    """数据验证错误基类"""
    pass


class MissingColumnError(DataValidationError):
    """缺失必需列错误"""

    def __init__(self, file_type: str, missing_columns: list[str]):
        self.file_type = file_type
        self.missing_columns = missing_columns
        super().__init__(
            f"{file_type}文件缺失必需列: {', '.join(missing_columns)}"
        )


class InvalidDataTypeError(DataValidationError):
    """无效数据类型错误"""

    def __init__(self, column: str, expected_type: str, actual_type: str):
        self.column = column
        self.expected_type = expected_type
        self.actual_type = actual_type
        super().__init__(
            f"列 '{column}' 类型错误: 期望 {expected_type}, 实际 {actual_type}"
        )


class InvalidValueError(DataValidationError):
    """无效值错误"""

    def __init__(self, column: str, description: str):
        self.column = column
        self.description = description
        super().__init__(
            f"列 '{column}' 包含无效值: {description}"
        )


class PatientMismatchError(DataValidationError):
    """患者ID不匹配错误"""

    def __init__(self, feature_count: int, survival_count: int):
        self.feature_count = feature_count
        self.survival_count = survival_count
        super().__init__(
            f"特征文件和生存文件的患者ID数量不匹配: "
            f"特征文件={feature_count}, 生存文件={survival_count}"
        )


# =============================================================================
# Data Validator
# =============================================================================

class DataValidator:
    """
    数据格式验证器

    验证输入数据文件的格式、类型和完整性，确保数据符合ml_survival包的要求。
    """

    # Required columns for each file type
    FEATURE_REQUIRED_COLUMNS = ["patient"]
    SURVIVAL_REQUIRED_COLUMNS = ["events", "OS"]

    # Valid values for categorical columns
    VALID_EVENTS_VALUES = {0, 1, 0.0, 1.0}

    def __init__(self, strict: bool = True):
        """
        初始化数据验证器

        参数:
            strict: 是否使用严格模式（严格模式下遇到错误立即抛出异常）
        """
        self.strict = strict
        self.errors: list[str] = []
        self.warnings: list[str] = []

    # ---------------------------------------------------------------------
    # Public Validation Methods
    # ---------------------------------------------------------------------

    def validate_dataset(
        self,
        df_features: pd.DataFrame,
        df_survival: pd.DataFrame,
        dataset_name: str = "dataset"
    ) -> bool:
        """
        综合验证数据集

        参数:
            df_features: 特征DataFrame
            df_survival: 生存数据DataFrame
            dataset_name: 数据集名称（用于日志）

        返回:
            bool: 验证是否通过
        """
        logging.info(f"验证数据集: {dataset_name}")
        self._clear_messages()

        try:
            self.validate_feature_file(df_features, dataset_name)
            self.validate_survival_file(df_survival, dataset_name)
            self.validate_patient_matching(df_features, df_survival, dataset_name)

            if self.warnings:
                for warning in self.warnings:
                    logging.warning(warning)

            logging.info(f"数据集 {dataset_name} 验证通过")
            return True

        except DataValidationError as e:
            logging.error(f"数据集 {dataset_name} 验证失败: {e}")
            if self.strict:
                raise
            return False

    def validate_feature_file(
        self,
        df: pd.DataFrame,
        file_name: str = "feature file"
    ) -> bool:
        """
        验证特征文件格式

        参数:
            df: 特征DataFrame
            file_name: 文件名称（用于日志）

        返回:
            bool: 验证是否通过

        抛出:
            MissingColumnError: 缺失必需列
        """
        logging.info(f"  验证特征文件: {file_name}")

        # Check required columns
        missing_cols = [
            col for col in self.FEATURE_REQUIRED_COLUMNS
            if col not in df.columns
        ]
        if missing_cols:
            raise MissingColumnError("特征文件", missing_cols)

        # Check patient column type
        if not pd.api.types.is_string_dtype(df["patient"]):
            self.warnings.append(
                f"特征文件 'patient' 列非字符串类型，可能影响患者ID匹配"
            )

        # Check for empty patient IDs
        empty_patients = df["patient"].isna().sum()
        if empty_patients > 0:
            raise InvalidValueError(
                "patient",
                f"发现 {empty_patients} 个空患者ID"
            )

        # Check for duplicate patient IDs
        duplicates = df["patient"].duplicated().sum()
        if duplicates > 0:
            self.warnings.append(
                f"特征文件发现 {duplicates} 个重复患者ID"
            )

        logging.info(f"  特征文件验证通过: {len(df)} 行, {len(df.columns)} 列")
        return True

    def validate_survival_file(
        self,
        df: pd.DataFrame,
        file_name: str = "survival file"
    ) -> bool:
        """
        验证生存数据文件格式

        参数:
            df: 生存数据DataFrame
            file_name: 文件名称（用于日志）

        返回:
            bool: 验证是否通过

        抛出:
            MissingColumnError: 缺失必需列
            InvalidDataTypeError: 列数据类型错误
            InvalidValueError: 包含无效值
        """
        logging.info(f"  验证生存文件: {file_name}")

        # Check required columns (support both 'Name' and 'me' as patient ID column)
        patient_col = None
        for col in ["patient"]:
            if col in df.columns:
                patient_col = col
                break

        if patient_col is None:
            print(df.columns)
            raise MissingColumnError("生存文件", ["Patient"])

        # Check events and OS columns
        missing_cols = [
            col for col in self.SURVIVAL_REQUIRED_COLUMNS
            if col not in df.columns
        ]
        if missing_cols:
            raise MissingColumnError("生存文件", missing_cols)

        # Validate events column - CRITICAL: must be numeric, not strings like "Dead"/"Alive"
        self._validate_events_column(df)

        # Validate OS column
        self._validate_os_column(df)

        # Check for missing critical values
        na_events = df["events"].isna().sum()
        na_os = df["OS"].isna().sum()

        # Events NA values: warn instead of error (will be filtered during loading)
        if na_events > 0:
            self.warnings.append(
                f"生存文件 'events' 列发现 {na_events} 个NA值，这些患者将被过滤"
            )

        # OS NA values: warn (will be filtered during loading)
        if na_os > 0:
            self.warnings.append(
                f"生存文件 'OS' 列发现 {na_os} 个NA值，这些患者将被过滤"
            )

        # Check for zero/negative survival times
        invalid_os = (df["OS"] <= 0).sum()
        if invalid_os > 0:
            self.warnings.append(
                f"生存文件发现 {invalid_os} 个OS值 <= 0，这些患者将被过滤"
            )

        logging.info(
            f"  生存文件验证通过: {len(df)} 行, "
            f"事件率={df['events'].astype(bool).mean():.2%}"
        )
        return True

    def validate_patient_matching(
        self,
        df_features: pd.DataFrame,
        df_survival: pd.DataFrame,
        dataset_name: str = "dataset"
    ) -> bool:
        """
        验证特征文件和生存文件的患者ID匹配

        参数:
            df_features: 特征DataFrame
            df_survival: 生存数据DataFrame
            dataset_name: 数据集名称（用于日志）

        返回:
            bool: 验证是否通过

        抛出:
            PatientMismatchError: 患者ID数量严重不匹配
        """
        logging.info(f"  验证患者ID匹配: {dataset_name}")

        # Get patient ID column name from survival file
        patient_col = "patient"

        feature_patients = set(df_features["patient"].astype(str))
        survival_patients = set(df_survival[patient_col].astype(str))

        common_patients = feature_patients & survival_patients
        only_in_features = feature_patients - survival_patients
        only_in_survival = survival_patients - feature_patients

        logging.info(f"    特征文件患者数: {len(feature_patients)}")
        logging.info(f"    生存文件患者数: {len(survival_patients)}")
        logging.info(f"    共有患者数: {len(common_patients)}")

        if only_in_features:
            self.warnings.append(
                f"仅存在于特征文件的患者: {len(only_in_features)} 个"
            )

        if only_in_survival:
            self.warnings.append(
                f"仅存在于生存文件的患者: {len(only_in_survival)} 个"
            )

        # Critical check: if overlap is too small, raise error
        overlap_ratio = len(common_patients) / min(len(feature_patients), len(survival_patients))
        if overlap_ratio < 0.5:
            raise PatientMismatchError(len(feature_patients), len(survival_patients))

        logging.info(f"  患者ID匹配验证通过: {len(common_patients)} 个共有患者")
        return True

    # ---------------------------------------------------------------------
    # Private Validation Methods
    # ---------------------------------------------------------------------

    def _validate_events_column(self, df: pd.DataFrame) -> None:
        """
        验证events列的数据类型和值

        关键验证：拒绝字符串类型（如"Dead"/"Alive"），只接受数值0/1

        抛出:
            InvalidDataTypeError: events列不是数值类型
            InvalidValueError: events列包含无效值
        """
        events_col = df["events"]

        # Check if column is string type (critical error!)
        if pd.api.types.is_string_dtype(events_col):
            # Check for common string values that should NOT be used
            unique_vals = events_col.dropna().unique()
            string_vals = [str(v).lower() for v in unique_vals if isinstance(v, str)]

            if any(val in string_vals for val in ["dead", "alive", "death", "survived"]):
                raise InvalidDataTypeError(
                    "events",
                    "numeric (0/1)",
                    f"string (发现值: {', '.join(map(str, unique_vals[:5]))})"
                )

        # Check numeric type
        if not pd.api.types.is_numeric_dtype(events_col):
            try:
                # Try to convert to numeric
                pd.to_numeric(events_col, errors="raise")
            except (ValueError, TypeError):
                raise InvalidDataTypeError(
                    "events",
                    "numeric (0/1)",
                    str(events_col.dtype)
                )

        # Convert to numeric for value checking
        events_numeric = pd.to_numeric(events_col, errors="coerce")

        # Check for invalid values (not 0 or 1)
        unique_values = set(events_numeric.dropna().unique())
        invalid_values = unique_values - self.VALID_EVENTS_VALUES

        if invalid_values:
            raise InvalidValueError(
                "events",
                f"发现无效值 {invalid_values}, 只接受 0 或 1"
            )

    def _validate_os_column(self, df: pd.DataFrame) -> None:
        """
        验证OS列的数据类型

        抛出:
            InvalidDataTypeError: OS列不是数值类型
        """
        os_col = df["OS"]

        if not pd.api.types.is_numeric_dtype(os_col):
            try:
                pd.to_numeric(os_col, errors="raise")
            except (ValueError, TypeError):
                raise InvalidDataTypeError(
                    "OS",
                    "numeric",
                    str(os_col.dtype)
                )

    # ---------------------------------------------------------------------
    # Utility Methods
    # ---------------------------------------------------------------------

    def _clear_messages(self) -> None:
        """清除错误和警告信息"""
        self.errors.clear()
        self.warnings.clear()

    def get_validation_report(self) -> str:
        """
        获取验证报告

        返回:
            验证报告字符串
        """
        report_lines = [
            "=" * 60,
            "Data Validation Report",
            "=" * 60,
        ]

        if self.errors:
            report_lines.append("\nErrors:")
            for error in self.errors:
                report_lines.append(f"  [ERROR] {error}")

        if self.warnings:
            report_lines.append("\nWarnings:")
            for warning in self.warnings:
                report_lines.append(f"  [WARN] {warning}")

        if not self.errors and not self.warnings:
            report_lines.append("\nNo issues found. Data validation passed.")

        report_lines.append("=" * 60)

        return "\n".join(report_lines)
