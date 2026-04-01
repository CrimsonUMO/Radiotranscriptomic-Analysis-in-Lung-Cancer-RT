# Radiotranscriptomic Analysis in Lung Cancer RT

A radiotranscriptomic analysis project for lung cancer radiotherapy, combining radiomics features and transcriptomic data for survival analysis modeling.

## Overview

This project aims to extract radiomics features from CT images, build machine learning models to predict survival prognosis in lung cancer patients, and integrate transcriptomic data for multi-omics joint analysis.

## Directory Structure

```
Radiotranscriptomic Analysis in Lung Cancer RT/
├── scripts/                        # Analysis scripts
│   ├── ml_survival.py              # Survival analysis main entry
│   └── ml_survival/                # Modular survival analysis package
│       ├── __main__.py             # Main workflow controller
│       ├── config.py               # Configuration management
│       ├── data_manager.py         # Data loading and preprocessing
│       ├── feature_selector.py     # Feature selection
│       ├── models.py               # Model definitions
│       ├── trainer.py              # Training and evaluation
│       ├── visualizer.py           # Visualization
│       └── ...
├── data/                           # Data directory
│   ├── X_train.csv                 # Training set features
│   ├── X_test.csv                  # Test set features
│   ├── y_train.csv                 # Training set survival data
│   └── y_test.csv                  # Test set survival data
└── README.md                       # This file
```

## Core Modules

### Survival Analysis (scripts/ml_survival/)

A modular survival analysis package supporting multiple survival models:

- **Cox Proportional Hazards Model** (Cox PH, Cox Net)
- **Random Survival Forest** (RSF)
- **Gradient Boosting Machine** (GBM)
- **Survival Support Vector Machine** (Fast SVM)

**Key Features**:

- Data validation and preprocessing
- Feature selection (variance filtering, correlation filtering, univariate Cox)
- Batch effect correction (ComBat)
- Model training and evaluation
- Visualization (KM curves, ROC curves, SHAP plots)

**Quick Start**:

```bash
cd scripts
python ml_survival.py --help
```

## Data Format Requirements

### Feature Files

- Must contain `patient` column as patient ID
- Remaining columns are numerical radiomics features

### Survival Data Files

- `patient`: Patient ID
- `OS`: Overall survival time (months)
- `events`: Event status (0=censored, 1=event)

## Output Results

After running survival analysis, results are saved in the `results/` directory:

- `survival_cv_results.csv` - Model performance metrics
- `cindex_data.csv` - C-index per fold
- `{Model}_risk_scores.csv` - Patient risk scores
- `{Model}_KM.svg` - Kaplan-Meier survival curves
- `{Model}_ROC.svg` - Time-dependent ROC curves
- `SHAP_{Model}.svg` - SHAP feature importance plots

---

# 影像转录组分析项目（肺癌放疗）

肺癌放疗的影像转录组分析项目，结合影像组学特征和转录组数据进行生存分析建模。

## 项目概述

本项目旨在从CT影像中提取影像组学特征，构建机器学习模型预测肺癌患者的生存预后，并结合转录组数据进行多组学联合分析。

## 目录结构

```
Radiotranscriptomic Analysis in Lung Cancer RT/
├── scripts/                        # 分析脚本目录
│   ├── ml_survival.py              # 生存分析主入口
│   └── ml_survival/                # 生存分析模块化包
│       ├── __main__.py             # 主流程控制器
│       ├── config.py               # 配置管理
│       ├── data_manager.py         # 数据加载与预处理
│       ├── feature_selector.py     # 特征选择
│       ├── models.py               # 模型定义
│       ├── trainer.py              # 训练与评估
│       ├── visualizer.py           # 可视化
│       └── ...
├── data/                           # 数据目录
│   ├── X_train.csv                 # 训练集特征
│   ├── X_test.csv                  # 测试集特征
│   ├── y_train.csv                 # 训练集生存数据
│   └── y_test.csv                  # 测试集生存数据
└── README.md                       # 本文件
```

## 核心功能模块

### 生存分析 (scripts/ml_survival/)

模块化的生存分析包，支持多种生存模型：

- **Cox比例风险模型** (Cox PH, Cox Net)
- **随机生存森林** (RSF)
- **梯度提升** (GBM)
- **生存支持向量机** (Fast SVM)

**主要功能**：

- 数据验证与预处理
- 特征选择（方差过滤、相关性过滤、单变量Cox）
- 批效应校正（ComBat）
- 模型训练与评估
- 可视化（KM曲线、ROC曲线、SHAP图）

**快速开始**：

```bash
cd scripts
python ml_survival.py --help
```

## 数据格式要求

### 特征文件

- 必须包含 `patient` 列作为患者ID
- 其余列为数值型影像组学特征

### 生存数据文件

- `patient`: 患者ID
- `OS`: 总生存时间（月）
- `events`: 事件状态（0=删失，1=事件）

## 输出结果

运行生存分析后，结果保存在 `results/` 目录：

- `survival_cv_results.csv` - 模型性能指标
- `cindex_data.csv` - 每折C-index
- `{Model}_risk_scores.csv` - 患者风险评分
- `{Model}_KM.svg` - KM生存曲线
- `{Model}_ROC.svg` - 时间依赖ROC曲线
- `SHAP_{Model}.svg` - SHAP特征重要性图
