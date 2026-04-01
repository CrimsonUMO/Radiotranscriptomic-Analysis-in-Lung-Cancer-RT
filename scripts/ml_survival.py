#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生存分析主脚本 - 模块化包版本

使用 ml_survival 包进行癌症患者的生存分析。
支持多种机器学习模型：Cox Net, Cox PH, RSF, GBM, SVM

用法示例:
    python ml_survival.py --tr train_features.csv \\
                          --trs train_survival.csv \\
                          --ts test_features.csv \\
                          --tss test_survival.csv \\
                          --cv_json cv_splits.json \\
                          --enable_cox --var 0.01 --cor 0.95 --k 10 \\
                          --output results
"""

from ml_survival import main

if __name__ == "__main__":
    main()
