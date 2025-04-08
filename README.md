# Credit-Card-Fraud-Detection
## 項目概述

本專案旨在建構一套能夠準確偵測信用卡交易中詐騙行為的機器學習模型。使用了 XGBoost 作為分類器，並結合 ADASYN 過取樣技術處理不平衡資料，最終模型在最佳化閾值後能顯著提升預測詐騙的準確性與召回率。

## 功能特色

使用 XGBoost 模型進行分類

資料標準化（Amount 與 Time）

採用 ADASYN 解決不平衡資料問題

模型訓練與預測

自動進行閾值優化以最大化 F1 分數

視覺化：

特徵重要性圖

精確率-召回率（PR）曲線


## 使用的套件 
pandas, numpy

matplotlib, seaborn

scikit-learn

xgboost

imbalanced-learn（ADASYN）

joblib, pickle

Google Colab 支援



## 資料集
本系統使用的數據集來自 European Credit Card Fraud Detection，包含 2013 年 9 月的歐洲信用卡交易數據。

特徵數量：30

資料筆數：284,807

詐騙樣本比例：僅約 0.17%

### 數據分佈

類別          /原始數據	         /ADASYN 後

0（正常）	    /284,315	       /227,451


1（詐騙）	    /492	           /45,516

欄位說明：

V1 - V28：經 PCA 轉換的 28 個匿名特徵。

Time：交易發生時間（與第一筆交易的秒數差）。

Amount：交易金額。

Class：標籤，1 代表詐欺交易，0 代表正常交易。

## 最佳化閾值評估結果（Threshold = 0.9301）
F1-score：0.8478

精確率：0.9070

召回率：0.7959
