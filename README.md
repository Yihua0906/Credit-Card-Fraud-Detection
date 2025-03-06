# Credit-Card-Fraud-Detection
項目概述
這個信用卡詐欺檢測系統旨在通過機器學習技術識別信用卡交易中的詐欺行為。該系統使用 XGBoost 模型訓練，並應用了過抽樣技術（如 SMOTE）來處理類別不平衡問題。該模型能夠檢測不平衡數據集中的詐欺交易，並提供 AUPRC（平均精確度-召回率曲線）和 ROC-AUC（接收者操作特徵曲線-面積）等評估指標。

使用技術
XGBoost：一個高效能的提升樹模型，用於二分類問題。
SMOTE (Synthetic Minority Over-sampling Technique)：用於解決類別不平衡問題，生成合成少數類樣本。
Scikit-learn：用於數據預處理、模型訓練、模型評估。
Joblib：用於儲存和載入訓練好的模型。

安裝要求
安裝 Python 3.6 及以上版本。
安裝所需的依賴庫：
pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn joblib

資料集
這個項目使用的數據集來自於 European Credit Card Fraud Detection，包含了 2013 年 9 月的歐洲信用卡交易資料，其中有 492 條詐欺交易和 284,807 條正常交易。由於數據集的高度不平衡，詐欺交易佔比僅為 0.172%。
數據集的欄位：
V1-V28: 經過 PCA 轉換的 28 個特徵。
Time: 交易發生的時間，與第一筆交易的秒數差。
Amount: 交易金額。
Class: 標籤，1 代表詐欺交易，0 代表正常交易。

使用方法
訓練模型：
在運行模型前，請將 creditcard.csv 放置在項目目錄下。
執行以下代碼來訓練並儲存模型：

python model.py
模型推斷：

使用訓練好的模型進行預測，可以在 predict.py 中找到代碼範本來對新交易進行預測。
評估指標：

該系統的性能通過 AUPRC 和 ROC-AUC 來評估：
AUPRC (Average Precision-Recall Curve)：反映了模型在處理不平衡數據時的性能。
ROC-AUC：表示模型區分正負類別的能力。

相關文獻
Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015). Calibrating Probability with Undersampling for Unbalanced Classification. In Symposium on Computational Intelligence and Data Mining (CIDM).
Dal Pozzolo, A., Caelen, O., Le Borgne, Y.-A., Waterschoot, S., & Bontempi, G. (2014). Learned lessons in credit card fraud detection from a practitioner perspective. Expert systems with applications, 41(10), 4915-4928.
