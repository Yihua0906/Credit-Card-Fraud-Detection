# Credit-Card-Fraud-Detection
## 項目概述

這個信用卡詐欺檢測系統利用機器學習技術來識別信用卡交易中的詐欺行為。本系統使用隨機森林分類器（RandomForestClassifier）作為主要模型，並應用 SMOTE（合成少數類過採樣技術）來解決類別不平衡問題。模型的性能評估指標包括 AUPRC（平均精確度-召回率曲線）和 ROC-AUC（接收者操作特徵曲線-面積）。

## 使用技術
RandomForestClassifier：一種集成學習方法，通過多棵決策樹的投票結果進行分類。

SMOTE（Synthetic Minority Over-sampling Technique）：用於生成合成少數類樣本，以改善模型在不平衡數據上的表現。

Scikit-learn：用於數據預處理、模型訓練和評估。

Imbalanced-learn：提供 SMOTE 技術的實現。

Pandas & NumPy：用於數據處理和操作。

Matplotlib & Seaborn：用於可視化數據和評估結果。

Joblib：用於儲存和載入訓練好的模型。


## 安裝要求
安裝 Python 3.6 及以上版本。
安裝所需的依賴庫：
pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn joblib

## 資料集
本系統使用的數據集來自 European Credit Card Fraud Detection，包含 2013 年 9 月的歐洲信用卡交易數據。

總樣本數：284,807 筆交易

詐欺交易數：492 筆（約 0.172%）

欄位說明：

V1 - V28：經 PCA 轉換的 28 個匿名特徵。

Time：交易發生時間（與第一筆交易的秒數差）。

Amount：交易金額。

Class：標籤，1 代表詐欺交易，0 代表正常交易。

## 使用方法

### 訓練模型：
在運行模型前，請將 creditcard.csv 放置在項目目錄下。
執行以下代碼來訓練並儲存模型：

python model.py
### 模型推斷：

使用訓練好的模型進行預測，可以在 predict.py 中找到代碼範本來對新交易進行預測。

### API 接口使用：

本系統提供了一個 API 接口，允許外部系統通過 HTTP 請求進行詐欺預測。

#### 啟動 Flask 伺服器：
首先，確保你已經訓練並儲存了模型。然後執行以下命令啟動 Flask 伺服器：
python app.py
伺服器將在 http://127.0.0.1:5000/ 運行。

#### 請求格式：
該 API 支援 POST 請求，請將交易資料以 JSON 格式發送到 /predict 端點。請求的資料格式如下：

json
{
  "input": [
    {
      "V1": -0.1743,
      "V2": 0.2345,
      "V3": -0.3456,
      ...
      "Time": 123456,
      "Amount": 100.50
    }
  ]
}

#### 響應格式：
API 將返回以下格式的 JSON 響應，告訴您該交易是否為詐欺：

json
{
  "prediction": 1
}

prediction: 1 表示詐欺交易，0 表示正常交易。

### 評估指標：

該系統的性能通過以下指標來評估：

AUPRC（Average Precision-Recall Curve）：衡量模型在處理不平衡數據時的效果。

ROC-AUC（Receiver Operating Characteristic - Area Under Curve）：評估模型區分正負類的能力。
## 相關文獻
Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015). Calibrating Probability with Undersampling for Unbalanced Classification. In Symposium on Computational Intelligence and Data Mining (CIDM).

Dal Pozzolo, A., Caelen, O., Le Borgne, Y.-A., Waterschoot, S., & Bontempi, G. (2014). Learned lessons in credit card fraud detection from a practitioner perspective. Expert systems with applications, 41(10), 4915-4928.
