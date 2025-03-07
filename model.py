import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, average_precision_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 讀取信用卡詐欺數據集
data = pd.read_csv('creditcard.csv')

# 確保 Amount 欄位存在
if 'Amount' in data.columns:
    data['Amount'] = data['Amount'].fillna(data['Amount'].median())

# 確保 Class 欄位存在
if 'Class' in data.columns:
    data['Class'] = data['Class'].fillna(data['Class'].mode()[0])

# 填補 `V` 欄位的缺失值
for column in data.columns:
    if column.startswith('V'):
        data[column] = data[column].fillna(data[column].median())

# 標準化 `Time` 和 `Amount`
scaler = StandardScaler()
columns_to_scale = [col for col in ['Time', 'Amount'] if col in data.columns]
data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])

# 分割數據集
X = data.drop('Class', axis=1)
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用 SMOTE 進行過采樣來處理不平衡數據
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 建立隨機森林模型
rf_model = RandomForestClassifier(random_state=42, n_estimators=200, class_weight='balanced')

# 訓練模型
rf_model.fit(X_train_res, y_train_res)

# 預測結果
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]  # 獲得詐欺交易的機率

# 顯示 ROC-AUC 和 AUPRC
roc_auc = roc_auc_score(y_test, y_pred_proba)
auprc = average_precision_score(y_test, y_pred_proba)

print(f"Model AUPRC: {auprc:.4f}")
print(f"Model ROC-AUC: {roc_auc:.4f}")

# 顯示混淆矩陣
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 顯示分類報告
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 繪製混淆矩陣的熱圖
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Fraud', 'Fraud'], yticklabels=['No Fraud', 'Fraud'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# 儲存模型
joblib.dump(rf_model, 'fraud_detection_model_rf.pkl')
scaler_filename = "scaler.pkl"
joblib.dump(scaler, scaler_filename)  # 儲存標準化器
print("隨機森林模型已儲存為 fraud_detection_model_rf.pkl")
