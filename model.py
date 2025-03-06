import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
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

# 確保 Class 欄位存在
if 'Class' not in data.columns:
    raise ValueError("Class 欄位不存在於數據集中，請確認數據格式。")

# 分割數據集
X = data.drop('Class', axis=1)
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 訓練隨機森林模型
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# 預測結果
rf_preds = rf_model.predict(X_test)

# 顯示正確率
accuracy = rf_model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy}")

# 顯示混淆矩陣
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_preds))

# 顯示分類報告
print("Classification Report:")
print(classification_report(y_test, rf_preds))

# 繪製混淆矩陣的熱圖
cm = confusion_matrix(y_test, rf_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Fraud', 'Fraud'], yticklabels=['No Fraud', 'Fraud'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# 儲存模型
joblib.dump(rf_model, "fraud_detection_model.pkl")
joblib.dump(scaler, "scaler.pkl")  # 儲存標準化器，以便 API 使用
print("模型已儲存為 fraud_detection_model.pkl")
