from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd  
app = Flask(__name__)

# 載入訓練好的模型與標準化器
rf_model = joblib.load("fraud_detection_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 從 POST 請求中獲取 JSON 資料
        data = request.get_json()

        # 檢查 'input' 是否存在並獲取其內容
        if 'input' not in data:
            return jsonify({"error": "'input' key is missing in the request"}), 400
        
        input_data = data['input']
        
        # 確保資料是正確的格式（轉換為 DataFrame）
        df = pd.DataFrame([input_data])

        # 標準化 'Time' 和 'Amount' 欄位
        columns_to_scale = ['Time', 'Amount']
        df[columns_to_scale] = scaler.transform(df[columns_to_scale])

        # 預測詐騙 (Class)
        prediction = rf_model.predict(df)

        # 回傳預測結果
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("啟動 Flask 伺服器...")
    app.run(debug=True)
