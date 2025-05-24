# Pattern Recognition API

## 概要
このAPIは、パターン認識モデルを使用して入力パラメータからクラスを予測します。

## エンドポイント

### ヘルスチェック
```
GET /health
```

#### レスポンス
```json
{
    "status": "healthy",
    "message": "API is running"
}
```

### 予測
```
POST /predict
```

#### リクエスト
Content-Type: application/json

```json
[
    {
        "param1": 0.5,
        "param2": 1.2,
        "param3": -0.7,
        "param4": 3.4,
        "param5": 0.0,
        "param6": 2.1
    },
    {
        "param1": 1.0,
        "param2": 2.3,
        "param3": -1.5,
        "param4": 4.1,
        "param5": 0.5,
        "param6": 3.2
    }
]
```

#### パラメータ
- `param1` - `param6`: 数値型（float）のパラメータ
  - 必須パラメータ
  - 数値以外の値はエラーとなります

#### レスポンス
```json
{
    "predictions": [
        {
            "predicted_class": "C1",
            "probabilities": {
                "C1": 0.8,
                "C2": 0.1,
                "C3": 0.1
            }
        },
        {
            "predicted_class": "C2",
            "probabilities": {
                "C1": 0.2,
                "C2": 0.7,
                "C3": 0.1
            }
        }
    ]
}
```

#### エラーレスポンス
```json
{
    "error": "エラーメッセージ",
    "message": "詳細なエラー情報"
}
```

##### エラーコード
- 400 Bad Request
  - 必須パラメータが不足している場合
  - パラメータの型が不正な場合
  - リクエストの形式が不正な場合（配列でない場合）
- 500 Internal Server Error
  - サーバー内部でエラーが発生した場合

## 使用例

### cURL
```bash
curl -X POST https://your-app-name.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '[
    {
      "param1": 0.5,
      "param2": 1.2,
      "param3": -0.7,
      "param4": 3.4,
      "param5": 0.0,
      "param6": 2.1
    },
    {
      "param1": 1.0,
      "param2": 2.3,
      "param3": -1.5,
      "param4": 4.1,
      "param5": 0.5,
      "param6": 3.2
    }
  ]'
```

### Python
```python
import requests
import json

url = "https://your-app-name.onrender.com/predict"
data = [
    {
        "param1": 0.5,
        "param2": 1.2,
        "param3": -0.7,
        "param4": 3.4,
        "param5": 0.0,
        "param6": 2.1
    },
    {
        "param1": 1.0,
        "param2": 2.3,
        "param3": -1.5,
        "param4": 4.1,
        "param5": 0.5,
        "param6": 3.2
    }
]

response = requests.post(url, json=data)
predictions = response.json()

# 予測結果の処理
for i, pred in enumerate(predictions["predictions"]):
    print(f"予測 {i+1}:")
    print(f"  予測クラス: {pred['predicted_class']}")
    print(f"  確率: {pred['probabilities']}")
```
