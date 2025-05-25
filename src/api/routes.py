from flask import jsonify, request
import numpy as np
import logging
from .validators import validate_predict_input

logger = logging.getLogger(__name__)

def register_routes(app):
    """ルートの登録"""
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """ヘルスチェックエンドポイント"""
        return jsonify({
            "status": "healthy",
            "message": "API is running"
        })

    @app.route('/predict', methods=['POST'])
    def predict():
        """予測エンドポイント"""
        try:
            # 入力データの取得とバリデーション
            data = request.get_json()
            if not isinstance(data, list):
                raise ValueError("Input must be a list of objects")
            
            # 入力データのバリデーション
            validated_data = [validate_predict_input(item) for item in data]
            logger.info("Validation ok")
            
            # モデルが読み込まれていない場合のエラー
            if app.model is None:
                raise RuntimeError("Model is not loaded")
            
            logger.info("loading Model ok")
            # 予測の実行
            predictions = []
            for item in validated_data:
                # 入力データをnumpy配列に変換
                features = np.array([[
                    item['param1'],
                    item['param2'],
                    item['param3'],
                    item['param4'],
                    item['param5'],
                    item['param6']
                ]])
                
                logger.info(features)
                
                # 予測の実行
                predicted_class = app.model.predict(features)[0]
                probabilities = app.model.predict_proba(features)[0]
                
                logger.info("prediction ok")
                
                # クラス名と確率のマッピング
                class_names = app.model.classes_
                prob_dict = {
                    f"C{i+1}": float(prob)
                    for i, prob in enumerate(probabilities)
                }
                
                predictions.append({
                    "predicted_class": f"C{int(predicted_class) + 1}",
                    "probabilities": prob_dict
                })
            
            return jsonify({"predictions": predictions})
            
        except ValueError as e:
            logger.error(f"Validation error: {str(e)}")
            return jsonify({
                "error": "Bad Request",
                "message": str(e)
            }), 400
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return jsonify({
                "error": "Internal Server Error",
                "message": "An error occurred during prediction"
            }), 500 
