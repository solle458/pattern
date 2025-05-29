from flask import jsonify, request, current_app as app # current_app をインポート
import numpy as np
import logging
from .validators import validate_predict_input

logger = logging.getLogger(__name__)

def register_routes(app_context): # app との衝突を避けるため app を app_context に変更
    """ルートの登録"""

    @app_context.route('/health', methods=['GET'])
    def health_check():
        """ヘルスチェックエンドポイント"""
        return jsonify({
            "status": "healthy",
            "message": "API is running"
        })

    @app_context.route('/predict', methods=['POST'])
    def predict():
        """予測エンドポイント"""
        try:
            # 入力データの取得とバリデーション
            data = request.get_json()
            if not isinstance(data, list):
                # リストでない場合はリストにラップして処理を継続
                if isinstance(data, dict):
                    data = [data]
                else:
                    raise ValueError("入力はオブジェクトのリストまたは単一のオブジェクトである必要があります")

            # 入力データのバリデーション
            validated_data = [validate_predict_input(item) for item in data]
            logger.info("バリデーションOK")

            # モデル、スケーラー、エンコーダーが読み込まれていない場合のエラー
            if app.model is None:
                logger.error("モデルが読み込まれていません")
                raise RuntimeError("モデルが読み込まれていません")
            if app.scaler is None:
                logger.error("スケーラーが読み込まれていません")
                raise RuntimeError("スケーラーが読み込まれていません")
            if app.label_encoder is None:
                logger.error("ラベルエンコーダーが読み込まれていません")
                raise RuntimeError("ラベルエンコーダーが読み込まれていません")

            logger.info("モデル、スケーラー、ラベルエンコーダーの読み込みOK")

            predictions = []
            for item in validated_data:
                # 入力データをnumpy配列に変換
                features_raw = np.array([[
                    item['param1'],
                    item['param2'],
                    item['param3'],
                    item['param4'],
                    item['param5'],
                    item['param6']
                ]])

                logger.info(f"生の入力特徴量: {features_raw}")

                # スケーラーを適用
                scaled_features = app.scaler.transform(features_raw)
                logger.info(f"スケーリング後の特徴量: {scaled_features}")

                # 予測の実行
                predicted_encoded_class = app.model.predict(scaled_features)[0]
                probabilities = app.model.predict_proba(scaled_features)[0]

                logger.info(f"予測されたエンコード済みクラス: {predicted_encoded_class}, 確率: {probabilities}")

                # エンコードされたクラスを元のラベルに戻す
                # inverse_transformは通常1D配列を期待するので、単一の値をリスト/配列に入れる
                predicted_original_label = app.label_encoder.inverse_transform([predicted_encoded_class])[0]

                # クラス名と確率のマッピング (ラベルエンコーダーのクラスを使用)
                prob_dict = {
                    str(cls_name): float(prob)
                    for cls_name, prob in zip(app.label_encoder.classes_, probabilities)
                }

                predictions.append({
                    "predicted_class": str(predicted_original_label), # 元のラベルを使用
                    "probabilities": prob_dict
                })

            return jsonify({"predictions": predictions})

        except ValueError as e:
            logger.error(f"バリデーションまたは入力エラー: {str(e)}")
            return jsonify({
                "error": "Bad Request",
                "message": str(e)
            }), 400

        except RuntimeError as e: # 読み込み問題に特化
            logger.error(f"設定エラー: {str(e)}")
            return jsonify({
                "error": "Internal Server Error",
                "message": str(e) # またはクライアント向けのより一般的なメッセージ
            }), 500

        except Exception as e:
            logger.error(f"予測エラー: {str(e)}", exc_info=True) # ログに詳細な情報を残すため exc_info=True を追加
            return jsonify({
                "error": "Internal Server Error",
                "message": "予測中にエラーが発生しました"
            }), 500
