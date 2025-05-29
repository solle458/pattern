from flask import Flask
from flask.json import jsonify
import logging
from pathlib import Path
import joblib
import os
from dotenv import load_dotenv

load_dotenv()

# ロギングの設定 (必要に応じてコメントを解除)
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

def create_app():
    """Flaskアプリケーションのファクトリー関数"""
    app = Flask(__name__)

    # モデル、スケーラー、ラベルエンコーダーのベースパス
    base_artifact_path = os.getenv('MODEL_BASE_PATH', str(Path(__file__).parent.parent.parent))
    models_dir = Path(base_artifact_path) / 'models'

    # モデルの読み込み
    model_name = os.getenv('MODEL_NAME', 'model.pkl')
    model_path = models_dir / model_name
    # logger.info(f"モデルを次のパスから読み込みます: {model_path}")
    try:
        loaded_data = joblib.load(model_path)
        if isinstance(loaded_data, dict) and 'model' in loaded_data:
            app.model = loaded_data['model']
        else:
            app.model = loaded_data
        # logger.info(f"モデルの読み込みに成功しました (型: {type(app.model)})")
    except Exception as e:
        # logger.error(f"モデルの読み込みに失敗しました: {str(e)}")
        app.model = None

    # スケーラーの読み込み
    scaler_name = os.getenv('SCALER_NAME', 'scaler.pkl') # 例: scaler_c5_lgbm.pkl
    scaler_path = models_dir / scaler_name
    # logger.info(f"スケーラーを次のパスから読み込みます: {scaler_path}")
    try:
        app.scaler = joblib.load(scaler_path)
        # logger.info(f"スケーラーの読み込みに成功しました (型: {type(app.scaler)})")
    except Exception as e:
        # logger.error(f"スケーラーの読み込みに失敗しました: {str(e)}")
        app.scaler = None

    # ラベルエンコーダーの読み込み
    label_encoder_name = os.getenv('LABEL_ENCODER_NAME', 'label_encoder.pkl') # 例: label_encoder_c5_lgbm.pkl
    label_encoder_path = models_dir / label_encoder_name
    # logger.info(f"ラベルエンコーダーを次のパスから読み込みます: {label_encoder_path}")
    try:
        app.label_encoder = joblib.load(label_encoder_path)
        # logger.info(f"ラベルエンコーダーの読み込みに成功しました (型: {type(app.label_encoder)})")
    except Exception as e:
        # logger.error(f"ラベルエンコーダーの読み込みに失敗しました: {str(e)}")
        app.label_encoder = None

    # lightgbmなどのモデルライブラリのインポート (必要な場合)
    # test.py にあったように、特定のモデルでは読み込み時にライブラリのインポートが必要なことがあります。
    # joblibで保存されたscikit-learn API互換モデルなら通常は不要ですが、
    # LightGBMの場合、警告を避けるためやカスタムオブジェクトのために必要になることがあります。
    try:
        import lightgbm # noqa F401: LGBMモデルの読み込み時の警告回避や登録のため
    except ImportError:
        # logger.info("LightGBMがインストールされていないため、インポートをスキップします。")
        pass


    # ルートの登録
    from .routes import register_routes
    register_routes(app)

    # エラーハンドラーの登録
    from .errors import register_error_handlers
    register_error_handlers(app)

    return app

# グローバルなappインスタンスを作成
app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5001)), debug=False)
